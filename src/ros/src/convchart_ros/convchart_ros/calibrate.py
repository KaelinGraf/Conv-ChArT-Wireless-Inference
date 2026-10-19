"""
Camera calibration node and GUI for the ChArUco calibration of the camera intrinsics.

Live preview, human-triggered full-resolution captures, and a solve that is refined with every
further kept view. CalibrateCam shows the 'image' stream and calls the camera's 'image_full_res'
service (convchart_interfaces/GetFullResImage, served by the Pi's camera node) for full-resolution
stills. Each capture is checked at once (calib_session), saved in the session
folder and shown for review_s seconds. Once min_views views are kept, a background thread solves
the calibration after every kept view, then writes the log and, with auto_replace, the config's
CAMERA block. run_gui owns the OpenCV window on the main thread while a MultiThreadedExecutor
spins the node in a background thread.

The corners come from the deployed model (detector model, the default: the config's MODEL block
through the inference pipeline, calib_model) or from OpenCV's ChArUco detector (detector opencv).
With the model, the live view also shows the board's distance from the camera in large type, from
the stream and the config's current CAMERA K and dist, to set the lens focus by before calibrating.

resume carries on in an earlier session's folder (or the latest one): its kept views are
re-detected from their saved frames and solved at once, and new captures add to them.

    ros2 run convchart_ros calibrate --ros-args -p config:=<cfg.yaml> [-p auto_replace:=true]
        [-p detector:=opencv] [-p resume:=<session folder>|latest]
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import functools
import itertools
import os
from pathlib import Path
import threading
import time
import traceback

from convchart_interfaces.srv import GetFullResImage
from convchart_qos.qos import IMAGE_SUB_QOS
import cv2
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.exceptions import ParameterException
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.task import Future
from sensor_msgs.msg import CompressedImage
import yaml

from .calib_utils.calib_board import BoardSpec, make_board
from .calib_utils.calib_session import (
    build_log, CalibrationSession, Detection, latest_saved_session, read_saved_session,
    SavedSession, scale_intrinsics, settings_from_cfg, Solution, write_camera_to_cfg, write_log)
from .calib_utils.calib_view import compose_live, compose_review, HudState
from .screen import fill_scale, screen_size

WINDOW_NAME = "calibrate"
LOG_NAME = "calibration_log.yaml"
FRAMES_DIR = "frames"
DEFAULT_CANVAS_SIZE = (640, 480)
DETECTORS = ("model", "opencv")

_MESSAGE_S = 4.0
_STALL_S = 1.0
_TIMEOUT_CHECK_S = 0.05
_SPIN_WAIT_S = 0.1
_GUI_WAIT_MS = 30
_RANGE_PERIOD_S = 0.1
_RANGE_HOLD_S = 0.5
_HELP_DISTANCE = "SPACE capture   U undo   M coverage   D distance   Q quit"
_NOTICE_S = 6.0
_HISTORY_KEYS = {"n_views", "rms_px", "fx", "fy", "cx", "cy"}
_QUIT_KEYS = (ord("q"), ord("Q"), 27)
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class Review:
    """Verdict on a capture, shown in review mode until `until`.

    Args:
        frame: Captured frame, mono, full resolution.
        corners: (N, 2) ChArUco corners found in it, full-res pixels (pixel-centre convention).
        kept: The capture became a calibration view.
        verdict: "KEPT" or "REJECTED: <reason>".
        until: time.monotonic() at which the review ends.
    """
    frame: np.ndarray
    corners: np.ndarray
    kept: bool
    verdict: str
    until: float


@dataclass(frozen=True)
class GuiState:
    """Snapshot of the node for one GUI frame: everything the window shows, plus its activity.

    Args:
        frame: Latest stream frame (mono), None until the first one arrives.
        canvas_size: (W, H) of the window canvas: the stream frame size, else (640, 480).
        hud: HUD contents: counters, coverage %, status line, verdict (review mode), message.
        coverage: (rows, cols) corner counts of the active views per coverage cell.
        review: Latest capture while its review is showing, else None (live mode).
        capturing: A capture request is in flight (sent, reply not yet handled).
        solving: A solve is running or queued.
        solution: Latest solution (full resolution), None before the first solve.
    """
    frame: np.ndarray | None
    canvas_size: tuple[int, int]
    hud: HudState
    coverage: np.ndarray
    review: Review | None
    capturing: bool
    solving: bool
    solution: Solution | None


@dataclass
class _Pending:
    """Capture request in flight.

    Args:
        deadline: time.monotonic() after which the request is cancelled.
        future: Service call future, None until the request is sent.
    """
    deadline: float
    future: Future | None = None


class CalibrateCam(Node):
    """Node that ingests camera frames to calibrate the camera intrinsics and distortion.

    Constructor arguments are defaults for the ROS parameters of the same names; parameters win.

    Args:
        cfg_pth: Path to cfg.yaml; CALIBRATION sets the board and settings, CAMERA gets the result.
        auto_replace: Write CAMERA.K, CAMERA.dist and CAMERA.image_size after every solve.
        output_dir: Folder for the session folders; "" means <config dir>/../calibration_output.
        capture_timeout_s: Seconds a capture request may take before it is cancelled.
        capture_max_age_s: Oldest frame a capture accepts, in seconds; 0 accepts any frame.
        detector: "model" (the config's MODEL, with distance readout) or "opencv" (any board).
        resume: Earlier session folder to carry on in, "latest", or "" for a new session.
    """

    def __init__(self, cfg_pth: str = "", auto_replace: bool = False, output_dir: str = "",
                 capture_timeout_s: float = 5.0, capture_max_age_s: float = 0.5,
                 detector: str = DETECTORS[0], resume: str = ""):
        super().__init__("calibrate_cam_node")
        try:
            self._cfg_pth: str = str(self.declare_parameter("config", str(cfg_pth)).value)
            self._auto_replace = bool(
                self.declare_parameter("auto_replace", bool(auto_replace)).value)
            output_dir = str(self.declare_parameter("output_dir", str(output_dir)).value)
            timeout = self.declare_parameter("capture_timeout_s", float(capture_timeout_s),
                                             ParameterDescriptor(dynamic_typing=True)).value
            self._capture_timeout_s = float(timeout)
            if not self._capture_timeout_s > 0:
                raise ValueError(f"capture_timeout_s must be positive, got {timeout}")
            max_age = self.declare_parameter("capture_max_age_s", float(capture_max_age_s),
                                             ParameterDescriptor(dynamic_typing=True)).value
            self._capture_max_age_s = float(max_age)
            if not self._capture_max_age_s >= 0:
                raise ValueError(f"capture_max_age_s must be >= 0, got {max_age}")
            self._detector = str(self.declare_parameter("detector", str(detector)).value)
            if self._detector not in DETECTORS:
                raise ValueError(f"detector must be one of {', '.join(DETECTORS)}, "
                                 f"got {self._detector!r}")
            resume = str(self.declare_parameter("resume", str(resume)).value).strip()
            if not self._cfg_pth:
                raise ValueError("configuration path is not set (-p config:=<cfg.yaml>)")
            self._config_path = Path(os.path.abspath(self._cfg_pth))
            with open(self._config_path, "r") as file:
                cfg = yaml.safe_load(file) or {}
            if not isinstance(cfg, dict):
                raise ValueError(f"{self._config_path} does not hold a YAML mapping")
            self._settings = settings_from_cfg(cfg)
            make_board(self._settings.board)
            self._model = None
            if self._detector == "model":
                from .calib_utils.calib_model import check_model_board, ModelDetector
                check_model_board(self._settings.board)
                self._model = ModelDetector(self._config_path)
            self._session = CalibrationSession(
                self._settings, detect=None if self._model is None else self._model.corners)
            self._camera = _camera_from_cfg(cfg)
            started = datetime.now().astimezone()
            root = (Path(os.path.abspath(output_dir)) if output_dir
                    else self._config_path.parent.parent / "calibration_output")
            saved = None
            if not resume:
                self._session_dir = _new_session_dir(root, started)
            else:
                folder = (latest_saved_session(root) if resume == "latest"
                          else Path(os.path.abspath(resume)))
                if folder is None:
                    raise ValueError(f"resume:=latest: no session folder in {root} holds a kept "
                                     f"view")
                saved = read_saved_session(folder)
                _check_saved_board(saved, self._settings.board)
                self._session_dir = saved.folder
                view_files, now_rejected = self._restore(saved)
        except Exception as error:
            self.get_logger().error(str(error))
            self.destroy_node()
            raise
        earlier = (saved.log or {}).get("session") if saved is not None else None
        earlier = earlier if isinstance(earlier, dict) else {}
        self._started = (str(earlier["started"]) if earlier.get("started")
                         else started.isoformat(timespec="seconds"))
        resumed = earlier.get("resumed")
        self._resumed: list[dict] = [dict(entry) for entry in
                                     (resumed if isinstance(resumed, list) else [])
                                     if isinstance(entry, dict)]
        if saved is not None:
            self._resumed.append({"at": started.isoformat(timespec="seconds"),
                                  "detector": self._detector, "views_restored": len(view_files),
                                  "views_now_rejected": now_rejected})
        self._log_path = self._session_dir / LOG_NAME
        self._backup_path = self._session_dir / f"{self._config_path.name}.bak"

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frame: np.ndarray | None = None
        self._frame_time = 0.0
        self._stream_size: tuple[int, int] | None = None
        self._pending: _Pending | None = None
        self._review: Review | None = None
        self._message: str | None = None
        self._message_until = 0.0
        self._dirty = False
        self._solving = False
        self._closing = False
        self._closed = False
        self._config_written = bool(earlier.get("config_written", False))
        self._view_files: dict[int, str] = {} if saved is None else view_files
        self._views_lock = threading.Lock()
        self._n_rejected_files = 0 if saved is None else saved.rejected
        self._notice: str | None = None
        self._notice_until = 0.0
        self._range = None
        self._range_time = 0.0
        self._range_stop = threading.Event()

        self._video_cb_group = MutuallyExclusiveCallbackGroup()
        self._calib_cb_group = MutuallyExclusiveCallbackGroup()

        self._vid_sub = self.create_subscription(CompressedImage, "image", self._video_stream,
                                                 IMAGE_SUB_QOS,
                                                 callback_group=self._video_cb_group)
        self._capture_client = self.create_client(GetFullResImage, "image_full_res",
                                                  callback_group=self._calib_cb_group)
        self._timeout_timer = self.create_timer(_TIMEOUT_CHECK_S, self._check_timeout,
                                                callback_group=self._calib_cb_group)
        self._worker = threading.Thread(target=self._solve_worker, name="calibrate_solve",
                                        daemon=True)
        self._worker.start()
        self._range_worker_thread = None
        if self._model is not None and self._camera is not None:
            self._range_worker_thread = threading.Thread(target=self._range_worker,
                                                         name="calibrate_range", daemon=True)
            self._range_worker_thread.start()

        spec = self._settings.board
        self.get_logger().info(
            f"{spec.squares[0]}x{spec.squares[1]} {spec.dictionary} board (markers from "
            f"{spec.first_marker_id}), min_views {self._settings.min_views}; corners from "
            f"{'the model' if self._model is not None else 'OpenCV'}; session folder "
            f"{self._session_dir}; auto_replace "
            f"{'on: ' + str(self._config_path) if self._auto_replace else 'off'}")
        if self._model is not None and self._camera is None:
            self.get_logger().warning("the config has no CAMERA.K, so the distance readout is off")
        if saved is not None:
            restored = len(view_files)
            self._notice = (f"resumed {self._session_dir.name}: {restored} views"
                            + (f", {now_rejected} now rejected" if now_rejected else ""))
            self._notice_until = time.monotonic() + _NOTICE_S
            self.get_logger().info(
                f"resumed {self._session_dir}: {restored} views restored"
                + (f" ({now_rejected} more now rejected)" if now_rejected else "")
                + f", {saved.rejected} rejected captures; new views from "
                f"{FRAMES_DIR}/view_{max(saved.next_index, max(view_files, default=-1) + 1):03d}"
                f".png")
            if self._session.ready:
                self._queue_solve()
            else:
                self._write_log()

    @property
    def session_dir(self) -> Path:
        """Return this session's folder (calibration_log.yaml, frames/, the config backup)."""
        return self._session_dir

    def request_capture(self) -> bool:
        """Ask the camera for a full-resolution still, as the SPACE key does."""
        now = time.monotonic()
        with self._lock:
            reviewing = self._review is not None and now < self._review.until
            if self._closed or self._pending is not None or reviewing:
                return False
            if not self._capture_client.service_is_ready():
                self._set_message("camera service image_full_res not available", now)
                return False
            pending = _Pending(deadline=now + self._capture_timeout_s)
            pending.future = self._capture_client.call_async(
                GetFullResImage.Request(max_age=self._capture_max_age_s))
            self._pending = pending
            self._message = None
        pending.future.add_done_callback(functools.partial(self._on_capture_reply, pending))
        return True

    def undo(self) -> bool:
        """Remove the last kept view, as the U key does."""
        with self._views_lock:
            with self._lock:
                if self._closed:
                    return False
            index = self._session.undo()
            if index is None:
                self._report("nothing to undo")
                return False
            with self._lock:
                name = self._view_files.pop(index, f"{FRAMES_DIR}/view_{index:03d}.png")
            try:
                (self._session_dir / name).rename(
                    self._session_dir / FRAMES_DIR / f"undone_{index:03d}.png")
            except OSError as error:
                self._report(f"could not rename {name}: {error}", error=True)
        self.get_logger().info(f"undid view {index}")
        if self._session.ready:
            self._queue_solve()
        return True

    def state(self) -> GuiState:
        """Take a snapshot of what the GUI shows."""
        now = time.monotonic()
        with self._lock:
            frame, frame_time, stream_size = self._frame, self._frame_time, self._stream_size
            review = self._review
            if review is not None and now >= review.until:
                review = None
            message = self._message if now < self._message_until else None
            capturing = self._pending is not None
            solving = self._solving or self._dirty
            reading, reading_time = self._range, self._range_time
        session = self._session
        kept, rejected = session.kept_count, session.rejected_count
        active, dropped = session.active_count, session.dropped_count
        coverage = session.coverage()
        solution = session.solution
        status, converged = self._status(active, solving, solution)
        if dropped:
            status += f" | {dropped} outlier{'' if dropped == 1 else 's'} dropped"
        if capturing:
            status += " | capturing..."
        if self._notice is not None and now < self._notice_until:
            status += f" | {self._notice}"
        if message is None and frame is None:
            message = "waiting for the image stream"
        elif message is None and now - frame_time > _STALL_S:
            message = f"stream stalled: no frame for {now - frame_time:.0f} s"
        distance = None
        if self._model is not None:
            if self._camera is None:
                distance = "no CAMERA.K"
            elif self._model.square_length_m is None:
                distance = "no BOARD.square_length_m"
            elif reading is not None and now - reading_time <= _RANGE_HOLD_S:
                distance = (reading.z, reading.range)
            else:
                distance = "no board"
        hud = HudState(kept=kept, rejected=rejected, min_views=self._settings.min_views,
                       status=status,
                       coverage_fraction=float(np.count_nonzero(coverage)) / coverage.size,
                       converged=converged,
                       verdict=None if review is None else review.verdict,
                       verdict_ok=None if review is None else review.kept,
                       message=message, distance=distance,
                       **({} if self._model is None else {"help": _HELP_DISTANCE}))
        return GuiState(frame=frame, canvas_size=stream_size or DEFAULT_CANVAS_SIZE, hud=hud,
                        coverage=coverage, review=review, capturing=capturing, solving=solving,
                        solution=solution)

    def close(self) -> None:
        """End the session and write the final log."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pending, self._pending = self._pending, None
        with self._views_lock:
            pass
        if pending is not None and pending.future is not None:
            self._capture_client.remove_pending_request(pending.future)
            pending.future.cancel()
        with self._cond:
            self._closing = True
            self._cond.notify_all()
        self._range_stop.set()
        if self._range_worker_thread is not None:
            self._range_worker_thread.join()
        self._worker.join()
        self._write_log()

    def _video_stream(self, msg: CompressedImage) -> None:
        """Keep the latest stream frame (mono), its arrival time and the stream size for display.

        Args:
            msg: Compressed frame from the 'image' stream.
        """
        frame = _decode(msg.data)
        if frame is None:
            self.get_logger().warning("could not decode a stream frame", throttle_duration_sec=5.0)
            return
        with self._lock:
            self._frame = frame
            self._frame_time = time.monotonic()
            self._stream_size = (int(frame.shape[1]), int(frame.shape[0]))

    def _check_timeout(self) -> None:
        """Cancel the capture request in flight once capture_timeout_s has passed."""
        message = f"capture timed out ({self._capture_timeout_s:g} s)"
        with self._lock:
            pending = self._pending
            now = time.monotonic()
            if (pending is None or pending.future is None or pending.future.done()
                    or now < pending.deadline):
                return
            self._pending = None
            self._set_message(message, now)
        self._capture_client.remove_pending_request(pending.future)
        pending.future.cancel()
        self.get_logger().warning(message)

    def _on_capture_reply(self, pending: _Pending, future: Future) -> None:
        """Record a capture reply, start its review and queue a solve.

        Requests that timed out or that close() abandoned are ignored.

        Args:
            pending: Request the reply belongs to.
            future: Completed service call future.
        """
        with self._views_lock:
            with self._lock:
                if self._pending is not pending or not future.done():
                    return
            frame, detection, message = None, None, None
            try:
                frame, detection, message = self._record_capture(future.result())
            except Exception as error:
                message = f"capture failed: {error}"
                self.get_logger().error(f"capture failed:\n{traceback.format_exc()}")
            if message is not None:
                self.get_logger().warning(message)
            queue = detection is not None and detection.kept and self._session.ready
            with self._lock:
                self._pending = None
                now = time.monotonic()
                if detection is not None:
                    verdict = "KEPT" if detection.kept else f"REJECTED: {detection.reason}"
                    self._review = Review(frame, detection.corners, detection.kept, verdict,
                                          now + self._settings.review_s)
                if message is not None:
                    self._set_message(message, now)
                if queue:
                    self._dirty = True
                    self._cond.notify_all()

    def _record_capture(self, response: GetFullResImage.Response
                        ) -> tuple[np.ndarray | None, Detection | None, str | None]:
        """Decode, evaluate and submit one capture reply and save its PNG.

        Returns (frame, detection, message); message is None when all went well.

        Args:
            response: GetFullResImage reply.
        """
        if not response.success:
            return None, None, f"capture failed: {response.message or 'no reason given'}"
        data = np.frombuffer(response.image.data, dtype=np.uint8)
        frame = _decode(data)
        if frame is None:
            return None, None, "capture failed: the image could not be decoded"
        detection = self._session.evaluate(frame)
        index = self._session.submit(detection)
        if index is None:
            name = f"{FRAMES_DIR}/rejected_{self._n_rejected_files:03d}.png"
            self._n_rejected_files += 1
        else:
            name = f"{FRAMES_DIR}/view_{index:03d}.png"
            with self._lock:
                self._view_files[index] = name
        message = None
        try:
            _save_png(self._session_dir / name, data, frame)
        except OSError as error:
            message = f"could not save {name}: {error}"
        verdict = "kept" if detection.kept else f"rejected ({detection.reason})"
        self.get_logger().info(f"capture {verdict}: {len(detection.ids)} corners, {name}")
        return frame, detection, message

    def _solve_worker(self) -> None:
        """Run the queued solves one at a time until close()."""
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._dirty or self._closing)
                if not self._dirty:
                    return
                self._dirty = False
                self._solving = True
            try:
                self._solve_and_write()
            except Exception:
                self.get_logger().error(f"solve worker:\n{traceback.format_exc()}")
            finally:
                with self._cond:
                    self._solving = False

    def _restore(self, saved: SavedSession) -> tuple[dict[int, str], int]:
        """Re-detect an earlier session's kept views and submit them under their own indices.

        Returns (view files by index, number of views rejected on re-detection).

        Args:
            saved: Earlier session to resume.
        """
        view_files: dict[int, str] = {}
        now_rejected = unreadable = 0
        for index, name, dropped in saved.views:
            frame = cv2.imread(str(saved.folder / name), cv2.IMREAD_GRAYSCALE)
            if frame is None:
                self.get_logger().warning(f"resume: cannot read {name}; counted as rejected")
                now_rejected += 1
                unreadable += 1
                continue
            detection = self._session.evaluate(frame)
            if self._session.submit(detection, index=index, dropped=dropped) is None:
                self.get_logger().warning(f"resume: {name} is now rejected: {detection.reason}")
                now_rejected += 1
            else:
                view_files[index] = name
        history = [entry for entry in (saved.log or {}).get("history") or []
                   if isinstance(entry, dict) and _HISTORY_KEYS <= set(entry)]
        self._session.carry_over(rejected=saved.rejected + unreadable,
                                 next_index=saved.next_index, history=history)
        return view_files, now_rejected

    def _range_worker(self) -> None:
        """Measure the board's distance on each new stream frame, at most every _RANGE_PERIOD_S."""
        measured = None
        while not self._range_stop.is_set():
            started = time.monotonic()
            with self._lock:
                frame, frame_time = self._frame, self._frame_time
            if frame is not None and frame_time != measured:
                measured = frame_time
                reading = None
                try:
                    K, dist = self._camera_at((int(frame.shape[1]), int(frame.shape[0])))
                    reading = self._model.board_range(frame, K, dist)
                except Exception as error:
                    self.get_logger().error(f"distance readout: {error}",
                                            throttle_duration_sec=5.0)
                if reading is not None:
                    with self._lock:
                        self._range, self._range_time = reading, frame_time
            self._range_stop.wait(max(0.0, _RANGE_PERIOD_S - (time.monotonic() - started)))

    def _camera_at(self, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Return the config's CAMERA K and dist for frames of size (W, H).

        Args:
            size: (W, H) of the frames.
        """
        K, dist, camera_size = self._camera
        if camera_size is not None and camera_size != size:
            K = scale_intrinsics(K, camera_size, size)
        return K, dist

    def _solve_and_write(self) -> None:
        """Solve, then write the config's CAMERA block (auto_replace) and the log."""
        try:
            solution = self._session.solve()
        except RuntimeError:
            return
        except cv2.error as error:
            self.get_logger().error(f"solve failed: {error}")
            self._report("solve failed (details in the terminal)")
            self._write_log()
            return
        K = solution.K
        self.get_logger().info(
            f"solved over {solution.n_views} views: rms {solution.rms:.3f} px, fx {K[0, 0]:.2f}, "
            f"fy {K[1, 1]:.2f}, cx {K[0, 2]:.2f}, cy {K[1, 2]:.2f}"
            + (f", dropped views {solution.dropped}" if solution.dropped else "")
            + (", converged" if solution.converged else ""))
        if self._auto_replace:
            self._write_camera(solution)
        self._write_log()

    def _write_camera(self, solution: Solution) -> None:
        """Write the solution at stream resolution into the config's CAMERA block.

        Args:
            solution: Full-resolution solution to write.
        """
        with self._lock:
            stream_size = self._stream_size
        if stream_size is None:
            self.get_logger().warning("no stream frame seen yet, so CAMERA (stream resolution) "
                                      "is not written; the log shows config_written: false")
            return
        if not (np.isfinite(solution.K).all() and np.isfinite(solution.dist).all()):
            self._report("the solve is not finite; CAMERA not written", error=True)
            return
        try:
            K_stream = scale_intrinsics(solution.K, solution.image_size, stream_size)
            write_camera_to_cfg(self._config_path, K_stream, solution.dist, stream_size,
                                backup_path=self._backup_path)
        except (OSError, ValueError, yaml.YAMLError) as error:
            self._report(f"CAMERA not written: {error}", error=True)
            return
        with self._lock:
            self._config_written = True

    def _write_log(self) -> None:
        """Write calibration_log.yaml for the session in its current state."""
        backup = str(self._backup_path) if self._backup_path.exists() else None
        with self._lock:
            meta = {"started": self._started, "config": str(self._config_path),
                    "auto_replace": self._auto_replace, "config_written": self._config_written,
                    "config_backup": backup, "detector": self._detector,
                    "resumed": list(self._resumed), "stream_size": self._stream_size,
                    "view_files": dict(self._view_files)}
        try:
            write_log(self._log_path, build_log(self._session, self._settings, meta))
        except OSError as error:
            self._report(f"log not written: {error}", error=True)

    def _queue_solve(self) -> None:
        with self._cond:
            self._dirty = True
            self._cond.notify_all()

    def _status(self, active: int, solving: bool,
                solution: Solution | None) -> tuple[str, bool]:
        """Return the HUD status line and whether it reports a converged solution.

        Args:
            active: Number of active views.
            solving: A solve is running or queued.
            solution: Latest solution, None before the first solve.
        """
        min_views = self._settings.min_views
        if solution is None and solving:
            return "solving...", False
        if solution is None or active < min_views:
            return f"collecting {active}/{min_views}", False
        K, std = solution.K, solution.std
        parts = [f"rms {solution.rms:.2f} px"]
        parts += [f"{name} {value:.1f} +- {std[name]:.1f}"
                  for name, value in (("fx", K[0, 0]), ("fy", K[1, 1]), ("cx", K[0, 2]),
                                      ("cy", K[1, 2]))]
        if solution.converged:
            parts.append("converged")
        if solving:
            parts.append("solving...")
        return " | ".join(parts), solution.converged

    def _set_message(self, text: str, now: float) -> None:
        """Show text as the HUD's transient message; the caller holds the lock.

        Args:
            text: Message text.
            now: time.monotonic() at which it is shown.
        """
        self._message = text
        self._message_until = now + _MESSAGE_S

    def _report(self, text: str, error: bool = False) -> None:
        """Log text and show it as the HUD's message; the caller must not hold the lock.

        Args:
            text: Message text.
            error: Log as an error instead of a warning.
        """
        (self.get_logger().error if error else self.get_logger().warning)(text)
        with self._lock:
            self._set_message(text, time.monotonic())


def run_gui(node: CalibrateCam) -> None:
    """Show the calibration window until it is quit; main thread only.

    Keys: SPACE capture, U undo, M coverage map, D distance readout, Q or ESC quit.

    Args:
        node: Calibration node, spun by an executor on another thread.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("run_gui must run on the main thread")
    show_coverage = True
    show_distance = True
    was_visible = False
    screen, scale = screen_size(), None
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    try:
        while rclpy.ok():
            canvas = _compose(node.state(), show_coverage, show_distance)
            if scale is None:
                scale = fill_scale(canvas.shape[1], canvas.shape[0], screen)
            if scale != 1.0:
                canvas = cv2.resize(canvas, None, fx=scale, fy=scale,
                                    interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            cv2.imshow(WINDOW_NAME, canvas)
            key = _key_code(cv2.waitKeyEx(_GUI_WAIT_MS))
            if key in _QUIT_KEYS:
                break
            if key == ord(" "):
                node.request_capture()
            elif key in (ord("u"), ord("U")):
                node.undo()
            elif key in (ord("m"), ord("M")):
                show_coverage = not show_coverage
            elif key in (ord("d"), ord("D")):
                show_distance = not show_distance
            visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
            if was_visible and not visible:
                break
            was_visible = was_visible or visible
    finally:
        try:
            cv2.destroyWindow(WINDOW_NAME)
            cv2.waitKey(1)
        except cv2.error:
            pass


def main(args: list[str] | None = None) -> None:
    """Run the calibration GUI: the node spins in a background thread, the window on the main one.

    ROS parameters: config (required), auto_replace, output_dir, capture_*, detector, resume.

    Args:
        args: Command-line arguments for rclpy.init; None uses sys.argv.
    """
    rclpy.init(args=args)
    try:
        node = CalibrateCam()
    except (OSError, ValueError, TypeError, AttributeError, RuntimeError, yaml.YAMLError,
            ParameterException) as error:
        rclpy.try_shutdown()
        raise SystemExit(f"calibrate: {error}") from None
    except BaseException:
        rclpy.try_shutdown()
        raise
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    stop = threading.Event()
    spinner = threading.Thread(target=_spin, args=(executor, stop), name="calibrate_executor",
                               daemon=True)
    spinner.start()
    try:
        run_gui(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        stop.set()
        spinner.join()
        node.destroy_node()
        rclpy.try_shutdown()


def _compose(state: GuiState, show_coverage: bool, show_distance: bool = True) -> np.ndarray:
    """Return the window canvas: review mode while a capture's review is showing, else live mode.

    Args:
        state: GUI snapshot to draw.
        show_coverage: Draw the coverage map in live mode.
        show_distance: Draw the distance readout in live mode.
    """
    review = state.review
    if review is not None:
        return compose_review(review.frame, review.corners, review.kept, state.hud,
                              state.canvas_size)
    hud = state.hud if show_distance else replace(state.hud, distance=None)
    return compose_live(state.frame, hud, state.coverage, state.canvas_size, show_coverage)


def _camera_from_cfg(cfg: dict) -> tuple[np.ndarray, np.ndarray, tuple[int, int] | None] | None:
    """Return the config's CAMERA block as (K, dist, image_size), None without a K.

    A missing dist means none; a missing image_size means K is at the stream's resolution.

    Args:
        cfg: Loaded config mapping.
    """
    camera = cfg.get("CAMERA") or {}
    if camera.get("K") is None:
        return None
    K = np.asarray(camera["K"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(camera.get("dist") or [0.0] * 5, dtype=np.float64).ravel()
    size = camera.get("image_size")
    return K, dist, None if size is None else (int(size[0]), int(size[1]))


def _check_saved_board(saved: SavedSession, spec: BoardSpec) -> None:
    """Raise ValueError when the saved session's log names a board other than spec's layout.

    The square length may differ; a session without a log is not checked.

    Args:
        saved: Earlier session to resume.
        spec: Board of the current settings.
    """
    board = (saved.log or {}).get("board")
    if not isinstance(board, dict):
        return
    try:
        theirs = (tuple(int(v) for v in board["squares"]), str(board["dictionary"]),
                  int(board["first_marker_id"]), bool(board["legacy_pattern"]),
                  round(float(board["marker_length_m"]) / float(board["square_length_m"]), 6))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        raise ValueError(f"resume: the board block of {saved.folder / LOG_NAME} is "
                         f"malformed") from None
    ours = (tuple(spec.squares), spec.dictionary, spec.first_marker_id, spec.legacy_pattern,
            round(spec.marker_side / spec.square_length, 6))
    if theirs != ours:
        raise ValueError(f"resume: {saved.folder} was taken with a {theirs[0][0]}x{theirs[0][1]} "
                         f"{theirs[1]} board (markers from {theirs[2]}), but CALIBRATION.board "
                         f"is a {ours[0][0]}x{ours[0][1]} {ours[1]} board (markers from "
                         f"{ours[2]})")


def _spin(executor: MultiThreadedExecutor, stop: threading.Event) -> None:
    """Spin executor until stop is set or ROS shuts down; the body of main()'s executor thread.

    Args:
        executor: Executor holding the node.
        stop: Event set to end the loop.
    """
    while not stop.is_set() and executor.context.ok():
        try:
            executor.spin_once(timeout_sec=_SPIN_WAIT_S)
        except Exception:
            if stop.is_set() or not executor.context.ok():
                return
            raise


def _key_code(code: int) -> int:
    """Return the key a cv2.waitKeyEx() code stands for: its low 16 bits, or -1 for no key.

    Args:
        code: Value returned by cv2.waitKeyEx().
    """
    return -1 if code == -1 else code & 0xFFFF


def _decode(data) -> np.ndarray | None:
    """Decode an image message's bytes to a mono uint8 frame; None if they hold no image.

    Args:
        data: Encoded image bytes of a CompressedImage message.
    """
    buffer = np.frombuffer(data, dtype=np.uint8)
    if buffer.size == 0:
        return None
    try:
        return cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    except cv2.error:
        return None


def _save_png(path: Path, data: np.ndarray, frame: np.ndarray) -> None:
    """Save a capture as PNG: the camera's own bytes when they are PNG, else frame re-encoded.

    Args:
        path: Destination file.
        data: Encoded image bytes from the camera.
        frame: Decoded frame, re-encoded when data is not PNG.
    """
    if data[:len(_PNG_SIGNATURE)].tobytes() != _PNG_SIGNATURE:
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            raise OSError(f"could not encode {path.name} as PNG")
        data = encoded
    path.write_bytes(data.tobytes())


def _new_session_dir(root: Path, started: datetime) -> Path:
    """Create and return <root>/<YYYYmmdd-HHMMSS>/ with its frames/ folder.

    A session started in the same second as another gets a -2, -3, ... suffix.

    Args:
        root: Folder that holds the session folders.
        started: Session start time, naming the folder.
    """
    root.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime("%Y%m%d-%H%M%S")
    for n in itertools.count(1):
        path = root / (stamp if n == 1 else f"{stamp}-{n}")
        try:
            path.mkdir()
        except FileExistsError:
            continue
        (path / FRAMES_DIR).mkdir()
        return path
    raise AssertionError("unreachable")


if __name__ == "__main__":
    main()
