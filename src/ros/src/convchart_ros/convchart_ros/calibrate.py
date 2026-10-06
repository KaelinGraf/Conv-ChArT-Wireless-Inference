"""
Camera calibration node and GUI for the ChArUco calibration of the camera intrinsics.

Live preview, human-triggered full-resolution captures, and a solve that is refined with every
further kept view. CalibrateCam shows the 'image' stream and calls the 'capture' service for
full-resolution stills. Each capture is checked at once (calib_session), saved in the session
folder and shown for review_s seconds. Once min_views views are kept, a background thread solves
the calibration after every kept view, then writes the log and, with auto_replace, the config's
CAMERA block. run_gui owns the OpenCV window on the main thread while a MultiThreadedExecutor
spins the node in a background thread.

    ros2 run convchart_ros calibrate --ros-args -p config:=<cfg.yaml> [-p auto_replace:=true]
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import functools
import itertools
import os
from pathlib import Path
import threading
import time
import traceback

from convchart_interfaces.srv import CaptureFrame
import cv2
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.exceptions import ParameterException
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSLivelinessPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.task import Future
from sensor_msgs.msg import CompressedImage
import yaml

from .calib_utils.calib_session import (build_log, CalibrationSession, Detection, scale_intrinsics,
                            settings_from_cfg, Solution, write_camera_to_cfg, write_log)
from .calib_utils.calib_view import compose_live, compose_review, HudState

WINDOW_NAME = "calibrate"
LOG_NAME = "calibration_log.yaml"
FRAMES_DIR = "frames"
DEFAULT_CANVAS_SIZE = (640, 480)        # (W, H) of the window until the first stream frame arrives

_MESSAGE_S = 4.0                        # how long a transient message stays on the HUD
_STALL_S = 1.0                          # no stream frame for longer: the HUD says it stalled
_TIMEOUT_CHECK_S = 0.05                 # period of the capture timeout check
_SPIN_WAIT_S = 0.1                      # longest executor wait before the spin loop checks stop
_GUI_WAIT_MS = 30                       # waitKeyEx period of the GUI loop
_QUIT_KEYS = (ord("q"), ord("Q"), 27)   # 27: ESC
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class Review:
    """
    A capture's verdict, shown in review mode until `until`.
      @args:
        frame: the captured frame, mono, full resolution.
        corners: (N, 2) ChArUco corners found in it, full-res pixels (pixel-centre convention).
        kept: the capture became a calibration view.
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
    """
    Snapshot of the node for one GUI frame: everything the window shows, plus the node's activity.
      @args:
        frame: latest stream frame (mono), None until the first one arrives.
        canvas_size: (W, H) of the window canvas: the stream frame size, (640, 480) until a frame
                     arrives.
        hud: HUD contents: counters, coverage %, status line, verdict (review mode), message.
        coverage: (rows, cols) corner counts of the active views per coverage cell.
        review: the latest capture while its review is showing, else None (live mode).
        capturing: a capture request is in flight (sent, reply not yet handled).
        solving: a solve is running or queued.
        solution: the latest solution (full resolution), None before the first solve.
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
    """A capture request in flight."""
    deadline: float                     # time.monotonic() after which it is cancelled
    future: Future | None = None


class CalibrateCam(Node):
    """
    Ingest camera frames to calibrate the camera intrinsics and distortion coefficients.

    Subscribes to the 'image' stream (callback group "video") for the preview and calls the
    'capture' service (callback group "calib") for full-resolution stills. A capture's reply is
    checked, recorded and saved as frames/view_NNN.png or frames/rejected_NNN.png in the session
    folder <output_dir>/<YYYYmmdd-HHMMSS>/; its review then shows for review_s seconds, and a kept
    view queues a solve once min_views are active. One worker thread runs the solves, one at a
    time (a capture during a solve queues exactly one more), and after each writes
    calibration_log.yaml and, with auto_replace, the config's CAMERA block at stream resolution,
    after backing the config up once into the session folder. The constructor arguments are the
    defaults of the ROS parameters config, auto_replace, output_dir and capture_timeout_s;
    parameters win.
      @args:
        cfg_pth: path to the configuration file (cfg.yaml). Its CALIBRATION block sets the board
                 and the session settings; with auto_replace its CAMERA block receives the result.
        auto_replace: if True, automatically replace CAMERA.K, CAMERA.dist and CAMERA.image_size
                      in the configuration file with the calibration after every solve.
        output_dir: folder for the session folders; "" means <config dir>/../calibration_output.
        capture_timeout_s: seconds a capture request may take before it is cancelled.
    """

    def __init__(self, cfg_pth: str = "", auto_replace: bool = False, output_dir: str = "",
                 capture_timeout_s: float = 5.0):
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
            if not self._cfg_pth:
                raise ValueError("configuration path is not set (-p config:=<cfg.yaml>)")
            self._config_path = Path(os.path.abspath(self._cfg_pth))
            with open(self._config_path, "r") as file:
                cfg = yaml.safe_load(file) or {}
            if not isinstance(cfg, dict):
                raise ValueError(f"{self._config_path} does not hold a YAML mapping")
            self._settings = settings_from_cfg(cfg)
            self._session = CalibrationSession(self._settings)
            started = datetime.now().astimezone()
            root = (Path(os.path.abspath(output_dir)) if output_dir
                    else self._config_path.parent.parent / "calibration_output")
            self._session_dir = _new_session_dir(root, started)
        except Exception as error:
            self.get_logger().error(str(error))
            self.destroy_node()
            raise
        self._started = started.isoformat(timespec="seconds")
        self._log_path = self._session_dir / LOG_NAME
        self._backup_path = self._session_dir / f"{self._config_path.name}.bak"

        # Shared state, behind one lock; the GUI reads it only through state().
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frame: np.ndarray | None = None
        self._frame_time = 0.0              # time.monotonic() at which _frame arrived
        self._stream_size: tuple[int, int] | None = None
        self._pending: _Pending | None = None
        self._review: Review | None = None
        self._message: str | None = None
        self._message_until = 0.0
        self._dirty = False                 # a solve is queued
        self._solving = False               # a solve is running
        self._closing = False               # the solve worker is to exit once nothing is queued
        self._closed = False
        self._config_written = False
        self._view_files: dict[int, str] = {}
        # Serialises recording a capture, undo and close(), so a view's file never lags its record.
        self._views_lock = threading.Lock()
        self._n_rejected_files = 0

        self._video_cb_group = MutuallyExclusiveCallbackGroup()
        self._calib_cb_group = MutuallyExclusiveCallbackGroup()

        imgQOSProfile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            liveliness=QoSLivelinessPolicy.AUTOMATIC,
        )

        self._vid_sub = self.create_subscription(CompressedImage, "image", self._video_stream,
                                                 imgQOSProfile,
                                                 callback_group=self._video_cb_group)
        self._capture_client = self.create_client(CaptureFrame, "capture",
                                                  callback_group=self._calib_cb_group)
        self._timeout_timer = self.create_timer(_TIMEOUT_CHECK_S, self._check_timeout,
                                                callback_group=self._calib_cb_group)
        self._worker = threading.Thread(target=self._solve_worker, name="calibrate_solve",
                                        daemon=True)
        self._worker.start()

        spec = self._settings.board
        self.get_logger().info(
            f"{spec.squares[0]}x{spec.squares[1]} {spec.dictionary} board (markers from "
            f"{spec.first_marker_id}), min_views {self._settings.min_views}; session folder "
            f"{self._session_dir}; auto_replace "
            f"{'on: ' + str(self._config_path) if self._auto_replace else 'off'}")

    @property
    def session_dir(self) -> Path:
        """
        Return this session's folder (calibration_log.yaml, frames/, the config backup).
        """
        return self._session_dir

    def request_capture(self) -> bool:
        """
        Ask the camera for a full-resolution still, as the SPACE key does.

        Returns False and does nothing while a request is in flight or a review is showing, and
        after close(); returns False with a HUD message when the 'capture' service is not
        available.
        """
        now = time.monotonic()
        with self._lock:
            reviewing = self._review is not None and now < self._review.until
            if self._closed or self._pending is not None or reviewing:
                return False
            if not self._capture_client.service_is_ready():
                self._set_message("capture service not available", now)
                return False
            pending = _Pending(deadline=now + self._capture_timeout_s)
            pending.future = self._capture_client.call_async(CaptureFrame.Request())
            self._pending = pending
            self._message = None
        pending.future.add_done_callback(functools.partial(self._on_capture_reply, pending))
        return True

    def undo(self) -> bool:
        """
        Remove the last kept view, as the U key does.

        Its file becomes frames/undone_NNN.png, and the calibration is solved again if enough views
        remain. Returns False when there is nothing to undo, and after close().
        """
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
        """
        Take a snapshot of what the GUI shows.

        The HUD's message is the latest transient message while it lasts, else a missing or stalled
        image stream (no frame for over _STALL_S seconds) is reported there.
        """
        now = time.monotonic()
        with self._lock:
            frame, frame_time, stream_size = self._frame, self._frame_time, self._stream_size
            review = self._review
            if review is not None and now >= review.until:
                review = None
            message = self._message if now < self._message_until else None
            capturing = self._pending is not None
            solving = self._solving or self._dirty
        # Read after the flags: once they show no solve, the solution is the latest one.
        session = self._session
        kept, rejected = session.kept_count, session.rejected_count
        active, dropped = session.active_count, session.dropped_count
        coverage = session.coverage()
        solution = session.solution
        status, converged = self._status(active, solving, solution)
        if dropped:                         # kept counts them; the solution and coverage do not
            status += f" | {dropped} outlier{'' if dropped == 1 else 's'} dropped"
        if capturing:                       # SPACE is ignored until the reply
            status += " | capturing..."
        if message is None and frame is None:
            message = "waiting for the image stream"
        elif message is None and now - frame_time > _STALL_S:
            message = f"stream stalled: no frame for {now - frame_time:.0f} s"
        hud = HudState(kept=kept, rejected=rejected, min_views=self._settings.min_views,
                       status=status,
                       coverage_fraction=float(np.count_nonzero(coverage)) / coverage.size,
                       converged=converged,
                       verdict=None if review is None else review.verdict,
                       verdict_ok=None if review is None else review.kept,
                       message=message)
        return GuiState(frame=frame, canvas_size=stream_size or DEFAULT_CANVAS_SIZE, hud=hud,
                        coverage=coverage, review=review, capturing=capturing, solving=solving,
                        solution=solution)

    def close(self) -> None:
        """
        End the session and write the final log.

        Refuses further captures and abandons one in flight, then lets the solve worker finish
        what is queued. Calls after the first do nothing.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pending, self._pending = self._pending, None
        with self._views_lock:          # a reply being recorded, or an undo, finishes first
            pass
        if pending is not None and pending.future is not None:
            self._capture_client.remove_pending_request(pending.future)
            pending.future.cancel()
        with self._cond:
            self._closing = True
            self._cond.notify_all()
        self._worker.join()
        self._write_log()

    def _video_stream(self, msg: CompressedImage) -> None:
        """
        Use the inference image stream to display the camera feed for calibration to the user.

        Keeps the latest frame (mono), when it arrived, and the stream size, the resolution K is
        scaled to for the config.
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
        """
        Timer ("calib" group): cancel the request in flight once capture_timeout_s has passed.
        """
        message = f"capture timed out ({self._capture_timeout_s:g} s)"
        with self._lock:
            pending = self._pending
            now = time.monotonic()
            if (pending is None or pending.future is None or pending.future.done()
                    or now < pending.deadline):
                return
            self._pending = None
            self._set_message(message, now)     # with the request gone, never after it
        self._capture_client.remove_pending_request(pending.future)
        pending.future.cancel()
        self.get_logger().warning(message)

    def _on_capture_reply(self, pending: _Pending, future: Future) -> None:
        """
        Done callback of a capture request: record the capture, start its review, queue a solve.

        Requests that timed out or that close() abandoned are ignored.
        """
        with self._views_lock:
            with self._lock:
                if self._pending is not pending or not future.done():
                    return
            frame, detection, message = None, None, None
            try:
                frame, detection, message = self._record_capture(future.result())
            except Exception as error:      # one bad reply must not end the session
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

    def _record_capture(self, response: CaptureFrame.Response
                        ) -> tuple[np.ndarray | None, Detection | None, str | None]:
        """
        Decode, evaluate and submit one capture reply and save its PNG.

        Returns (frame, detection, message): frame and detection are None when the reply holds no
        usable image; message is None when all went well.
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
        """
        Run the queued solves one at a time until close().

        However many kept views queue a solve while one runs, exactly one more runs after it.
        Returns once closing with nothing queued.
        """
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._dirty or self._closing)
                if not self._dirty:
                    return
                self._dirty = False
                self._solving = True
            try:
                self._solve_and_write()
            except Exception:               # the worker must outlive any one solve
                self.get_logger().error(f"solve worker:\n{traceback.format_exc()}")
            finally:
                with self._cond:
                    self._solving = False

    def _solve_and_write(self) -> None:
        """
        Solve, then write the config's CAMERA block (auto_replace) and the log.
        """
        try:
            solution = self._session.solve()
        except RuntimeError:
            return                          # an undo left too few views since the solve was queued
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
        """
        Write the solution at stream resolution into the config's CAMERA block.

        The config is backed up first if this session has not yet. Skipped until a stream frame
        has been seen, since the stream resolution is unknown until then.
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
        """
        Write calibration_log.yaml for the session as it is now.
        """
        backup = str(self._backup_path) if self._backup_path.exists() else None
        with self._lock:
            meta = {"started": self._started, "config": str(self._config_path),
                    "auto_replace": self._auto_replace, "config_written": self._config_written,
                    "config_backup": backup, "stream_size": self._stream_size,
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
        """
        Return the HUD status line and whether it reports a converged solution.
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
        """Show text as the HUD's transient message; the caller holds the lock."""
        self._message = text
        self._message_until = now + _MESSAGE_S

    def _report(self, text: str, error: bool = False) -> None:
        """Log text and show it as the HUD's message; the caller must not hold the lock."""
        (self.get_logger().error if error else self.get_logger().warning)(text)
        with self._lock:
            self._set_message(text, time.monotonic())


def run_gui(node: CalibrateCam) -> None:
    """
    Show the calibration window until the user quits; main thread only.

    Live mode shows the stream under the coverage map and the HUD; after a capture, review mode
    shows the captured frame with its corners and verdict for review_s seconds. Keys: SPACE
    capture, U undo, M coverage map on/off, Q or ESC quit; closing the window quits too.
      @args:
        node: the calibration node, spun by an executor on another thread.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("run_gui must run on the main thread")
    show_coverage = True
    was_visible = False
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    try:
        while rclpy.ok():
            cv2.imshow(WINDOW_NAME, _compose(node.state(), show_coverage))
            key = _key_code(cv2.waitKeyEx(_GUI_WAIT_MS))
            if key in _QUIT_KEYS:
                break
            if key == ord(" "):
                node.request_capture()
            elif key in (ord("u"), ord("U")):
                node.undo()
            elif key in (ord("m"), ord("M")):
                show_coverage = not show_coverage
            visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
            if was_visible and not visible:
                break                       # closed by the window manager
            was_visible = was_visible or visible
    finally:
        try:
            cv2.destroyWindow(WINDOW_NAME)
            cv2.waitKey(1)
        except cv2.error:
            pass


def main(args: list[str] | None = None) -> None:
    """
    Run the calibration GUI: the node spins in a background thread, the window on the main thread.

    ROS parameters: config (cfg.yaml path, required), auto_replace, output_dir, capture_timeout_s.
    A bad config or parameter value ends it with the line "calibrate: <reason>" and exit status 1.
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
        # Stop the spin loop and join it before tearing down, and never call executor.shutdown():
        # in rclpy 7.1 it destroys the executor's guard conditions while the loop may be building a
        # wait set (which can then wait for ever) or a callback in the pool may still trigger them.
        stop.set()
        spinner.join()
        node.destroy_node()
        rclpy.try_shutdown()


def _compose(state: GuiState, show_coverage: bool) -> np.ndarray:
    """
    Return the window canvas: review mode while a capture's review is showing, else live mode.
    """
    review = state.review
    if review is not None:
        return compose_review(review.frame, review.corners, review.kept, state.hud,
                              state.canvas_size)
    return compose_live(state.frame, state.hud, state.coverage, state.canvas_size, show_coverage)


def _spin(executor: MultiThreadedExecutor, stop: threading.Event) -> None:
    """
    Spin executor until stop is set or ROS shuts down; the body of main()'s executor thread.

    Every wait ends within _SPIN_WAIT_S, so the loop notices stop without being woken, even while
    nothing it waits on can fire (no stream, the "calib" group busy). Errors from entities torn
    down under it once stopping end the loop quietly; any other error is raised.
      @args:
        executor: the executor holding the node.
        stop: set to end the loop.
    """
    while not stop.is_set() and executor.context.ok():
        try:
            executor.spin_once(timeout_sec=_SPIN_WAIT_S)
        except Exception:
            if stop.is_set() or not executor.context.ok():
                return
            raise


def _key_code(code: int) -> int:
    """
    Return the key a cv2.waitKeyEx() code stands for: its low 16 bits, or -1 for no key.

    GTK adds the modifier state above bit 16 (with NumLock on, q arrives as 0x100071). Unlike
    waitKey's 8-bit mask, 16 bits keep X11 keysyms apart from the command keys: Left (0xFF51) and
    PageUp (0xFF55) would otherwise read as Q (quit) and U (undo).
    """
    return -1 if code == -1 else code & 0xFFFF


def _decode(data) -> np.ndarray | None:
    """
    Decode an image message's bytes to a mono uint8 frame; None if they hold no image.
    """
    buffer = np.frombuffer(data, dtype=np.uint8)
    if buffer.size == 0:
        return None
    try:
        return cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    except cv2.error:
        return None


def _save_png(path: Path, data: np.ndarray, frame: np.ndarray) -> None:
    """
    Save a capture as PNG: the camera's own bytes when they are PNG, else frame re-encoded.
    """
    if data[:len(_PNG_SIGNATURE)].tobytes() != _PNG_SIGNATURE:
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            raise OSError(f"could not encode {path.name} as PNG")
        data = encoded
    path.write_bytes(data.tobytes())


def _new_session_dir(root: Path, started: datetime) -> Path:
    """
    Create and return <root>/<YYYYmmdd-HHMMSS>/ with its frames/ folder.

    A session started in the same second as another gets a -2, -3, ... suffix.
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
