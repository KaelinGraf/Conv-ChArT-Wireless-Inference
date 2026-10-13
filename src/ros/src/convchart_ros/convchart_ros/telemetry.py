"""
Telemetry: a live debugging dashboard of the docking robot, drawn with OpenCV (the telemetry task).

    ros2 run convchart_ros telemetry [--ros-args -p snapshot:=dash.png -p headless:=true -p record:=out.mp4]

It subscribes, relative to its namespace, to image (the Pi's camera), dock/estimate (the EKF), fiducial/fix,
cmd_vel (the MPC's command), dock/stage, dock/mpc_info, drive/status (the serial bridge) and inference_result
(the laptop's poses), and publishes nothing. A control strip along the bottom calls the two arming services
(std_srvs/SetBool): ARM DRIVE (serial_bridge/arm), START DOCKING (the drive, then dock/arm), CANCEL DOCKING
(dock/arm false: back to IDLE, the drive still armed and holding still) and STOP (both false; the space bar does
the same). Each reply shows in the strip and the log. Panels: the camera with the pose pipeline's verdict; a top-down
view of the target frame (the estimate with its 1-sigma ellipse and trail, the recent fixes, the MPC's
predicted path, the reference); the stage, MPC, drive, EKF and vision readouts; a row per stream (rate,
messages, seconds since the last one, matched publishers), red once it goes stale and amber when the silence
is expected (no command while the stage disables it, no fix while the pipeline finds no pose); and the last
history_s seconds of x, y, theta, vx, vy (estimate +-1 sigma, fixes, reference) and of the command u_b, v_b,
r_b (with the stage's caps and a band of stages). Callbacks only store: the 150 Hz estimate is kept at
history_hz, and the dashboard is drawn at render_hz. In the window each camera frame is also drawn as it
comes, and only then: the newest only (the image subscription keeps one), and a frame older than the one on
show is dropped. The newest pose is drawn on it, so a moving board's overlay trails by about a frame (the
pipeline's ~50 ms). snapshot rewrites a PNG every snapshot_period_s, record
writes the render_hz frames to an mp4, headless runs without the window (snapshot or record required), and
image_topic '' leaves the camera out. The window is scaled to fill the screen (scale 0, the default; any
other scale is the canvas's factor). Without dock_msgs (the controller workspace not sourced) only the camera,
cmd_vel, drive status and poses are shown, and the dashboard says why. q or Esc in the window quits.
"""
from __future__ import annotations

from collections import deque
import math
import os
from pathlib import Path
import signal
import threading
import time

from convchart_interfaces.msg import DriveStatus, RosInferenceResult
from convchart_qos.qos import IMAGE_SUB_QOS, INFERENCE_RESULT_QOS
import cv2
from geometry_msgs.msg import Twist
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.event_handler import SubscriptionEventCallbacks
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy,
                       qos_policy_name_from_kind)
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import SetBool
import yaml

from convchart_ros.screen import fill_scale, screen_size

try:
    from dock_msgs.msg import DockEstimate, DockStage, FiducialFix, MpcInfo
except ImportError:
    DockEstimate = DockStage = FiducialFix = MpcInfo = None

WINDOW = "docking telemetry"
POLL_S = 0.005
QUIT_KEYS = (27, ord("q"), ord("Q"))
FAST_QOS = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
CMD_QOS = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.RELIABLE)
STATUS_QOS = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
POSE_QOS = QoSProfile(history=INFERENCE_RESULT_QOS.history, depth=INFERENCE_RESULT_QOS.depth,
                      reliability=INFERENCE_RESULT_QOS.reliability, durability=QoSDurabilityPolicy.VOLATILE)

STAGES = ("IDLE", "ACQUIRE", "APPROACH", "FINAL", "DOCKED", "HOLD", "ABORT")
DRIVE_STATES = ("NO_LINK", "NO_TELEMETRY", "DISARMED", "STALLED", "RUNNING")
ARMED = (3, 4)
IMU_HZ = 150.0
IMU_MIN = 0.9 * IMU_HZ
SOLVE_BUDGET_MS = 25.0
HOLD_AFTER_S = 0.5
GAP_S = 0.3
TRAIL_S = 10.0
FIX_TRAIL_S = 3.0

W, H = 1600, 960
CAMERA = (0, 0, 640, 480)
TOPDOWN = (0, 484, 640, 416)
STATUS = (644, 0, 396, 480)
STREAMS = (644, 484, 396, 416)
SERIES = (1044, 0, 556, 900)
CONTROLS = (0, 904, 1600, 56)
BUTTONS = (("ARM DRIVE", "arm_drive"), ("START DOCKING", "start"), ("CANCEL DOCKING", "cancel"),
           ("STOP  (space)", "stop"))
STOP_KEYS = (ord(" "),)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _bgr(hex_colour):
    v = int(hex_colour.lstrip("#"), 16)
    return v & 255, (v >> 8) & 255, v >> 16


PAGE, SURFACE, GRID, AXIS = _bgr("#0d0d0d"), _bgr("#1a1a19"), _bgr("#2c2c2a"), _bgr("#383835")
INK, INK2, MUTED, DIM = _bgr("#ffffff"), _bgr("#c3c2b7"), _bgr("#898781"), _bgr("#52514e")
EST, FIX, MPC = _bgr("#3987e5"), _bgr("#d95926"), _bgr("#199e70")
GOOD, WARN, CRIT = _bgr("#0ca30c"), _bgr("#fab219"), _bgr("#d03b3b")
STAGE_LOOK = ((AXIS, INK2), (DIM, INK), (MUTED, PAGE), (INK2, PAGE), (GOOD, INK), (WARN, PAGE), (CRIT, INK))
DRIVE_LOOK = ((CRIT, INK), (CRIT, INK), (AXIS, INK2), (WARN, PAGE), (GOOD, INK))
STATE_LOOK = {"fresh": GOOD, "stale": CRIT, "never": CRIT, "quiet": WARN, "off": DIM}


def put(img, s, x, y, colour=INK, scale=0.45, thick=1):
    cv2.putText(img, s, (int(x), int(y)), FONT, scale, colour, thick, cv2.LINE_AA)


def width(s, scale=0.45, thick=1):
    return cv2.getTextSize(s, FONT, scale, thick)[0][0]


def put_right(img, s, x, y, colour=INK, scale=0.45, thick=1):
    put(img, s, x - width(s, scale, thick), y, colour, scale, thick)


def outlined(img, s, x, y, colour=INK, scale=0.5, thick=1):
    cv2.putText(img, s, (int(x), int(y)), FONT, scale, PAGE, thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, (int(x), int(y)), FONT, scale, colour, thick, cv2.LINE_AA)


def fit(s, max_w, scale=0.45):
    if width(s, scale) <= max_w:
        return s
    while s and width(s + "...", scale) > max_w:
        s = s[:-1]
    return s + "..."


def pill(img, s, x, y, look, scale=0.45, thick=1):
    """A filled label with its baseline at y; returns its right edge."""
    (w, h), _ = cv2.getTextSize(s, FONT, scale, thick)
    cv2.rectangle(img, (int(x), int(y - h - 5)), (int(x + w + 12), int(y + 5)), look[0], -1)
    put(img, s, x + 6, y, look[1], scale, thick)
    return int(x + w + 12)


def panel(img, rect, title=None, note=None, note_colour=MUTED):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), SURFACE, -1)
    if title:
        put(img, title, x + 10, y + 18, INK2, 0.45)
    if note:
        put(img, note, x + 16 + width(title or "", 0.45), y + 18, note_colour, 0.4)


def nice_step(span, n):
    raw = max(span, 1e-12) / n
    mag = 10.0 ** math.floor(math.log10(raw))
    return next(m * mag for m in (1.0, 2.0, 2.5, 5.0, 10.0) if m * mag >= raw * 0.999)


def tick_label(v, step):
    if abs(v) < step / 2:
        return "0"
    mag = 10.0 ** math.floor(math.log10(step))
    d = max(0, -int(math.floor(math.log10(step) + 1e-9))) + (1 if round(step / mag, 3) == 2.5 and step < 10 else 0)
    return f"{v:+.{d}f}"


def segments(points, t):
    """Split a polyline where the stream it came from paused for longer than GAP_S."""
    if len(t) < 2:
        return [points] if len(t) else []
    return [p for p in np.split(points, np.where(np.diff(t) > GAP_S)[0] + 1) if len(p)]


def finite(rows):
    return rows[np.all(np.isfinite(rows), axis=1)]


def age_text(age):
    return "never" if age is None else f"{age:.1f} s" if age < 100 else f"{age / 60:.0f} min"


def load_overlay(config=""):
    """Return ((K, dist, (cols, rows), square_m), note) for the board overlay, or (None, why not).
    """
    path = Path(config) if config else None
    if path is None:
        from convchart_ros.launch_plan import find_repo
        repo = find_repo(Path(__file__).resolve())
        path = Path(repo) / "cfg" / "cfg.yaml" if repo else None
    if path is None or not path.is_file():
        return None, "no pipeline config found: the camera panel draws no board overlay"
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    cam, board = cfg.get("CAMERA") or {}, cfg.get("BOARD") or {}
    if cam.get("K") is None or board.get("square_length_m") is None:
        return None, f"{path} has no CAMERA.K or BOARD.square_length_m: no board overlay"
    cols, rows = (int(n) for n in board.get("squares", (5, 5)))
    model = (np.asarray(cam["K"], np.float64).reshape(3, 3), np.asarray(cam.get("dist") or [], np.float64),
             (cols, rows), float(board["square_length_m"]))
    return model, f"board overlay from {path}"


def board_overlay(pose, model):
    """Project the board with one pose (geometry_msgs/Pose, the board in the camera's optical
    frame).
    """
    K, dist, (cols, rows), sq = model
    q, t = pose.orientation, pose.position
    if t.z <= 0.0:
        return None
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], np.float64)
    corners = [(i, j) for j in range(1, rows) for i in range(1, cols)]
    pts = np.array([(0, 0), (cols, 0), (cols, rows), (0, rows), *corners, (cols / 2, 0), (0, rows / 2)], np.float64)
    obj = np.hstack([pts * sq, np.zeros((len(pts), 1))])
    cam = obj @ R.T + np.array([t.x, t.y, t.z])
    if np.any(cam[:, 2] <= 0.02) or np.any(np.hypot(cam[:, 0], cam[:, 1]) > 1.5 * cam[:, 2]):
        return None
    img, _ = cv2.projectPoints(obj, cv2.Rodrigues(R)[0], np.array([t.x, t.y, t.z]), K, dist)
    img = img.reshape(-1, 2)
    if not np.all(np.isfinite(img)) or np.any(np.abs(img) > 1e4):
        return None
    n = len(corners)
    return {"border": img[:4], "corners": img[4:4 + n], "origin": img[0], "x": img[4 + n], "y": img[5 + n]}


class Stream:
    """One subscription: its latest message, when it came, how many, how fast, who publishes it."""

    def __init__(self, topic, rate_hz, stale_s, enabled=True, keep=0, keep_every_s=0.0):
        self.topic, self.rate_hz, self.stale_s, self.enabled = topic, rate_hz, stale_s, enabled
        self.latest = self.last = None
        self.count, self.rate, self.publishers, self.qos_issue = 0, 0.0, 0, ""
        self.history = deque(maxlen=keep) if keep else None
        self.keep_every_s = keep_every_s
        self._counts = deque()

    def store(self, msg, now):
        self.latest, self.last, self.count = msg, now, self.count + 1
        if self.history is not None and (not self.history or now - self.history[-1][0] >= self.keep_every_s):
            self.history.append((now, msg))

    def sample_rate(self, now, window_s=2.0):
        self._counts.append((now, self.count))
        while len(self._counts) > 2 and now - self._counts[1][0] >= window_s:
            self._counts.popleft()
        t0, c0 = self._counts[0]
        self.rate = (self.count - c0) / (now - t0) if now - t0 > 0.2 else 0.0

    def state(self, now):
        if not self.enabled:
            return "off"
        if self.last is None:
            return "never"
        return "stale" if now - self.last > self.stale_s else "fresh"

    def age(self, now):
        return None if self.last is None else max(0.0, now - self.last)


class Telemetry(Node):

    def __init__(self, **overrides):
        super().__init__("telemetry")
        dyn = ParameterDescriptor(dynamic_typing=True)
        p = {k: self.declare_parameter(k, overrides.get(k, v), dyn).value for k, v in {
            "image_topic": "image", "headless": False, "snapshot": "", "snapshot_period_s": 1.0, "record": "",
            "render_hz": 10.0, "history_s": 20.0, "history_hz": 25.0, "scale": 0.0, "config": ""}.items()}
        self.headless, self.snapshot, self.record = bool(p["headless"]), str(p["snapshot"]), str(p["record"])
        self.snapshot_period_s, self.render_hz = float(p["snapshot_period_s"]), float(p["render_hz"])
        self.history_s, self.scale, history_hz = float(p["history_s"]), float(p["scale"]), float(p["history_hz"])
        if self.headless and not (self.snapshot or self.record):
            raise ValueError("headless:=true needs snapshot:=<out.png> or record:=<out.mp4>, or nothing would "
                             "come of it")
        if self.snapshot and Path(self.snapshot).suffix.lower() != ".png":
            raise ValueError(f"snapshot must be a .png file, got {self.snapshot!r}")
        if min(self.snapshot_period_s, self.render_hz, self.history_s, history_hz) <= 0 or self.scale < 0:
            raise ValueError("snapshot_period_s, render_hz, history_s and history_hz must be > 0, and scale >= 0 "
                             "(0 fills the screen)")
        self.dock = DockEstimate is not None
        image = str(p["image_topic"]).strip()
        n = int(self.history_s * 50) + 16
        self.streams = {
            "image": Stream(image or "image", 15.0, 0.5, bool(image)),
            "estimate": Stream("dock/estimate", IMU_HZ, 0.25, self.dock, int(self.history_s * history_hz * 1.25) + 16,
                               1.0 / history_hz),
            "fix": Stream("fiducial/fix", 15.0, 0.5, self.dock, n),
            "cmd": Stream("cmd_vel", 20.0, 0.5, True, n),
            "stage": Stream("dock/stage", 20.0, 0.5, self.dock, n),
            "mpc": Stream("dock/mpc_info", 20.0, 0.5, self.dock),
            "drive": Stream("drive/status", 2.0, 1.5),
            "pose": Stream("inference_result", 15.0, 0.5),
        }
        kinds = {"image": (CompressedImage, IMAGE_SUB_QOS), "estimate": (DockEstimate, FAST_QOS),
                 "fix": (FiducialFix, FAST_QOS), "cmd": (Twist, CMD_QOS), "stage": (DockStage, FAST_QOS),
                 "mpc": (MpcInfo, FAST_QOS), "drive": (DriveStatus, STATUS_QOS),
                 "pose": (RosInferenceResult, POSE_QOS)}
        self._lock = threading.Lock()
        for key, s in self.streams.items():
            if s.enabled:
                self._subscribe(s, *kinds[key])
        ns = self.get_namespace()
        self.namespace = ns if ns.endswith("/") else ns + "/"
        self.domain = self.context.get_domain_id()
        self.rendered = self.snapshots = 0
        self._frame = self._frame_note = None
        self._frame_msg = None
        self._frame_ns = -1
        self._ranges, self._view, self._logged = {}, None, {}
        self._blank = np.full((H, W, 3), PAGE, np.uint8)
        self._canvas = self._blank
        self.drive_arm = self.create_client(SetBool, "serial_bridge/arm")
        self.dock_arm = self.create_client(SetBool, "dock/arm")
        self.buttons = {}
        self.overlay, overlay_note = load_overlay(str(p["config"]))
        self._frame_scale = 1.0
        self.get_logger().info(overlay_note)
        self.replies = deque(maxlen=2)
        self.get_logger().info(f"telemetry of {self.namespace}: "
                               + ", ".join(s.topic for s in self.streams.values() if s.enabled)
                               + (f"; snapshot to {self.snapshot}" if self.snapshot else "")
                               + (f"; recording to {self.record}" if self.record else "")
                               + ("; headless" if self.headless else ""))
        if not self.dock:
            self.get_logger().warning("dock_msgs is not importable (source the controller workspace's "
                                      "install/setup.bash): no EKF, stage or MPC; showing the camera, cmd_vel, "
                                      "drive status and poses only")

    def _subscribe(self, s, kind, qos):
        def on_message(msg):
            now = time.monotonic()
            with self._lock:
                s.store(msg, now)

        def on_matched(info):
            s.publishers = info.current_count

        def on_incompatible(info):
            s.qos_issue = qos_policy_name_from_kind(info.last_policy_kind)

        events = SubscriptionEventCallbacks(matched=on_matched, incompatible_qos=on_incompatible)
        self.create_subscription(kind, s.topic, on_message, qos, event_callbacks=events)

    def states(self, now=None) -> dict:
        """Each stream's state: off, never, fresh, stale, or quiet (silent, and the stage or the pipeline says why)."""
        now = time.monotonic() if now is None else now
        st = {k: s.state(now) for k, s in self.streams.items()}
        stage, pose = self.streams["stage"].latest, self.streams["pose"].latest
        if st["stage"] == "fresh" and not stage.command_enabled:
            for k in ("cmd", "mpc"):
                if st[k] in ("stale", "never"):
                    st[k] = "quiet"
        if st["pose"] == "fresh" and pose.reason and st["fix"] in ("stale", "never"):
            st["fix"] = "quiet"
        return st

    def log_changes(self, states) -> None:
        for k, state in states.items():
            was = self._logged.get(k)
            if state == was or state in ("never", "off"):
                continue
            self._logged[k] = state
            s = self.streams[k]
            if state == "stale":
                self.get_logger().warning(f"{s.topic}: no message for {s.stale_s:g} s")
            elif state == "fresh":
                self.get_logger().info(f"{s.topic}: " + ("flowing again" if was else "first message"))

    def act(self, action) -> None:
        """Run a control-strip action: ARM DRIVE, START DOCKING (the drive first), CANCEL DOCKING or STOP."""
        if action == "stop":
            self._call(self.dock_arm, False)
            self._call(self.drive_arm, False)
        elif action == "cancel":
            self._call(self.dock_arm, False)
        elif action == "arm_drive":
            self._call(self.drive_arm, True)
        elif action == "start":
            self._call(self.drive_arm, True, then=lambda: self._call(self.dock_arm, True))

    def _call(self, client, value, then=None) -> None:
        """Call one arming service without waiting; record the reply and, on success, run then."""
        name = client.srv_name
        if not client.service_is_ready():
            self._reply(f"{name}: no service (is its node up, on domain {self.domain}?)", CRIT)
            return

        def done(future):
            try:
                result = future.result()
            except Exception as error:          # noqa: BLE001
                self._reply(f"{name} {value}: {error}", CRIT)
                return
            self._reply(f"{name} {value}: {result.message or ('ok' if result.success else 'refused')}",
                        GOOD if result.success else CRIT)
            if result.success and then is not None:
                then()

        self.get_logger().info(f"{name} -> {value}")
        client.call_async(SetBool.Request(data=value)).add_done_callback(done)

    def _reply(self, s, colour) -> None:
        if colour == CRIT:
            self.get_logger().warning(s)
        else:
            self.get_logger().info(s)
        with self._lock:
            self.replies.append((time.strftime("%H:%M:%S  ") + s, colour))

    def render(self, now=None) -> np.ndarray:
        now = time.monotonic() if now is None else now
        with self._lock:
            for s in self.streams.values():
                s.sample_rate(now)
            hist = {k: list(s.history) for k, s in self.streams.items() if s.history is not None}
            replies = list(self.replies)
        st = self.states(now)
        img = self._blank.copy()
        self._draw_camera(img, now, st)
        self._draw_topdown(img, now, st, hist)
        self._draw_status(img, now, st)
        self._draw_streams(img, now, st)
        self._draw_series(img, now, st, hist)
        self._draw_controls(img, replies)
        self._canvas = img
        self.rendered += 1
        return img

    def render_frame(self, now=None) -> np.ndarray:
        """The last full drawing with the camera panel drawn again: a new frame between the render_hz redraws."""
        now = time.monotonic() if now is None else now
        img = self._canvas.copy()
        self._draw_camera(img, now, self.states(now))
        self.rendered += 1
        return img

    def _draw_controls(self, img, replies):
        x, y, w, h = CONTROLS
        panel(img, (x, y, w, h))
        bx, by, bh = x + 10, y + 8, h - 16
        for label, action in BUTTONS:
            bw = width(label, 0.55) + 28
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), {"start": GOOD, "stop": CRIT}.get(action, AXIS), -1)
            put(img, label, bx + 14, by + bh // 2 + 6, INK, 0.55)
            self.buttons[action] = (bx, by, bx + bw, by + bh)
            bx += bw + 12
        bx += 12
        for i, client in enumerate((self.drive_arm, self.dock_arm)):
            ready = client.service_is_ready()
            cv2.circle(img, (bx + 5, y + 18 + 22 * i), 5, GOOD if ready else CRIT, -1, cv2.LINE_AA)
            put(img, client.srv_name + ("" if ready else "  (no service)"), bx + 16, y + 23 + 22 * i,
                INK2 if ready else MUTED, 0.42)
        bx += 300
        for i, (s, colour) in enumerate(replies):
            put(img, fit(s, x + w - 10 - bx, 0.42), bx, y + 23 + 22 * i, colour, 0.42)

    def _draw_camera(self, img, now, st):
        x, y, w, h = CAMERA
        s = self.streams["image"]
        cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), SURFACE, -1)
        msg = s.latest
        ns = -1 if msg is None or msg is self._frame_msg else stamp_ns(msg.header.stamp)
        self._frame_msg = msg
        if ns > self._frame_ns or 0 <= ns < self._frame_ns - 1_000_000_000:
            self._frame_ns = ns
            frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                self._frame_note = f"cannot decode the image ({msg.format or 'no format'})"
            else:
                fh, fw = frame.shape[:2]
                k = min(w / fw, h / fh)
                if k != 1.0:
                    frame = cv2.resize(frame, (max(1, int(fw * k)), max(1, int(fh * k))),
                                       interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
                self._frame, self._frame_note, self._frame_scale = frame, f"{fw}x{fh}", k
        state = st["image"]
        if self._frame is not None:
            fh, fw = self._frame.shape[:2]
            ox, oy = x + (w - fw) // 2, y + (h - fh) // 2
            view = img[oy:oy + fh, ox:ox + fw]
            view[:] = self._frame if state == "fresh" else (self._frame * 0.35).astype(np.uint8)
            pose = self.streams["pose"].latest
            if (self.overlay is not None and state == "fresh" and st["pose"] == "fresh" and pose is not None
                    and not pose.reason):
                try:
                    self._draw_board(img, pose, ox, oy, self._frame_scale)
                except (cv2.error, ValueError, OverflowError) as e:
                    self.get_logger().warning(f"board overlay skipped: {e}", throttle_duration_sec=10.0)
        head = f"{s.topic}   {s.rate:.1f} Hz" + (f"   {self._frame_note}" if self._frame_note else "")
        outlined(img, head, x + 8, y + 20, INK2, 0.45)
        centre = {"off": ("camera off (image_topic is empty)", MUTED),
                  "never": (f"waiting for {self.namespace}{s.topic}", CRIT),
                  "stale": (f"NO IMAGE FOR {age_text(s.age(now))}", CRIT)}.get(state)
        if centre:
            outlined(img, centre[0], x + (w - width(centre[0], 0.8, 2)) // 2, y + h // 2, centre[1], 0.8, 2)
        pose, pose_state = self.streams["pose"].latest, st["pose"]
        if pose is not None:
            if pose.reason:
                line, colour = f"pose: {pose.reason}", WARN
            else:
                line = (f"pose OK   rms {pose.rms:.2f} px   {pose.num_used} corners"
                        + ("" if pose.covariance_valid else "   covariance INVALID")
                        + ("   AMBIGUOUS" if pose.ambiguous else ""))
                colour = GOOD if pose.covariance_valid and not pose.ambiguous else WARN
            if pose_state != "fresh":
                line, colour = line + f"   ({age_text(self.streams['pose'].age(now))} old)", MUTED
            outlined(img, fit(line, w - 16, 0.5), x + 8, y + h - 12, colour, 0.5)

    def _draw_board(self, img, pose, ox, oy, k):
        """Overlay the newest pose on the frame on show: its own, or (the pipeline taking ~50 ms)
        the one before.
        """
        pts = board_overlay(pose.pose.pose, self.overlay)
        if pts is None:
            return

        def px(p):
            return int(round(ox + k * p[0])), int(round(oy + k * p[1]))

        cv2.polylines(img, [np.array([px(p) for p in pts["border"]], np.int32)], True,
                      WARN if pose.ambiguous else INK, 2, cv2.LINE_AA)
        for p in pts["corners"]:
            cv2.circle(img, px(p), 3, WARN, -1, cv2.LINE_AA)
        origin = px(pts["origin"])
        for tip, label, colour in ((pts["x"], "x", CRIT), (pts["y"], "y", GOOD)):
            cv2.arrowedLine(img, origin, px(tip), colour, 3, cv2.LINE_AA, tipLength=0.2)
            outlined(img, label, px(tip)[0] + 6, px(tip)[1] + 6, colour, 0.7, 2)
        outlined(img, "0", origin[0] - 18, origin[1] - 8, INK, 0.7, 2)

    def _draw_topdown(self, img, now, st, hist):
        x, y, w, h = TOPDOWN
        panel(img, (x, y, w, h), "TOP-DOWN", "target frame [m], x right, y up")
        legend = (("estimate", EST, "dot"), ("fix", FIX, "dot"), ("MPC plan", MPC, "line"),
                  ("reference", INK2, "ring"))
        lx = x + w - 10
        for name, colour, mark in reversed(legend):
            lx -= width(name, 0.4)
            put(img, name, lx, y + 18, INK2, 0.4)
            lx -= 14
            if mark == "line":
                cv2.line(img, (lx - 4, y + 14), (lx + 8, y + 14), colour, 2, cv2.LINE_AA)
            elif mark == "ring":
                cv2.circle(img, (lx + 2, y + 14), 4, colour, 1, cv2.LINE_AA)
            else:
                cv2.circle(img, (lx + 2, y + 14), 4, colour, -1, cv2.LINE_AA)
            lx -= 14
        px0, py0, px1, py1 = x + 10, y + 28, x + w - 10, y + h - 26
        if not self.dock:
            msg = "dock_msgs not found: no estimate to draw"
            put(img, msg, (px0 + px1 - width(msg, 0.6)) // 2, (py0 + py1) // 2, CRIT, 0.6)
            return
        est = self.streams["estimate"].latest
        stage, mpc = self.streams["stage"].latest, self.streams["mpc"].latest
        if est is not None and not np.all(np.isfinite(est.x[:3])):
            est = None
        trail = finite(np.array([m.x[:2] for t, m in hist["estimate"] if now - t <= TRAIL_S]).reshape(-1, 2))
        fixes = finite(np.array([(m.x, m.y) for t, m in hist["fix"] if now - t <= FIX_TRAIL_S]).reshape(-1, 2))
        plan = np.zeros((0, 3))
        if mpc is not None and st["mpc"] == "fresh" and len(mpc.y_pred) >= 3 and len(mpc.y_pred) % 3 == 0:
            plan = finite(np.asarray(mpc.y_pred, np.float64).reshape(-1, 3))
        ref = np.asarray(stage.y_ref, np.float64) if stage is not None else None
        pts = [np.zeros((1, 2)), trail, fixes, plan[:, :2]]
        if est is not None:
            pts.append(np.asarray(est.x[:2]).reshape(1, 2))
        if ref is not None:
            pts.append(ref[:2].reshape(1, 2))
        pts = finite(np.concatenate(pts))
        aspect = (px1 - px0) / (py1 - py0)
        lo, hi = pts.min(0), pts.max(0)
        centre = (lo + hi) / 2
        half_x = max((hi[0] - lo[0]) / 2 * 1.2, (hi[1] - lo[1]) / 2 * 1.2 * aspect, 0.25 * aspect)
        if self._view is None or half_x > self._view[1]:
            self._view = [centre, half_x]
        else:
            (c0, h0) = self._view
            self._view = [c0 + 0.3 * (centre - c0), h0 + 0.1 * (half_x - h0)]
        (cx, cy), half_x = self._view
        k = (px1 - px0) / 2 / half_x
        mx, my = (px0 + px1) / 2, (py0 + py1) / 2

        def to_px(xy):
            xy = np.asarray(xy, np.float64).reshape(-1, 2)
            return np.stack([mx + (xy[:, 0] - cx) * k, my - (xy[:, 1] - cy) * k], 1).round().astype(np.int32)

        step = nice_step(2 * half_x, 6)
        for gx in np.arange(math.ceil((cx - half_x) / step) * step, cx + half_x, step):
            u = int(round(mx + (gx - cx) * k))
            cv2.line(img, (u, py0), (u, py1), AXIS if abs(gx) < step / 2 else GRID, 1)
            if u + 3 + width(tick_label(gx, step), 0.38) < px1:
                put(img, tick_label(gx, step), u + 3, py1 - 5, MUTED, 0.38)
        half_y = half_x / aspect
        for gy in np.arange(math.ceil((cy - half_y) / step) * step, cy + half_y, step):
            v = int(round(my - (gy - cy) * k))
            cv2.line(img, (px0, v), (px1, v), AXIS if abs(gy) < step / 2 else GRID, 1)
            if v - 14 > py0:
                put(img, tick_label(gy, step), px0 + 3, v - 3, MUTED, 0.38)
        roi = img[py0:py1, px0:px1]
        off = np.array([px0, py0], np.int32)
        (dx, dy), = to_px((0.0, 0.0)) - off
        cv2.line(roi, (int(dx), int(dy) - 16), (int(dx), int(dy) + 16), INK, 3, cv2.LINE_AA)
        put(roi, "dock", int(dx) + 6, int(dy) - 8, INK2, 0.4)
        if len(trail) > 1:
            cv2.polylines(roi, [to_px(trail) - off], False, EST, 1, cv2.LINE_AA)
        for u, v in to_px(fixes) - off:
            cv2.circle(roi, (int(u), int(v)), 3, FIX, -1, cv2.LINE_AA)
        if len(plan) > 1:
            pp = to_px(plan[:, :2]) - off
            cv2.polylines(roi, [pp], False, MPC, 2, cv2.LINE_AA)
            for u, v in pp[4::5]:
                cv2.circle(roi, (int(u), int(v)), 3, MPC, -1, cv2.LINE_AA)
        if ref is not None and np.all(np.isfinite(ref)):
            (u, v), = to_px(ref[:2]) - off
            cv2.circle(roi, (int(u), int(v)), 9, INK2, 1, cv2.LINE_AA)
            cv2.line(roi, (int(u), int(v)), (int(u + 18 * math.cos(ref[2])), int(v - 18 * math.sin(ref[2]))),
                     INK2, 1, cv2.LINE_AA)
        if est is None:
            put(img, "no estimate: the EKF publishes after its first fix" if self.streams["estimate"].latest is None
                else "the estimate is not finite", x + 10, y + h - 8, CRIT, 0.42)
            return
        stale = st["estimate"] != "fresh"
        colour = MUTED if stale else EST
        (u, v), = to_px(est.x[:2]) - off
        sx, sy = (math.sqrt(max(est.p_diag[i], 0.0)) * k for i in (0, 1))
        if sx >= 1 or sy >= 1:
            cv2.ellipse(roi, (int(u), int(v)), (max(1, int(sx)), max(1, int(sy))), 0, 0, 360, colour, 1, cv2.LINE_AA)
        th = float(est.x[2])
        cv2.line(roi, (int(u), int(v)), (int(u + 26 * math.cos(th)), int(v - 26 * math.sin(th))), colour, 2,
                 cv2.LINE_AA)
        cv2.circle(roi, (int(u), int(v)), 7, SURFACE, -1, cv2.LINE_AA)
        cv2.circle(roi, (int(u), int(v)), 5, colour, -1, cv2.LINE_AA)
        line = f"x {est.x[0]:+.3f} m   y {est.x[1]:+.3f} m   theta {math.degrees(th):+.1f} deg"
        if ref is not None:
            line += f"   to ref {math.hypot(ref[0] - est.x[0], ref[1] - est.x[1]):.3f} m"
        put(img, line, x + 10, y + h - 8, CRIT if stale else INK, 0.42)
        if stale:
            put_right(img, f"estimate stale {age_text(self.streams['estimate'].age(now))}", x + w - 10, y + h - 8,
                      CRIT, 0.42)

    def _draw_status(self, img, now, st):
        x, y, w, h = STATUS
        panel(img, (x, y, w, h))
        left, right, row = x + 10, x + w - 10, y + 22

        def header(name, key):
            nonlocal row
            put(img, name, left, row, MUTED, 0.42)
            state = st[key]
            if state != "fresh":
                note = {"off": "off", "never": "no message yet", "quiet": "no pose" if key == "fix" else "command off",
                        "stale": f"stale {age_text(self.streams[key].age(now))}"}[state]
                put(img, note, left + width(name, 0.42) + 10, row, STATE_LOOK[state], 0.42)
            row += 24
            return state == "fresh"

        def line(s, colour=INK, scale=0.45):
            nonlocal row
            put(img, fit(s, right - left, scale), left, row, colour, scale)
            row += 20

        if not self.dock:
            put(img, "STAGE / MPC / EKF", left, row, MUTED, 0.42)
            row += 28
            pill(img, "dock_msgs NOT FOUND", left, row, (CRIT, INK), 0.55, 1)
            row += 28
            for s in ("Build controller_ws's dock_msgs and source", "controller_ws/install/setup.bash, then",
                      "restart: the EKF, stage and MPC need it."):
                line(s, INK2)
            row += 10
        else:
            fresh = header("STAGE", "stage")
            stage = self.streams["stage"].latest
            if stage is not None:
                tint = INK if fresh else MUTED
                look = STAGE_LOOK[stage.stage] if stage.stage < len(STAGE_LOOK) else (CRIT, INK)
                name = STAGES[stage.stage] if stage.stage < len(STAGES) else f"STAGE {stage.stage}"
                edge = pill(img, name, left, row + 4, look if fresh else (AXIS, MUTED), 0.8, 2)
                pill(img, "COMMAND ON" if stage.command_enabled else "COMMAND OFF", edge + 10, row + 2,
                     ((INK2, PAGE) if stage.command_enabled else (AXIS, INK2)) if fresh else (AXIS, MUTED), 0.5)
                row += 32
                line(f"v_max {stage.v_max:.3f} m/s    r_max {stage.r_max:.3f} rad/s", tint)
                r = stage.y_ref
                line(f"y_ref  x {r[0]:+.3f}  y {r[1]:+.3f} m  theta {math.degrees(r[2]):+.1f} deg", tint)
            row += 8
            fresh = header("MPC", "mpc")
            mpc = self.streams["mpc"].latest
            if mpc is not None:
                tint = INK if fresh else MUTED
                put(img, f"solve {mpc.solve_ms:.2f} ms", left, row,
                    CRIT if fresh and mpc.solve_ms > SOLVE_BUDGET_MS else tint)
                put(img, f"exit {mpc.exit_flag}", left + 140, row, CRIT if fresh and mpc.exit_flag < 1 else tint)
                put(img, f"slack {mpc.slack:.2g}", left + 230, row, tint)
                row += 24
                if mpc.fallback:
                    edge = pill(img, "FALLBACK", left, row, (CRIT, INK) if fresh else (AXIS, MUTED), 0.45)
                else:
                    edge = left + width("no fallback") + 4
                    put(img, "no fallback", left, row, MUTED)
                put(img, f"u  {mpc.u[0]:+.3f}  {mpc.u[1]:+.3f}  {mpc.u[2]:+.3f}", edge + 12, row, tint)
                row += 20
            row += 8
        fresh = header("DRIVE", "drive")
        drive = self.streams["drive"].latest
        if drive is not None:
            tint = INK if fresh else MUTED
            name = DRIVE_STATES[drive.state] if drive.state < len(DRIVE_STATES) else f"STATE {drive.state}"
            look = DRIVE_LOOK[drive.state] if drive.state < len(DRIVE_LOOK) else (CRIT, INK)
            edge = pill(img, name, left, row + 2, look if fresh else (AXIS, MUTED), 0.55)
            armed = "ARMED" if drive.state in ARMED else "not armed"
            put(img, f"{armed}   want_armed {'yes' if drive.want_armed else 'no'}", edge + 12, row + 2, tint)
            row += 26
            line(f"cmd_age {drive.cmd_age:.2f} s   telemetry_age {drive.telemetry_age:.2f} s", tint)
            line(drive.detail, INK2 if fresh else MUTED, 0.42)
        row += 8
        if self.dock:
            fresh = header("EKF", "estimate")
            est = self.streams["estimate"].latest
            if est is not None:
                tint = INK if fresh else MUTED
                imu_ok = est.imu_rate >= IMU_MIN
                edge = pill(img, f"IMU {est.imu_rate:.0f} Hz", left, row + 2,
                            ((GOOD, INK) if imu_ok else (CRIT, INK)) if fresh else (AXIS, MUTED), 0.55)
                put(img, f"fix_age {est.fix_age:.2f} s", edge + 12, row + 2,
                    CRIT if fresh and est.fix_age > HOLD_AFTER_S else tint)
                put_right(img, "fix accepted" if est.last_fix_accepted else "fix REJECTED", right, row + 2,
                          (GOOD if est.last_fix_accepted else CRIT) if fresh else MUTED)
                row += 26
                if fresh and not imu_ok:
                    line(f"below {IMU_MIN:.0f} Hz: docking will not start", CRIT, 0.42)
                sd = np.sqrt(np.maximum(np.asarray(est.p_diag, np.float64), 0.0))
                line(f"NIS {est.last_nis:.2f}    sd  x {sd[0] * 1e3:.1f}  y {sd[1] * 1e3:.1f} mm  theta "
                     f"{math.degrees(sd[2]):.2f} deg", tint, 0.42)
            row += 8
            fresh = header("VISION FIX", "fix")
            fix = self.streams["fix"].latest
            if fix is not None:
                tint = INK if fresh else MUTED
                line(f"range {fix.range:.3f} m   roll {math.degrees(fix.roll):+.1f}   pitch "
                     f"{math.degrees(fix.pitch):+.1f} deg", tint)
                flags = ("cov valid" if fix.covariance_valid else "cov INVALID") + ("   AMBIGUOUS" * fix.ambiguous)
                line(flags, (GOOD if fix.covariance_valid and not fix.ambiguous else WARN) if fresh else MUTED)

    def _draw_streams(self, img, now, st):
        x, y, w, h = STREAMS
        panel(img, (x, y, w, h), "STREAMS", f"domain {self.domain}   namespace {self.namespace}")
        cols = (x + 30, x + 196, x + 262, x + 336, x + w - 10)
        row = y + 44
        put(img, "topic", cols[0], row, MUTED, 0.4)
        for col, name in zip(cols[1:], ("Hz", "msgs", "last", "pubs")):
            put_right(img, name, col, row, MUTED, 0.4)
        row += 8
        notes = []
        for key, s in self.streams.items():
            row += 24
            state = st[key]
            look = STATE_LOOK[state]
            cv2.circle(img, (x + 16, row - 5), 5, look, -1, cv2.LINE_AA)
            put(img, fit(s.topic, cols[1] - cols[0] - 50), cols[0], row, INK if s.enabled else MUTED)
            if not s.enabled:
                put_right(img, "off", cols[3], row, MUTED)
                continue
            put_right(img, f"{s.rate:.1f}", cols[1], row, INK2)
            put_right(img, f"{s.count}", cols[2], row, INK2)
            put_right(img, age_text(s.age(now)), cols[3], row, INK if state == "fresh" else look)
            if s.qos_issue:
                put_right(img, "QoS!", cols[4], row, CRIT)
                notes.append((f"QoS! {s.topic}: a publisher's {s.qos_issue} does not match", CRIT))
            else:
                put_right(img, str(s.publishers), cols[4], row, CRIT if s.publishers == 0 else INK2)
        row += 30
        if st["cmd"] == "quiet" or st["mpc"] == "quiet":
            notes.append(("amber: silent as expected, the command is off", WARN))
        if st["fix"] == "quiet":
            notes.append(("amber: no fix, as the pipeline finds no pose", WARN))
        if not self.dock:
            notes.append(("off: dock_msgs is not importable", MUTED))
        notes.append(("red: silent past its limit, or never heard", MUTED))
        notes.append(("pubs: matched publishers (0: nobody publishes it)", MUTED))
        for s, colour in notes:
            put(img, fit(s, w - 20, 0.4), x + 10, row, colour, 0.4)
            row += 18
        footer = (f"drawn at {self.render_hz:g} Hz" if self.headless else
                  f"drawn per frame + {self.render_hz:g} Hz;  space STOP;  q or Esc quits")
        put(img, footer, x + 10, y + h - 10, MUTED, 0.4)

    def _draw_series(self, img, now, st, hist):
        x, y, w, h = SERIES
        panel(img, (x, y, w, h))
        span = self.history_s
        px0, px1 = x + 58, x + w - 10
        est_t = np.array([t for t, _ in hist.get("estimate", ())])
        est_x = np.array([m.x for _, m in hist.get("estimate", ())]).reshape(-1, 5)
        est_sd = np.sqrt(np.maximum(np.array([m.p_diag for _, m in hist.get("estimate", ())]).reshape(-1, 5), 0.0))
        fix_t = np.array([t for t, _ in hist.get("fix", ())])
        fix_v = np.array([(m.x, m.y, m.theta) for _, m in hist.get("fix", ())]).reshape(-1, 3)
        stage_t = np.array([t for t, _ in hist.get("stage", ())])
        stage_ref = np.array([m.y_ref for _, m in hist.get("stage", ())]).reshape(-1, 3)
        stage_cap = np.array([(m.v_max, m.v_max, m.r_max) for _, m in hist.get("stage", ())]).reshape(-1, 3)
        cmd_t = np.array([t for t, _ in hist["cmd"]])
        cmd_v = np.array([(m.linear.x, m.linear.y, m.angular.z) for _, m in hist["cmd"]]).reshape(-1, 3)
        deg = (1.0, 1.0, math.degrees(1.0), 1.0, 1.0)

        def legend(items, row):
            lx = px1
            for name, colour, mark in reversed(items):
                lx -= width(name, 0.4)
                put(img, name, lx, row, INK2, 0.4)
                lx -= 16
                if mark == "dot":
                    cv2.circle(img, (lx + 5, row - 4), 3, colour, -1, cv2.LINE_AA)
                else:
                    cv2.line(img, (lx - 2, row - 4), (lx + 11, row - 4), colour, 2 if mark == "line" else 1,
                             cv2.LINE_AA)
                lx -= 14

        est_state = st["estimate"]
        put(img, "EKF STATE", x + 10, y + 18, INK2, 0.45)
        put(img, "dock/estimate" + ("" if est_state == "fresh" else f"  {est_state}"), x + 104, y + 18,
            MUTED if est_state == "fresh" else STATE_LOOK[est_state], 0.4)
        legend((("estimate +-1 sd", EST, "line"), ("fix", FIX, "dot"), ("reference", INK2, "thin")), y + 18)
        rows = [("x [m]", 0), ("y [m]", 1), ("theta [deg]", 2), ("vx [m/s]", 3), ("vy [m/s]", 4)]
        top, sh = y + 26, 98
        for title, i in rows:
            layers = []
            if len(est_t):
                layers.append(("band", est_t, est_x[:, i] * deg[i], est_sd[:, i] * deg[i], EST))
            if i < 3 and len(stage_t):
                layers.append(("thin", stage_t, stage_ref[:, i] * deg[i], None, INK2))
            if i < 3 and len(fix_t):
                layers.append(("dot", fix_t, fix_v[:, i] * deg[i], None, FIX))
            if len(est_t):
                layers.append(("line", est_t, est_x[:, i] * deg[i], None, EST))
            readout, colour = "", INK
            est = self.streams["estimate"].latest
            if est is not None:
                v, sd = est.x[i] * deg[i], math.sqrt(max(est.p_diag[i], 0.0)) * deg[i]
                readout = f"{v:+.3f}  +-{sd:.3f}" if i != 2 else f"{v:+.2f}  +-{sd:.2f}"
                colour = INK if est_state == "fresh" else CRIT
            self._strip(img, (x, top, w, sh), px0, px1, now, span, title, readout, colour, layers,
                        (0.02, 0.02, 1.0, 0.02, 0.02)[i], f"ekf{i}")
            top += sh
        top += 6
        cmd_state = st["cmd"]
        put(img, "MPC COMMAND", x + 10, top + 18, INK2, 0.45)
        put(img, "cmd_vel" + ("" if cmd_state == "fresh" else f"  {cmd_state}"), x + 118, top + 18,
            MUTED if cmd_state == "fresh" else STATE_LOOK[cmd_state], 0.4)
        legend((("command", MPC, "line"), ("cap (stage)", MUTED, "thin")), top + 18)
        top += 26
        cmd = self.streams["cmd"].latest
        for title, i, unit in (("u_b [m/s]", 0, "m/s"), ("v_b [m/s]", 1, "m/s"), ("r_b [rad/s]", 2, "rad/s")):
            layers = []
            if len(stage_t):
                layers.append(("thin", stage_t, stage_cap[:, i], None, MUTED))
                layers.append(("thin", stage_t, -stage_cap[:, i], None, MUTED))
            if len(cmd_t):
                layers.append(("line", cmd_t, cmd_v[:, i], None, MPC))
            readout, colour = "", INK
            if cmd is not None:
                v = (cmd.linear.x, cmd.linear.y, cmd.angular.z)[i]
                readout, colour = f"{v:+.3f} {unit}", (INK if cmd_state == "fresh" else STATE_LOOK[cmd_state])
            self._strip(img, (x, top, w, sh), px0, px1, now, span, title, readout, colour, layers,
                        (0.05, 0.05, 0.1)[i], f"cmd{i}")
            top += sh
        top += 4
        put_right(img, "stage", px0 - 6, top + 17, MUTED, 0.38)
        cv2.rectangle(img, (px0, top), (px1, top + 24), GRID, -1)
        hist_stage = hist.get("stage", ())
        if len(stage_t):
            runs, start = [], 0
            for j in range(1, len(stage_t) + 1):
                if j == len(stage_t) or hist_stage[j][1].stage != hist_stage[start][1].stage \
                        or stage_t[j] - stage_t[j - 1] > GAP_S:
                    runs.append((stage_t[start], stage_t[j - 1], hist_stage[start][1].stage))
                    start = j
            for t0, t1, sid in runs:
                u0 = max(px0, int(px1 - (now - t0) / span * (px1 - px0)))
                u1 = int(px1 - (now - t1) / span * (px1 - px0))
                if u1 <= px0:
                    continue
                bg, fg = STAGE_LOOK[sid] if sid < len(STAGE_LOOK) else (CRIT, INK)
                cv2.rectangle(img, (u0, top), (max(u0 + 1, u1), top + 24), bg, -1)
                name = STAGES[sid] if sid < len(STAGES) else str(sid)
                if u1 - u0 > width(name, 0.38) + 8:
                    put(img, name, u0 + 4, top + 17, fg, 0.38)
        top += 24
        for k in range(int(span // 5) + 1):
            u = int(px1 - k * 5 / span * (px1 - px0))
            label = "now" if k == 0 else f"-{k * 5} s"
            put(img, label, u - width(label, 0.38) // 2 if k else u - width(label, 0.38), top + 16, MUTED, 0.38)

    def _strip(self, img, rect, px0, px1, now, span, title, readout, colour, layers, min_span, key):
        x, y, w, h = rect
        py0, py1 = y + 20, y + h - 6
        put(img, title, px0, y + 14, INK2, 0.42)
        if readout:
            put_right(img, readout, px1, y + 14, colour, 0.42)
        t_lo = now - span - 1.0
        vals = []
        for kind, t, v, sd, _ in layers:
            m = (t >= t_lo) & np.isfinite(v) & (np.isfinite(sd) if kind == "band" else True)
            if kind == "band":
                vals += [v[m] - sd[m], v[m] + sd[m]]
            else:
                vals.append(v[m])
        vals = np.concatenate(vals) if vals else np.zeros(0)
        vals = vals[np.isfinite(vals)]
        lo, hi = (float(vals.min()), float(vals.max())) if len(vals) else (-min_span / 2, min_span / 2)
        if hi - lo < min_span:
            lo, hi = (lo + hi - min_span) / 2, (lo + hi + min_span) / 2
        pad = 0.08 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        prev = self._ranges.get(key)
        if prev is not None and lo >= prev[0] and hi <= prev[1]:
            lo, hi = prev[0] + 0.05 * (lo - prev[0]), prev[1] + 0.05 * (hi - prev[1])
        self._ranges[key] = (lo, hi)
        for k in range(int(span // 5) + 1):
            u = int(px1 - k * 5 / span * (px1 - px0))
            cv2.line(img, (u, py0), (u, py1), GRID, 1)
        step = nice_step(hi - lo, 3)
        for v in np.arange(math.ceil(lo / step) * step, hi, step):
            vv = int(round(py1 - (v - lo) / (hi - lo) * (py1 - py0)))
            cv2.line(img, (px0, vv), (px1, vv), AXIS if abs(v) < step / 2 else GRID, 1)
            put_right(img, tick_label(v, step), px0 - 6, vv + 4, MUTED, 0.38)
        roi = img[py0:py1 + 1, px0:px1 + 1]
        sx, sy = (px1 - px0) / span, (py1 - py0) / (hi - lo)

        def to_px(t, v):
            return np.stack([(span - (now - t)) * sx, (hi - v) * sy], 1).round().astype(np.int32)

        for kind, t, v, sd, c in layers:
            m = (t >= t_lo) & np.isfinite(v) & (np.isfinite(sd) if kind == "band" else True)
            if not m.any():
                continue
            t, v = t[m], v[m]
            if kind == "band":
                sd = sd[m]
                over = roi.copy()
                for up, down in zip(segments(to_px(t, v + sd), t), segments(to_px(t, v - sd), t)):
                    cv2.fillPoly(over, [np.concatenate([up, down[::-1]])], c, cv2.LINE_AA)
                cv2.addWeighted(over, 0.3, roi, 0.7, 0, dst=roi)
            elif kind == "dot":
                for u, vv in to_px(t, v):
                    cv2.circle(roi, (int(u), int(vv)), 2, c, -1, cv2.LINE_AA)
            else:
                cv2.polylines(roi, segments(to_px(t, v), t), False, c, 2 if kind == "line" else 1, cv2.LINE_AA)


def stamp_ns(stamp) -> int:
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def run(node: Telemetry, stop: threading.Event | None = None) -> None:
    """The draw loop, on the calling (main) thread: OpenCV's window must live there."""
    writer, written, period = None, 0, 1.0 / node.render_hz
    next_draw = next_record = time.monotonic()
    next_snapshot, shown = 0.0, -1
    if not node.headless:
        screen = screen_size()
        node.scale = node.scale or fill_scale(W, H, screen)
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        if screen is not None:
            cv2.moveWindow(WINDOW, max(0, (screen[0] - round(W * node.scale)) // 2), 0)
        cv2.setMouseCallback(WINDOW, _on_mouse, node)
        node.get_logger().info(f"window {round(W * node.scale)}x{round(H * node.scale)} (scale {node.scale:.2f})")
    try:
        while rclpy.ok() and not (stop is not None and stop.is_set()):
            now = time.monotonic()
            count = node.streams["image"].count
            new_frame = not node.headless and count != shown
            if now < next_draw and not new_frame:
                if node.headless:
                    time.sleep(min(next_draw - now, 0.05))
                elif _key(node, cv2.waitKey(max(1, int(min(next_draw - now, POLL_S) * 1e3))) & 0xFF):
                    return
                continue
            shown = count
            if now >= next_draw:
                next_draw = max(next_draw + period, now)
                states = node.states(now)
                node.log_changes(states)
                vis = node.render(now)
            else:
                vis = node.render_frame(now)
            if node.snapshot and now >= next_snapshot:
                next_snapshot = now + node.snapshot_period_s
                ok, png = cv2.imencode(".png", vis, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                if ok:
                    path = Path(node.snapshot)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_name(f".{path.name}.tmp")
                    tmp.write_bytes(png.tobytes())
                    os.replace(tmp, path)
                    if not node.snapshots:
                        node.get_logger().info(f"snapshot written to {path}")
                    node.snapshots += 1
            if node.record and now >= next_record:
                next_record = max(next_record + period, now)
                if writer is None:
                    Path(node.record).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(node.record, cv2.VideoWriter_fourcc(*"mp4v"), node.render_hz, (W, H))
                    if not writer.isOpened():
                        raise RuntimeError(f"could not open {node.record} for writing")
                writer.write(vis)
                written += 1
            if not node.headless:
                if node.scale != 1.0:
                    vis = cv2.resize(vis, None, fx=node.scale, fy=node.scale,
                                     interpolation=cv2.INTER_AREA if node.scale < 1 else cv2.INTER_CUBIC)
                cv2.imshow(WINDOW, vis)
                if _key(node, cv2.waitKey(1) & 0xFF) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    return
    finally:
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        if writer is not None:
            writer.release()
            node.get_logger().info(f"wrote {written} frames to {node.record}")
        if not node.headless:
            cv2.destroyAllWindows()


def _key(node: Telemetry, key: int) -> bool:
    """Handle a key: the space bar is STOP. Return whether the key quits."""
    if key in STOP_KEYS:
        node.act("stop")
    return key in QUIT_KEYS


def _on_mouse(event, x, y, flags, node: Telemetry) -> None:
    """A left click on a control-strip button runs its action. The window shows the canvas times scale."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    k = node.scale or 1.0
    px, py = x / k, y / k
    for action, (x0, y0, x1, y1) in node.buttons.items():
        if x0 <= px <= x1 and y0 <= py <= y1:
            node.act(action)
            return


def _spin(executor: SingleThreadedExecutor, stop: threading.Event) -> None:
    try:
        while not stop.is_set() and rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
    except ExternalShutdownException:
        pass
    except Exception:
        if rclpy.ok():
            raise


def main(args=None) -> None:
    """Run the dashboard. A bad parameter ends it with "telemetry: <reason>" and exit status 1."""
    rclpy.init(args=args)
    try:
        node = Telemetry()
    except (OSError, ValueError) as error:
        rclpy.try_shutdown()
        raise SystemExit(f"telemetry: {error}") from None
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    stop = threading.Event()
    spinner = threading.Thread(target=_spin, args=(executor, stop), name="telemetry_executor", daemon=True)
    spinner.start()
    try:
        run(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        stop.set()
        spinner.join()
        node.destroy_node()
        rclpy.try_shutdown()
