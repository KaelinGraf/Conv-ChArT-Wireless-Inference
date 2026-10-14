"""
The telemetry dashboard (telemetry.py) over ROS 2: every stream tracked, a stopped one flagged, nothing published.

    python3 -m pytest src/ros/src/convchart_ros/test/test_telemetry.py -q

Synthetic traffic on every topic the dashboard reads, at the robot's rates and with its publishers' QoS (a PNG
frame at 15 Hz, the estimate at 200 Hz, the command, stage and MPC diagnostics at 20 Hz, fixes and poses at
15 Hz, the bridge's latched status at 2 Hz), drives a headless dashboard in process, through main() without
dock_msgs, and through `ros2 launch ... task:=telemetry`. Isolated domain, discovery limited to localhost and a
private namespace, so not even a synthetic cmd_vel can reach a robot. Needs dock_msgs (source the controller
workspace's install/setup.bash) for all but the test without it.
"""
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

import cv2
from convchart_interfaces.msg import DriveStatus, RosInferenceResult
from convchart_qos.qos import image_qos_pub, INFERENCE_RESULT_QOS
from geometry_msgs.msg import Twist
import numpy as np
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import SetBool

from convchart_ros import telemetry

DOMAIN_ID = 31 + os.getpid() % 20
if DOMAIN_ID == 42:
    DOMAIN_ID = 50
NAMESPACE = "/telemetry_test"
TIMEOUT_S = 20.0
HAVE_DOCK = telemetry.DockEstimate is not None
needs_dock = pytest.mark.skipif(not HAVE_DOCK, reason="needs dock_msgs: source controller_ws/install/setup.bash")
CMD_QOS = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.RELIABLE)
STATUS_QOS = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=5, reliability=QoSReliabilityPolicy.RELIABLE,
                        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)


@pytest.fixture(scope="module")
def ros():
    saved = os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE")
    os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    rclpy.init(args=["--ros-args", "-r", f"__ns:={NAMESPACE}"], domain_id=DOMAIN_ID)
    try:
        yield
    finally:
        rclpy.shutdown()
        if saved is None:
            os.environ.pop("ROS_AUTOMATIC_DISCOVERY_RANGE", None)
        else:
            os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = saved


def camera_frame():
    """A 640x480 checkerboard over a gradient, as the Pi's mono8 PNG."""
    gray = np.tile(np.linspace(40, 200, 640).astype(np.uint8), (480, 1))
    for i in range(6):
        for j in range(6):
            cv2.rectangle(gray, (200 + 40 * i, 120 + 40 * j), (239 + 40 * i, 159 + 40 * j), 250 * ((i + j) % 2), -1)
    return gray


class Robot:
    """What the docking stack publishes, at its rates and with its publishers' QoS, from a background thread."""

    def __init__(self, node):
        from dock_msgs.msg import DockEstimate, DockStage, FiducialFix, MpcInfo
        self.types = {"estimate": DockEstimate, "fix": FiducialFix, "stage": DockStage, "mpc": MpcInfo}
        self.node = node
        self.pubs = {
            "image": node.create_publisher(CompressedImage, "image", image_qos_pub()),
            "estimate": node.create_publisher(DockEstimate, "dock/estimate", 10),
            "fix": node.create_publisher(FiducialFix, "fiducial/fix", 10),
            "cmd": node.create_publisher(Twist, "cmd_vel", CMD_QOS),
            "stage": node.create_publisher(DockStage, "dock/stage", 10),
            "mpc": node.create_publisher(MpcInfo, "dock/mpc_info", 10),
            "drive": node.create_publisher(DriveStatus, "drive/status", STATUS_QOS),
            "pose": node.create_publisher(RosInferenceResult, "inference_result", INFERENCE_RESULT_QOS),
        }
        ok, png = cv2.imencode(".png", camera_frame())
        self.image = CompressedImage(format="mono8; png compressed mono8", data=png.tobytes())
        self.active = set(self.pubs)
        self.command_enabled, self.reason, self.imu_rate, self.nan = True, "", 150.0, False
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        T = self.types
        t0 = next_t = time.monotonic()
        i = 0
        while not self.stop.is_set():
            t = time.monotonic() - t0
            x = (-1.2 + 0.05 * t, 0.05 * math.sin(t), 0.05 * math.sin(0.5 * t), 0.05, 0.05 * math.cos(t))
            if self.nan:
                x = (math.nan,) * 5
            send = {"estimate": lambda: T["estimate"](x=x, p_diag=[1.6e-5, 1.6e-5, 3e-5, 1e-4, 1e-4],
                                                      imu_rate=self.imu_rate, fix_age=0.04, last_nis=1.2,
                                                      last_fix_accepted=True)}
            if i % 10 == 0:
                send["stage"] = lambda: T["stage"](stage=2 if self.command_enabled else 0, y_ref=[-0.3, 0.0, 0.0],
                                                   v_max=0.3, r_max=0.5, command_enabled=self.command_enabled)
                send["cmd"] = lambda: Twist()
                plan = [v for k in range(40) for v in (x[0] + 0.01 * k, x[1], x[2])]
                send["mpc"] = lambda: T["mpc"](u=[0.05, 0.0, 0.01], y_pred=plan, slack=0.0, exit_flag=1,
                                               solve_ms=0.25, fallback=False)
            if i % 13 == 0:
                send["image"] = lambda: self.image
                send["fix"] = lambda: T["fix"](x=x[0], y=x[1], theta=x[2], covariance_valid=True, range=1.5)
                send["pose"] = lambda: RosInferenceResult(reason=self.reason, rms=0.3, num_used=24,
                                                          covariance_valid=True)
            if i % 100 == 0:
                send["drive"] = lambda: DriveStatus(state=DriveStatus.RUNNING, detail="RUNNING", want_armed=True,
                                                    cmd_age=0.03, telemetry_age=0.02)
            for key, make in send.items():
                if key in self.active:
                    self.pubs[key].publish(make())
            i += 1
            next_t += 0.005
            time.sleep(max(0.0, next_t - time.monotonic()))

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=TIMEOUT_S)


@pytest.fixture
def dashboard(ros, tmp_path):
    node = telemetry.Telemetry(headless=True, snapshot=str(tmp_path / "dash.png"), snapshot_period_s=0.2,
                               capture_dir=str(tmp_path / "captures"))
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            executor.spin_once(timeout_sec=0.05)

    threads = [threading.Thread(target=spin, daemon=True),
               threading.Thread(target=telemetry.run, args=(node, stop), daemon=True)]
    for thread in threads:
        thread.start()
    try:
        yield node
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=TIMEOUT_S)
        node.destroy_node()


def wait_for(predicate, what, timeout_s=TIMEOUT_S):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail(f"timed out waiting for {what}")
        time.sleep(0.02)


def snapshot_showing(path, predicate, what, timeout_s=5.0):
    """The newest snapshot once it shows what is expected: one written just after a change may predate it."""
    deadline = time.monotonic() + timeout_s
    while True:
        img = cv2.imread(str(path)) if path.exists() else None
        if img is not None:
            assert img.shape == (telemetry.H, telemetry.W, 3)
            if predicate(img):
                return img
        if time.monotonic() > deadline:
            pytest.fail(f"no snapshot showing {what}")
        time.sleep(0.05)


def red_pixels(img, rect):
    x, y, w, h = rect
    return int(np.all(img[y:y + h, x:x + w] == telemetry.CRIT, axis=2).sum())


def shows_the_camera(img):
    crop = (slice(60, 420), slice(40, 600))
    return np.mean(img[crop] == cv2.cvtColor(camera_frame(), cv2.COLOR_GRAY2BGR)[crop]) > 0.99


def test_a_missing_arm_service_is_reported_not_waited_on(dashboard):
    """With no bridge or controller up, START says so in the strip instead of hanging."""
    dashboard.act("start")
    wait_for(lambda: any("no service" in s for s, _ in dashboard.replies), "the missing service to be reported", 5.0)


def test_replies_of_either_severity_can_follow_each_other(dashboard):
    """A failed reply after a good one must not raise. rclpy refuses two severities from one logging line,
    and the exception, raised on the executor thread, once stopped the dashboard receiving anything."""
    for colour in (telemetry.GOOD, telemetry.CRIT, telemetry.GOOD):
        dashboard._reply("reply", colour)
    assert [c for _, c in dashboard.replies] == [telemetry.CRIT, telemetry.GOOD]


def test_start_arms_the_drive_then_the_docking_and_stop_disarms_both(dashboard):
    """A click on START DOCKING arms the drive and only then the docking; STOP disarms both."""
    probe = rclpy.create_node("arm_probe")
    calls = []

    def serve(name):
        def on_call(request, response):
            calls.append((name, request.data))
            response.success = True
            return response
        probe.create_service(SetBool, name, on_call)

    serve("serial_bridge/arm")
    serve("dock/arm")
    executor = SingleThreadedExecutor()
    executor.add_node(probe)
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            executor.spin_once(timeout_sec=0.05)

    spinner = threading.Thread(target=spin, daemon=True)
    spinner.start()
    try:
        wait_for(lambda: dashboard.drive_arm.service_is_ready() and dashboard.dock_arm.service_is_ready(),
                 "both arming services to be seen")
        wait_for(lambda: "start" in dashboard.buttons, "the control strip to be drawn")
        x0, y0, x1, y1 = dashboard.buttons["start"]
        telemetry._on_mouse(cv2.EVENT_LBUTTONDOWN, (x0 + x1) // 2, (y0 + y1) // 2, 0, dashboard)
        wait_for(lambda: len(calls) == 2, "the drive and then the docking to be armed", 5.0)
        assert calls == [("serial_bridge/arm", True), ("dock/arm", True)]
        dashboard.act("stop")
        wait_for(lambda: len(calls) == 4, "both to be disarmed", 5.0)
        assert sorted(calls[2:]) == [("dock/arm", False), ("serial_bridge/arm", False)]
    finally:
        stop.set()
        spinner.join(timeout=TIMEOUT_S)
        probe.destroy_node()


def decoded(path, index, w, h, pix):
    """One frame of a capture's video, as rgba or bgr24."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{index})", "-vsync", "0",
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", pix, "-"], capture_output=True,
                         check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w, 4 if pix == "rgba" else 3)


@needs_dock
@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg")
def test_start_docking_records_the_window_and_each_panel_until_stop(dashboard, tmp_path):
    """START DOCKING records the whole window, the camera, and the EKF/MPC and top-down panels without their
    background, all frame for frame alike; a second START carries on; STOP ends it and the files are finished."""
    probe = rclpy.create_node("capture_probe")
    try:
        with Robot(probe):
            wait_for(lambda: all(s == "fresh" for s in dashboard.states().values()), "every stream to flow")
            dashboard.act("start")
            cap = dashboard.capture
            assert cap is not None and cap.recording and cap.folder.parent == tmp_path / "captures"
            wait_for(lambda: cap.elapsed(time.monotonic()) > 2.5, "2.5 s of capture")
            dashboard.act("start")
            assert dashboard.capture is cap and cap.recording
            dashboard.act("stop")
    finally:
        probe.destroy_node()
    assert not cap.recording and cap.wait(30.0) and not cap.errors(), cap.errors()
    wait_for(lambda: any("capture: dock_" in s for s, _ in dashboard.replies), "the capture's summary")
    record = json.loads((cap.folder / "capture.json").read_text())
    assert record["status"] == "complete" and record["stopped_by"] == "STOP"
    assert record["events"][1]["event"] == "stage APPROACH"
    n = record["frames"]
    assert n == round(30 * record["duration_s"]) and n >= 75
    files = sorted(p.name for p in cap.folder.iterdir())
    assert files == ["camera.mp4", "capture.json", "dashboard.mp4", "ekf_mpc_hud.mov", "topdown_hud.mov"]
    for key, (rect, kind) in telemetry.CAPTURE.items():
        out = record["outputs"][key]
        assert (out["width"], out["height"], out["alpha"], out["frames"]) == (rect[2], rect[3], kind == "bgra", n)
        assert out["pictures_drawn"] >= 0.8 * n, out
    k = n // 2
    full = decoded(cap.folder / "dashboard.mp4", k, telemetry.W, telemetry.H, "bgr24").astype(np.float32)
    for key in ("ekf_mpc_hud", "topdown_hud"):
        x, y, w, h = telemetry.CAPTURE[key][0]
        hud = decoded(cap.folder / f"{key}.mov", k, w, h, "rgba").astype(np.float32)
        alpha = hud[..., 3:] / 255.0
        assert alpha[3, 3, 0] == 0 and (alpha > 0.99).mean() > 0.01, key
        laid = hud[..., 2::-1] * alpha + np.array(telemetry.SURFACE, np.float32) * (1.0 - alpha)
        assert np.abs(laid - full[y:y + h, x:x + w]).mean() < 4.0, key
    x, y, w, h = telemetry.CAMERA
    camera = decoded(cap.folder / "camera.mp4", k, w, h, "bgr24").astype(np.float32)
    assert np.abs(camera - full[y:y + h, x:x + w]).mean() < 3.0
    expected = cv2.cvtColor(camera_frame(), cv2.COLOR_GRAY2BGR).astype(np.float32)
    assert np.abs(camera[60:420, 40:600] - expected[60:420, 40:600]).mean() < 6.0


@needs_dock
def test_the_dashboard_tracks_every_stream_and_flags_the_ones_that_stop(dashboard, tmp_path):
    from dock_msgs.msg import DockStage
    probe = rclpy.create_node("telemetry_probe")
    snap = tmp_path / "dash.png"
    with Robot(probe) as robot:
        wait_for(lambda: all(s == "fresh" for s in dashboard.states().values()), "every stream to flow")
        time.sleep(2.5)
        wait_for(lambda: all(s == "fresh" for s in dashboard.states().values()), "every stream still flowing")
        s = dashboard.streams
        assert 120 < s["estimate"].rate < 260 and 12 < s["cmd"].rate < 28 and 9 < s["image"].rate < 21
        assert s["estimate"].count > 3 * len(s["estimate"].history)
        assert all(stream.publishers == 1 and not stream.qos_issue for stream in s.values())
        assert s["stage"].latest.stage == DockStage.APPROACH and s["stage"].latest.command_enabled
        assert s["drive"].latest.state == DriveStatus.RUNNING and s["mpc"].latest.solve_ms == pytest.approx(0.25)
        assert len(s["mpc"].latest.y_pred) == 120 and s["pose"].latest.num_used == 24
        live = snapshot_showing(snap, lambda img: shows_the_camera(img) and red_pixels(img, telemetry.STREAMS)
                                == red_pixels(img, telemetry.STATUS) == 0, "the camera and no red")
        cv2.imwrite(str(tmp_path / "live.png"), live)

        robot.imu_rate = 100.0
        wait_for(lambda: s["estimate"].latest.imu_rate == 100.0, "the slow IMU rate to arrive", 5.0)
        slow = snapshot_showing(snap, lambda img: red_pixels(img, telemetry.STATUS) > 0, "the IMU rate flagged")
        cv2.imwrite(str(tmp_path / "imu_slow.png"), slow)
        robot.imu_rate, robot.nan = 150.0, True
        wait_for(lambda: math.isnan(s["estimate"].latest.x[0]), "a NaN estimate to arrive", 5.0)
        rendered = dashboard.rendered
        wait_for(lambda: dashboard.rendered > rendered + 3, "the dashboard to keep drawing through NaNs", 5.0)
        robot.nan = False
        wait_for(lambda: math.isfinite(s["estimate"].latest.x[0]), "the estimate to recover", 5.0)

        robot.active.discard("estimate")
        t_stop = time.monotonic()
        wait_for(lambda: dashboard.states()["estimate"] == "stale", "the estimate to go stale", 5.0)
        assert time.monotonic() - t_stop >= s["estimate"].stale_s
        states = dashboard.states()
        assert all(states[k] == "fresh" for k in states if k != "estimate"), states
        stale = snapshot_showing(snap, lambda img: red_pixels(img, telemetry.STREAMS) > 0, "the stale estimate in red")
        cv2.imwrite(str(tmp_path / "estimate_stale.png"), stale)

        robot.command_enabled, robot.reason = False, "too few corners"
        robot.active -= {"cmd", "mpc", "fix"}
        wait_for(lambda: [dashboard.states()[k] for k in ("cmd", "mpc", "fix")] == ["quiet"] * 3,
                 "cmd_vel, mpc_info and the fix to fall quiet as the stage and the pipeline expect", 5.0)
        assert dashboard.streams["stage"].latest.stage == DockStage.IDLE

        robot.active.clear()
        wait_for(lambda: all(dashboard.states()[k] == "stale" for k in ("image", "stage", "drive", "pose")),
                 "every stream to go stale once the robot stops", 5.0)
        snapshot_showing(snap, lambda img: not shows_the_camera(img), "the camera dimmed")
        for pub in robot.pubs.values():
            probe.destroy_publisher(pub)
        wait_for(lambda: all(stream.publishers == 0 for stream in s.values()), "every publisher to unmatch", 5.0)
    names = probe.get_publisher_names_and_types_by_node(dashboard.get_name(), dashboard.get_namespace())
    published = {name for name, _ in names}
    assert published <= {"/rosout", "/parameter_events"}, published
    assert dashboard.snapshots >= 5 and dashboard.rendered >= 40
    probe.destroy_node()


def test_each_frame_is_drawn_as_it_comes_and_one_older_than_the_frame_on_show_is_dropped(ros, tmp_path):
    """The picture only moves forward: a late older frame is dropped; a clock that restarted is followed."""
    node = telemetry.Telemetry(headless=True, snapshot=str(tmp_path / "dash.png"), capture=False)
    pub = rclpy.create_node("order_probe")
    executor = SingleThreadedExecutor()
    for n in (node, pub):
        executor.add_node(n)
    images = pub.create_publisher(CompressedImage, "image", image_qos_pub())
    ok, png = cv2.imencode(".png", camera_frame())

    def spin_until(done, what):
        deadline = time.monotonic() + TIMEOUT_S
        while not done():
            assert time.monotonic() < deadline, what
            executor.spin_once(timeout_sec=0.02)

    try:
        spin_until(lambda: images.get_subscription_count(), "the dashboard never subscribed")
        for n, (ms, shown) in enumerate(((5000, 5000), (5200, 5200), (5100, 5200), (5300, 5300), (2000, 2000)), 1):
            msg = CompressedImage(format="mono8; png compressed mono8", data=png.tobytes())
            msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(ms * 1_000_000, 1_000_000_000)
            images.publish(msg)
            spin_until(lambda: node.streams["image"].count == n, f"frame {ms} ms not received")
            node.render_frame()
            assert node._frame_ns == shown * 1_000_000, f"after the frame stamped {ms} ms"
    finally:
        pub.destroy_node()
        node.destroy_node()


def test_without_dock_msgs_it_still_shows_the_camera_and_says_why(ros, tmp_path):
    snap, video = tmp_path / "dash.png", tmp_path / "dash.mp4"
    env = {**os.environ, "ROS_DOMAIN_ID": str(DOMAIN_ID), "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST"}
    code = "import sys; sys.modules['dock_msgs'] = None; from convchart_ros.telemetry import main; main()"
    proc = subprocess.Popen([sys.executable, "-c", code, "--ros-args", "-r", f"__ns:={NAMESPACE}", "-p",
                             "headless:=true", "-p", f"snapshot:={snap}", "-p", "snapshot_period_s:=0.2", "-p",
                             f"record:={video}"], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    probe = rclpy.create_node("telemetry_probe")
    pub = probe.create_publisher(CompressedImage, "image", image_qos_pub())
    ok, png = cv2.imencode(".png", camera_frame())
    try:
        deadline = time.monotonic() + TIMEOUT_S
        while not (snap.exists() and shows_the_camera(cv2.imread(str(snap)))):
            assert proc.poll() is None, proc.communicate()[0]
            assert time.monotonic() < deadline, "no snapshot with the camera in it"
            pub.publish(CompressedImage(format="mono8; png compressed mono8", data=png.tobytes()))
            time.sleep(1 / 15)
        for _ in range(8):
            pub.publish(CompressedImage(format="mono8; png compressed mono8", data=png.tobytes()))
            time.sleep(1 / 15)
        os.killpg(proc.pid, signal.SIGINT)
        out = proc.communicate(timeout=TIMEOUT_S)[0]
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        probe.destroy_node()
    assert proc.returncode == 0, out
    assert "dock_msgs is not importable" in out and "image, cmd_vel, drive/status, inference_result" in out, out
    written = re.search(rf"wrote (\d+) frames to {re.escape(str(video))}", out)
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    assert written and ok and frame.shape == (telemetry.H, telemetry.W, 3), out
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert frames >= 3 and abs(frames - int(written.group(1))) <= 1


def _installed():
    try:
        from ament_index_python.packages import get_package_share_directory
        return (Path(get_package_share_directory("convchart_ros")) / "launch" / "convchart.launch.py").exists()
    except Exception:
        return False


@needs_dock
@pytest.mark.skipif(not _installed() or shutil.which("ros2") is None, reason="needs the built workspace and ros2")
def test_task_telemetry_launches_a_dashboard_that_publishes_nothing(ros, tmp_path):
    snap = tmp_path / "dash.png"
    env = {**os.environ, "ROS_DOMAIN_ID": str(DOMAIN_ID), "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST"}
    log = open(tmp_path / "launch.log", "w")
    proc = subprocess.Popen(["ros2", "launch", "convchart_ros", "convchart.launch.py", "task:=telemetry",
                             f"namespace:={NAMESPACE.strip('/')}", "headless:=true", f"snapshot:={snap}"], env=env,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    probe = rclpy.create_node("telemetry_probe")
    try:
        with Robot(probe):
            deadline = time.monotonic() + TIMEOUT_S
            while not (snap.exists() and shows_the_camera(cv2.imread(str(snap)))):
                assert proc.poll() is None, (tmp_path / "launch.log").read_text()[-3000:]
                assert time.monotonic() < deadline, "no snapshot with the camera in it"
                time.sleep(0.1)
            published = {name for name, _ in probe.get_publisher_names_and_types_by_node("telemetry", NAMESPACE)}
            subscribed = {name for name, _ in probe.get_subscriber_names_and_types_by_node("telemetry", NAMESPACE)}
        assert published <= {"/rosout", "/parameter_events"}, published
        assert {f"{NAMESPACE}/{t}" for t in ("image", "dock/estimate", "fiducial/fix", "cmd_vel", "dock/stage",
                                             "dock/mpc_info", "drive/status", "inference_result")} <= subscribed
        os.killpg(proc.pid, signal.SIGINT)
        assert proc.wait(TIMEOUT_S) == 0, (tmp_path / "launch.log").read_text()[-3000:]
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        log.close()
        probe.destroy_node()
    text = (tmp_path / "launch.log").read_text()
    note = f"telemetry dashboard of {NAMESPACE}, subscribing only, snapshots to {snap.resolve()} (headless)"
    assert note in text, text
