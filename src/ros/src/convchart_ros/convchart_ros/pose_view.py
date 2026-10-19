"""
Pose view: the 'image' stream through the conv-chart pipeline, drawn with OpenCV (the demo task).

    ros2 run convchart_ros pose_view --ros-args -p config:=<cfg.yaml> [-p record:=out.mp4] [-p headless:=true]

Green circle + number: identified corner and its ID. Red circle: detected, not identified. Axes and the
blue outline: the PnP pose (x red, y green, z blue, one square long). With CAMERA.K null a nominal K is
used, so the overlay is right but the pose numbers are rough. q or Esc in the window quits. record
writes every drawn frame to an mp4 at record_fps; headless runs without a window (record required).
exit_after_idle_s > 0 ends the node once the stream has been silent that long after its first frame,
which is how a launched demo ends with its video. Frames queue (queue_size) rather than being skipped,
so a recording keeps every frame the pipeline keeps up with; the HUD counts any dropped.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
import signal
import sys
import threading
import time

from convchart_qos.qos import IMAGE_SUB_QOS
import cv2
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

WINDOW = "conv-chart pose view"
QUIT_KEYS = (27, ord("q"), ord("Q"))
NOMINAL_K = np.array([[560.0, 0, 319.5], [0, 560.0, 239.5], [0, 0, 1]])


def text(img, s, row, colour=(255, 255, 255)):
    org = (8, 24 + 24 * row)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA)


class PoseView(Node):

    def __init__(self, **overrides):
        super().__init__("pose_view")
        dyn = ParameterDescriptor(dynamic_typing=True)
        p = {k: self.declare_parameter(k, overrides.get(k, v), dyn).value for k, v in {
            "config": "", "scale": 1.5, "record": "", "record_fps": 10.0, "headless": False,
            "exit_after_idle_s": 0.0, "queue_size": 30}.items()}
        if not p["config"]:
            raise ValueError("set config: the cfg.yaml the pipeline runs with")
        config = Path(str(p["config"])).resolve()
        if not config.is_file():
            raise ValueError(f"config {config} does not exist")
        self.scale, self.record = float(p["scale"]), str(p["record"])
        self.record_fps, self.headless = float(p["record_fps"]), bool(p["headless"])
        self.idle_s = float(p["exit_after_idle_s"])
        if self.headless and not self.record:
            raise ValueError("headless:=true needs record:=<out.mp4>, or nothing would come of it")
        src = config.parent.parent / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        import inference as inf
        self.pipe = inf.inference_pipeline(str(config))
        self.nominal_k = self.pipe.camera.K is None
        self.K = NOMINAL_K if self.nominal_k else self.pipe.camera.K
        self.dist = np.zeros(5) if self.nominal_k or self.pipe.camera.dist is None else self.pipe.camera.dist
        self.sq = self.pipe.board.square_length_m or 1.0
        self.unit = "m" if self.pipe.board.square_length_m else "sq"
        lattice = inf._CANON if self.pipe.board.n == 4 else inf.canon_lattice(self.pipe.board.n)
        (x0, y0), (x1, y1) = lattice.min(0), lattice.max(0)
        self.outline = np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]], np.float64) * self.sq
        self.K_disp = self.K.copy()
        self.K_disp[0, 0] *= self.scale
        self.K_disp[1, 1] *= self.scale
        self.K_disp[:2, 2] = self.scale * (self.K[:2, 2] + 0.5) - 0.5
        self._queue: deque = deque(maxlen=int(p["queue_size"]))
        self._cv = threading.Condition()
        self.received = self.drops = self.shown = 0
        self.last_arrival = None
        sub = self.create_subscription(CompressedImage, "image", self._on_image, IMAGE_SUB_QOS)
        self.get_logger().info(f"pose view on {sub.topic_name}"
                               + (f", recording to {self.record}" if self.record else "")
                               + (", headless" if self.headless else ""))

    def _on_image(self, msg: CompressedImage) -> None:
        with self._cv:
            if len(self._queue) == self._queue.maxlen:
                self.drops += 1
            self._queue.append(msg)
            self.received += 1
            self.last_arrival = time.monotonic()
            self._cv.notify()

    def next_frame(self, timeout_s: float):
        with self._cv:
            if not self._queue:
                self._cv.wait(timeout_s)
            return self._queue.popleft() if self._queue else None

    def idle(self) -> bool:
        with self._cv:
            return (self.idle_s > 0 and self.last_arrival is not None and not self._queue
                    and time.monotonic() - self.last_arrival > self.idle_s)

    def draw(self, gray, res, infer_ms, hz):
        s = self.scale
        vis = cv2.cvtColor(cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR),
                           cv2.COLOR_GRAY2BGR)
        for c in res["corners"]:
            pt = (int(round(s * (c["x"] + 0.5) - 0.5)), int(round(s * (c["y"] + 0.5) - 0.5)))
            if c["index"] is None:
                cv2.circle(vis, pt, 4, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(vis, pt, 5, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(vis, str(c["index"]), (pt[0] + 6, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 0), 1, cv2.LINE_AA)
        n_id = sum(c["index"] is not None for c in res["corners"])
        text(vis, f"{res['reason'] or 'OK'}   ids {n_id}/{len(res['corners'])}   "
                  f"infer {infer_ms:.0f} ms   {hz:.1f} Hz" + (f"   dropped {self.drops}" if self.drops else ""), 0)
        if res["rvec"] is not None:
            rvec, tvec = res["rvec"], res["tvec"]
            img_pts, _ = cv2.projectPoints(self.outline, rvec, tvec, self.K_disp, self.dist)
            cv2.polylines(vis, [np.round(img_pts).astype(np.int32)], True, (255, 128, 0), 2, cv2.LINE_AA)
            cv2.drawFrameAxes(vis, self.K_disp, self.dist, rvec, tvec, self.sq, 2)
            t = np.ravel(tvec)
            ang = np.degrees(np.ravel(rvec))
            text(vis, f"t = ({t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f}) {self.unit}   range "
                      f"{np.linalg.norm(t):.2f} {self.unit}   rms {res['rms']:.2f} px"
                      + ("   AMBIGUOUS" if res["ambiguous"] else ""), 1,
                 (0, 200, 255) if res["ambiguous"] else (255, 255, 255))
            text(vis, f"rvec = ({ang[0]:+.1f}, {ang[1]:+.1f}, {ang[2]:+.1f}) deg", 2)
        if self.nominal_k:
            text(vis, "UNCALIBRATED K (nominal f = 560 px): pose numbers approximate", 3, (0, 200, 255))
        return vis


def run(node: PoseView) -> None:
    """Run the view loop on the calling (main) thread, where OpenCV's window must live.

    Args:
        node: PoseView node to show.
    """
    writer, hz, t_prev, t_log = None, 0.0, None, time.monotonic()
    if not node.headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        while rclpy.ok():
            msg = node.next_frame(0.05)
            if msg is None:
                if node.idle():
                    node.get_logger().info(f"stream idle for {node.idle_s:g} s; {node.shown} frames shown")
                    return
                if not node.headless and (cv2.waitKey(5) & 0xFF) in QUIT_KEYS:
                    return
                continue
            gray = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            t0 = time.perf_counter()
            res = node.pipe.run_inference(gray, K=node.K, dist=node.dist)
            infer_ms = (time.perf_counter() - t0) * 1e3
            now = time.monotonic()
            if t_prev is not None:
                dt = max(now - t_prev, 1e-3)
                hz = 0.9 * hz + 0.1 / dt if hz else 1.0 / dt
            t_prev = now
            vis = node.draw(gray, res, infer_ms, hz)
            node.shown += 1
            if node.record:
                if writer is None:
                    Path(node.record).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(node.record, cv2.VideoWriter_fourcc(*"mp4v"), node.record_fps,
                                             (vis.shape[1], vis.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f"could not open {node.record} for writing")
                writer.write(vis)
            if now - t_log > 3.0:
                t_log = now
                node.get_logger().info(f"{node.shown} frames  {hz:.1f} Hz  infer {infer_ms:.0f} ms  "
                                       f"{res['reason'] or 'OK'}  dropped {node.drops}")
            if not node.headless:
                cv2.imshow(WINDOW, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in QUIT_KEYS or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    return
    finally:
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        if writer is not None:
            writer.release()
            node.get_logger().info(f"wrote {node.shown} frames to {node.record}")
        if not node.headless:
            cv2.destroyAllWindows()


def _spin(executor: MultiThreadedExecutor, stop: threading.Event) -> None:
    try:
        while not stop.is_set() and rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
    except ExternalShutdownException:
        pass


def main(args=None) -> None:
    """Run the pose view; a bad parameter ends it with "pose_view: <reason>" and exit status 1.

    Args:
        args: Command-line arguments passed to rclpy.init().
    """
    rclpy.init(args=args)
    try:
        node = PoseView()
    except (OSError, ValueError) as error:
        rclpy.try_shutdown()
        raise SystemExit(f"pose_view: {error}") from None
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    stop = threading.Event()
    spinner = threading.Thread(target=_spin, args=(executor, stop), name="pose_view_executor", daemon=True)
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
