"""
Replay node: a recorded video or image folder published on 'image' the way the Pi camera node does.

    ros2 run convchart_ros replay --ros-args -r __ns:=/local -p source:=<video file | image dir> \
        [-p rate:=10.0] [-p loop:=1] [-p wait_for_subscribers:=1]

Frames are centre-cropped to 4:3, area-downscaled to size (640x480), PNG-encoded as mono8 and stamped
with the publish time, under the camera's QoS (convchart_qos.image_qos_pub, offering the deadline the
camera node offers). loop is how many passes to play, 0 for ever; the node exits after the last frame.
wait_for_subscribers holds the first frame back until that many subscribers have matched, so a
consumer still loading its model does not miss the start. It refuses to run in the root namespace
unless allow_root_namespace is set: there, on the robot's domain, recorded frames would reach the
robot's inference node and through it the controller.
"""
from __future__ import annotations

from pathlib import Path
import signal

from convchart_qos.qos import IMAGE_DEADLINE_CEILING_S, image_qos_pub
import cv2
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}
PNG_FORMAT = "mono8; png compressed mono8"


def frames(source: Path):
    """Yield the frames of a video file, or of the images in a folder in name order, as mono uint8.

    Args:
        source: Video file or folder of images.
    """
    if source.is_dir():
        for f in sorted(x for x in source.iterdir() if x.suffix.lower() in IMAGE_EXTS):
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                yield img
        return
    cap = cv2.VideoCapture(str(source))
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                return
            yield cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    finally:
        cap.release()


def fit(img, w: int, h: int):
    """Centre-crop to 4:3 and area-resize to (w, h), as the camera node's 1600x1200 -> 640x480.

    Args:
        img: Mono image.
        w: Output width, in pixels.
        h: Output height, in pixels.
    """
    H, W = img.shape
    k = min(W // 4, H // 3)
    img = img[(H - 3 * k) // 2:(H - 3 * k) // 2 + 3 * k, (W - 4 * k) // 2:(W - 4 * k) // 2 + 4 * k]
    return img if img.shape == (h, w) else cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


class Replay(Node):

    def __init__(self, **overrides):
        super().__init__("replay")
        dyn = ParameterDescriptor(dynamic_typing=True)
        p = {k: self.declare_parameter(k, overrides.get(k, v), dyn).value for k, v in {
            "source": "", "rate": 10.0, "loop": 1, "size": "640x480", "frame_id": "replay",
            "wait_for_subscribers": 1, "allow_root_namespace": False}.items()}
        self.source = Path(str(p["source"])).expanduser()
        self.rate, self.loop = float(p["rate"]), int(p["loop"])
        self.need = int(p["wait_for_subscribers"])
        self.frame_id = str(p["frame_id"])
        if not str(p["source"]):
            raise ValueError("set source: a video file or a folder of images")
        if not self.source.exists():
            raise ValueError(f"source {self.source} does not exist")
        if next(frames(self.source), None) is None:
            raise ValueError(f"source {self.source} holds no readable frames")
        if not self.rate > 0 or self.loop < 0 or self.need < 0:
            raise ValueError(f"need rate > 0, loop >= 0 and wait_for_subscribers >= 0, got "
                             f"{self.rate}, {self.loop}, {self.need}")
        self.w, self.h = (int(v) for v in str(p["size"]).lower().split("x"))
        if self.get_namespace() == "/" and not bool(p["allow_root_namespace"]):
            raise ValueError("refusing to publish recorded frames on /image, where the robot's inference "
                             "node listens: run in a namespace (-r __ns:=/local) or set "
                             "allow_root_namespace:=true")
        offered = min(IMAGE_DEADLINE_CEILING_S, 1.5 / self.rate)
        self.pub = self.create_publisher(CompressedImage, "image", image_qos_pub(deadline_s=offered))
        self.published = 0
        self.done = False
        self._frames = self._passes()
        self._waiting_logged = False
        self.create_timer(1.0 / self.rate, self._tick)
        self.get_logger().info(f"replaying {self.source} at {self.rate:g} Hz, "
                               f"{'for ever' if self.loop == 0 else f'{self.loop} pass(es)'}, "
                               f"on {self.pub.topic_name}")

    def _passes(self):
        n = 0
        while self.loop == 0 or n < self.loop:
            yield from frames(self.source)
            n += 1

    def _tick(self) -> None:
        if self.done:
            return
        if self.published == 0 and self.pub.get_subscription_count() < self.need:
            if not self._waiting_logged:
                self._waiting_logged = True
                self.get_logger().info(f"waiting for {self.need} subscriber(s) on {self.pub.topic_name}")
            return
        img = next(self._frames, None)
        if img is None:
            self.done = True
            self.get_logger().info(f"published {self.published} frames from {self.source}; done")
            return
        ok, png = cv2.imencode(".png", fit(img, self.w, self.h), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        msg = CompressedImage(format=PNG_FORMAT, data=png.tobytes())
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.pub.publish(msg)
        self.published += 1


def main(args=None) -> None:
    """Run the replay until its last frame; a bad parameter ends it with "replay: <reason>".

    Args:
        args: Command-line arguments passed to rclpy.init().
    """
    rclpy.init(args=args)
    try:
        node = Replay()
    except ValueError as error:
        rclpy.try_shutdown()
        raise SystemExit(f"replay: {error}") from None
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        while rclpy.ok() and not node.done:
            executor.spin_once(timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.destroy_node()
        rclpy.try_shutdown()
