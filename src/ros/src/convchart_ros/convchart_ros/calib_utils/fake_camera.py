"""
Fake camera node: the Pi camera's 'image' stream and 'capture' service, served from synthetic renders.

It stands in for the (deferred) Pi camera node during development and in tests. The board steps through
calib_synth.diverse_poses: the stream shows the current pose, and each capture returns the current pose at
full resolution with fresh noise, then moves on to the next pose. The ground truth a calibration should
recover (K_full, dist) and the poses are exposed as attributes.

    ros2 run convchart_ros fake_camera --ros-args -p config:=<cfg.yaml> -p blank_every:=5
"""
from __future__ import annotations

import array
import threading
import time
from typing import Any

from builtin_interfaces.msg import Time
from convchart_interfaces.srv import CaptureFrame
import cv2
import numpy as np
from numpy.typing import ArrayLike
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSLivelinessPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage
import yaml

from .calib_board import board_spec_from_cfg, BoardSpec, make_board
from .calib_synth import (blank_frame, DEFAULT_DIST, DEFAULT_FULL_SIZE, DEFAULT_K_FULL, DEFAULT_STREAM_SIZE,
                          diverse_poses, render_view)

NODE_NAME = "fake_camera"
FRAME_ID = "fake_camera"
PNG_FORMAT = "mono8; png compressed mono8"

# RELIABLE, VOLATILE, KEEP_LAST 1 as the interface specifies. The deadline and liveliness lease meet what
# convchart's inference node requests of its image publisher (deadline <= 0.2 s, lease <= 1 s), as the Pi
# camera does, so that node runs on the fake stream too; subscribers that ask for neither are unaffected.
IMAGE_QOS = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                       reliability=QoSReliabilityPolicy.RELIABLE,
                       durability=QoSDurabilityPolicy.VOLATILE,
                       deadline=Duration(seconds=0.2),
                       liveliness=QoSLivelinessPolicy.AUTOMATIC,
                       liveliness_lease_duration=Duration(seconds=1.0))

_CAPTURE_SEED = 1_000_000       # capture k uses noise seed _CAPTURE_SEED + k; the stream uses the pose index


class FakeCamera(Node):
    """
    Synthetic camera serving 'image' (preview stream) and 'capture' (full-resolution CaptureFrame stills).
    The stream publishes the current pose rendered at full resolution, resized to stream_size (INTER_AREA),
    as PNG at rate_hz. A capture waits reply_delay_s, renders the current pose at full resolution with
    fresh noise (or a blank frame on every blank_every-th capture), replies, and advances to the next pose
    (cycling). Timer and service sit in separate callback groups, so under a MultiThreadedExecutor a slow
    reply never stalls the stream.
      @args:
        board_spec: board to render.
        K_full: (3, 3) camera matrix at full resolution: the ground truth a calibration should recover.
        dist: distortion coefficients (OpenCV order), ground truth too.
        full_size: (W, H) of captures.
        stream_size: (W, H) of the 'image' stream.
        rate_hz: stream rate.
        blank_every: every Nth capture is a blank frame without the board (it still uses up a pose);
                     0 = never.
        reply_delay_s: how long a capture takes before it is taken and answered.
    """

    def __init__(self, board_spec: BoardSpec = BoardSpec(), K_full: ArrayLike = DEFAULT_K_FULL,
                 dist: ArrayLike = DEFAULT_DIST, full_size: tuple[int, int] = DEFAULT_FULL_SIZE,
                 stream_size: tuple[int, int] = DEFAULT_STREAM_SIZE, rate_hz: float = 10.0,
                 blank_every: int = 0, reply_delay_s: float = 0.0):
        if not rate_hz > 0:
            raise ValueError(f"rate_hz must be positive, got {rate_hz}")
        if blank_every < 0:
            raise ValueError(f"blank_every must be >= 0, got {blank_every}")
        if reply_delay_s < 0:
            raise ValueError(f"reply_delay_s must be >= 0, got {reply_delay_s}")
        board = make_board(board_spec)
        super().__init__(NODE_NAME)

        self.board_spec = board_spec
        self.board = board
        self.K_full = np.array(K_full, dtype=np.float64).reshape(3, 3)
        self.dist = np.array(dist, dtype=np.float64).ravel()
        self.full_size = (int(full_size[0]), int(full_size[1]))
        self.stream_size = (int(stream_size[0]), int(stream_size[1]))
        self.poses = diverse_poses(board_spec, self.K_full, self.full_size)
        self._blank_every = int(blank_every)
        self._reply_delay_s = float(reply_delay_s)

        self._lock = threading.Lock()
        self._pose_index = 0
        self._captures = 0
        self._stream_png: tuple[int, array.array] | None = None     # (pose index, PNG) last published

        stream_group = MutuallyExclusiveCallbackGroup()
        capture_group = MutuallyExclusiveCallbackGroup()
        self._image_pub = self.create_publisher(CompressedImage, "image", IMAGE_QOS,
                                                callback_group=stream_group)
        self._timer = self.create_timer(1.0 / rate_hz, self._publish_frame, callback_group=stream_group)
        self._capture_srv = self.create_service(CaptureFrame, "capture", self._capture,
                                                callback_group=capture_group)
        cols, rows = board_spec.squares
        self.get_logger().info(
            f"{len(self.poses)} poses of a {cols}x{rows} {board_spec.dictionary} board; captures "
            f"{self.full_size[0]}x{self.full_size[1]}, stream {self.stream_size[0]}x{self.stream_size[1]} at "
            f"{rate_hz:g} Hz, blank_every {self._blank_every}, reply_delay_s {self._reply_delay_s:g}")

    @property
    def pose_index(self) -> int:
        """
        Index into poses of the pose the stream shows and the next capture takes.
        """
        with self._lock:
            return self._pose_index

    def _publish_frame(self) -> None:
        """
        Stream timer: publish the current pose at the stream size. A pose is rendered once and its PNG reused
        until a capture moves on.
        """
        with self._lock:
            index, cached = self._pose_index, self._stream_png
        if cached is None or cached[0] != index:
            full = self._render(index, seed=index)
            frame = cv2.resize(full, self.stream_size, interpolation=cv2.INTER_AREA)
            cached = (index, _png(frame))
            with self._lock:
                self._stream_png = cached
        self._image_pub.publish(self._message(cached[1], self.get_clock().now().to_msg()))

    def _capture(self, request: CaptureFrame.Request,
                 response: CaptureFrame.Response) -> CaptureFrame.Response:
        """
        'capture' service: a full-resolution still of the current pose taken after the request (and after
        reply_delay_s), then the next pose.
        """
        if self._reply_delay_s > 0:
            time.sleep(self._reply_delay_s)
        stamp = self.get_clock().now().to_msg()
        with self._lock:
            self._captures += 1
            count, index = self._captures, self._pose_index
        blank = self._blank_every > 0 and count % self._blank_every == 0
        seed = _CAPTURE_SEED + count
        frame = blank_frame(self.full_size, seed=seed) if blank else self._render(index, seed=seed)
        response.image = self._message(_png(frame), stamp)
        response.success = True
        shown = "blank frame" if blank else f"pose {index + 1}/{len(self.poses)}"
        response.message = f"fake capture {count}: {shown}"
        with self._lock:
            self._pose_index = (index + 1) % len(self.poses)
        self.get_logger().info(response.message)
        return response

    def _render(self, index: int, seed: int) -> np.ndarray:
        rvec, tvec = self.poses[index]
        frame, _, _ = render_view(self.board, self.board_spec, self.K_full, self.dist, rvec, tvec,
                                  self.full_size, seed=seed)
        return frame

    @staticmethod
    def _message(png: array.array, stamp: Time) -> CompressedImage:
        msg = CompressedImage(format=PNG_FORMAT)
        msg.header.stamp = stamp
        msg.header.frame_id = FRAME_ID
        msg.data = png
        return msg


def _png(frame: np.ndarray) -> array.array:
    """
    PNG bytes of a mono8 frame, as the array type a uint8[] message field takes without copying per element.
    """
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return array.array("B", encoded.tobytes())


def _load_board_spec(config_path: str) -> BoardSpec:
    """
    The board of a config file's CALIBRATION.board block (defaults for whatever is missing).
      @args:
        config_path: path of a cfg.yaml.
    """
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file) or {}
    return board_spec_from_cfg((cfg.get("CALIBRATION") or {}).get("board"))


def _read_parameters() -> dict[str, Any]:
    """
    FakeCamera arguments from the ROS parameters config, rate_hz, blank_every and reply_delay_s, declared on
    a short-lived node with the camera's name so global and node-scoped overrides both apply.
    """
    params = rclpy.create_node(NODE_NAME, start_parameter_services=False, enable_rosout=False)
    try:
        def value(name: str, default: Any) -> Any:
            return params.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=True)).value

        config = str(value("config", "") or "")
        args = {"rate_hz": float(value("rate_hz", 10.0)), "blank_every": int(value("blank_every", 0)),
                "reply_delay_s": float(value("reply_delay_s", 0.0))}
    finally:
        params.destroy_node()
    if config:
        args["board_spec"] = _load_board_spec(config)
    return args


def main(args: list[str] | None = None) -> None:
    """
    Run the fake camera until interrupted. ROS parameters: config (cfg.yaml path; its CALIBRATION.board
    sets the board), rate_hz, blank_every, reply_delay_s.
    """
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    node = None
    try:
        node = FakeCamera(**_read_parameters())
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
