"""ROS 2 camera node for the Arducam OV2311 on the Raspberry Pi 5.

It owns the camera and nothing else: no filtering, no pose, no control. Those
belong in the nodes on either side of it.

One topic, one service, and deliberately not two topics:

    image             sensor_msgs/CompressedImage   out  640x480 mono8 PNG, 15 Hz
    camera_info       sensor_msgs/CameraInfo        out  intrinsics FOR THAT 640x480
    image_full_res    convchart_interfaces/..       srv  the newest 1600x1200 frame

The stream is 640x480 because that is exactly the detector's input size, so
inference.py's own downscale (`r = W_in / Ws`) comes out at 1.0 and does nothing.
The full-resolution frame is a SERVICE rather than a second topic because almost
nothing wants it: a 1600x1200 PNG is tens of milliseconds to encode and over a
megabyte on the wire, and publishing that continuously is the ~230 Mbit/s problem
that docker/README.md exists to avoid. Calibration and single-shot diagnostics
ask for it; the pose pipeline never does.

The geometry is fixed and load-bearing -- see frames.py. 1600x1300 active area,
cropped to a 1600x1200 4:3 window, downscaled by exactly 2.5. Intrinsics must
therefore be calibrated AT 640x480, which is what camera_info publishes and what
camera_info_url is validated against.

Unlike p4p_serial_bridge, this node uses a MultiThreadedExecutor and a dedicated
capture thread. That is not gratuitous: the bridge's operations are all
sub-millisecond, whereas a full-res PNG encode is 25-60 ms against a 100 ms
stream period, so the service must not share a thread with the publisher.

Three invariants worth not breaking:

  * Only the capture thread ever touches the backend. libcamera objects are not
    thread-safe and a stop() racing a blocked read() is a segfault, so the
    watchdog asks for a restart rather than performing one.
  * Frames handed out by a backend are copies, not views into its DMA buffer, so
    the service can spend 40 ms encoding one while capture moves on.
  * close() joins the capture thread BEFORE destroy_node(). publish() on a
    destroyed publisher is a segfault, and at 15 Hz the thread is mid-publish a
    noticeable fraction of the time.
"""
from __future__ import annotations

import os
import threading
import time

from convchart_interfaces.srv import GetFullResImage
from convchart_qos.qos import IMAGE_DEADLINE_CEILING_S, image_qos_pub
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage
import yaml

from . import frames
from .backends import CameraError, make_backend

_MAX_SENSOR_LATENCY_S = 0.5

_MIN_FRAME_RATE_HZ = 1.0 / IMAGE_DEADLINE_CEILING_S

_SURPLUS_GUARD_FRACTION = 0.9


class CameraNode(Node):
    """Capture, downscale, publish, and serve the newest full-resolution frame."""

    def __init__(self, **kwargs):
        super().__init__('camera_node', **kwargs)

        self.declare_parameter('camera_backend', 'v4l2')
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('device', '/dev/video0')
        self.declare_parameter('frame_rate', 15.0)
        self.declare_parameter('frame_id', 'pi_camera')
        self.declare_parameter('png_level', 1)
        self.declare_parameter('crop_top', frames.CROP_TOP)
        self.declare_parameter('prefer_8bit', True)
        self.declare_parameter('exposure_time_us', 0)
        self.declare_parameter('analogue_gain', 0.0)
        self.declare_parameter('frame_timeout', 1.0)
        self.declare_parameter('reconnect_period', 1.0)
        self.declare_parameter('buffer_count', 4)
        self.declare_parameter('publish_camera_info', True)
        self.declare_parameter('camera_info_url', '')
        self.declare_parameter('use_sensor_timestamp', True)
        self.declare_parameter('mock_image', '')
        self.declare_parameter('mock_latency_ms', 0.0)
        self.declare_parameter('mock_rate', 0.0)
        self.declare_parameter('status_period', 5.0)

        g = self.get_parameter
        self._backend_name = g('camera_backend').value
        if self._backend_name not in ('picamera2', 'v4l2', 'mock'):
            raise ValueError(
                f'camera_backend must be picamera2, v4l2 or mock, '
                f'not {self._backend_name!r}')

        self._frame_rate = self._rate('frame_rate', _MIN_FRAME_RATE_HZ)
        self._frame_timeout = self._rate('frame_timeout', 0.05)
        self._reconnect_period = self._rate('reconnect_period', 0.1)
        self._status_period = self._rate('status_period', 1.0)
        self._frame_id = g('frame_id').value
        self._png_level = int(np.clip(g('png_level').value, 0, 9))
        self._use_sensor_ts = bool(g('use_sensor_timestamp').value)
        self._publish_info = bool(g('publish_camera_info').value)

        self._period = 1.0 / self._frame_rate

        offered = min(IMAGE_DEADLINE_CEILING_S, 1.5 * self._period)
        if offered < 1.5 * self._period:
            self.get_logger().warning(
                f'frame_rate={self._frame_rate} Hz needs a {1.5 * self._period:.3f} s '
                f'deadline for jitter margin, but {IMAGE_DEADLINE_CEILING_S} s is the '
                'ceiling the inference node requests. Offering the ceiling: the link '
                'still matches, but subscribers may report missed deadlines.')
        self._img_pub = self.create_publisher(
            CompressedImage, 'image', image_qos_pub(deadline_s=offered))
        self._info_pub = self.create_publisher(
            CameraInfo, 'camera_info', image_qos_pub(deadline_s=offered))
        self._info = self._load_camera_info(g('camera_info_url').value)

        self._lock = threading.Lock()
        self._latest: tuple[np.ndarray, object] | None = None
        self._encode_lock = threading.Lock()
        self._cache_key: tuple[int, int] | None = None
        self._cache_data: bytes = b''

        self._srv = self.create_service(
            GetFullResImage, 'image_full_res', self._on_full_res,
            callback_group=MutuallyExclusiveCallbackGroup())

        self._backend = None
        self._running = True
        self._want_restart = threading.Event()
        self._next_open = 0.0
        self._last_open_error = ''
        self._frames = 0
        self._drops = 0
        self._surplus = 0
        self._last_arrival_mono = 0.0
        self._reconnects = 0
        self._last_frame_mono = 0.0
        self._last_latency_s = float('nan')
        self._encode_ms = float('nan')
        self._boot_clock = time.CLOCK_BOOTTIME
        self._clock_probed = False
        self._stamp_trusted = False

        if self._use_sensor_ts and getattr(self.get_clock(), 'ros_time_is_active', False):
            self.get_logger().warning(
                'use_sensor_timestamp is set but ROS time is active (use_sim_time); '
                'the sensor clock is unrelated to /clock, so stamping with the ROS '
                'clock at read instead.')
            self._use_sensor_ts = False

        self._params = {
            'camera_index': g('camera_index').value,
            'device': g('device').value,
            'frame_rate': self._frame_rate,
            'crop_top': int(g('crop_top').value),
            'prefer_8bit': bool(g('prefer_8bit').value),
            'exposure_time_us': int(g('exposure_time_us').value),
            'analogue_gain': float(g('analogue_gain').value),
            'buffer_count': int(g('buffer_count').value),
            'mock_image': g('mock_image').value,
            'mock_latency_ms': float(g('mock_latency_ms').value),
            'mock_rate': float(g('mock_rate').value),
        }

        self._status_timer = self.create_timer(
            self._status_period, self._status_tick,
            callback_group=MutuallyExclusiveCallbackGroup())

        self._thread = threading.Thread(
            target=self._capture_loop, name='p4p_camera_capture', daemon=True)
        self._thread.start()

        self.get_logger().info(
            f'camera_node up: backend={self._backend_name} '
            f'{frames.ACTIVE[1]}x{frames.ACTIVE[0]} -> crop {self._params["crop_top"]} -> '
            f'{frames.FULL[1]}x{frames.FULL[0]} -> /{frames.SCALE} -> '
            f'{frames.STREAM[1]}x{frames.STREAM[0]} at {self._frame_rate} Hz, '
            f'offered deadline {offered:.3f} s')

    def _rate(self, name: str, minimum: float) -> float:
        """Read a positive rate, clamping loudly rather than refusing to start."""
        value = float(self.get_parameter(name).value)
        if value < minimum:
            self.get_logger().error(
                f'{name}={value} is below the usable minimum {minimum}; clamped.')
            return minimum
        return value

    def _load_camera_info(self, url: str) -> CameraInfo:
        """Build the CameraInfo to publish, validating the resolution hard."""
        info = CameraInfo()
        info.width, info.height = frames.STREAM[1], frames.STREAM[0]
        info.distortion_model = 'plumb_bob'
        url = str(url or '')
        if not url:
            self.get_logger().warning(
                'no camera_info_url: publishing UNCALIBRATED camera_info (k and d '
                'zeroed). Calibrate at '
                f'{frames.STREAM[1]}x{frames.STREAM[0]}, not at sensor resolution, '
                'and set cfg/cfg.yaml CAMERA.K and CAMERA.dist to match.')
            return info

        path = url[len('file://'):] if url.startswith('file://') else url
        try:
            with open(os.path.expanduser(path), 'r') as fh:
                cal = yaml.safe_load(fh) or {}
        except (OSError, yaml.YAMLError) as e:
            self.get_logger().error(
                f'cannot read camera_info_url {path!r}: {e}; publishing uncalibrated.')
            return info

        want_w, want_h = frames.STREAM[1], frames.STREAM[0]
        got_w, got_h = cal.get('image_width'), cal.get('image_height')
        if (got_w, got_h) != (want_w, want_h):
            self.get_logger().error(
                f'camera_info_url {path!r} declares {got_w}x{got_h}, but this node '
                f'streams {want_w}x{want_h}. Intrinsics for a different resolution '
                'would be silently wrong, so the file is IGNORED. Convert them with '
                'p4p_camera.frames.scale_intrinsics (note the principal point is '
                'not simply cx/2.5) and recalibrate or rescale.')
            return info

        try:
            info.k = [float(v) for v in cal['camera_matrix']['data']]
            info.d = [float(v) for v in cal['distortion_coefficients']['data']]
            info.r = [float(v) for v in cal.get('rectification_matrix', {}).get(
                'data', [1, 0, 0, 0, 1, 0, 0, 0, 1])]
            info.p = [float(v) for v in cal.get('projection_matrix', {}).get(
                'data', list(info.k[:3]) + [0.0] + list(info.k[3:6]) + [0.0]
                + list(info.k[6:9]) + [0.0])]
            info.distortion_model = str(cal.get('distortion_model', 'plumb_bob'))
        except (KeyError, TypeError, ValueError) as e:
            self.get_logger().error(
                f'camera_info_url {path!r} is missing or malformed ({e}); '
                'publishing uncalibrated.')
            return CameraInfo(width=want_w, height=want_h,
                              distortion_model='plumb_bob')
        self.get_logger().info(f'camera_info loaded from {path} at {want_w}x{want_h}')
        return info

    def _capture_loop(self) -> None:
        """Own the backend, publish every frame, and never let an error escape."""
        try:
            while self._running:
                if self._backend is None:
                    self._try_open()
                    continue
                if self._want_restart.is_set():
                    self._want_restart.clear()
                    self._drop('the watchdog asked for a restart')
                    continue
                try:
                    full, sensor_ns = self._backend.read(self._frame_timeout)
                except TimeoutError as e:
                    self._drop(f'read timed out: {e}')
                    continue
                except (CameraError, OSError, RuntimeError, ValueError) as e:
                    self._drop(f'read failed: {e}')
                    continue
                t_read = time.monotonic()
                if t_read - self._last_arrival_mono < (
                        _SURPLUS_GUARD_FRACTION * self._period):
                    self._surplus += 1
                    continue
                self._last_arrival_mono = t_read
                try:
                    self._on_frame(full, sensor_ns)
                except Exception as e:      # noqa: BLE001
                    self.get_logger().error(
                        f'dropping a frame: {e}', throttle_duration_sec=5.0)
                    self._drops += 1
        finally:
            backend, self._backend = self._backend, None
            if backend is not None:
                backend.stop()

    def _try_open(self) -> None:
        """Open the backend, rate-limited, logging the reason it will not open."""
        now = time.monotonic()
        if now < self._next_open:
            time.sleep(min(0.05, self._next_open - now))
            return
        self._next_open = now + self._reconnect_period
        try:
            backend = make_backend(self._backend_name, self._params, self.get_logger())
            backend.start()
        except (CameraError, OSError, RuntimeError, ValueError, ImportError) as e:
            self._last_open_error = str(e)
            self.get_logger().error(
                f'cannot open the {self._backend_name} camera: {e}',
                throttle_duration_sec=5.0)
            return
        self._backend = backend
        self._last_open_error = ''
        self._last_frame_mono = time.monotonic()
        self.get_logger().info(f'{self._backend_name} camera open')

    def _drop(self, why: str) -> None:
        """Close the backend so the next loop reopens it. Mirrors the bridge."""
        self._drops += 1
        self._reconnects += 1
        self.get_logger().error(f'{why}; reopening the camera',
                                throttle_duration_sec=5.0)
        backend, self._backend = self._backend, None
        if backend is not None:
            backend.stop()

    def _on_frame(self, full: np.ndarray, sensor_ns) -> None:
        """Stamp, retain, downscale, encode and publish one frame."""
        now_ros_ns = self.get_clock().now().nanoseconds
        now_boot_ns = time.clock_gettime_ns(self._boot_clock)

        if self._use_sensor_ts:
            if not self._clock_probed:
                self._probe_capture_clock(sensor_ns, now_ros_ns)
                now_boot_ns = time.clock_gettime_ns(self._boot_clock)
            stamp_ns, latency_s, trusted = frames.ros_stamp_ns(
                sensor_ns, now_ros_ns, now_boot_ns, _MAX_SENSOR_LATENCY_S)
            self._last_latency_s = latency_s
            if not trusted:
                self.get_logger().warning(
                    f'sensor timestamp implies a {latency_s:.3f} s capture latency, '
                    f'outside [0, {_MAX_SENSOR_LATENCY_S}]; stamping with the ROS '
                    'clock instead. The two clocks are not what we think they are.',
                    throttle_duration_sec=10.0)
            self._stamp_trusted = trusted
        else:
            stamp_ns = now_ros_ns

        stamp = Time(nanoseconds=stamp_ns,
                     clock_type=self.get_clock().clock_type).to_msg()

        full.flags.writeable = False
        with self._lock:
            self._latest = (full, stamp)

        stream = frames.downscale(full)
        t0 = time.monotonic()
        data = frames.encode_png(stream, self._png_level)
        self._encode_ms = (time.monotonic() - t0) * 1e3

        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id
        msg.format = frames.PNG_FORMAT
        msg.data = data
        self._img_pub.publish(msg)

        if self._publish_info:
            self._info.header.stamp = stamp
            self._info.header.frame_id = self._frame_id
            self._info_pub.publish(self._info)

        self._frames += 1
        self._last_frame_mono = time.monotonic()

    def _probe_capture_clock(self, sensor_ns, now_ros_ns: int) -> None:
        """Latch whichever POSIX clock the sensor timestamps are actually on."""
        self._clock_probed = True
        if sensor_ns is None:
            return
        candidates = (('CLOCK_BOOTTIME', time.CLOCK_BOOTTIME),
                      ('CLOCK_MONOTONIC', time.CLOCK_MONOTONIC))
        report, plausible = [], None
        for name, clk in candidates:
            latency = (time.clock_gettime_ns(clk) - int(sensor_ns)) / 1e9
            report.append(f'{name} {latency:+.4f} s')
            if plausible is None and 0.0 <= latency <= _MAX_SENSOR_LATENCY_S:
                plausible = (name, clk)
        if plausible is None:
            self.get_logger().warning(
                'the sensor timestamp matches no POSIX clock plausibly '
                f'({", ".join(report)}); falling back to the ROS clock at read.')
            self._use_sensor_ts = False
            return
        self._boot_clock = plausible[1]
        self.get_logger().info(
            f'capture clock: implied latency {", ".join(report)}; '
            f'using {plausible[0]}')

    def _on_full_res(self, request, response):
        """Return the newest retained full-resolution frame, encoded on demand."""
        response.success = False
        response.image.format = ''
        response.image.data = []

        with self._lock:
            latest = self._latest
        if latest is None:
            reason = self._last_open_error or 'waiting for the first frame'
            response.message = f'no frame captured yet: {reason}'
            return response

        full, stamp = latest
        age_s = (self.get_clock().now() - Time.from_msg(stamp)).nanoseconds / 1e9
        if request.max_age > 0.0 and age_s > request.max_age:
            response.message = (
                f'retained frame is {age_s:.3f} s old (max_age {request.max_age:.3f}); '
                f'the camera is {"open" if self._backend is not None else "not open"}')
            response.capture_stamp = stamp
            return response

        key = (stamp.sec, stamp.nanosec)
        if key == self._cache_key:
            data = self._cache_data
        else:
            if not self._encode_lock.acquire(blocking=False):
                response.message = 'an encode is already in flight; retry'
                return response
            try:
                data = frames.encode_png(full, self._png_level)
                self._cache_key, self._cache_data = key, data
            except Exception as e:      # noqa: BLE001
                response.message = f'could not encode the frame: {e}'
                return response
            finally:
                self._encode_lock.release()

        response.success = True
        response.message = (
            f'{frames.FULL[1]}x{frames.FULL[0]} mono8 PNG, {len(data)} bytes, '
            f'captured {age_s:.3f} s ago')
        response.capture_stamp = stamp
        response.image.header.stamp = stamp
        response.image.header.frame_id = self._frame_id
        response.image.format = frames.PNG_FORMAT
        response.image.data = data
        return response

    def _status_tick(self) -> None:
        """Report health, and ask the capture thread to restart a dead stream."""
        if self._backend is not None:
            idle = time.monotonic() - self._last_frame_mono
            if idle > max(self._frame_timeout, 2.0 * self._period):
                self.get_logger().error(
                    f'no frame for {idle:.2f} s; asking the capture thread to '
                    'restart the camera')
                self._want_restart.set()

        state = 'streaming' if self._backend is not None else 'no camera'
        latency = ('n/a' if self._last_latency_s != self._last_latency_s
                   else f'{self._last_latency_s * 1e3:.1f} ms')
        encode = ('n/a' if self._encode_ms != self._encode_ms
                  else f'{self._encode_ms:.1f} ms')
        self.get_logger().info(
            f'{state}: {self._frames} frames, {self._drops} dropped, '
            f'{self._surplus} surplus, {self._reconnects} reconnects, '
            f'sensor->ros {latency}, stream encode {encode}'
            + ('' if self._backend is not None
               else f', last error: {self._last_open_error or "none"}'))

    def close(self) -> None:
        """Stop capturing and release the camera."""
        self._running = False
        thread, self._thread = getattr(self, '_thread', None), None
        if thread is not None and thread.is_alive():
            thread.join(timeout=self._frame_timeout + 1.0)
            if thread.is_alive():
                self.get_logger().error(
                    'the capture thread did not stop in time; the camera read is '
                    'wedged and this process may need to be killed')
        backend, self._backend = self._backend, None
        if backend is not None:
            try:
                backend.stop()
            except Exception as e:      # noqa: BLE001
                self.get_logger().error(f'could not release the camera: {e}')


def main(args: list[str] | None = None) -> None:
    """Spin the camera node, releasing the camera on the way out."""
    rclpy.init(args=args)
    node = CameraNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
