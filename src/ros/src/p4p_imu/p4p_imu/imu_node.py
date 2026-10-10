"""ROS 2 driver for the BNO085 on the Pi's I2C bus.

It owns the sensor and nothing else: no filtering, no fusing, no control. Those
belong in the nodes downstream of it.

One direction, one topic:

    imu/data   sensor_msgs/Imu   out   heading about z, yaw rate, acceleration

and deliberately no custom message alongside it. The BNO085 used to hang off the
Arduino Mega, and p4p_serial_bridge carried its heading twice: unwrapped on
drive/telemetry for this project's filter, and as a quaternion on imu/data for
off-the-shelf consumers. The sensor is on the Pi now and only the standard
message is published, so NOTHING PUBLISHES AN UNWRAPPED HEADING ANY MORE. A
quaternion cannot represent one -- it wraps at +-pi by construction -- so a
consumer that differences heading has to unwrap it itself.

Three things about this sensor that shape the whole file:

THE LIBRARY HANDS OVER PLACEHOLDERS. enable_feature() returns as soon as the
feature id appears in its readings dict, and the entry it waits for is seeded as
a zero quaternion and zero acceleration. A node that published what it was handed
would emit a fabricated yaw of exactly 0 and no gravity for its first few cycles.
Hence orientation.validate, and hence a start() that waits for real data.

A RESET LOOKS LIKE SUCCESS. The library exposes no new-data flag and no sensor
timestamp, and its readings dict holds last-known values. When the chip resets
and stops streaming, every read keeps returning the same tuple with no error, for
as long as you care to ask. Staleness therefore has to be measured on the VALUES
(orientation.samples_differ), not on whether a read succeeded.

THE STAMP IS THE ROS CLOCK, NOT THE SENSOR'S. The SH-2 protocol carries a report
timebase and the library parses it, but drops it rather than exposing it. So
unlike p4p_camera, which back-dates to capture time, there is nothing here to
back-date against: the stamp is taken immediately before the read and carries an
uncharacterised latency of roughly half a poll period plus one report interval
plus the I2C transfer. That matters to anything fusing this against the camera's
properly back-dated stamps.

Reads run on their own thread, never on a ROS timer. The vendor library's drain
loops while its data-ready flag is set and swallows parse errors inside that
loop, so a babbling bus can hang it, and its recovery paths sleep for over a
second. The heartbeat timer therefore ASKS the read thread to restart the sensor
and never touches it itself.
"""
from __future__ import annotations

import threading
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Imu

from . import orientation
from .backends import BACKENDS, ImuError, ImuResetError, make_backend

_MIN_SAMPLE_RATE_HZ = 1.0
_MAX_SAMPLE_RATE_HZ = 400.0
_MIN_POLL_PERIOD_S = 0.001
_BUFFER_S = 0.1


class ImuNode(Node):
    """Poll the BNO085 and publish sensor_msgs/Imu."""

    def __init__(self, **kwargs) -> None:
        super().__init__('imu_node', **kwargs)

        self.declare_parameter('imu_backend', 'bno08x')
        self.declare_parameter('i2c_bus', 1)
        self.declare_parameter('i2c_address', 0x4A)
        self.declare_parameter('sample_rate_hz', 150.0)
        self.declare_parameter('oversample', 2.0)
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('data_timeout', 0.5)
        self.declare_parameter('warmup_timeout', 2.0)
        self.declare_parameter('reconnect_period', 1.0)
        self.declare_parameter('status_period', 5.0)
        self.declare_parameter('jump_tolerance', 0.35)
        self.declare_parameter('var_yaw', 0.01)
        self.declare_parameter('var_roll_pitch', 1e6)
        self.declare_parameter('var_gyro', 1e-4)
        self.declare_parameter('var_accel', 0.01)
        self.declare_parameter('mock_yaw_rate', 0.4)
        self.declare_parameter('mock_noise', 0.01)
        self.declare_parameter('mock_seed', 0xB0085)
        self.declare_parameter('mock_reset_after', 0)
        self.declare_parameter('mock_stall_after', 0)
        self.declare_parameter('mock_fail_open', 0)
        self.declare_parameter('mock_placeholder_reads', 0)

        g = self.get_parameter
        self._backend_name = str(g('imu_backend').value)
        if self._backend_name not in BACKENDS:
            raise ValueError(
                f'imu_backend must be one of {sorted(BACKENDS)}, '
                f'not {self._backend_name!r}')
        self._frame_id = str(g('frame_id').value)
        self._sample_rate = min(self._rate('sample_rate_hz', _MIN_SAMPLE_RATE_HZ),
                                _MAX_SAMPLE_RATE_HZ)
        self._report_period = 1.0 / self._sample_rate
        self._oversample = self._rate('oversample', 1.0)
        self._poll_period = max(self._report_period / self._oversample, _MIN_POLL_PERIOD_S)
        self._data_timeout = self._rate('data_timeout', 2.0 * self._report_period)
        self._reconnect_period = self._rate('reconnect_period', 0.05)
        self._status_period = self._rate('status_period', 0.5)
        self._warmup = float(g('warmup_timeout').value)
        self._jump_tolerance = float(g('jump_tolerance').value)

        var_rp = float(g('var_roll_pitch').value)
        var_gyro = float(g('var_gyro').value)
        var_accel = float(g('var_accel').value)
        self._orientation_cov = orientation.covariance(var_rp, var_rp, float(g('var_yaw').value))
        self._gyro_cov = orientation.covariance(var_gyro, var_gyro, var_gyro)
        self._accel_cov = orientation.covariance(var_accel, var_accel, var_accel)

        self._params = {
            'i2c_bus': int(g('i2c_bus').value),
            'i2c_address': int(g('i2c_address').value),
            'sample_rate_hz': self._sample_rate,
            'warmup_timeout': self._warmup,
            'mock_yaw_rate': float(g('mock_yaw_rate').value),
            'mock_noise': float(g('mock_noise').value),
            'mock_seed': int(g('mock_seed').value),
            'mock_reset_after': int(g('mock_reset_after').value),
            'mock_stall_after': int(g('mock_stall_after').value),
            'mock_fail_open': int(g('mock_fail_open').value),
            'mock_placeholder_reads': int(g('mock_placeholder_reads').value),
        }

        depth = max(10, round(_BUFFER_S * self._sample_rate))
        self._pub = self.create_publisher(
            Imu, 'imu/data',
            QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=depth,
                       reliability=QoSReliabilityPolicy.RELIABLE))

        self._backend = None
        self._running = True
        self._want_restart = threading.Event()
        self._next_open = 0.0
        self._prev: orientation.ImuSample | None = None
        self._prev_yaw: float | None = None
        self._last_yaw = float('nan')
        self._last_change_mono: float | None = None
        self._last_open_error: str | None = None
        self._samples = 0
        self._invalid = 0
        self._polls = 0
        self._empty_polls = 0
        self._last_publish_mono: float | None = None
        self._resets = 0
        self._jumps = 0
        self._reconnects = 0
        self._read_ms_max = 0.0
        self._status_mono = time.monotonic()
        self._status_samples = 0
        self._status_polls = 0

        self._status_timer = self.create_timer(self._status_period, self._status_tick)
        self._thread = threading.Thread(
            target=self._read_loop, name='p4p_imu_read', daemon=True)
        self._thread.start()

        self.get_logger().info(
            f'imu_node on {self._backend_name}: /dev/i2c-{self._params["i2c_bus"]} '
            f'at {self._params["i2c_address"]:#04x}, {self._sample_rate} Hz reports '
            f'polled at {1.0 / self._poll_period:.0f} Hz ({self._oversample}x), '
            f'frame_id={self._frame_id}')

    def _rate(self, name: str, minimum: float) -> float:
        """Read a positive rate, clamping loudly rather than refusing to start."""
        value = float(self.get_parameter(name).value)
        if value < minimum:
            self.get_logger().error(
                f'{name}={value} is below the usable minimum {minimum}; clamped.')
            return minimum
        return value

    def _read_loop(self) -> None:
        """Own the sensor: open it, poll it, publish, and recover from faults."""
        next_due = time.monotonic()
        try:
            while self._running:
                if self._backend is None:
                    self._try_open()
                    next_due = time.monotonic()
                    continue
                if self._want_restart.is_set():
                    self._want_restart.clear()
                    self._drop('the watchdog asked for a restart')
                    continue

                now = time.monotonic()
                if next_due > now:
                    time.sleep(next_due - now)
                next_due = max(next_due + self._poll_period, time.monotonic())
                self._polls += 1

                stamp = self.get_clock().now().to_msg()
                started = time.monotonic()
                try:
                    sample = self._backend.read()
                except ImuResetError as e:
                    self._on_reset(e)
                    continue
                except (ImuError, OSError) as e:
                    self._drop(str(e))
                    continue
                read_ms = (time.monotonic() - started) * 1e3

                why = orientation.validate(sample)
                if why is not None:
                    self._invalid += 1
                    self.get_logger().warning(f'discarding a sample: {why}',
                                              throttle_duration_sec=5.0)
                    continue
                if not orientation.samples_differ(self._prev, sample):
                    self._empty_polls += 1
                    continue

                self._prev = sample
                self._last_change_mono = time.monotonic()
                self._read_ms_max = max(self._read_ms_max, read_ms)
                self._publish(sample, stamp)
        except Exception as e:  # noqa: BLE001
            self.get_logger().fatal(f'the IMU read thread has died: {e!r}')
        finally:
            backend, self._backend = self._backend, None
            if backend is not None:
                backend.stop()

    def _publish(self, sample: orientation.ImuSample, stamp) -> None:
        """Fill and publish one Imu, then check whether heading jumped."""
        quat = orientation.normalise_quaternion(sample.quat)
        if quat is None:
            self._invalid += 1
            return

        msg = Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id
        (msg.orientation.x, msg.orientation.y,
         msg.orientation.z, msg.orientation.w) = quat
        msg.orientation_covariance = self._orientation_cov
        (msg.angular_velocity.x, msg.angular_velocity.y,
         msg.angular_velocity.z) = sample.gyro
        msg.angular_velocity_covariance = self._gyro_cov
        (msg.linear_acceleration.x, msg.linear_acceleration.y,
         msg.linear_acceleration.z) = sample.accel
        msg.linear_acceleration_covariance = self._accel_cov
        self._pub.publish(msg)
        self._samples += 1

        now = time.monotonic()
        dt = 0.0 if self._last_publish_mono is None else now - self._last_publish_mono
        self._last_publish_mono = now

        yaw = orientation.yaw_from_quaternion(quat)
        if self._prev_yaw is not None and not orientation.explains_jump(
                self._prev_yaw, yaw, sample.gyro[2], dt, self._jump_tolerance):
            self._jumps += 1
            self.get_logger().warning(
                f'heading jumped {self._prev_yaw:+.3f} -> {yaw:+.3f} rad with a gyro '
                f'reading {sample.gyro[2]:+.3f} rad/s, which does not account for it. '
                'A sensor reset is likely and continuity is broken; the raw heading is '
                'published uncompensated.',
                throttle_duration_sec=5.0)
        self._prev_yaw = yaw
        self._last_yaw = yaw

    def _on_reset(self, exc: Exception) -> None:
        """Count a self-reset, then try to re-enable the reports in place."""
        self._resets += 1
        self.get_logger().error(
            f'the BNO085 reset itself ({exc}); its heading has re-zeroed, so continuity '
            'across this point is BROKEN. Publishing the raw heading uncompensated.',
            throttle_duration_sec=5.0)
        try:
            self._backend.recover()
        except (ImuError, OSError) as e:
            self._drop(f'recovery after a reset failed: {e}')
            return
        self._prev = None
        self._prev_yaw = None
        self._last_publish_mono = None
        self._last_change_mono = time.monotonic()

    def _try_open(self) -> None:
        """Open the sensor, no more often than reconnect_period."""
        now = time.monotonic()
        if now < self._next_open:
            time.sleep(min(0.05, self._next_open - now))
            return
        self._next_open = now + self._reconnect_period

        try:
            backend = make_backend(self._backend_name, self._params, self.get_logger())
            backend.start()
        except (ImuError, OSError) as e:
            self._last_open_error = str(e)
            self.get_logger().error(f'cannot open the IMU: {e}', throttle_duration_sec=5.0)
            return

        self._backend = backend
        self._last_open_error = None
        self._prev = None
        self._prev_yaw = None
        self._last_publish_mono = None
        self._last_change_mono = time.monotonic()

    def _drop(self, why: str) -> None:
        """Release the sensor so the next pass reopens it. Mirrors the bridge's."""
        self._reconnects += 1
        self._last_open_error = why
        self.get_logger().error(f'dropping the IMU: {why}', throttle_duration_sec=5.0)
        backend, self._backend = self._backend, None
        self._prev = None
        self._prev_yaw = None
        self._last_publish_mono = None
        self._last_change_mono = None
        if backend is not None:
            backend.stop()

    def _status_tick(self) -> None:
        """Watch for a stalled stream, then log one line of health."""
        now = time.monotonic()
        if self._backend is not None and self._last_change_mono is not None:
            idle = now - self._last_change_mono
            if idle > self._data_timeout:
                self.get_logger().error(
                    f'no NEW reading for {idle:.2f} s: the sensor is serving cached '
                    'values, which is what a silent reset looks like. Asking the read '
                    'thread to restart it.')
                self._want_restart.set()

        elapsed = now - self._status_mono
        hz = (self._samples - self._status_samples) / elapsed if elapsed > 0 else 0.0
        poll_hz = (self._polls - self._status_polls) / elapsed if elapsed > 0 else 0.0
        self._status_mono = now
        self._status_samples = self._samples
        self._status_polls = self._polls
        read_max = self._read_ms_max
        self._read_ms_max = 0.0

        yaw = 'none' if self._last_yaw != self._last_yaw else f'{self._last_yaw:+.3f} rad'
        state = 'streaming' if self._backend is not None else 'no imu'
        self.get_logger().info(
            f'{state}: {self._samples} samples at {hz:.1f} Hz (asked '
            f'{self._sample_rate}), polling {poll_hz:.0f} Hz with '
            f'{self._empty_polls} empty, {self._invalid} invalid, '
            f'{self._resets} resets, {self._jumps} jumps, {self._reconnects} reconnects, '
            f'read {read_max:.1f} ms max, yaw {yaw}'
            + ('' if self._backend is not None
               else f', last error: {self._last_open_error or "none"}'))

    def close(self) -> None:
        """Stop the read thread and release the sensor."""
        self._running = False
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(5.0, self._warmup + 3.0))
            if thread.is_alive():
                self.get_logger().warning(
                    'the IMU read thread did not stop; leaking the bus handle rather '
                    'than blocking shutdown further.')
        backend, self._backend = self._backend, None
        if backend is not None:
            try:
                backend.stop()
            except Exception as e:  # noqa: BLE001
                self.get_logger().warning(f'releasing the IMU failed: {e}')


def main(args: list[str] | None = None) -> None:
    """Spin the IMU node, releasing the sensor on the way out."""
    rclpy.init(args=args)
    node = ImuNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
