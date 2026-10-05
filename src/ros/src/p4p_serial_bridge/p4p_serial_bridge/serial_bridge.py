"""ROS 2 bridge between the Pi and the p4p_arduino firmware on the Mega.

It owns the USB serial link and nothing else: no filtering, no fusing, no
control. Those belong in the nodes on either side of it.

Two directions, three topics, and deliberately NOT one topic for both. A node
that subscribed to what it published would build a feedback loop, the two
directions carry unrelated types, and the controller would have to filter the
bridge's own telemetry back out of its command stream:

    cmd_vel          geometry_msgs/Twist         in    the controller's request
    drive/telemetry  convchart_interfaces/..     out   everything the Mega reports
    drive/status     convchart_interfaces/..     out   link and command health:
                                                       running, stopped, stalled
    imu/data         sensor_msgs/Imu             out   the same heading and rate,
                                                       in the standard type

The downlink runs on a FIXED-RATE TIMER, not on the cmd_vel callback. The
firmware zeroes the motors if no V arrives within 400 ms, so a controller that
goes quiet -- or merely publishes unevenly -- would stall the bot. The timer
re-sends the newest command at command_rate_hz regardless, and substitutes zeros
once that command is older than cmd_timeout. Two watchdogs in series: ours holds
station and keeps feeding theirs, theirs catches us dying altogether.

Arming is a service rather than a topic, because it is a one-shot state change
and the disarm half is the software e-stop, which must not queue behind anything:

    ros2 service call /serial_bridge/arm std_srvs/srv/SetBool "{data: true}"

The bot boots disarmed, and opening the port resets the Mega over DTR, so every
single reconnect lands disarmed. The bridge re-arms itself afterwards only if it
was armed beforehand.
"""
from __future__ import annotations

import math

from convchart_interfaces.msg import DriveStatus, DriveTelemetry
from geometry_msgs.msg import Twist
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Imu
import serial
from std_srvs.srv import SetBool

from . import protocol

_VAR_UNKNOWN = 1e6
_VAR_YAW = 0.01
_VAR_YAW_RATE = 1e-4


class SerialBridge(Node):
    """Pump bytes between the cmd_vel / telemetry topics and the Mega."""

    def __init__(self) -> None:
        super().__init__('serial_bridge')

        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('command_rate_hz', 20.0)
        self.declare_parameter('read_rate_hz', 200.0)
        self.declare_parameter('cmd_timeout', 0.25)
        self.declare_parameter('auto_arm', False)
        self.declare_parameter('max_linear', 0.5)
        self.declare_parameter('max_angular', 2.08)
        self.declare_parameter('publish_imu', True)
        self.declare_parameter('status_rate_hz', 2.0)
        self.declare_parameter('telemetry_timeout', 0.5)
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('reconnect_period', 1.0)

        self._port = self.get_parameter('port').value
        self._baud = int(self.get_parameter('baud').value)
        self._cmd_timeout = float(self.get_parameter('cmd_timeout').value)
        self._max_linear = float(self.get_parameter('max_linear').value)
        self._max_angular = float(self.get_parameter('max_angular').value)
        self._frame_id = self.get_parameter('frame_id').value
        self._reconnect_period = float(self.get_parameter('reconnect_period').value)
        self._telemetry_timeout = float(self.get_parameter('telemetry_timeout').value)
        self._auto_arm = bool(self.get_parameter('auto_arm').value)
        self._want_armed = self._auto_arm

        if self._cmd_timeout >= protocol.FIRMWARE_CMD_TIMEOUT_S:
            safe = protocol.FIRMWARE_CMD_TIMEOUT_S / 2.0
            self.get_logger().error(
                f'cmd_timeout={self._cmd_timeout} is not below the firmware watchdog '
                f'({protocol.FIRMWARE_CMD_TIMEOUT_S} s); our zeros would keep feeding it a '
                f'stale velocity forever. Clamped to {safe}.')
            self._cmd_timeout = safe
        cmd_hz = self._rate('command_rate_hz', 2.0 / protocol.FIRMWARE_CMD_TIMEOUT_S)
        read_hz = self._rate('read_rate_hz', 1.0)
        status_hz = self._rate('status_rate_hz', 0.1)

        self._ser: serial.Serial | None = None
        self._reader = protocol.LineReader()
        self._cmd = (0.0, 0.0, 0.0)
        self._cmd_time = self.get_clock().now()
        self._have_cmd = False
        self._cmd_stale = True
        self._booted = False
        self._next_open = self.get_clock().now()
        self._dropped_rows = 0
        self._last_flags: int | None = None
        self._last_row_time = None
        self._open_error = ''
        self._reconnects = 0
        self._opened_once = False
        self._state = DriveStatus.NO_LINK

        cmd_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                             reliability=QoSReliabilityPolicy.RELIABLE)
        tel_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10,
                             reliability=QoSReliabilityPolicy.RELIABLE)

        self._cmd_sub = self.create_subscription(Twist, 'cmd_vel', self._on_cmd_vel, cmd_qos)
        self._tel_pub = self.create_publisher(DriveTelemetry, 'drive/telemetry', tel_qos)
        self._imu_pub = (self.create_publisher(Imu, 'imu/data', tel_qos)
                         if self.get_parameter('publish_imu').value else None)
        status_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=5,
                                reliability=QoSReliabilityPolicy.RELIABLE,
                                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._status_pub = self.create_publisher(DriveStatus, 'drive/status', status_qos)
        self._arm_srv = self.create_service(SetBool, '~/arm', self._on_arm)

        self._read_timer = self.create_timer(1.0 / read_hz, self._read_tick)
        self._write_timer = self.create_timer(1.0 / cmd_hz, self._write_tick)
        self._status_timer = self.create_timer(1.0 / status_hz, self._publish_status)

        self.get_logger().info(
            f'bridging {self._port} at {self._baud} baud, commands at {cmd_hz} Hz, '
            f'cmd_timeout={self._cmd_timeout} s, auto_arm={self._auto_arm}')

    def _rate(self, name: str, minimum: float) -> float:
        """Read a rate parameter, refusing zero (which would divide) and anything slow enough to be
        unsafe.
        """
        value = float(self.get_parameter(name).value)
        if value < minimum:
            self.get_logger().error(
                f'{name}={value} is below the safe minimum {minimum}; clamped')
            return minimum
        return value

    def _open(self) -> None:
        """Try to open the port, quietly, retrying no faster than reconnect_period."""
        now = self.get_clock().now()
        if now < self._next_open:
            return
        self._next_open = now + Duration(seconds=self._reconnect_period)
        try:
            self._ser = serial.Serial(self._port, self._baud, timeout=0, write_timeout=0.05)
        except (OSError, serial.SerialException) as e:
            self.get_logger().warning(f'cannot open {self._port}: {e}', throttle_duration_sec=5.0)
            self._ser = None
            self._open_error = str(e)
            self._publish_status()
            return
        self._reader.reset()
        try:
            self._ser.reset_input_buffer()
        except (OSError, serial.SerialException):
            pass
        self._forget_firmware_state('port reopened')
        self._open_error = ''
        if self._opened_once:
            self._reconnects += 1
        self._opened_once = True
        self.get_logger().info(
            f'opened {self._port}; the Mega resets on DTR, waiting for its banner')
        self._publish_status()

    def _forget_firmware_state(self, why: str) -> None:
        """Reset everything that a firmware reboot invalidates."""
        self._booted = False
        self._last_flags = None
        self._last_row_time = None
        self._have_cmd = False
        self._cmd = (0.0, 0.0, 0.0)
        self._cmd_stale = True
        if self._want_armed and not self._auto_arm:
            self._want_armed = False
            self.get_logger().warning(
                f'{why}: staying disarmed, call ~/arm again (set auto_arm to re-arm '
                f'automatically)')

    def _drop(self, why: str) -> None:
        """Close the link after an I/O error so the next tick reopens it."""
        self.get_logger().error(f'{self._port} lost: {why}')
        try:
            if self._ser is not None:
                self._ser.close()
        except (OSError, serial.SerialException):
            pass
        self._ser = None
        self._forget_firmware_state('link lost')
        self._open_error = why
        self._publish_status()

    def _write(self, data: bytes) -> bool:
        """Send bytes, dropping the link on failure. True when it went out."""
        if self._ser is None:
            return False
        try:
            self._ser.write(data)
            return True
        except (OSError, serial.SerialException) as e:
            self._drop(f'write failed: {e}')
            return False

    def _read_tick(self) -> None:
        if self._ser is None:
            self._open()
            return
        try:
            data = self._ser.read(4096)
        except (OSError, serial.SerialException) as e:
            self._drop(f'read failed: {e}')
            return
        for line in self._reader.feed(data):
            self._on_line(line)

    def _on_line(self, line: str) -> None:
        if line.startswith('#'):
            self.get_logger().info(f'mega: {line}')
            if line.startswith(protocol.BOOT_BANNER) and self._booted:
                self.get_logger().warning('the Mega rebooted without dropping the port')
                self._forget_firmware_state('firmware rebooted')
                self._publish_status()
            return
        if not line or line == protocol.TELEMETRY_HEADER:
            return
        tel = protocol.parse_line(line)
        if tel is None:
            self._dropped_rows += 1
            self.get_logger().warning(f'unparseable telemetry row ({self._dropped_rows} so far): '
                                      f'{line!r}', throttle_duration_sec=5.0)
            return
        self._last_row_time = self.get_clock().now()
        first = not self._booted
        if first:
            self._booted = True
            self.get_logger().info('Mega is streaming')
            if self._want_armed:
                self._write(protocol.ARM)
        self._publish(tel)
        if first:
            self._publish_status()

    def _publish(self, tel: protocol.Telemetry) -> None:
        stamp = self.get_clock().now().to_msg()
        yaw_rate = math.radians(tel.yaw_rate_dps)

        msg = DriveTelemetry()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id
        msg.mega_t_ms = tel.t_ms & 0xFFFFFFFF
        msg.heading_rad = tel.heading_rad
        msg.yaw_rate = yaw_rate
        msg.applied.linear.x, msg.applied.linear.y, msg.applied.angular.z = tel.applied
        msg.armed = tel.armed
        msg.timed_out = tel.timed_out
        msg.saturated = tel.saturated
        msg.flags = tel.flags & 0xFF
        msg.imu_resets = tel.imu_resets & 0xFFFF
        msg.bad_commands = tel.bad_commands & 0xFFFF
        self._tel_pub.publish(msg)

        if tel.flags != self._last_flags:
            self.get_logger().info(f'mega state: {protocol.describe_flags(tel.flags)}')
            self._last_flags = tel.flags
            self._publish_status()

        if self._imu_pub is not None:
            imu = Imu()
            imu.header.stamp = stamp
            imu.header.frame_id = self._frame_id
            yaw = math.remainder(tel.heading_rad, 2.0 * math.pi)
            imu.orientation.z = math.sin(yaw / 2.0)
            imu.orientation.w = math.cos(yaw / 2.0)
            imu.orientation_covariance = [_VAR_UNKNOWN, 0.0, 0.0,
                                          0.0, _VAR_UNKNOWN, 0.0,
                                          0.0, 0.0, _VAR_YAW]
            imu.angular_velocity.z = yaw_rate
            imu.angular_velocity_covariance = [_VAR_UNKNOWN, 0.0, 0.0,
                                               0.0, _VAR_UNKNOWN, 0.0,
                                               0.0, 0.0, _VAR_YAW_RATE]
            imu.linear_acceleration_covariance[0] = -1.0
            self._imu_pub.publish(imu)

    def _on_cmd_vel(self, msg: Twist) -> None:
        raw = (msg.linear.x, msg.linear.y, msg.angular.z)
        if not all(math.isfinite(v) for v in raw):
            self.get_logger().warning(f'ignoring a cmd_vel carrying NaN or inf: {raw}')
            return
        vx = _clamp(raw[0], self._max_linear)
        vy = _clamp(raw[1], self._max_linear)
        wz = _clamp(raw[2], self._max_angular)
        self._cmd = (vx, vy, wz)
        self._cmd_time = self.get_clock().now()
        self._have_cmd = True

    def _write_tick(self) -> None:
        """Send the newest command, or zeros if it has gone stale. Never skip."""
        if self._ser is None or not self._booted:
            return
        if not self._want_armed or not self._have_cmd:
            if not self._have_cmd:
                self.get_logger().info('no cmd_vel yet, holding station',
                                       throttle_duration_sec=10.0)
            vx, vy, wz = 0.0, 0.0, 0.0
        else:
            age = (self.get_clock().now() - self._cmd_time).nanoseconds * 1e-9
            if age > self._cmd_timeout:
                if not self._cmd_stale:
                    self.get_logger().warning(f'cmd_vel is {age:.2f} s old, sending zeros')
                    self._cmd_stale = True
                    self._publish_status()
                vx, vy, wz = 0.0, 0.0, 0.0
            else:
                if self._cmd_stale:
                    self.get_logger().info('cmd_vel is flowing again')
                    self._cmd_stale = False
                    self._publish_status()
                vx, vy, wz = self._cmd
        try:
            self._write(protocol.format_velocity(vx, vy, wz))
        except ValueError as e:
            self.get_logger().error(f'{e}; sending zeros instead')
            self._write(protocol.format_velocity(0.0, 0.0, 0.0))

    def _on_arm(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        want = bool(request.data)
        sent = self._write(protocol.ARM if want else protocol.DISARM)
        if sent or not want:
            self._want_armed = want
        response.success = sent
        response.message = (('armed' if want else 'disarmed') if sent
                            else f'{self._port} is not open, still '
                                 f'{"armed" if self._want_armed else "disarmed"}')
        self.get_logger().info(f'arm request: {response.message}')
        self._publish_status()
        return response

    def _age(self, then) -> float:
        """Seconds since `then`, or -1.0 when it never happened."""
        if then is None:
            return -1.0
        return (self.get_clock().now() - then).nanoseconds * 1e-9

    def _classify(self) -> tuple[int, str]:
        """Work out the single state that best describes the link right now."""
        if self._ser is None:
            return DriveStatus.NO_LINK, f'NO_LINK: {self._open_error or "port closed"}'
        if not self._booted:
            return DriveStatus.NO_TELEMETRY, 'NO_TELEMETRY: waiting for the Mega to boot'
        silent = self._age(self._last_row_time)
        if silent > self._telemetry_timeout:
            return DriveStatus.NO_TELEMETRY, f'NO_TELEMETRY: silent for {silent:.1f} s'
        if not (self._last_flags or 0) & protocol.FLAG_ARMED:
            if self._want_armed:
                return DriveStatus.DISARMED, 'DISARMED: an arm is pending, awaiting the echo'
            return DriveStatus.DISARMED, 'DISARMED: call ~/arm with data: true to enable'
        if not self._have_cmd:
            return DriveStatus.STALLED, 'STALLED: no cmd_vel has arrived yet'
        cmd_age = self._age(self._cmd_time)
        if cmd_age > self._cmd_timeout:
            return DriveStatus.STALLED, f'STALLED: cmd_vel is {cmd_age:.2f} s old'
        return DriveStatus.RUNNING, 'RUNNING'

    def _publish_status(self) -> None:
        """Publish the current state, and log it whenever it changes."""
        state, detail = self._classify()
        msg = DriveStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.state = state
        msg.detail = detail
        msg.want_armed = self._want_armed
        msg.port = self._port
        msg.cmd_age = self._age(self._cmd_time) if self._have_cmd else -1.0
        msg.telemetry_age = self._age(self._last_row_time)
        msg.dropped_rows = self._dropped_rows
        msg.reconnects = self._reconnects
        self._status_pub.publish(msg)
        if state != self._state:
            self.get_logger().info(f'status: {detail}')
            self._state = state

    def close(self) -> None:
        """Disarm the bot and release the port."""
        ser, self._ser = self._ser, None
        if ser is None:
            return
        try:
            ser.write(protocol.DISARM)
            ser.flush()
        except (OSError, serial.SerialException) as e:
            self.get_logger().error(f'could not disarm on the way out: {e}')
        try:
            ser.close()
        except (OSError, serial.SerialException):
            pass


def _clamp(v: float, limit: float) -> float:
    return max(-limit, min(limit, v))


def main(args: list[str] | None = None) -> None:
    """Spin the bridge, disarming the bot on the way out."""
    rclpy.init(args=args)
    node = SerialBridge()
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
