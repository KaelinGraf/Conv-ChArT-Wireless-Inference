"""Arithmetic on one BNO085 sample.

No ROS and no adafruit in here on purpose: this is the part most likely to be
wrong, and all of it is reachable from a plain pytest with nothing installed. The
driver module under backends/ is a thin wrapper that calls into here, so that the
code which cannot be tested without the sensor contains as little arithmetic as
possible. Same split as p4p_camera's frames.py and p4p_serial_bridge's protocol.py.

The frame is REP-103 as the sensor sits on the bot: +x forward, +y left, +z up,
right handed. The board is mounted flat and axis-aligned with base_link, so there
is no remap here -- if that ever changes, this is where it belongs, not in the
backend and not in the node.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

_MIN_QUAT_NORM = 1e-6


@dataclass(frozen=True)
class ImuSample:
    """One coherent set of readings, taken at a single instant."""

    accel: tuple[float, float, float]
    gyro: tuple[float, float, float]
    quat: tuple[float, float, float, float]


def normalise_quaternion(
        quat: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    """Scale a quaternion to unit length, or return None if it has no direction."""
    i, j, k, real = quat
    norm = math.sqrt(i * i + j * j + k * k + real * real)
    if not math.isfinite(norm) or norm < _MIN_QUAT_NORM:
        return None
    return (i / norm, j / norm, k / norm, real / norm)


def validate(sample: ImuSample) -> str | None:
    """Return why the sample is unusable, or None when it is good."""
    if not all(math.isfinite(v) for v in (*sample.accel, *sample.gyro, *sample.quat)):
        return f'non-finite reading: accel={sample.accel} gyro={sample.gyro} quat={sample.quat}'
    if normalise_quaternion(sample.quat) is None:
        return f'quaternion has no direction: {sample.quat} (sensor not streaming yet?)'
    if sample.accel == (0.0, 0.0, 0.0):
        return 'acceleration is exactly zero (sensor not streaming yet?)'
    return None


def samples_differ(a: ImuSample | None, b: ImuSample) -> bool:
    """Report whether b carries anything new."""
    return a is None or (a.accel, a.gyro, a.quat) != (b.accel, b.gyro, b.quat)


def yaw_from_quaternion(quat: tuple[float, float, float, float]) -> float:
    """Extract rotation about +z, in (-pi, pi]."""
    i, j, k, real = quat
    return math.atan2(2.0 * (real * k + i * j),
                      1.0 - 2.0 * (j * j + k * k))


def quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    """Build the (i, j, k, real) quaternion for a rotation of yaw about +z."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def covariance(var_x: float, var_y: float, var_z: float) -> list[float]:
    """Build a row-major 3x3 diagonal covariance for a sensor_msgs/Imu field."""
    return [var_x, 0.0, 0.0,
            0.0, var_y, 0.0,
            0.0, 0.0, var_z]


def explains_jump(prev_yaw: float, yaw: float, yaw_rate: float,
                  dt: float, tolerance: float) -> bool:
    """Report whether the gyro accounts for the step in yaw since the last sample."""
    if not dt > 0.0:
        return True
    if not all(math.isfinite(v) for v in (prev_yaw, yaw, yaw_rate, dt, tolerance)):
        return True
    step = math.remainder(yaw - prev_yaw, 2.0 * math.pi)
    return abs(step - yaw_rate * dt) <= tolerance
