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

# Below this the quaternion carries no usable direction and normalising it would
# amplify noise into a confident wrong answer. The BNO reports a unit quaternion,
# so anything near zero means a corrupt report rather than a real orientation.
_MIN_QUAT_NORM = 1e-6


@dataclass(frozen=True)
class ImuSample:
    """One coherent set of readings, taken at a single instant.

    Coherent matters: the three BNO reports arrive in separate packets, and a
    sample built from an accelerometer reading at one instant and a gyro reading
    at the next is wrong in a way nothing downstream can detect. A backend must
    fill all three fields from one drain of the sensor's queue.

    accel is m/s^2 INCLUDING gravity, which is what sensor_msgs/Imu specifies for
    linear_acceleration. gyro is rad/s. quat is the game rotation vector as
    (i, j, k, real) -- the same ordering the Adafruit library returns, kept rather
    than reshuffled so the backend does no arithmetic of its own.
    """

    accel: tuple[float, float, float]
    gyro: tuple[float, float, float]
    quat: tuple[float, float, float, float]


def normalise_quaternion(
        quat: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    """Scale a quaternion to unit length, or return None if it has no direction.

    sensor_msgs/Imu consumers are entitled to assume unit length. The BNO's game
    rotation vector is already unit to within its Q14 fixed-point resolution, so
    this is a guard against a corrupt report rather than a routine correction.
    """
    i, j, k, real = quat
    norm = math.sqrt(i * i + j * j + k * k + real * real)
    if not math.isfinite(norm) or norm < _MIN_QUAT_NORM:
        return None
    return (i / norm, j / norm, k / norm, real / norm)


def validate(sample: ImuSample) -> str | None:
    """Return why the sample is unusable, or None when it is good.

    Three gates, each for a failure the Adafruit library hands over silently:

    NON-FINITE. A NaN would poison a filter without ever raising, exactly as
    protocol.py guards against on the serial link.

    ZERO-NORM QUATERNION. enable_feature() returns as soon as the feature id
    appears in the library's readings dict, and the dict entry it waits for is a
    PLACEHOLDER -- _INITIAL_REPORTS seeds the game rotation vector as
    (0, 0, 0, 0). So for the first few cycles after enabling, the sensor reads as
    a zero quaternion with no error, and a node that publishes what it is handed
    emits a fabricated yaw of exactly 0.

    ALL-ZERO ACCELERATION. The same placeholder, as (0, 0, 0). It is also
    physically impossible: a stationary accelerometer reads gravity, so a true
    zero means the chip is not measuring. Note this is an EXACT comparison, not a
    tolerance -- real readings are never exactly zero, and a tolerance would throw
    away genuine free-fall data.
    """
    if not all(math.isfinite(v) for v in (*sample.accel, *sample.gyro, *sample.quat)):
        return f'non-finite reading: accel={sample.accel} gyro={sample.gyro} quat={sample.quat}'
    if normalise_quaternion(sample.quat) is None:
        return f'quaternion has no direction: {sample.quat} (sensor not streaming yet?)'
    if sample.accel == (0.0, 0.0, 0.0):
        return 'acceleration is exactly zero (sensor not streaming yet?)'
    return None


def samples_differ(a: ImuSample | None, b: ImuSample) -> bool:
    """Report whether b carries anything new.

    The ONLY liveness signal available. The library exposes no new-data flag and
    no sensor timestamp, and its readings dict holds last-known values: once the
    chip stops streaming -- which is what a spontaneous reset looks like, with the
    enabled features cleared -- every read keeps returning the same tuple
    successfully, forever. Nothing raises and nothing is stale-marked.

    Exact comparison, deliberately. Gyro noise alone changes the low bits of a
    live sensor every report, so two byte-identical samples mean the dict was not
    refreshed rather than that the bot held unusually still.
    """
    return a is None or (a.accel, a.gyro, a.quat) != (b.accel, b.gyro, b.quat)


def yaw_from_quaternion(quat: tuple[float, float, float, float]) -> float:
    """Extract rotation about +z, in (-pi, pi].

    This is NOT what gets published -- the message carries the quaternion itself.
    It exists for the log lines and for explains_jump, both of which need a single
    number. The result necessarily wraps, which is the whole reason the serial
    bridge used to carry an unwrapped heading alongside the quaternion; nothing
    does any more, so a consumer that differences heading must unwrap it itself.
    """
    i, j, k, real = quat
    return math.atan2(2.0 * (real * k + i * j),
                      1.0 - 2.0 * (j * j + k * k))


def quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    """Build the (i, j, k, real) quaternion for a rotation of yaw about +z.

    The inverse of yaw_from_quaternion for any level orientation. Not used on the
    sensor path -- the BNO's quaternion is published as it comes -- but the mock
    backend builds its synthetic orientation with it, which lets a pure test
    round-trip the two against each other instead of restating either formula.
    """
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def covariance(var_x: float, var_y: float, var_z: float) -> list[float]:
    """Build a row-major 3x3 diagonal covariance for a sensor_msgs/Imu field.

    Diagonal because the BNO reports no cross-axis correlation and inventing one
    would be fiction. Note what is NOT here: the -1.0 sentinel in element 0 that
    sensor_msgs/Imu uses for "this quantity is not measured". Every quantity this
    node publishes IS measured, and a stray -1 would tell an off-the-shelf
    consumer to discard it.
    """
    return [var_x, 0.0, 0.0,
            0.0, var_y, 0.0,
            0.0, 0.0, var_z]


def explains_jump(prev_yaw: float, yaw: float, yaw_rate: float,
                  dt: float, tolerance: float) -> bool:
    """Report whether the gyro accounts for the step in yaw since the last sample.

    A BNO085 reset re-zeroes the game rotation vector, so heading teleports while
    the gyro reports nothing unusual. That disagreement is the only evidence of a
    reset the Adafruit library leaves us -- it exposes no reset flag, and with no
    reset pin wired we cannot even force one to compare against.

    Compares the shortest-arc yaw step against yaw_rate * dt. False means the step
    is unexplained and a reset is likely.

    Two honest limitations. A reset that lands while heading is already near zero
    produces no detectable step, so this undercounts rather than overcounts. And
    dt <= 0 or a non-finite input returns True: with nothing to compare, staying
    quiet beats crying wolf on every startup sample.
    """
    if not dt > 0.0:
        return True
    if not all(math.isfinite(v) for v in (prev_yaw, yaw, yaw_rate, dt, tolerance)):
        return True
    # remainder, not fmod: the step has to be the shortest arc, so that a sample
    # crossing +pi reads as a small positive move rather than a ~2pi jump.
    step = math.remainder(yaw - prev_yaw, 2.0 * math.pi)
    return abs(step - yaw_rate * dt) <= tolerance
