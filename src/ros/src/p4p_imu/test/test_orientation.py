"""Pure tests for p4p_imu.orientation: no ROS, no adafruit, no sensor.

Run them anywhere:

    PYTHONPATH=src/ros/src/p4p_imu python3 -m pytest src/ros/src/p4p_imu/test/test_orientation.py
"""
import math

from p4p_imu import orientation
from p4p_imu.orientation import ImuSample
import pytest

GOOD = ImuSample(accel=(0.1, -0.2, 9.8), gyro=(0.0, 0.0, 0.5),
                 quat=(0.0, 0.0, 0.0, 1.0))


# --- an independent yaw, for checking the formula against ---------------------

def _qmul(a, b):
    """Hamilton product of two (w, x, y, z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def _yaw_by_rotating_x(quat):
    """Yaw as the azimuth of the rotated +x axis, via explicit quaternion algebra.

    Deliberately NOT the atan2 identity yaw_from_quaternion uses, so that the test
    checks the function instead of restating it.
    """
    i, j, k, real = quat
    q = (real, i, j, k)
    conj = (real, -i, -j, -k)
    rotated = _qmul(_qmul(q, (0.0, 1.0, 0.0, 0.0)), conj)
    return math.atan2(rotated[2], rotated[1])


# --- yaw ----------------------------------------------------------------------

def test_a_quarter_turn_about_z_reads_as_ninety_degrees():
    """Pins the (i, j, k, real) ordering the Adafruit library returns."""
    quat = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    assert orientation.yaw_from_quaternion(quat) == pytest.approx(math.pi / 2)


def test_identity_is_zero_yaw():
    assert orientation.yaw_from_quaternion((0.0, 0.0, 0.0, 1.0)) == pytest.approx(0.0)


@pytest.mark.parametrize('yaw', [-3.0, -1.5, -0.1, 0.0, 0.1, 1.5, 3.0, 3.14])
def test_yaw_round_trips_through_the_quaternion(yaw):
    quat = orientation.quaternion_from_yaw(yaw)
    assert orientation.yaw_from_quaternion(quat) == pytest.approx(yaw)
    assert _yaw_by_rotating_x(quat) == pytest.approx(yaw)


@pytest.mark.parametrize('roll,pitch,yaw', [
    (0.3, 0.0, 1.0), (0.0, 0.4, -2.0), (-0.2, 0.25, 2.5),
])
def test_yaw_is_the_zyx_yaw_even_when_the_board_is_not_level(roll, pitch, yaw):
    """The flat-mount assumption broken: roll and pitch must not leak into yaw."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    quat = (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)
    assert orientation.yaw_from_quaternion(quat) == pytest.approx(yaw)
    assert _yaw_by_rotating_x(quat) == pytest.approx(yaw)


# --- normalisation ------------------------------------------------------------

def test_a_slightly_long_quaternion_is_scaled_rather_than_rejected():
    """The BNO's Q14 fixed point lands just off unit length; that is not an error."""
    quat = orientation.normalise_quaternion((0.0, 0.0, 0.01, 1.01))
    assert quat is not None
    assert math.sqrt(sum(c * c for c in quat)) == pytest.approx(1.0)


def test_a_zero_quaternion_has_no_direction():
    """The vendor library's placeholder. Normalising it would invent a heading."""
    assert orientation.normalise_quaternion((0.0, 0.0, 0.0, 0.0)) is None


def test_a_non_finite_quaternion_has_no_direction():
    assert orientation.normalise_quaternion((0.0, 0.0, 0.0, float('nan'))) is None


# --- validate -----------------------------------------------------------------

def test_a_real_sample_is_valid():
    assert orientation.validate(GOOD) is None


def test_the_libraries_placeholder_sample_is_rejected():
    """enable_feature() returns carrying exactly this; publishing it would fabricate
    a yaw of 0 and claim the bot is in free fall."""
    placeholder = ImuSample(accel=(0.0, 0.0, 0.0), gyro=(0.0, 0.0, 0.0),
                            quat=(0.0, 0.0, 0.0, 0.0))
    assert orientation.validate(placeholder) is not None


def test_exactly_zero_acceleration_is_rejected():
    """A stationary accelerometer reads gravity, so a true zero means not streaming."""
    sample = ImuSample(accel=(0.0, 0.0, 0.0), gyro=GOOD.gyro, quat=GOOD.quat)
    assert 'exactly zero' in orientation.validate(sample)


@pytest.mark.parametrize('field', ['accel', 'gyro', 'quat'])
def test_a_nan_anywhere_is_rejected(field):
    """A NaN would poison a filter without ever raising."""
    parts = {'accel': GOOD.accel, 'gyro': GOOD.gyro, 'quat': GOOD.quat}
    parts[field] = tuple([float('nan')] + list(parts[field])[1:])
    assert 'non-finite' in orientation.validate(ImuSample(**parts))


# --- liveness -----------------------------------------------------------------

def test_the_same_sample_twice_carries_nothing_new():
    """How a silent reset shows up: the library serves its cache forever."""
    assert not orientation.samples_differ(GOOD, GOOD)


def test_one_changed_bit_counts_as_new():
    moved = ImuSample(accel=GOOD.accel, gyro=(0.0, 0.0, 0.5000001), quat=GOOD.quat)
    assert orientation.samples_differ(GOOD, moved)


def test_the_first_sample_always_counts_as_new():
    assert orientation.samples_differ(None, GOOD)


# --- covariance ---------------------------------------------------------------

def test_covariance_is_row_major_diagonal():
    assert orientation.covariance(1.0, 2.0, 3.0) == [1.0, 0.0, 0.0,
                                                     0.0, 2.0, 0.0,
                                                     0.0, 0.0, 3.0]


def test_covariance_never_emits_the_not_measured_sentinel():
    """REGRESSION. p4p_serial_bridge set linear_acceleration_covariance[0] = -1.0
    because the Mega never sent acceleration. This node measures ax/ay/az, so a
    -1 would now be a false statement that tells robot_localization to discard
    the very data the node exists to publish. Re-adding it while copying from the
    bridge is the obvious mistake, so it is pinned here."""
    for cov in (orientation.covariance(1e6, 1e6, 0.01),
                orientation.covariance(1e-4, 1e-4, 1e-4),
                orientation.covariance(0.01, 0.01, 0.01)):
        assert len(cov) == 9
        assert -1.0 not in cov


# --- reset detection ----------------------------------------------------------

def test_a_re_zeroed_heading_is_not_explained_by_the_gyro():
    """The signature of a spontaneous BNO085 reset."""
    assert not orientation.explains_jump(2.0, 0.0, 0.1, 0.01, 0.35)


def test_a_fast_but_consistent_turn_is_explained():
    assert orientation.explains_jump(0.0, 0.02, 2.0, 0.01, 0.35)


def test_crossing_pi_is_explained_rather_than_read_as_a_jump():
    """remainder(), not fmod(): the step has to be the shortest arc."""
    assert orientation.explains_jump(3.13, -3.13, 2.0, 0.01, 0.35)


def test_without_a_time_base_we_stay_quiet():
    """dt <= 0 happens on the first sample; crying wolf there would be noise."""
    assert orientation.explains_jump(2.0, 0.0, 0.0, 0.0, 0.35)


def test_a_non_finite_input_stays_quiet():
    assert orientation.explains_jump(float('nan'), 0.0, 0.0, 0.01, 0.35)
