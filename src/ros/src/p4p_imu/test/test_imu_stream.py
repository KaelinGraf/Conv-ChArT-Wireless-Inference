"""What actually lands on imu/data.

The contract these pin down is the one a consumer reads off the message: the
frame, a unit quaternion, every covariance populated with no "not measured"
sentinel, and all three vectors filled. Plus the accepted consequence of
publishing sensor_msgs/Imu alone -- heading wraps, and nothing unwraps it.
"""
import math

from p4p_imu import orientation
import pytest


def _quat(msg):
    o = msg.orientation
    return (o.x, o.y, o.z, o.w)


def test_samples_arrive_at_all(rig):
    assert rig.wait_for_samples(5)


def test_the_frame_is_base_link(rig):
    """What the serial bridge used for this same data."""
    assert rig.wait_for_samples(1)[0].header.frame_id == 'base_link'


def test_the_quaternion_is_unit_length(rig):
    """Consumers are entitled to assume this; the BNO's Q14 output is only close."""
    for msg in rig.wait_for_samples(10):
        assert math.sqrt(sum(c * c for c in _quat(msg))) == pytest.approx(1.0, abs=1e-6)


def test_stamps_advance(rig):
    stamps = [m.header.stamp.sec + m.header.stamp.nanosec / 1e9
              for m in rig.wait_for_samples(10)]
    assert all(b >= a for a, b in zip(stamps, stamps[1:]))


def test_the_turn_rate_reaches_angular_velocity_z(make_rig):
    """mock_yaw_rate goes in, angular_velocity.z comes out: the node is not
    dropping or reordering the gyro vector."""
    rig = make_rig(mock_yaw_rate=1.25, mock_noise=0.0)
    for msg in rig.wait_for_samples(5):
        assert msg.angular_velocity.z == pytest.approx(1.25, abs=1e-6)


def test_acceleration_is_published_with_gravity_in_z(make_rig):
    """BNO_REPORT_ACCELEROMETER, not LINEAR_ACCELERATION: sensor_msgs/Imu
    specifies linear_acceleration as including gravity."""
    rig = make_rig(mock_noise=0.0)
    msg = rig.wait_for_samples(1)[0]
    assert msg.linear_acceleration.z == pytest.approx(9.80665, abs=1e-6)


def test_heading_wraps_at_pi_and_nothing_unwraps_it(make_rig):
    """The accepted cost of publishing sensor_msgs/Imu alone. Asserted rather
    than assumed, so that the limitation stays documented in behaviour."""
    rig = make_rig(mock_yaw_rate=4.0, mock_noise=0.0)
    yaws = [orientation.yaw_from_quaternion(_quat(m)) for m in rig.wait_for_samples(300)]
    assert all(-math.pi <= y <= math.pi for y in yaws)
    assert any(b - a < -math.pi for a, b in zip(yaws, yaws[1:])), \
        'heading never wrapped; the test is not exercising the wrap'


# --- covariances --------------------------------------------------------------

def test_every_covariance_is_populated(make_rig):
    rig = make_rig(var_yaw=0.02, var_roll_pitch=1e6, var_gyro=2e-4, var_accel=0.03)
    msg = rig.wait_for_samples(1)[0]
    assert list(msg.orientation_covariance) == orientation.covariance(1e6, 1e6, 0.02)
    assert list(msg.angular_velocity_covariance) == orientation.covariance(2e-4, 2e-4, 2e-4)
    assert list(msg.linear_acceleration_covariance) == orientation.covariance(0.03, 0.03, 0.03)


def test_no_covariance_carries_the_not_measured_sentinel(rig):
    """REGRESSION. p4p_serial_bridge set linear_acceleration_covariance[0] = -1.0
    because the Mega never sent acceleration. This node measures ax/ay/az, so a
    -1 here would tell robot_localization to discard the data the node exists to
    publish."""
    msg = rig.wait_for_samples(1)[0]
    for name in ('orientation_covariance', 'angular_velocity_covariance',
                 'linear_acceleration_covariance'):
        cov = list(getattr(msg, name))
        assert len(cov) == 9, name
        assert -1.0 not in cov, f'{name} carries the "not measured" sentinel'


def test_roll_and_pitch_are_flagged_uncharacterised_not_unmeasured(rig):
    """1e6 on the diagonal says "do not trust this yet". The sentinel for genuinely
    unmeasured is -1.0 in element 0, which would condemn the whole quaternion."""
    cov = list(rig.wait_for_samples(1)[0].orientation_covariance)
    assert cov[0] == 1e6 and cov[4] == 1e6
    assert cov[8] < 1.0
