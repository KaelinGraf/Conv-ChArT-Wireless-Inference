"""
Unit tests for packing an InferenceResult into convchart_interfaces/RosInferenceResult.

No GPU and no ROS graph: ConvChartROS._package_result is fed synthetic results with known
values, and every expectation comes from an independent reference (scipy for rotations, the
measured noise law written out below for the covariance), never from the code under test.
"""

import math

from convchart_interfaces.msg import RosInferenceResult
from convchart_ros import convchart
from geometry_msgs.msg import PoseWithCovariance
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CompressedImage

NOISE_A = (1.1196e-4, 1.5808e-4, 3.8022e-5, 5.2027e-5, 4.6866e-5, 1.0236e-4)
NOISE_K = (0.9674, 0.8074, 1.0171, 1.5016, 1.4944, 1.7337)
Z_FITTED = (3.88, 70.67)
SQUARE_M = 0.04
RVEC = (0.35, -0.25, 0.15)
TVEC = (-0.104, -0.088, 0.56)
RVEC_ALT = (-0.30, 0.28, 0.14)
TVEC_ALT = (-0.110, -0.085, 0.57)
REASONS = ['too_few', 'collinear', 'no_intrinsics', 'vacuous_uncorroborated',
           'too_few_correspondences', 'pnp_solver_failed']
STAMP = (1712345678, 123456789)
FRAME_ID = 'pi_camera'


def corner(index, sigma_px=0.08):
    return {'x': 100.0, 'y': 200.0, 'index': index, 'x_coarse': 100.0, 'y_coarse': 200.0,
            'source': None if index is None else 'head', 'p_hm': 0.9, 'p_id': 0.9,
            'sigma_px': sigma_px}


def covariance(seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(6, 6))
    scale = np.sqrt([1e-5] * 3 + [1e-7] * 3)
    return (a @ a.T + 6.0 * np.eye(6)) * np.outer(scale, scale)


def make_result(rvec=RVEC, tvec=TVEC, *, reason=None, ambiguous=False, n_identified=13,
                n_unidentified=3, with_alt=True, with_cov=True):
    """Build an InferenceResult shaped like the ones src/inference.py returns.

    Args:
        rvec: Rodrigues rotation vector of the pose.
        tvec: Translation vector of the pose, in metres.
        reason: Refusal reason, or None for a solved pose.
        ambiguous: Whether the pose is flagged ambiguous.
        n_identified: Number of identified corners.
        n_unidentified: Number of unidentified corners.
        with_alt: Whether the second IPPE solution is included.
        with_cov: Whether the pose covariances are included.
    """
    solved = reason is None
    alt = solved and with_alt

    def column(v):
        return np.asarray(v, dtype=np.float64).reshape(3, 1)

    corners = [corner(i) for i in range(n_identified)]
    corners += [corner(None) for _ in range(n_unidentified)]
    return {
        'rvec': column(rvec) if solved else None,
        'tvec': column(tvec) if solved else None,
        'rms': 0.21 if solved else None,
        'reason': reason,
        'corners': corners,
        'pose_cov': covariance(0).tolist() if solved and with_cov else None,
        'ambiguous': ambiguous,
        'rvec_alt': column(RVEC_ALT) if alt else None,
        'tvec_alt': column(TVEC_ALT) if alt else None,
        'rms_alt': 0.43 if alt else None,
        'pose_cov_alt': covariance(1).tolist() if alt and with_cov else None,
        'demoted': 1,
        'recovered': 2,
    }


def expected_diagonal(tvec_z, square_m=SQUARE_M):
    """Return the variances in PoseWithCovariance order (x, y, z, rot_x, rot_y, rot_z), in metres.

    Args:
        tvec_z: Range of the board, in metres.
        square_m: Board square length, in metres.
    """
    z = tvec_z / square_m
    sigma = [a * z ** k for a, k in zip(NOISE_A, NOISE_K)]
    return np.array([s * square_m for s in sigma[3:]] + sigma[:3]) ** 2


def image_msg():
    msg = CompressedImage(format='mono8; png compressed mono8')
    msg.header.stamp.sec, msg.header.stamp.nanosec = STAMP
    msg.header.frame_id = FRAME_ID
    return msg


def xyzw(q):
    return np.array([q.x, q.y, q.z, q.w])


def same_rotation(q, expected_xyzw):
    return abs(float(np.dot(q, expected_xyzw))) == pytest.approx(1.0, abs=1e-9)


@pytest.fixture
def pack():
    node = convchart.ConvChartROS.__new__(convchart.ConvChartROS)
    return lambda img, res: node._package_result(img, res, SQUARE_M)


def test_returns_a_ros_inference_result(pack):
    out = pack(image_msg(), make_result())
    assert isinstance(out, RosInferenceResult)
    assert isinstance(out.pose, PoseWithCovariance)
    assert isinstance(out.pose_alt, PoseWithCovariance)


def test_header_is_the_image_capture_header(pack):
    out = pack(image_msg(), make_result())
    assert (out.header.stamp.sec, out.header.stamp.nanosec) == STAMP
    assert out.header.frame_id == FRAME_ID


@pytest.mark.parametrize('rvec', [RVEC, (0.0, 0.0, math.pi / 2), (2.9, 0.4, -0.3),
                                  (1e-9, 0.0, 0.0)])
def test_pose_is_rvec_tvec_with_a_ros_quaternion(pack, rvec):
    pose = pack(image_msg(), make_result(rvec=rvec)).pose.pose
    assert [pose.position.x, pose.position.y, pose.position.z] == pytest.approx(TVEC)
    q = xyzw(pose.orientation)
    assert np.linalg.norm(q) == pytest.approx(1.0)
    expected = Rotation.from_rotvec(rvec).as_quat()
    assert same_rotation(q, expected), \
        f'orientation (x, y, z, w) = {q}, expected {expected} for rvec {rvec}'


def test_to_4x4_matrix_is_the_board_to_camera_transform():
    T = convchart.to_4x4_matrix(np.reshape(TVEC, (3, 1)), np.reshape(RVEC, (3, 1)))
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T[:3, :3], Rotation.from_rotvec(RVEC).as_matrix(), atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], TVEC)
    np.testing.assert_array_equal(T[3], [0.0, 0.0, 0.0, 1.0])


def test_covariance_is_the_range_scaled_measurement_noise_in_ros_order(pack):
    out = pack(image_msg(), make_result())
    packed = [(out.pose, TVEC, out.covariance_valid), (out.pose_alt, TVEC_ALT, out.covariance_valid_alt)]
    for pose, tvec, valid in packed:
        got = np.asarray(pose.covariance)
        assert got.shape == (36,)
        cov = got.reshape(6, 6)
        np.testing.assert_allclose(
            np.diag(cov), expected_diagonal(tvec[2]), rtol=1e-12, atol=0.0,
            err_msg='expected sigma_i(z)^2 in (x, y, z, rot_x, rot_y, rot_z) order, translation in metres')
        assert np.count_nonzero(cov - np.diag(np.diag(cov))) == 0, 'off-diagonals must be zero'
        assert valid


def test_the_pipelines_own_pose_cov_never_reaches_the_wire(pack):
    with_cov = pack(image_msg(), make_result(with_cov=True))
    without = pack(image_msg(), make_result(with_cov=False))
    assert list(with_cov.pose.covariance) == list(without.pose.covariance)
    assert with_cov.covariance_valid and without.covariance_valid


@pytest.mark.parametrize('tvec_z, z_clamped', [(0.05, Z_FITTED[0]), (3.0, Z_FITTED[1])])
def test_a_range_outside_the_fit_is_clamped_and_flagged(pack, tvec_z, z_clamped):
    tvec = (TVEC[0], TVEC[1], tvec_z)
    out = pack(image_msg(), make_result(tvec=tvec, with_alt=False))
    assert not out.covariance_valid
    cov = np.asarray(out.pose.covariance).reshape(6, 6)
    np.testing.assert_allclose(np.diag(cov), expected_diagonal(z_clamped * SQUARE_M), rtol=1e-12, atol=0.0)
    position = out.pose.pose.position
    assert [position.x, position.y, position.z] == pytest.approx(tvec)


def test_quality_fields(pack):
    out = pack(image_msg(), make_result(n_identified=13, n_unidentified=3, ambiguous=True))
    assert out.reason == ''
    assert out.ambiguous is True
    assert out.num_used == 13
    assert out.num_used_alt == 13
    assert out.rms == pytest.approx(0.21, rel=1e-6)
    assert out.rms_alt == pytest.approx(0.43, rel=1e-6)


def test_alternative_solution_is_packed(pack):
    alt = pack(image_msg(), make_result(ambiguous=True)).pose_alt.pose
    assert [alt.position.x, alt.position.y, alt.position.z] == pytest.approx(TVEC_ALT)
    assert same_rotation(xyzw(alt.orientation), Rotation.from_rotvec(RVEC_ALT).as_quat())


def test_single_solution_marks_the_alternative_invalid(pack):
    out = pack(image_msg(), make_result(with_alt=False))
    assert out.covariance_valid
    assert not out.covariance_valid_alt


@pytest.mark.parametrize('reason', REASONS)
def test_refusal_carries_its_reason_and_no_covariance(pack, reason):
    out = pack(image_msg(), make_result(reason=reason))
    assert out.reason == reason
    assert not out.covariance_valid
    assert not out.covariance_valid_alt
    assert (out.header.stamp.sec, out.header.stamp.nanosec) == STAMP
