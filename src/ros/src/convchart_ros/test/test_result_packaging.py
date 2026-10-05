"""
Unit tests for packing an InferenceResult into convchart_interfaces/RosInferenceResult.

No GPU and no ROS graph: ConvChartROS._package_result is fed synthetic results with known
values, and every expectation comes from an independent reference (scipy for rotations, an
explicit index permutation for the covariance), never from the code under test.
"""

import math

from convchart_interfaces.msg import RosInferenceResult
from convchart_ros import convchart
from geometry_msgs.msg import PoseWithCovariance
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CompressedImage

# PoseWithCovariance.covariance is a row-major 6x6 over (x, y, z, rot_x, rot_y, rot_z);
# the pipeline's pose_cov is over (rvec, tvec). The same permutation maps either way.
ROS_ORDER = [3, 4, 5, 0, 1, 2]
RVEC = (0.35, -0.25, 0.15)
TVEC = (-0.104, -0.088, 0.56)           # metres: 40 mm squares, about 0.56 m away
RVEC_ALT = (-0.30, 0.28, 0.14)          # the other planar (IPPE) solution, tilt flipped
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
    # Symmetric positive definite, rotation and translation blocks on different scales and all
    # entries distinct, so any misplaced element changes the packed array.
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(6, 6))
    scale = np.sqrt([1e-5] * 3 + [1e-7] * 3)
    return (a @ a.T + 6.0 * np.eye(6)) * np.outer(scale, scale)


def make_result(rvec=RVEC, tvec=TVEC, *, reason=None, ambiguous=False, n_identified=13,
                n_unidentified=3, with_alt=True, with_cov=True):
    """Build an InferenceResult shaped like the ones src/inference.py returns."""
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
    # Skip __init__ (cfg, ONNX sessions, ROS graph): packing needs only its arguments.
    node = convchart.ConvChartROS.__new__(convchart.ConvChartROS)
    return node._package_result


def test_returns_a_ros_inference_result(pack):
    out = pack(image_msg(), make_result())
    assert isinstance(out, RosInferenceResult)
    assert isinstance(out.pose, PoseWithCovariance)
    assert isinstance(out.pose_alt, PoseWithCovariance)


def test_header_is_the_image_capture_header(pack):
    # The Pi's filter replays from the capture time, so this must not be the inference time.
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
    expected = Rotation.from_rotvec(rvec).as_quat()     # scipy uses ROS's (x, y, z, w) order
    assert same_rotation(q, expected), \
        f'orientation (x, y, z, w) = {q}, expected {expected} for rvec {rvec}'


def test_to_4x4_matrix_is_the_board_to_camera_transform():
    T = convchart.to_4x4_matrix(np.reshape(TVEC, (3, 1)), np.reshape(RVEC, (3, 1)))
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T[:3, :3], Rotation.from_rotvec(RVEC).as_matrix(), atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], TVEC)
    np.testing.assert_array_equal(T[3], [0.0, 0.0, 0.0, 1.0])


def test_covariance_is_row_major_in_ros_order(pack):
    result = make_result()
    out = pack(image_msg(), result)
    packed = [(out.pose, result['pose_cov'], out.covariance_valid),
              (out.pose_alt, result['pose_cov_alt'], out.covariance_valid_alt)]
    for pose, sigma6, valid in packed:
        expected = np.asarray(sigma6)[np.ix_(ROS_ORDER, ROS_ORDER)].ravel()
        got = np.asarray(pose.covariance)
        assert got.shape == (36,)
        np.testing.assert_allclose(
            got, expected, rtol=1e-12, atol=0.0,
            err_msg='expected pose_cov permuted to (x, y, z, rot_x, rot_y, rot_z), row-major')
        assert valid


def test_missing_covariance_is_flagged_not_faked(pack):
    # pose_cov is None when a used corner fell back to its coarse peak: the pose stands, but
    # the Pi must switch to its fallback R rather than trust the covariance field.
    out = pack(image_msg(), make_result(with_cov=False))
    assert not out.covariance_valid
    assert not out.covariance_valid_alt
    position = out.pose.pose.position
    assert [position.x, position.y, position.z] == pytest.approx(TVEC)


def test_quality_fields(pack):
    out = pack(image_msg(), make_result(n_identified=13, n_unidentified=3, ambiguous=True))
    assert out.reason == ''
    assert out.ambiguous is True
    assert out.num_used == 13           # identified corners only: the ones PnP used
    assert out.num_used_alt == 13       # both IPPE solutions come from the same corners
    assert out.rms == pytest.approx(0.21, rel=1e-6)     # float32 on the wire
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
