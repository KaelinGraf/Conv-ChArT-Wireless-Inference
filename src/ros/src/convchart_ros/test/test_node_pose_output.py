"""
End-to-end test of the convchart_ros node over ROS 2.

A synthetic ChArUco frame with a known board pose (built as in tests/test_parity.py) is published
as a PNG CompressedImage on 'image'; the node must answer on 'inference_result' with that pose in
metres, in ROS conventions, stamped with the image's capture time. Everything runs in-process on
an isolated domain with discovery limited to localhost, so it never reaches a real robot. Needs
CUDA, the ONNX checkpoints and the Conv-ChArT checkout (dcc); skipped without them.
"""

import math
import os
from pathlib import Path
import time

from convchart_interfaces.msg import RosInferenceResult
import cv2
import numpy as np
import pytest
import rclpy
from rclpy.duration import Duration
from rclpy.event_handler import PublisherEventCallbacks
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSLivelinessPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CompressedImage
import yaml

SENSOR = (1200, 1600)                   # native OV2311 frame the Pi sends
K = np.array([[1400.0, 0.0, 799.5], [0.0, 1400.0, 599.5], [0.0, 0.0, 1.0]])
RVEC_GT = np.array([0.35, -0.25, 0.15])
TVEC_GT_SQUARES = np.array([-2.6, -2.2, 14.0])
SQUARE_M = 0.04
BOARD_STAMP = (1000, 250000000)
BLANK_STAMP = (2000, 0)
FRAME_ID = 'pi_camera'
DOMAIN_ID = 77 + os.getpid() % 20
TIMEOUT_S = 60.0

# What the Pi's image publisher has to offer for the node's subscription to match it:
# reliable, deadline <= 0.2 s, liveliness lease <= 1 s.
IMAGE_QOS = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                       reliability=QoSReliabilityPolicy.RELIABLE,
                       durability=QoSDurabilityPolicy.VOLATILE,
                       deadline=Duration(seconds=0.1),
                       liveliness=QoSLivelinessPolicy.AUTOMATIC,
                       liveliness_lease_duration=Duration(seconds=0.5))
# A plain consumer, like the Pi's filter.
RESULT_QOS = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10,
                        reliability=QoSReliabilityPolicy.RELIABLE,
                        durability=QoSDurabilityPolicy.VOLATILE)


def find_repo():
    for parent in Path(__file__).resolve().parents:
        if (parent / 'cfg' / 'cfg.yaml').is_file() and (parent / 'checkpoints').is_dir():
            return parent
    return None


def board_frame(render_board):
    """Render the 5x5 board under the known pose into a native-resolution frame."""
    rng = np.random.default_rng(0)
    board, _ = render_board(480)
    sq = 480 // 5
    lattice = np.array([[1 / sq, 0, 0.5 / sq], [0, 1 / sq, 0.5 / sq], [0, 0, 1]])
    rot, _ = cv2.Rodrigues(RVEC_GT)
    homography = K @ np.column_stack([rot[:, 0], rot[:, 1], TVEC_GT_SQUARES]) @ lattice
    canvas = np.full(SENSOR, 140, np.uint8)
    warped = cv2.warpPerspective(board, homography, (SENSOR[1], SENSOR[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT,
                                 dst=canvas.copy())
    warped = cv2.GaussianBlur(warped, (0, 0), 1.2)
    noisy = warped.astype(np.float32) * 0.8 + rng.normal(0, 4, SENSOR)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def blank_frame():
    rng = np.random.default_rng(1)
    return np.clip(rng.normal(112, 4, SENSOR), 0, 255).astype(np.uint8)


def xyzw(q):
    return [q.x, q.y, q.z, q.w]


class Rig:
    """Hold the node under test plus a probe node that plays the Pi."""

    def __init__(self, node, render_board):
        self.node = node
        self.render_board = render_board
        self.probe = rclpy.create_node('convchart_test_probe')
        self.results = []
        self.incompatible = []
        self.probe.create_subscription(RosInferenceResult, 'inference_result',
                                       self.results.append, RESULT_QOS)
        events = PublisherEventCallbacks(
            incompatible_qos=lambda event: self.incompatible.append(event.last_policy_kind.name))
        self.pub = self.probe.create_publisher(CompressedImage, 'image', IMAGE_QOS,
                                               event_callbacks=events)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor.add_node(self.probe)

    def exchange(self, frame, stamp):
        """Publish a frame until a result carrying its stamp comes back, and return that."""
        msg = CompressedImage(format='mono8; png compressed mono8',
                              data=cv2.imencode('.png', frame)[1].tobytes())
        msg.header.stamp.sec, msg.header.stamp.nanosec = stamp
        msg.header.frame_id = FRAME_ID
        deadline, next_send = time.monotonic() + TIMEOUT_S, 0.0
        while time.monotonic() < deadline:
            for result in self.results:
                if (result.header.stamp.sec, result.header.stamp.nanosec) == stamp:
                    return result
            assert not self.incompatible, \
                f'image QoS incompatible with the node subscription: {self.incompatible}'
            # Re-send every second: the subscription's 0.15 s lifespan drops a copy that waits
            # out a slow callback (the first inference also pays for CUDA warm-up).
            if time.monotonic() >= next_send and self.pub.get_subscription_count() > 0:
                self.pub.publish(msg)
                next_send = time.monotonic() + 1.0
            self.executor.spin_once(timeout_sec=0.05)
        pytest.fail(f'no RosInferenceResult stamped {stamp} on inference_result within '
                    f'{TIMEOUT_S:.0f} s')

    def close(self):
        self.executor.shutdown()
        self.probe.destroy_node()
        self.node.destroy_node()


@pytest.fixture(scope='module')
def rig(tmp_path_factory):
    ort = pytest.importorskip('onnxruntime')
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        pytest.skip('the node runs the ONNX pipeline on CUDA')
    dcc_board = pytest.importorskip('dcc.board', reason='needs the Conv-ChArT checkout (dcc)')
    repo = find_repo()
    if repo is None:
        pytest.skip('repo root (cfg/cfg.yaml + checkpoints/) not found above this file')
    from convchart_ros import convchart

    cfg = yaml.safe_load((repo / 'cfg' / 'cfg.yaml').read_text())
    cfg['MODEL'] = {name: str(repo / path) for name, path in cfg['MODEL'].items()}
    cfg['BOARD'] = {'squares': [5, 5], 'square_length_m': SQUARE_M}
    cfg['CAMERA'] = {'K': K.tolist(), 'dist': [0.0] * 5}
    cfg_path = tmp_path_factory.mktemp('convchart') / 'cfg.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg))

    saved_range = os.environ.get('ROS_AUTOMATIC_DISCOVERY_RANGE')
    os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'
    rclpy.init(args=['--ros-args', '-r', '__ns:=/convchart_test'], domain_id=DOMAIN_ID)
    rig = None
    try:
        rig = Rig(convchart.ConvChartROS(cfg_pth=str(cfg_path)), dcc_board.render_board)
        yield rig
    finally:
        if rig is not None:
            rig.close()
        rclpy.shutdown()
        if saved_range is None:
            os.environ.pop('ROS_AUTOMATIC_DISCOVERY_RANGE', None)
        else:
            os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = saved_range


@pytest.fixture(scope='module')
def board_result(rig):
    return rig.exchange(board_frame(rig.render_board), BOARD_STAMP)


def test_result_is_stamped_with_the_image_capture_time(board_result):
    assert (board_result.header.stamp.sec, board_result.header.stamp.nanosec) == BOARD_STAMP
    assert board_result.header.frame_id == FRAME_ID


def test_board_frame_gives_a_valid_unambiguous_pose(board_result):
    assert board_result.reason == ''
    assert not board_result.ambiguous
    assert board_result.covariance_valid
    assert board_result.covariance_valid_alt
    assert 12 <= board_result.num_used <= 16
    assert board_result.num_used_alt == board_result.num_used
    assert 0.0 < board_result.rms < 0.5
    assert board_result.rms_alt >= board_result.rms


def test_pose_matches_the_rendered_board(board_result):
    pose = board_result.pose.pose
    position = [pose.position.x, pose.position.y, pose.position.z]
    np.testing.assert_allclose(position, SQUARE_M * TVEC_GT_SQUARES, atol=0.05 * SQUARE_M)
    q = xyzw(pose.orientation)
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-6)
    error = Rotation.from_quat(q) * Rotation.from_rotvec(RVEC_GT).inv()
    assert math.degrees(error.magnitude()) < 0.5, \
        f'orientation (x, y, z, w) = {q} is not the rendered board pose'


def test_covariance_is_ros_ordered(board_result):
    cov = np.asarray(board_result.pose.covariance).reshape(6, 6)
    np.testing.assert_allclose(cov, cov.T, rtol=1e-9, atol=1e-18)
    assert np.all(np.linalg.eigvalsh(cov) > 0)
    var = np.diag(cov)
    # (x, y, z, rot_x, rot_y, rot_z): depth along the optical axis is the worst-observed
    # translation and rotation about it the best-observed rotation. The pipeline's unpermuted
    # (rvec, tvec) order breaks both.
    assert var[2] > max(var[0], var[1])
    assert var[5] < min(var[3], var[4])


def test_alternative_is_the_other_planar_solution(board_result):
    main_q = xyzw(board_result.pose.pose.orientation)
    alt_q = xyzw(board_result.pose_alt.pose.orientation)
    assert np.linalg.norm(alt_q) == pytest.approx(1.0, abs=1e-6)
    angle = (Rotation.from_quat(alt_q) * Rotation.from_quat(main_q).inv()).magnitude()
    assert math.degrees(angle) > 1.0


def test_frame_without_a_board_is_answered_with_a_reason(rig):
    result = rig.exchange(blank_frame(), BLANK_STAMP)
    assert result.reason != ''
    assert not result.covariance_valid
