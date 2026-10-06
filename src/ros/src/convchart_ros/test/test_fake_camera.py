"""
Tests for the fake camera node (fake_camera.py) over ROS 2.

FakeCamera runs in-process beside a probe node that subscribes to 'image' and calls 'capture', all
on an isolated domain with discovery limited to localhost and a private namespace, so nothing here
can reach a real robot or another test run; main() runs as a subprocess under the same isolation.
Captures are checked against the camera's own ground truth (K_full, dist, poses) with OpenCV's
ChArUco detector.
"""

import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from convchart_interfaces.srv import CaptureFrame
import convchart_ros
from convchart_ros.calib_utils.calib_board import BoardSpec, make_board
from convchart_ros.calib_utils.calib_session import charuco_offset_px
from convchart_ros.calib_utils.calib_synth import (
    DEFAULT_DIST, DEFAULT_FULL_SIZE, DEFAULT_K_FULL, DEFAULT_STREAM_SIZE, diverse_poses,
    render_view)
from convchart_ros.calib_utils.fake_camera import FakeCamera
import cv2
import numpy as np
import pytest
import rclpy
from rclpy.duration import Duration
from rclpy.event_handler import SubscriptionEventCallbacks
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSLivelinessPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage
import yaml

DOMAIN_ID = 51 + os.getpid() % 20       # private: not the robot's domain, not the other ROS test's
NAMESPACE = f'/fake_camera_test_{os.getpid()}'     # concurrent runs never share topics
TIMEOUT_S = 30.0
PNG_FORMAT = 'mono8; png compressed mono8'
FULL_SHAPE = (DEFAULT_FULL_SIZE[1], DEFAULT_FULL_SIZE[0])
STREAM_SHAPE = (DEFAULT_STREAM_SIZE[1], DEFAULT_STREAM_SIZE[0])
# DEFAULT_K_FULL scaled to 400x300 (pixel-centre convention): a small camera for fast captures.
K_SMALL = [[350.0, 0.0, 199.5], [0.0, 350.0, 149.5], [0.0, 0.0, 1.0]]
# How far the installed cv2.aruco.CharucoDetector reports corners off the pixel-centre convention
# of cv2.projectPoints (see test_calib_synth.py).
DETECTOR_SHIFT = charuco_offset_px()

# The interface's QoS for 'image'.
STREAM_QOS = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                        reliability=QoSReliabilityPolicy.RELIABLE,
                        durability=QoSDurabilityPolicy.VOLATILE)
# What convchart's inference node asks of its 'image' publisher.
INFERENCE_QOS = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                           reliability=QoSReliabilityPolicy.RELIABLE,
                           durability=QoSDurabilityPolicy.VOLATILE,
                           deadline=Duration(seconds=0.2),
                           lifespan=Duration(seconds=0.15),
                           liveliness=QoSLivelinessPolicy.AUTOMATIC,
                           liveliness_lease_duration=Duration(seconds=1.0))


def wait_until(condition, what, timeout=TIMEOUT_S):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            pytest.fail(f'timed out after {timeout:.0f} s waiting for {what}')
        time.sleep(0.005)


class Rig:
    """Spin a probe node (stream subscriber, capture client) and the camera under test, if any."""

    def __init__(self, camera=None, prefix=''):
        self.camera = camera
        self.probe = rclpy.create_node('fake_camera_test_probe')
        self.frames = []                # (monotonic arrival time, CompressedImage)
        self.probe.create_subscription(CompressedImage, prefix + 'image', self._on_frame,
                                       STREAM_QOS)
        self.client = self.probe.create_client(CaptureFrame, prefix + 'capture')
        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(self.probe)
        if camera is not None:
            self.executor.add_node(camera)
        self.spinner = threading.Thread(target=self.executor.spin, daemon=True)
        self.spinner.start()

    def _on_frame(self, msg):
        self.frames.append((time.monotonic(), msg))

    def wait_for_frames(self, count):
        """Wait until count stream frames have arrived; return the latest one."""
        wait_until(lambda: len(self.frames) >= count, f'{count} stream frames')
        return self.frames[-1][1]

    def capture(self):
        """Call 'capture' once; return the reply and the call's (start, end) monotonic times."""
        assert self.client.wait_for_service(timeout_sec=TIMEOUT_S), "no 'capture' service"
        start = time.monotonic()
        future = self.client.call_async(CaptureFrame.Request())
        wait_until(future.done, 'a capture reply')
        return future.result(), (start, time.monotonic())

    def close(self):
        self.executor.shutdown()
        self.spinner.join(timeout=TIMEOUT_S)
        self.probe.destroy_node()
        if self.camera is not None:
            self.camera.destroy_node()


@pytest.fixture(scope='module')
def ros():
    saved_range = os.environ.get('ROS_AUTOMATIC_DISCOVERY_RANGE')
    os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'
    rclpy.init(args=['--ros-args', '-r', f'__ns:={NAMESPACE}'], domain_id=DOMAIN_ID)
    try:
        yield
    finally:
        rclpy.shutdown()
        if saved_range is None:
            os.environ.pop('ROS_AUTOMATIC_DISCOVERY_RANGE', None)
        else:
            os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = saved_range


@pytest.fixture
def make_rig(ros):
    rigs = []

    def make(**camera_args):
        rigs.append(Rig(FakeCamera(**camera_args)))
        return rigs[-1]

    yield make
    for rig in rigs:
        rig.close()


def decode(image):
    """Decode a CompressedImage that must be a mono8 PNG."""
    assert image.format == PNG_FORMAT
    data = np.frombuffer(image.data, dtype=np.uint8)
    assert data[:8].tobytes() == b'\x89PNG\r\n\x1a\n'
    frame = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    assert frame is not None
    assert frame.dtype == np.uint8
    assert frame.ndim == 2
    return frame


def corner_error(frame, spec, pose, k_full=DEFAULT_K_FULL, dist=DEFAULT_DIST):
    """Median distance (px) from the corners detected in frame to a pose's projected corners."""
    board = make_board(spec)
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(frame)
    if ids is None:
        return np.inf
    object_points = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    truth = cv2.projectPoints(object_points, pose[0], pose[1], np.asarray(k_full),
                              np.asarray(dist))[0].reshape(-1, 2)
    found = corners.reshape(-1, 2) - DETECTOR_SHIFT
    return float(np.median(np.linalg.norm(found - truth[ids.ravel()], axis=1)))


def pose_error(camera, frame, index):
    """Median distance (px) from the corners detected in frame to the camera's pose index."""
    return corner_error(frame, camera.board_spec, camera.poses[index], camera.K_full, camera.dist)


def stream_view(camera, index):
    """Render a pose the way the stream shows it, with noise of its own."""
    frame, _, _ = render_view(camera.board, camera.board_spec, camera.K_full, camera.dist,
                              *camera.poses[index], camera.full_size, seed=99)
    return cv2.resize(frame, camera.stream_size, interpolation=cv2.INTER_AREA)


def mean_difference(a, b):
    return float(np.mean(np.abs(a.astype(np.float64) - b)))


def test_ground_truth_is_exposed(ros):
    camera = FakeCamera()
    try:
        np.testing.assert_array_equal(camera.K_full, DEFAULT_K_FULL)
        np.testing.assert_array_equal(camera.dist, DEFAULT_DIST)
        expected = diverse_poses(BoardSpec(), DEFAULT_K_FULL, DEFAULT_FULL_SIZE)
        assert len(camera.poses) == len(expected)
        for (rvec, tvec), (rvec_expected, tvec_expected) in zip(camera.poses, expected):
            np.testing.assert_array_equal(rvec, rvec_expected)
            np.testing.assert_array_equal(tvec, tvec_expected)
        assert camera.pose_index == 0
    finally:
        camera.destroy_node()


def test_stream_frames_arrive_at_the_stream_size(make_rig):
    rig = make_rig()
    rig.wait_for_frames(3)
    frames = [msg for _, msg in rig.frames]
    for msg in frames:
        assert decode(msg).shape == STREAM_SHAPE
    stamps = [Time.from_msg(msg.header.stamp).nanoseconds for msg in frames]
    assert all(later > earlier for earlier, later in zip(stamps, stamps[1:]))
    now = rig.probe.get_clock().now().nanoseconds
    assert abs(now - stamps[-1]) < 5e9, 'stream frames must carry the node clock'


def test_stream_shows_the_current_pose_and_follows_captures(make_rig):
    rig = make_rig()
    pose_0, pose_1 = (stream_view(rig.camera, index) for index in (0, 1))
    latest = decode(rig.wait_for_frames(1))
    assert mean_difference(latest, pose_0) < 2.0
    assert mean_difference(latest, pose_1) > 10.0
    rig.capture()
    wait_until(lambda: mean_difference(decode(rig.frames[-1][1]), pose_1) < 2.0,
               'the stream to show pose 1 after a capture')


def test_capture_returns_a_full_resolution_png_of_the_current_pose(make_rig):
    rig = make_rig()
    response, _ = rig.capture()
    assert response.success
    frame = decode(response.image)
    assert frame.shape == FULL_SHAPE
    assert pose_error(rig.camera, frame, 0) < 0.3
    assert rig.camera.pose_index == 1


def test_consecutive_captures_show_consecutive_poses(make_rig):
    rig = make_rig()
    first, second = (decode(rig.capture()[0].image) for _ in range(2))
    assert pose_error(rig.camera, first, 0) < 0.3
    assert pose_error(rig.camera, second, 1) < 0.3
    assert pose_error(rig.camera, second, 0) > 5.0


def test_captures_cycle_through_all_poses(make_rig):
    rig = make_rig(K_full=K_SMALL, full_size=(400, 300), stream_size=(160, 120))
    n = len(rig.camera.poses)
    frames = [decode(rig.capture()[0].image) for _ in range(n + 1)]
    assert all(frame.shape == (300, 400) for frame in frames)
    for index in range(n):
        assert mean_difference(frames[index], frames[index + 1]) > 5.0, f'capture {index + 1}'
    assert mean_difference(frames[n], frames[0]) < 4.0, 'capture n + 1 must repeat pose 0'
    assert rig.camera.pose_index == 1


def test_blank_every_makes_every_nth_capture_blank(make_rig):
    rig = make_rig(blank_every=2)
    replies = [rig.capture()[0] for _ in range(4)]
    assert all(reply.success for reply in replies)
    frames = [decode(reply.image) for reply in replies]
    assert pose_error(rig.camera, frames[0], 0) < 0.3
    for blank in (frames[1], frames[3]):
        assert blank.shape == FULL_SHAPE
        assert blank.std() < 3.0
        assert np.isinf(pose_error(rig.camera, blank, 0)), 'a blank frame shows no board'
    # The blank capture used up pose 1.
    assert pose_error(rig.camera, frames[2], 2) < 0.3


def test_reply_delay_s_delays_the_reply_but_not_the_stream(make_rig):
    rig = make_rig(rate_hz=20.0, reply_delay_s=0.6)
    rig.wait_for_frames(1)
    assert rig.client.wait_for_service(timeout_sec=TIMEOUT_S)
    requested = rig.probe.get_clock().now()
    response, (start, end) = rig.capture()
    assert response.success
    assert end - start >= 0.6
    taken = Time.from_msg(response.image.header.stamp)
    assert (taken - requested).nanoseconds >= 0.6e9, 'the frame must be taken after the delay'
    during = [arrival for arrival, _ in rig.frames if start < arrival < end]
    assert len(during) >= 4, f'only {len(during)} stream frames during a 0.6 s reply at 20 Hz'


def test_stream_also_matches_the_inference_node_subscription(make_rig):
    rig = make_rig()
    frames, incompatible = [], []
    events = SubscriptionEventCallbacks(
        incompatible_qos=lambda event: incompatible.append(event.last_policy_kind.name))
    rig.probe.create_subscription(CompressedImage, 'image', frames.append, INFERENCE_QOS,
                                  event_callbacks=events)
    wait_until(lambda: frames or incompatible, 'a frame on the inference QoS')
    assert not incompatible, f'incompatible QoS: {incompatible}'


@pytest.mark.parametrize('camera_args, match', [
    ({'rate_hz': 0.0}, 'rate_hz'),
    ({'blank_every': -1}, 'blank_every'),
    ({'reply_delay_s': -0.5}, 'reply_delay_s'),
    ({'board_spec': BoardSpec(squares=(1, 5))}, 'at least 2x2'),
])
def test_bad_arguments_are_rejected(ros, camera_args, match):
    with pytest.raises(ValueError, match=match):
        FakeCamera(**camera_args)


def test_main_takes_its_parameters_and_shuts_down_cleanly(ros, tmp_path):
    spec = BoardSpec(squares=(7, 5), square_length=0.03, marker_length=0.022)
    config = tmp_path / 'cfg.yaml'
    config.write_text(yaml.safe_dump({'CALIBRATION': {'board': {
        'squares': [7, 5], 'square_length_m': 0.03, 'marker_length_m': 0.022}}}))
    namespace = NAMESPACE + '_main'
    package_root = str(Path(convchart_ros.__file__).resolve().parents[1])
    env = dict(os.environ, ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST',
               ROS_DOMAIN_ID=str(DOMAIN_ID), PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=os.pathsep.join([package_root, os.environ.get('PYTHONPATH', '')]))
    log_path = tmp_path / 'fake_camera.log'
    with open(log_path, 'w') as log:
        process = subprocess.Popen(
            [sys.executable, '-m', 'convchart_ros.calib_utils.fake_camera', '--ros-args',
             '-r', f'__ns:={namespace}', '-p', f'config:={config}', '-p', 'rate_hz:=20',
             '-p', 'blank_every:=2', '-p', 'reply_delay_s:=0.3'],
            env=env, stdout=log, stderr=subprocess.STDOUT)
    rig = Rig(prefix=namespace + '/')
    try:
        rig.wait_for_frames(5)
        first = len(rig.frames)
        rig.wait_for_frames(first + 20)
        arrivals = [arrival for arrival, _ in rig.frames[first - 1:]]
        rate = (len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
        assert 14.0 < rate < 30.0, f'stream at {rate:.1f} Hz, rate_hz is 20'

        board_reply, (start, end) = rig.capture()
        assert end - start >= 0.3
        pose = diverse_poses(spec, DEFAULT_K_FULL, DEFAULT_FULL_SIZE)[0]
        frame = decode(board_reply.image)
        assert corner_error(frame, spec, pose) < 0.3, 'not the configured board'
        blank = decode(rig.capture()[0].image)
        assert blank.std() < 3.0, 'blank_every:=2 must blank the second capture'

        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=TIMEOUT_S) == 0, log_path.read_text()
        assert 'Traceback' not in log_path.read_text()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        rig.close()
