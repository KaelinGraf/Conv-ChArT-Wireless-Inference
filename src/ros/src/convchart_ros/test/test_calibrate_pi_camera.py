"""
Calibration against the Pi's own camera node (p4p_camera, mock backend) rather than the fake camera.

The camera node replays one rendered 1600x1200 board through its real image_full_res service
(GetFullResImage, max_age included), so a capture from CalibrateCam crosses the same interface as on
the robot: it must come back, be decoded and be kept as a view, and a frame older than
capture_max_age_s must be refused rather than calibrated on. Isolated domain, discovery limited to
localhost, private namespace. p4p_camera lives in holonomic_mpc_ekf (p4p_nodes/), so this runs with that
workspace sourced and is skipped without it.
"""

import importlib.util
import os
import threading
import time

import cv2
import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
import yaml

from convchart_ros.calib_utils.calib_board import BoardSpec, make_board
from convchart_ros.calib_utils.calib_synth import (
    DEFAULT_DIST, DEFAULT_FULL_SIZE, DEFAULT_K_FULL, diverse_poses, render_view)
from convchart_ros.calibrate import CalibrateCam

pytestmark = pytest.mark.skipif(importlib.util.find_spec('p4p_camera') is None,
                                reason="needs p4p_camera (holonomic_mpc_ekf's workspace, sourced)")

DOMAIN_ID = 1 + os.getpid() % 20
NAMESPACE = '/calibrate_pi_test'
TIMEOUT_S = 20.0


def wait_until(predicate, what, timeout_s=TIMEOUT_S):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail(f'timed out waiting for {what}')
        time.sleep(0.02)


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
def pi_rig(ros, tmp_path):
    """The Pi's camera node replaying a rendered board, a CalibrateCam, one executor for both."""
    from p4p_camera import camera_node
    spec = BoardSpec()
    rvec, tvec = diverse_poses(spec, DEFAULT_K_FULL, DEFAULT_FULL_SIZE)[0]
    frame, _, _ = render_view(make_board(spec), spec, DEFAULT_K_FULL, DEFAULT_DIST, rvec, tvec,
                              DEFAULT_FULL_SIZE, seed=0)
    board_png = tmp_path / 'board.png'
    cv2.imwrite(str(board_png), frame)
    config = tmp_path / 'cfg' / 'cfg.yaml'
    config.parent.mkdir()
    config.write_text(yaml.safe_dump({'CALIBRATION': {'review_s': 0.05}}))
    camera = camera_node.CameraNode(parameter_overrides=[
        Parameter('camera_backend', value='mock'), Parameter('mock_image', value=str(board_png)),
        Parameter('frame_rate', value=10.0), Parameter('status_period', value=1.0)])
    made = []

    def make(**node_args):
        node = CalibrateCam(cfg_pth=str(config), output_dir=str(tmp_path / 'out'),
                            **{'detector': 'opencv', **node_args})
        executor = MultiThreadedExecutor(num_threads=6)
        executor.add_node(camera)
        executor.add_node(node)
        spinner = threading.Thread(target=executor.spin, daemon=True)
        spinner.start()
        made.append((node, executor, spinner))
        wait_until(lambda: node.state().frame is not None, 'the camera stream to reach calibrate')
        wait_until(node._capture_client.service_is_ready, 'image_full_res on the graph')
        return node

    yield make
    for node, executor, spinner in made:
        node.close()
        executor.shutdown()
        spinner.join(timeout=TIMEOUT_S)
        node.destroy_node()
    camera.close()
    camera.destroy_node()


def test_a_capture_from_the_pi_camera_node_is_kept_as_a_full_resolution_view(pi_rig):
    node = pi_rig()
    assert node.request_capture()
    wait_until(lambda: node.state().hud.kept + node.state().hud.rejected == 1, 'the capture verdict')
    hud = node.state().hud
    assert hud.kept == 1, f'the capture was not kept: {hud.message}'
    view = cv2.imread(str(node.session_dir / 'frames' / 'view_000.png'), cv2.IMREAD_UNCHANGED)
    assert view.shape == (DEFAULT_FULL_SIZE[1], DEFAULT_FULL_SIZE[0])


def test_a_frame_older_than_capture_max_age_s_is_refused(pi_rig):
    node = pi_rig(capture_max_age_s=1e-6)
    assert node.request_capture()
    wait_until(lambda: node.state().hud.message is not None and 'old' in node.state().hud.message,
               'the refusal on the HUD')
    hud = node.state().hud
    assert hud.kept == 0 and hud.rejected == 0
    assert 'capture failed' in hud.message and 'max_age' in hud.message
