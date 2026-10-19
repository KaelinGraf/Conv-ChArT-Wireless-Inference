"""
End-to-end tests of the calibration node (calibrate.py) with the fake camera over ROS 2, headless.

FakeCamera and CalibrateCam run in one process, each on a MultiThreadedExecutor of its own, on an
isolated domain with discovery limited to localhost and a namespace private to the process, so
nothing here can reach a real robot or another test run; the node also starts as main() starts it,
from ROS parameters, in a subprocess under the same isolation. Captures go through
CalibrateCam.request_capture(), as the GUI's SPACE key does. run_gui runs against stand-ins for
OpenCV's window functions, so no window ever opens. The calibration must recover the fake camera's
ground truth (K_full, dist). These tests run the node with OpenCV's corner detector (custom boards,
configs without a model); test_calib_model.py covers the model's, with the GPU and the checkpoints.
"""

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import convchart_ros
from convchart_ros import calibrate
from convchart_ros.calib_utils.calib_board import board_spec_from_cfg
from convchart_ros.calib_utils.calib_session import scale_intrinsics
from convchart_ros.calib_utils.fake_camera import FakeCamera
from convchart_ros.calibrate import CalibrateCam, GuiState, run_gui
from convchart_ros.launch_plan import find_repo
import cv2
import numpy as np
import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
import yaml

DOMAIN_ID = 21 + os.getpid() % 20
NAMESPACE = f'/calibrate_test_{os.getpid()}'
TIMEOUT_S = 60.0
FULL_SHAPE = (1200, 1600)
STREAM_SIZE = (640, 480)
LOG_NAME = 'calibration_log.yaml'
BACKUP_NAME = 'cfg.yaml.bak'
MIN_VIEWS = 12
CUSTOM_BOARD = {'squares': [7, 5], 'square_length_m': 0.03, 'marker_length_m': 0.022,
                'dictionary': 'DICT_4X4_50', 'first_marker_id': 10}

CFG_TEXT = """\
MODEL:
  detector: "checkpoints/detector_882k.onnx"
  refiner: "checkpoints/refiner.onnx"

PIPELINE:
  tau_hm: 0.3
  id_readout: coarse

BOARD:
  squares: [5, 5]
  square_length_m: null

# written by the calibration GUI
CAMERA:
  K: null
  dist: null

"""

LOG_KEYS = {'session', 'board', 'images', 'counts', 'coverage_fraction', 'solution', 'views',
            'history'}
SESSION_KEYS = {'started', 'config', 'auto_replace', 'config_written', 'config_backup',
                'detector', 'resumed'}
BOARD_KEYS = {'squares', 'square_length_m', 'marker_length_m', 'dictionary', 'first_marker_id',
              'legacy_pattern'}
SOLUTION_KEYS = {'rms_px', 'converged', 'K_full', 'dist', 'std', 'K_stream', 'stream_size'}
STD_KEYS = {'fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3'}
VIEW_KEYS = {'index', 'file', 'n_corners', 'error_px', 'dropped'}
HISTORY_KEYS = {'n_views', 'rms_px', 'fx', 'fy', 'cx', 'cy'}

WINDOW_FUNCTIONS = ['namedWindow', 'imshow', 'waitKeyEx', 'waitKey', 'getWindowProperty',
                    'destroyWindow', 'destroyAllWindows', 'pollKey', 'setWindowProperty',
                    'resizeWindow', 'moveWindow', 'setWindowTitle', 'startWindowThread',
                    'selectROI', 'createTrackbar', 'setMouseCallback']


def wait_until(condition, what, timeout=TIMEOUT_S):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            pytest.fail(f'timed out after {timeout:.0f} s waiting for {what}')
        time.sleep(0.005)


def counts(state):
    return state.hud.kept, state.hud.rejected


def idle(node):
    state = node.state()
    return not state.capturing and not state.solving


def settled(node):
    """Check the node is idle and back in live mode: no capture, no solve, no review showing.

    Args:
        node: CalibrateCam under test.
    """
    state = node.state()
    return not state.capturing and not state.solving and state.review is None


def wait_for_stream(node):
    wait_until(lambda: node.state().frame is not None, 'the first stream frame')


def capture(node):
    """Capture as the SPACE key does and wait until the node has handled the reply.

    Args:
        node: CalibrateCam under test.
    """
    before = counts(node.state())
    wait_until(node.request_capture, 'the node to take a capture request')
    wait_until(lambda: not node.state().capturing, 'the capture reply to be handled')
    state = node.state()
    assert counts(state) != before, f'capture not recorded: {state.hud.message}'
    return state


def write_config(path, calibration):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CFG_TEXT + yaml.safe_dump({'CALIBRATION': calibration}, sort_keys=False))
    return path


def read_log(session_dir):
    return yaml.safe_load((session_dir / LOG_NAME).read_text())


def ground_truth_corners(camera, pose_index):
    """Project the camera's chessboard corners for one pose with its true K and dist.

    Args:
        camera: FakeCamera holding the poses and the true intrinsics.
        pose_index: Index into camera.poses.
    """
    rvec, tvec = camera.poses[pose_index]
    points = np.asarray(camera.board.getChessboardCorners(), dtype=np.float64)
    return cv2.projectPoints(points, rvec, tvec, camera.K_full, camera.dist)[0].reshape(-1, 2)


def distortion_error_px(camera, dist):
    """Return the largest image displacement (px) between dist and the camera's true distortion.

    Args:
        camera: FakeCamera holding the poses and the true intrinsics.
        dist: Estimated distortion coefficients.
    """
    points = np.asarray(camera.board.getChessboardCorners(), dtype=np.float64)
    worst = 0.0
    for rvec, tvec in camera.poses:
        in_camera = points @ cv2.Rodrigues(rvec)[0].T + tvec
        estimated, truth = (cv2.projectPoints(in_camera, np.zeros(3), np.zeros(3), camera.K_full,
                                              np.asarray(d, dtype=np.float64))[0].reshape(-1, 2)
                            for d in (dist, camera.dist))
        worst = max(worst, float(np.linalg.norm(estimated - truth, axis=1).max()))
    return worst


def spin_in_thread(node, num_threads):
    """Spin node in a MultiThreadedExecutor on a background thread; return a function to stop it.

    The stop function ends the loop and waits for running callbacks before shutting down.

    Args:
        node: Node to spin.
        num_threads: Executor thread count.
    """
    executor = MultiThreadedExecutor(num_threads=num_threads)
    executor.add_node(node)
    stopping = threading.Event()

    def loop():
        while not stopping.is_set() and executor.context.ok():
            executor.spin_once(timeout_sec=0.05)

    spinner = threading.Thread(target=loop, daemon=True)
    spinner.start()

    def stop():
        stopping.set()
        spinner.join(timeout=TIMEOUT_S)
        assert not spinner.is_alive(), 'the spin loop did not stop'
        executor._executor.shutdown(wait=True)
        executor.shutdown()

    return stop


class Rig:
    """CalibrateCam and FakeCamera, each spun in a MultiThreadedExecutor on a background thread.

    The camera has an executor of its own so it can be replaced while the node keeps spinning.

    Args:
        camera: FakeCamera to spin.
        node: CalibrateCam to spin.
        config: Path of the config the node uses.
    """

    def __init__(self, camera, node, config):
        self.camera = camera
        self.node = node
        self.config = config
        self._stop_node = spin_in_thread(node, num_threads=6)
        self._stop_camera = spin_in_thread(camera, num_threads=3)

    def replace_camera(self, make_camera):
        """Stop the camera, destroy it and start the one make_camera() makes in its place.

        Args:
            make_camera: Callable returning the new FakeCamera.
        """
        self._stop_camera()
        self.camera.destroy_node()
        self.camera = make_camera()
        self._stop_camera = spin_in_thread(self.camera, num_threads=3)

    def close(self):
        self.node.close()
        self._stop_node()
        self._stop_camera()
        self.node.destroy_node()
        self.camera.destroy_node()


def start_rig(tmp_path, calibration, camera_args=None, **node_args):
    """Write a config with this CALIBRATION block and start a camera with its board and a node.

    The node detects with OpenCV unless node_args say otherwise.

    Args:
        tmp_path: Folder for the config and the session output.
        calibration: CALIBRATION block of the config.
        camera_args: Extra FakeCamera arguments, or None.
        **node_args: Extra CalibrateCam arguments.
    """
    config = write_config(tmp_path / 'cfg' / 'cfg.yaml', calibration)
    camera = FakeCamera(board_spec=board_spec_from_cfg(calibration.get('board')),
                        **(camera_args or {}))
    try:
        node = CalibrateCam(cfg_pth=str(config), output_dir=str(tmp_path / 'out'),
                            **{'detector': 'opencv', **node_args})
    except BaseException:
        camera.destroy_node()
        raise
    return Rig(camera, node, config)


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
def make_rig(ros, tmp_path):
    """Start one rig for this test (nodes share names, so never two at once) and close it after."""
    rigs = []

    def make(calibration=None, camera_args=None, **node_args):
        assert not rigs, 'one rig per test'
        rigs.append(start_rig(tmp_path, {'review_s': 0.05, **(calibration or {})}, camera_args,
                              **node_args))
        return rigs[-1]

    yield make
    for rig in rigs:
        rig.close()


@dataclass
class Run:
    """Complete calibration session, captured and closed, with what it left behind.

    Args:
        camera: FakeCamera the session captured from.
        config: Path of the config.
        original: Config bytes before the session.
        session_dir: Session folder.
        auto_replace: The node's auto_replace setting.
        board: CALIBRATION.board block ({} for the default board).
        first_solve: Config, log and backup stat right after the first solve.
        state: GUI state after the last solve and review.
        last_solve: Log after the last solve.
        log_text: Final calibration_log.yaml text.
    """

    camera: FakeCamera
    config: Path
    original: bytes
    session_dir: Path
    auto_replace: bool
    board: dict
    first_solve: dict
    state: GuiState
    last_solve: dict
    log_text: str

    @property
    def log(self):
        return yaml.safe_load(self.log_text)


def full_session(tmp_path, auto_replace, board, blank_every, n_captures):
    """Run a whole calibration session and return it as a Run.

    Args:
        tmp_path: Folder for the config and the session output.
        auto_replace: The node's auto_replace setting.
        board: CALIBRATION.board block, or None for the default board.
        blank_every: Every Nth capture is blank; 0 means never.
        n_captures: Number of captures taken.
    """
    calibration = {'board': board, 'review_s': 0.05} if board else {'review_s': 0.05}
    rig = start_rig(tmp_path, calibration, {'blank_every': blank_every},
                    auto_replace=auto_replace)
    node, config = rig.node, rig.config
    try:
        original = config.read_bytes()
        wait_for_stream(node)
        first_solve = None
        for _ in range(n_captures):
            capture(node)
            if first_solve is None and node.state().hud.kept == MIN_VIEWS:
                wait_until(lambda: idle(node), 'the first solve')
                backup = node.session_dir / BACKUP_NAME
                first_solve = {'config': config.read_bytes(), 'log': read_log(node.session_dir),
                               'backup': backup.stat() if backup.exists() else None}
        wait_until(lambda: settled(node), 'the last solve and review')
        state = node.state()
        last_solve = read_log(node.session_dir)
        node.close()
        return Run(rig.camera, config, original, node.session_dir, auto_replace, board or {},
                   first_solve, state, last_solve, (node.session_dir / LOG_NAME).read_text())
    finally:
        rig.close()


@pytest.fixture(scope='module')
def replace_run(ros, tmp_path_factory):
    """Run a session with auto_replace on the default board, every 5th capture blank."""
    return full_session(tmp_path_factory.mktemp('replace'), auto_replace=True, board=None,
                        blank_every=5, n_captures=20)


@pytest.fixture(scope='module')
def keep_run(ros, tmp_path_factory):
    """Run a session without auto_replace on a custom 7x5 DICT_4X4_50 board, ids from 10."""
    return full_session(tmp_path_factory.mktemp('keep'), auto_replace=False, board=CUSTOM_BOARD,
                        blank_every=0, n_captures=20)


@pytest.fixture(params=['replace_run', 'keep_run'])
def run(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize('cfg_pth, calibration, node_args, error, match', [
    ('', None, {}, ValueError, 'configuration path is not set'),
    ('missing.yaml', None, {}, FileNotFoundError, 'missing.yaml'),
    ('cfg.yaml', {'min_view': 3}, {}, ValueError, 'unknown CALIBRATION keys'),
    ('cfg.yaml', {'board': {'squares': [1, 5]}}, {}, ValueError, 'at least 2x2'),
    ('cfg.yaml', None, {'capture_timeout_s': 0.0}, ValueError, 'capture_timeout_s'),
    ('cfg.yaml', None, {'detector': 'sift'}, ValueError, 'detector must be one of model, opencv'),
    ('cfg.yaml', {'board': CUSTOM_BOARD}, {'detector': 'model'}, ValueError,
     'use detector:=opencv'),
])
def test_bad_configuration_is_rejected(ros, tmp_path, cfg_pth, calibration, node_args, error,
                                       match):
    write_config(tmp_path / 'cfg.yaml', calibration or {})
    with pytest.raises(error, match=match):
        CalibrateCam(cfg_pth=str(tmp_path / cfg_pth) if cfg_pth else '',
                     output_dir=str(tmp_path / 'out'), **node_args)
    assert not (tmp_path / 'out').exists(), 'no session folder for a node that did not start'


def test_ros_parameters_configure_the_node_as_main_does(ros, tmp_path):
    config = write_config(tmp_path / 'cfg' / 'cfg.yaml', {'board': CUSTOM_BOARD})
    out = tmp_path / 'out'
    code = ('import sys\n'
            'import rclpy\n'
            'from convchart_ros.calibrate import CalibrateCam\n'
            'rclpy.init(args=sys.argv)\n'
            'node = CalibrateCam()\n'
            'node.close()\n'
            'node.destroy_node()\n'
            'rclpy.shutdown()\n')
    package_root = str(Path(convchart_ros.__file__).resolve().parents[1])
    env = dict(os.environ, ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST',
               ROS_DOMAIN_ID=str(DOMAIN_ID), PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=os.pathsep.join([package_root, os.environ.get('PYTHONPATH', '')]))
    result = subprocess.run(
        [sys.executable, '-c', code, '--ros-args', '-r', f'__ns:={NAMESPACE}_params',
         '-p', f'config:={config}', '-p', f'output_dir:={out}', '-p', 'auto_replace:=true',
         '-p', 'capture_timeout_s:=3',
         '-p', 'detector:=opencv'],
        env=env, capture_output=True, text=True, timeout=TIMEOUT_S)
    assert result.returncode == 0, result.stderr
    [session_dir] = out.iterdir()
    log = read_log(session_dir)
    assert log['session']['config'] == str(config)
    assert log['session']['auto_replace'] is True
    assert log['board']['squares'] == CUSTOM_BOARD['squares'], 'the board from that config'


def repo_model():
    """Return the repo's MODEL block, paths absolute; None without onnxruntime or checkpoints."""
    repo = find_repo(Path(__file__))
    if repo is None or importlib.util.find_spec('onnxruntime') is None:
        return None
    model = (yaml.safe_load((repo / 'cfg' / 'cfg.yaml').read_text()) or {}).get('MODEL') or {}
    files = {key: repo / str(model.get(key, '')) for key in ('detector', 'refiner')}
    return ({key: str(path) for key, path in files.items()}
            if all(path.is_file() for path in files.values()) else None)


def test_the_model_reads_the_boards_distance_and_calibrates(ros, tmp_path):
    model = repo_model()
    if model is None:
        pytest.skip("needs onnxruntime and the repo's MODEL checkpoints")
    camera = FakeCamera()
    K_stream = scale_intrinsics(camera.K_full, camera.full_size, STREAM_SIZE)
    config = tmp_path / 'cfg' / 'cfg.yaml'
    config.parent.mkdir()
    config.write_text(yaml.safe_dump({
        'MODEL': model,
        'BOARD': {'squares': [5, 5], 'square_length_m': 1.0},
        'CAMERA': {'K': K_stream.tolist(), 'dist': camera.dist.tolist(),
                   'image_size': list(STREAM_SIZE)},
        'CALIBRATION': {'review_s': 0.05}}))
    try:
        node = CalibrateCam(cfg_pth=str(config), output_dir=str(tmp_path / 'out'))
    except BaseException:
        camera.destroy_node()
        raise
    rig = Rig(camera, node, config)
    try:
        wait_for_stream(node)
        rvec, tvec = camera.poses[camera.pose_index]
        centre = cv2.Rodrigues(rvec)[0] @ np.array([2.5, 2.5, 0.0]) + tvec
        wait_until(lambda: isinstance(node.state().hud.distance, tuple), 'a distance reading')
        z, straight = node.state().hud.distance
        assert z == pytest.approx(centre[2], rel=0.02)
        assert straight == pytest.approx(np.linalg.norm(centre), rel=0.02)
        assert 'D distance' in node.state().hud.help
        for _ in range(MIN_VIEWS):
            capture(node)
        wait_until(lambda: settled(node) and node.state().solution is not None, 'the solve')
        state = node.state()
        assert (state.hud.kept, state.hud.rejected) == (MIN_VIEWS, 0)
        K, K_true = state.solution.K, camera.K_full
        assert state.solution.rms < 0.3
        np.testing.assert_allclose([K[0, 0], K[1, 1]], [K_true[0, 0], K_true[1, 1]], atol=3.0)
        np.testing.assert_allclose([K[0, 2], K[1, 2]], [K_true[0, 2], K_true[1, 2]], atol=4.0)
    finally:
        rig.close()


def test_opencv_shows_no_distance_readout(make_rig):
    rig = make_rig()
    wait_for_stream(rig.node)
    state = rig.node.state()
    assert state.hud.distance is None and 'distance' not in state.hud.help


def test_resume_carries_on_a_session_where_it_was_left(ros, tmp_path):
    calibration = {'review_s': 0.05, 'outlier_factor': 100.0}
    rig = start_rig(tmp_path, calibration, {'blank_every': 6})
    try:
        wait_for_stream(rig.node)
        for _ in range(15):
            capture(rig.node)
        wait_until(lambda: settled(rig.node), 'the solves')
        assert rig.node.undo()
        wait_until(lambda: settled(rig.node), 'the solve after undo')
        first = rig.node.state()
        folder = rig.node.session_dir
        rig.node.close()
    finally:
        rig.close()
    assert (first.hud.kept, first.hud.rejected) == (12, 2)
    first_log = read_log(folder)
    frames = folder / 'frames'
    assert (frames / 'undone_012.png').exists() and not (frames / 'view_012.png').exists()

    rig = start_rig(tmp_path, calibration, resume='latest')
    try:
        node = rig.node
        assert node.session_dir == folder, 'no new session folder'
        wait_until(lambda: settled(node) and node.state().solution is not None, 'the first solve')
        state = node.state()
        assert (state.hud.kept, state.hud.rejected) == (12, 2)
        assert f'resumed {folder.name}: 12 views' in state.hud.status
        solution, earlier = state.solution, first_log['solution']
        np.testing.assert_allclose(solution.K, earlier['K_full'], rtol=1e-4)
        assert solution.rms == pytest.approx(earlier['rms_px'], rel=1e-4)
        assert state.hud.coverage_fraction == pytest.approx(first_log['coverage_fraction'])
        wait_for_stream(node)
        capture(node)
        assert (frames / 'view_013.png').exists()
        wait_until(lambda: settled(node), 'the solve with the new view')
        assert node.undo(), 'the new view'
        wait_until(lambda: settled(node), 'the solve after undo')
        assert node.undo(), 'a restored view'
        assert (frames / 'undone_013.png').exists() and (frames / 'undone_011.png').exists()
        wait_until(lambda: settled(node), 'the last review')
        node.close()
    finally:
        rig.close()
    log = read_log(folder)
    assert [view['index'] for view in log['views']] == list(range(11))
    assert log['counts']['kept'] == 11 and log['counts']['rejected'] == 2
    assert log['session']['started'] == first_log['session']['started']
    [resumed] = log['session']['resumed']
    assert resumed['detector'] == 'opencv'
    assert (resumed['views_restored'], resumed['views_now_rejected']) == (12, 0)
    assert log['history'][:len(first_log['history'])] == first_log['history']
    assert len(log['history']) == len(first_log['history']) + 3


def test_resume_refuses_a_session_of_another_board_or_no_session(ros, tmp_path):
    folder = tmp_path / 'out' / '20261013-225444'
    (folder / 'frames').mkdir(parents=True)
    (folder / 'calibration_log.yaml').write_text(yaml.safe_dump({'board': {
        'squares': [7, 5], 'square_length_m': 0.03, 'marker_length_m': 0.022,
        'dictionary': 'DICT_4X4_50', 'first_marker_id': 10, 'legacy_pattern': False}}))
    config = write_config(tmp_path / 'cfg' / 'cfg.yaml', {})
    cases = [(str(folder), 'was taken with a 7x5 DICT_4X4_50 board'),
             (str(tmp_path), 'is not a calibration session folder'),
             ('latest', 'no session folder in')]
    for resume, match in cases:
        with pytest.raises(ValueError, match=match):
            CalibrateCam(cfg_pth=str(config), output_dir=str(tmp_path / 'out'),
                         detector='opencv', resume=resume)
    assert sorted(p.name for p in (tmp_path / 'out').iterdir()) == [folder.name], \
        'no new session folder'


def test_calibration_recovers_the_fake_camera(run):
    solution = run.state.solution
    assert solution is not None
    assert solution.image_size == (1600, 1200)
    assert MIN_VIEWS <= solution.n_views <= run.state.hud.kept
    assert solution.rms < 0.3
    K, K_true = solution.K, run.camera.K_full
    np.testing.assert_allclose([K[0, 0], K[1, 1]], [K_true[0, 0], K_true[1, 1]], atol=3.0)
    np.testing.assert_allclose([K[0, 2], K[1, 2]], [K_true[0, 2], K_true[1, 2]], atol=4.0)
    assert K[0, 1] == 0.0
    dist_error = np.abs(solution.dist[:4] - run.camera.dist[:4])
    assert np.all(dist_error < [0.01, 0.05, 0.001, 0.001]), f'k1 k2 p1 p2 off by {dist_error}'
    assert distortion_error_px(run.camera, solution.dist) < 1.0


def test_hud_reports_the_solution(run):
    state = run.state
    solution = state.solution
    assert state.review is None and not state.capturing and not state.solving
    assert state.canvas_size == STREAM_SIZE
    assert state.frame.shape == (STREAM_SIZE[1], STREAM_SIZE[0])
    assert state.hud.min_views == MIN_VIEWS
    status = f'rms {solution.rms:.2f} px | fx {solution.K[0, 0]:.1f} +- {solution.std["fx"]:.1f}'
    assert state.hud.status.startswith(status)
    assert ' | cy ' in state.hud.status
    assert ('converged' in state.hud.status) == solution.converged == state.hud.converged
    assert 0.5 < state.hud.coverage_fraction <= 1.0
    assert state.coverage.shape == (6, 8)


def test_log_has_the_spec_keys_and_values(run):
    log, state = run.log, run.state
    assert '&id' not in run.log_text and '*id' not in run.log_text, 'no YAML aliases in the log'
    assert set(log) == LOG_KEYS
    assert set(log['session']) == SESSION_KEYS
    assert log['session']['config'] == str(run.config)
    assert log['session']['auto_replace'] is run.auto_replace
    assert log['session']['detector'] == 'opencv' and log['session']['resumed'] == []
    assert set(log['board']) == BOARD_KEYS
    spec = board_spec_from_cfg(run.board)
    assert log['board']['squares'] == list(spec.squares)
    assert log['board']['dictionary'] == spec.dictionary
    assert log['board']['first_marker_id'] == spec.first_marker_id
    assert log['board']['marker_length_m'] == pytest.approx(spec.marker_side)
    assert log['images'] == {'full_size': [1600, 1200], 'stream_size': list(STREAM_SIZE)}
    assert log['counts'] == {'kept': state.hud.kept, 'rejected': state.hud.rejected,
                             'dropped': sum(view['dropped'] for view in log['views']),
                             'active': state.solution.n_views}
    assert log['coverage_fraction'] == pytest.approx(state.hud.coverage_fraction)

    solution = log['solution']
    assert set(solution) == SOLUTION_KEYS
    assert set(solution['std']) == STD_KEYS
    np.testing.assert_allclose(solution['K_full'], state.solution.K, rtol=1e-12)
    np.testing.assert_allclose(solution['dist'], state.solution.dist, rtol=1e-12)
    np.testing.assert_allclose(solution['K_stream'],
                               scale_intrinsics(state.solution.K, (1600, 1200), STREAM_SIZE),
                               rtol=1e-12)
    assert solution['stream_size'] == list(STREAM_SIZE)
    assert solution['rms_px'] == pytest.approx(state.solution.rms)

    assert len(log['views']) == state.hud.kept
    for view in log['views']:
        assert set(view) == VIEW_KEYS
        assert view['file'] == f'frames/view_{view["index"]:03d}.png'
        assert 0 < view['n_corners'] <= spec.n_inner_corners
    assert len(log['history']) >= 2
    for entry in log['history']:
        assert set(entry) == HISTORY_KEYS


def test_log_is_written_after_every_solve(run):
    first = run.first_solve['log']
    assert first['solution'] is not None, 'a log must exist after the first solve, before exit'
    assert first['counts']['kept'] == MIN_VIEWS
    last = run.last_solve
    np.testing.assert_allclose(last['solution']['K_full'], run.state.solution.K, rtol=1e-12)
    assert last['counts']['kept'] == run.state.hud.kept
    assert len(last['history']) == len(run.log['history']) > len(first['history'])


def test_frames_are_saved_for_every_capture(run):
    frames = run.session_dir / 'frames'
    kept, rejected = counts(run.state)
    expected = [f'rejected_{i:03d}.png' for i in range(rejected)]
    expected += [f'view_{i:03d}.png' for i in range(kept)]
    assert sorted(path.name for path in frames.iterdir()) == expected
    for name in (expected[0], expected[-1]):
        frame = cv2.imread(str(frames / name), cv2.IMREAD_UNCHANGED)
        assert frame.shape == FULL_SHAPE and frame.dtype == np.uint8


def test_auto_replace_writes_camera_at_stream_resolution(replace_run):
    run = replace_run
    cfg = yaml.safe_load(run.config.read_text())
    original = yaml.safe_load(run.original)
    assert list(cfg) == list(original), 'top-level keys and their order are kept'
    for key in original:
        if key != 'CAMERA':
            assert cfg[key] == original[key]
    solution = run.state.solution
    camera = cfg['CAMERA']
    assert list(camera) == ['K', 'dist', 'image_size']
    np.testing.assert_allclose(camera['K'],
                               scale_intrinsics(solution.K, solution.image_size, STREAM_SIZE),
                               rtol=1e-12)
    np.testing.assert_allclose(camera['dist'], solution.dist, rtol=1e-12)
    assert camera['image_size'] == list(STREAM_SIZE)
    np.testing.assert_allclose(camera['K'], run.log['solution']['K_stream'], rtol=1e-12)


def test_auto_replace_backs_up_the_config_exactly_once(replace_run):
    run = replace_run
    backup = run.session_dir / BACKUP_NAME
    assert backup.read_bytes() == run.original
    first = run.first_solve
    assert first['config'] != run.original, 'the first solve writes CAMERA'
    assert run.config.read_bytes() != first['config'], 'later solves write CAMERA again'
    stat = backup.stat()
    assert (stat.st_ino, stat.st_mtime_ns) == (first['backup'].st_ino, first['backup'].st_mtime_ns)
    assert sorted(path.name for path in run.session_dir.iterdir()) == \
        sorted([BACKUP_NAME, LOG_NAME, 'frames'])
    assert [path.name for path in run.config.parent.iterdir()] == ['cfg.yaml']
    assert run.log['session']['config_written'] is True
    assert run.log['session']['config_backup'] == str(backup)


def test_without_auto_replace_the_config_stays_byte_identical(keep_run):
    run = keep_run
    assert run.config.read_bytes() == run.original
    assert run.first_solve['config'] == run.original
    assert not (run.session_dir / BACKUP_NAME).exists()
    assert run.log['session']['config_written'] is False
    assert run.log['session']['config_backup'] is None


def test_auto_replace_waits_for_a_stream_frame(make_rig):
    rig = make_rig({'min_views': 3}, {'rate_hz': 0.001}, auto_replace=True)
    node = rig.node
    original = rig.config.read_bytes()
    for _ in range(3):
        capture(node)
    wait_until(lambda: idle(node), 'the solve')
    state = node.state()
    assert state.solution is not None
    assert state.frame is None and state.canvas_size == (640, 480)
    assert rig.config.read_bytes() == original
    assert not (node.session_dir / BACKUP_NAME).exists()
    log = read_log(node.session_dir)
    assert log['session']['config_written'] is False
    assert log['session']['config_backup'] is None
    assert log['images']['stream_size'] is None
    assert log['solution']['K_stream'] is None and log['solution']['stream_size'] is None


def test_review_shows_the_verdict_and_holds_off_captures(make_rig):
    rig = make_rig({'review_s': 0.5}, {'blank_every': 2})
    node = rig.node
    wait_for_stream(node)
    wait_until(node.request_capture, 'the node to take a capture request')
    assert node.state().capturing
    assert not node.request_capture(), 'no second request while one is in flight'
    wait_until(lambda: node.state().review is not None, 'the review')
    shown = time.monotonic()
    state = node.state()
    assert not state.capturing
    assert counts(state) == (1, 0)
    assert state.hud.verdict == 'KEPT' and state.hud.verdict_ok is True
    review = state.review
    assert review.kept and review.verdict == 'KEPT'
    assert review.frame.shape == FULL_SHAPE
    truth = ground_truth_corners(rig.camera, 0)
    assert review.corners.shape == (16, 2)
    nearest = np.linalg.norm(review.corners[:, None] - truth[None], axis=2).min(axis=1)
    assert np.median(nearest) < 0.3, 'corners in full-res pixels, pixel-centre convention'
    assert not node.request_capture(), 'SPACE is ignored while a review is showing'
    assert not node.state().capturing
    wait_until(lambda: node.state().review is None, 'the end of the review')
    assert time.monotonic() - shown > 0.4

    state = capture(node)
    assert counts(state) == (1, 1)
    assert state.review is not None and not state.review.kept
    assert state.hud.verdict == 'REJECTED: no board found'
    assert state.hud.verdict_ok is False
    assert len(state.review.corners) == 0
    assert (node.session_dir / 'frames' / 'rejected_000.png').exists()


def test_capture_timeout_is_reported_and_the_node_keeps_working(make_rig):
    rig = make_rig({'review_s': 0.5}, {'reply_delay_s': 2.0}, capture_timeout_s=1.0)
    node = rig.node
    wait_for_stream(node)
    wait_until(node.request_capture, 'the node to take a capture request')
    start = time.monotonic()
    wait_until(lambda: not node.state().capturing, 'the request to time out', timeout=5.0)
    assert 0.95 < time.monotonic() - start < 1.5, 'cancelled at capture_timeout_s'
    state = node.state()
    assert 'timed out' in state.hud.message
    assert counts(state) == (0, 0) and state.review is None
    wait_until(lambda: rig.camera.pose_index == 1, 'the late reply')
    time.sleep(0.3)
    state = node.state()
    assert counts(state) == (0, 0) and not state.capturing and state.review is None
    assert list((node.session_dir / 'frames').iterdir()) == []

    rig.replace_camera(FakeCamera)
    state = capture(node)
    assert counts(state) == (1, 0)
    assert state.hud.verdict == 'KEPT'
    assert 'timed out' not in (state.hud.message or ''), 'a new request clears the old message'
    assert (node.session_dir / 'frames' / 'view_000.png').exists()


def test_a_capture_during_a_solve_queues_exactly_one_more_solve(make_rig, monkeypatch):
    gate, solves = threading.Semaphore(0), []
    real = cv2.calibrateCameraExtended

    def held(object_points, *args, **kwargs):
        solves.append(len(object_points))
        assert gate.acquire(timeout=TIMEOUT_S)
        return real(object_points, *args, **kwargs)

    monkeypatch.setattr(cv2, 'calibrateCameraExtended', held)
    rig = make_rig({'min_views': 3, 'outlier_factor': 100.0})
    node = rig.node
    try:
        wait_for_stream(node)
        for _ in range(3):
            capture(node)
        wait_until(lambda: solves, 'the first solve')
        state = node.state()
        assert state.solving and state.solution is None
        assert state.hud.status == 'solving...'
        capture(node)
        capture(node)
        gate.release()
        wait_until(lambda: len(solves) > 1 or idle(node), 'the end of the first solve')
        assert solves == [3, 5], '... queue exactly one more solve, over all the views'
        state = node.state()
        assert state.solving and state.solution.n_views == 3
        assert state.hud.status.startswith('rms ') and 'solving...' in state.hud.status, \
            'a re-solve keeps the last solution on show'
    finally:
        gate.release(10)
    wait_until(lambda: idle(node), 'the follow-up solve')
    state = node.state()
    assert solves == [3, 5]
    assert state.solution.n_views == 5 and 'solving...' not in state.hud.status


def test_undo_renames_the_view_and_solves_again(make_rig):
    rig = make_rig({'min_views': 4, 'outlier_factor': 100.0})
    node = rig.node
    frames = node.session_dir / 'frames'
    wait_for_stream(node)
    for _ in range(5):
        capture(node)
    wait_until(lambda: idle(node), 'the solves')
    before = node.state().solution
    assert before is not None

    assert node.undo()
    assert not (frames / 'view_004.png').exists()
    assert (frames / 'undone_004.png').exists()
    wait_until(lambda: idle(node), 'the solve after the undo')
    state = node.state()
    assert state.hud.kept == 4
    assert state.solution is not before, 'still min_views kept: solved again'
    assert state.solution.n_views <= 4
    solved = state.solution

    assert node.undo()
    wait_until(lambda: idle(node), 'the node to settle')
    state = node.state()
    assert state.hud.kept == 3
    assert state.solution is solved
    assert state.hud.status.startswith('collecting ') and state.hud.status.endswith('/4')
    for _ in range(3):
        assert node.undo()
    assert not node.undo()
    assert node.state().hud.message == 'nothing to undo'
    assert sorted(path.name for path in frames.iterdir()) == \
        [f'undone_{i:03d}.png' for i in range(5)]
    assert read_log(node.session_dir)['counts']['kept'] == 4, 'the log as of the last solve'

    node.close()
    assert read_log(node.session_dir)['counts']['kept'] == 0, 'close() writes the final log'
    assert not node.request_capture()
    assert not node.undo()


class FakeWindow:
    """Stand-ins for OpenCV's window functions, so run_gui runs as on a display but opens nothing.

    Args:
        keys: Key codes waitKeyEx returns in turn (-1 once they run out).
        closed_after: Number of shown canvases after which the window reports closed, or None.
    """

    def __init__(self, keys=(), closed_after=None):
        self.keys = iter(keys)
        self.closed_after = closed_after
        self.created = []
        self.shown = []
        self.destroyed = []
        self.off_main_thread = []
        self.deadline = time.monotonic() + TIMEOUT_S

    def install(self, monkeypatch):
        monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
        for name in WINDOW_FUNCTIONS:
            monkeypatch.setattr(cv2, name, getattr(self, name, self._unexpected(name)),
                                raising=False)

    def _call(self, name):
        if threading.current_thread() is not threading.main_thread():
            self.off_main_thread.append(name)

    def _unexpected(self, name):
        def call(*args, **kwargs):
            pytest.fail(f'run_gui called cv2.{name}')
        return call

    def namedWindow(self, name, flags=None):
        self._call('namedWindow')
        self.created.append((name, flags))

    def imshow(self, name, canvas):
        self._call('imshow')
        self.shown.append((name, canvas.shape, canvas.dtype))

    def waitKeyEx(self, delay=0):
        self._call('waitKeyEx')
        if time.monotonic() > self.deadline:
            pytest.fail('the GUI loop is still running long after its script')
        time.sleep(max(delay, 1) / 1000.0)
        return next(self.keys, -1)

    def waitKey(self, delay=0):
        self._call('waitKey')
        if not self.destroyed:
            pytest.fail('run_gui read a key with cv2.waitKey: Left would read as Q, PageUp as U')
        return -1

    def getWindowProperty(self, name, prop):
        self._call('getWindowProperty')
        assert prop == cv2.WND_PROP_VISIBLE
        closed = self.closed_after is not None and len(self.shown) >= self.closed_after
        return 0.0 if closed else 1.0

    def destroyWindow(self, name):
        self._call('destroyWindow')
        self.destroyed.append(name)


def test_run_gui_drives_the_node_from_the_keys(make_rig, monkeypatch):
    rig = make_rig({'review_s': 0.3})
    node = rig.node
    live, review = [], []
    real_live, real_review = calibrate.compose_live, calibrate.compose_review

    def spy_live(frame, hud, coverage=None, canvas_size=(640, 480), show_coverage=True):
        live.append(show_coverage)
        return real_live(frame, hud, coverage, canvas_size, show_coverage)

    def spy_review(frame_full, corners_full, kept, hud, canvas_size=(640, 480)):
        review.append((frame_full.shape, hud.verdict))
        return real_review(frame_full, corners_full, kept, hud, canvas_size)

    monkeypatch.setattr(calibrate, 'compose_live', spy_live)
    monkeypatch.setattr(calibrate, 'compose_review', spy_review)
    steps = []

    def keys():
        """Yield key codes in turn, waiting for the node to react between them."""
        state = node.state
        while not (state().capturing or state().review is not None):
            yield ord(' ')
        steps.append('capture')
        while state().review is None:
            yield -1
        yield ord('m')
        steps.append('coverage off')
        while state().review is not None:
            yield -1
        yield ord('U')
        steps.append('undo')
        while state().hud.kept != 0:
            yield -1
        steps.append('quit')
        yield ord('q')

    window = FakeWindow(keys())
    window.install(monkeypatch)
    monkeypatch.setattr(calibrate, 'screen_size', lambda: (1000, 850))
    run_gui(node)

    assert steps == ['capture', 'coverage off', 'undo', 'quit']
    assert window.created == [(calibrate.WINDOW_NAME, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)]
    assert window.destroyed == [calibrate.WINDOW_NAME]
    assert window.off_main_thread == []
    assert {shape for _, shape, _ in window.shown} == {(720, 960, 3)}, 'scaled to the screen, and never resized'
    assert all(dtype == np.uint8 for _, _, dtype in window.shown)
    assert review and set(review) == {(FULL_SHAPE, 'KEPT')}
    assert live[0] is True and live[-1] is False
    assert True not in live[live.index(False):], 'M toggled the map off once'
    assert len(window.shown) == len(live) + len(review)
    assert counts(node.state()) == (0, 0)
    assert (node.session_dir / 'frames' / 'undone_000.png').exists()


@pytest.mark.parametrize('how', ['q', 'Q', 'ESC', 'window closed'])
def test_run_gui_quits(make_rig, monkeypatch, how):
    rig = make_rig()
    if how == 'window closed':
        window = FakeWindow(closed_after=3)
    else:
        window = FakeWindow([-1, -1, 27 if how == 'ESC' else ord(how)])
    window.install(monkeypatch)
    run_gui(rig.node)
    assert len(window.shown) == 3
    assert window.destroyed == [calibrate.WINDOW_NAME]


def test_run_gui_waits_for_the_window_to_show_before_watching_for_its_close(make_rig,
                                                                            monkeypatch):
    rig = make_rig()
    window = FakeWindow([-1, -1, -1, ord('q')], closed_after=0)
    window.install(monkeypatch)
    run_gui(rig.node)
    assert len(window.shown) == 4, 'only q ended the loop'


def test_run_gui_refuses_to_run_off_the_main_thread(monkeypatch):
    window = FakeWindow()
    window.install(monkeypatch)
    errors = []

    def gui():
        try:
            run_gui(None)
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=gui)
    thread.start()
    thread.join(timeout=TIMEOUT_S)
    assert errors == ['run_gui must run on the main thread']
    assert window.created == [] and window.shown == []


SPECIAL_KEYS = [0xFF51, 0xFF55, 0xFF20, 0xFE51, 0xFE55, 0xFF53, 0xFF52, 0xFF54, 0xFF56]
GTK_NUM_LOCK = 0x10 << 16


def test_run_gui_ignores_special_keys_and_modifier_bits(make_rig, monkeypatch):
    rig = make_rig()
    calls = []
    monkeypatch.setattr(rig.node, 'request_capture', lambda: calls.append('capture'))
    monkeypatch.setattr(rig.node, 'undo', lambda: calls.append('undo'))
    keys = SPECIAL_KEYS + [GTK_NUM_LOCK | ord(key) for key in ' uq']
    window = FakeWindow(keys)
    window.install(monkeypatch)
    run_gui(rig.node)
    assert len(window.shown) == len(keys), 'no special key quits; q with NumLock on does'
    assert calls == ['capture', 'undo'], 'no special key captures or undoes'


def test_spin_loop_stops_by_itself_while_nothing_can_wake_the_executor(ros):
    node = rclpy.create_node('spin_probe')
    held, release = threading.Event(), threading.Event()

    def hold():
        held.set()
        release.wait(TIMEOUT_S)

    node.create_timer(0.01, hold)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    stop = threading.Event()
    spinner = threading.Thread(target=calibrate._spin, args=(executor, stop), daemon=True)
    spinner.start()
    try:
        assert held.wait(5.0), 'the timer never ran'
        time.sleep(0.3)
        stop.set()
        spinner.join(timeout=2.0)
        assert not spinner.is_alive(), 'the spin loop missed stop'
    finally:
        release.set()
        stop.set()
        executor.shutdown()
        spinner.join(timeout=TIMEOUT_S)
        node.destroy_node()


def test_spin_loop_ends_quietly_when_ros_shuts_down(ros):
    context = rclpy.Context()
    rclpy.init(args=[], context=context, domain_id=DOMAIN_ID)
    node = rclpy.create_node('spin_probe', namespace=NAMESPACE, context=context)
    node.create_timer(0.005, lambda: None)
    executor = MultiThreadedExecutor(num_threads=2, context=context)
    executor.add_node(node)
    errors = []

    def spin():
        try:
            calibrate._spin(executor, threading.Event())
        except Exception as error:
            errors.append(error)

    spinner = threading.Thread(target=spin, daemon=True)
    spinner.start()
    time.sleep(0.3)
    context.try_shutdown()
    spinner.join(timeout=5.0)
    node.destroy_node()
    assert not spinner.is_alive() and errors == []


def patch_main(monkeypatch, make_node):
    """Let main() run in this process on the node make_node() builds; return its list of events.

    rclpy.init and rclpy.try_shutdown only record themselves there, so the tests' context stays up.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        make_node: Callable building the node main() runs.
    """
    events = []
    monkeypatch.setattr(rclpy, 'init', lambda args=None: events.append('init'))
    monkeypatch.setattr(rclpy, 'try_shutdown', lambda: events.append('try_shutdown'))
    monkeypatch.setattr(calibrate, 'CalibrateCam', make_node)
    return events


def test_main_stops_the_spin_loop_before_tearing_down(ros, tmp_path, monkeypatch):
    config = write_config(tmp_path / 'cfg' / 'cfg.yaml', {})
    real_spin = calibrate._spin

    class Executor(MultiThreadedExecutor):

        def shutdown(self, *args, **kwargs):
            events.append('executor.shutdown')
            return super().shutdown(*args, **kwargs)

    class Node(CalibrateCam):

        def close(self):
            events.append('close')
            super().close()

        def destroy_node(self):
            events.append('destroy_node')
            super().destroy_node()

    def spin(executor, stop):
        rescue = threading.Timer(TIMEOUT_S, executor.shutdown)
        rescue.daemon = True
        rescue.start()
        try:
            real_spin(executor, stop)
        finally:
            rescue.cancel()
        events.append('spin loop ended')

    events = patch_main(monkeypatch,
                        lambda: Node(cfg_pth=str(config), output_dir=str(tmp_path / 'out'),
                                     detector='opencv'))
    monkeypatch.setattr(calibrate, 'MultiThreadedExecutor', Executor)
    monkeypatch.setattr(calibrate, '_spin', spin)
    FakeWindow([-1, -1, ord('q')]).install(monkeypatch)
    calibrate.main()
    assert events == ['init', 'close', 'spin loop ended', 'destroy_node', 'try_shutdown']
    assert len(list((tmp_path / 'out').glob(f'*/{LOG_NAME}'))) == 1, 'close() wrote the log'


@pytest.mark.parametrize('calibration', [
    {'board': 5},
    {'board': {'squares': 5}},
    {'min_views': [3]},
])
def test_main_reports_a_mistyped_config_in_one_line(ros, tmp_path, monkeypatch, calibration):
    config = write_config(tmp_path / 'cfg' / 'cfg.yaml', calibration)
    events = patch_main(monkeypatch, lambda: CalibrateCam(cfg_pth=str(config),
                                                          output_dir=str(tmp_path / 'out')))
    with pytest.raises(SystemExit) as exit_info:
        calibrate.main()
    reason = exit_info.value.code
    assert isinstance(reason, str) and reason.startswith('calibrate: ') and '\n' not in reason
    assert events == ['init', 'try_shutdown']
    assert not (tmp_path / 'out').exists()


def test_main_reports_a_mistyped_parameter_in_one_line(ros, tmp_path):
    config = write_config(tmp_path / 'cfg' / 'cfg.yaml', {})
    out = tmp_path / 'out'
    package_root = str(Path(convchart_ros.__file__).resolve().parents[1])
    env = dict(os.environ, ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST',
               ROS_DOMAIN_ID=str(DOMAIN_ID), PYTHONDONTWRITEBYTECODE='1',
               QT_QPA_PLATFORM='offscreen',
               PYTHONPATH=os.pathsep.join([package_root, os.environ.get('PYTHONPATH', '')]))
    for name in ('DISPLAY', 'WAYLAND_DISPLAY'):
        env.pop(name, None)
    result = subprocess.run(
        [sys.executable, '-m', 'convchart_ros.calibrate', '--ros-args',
         '-r', f'__ns:={NAMESPACE}_main', '-p', f'config:={config}', '-p', f'output_dir:={out}',
         '-p', 'auto_replace:=1'],
        env=env, capture_output=True, text=True, timeout=TIMEOUT_S)
    assert result.returncode == 1, result.stderr
    assert 'Traceback' not in result.stderr, result.stderr
    reason = result.stderr.strip().splitlines()[-1]
    assert reason.startswith('calibrate: ') and "'auto_replace'" in reason, result.stderr
    assert not out.exists()


def test_hud_counts_the_views_dropped_as_outliers(make_rig):
    rig = make_rig({'min_views': 3, 'outlier_factor': 1.0, 'outlier_floor_px': 0.0})
    node = rig.node
    for _ in range(4):
        capture(node)
    wait_until(lambda: settled(node), 'the solves')
    state = node.state()
    assert state.hud.kept == 4 and state.solution.n_views == 3
    assert state.hud.status.startswith('rms ')
    assert state.hud.status.endswith(' | 1 outlier dropped'), state.hud.status


def test_hud_reports_a_missing_or_stalled_stream_and_a_capture_in_flight(make_rig):
    rig = make_rig(camera_args={'rate_hz': 0.001})
    node = rig.node
    state = node.state()
    assert state.frame is None and state.hud.message == 'waiting for the image stream'
    assert not node.undo()
    assert node.state().hud.message == 'nothing to undo', 'a transient message comes first'

    rig.replace_camera(lambda: FakeCamera(reply_delay_s=0.5))
    wait_for_stream(node)
    wait_until(node.request_capture, 'the node to take a capture request')
    state = node.state()
    assert state.capturing and state.hud.status.endswith(' | capturing...'), state.hud.status
    assert state.hud.message is None, 'nothing to report while the stream is live'
    wait_until(lambda: not node.state().capturing, 'the capture reply')
    assert 'capturing' not in node.state().hud.status

    rig.replace_camera(lambda: FakeCamera(rate_hz=0.001))
    stalled = 'stream stalled: no frame for '
    wait_until(lambda: (node.state().hud.message or '').startswith(stalled), 'the stall to show',
               timeout=5.0)
    assert node.state().frame is not None, 'the last frame stays on screen'
