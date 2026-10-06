"""
End-to-end tests of the calibration node (calibrate.py) with the fake camera over ROS 2, headless.

FakeCamera and CalibrateCam run in one process, each on a MultiThreadedExecutor of its own, on an
isolated domain with discovery limited to localhost and a namespace private to the process, so
nothing here can reach a real robot or another test run; the node also starts as main() starts it,
from ROS parameters, in a subprocess under the same isolation. Captures go through
CalibrateCam.request_capture(), as the GUI's SPACE key does. run_gui runs against stand-ins for
OpenCV's window functions, so no window ever opens. The calibration must recover the fake camera's
ground truth (K_full, dist).
"""

from dataclasses import dataclass
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
import cv2
import numpy as np
import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
import yaml

# 21-40: private, not the robot's 42 nor the other ROS tests' 51-70 and 77-96. Domains above 101
# would put the DDS ports in Linux's ephemeral port range, where any socket could hold them.
DOMAIN_ID = 21 + os.getpid() % 20
# Per process: two runs that land on the same domain still never see each other's topics.
NAMESPACE = f'/calibrate_test_{os.getpid()}'
TIMEOUT_S = 60.0
FULL_SHAPE = (1200, 1600)
STREAM_SIZE = (640, 480)
LOG_NAME = 'calibration_log.yaml'
BACKUP_NAME = 'cfg.yaml.bak'
MIN_VIEWS = 12                          # the default
# A board other than the default, set the way a user would: squares, sizes, dictionary, first id.
CUSTOM_BOARD = {'squares': [7, 5], 'square_length_m': 0.03, 'marker_length_m': 0.022,
                'dictionary': 'DICT_4X4_50', 'first_marker_id': 10}

# The repo's config without its CALIBRATION block. The comment makes 'byte-identical' strict: any
# rewrite through PyYAML would drop it.
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
SESSION_KEYS = {'started', 'config', 'auto_replace', 'config_written', 'config_backup'}
BOARD_KEYS = {'squares', 'square_length_m', 'marker_length_m', 'dictionary', 'first_marker_id',
              'legacy_pattern'}
SOLUTION_KEYS = {'rms_px', 'converged', 'K_full', 'dist', 'std', 'K_stream', 'stream_size'}
STD_KEYS = {'fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3'}
VIEW_KEYS = {'index', 'file', 'n_corners', 'error_px', 'dropped'}
HISTORY_KEYS = {'n_views', 'rms_px', 'fx', 'fy', 'cx', 'cy'}

# OpenCV window functions: run_gui may use the first six (waitKey only to let the window close);
# any other one fails the test.
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
    """Idle and back in live mode: no capture, no solve, no review showing."""
    state = node.state()
    return not state.capturing and not state.solving and state.review is None


def wait_for_stream(node):
    wait_until(lambda: node.state().frame is not None, 'the first stream frame')


def capture(node):
    """
    Capture as the SPACE key does and wait until the node has handled the reply.

    The request is repeated until the node takes it (none in flight and no review showing).
    Returns the state right after the reply.
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
    """Project the camera's chessboard corners for one pose with its true K and dist."""
    rvec, tvec = camera.poses[pose_index]
    points = np.asarray(camera.board.getChessboardCorners(), dtype=np.float64)
    return cv2.projectPoints(points, rvec, tvec, camera.K_full, camera.dist)[0].reshape(-1, 2)


def distortion_error_px(camera, dist):
    """
    Return the largest image displacement (px) between dist and the camera's true distortion.

    Both use the true K, over the board corners of all the camera's poses: the distortion model
    is compared where it was observed.
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
    """
    Spin node in a MultiThreadedExecutor on a background thread; return a function that stops it.

    The stop function ends the spin loop, waits for the callbacks still running, and only then
    shuts the executor down; once it returns, the node may be destroyed. In rclpy 7.1, shutdown()
    destroys the guard conditions that a loop still spinning may be about to wait on, and that a
    callback still running signals with, before it waits for the callbacks.
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
        executor._executor.shutdown(wait=True)      # its thread pool: the callbacks still running
        executor.shutdown()

    return stop


class Rig:
    """
    Spin a CalibrateCam and a FakeCamera, each in a MultiThreadedExecutor on a background thread.

    The camera has an executor of its own so it can be replaced while the node keeps spinning.
    """

    def __init__(self, camera, node, config):
        self.camera = camera
        self.node = node
        self.config = config
        self._stop_node = spin_in_thread(node, num_threads=6)
        self._stop_camera = spin_in_thread(camera, num_threads=3)

    def replace_camera(self, make_camera):
        """
        Stop the camera, destroy it and start the one make_camera() makes in its place.

        A node destroyed under one of its running callbacks (a stream render, a late reply) raises
        in the executor's thread, which then stops spinning. Only the camera's executor stops:
        stopping one that also spins the node and spinning the node again could leave one of its
        callback groups busy for good.
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
    """Write a config with this CALIBRATION block and start a camera with its board and a node."""
    config = write_config(tmp_path / 'cfg' / 'cfg.yaml', calibration)
    camera = FakeCamera(board_spec=board_spec_from_cfg(calibration.get('board')),
                        **(camera_args or {}))
    try:
        node = CalibrateCam(cfg_pth=str(config), output_dir=str(tmp_path / 'out'), **node_args)
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
    """A complete calibration session, captured and closed, with what it left behind."""

    camera: FakeCamera          # destroyed; its ground truth (K_full, dist, poses, board) remains
    config: Path
    original: bytes             # the config before the session
    session_dir: Path
    auto_replace: bool
    board: dict
    first_solve: dict           # config bytes, log and backup stat once min_views were first kept
    state: GuiState             # once idle after the last capture
    last_solve: dict            # the log as the last solve wrote it, read before close()
    log_text: str               # the final log, as close() wrote it

    @property
    def log(self):
        return yaml.safe_load(self.log_text)


def full_session(tmp_path, auto_replace, board, blank_every, n_captures):
    """
    Run a whole calibration session and return it as a Run.

    Captures n_captures stills through request_capture (solves start at MIN_VIEWS kept views),
    then closes the node. Every node is gone when this returns.
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
        last_solve = read_log(node.session_dir)     # before close(), which rewrites the log
        node.close()
        return Run(rig.camera, config, original, node.session_dir, auto_replace, board or {},
                   first_solve, state, last_solve, (node.session_dir / LOG_NAME).read_text())
    finally:
        rig.close()


@pytest.fixture(scope='module')
def replace_run(ros, tmp_path_factory):
    """auto_replace=True on the default (dcc) board; every 5th capture is blank."""
    return full_session(tmp_path_factory.mktemp('replace'), auto_replace=True, board=None,
                        blank_every=5, n_captures=20)


@pytest.fixture(scope='module')
def keep_run(ros, tmp_path_factory):
    """auto_replace=False on a user-specified 7x5 DICT_4X4_50 board with marker ids from 10."""
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
])
def test_bad_configuration_is_rejected(ros, tmp_path, cfg_pth, calibration, node_args, error,
                                       match):
    write_config(tmp_path / 'cfg.yaml', calibration or {})
    with pytest.raises(error, match=match):
        CalibrateCam(cfg_pth=str(tmp_path / cfg_pth) if cfg_pth else '',
                     output_dir=str(tmp_path / 'out'), **node_args)
    assert not (tmp_path / 'out').exists(), 'no session folder for a node that did not start'


def test_ros_parameters_configure_the_node_as_main_does(ros, tmp_path):
    # main() builds CalibrateCam() without arguments: the ROS parameters are all it is told.
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
         '-p', 'capture_timeout_s:=3'],             # an integer is accepted too
        env=env, capture_output=True, text=True, timeout=TIMEOUT_S)
    assert result.returncode == 0, result.stderr
    [session_dir] = out.iterdir()
    log = read_log(session_dir)
    assert log['session']['config'] == str(config)
    assert log['session']['auto_replace'] is True
    assert log['board']['squares'] == CUSTOM_BOARD['squares'], 'the board from that config'


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
    last = run.last_solve                       # the last solve's log, not the one close() wrote
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
    rig = make_rig({'min_views': 3}, {'rate_hz': 0.001}, auto_replace=True)   # no stream frame
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

    state = capture(node)                       # the fake camera blanks every 2nd capture
    assert counts(state) == (1, 1)
    assert state.review is not None and not state.review.kept
    assert state.hud.verdict == 'REJECTED: no board found'
    assert state.hud.verdict_ok is False
    assert len(state.review.corners) == 0
    assert (node.session_dir / 'frames' / 'rejected_000.png').exists()


def test_capture_timeout_is_reported_and_the_node_keeps_working(make_rig):
    # capture_timeout_s also bounds the capture after the swap below, which takes about 0.1 s on
    # an idle machine and several times that on a busy one.
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
    # The camera still answers after 2 s (and moves to its next pose); the node ignores it.
    wait_until(lambda: rig.camera.pose_index == 1, 'the late reply')
    time.sleep(0.3)
    state = node.state()
    assert counts(state) == (0, 0) and not state.capturing and state.review is None
    assert list((node.session_dir / 'frames').iterdir()) == []

    # The node keeps working: the next request is taken, and a camera that answers in time is
    # served as usual.
    rig.replace_camera(FakeCamera)
    state = capture(node)
    assert counts(state) == (1, 0)
    assert state.hud.verdict == 'KEPT'
    # (the HUD may report the stream's gap during the swap instead)
    assert 'timed out' not in (state.hud.message or ''), 'a new request clears the old message'
    assert (node.session_dir / 'frames' / 'view_000.png').exists()


def test_a_capture_during_a_solve_queues_exactly_one_more_solve(make_rig, monkeypatch):
    gate, solves = threading.Semaphore(0), []
    real = cv2.calibrateCameraExtended

    def held(object_points, *args, **kwargs):  # each solve waits here for a permit
        solves.append(len(object_points))
        assert gate.acquire(timeout=TIMEOUT_S)
        return real(object_points, *args, **kwargs)

    monkeypatch.setattr(cv2, 'calibrateCameraExtended', held)
    rig = make_rig({'min_views': 3, 'outlier_factor': 100.0})      # no outlier pass: 1 call/solve
    node = rig.node
    try:
        wait_for_stream(node)
        for _ in range(3):
            capture(node)
        wait_until(lambda: solves, 'the first solve')
        state = node.state()
        assert state.solving and state.solution is None
        assert state.hud.status == 'solving...'
        capture(node)                           # two kept views while the first solve runs ...
        capture(node)
        gate.release()
        wait_until(lambda: len(solves) > 1 or idle(node), 'the end of the first solve')
        assert solves == [3, 5], '... queue exactly one more solve, over all the views'
        state = node.state()
        assert state.solving and state.solution.n_views == 3
        assert state.hud.status.startswith('rms ') and 'solving...' in state.hud.status, \
            'a re-solve keeps the last solution on show'
    finally:
        gate.release(10)                        # no solve waits from here on; an extra one shows
    wait_until(lambda: idle(node), 'the follow-up solve')
    state = node.state()
    assert solves == [3, 5]
    assert state.solution.n_views == 5 and 'solving...' not in state.hud.status


def test_undo_renames_the_view_and_solves_again(make_rig):
    rig = make_rig({'min_views': 4, 'outlier_factor': 100.0})     # no view dropped as an outlier
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

    assert node.undo()                          # 3 left, below min_views: no solve
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
    """
    Stand-ins for OpenCV's window functions, so run_gui runs as if on a display but nothing opens.

    waitKeyEx returns the next key code from keys (-1 once they run out); waitKey, whose 8-bit mask
    would turn special keys into commands, may only follow destroyWindow. getWindowProperty reports
    the window closed once closed_after canvases have been shown, if set.
    """

    def __init__(self, keys=(), closed_after=None):
        self.keys = iter(keys)
        self.closed_after = closed_after
        self.created = []
        self.shown = []                 # (name, shape, dtype) of every canvas shown
        self.destroyed = []
        self.off_main_thread = []
        self.deadline = time.monotonic() + TIMEOUT_S

    def install(self, monkeypatch):
        # Belt and braces: even a window call that slipped past the stand-ins would stay offscreen.
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
        """Press keys the way a user would, waiting for the node between them."""
        state = node.state
        while not (state().capturing or state().review is not None):
            yield ord(' ')                  # SPACE until a capture is under way
        steps.append('capture')
        while state().review is None:
            yield -1
        yield ord('m')                      # coverage map off, during the review
        steps.append('coverage off')
        while state().review is not None:
            yield -1
        yield ord('U')                      # undo, after the review
        steps.append('undo')
        while state().hud.kept != 0:
            yield -1
        steps.append('quit')
        yield ord('q')

    window = FakeWindow(keys())
    window.install(monkeypatch)
    run_gui(node)

    assert steps == ['capture', 'coverage off', 'undo', 'quit']
    assert window.created == [(calibrate.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)]
    assert window.destroyed == [calibrate.WINDOW_NAME]
    assert window.off_main_thread == []
    assert {shape for _, shape, _ in window.shown} == {(480, 640, 3)}, 'the window never resizes'
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
    window = FakeWindow([-1, -1, -1, ord('q')], closed_after=0)     # never reports visible
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


# X11 keysyms that waitKey's 8-bit mask turned into commands: Left (Q), PageUp (U), Multi_key
# (SPACE), dead_acute (Q) and dead_breve (U); then Right, Up, Down and PageDown (S, R, T and V).
SPECIAL_KEYS = [0xFF51, 0xFF55, 0xFF20, 0xFE51, 0xFE55, 0xFF53, 0xFF52, 0xFF54, 0xFF56]
GTK_NUM_LOCK = 0x10 << 16       # GTK adds the modifier state above bit 16; NumLock is Mod2


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
    # main() never calls executor.shutdown(), so the loop must see stop on its own, even while its
    # node's only timer is held by its own callback (as the 'calib' group can be when the GUI
    # quits) and no stream frame comes.
    node = rclpy.create_node('spin_probe')
    held, release = threading.Event(), threading.Event()

    def hold():
        held.set()
        release.wait(TIMEOUT_S)

    node.create_timer(0.01, hold)           # in the node's default, mutually exclusive group
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    stop = threading.Event()
    spinner = threading.Thread(target=calibrate._spin, args=(executor, stop), daemon=True)
    spinner.start()
    try:
        assert held.wait(5.0), 'the timer never ran'
        time.sleep(0.3)                     # the executor now waits on nothing that can fire
        stop.set()
        spinner.join(timeout=2.0)
        assert not spinner.is_alive(), 'the spin loop missed stop'
    finally:
        release.set()
        stop.set()
        executor.shutdown()                 # ends a loop that missed stop
        spinner.join(timeout=TIMEOUT_S)
        node.destroy_node()


def test_spin_loop_ends_quietly_when_ros_shuts_down(ros):
    # Ctrl-C and SIGTERM shut the context down under the loop; it must end without an error.
    context = rclpy.Context()
    rclpy.init(args=[], context=context, domain_id=DOMAIN_ID)
    node = rclpy.create_node('spin_probe', namespace=NAMESPACE, context=context)
    node.create_timer(0.005, lambda: None)  # keeps the loop busy as the context goes down
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
    """
    Let main() run in this process on the node make_node() builds; return its list of events.

    rclpy.init and rclpy.try_shutdown only record themselves there, so the tests' context stays up.
    """
    events = []
    monkeypatch.setattr(rclpy, 'init', lambda args=None: events.append('init'))
    monkeypatch.setattr(rclpy, 'try_shutdown', lambda: events.append('try_shutdown'))
    monkeypatch.setattr(calibrate, 'CalibrateCam', make_node)
    return events


def test_main_stops_the_spin_loop_before_tearing_down(ros, tmp_path, monkeypatch):
    # Quitting closes the node, stops the spin loop and joins it, then destroys the node and shuts
    # ROS down, with no executor.shutdown(): in rclpy 7.1 that destroys guard conditions under a
    # wait set being built, which could leave the spin thread, and main() joining it, blocked.
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
        # A loop that missed stop would block main() in join(): after TIMEOUT_S, end it the old
        # way, so that the test fails instead of hanging.
        rescue = threading.Timer(TIMEOUT_S, executor.shutdown)
        rescue.daemon = True
        rescue.start()
        try:
            real_spin(executor, stop)
        finally:
            rescue.cancel()
        events.append('spin loop ended')

    events = patch_main(monkeypatch,
                        lambda: Node(cfg_pth=str(config), output_dir=str(tmp_path / 'out')))
    monkeypatch.setattr(calibrate, 'MultiThreadedExecutor', Executor)
    monkeypatch.setattr(calibrate, '_spin', spin)
    FakeWindow([-1, -1, ord('q')]).install(monkeypatch)
    calibrate.main()
    assert events == ['init', 'close', 'spin loop ended', 'destroy_node', 'try_shutdown']
    assert len(list((tmp_path / 'out').glob(f'*/{LOG_NAME}'))) == 1, 'close() wrote the log'


@pytest.mark.parametrize('calibration', [
    {'board': 5},                           # a number where the board's mapping goes
    {'board': {'squares': 5}},              # a number where [cols, rows] goes
    {'min_views': [3]},                     # a list where a number goes
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
    # As started from a shell: exit status 1 and a one-line reason, no traceback.
    config = write_config(tmp_path / 'cfg' / 'cfg.yaml', {})
    out = tmp_path / 'out'
    package_root = str(Path(convchart_ros.__file__).resolve().parents[1])
    env = dict(os.environ, ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST',
               ROS_DOMAIN_ID=str(DOMAIN_ID), PYTHONDONTWRITEBYTECODE='1',
               QT_QPA_PLATFORM='offscreen',
               PYTHONPATH=os.pathsep.join([package_root, os.environ.get('PYTHONPATH', '')]))
    for name in ('DISPLAY', 'WAYLAND_DISPLAY'):     # no window, even if the node did start
        env.pop(name, None)
    result = subprocess.run(
        [sys.executable, '-m', 'convchart_ros.calibrate', '--ros-args',
         '-r', f'__ns:={NAMESPACE}_main', '-p', f'config:={config}', '-p', f'output_dir:={out}',
         '-p', 'auto_replace:=1'],                  # an integer for a bool
        env=env, capture_output=True, text=True, timeout=TIMEOUT_S)
    assert result.returncode == 1, result.stderr
    assert 'Traceback' not in result.stderr, result.stderr
    reason = result.stderr.strip().splitlines()[-1]
    assert reason.startswith('calibrate: ') and "'auto_replace'" in reason, result.stderr
    assert not out.exists()


def test_hud_counts_the_views_dropped_as_outliers(make_rig):
    # outlier_factor 1: the solve over 4 views drops the worst one, down to min_views
    rig = make_rig({'min_views': 3, 'outlier_factor': 1.0})
    node = rig.node
    for _ in range(4):
        capture(node)
    wait_until(lambda: settled(node), 'the solves')
    state = node.state()
    assert state.hud.kept == 4 and state.solution.n_views == 3
    assert state.hud.status.startswith('rms ')
    assert state.hud.status.endswith(' | 1 outlier dropped'), state.hud.status


def test_hud_reports_a_missing_or_stalled_stream_and_a_capture_in_flight(make_rig):
    rig = make_rig(camera_args={'rate_hz': 0.001})             # no stream frame
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

    rig.replace_camera(lambda: FakeCamera(rate_hz=0.001))     # the stream stops
    stalled = 'stream stalled: no frame for '
    wait_until(lambda: (node.state().hud.message or '').startswith(stalled), 'the stall to show',
               timeout=5.0)
    assert node.state().frame is not None, 'the last frame stays on screen'
