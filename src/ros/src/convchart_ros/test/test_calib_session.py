"""
Tests for the calibration session (calib_session.py).

No ROS and no GPU. evaluate() runs on ideal pinhole renders of the board (generateImage warped by a
known homography); solve() runs on correspondences projected with a known K and dist plus small
noise. Expectations come from that ground truth wherever there is one.
"""

import stat
import threading

from convchart_ros.calib_utils.calib_board import BoardSpec, make_board
from convchart_ros.calib_utils.calib_session import (
    build_log, CalibrationSession, CalibrationSettings, charuco_offset_px, Detection,
    REJECT_COLLINEAR, REJECT_NO_BOARD, scale_intrinsics, settings_from_cfg, write_camera_to_cfg,
    write_log)
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

FULL = (1600, 1200)
STREAM = (640, 480)
K_TRUE = np.array([[1400.0, 0.0, 799.5], [0.0, 1400.0, 599.5], [0.0, 0.0, 1.0]])
DIST_TRUE = np.array([-0.12, 0.05, 0.0008, -0.0005, 0.0])
CHESS = make_board(BoardSpec()).getChessboardCorners().astype(np.float32)   # ids 0..15
OUTLINE = np.array([[0, 0, 0], [5, 0, 0], [5, 5, 0], [0, 5, 0]], dtype=np.float64)
PX_PER_SQUARE = 100
BACKGROUND = 140
RVEC = np.array([0.35, -0.25, 0.15])
TVEC = np.array([-2.5, -2.5, 12.0])     # 5x5 board of unit squares, centred in the frame
STD_KEYS = ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3']

# The CALIBRATION block as documented in docs/calibration_gui_design.md.
DOCUMENTED_BLOCK = """
CALIBRATION:
  board:
    squares: [5, 5]
    square_length_m: 1.0
    marker_length_m: null
    marker_ratio: 0.7
    dictionary: DICT_5X5_50
    first_marker_id: 0
    legacy_pattern: false
  min_corners: null
  min_views: 12
  review_s: 2.0
  outlier_factor: 3.0
  fix_k3: false
  coverage_grid: [6, 8]
  converge_rel_sigma_f: 0.002
  converge_sigma_c_px: 1.0
"""

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

CAMERA:
  K: null
  dist: null

CALIBRATION:
  min_views: 15
"""


def render_board(spec, rvec, tvec, visible=None):
    """
    Render spec's board under (rvec, tvec) through K_TRUE without distortion, as a mono frame.

    visible = (row0, row1, col0, col1) keeps only those squares; the rest shows the background.
    """
    cols, rows = spec.squares
    image = make_board(spec).generateImage((cols * PX_PER_SQUARE, rows * PX_PER_SQUARE),
                                           marginSize=0, borderBits=1)
    if visible is not None:
        row0, row1, col0, col1 = (PX_PER_SQUARE * v for v in visible)
        masked = np.full_like(image, BACKGROUND)
        masked[row0:row1, col0:col1] = image[row0:row1, col0:col1]
        image = masked
    # Board-image pixel (x, y) is centred on ((x + 0.5) * scale, (y + 0.5) * scale) on the board.
    scale = spec.square_length / PX_PER_SQUARE
    to_board = np.array([[scale, 0.0, 0.5 * scale], [0.0, scale, 0.5 * scale], [0.0, 0.0, 1.0]])
    rot = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
    homography = K_TRUE @ np.column_stack([rot[:, 0], rot[:, 1], tvec]) @ to_board
    frame = np.full((FULL[1], FULL[0]), BACKGROUND, np.uint8)
    frame = cv2.warpPerspective(image, homography, FULL, dst=frame, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_TRANSPARENT)
    return cv2.GaussianBlur(frame, (0, 0), 0.8)


def board_poses(n, seed, k=K_TRUE, size=FULL):
    """Return n board poses over a 3x3 grid of image regions, near and far, tilted 15-40 deg."""
    rng = np.random.default_rng(seed)
    poses = []
    for _ in range(100 * n):
        cell = len(poses) % 9
        u = (cell % 3 + 0.5 + rng.uniform(-0.3, 0.3)) / 3 * size[0]
        v = (cell // 3 + 0.5 + rng.uniform(-0.3, 0.3)) / 3 * size[1]
        tilt = rng.uniform(15, 40, 2) * rng.choice([-1, 1], 2)
        rot = Rotation.from_euler('xyz', [tilt[0], tilt[1], rng.uniform(-20, 20)], degrees=True)
        depth = rng.uniform(9, 18)      # the board spans 24-49% of the frame width at any scale
        tvec = depth * np.linalg.solve(k, [u, v, 1.0]) - rot.apply([2.5, 2.5, 0.0])
        rvec = rot.as_rotvec()
        outline, _ = cv2.projectPoints(OUTLINE, rvec, tvec, k, DIST_TRUE)
        outline = outline.reshape(-1, 2)
        if (outline > 10).all() and (outline < np.subtract(size, 10)).all():
            poses.append((rvec, tvec))
            if len(poses) == n:
                return poses
    raise AssertionError(f'only {len(poses)} of {n} poses keep the board inside the frame')


def kept_view(points, size=FULL):
    """Wrap image points of the first len(points) inner corners as a kept Detection."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return Detection(kept=True, reason='', corners=points, ids=np.arange(len(points)),
                     image_size=size, obj_points=CHESS[:len(points)].copy(),
                     img_points=points.astype(np.float32))


def synthetic_views(n, seed, noise_px=0.05, k=K_TRUE, size=FULL):
    """Project the inner corners under n diverse poses with k and DIST_TRUE, plus pixel noise."""
    rng = np.random.default_rng(seed + 1000)
    views = []
    for rvec, tvec in board_poses(n, seed, k, size):
        points, _ = cv2.projectPoints(CHESS, rvec, tvec, k, DIST_TRUE)
        views.append(kept_view(points.reshape(-1, 2) + rng.normal(0.0, noise_px, (len(CHESS), 2)),
                               size))
    return views


def noisy(view, sigma_px, seed=0):
    """Return a copy of a kept view with its image points perturbed by sigma_px of noise."""
    rng = np.random.default_rng(seed)
    return kept_view(view.corners + rng.normal(0.0, sigma_px, view.corners.shape), view.image_size)


def rejected(reason=REJECT_NO_BOARD):
    return Detection(False, reason, np.zeros((0, 2)), np.zeros(0, dtype=int), FULL, None, None)


def solved_session(views, settings=None):
    session = CalibrationSession(settings or CalibrationSettings())
    for view in views:
        session.submit(view)
    return session, session.solve()


def parameters(K, dist):
    return dict(zip(STD_KEYS, [K[0, 0], K[1, 1], K[0, 2], K[1, 2], *np.ravel(dist)]))


def assert_recovers(solution, k=K_TRUE):
    """Every parameter within 4 sigma (the solve's own std) of the truth; fx, fy within 0.5%."""
    estimate = parameters(solution.K, solution.dist)
    truth = parameters(k, DIST_TRUE)
    for name in STD_KEYS:
        error = abs(estimate[name] - truth[name])
        assert error <= 4.0 * solution.std[name], \
            f'{name} = {estimate[name]:.6g}, truth {truth[name]:.6g}, std {solution.std[name]:.3g}'
    assert abs(estimate['fx'] / truth['fx'] - 1.0) < 0.005
    assert abs(estimate['fy'] / truth['fy'] - 1.0) < 0.005


def record_calibrations(monkeypatch):
    """Spy on cv2.calibrateCameraExtended, recording the size, initial guess and flags."""
    calls = []
    real = cv2.calibrateCameraExtended

    def spy(obj, img, size, camera_matrix, dist_coeffs, *args, **kwargs):
        # Copy before calling: OpenCV writes its result into the guess arrays.
        calls.append({'size': tuple(size), 'flags': kwargs.get('flags', 0),
                      'K0': None if camera_matrix is None else np.array(camera_matrix),
                      'dist0': None if dist_coeffs is None else np.array(dist_coeffs)})
        return real(obj, img, size, camera_matrix, dist_coeffs, *args, **kwargs)

    monkeypatch.setattr(cv2, 'calibrateCameraExtended', spy)
    return calls


@pytest.fixture
def emulate_detector(monkeypatch):
    """
    Return emulate(shift), which swaps in a CharucoDetector with a given corner convention.

    The swapped-in detector reports corners shift px off pixel centres (one number for both axes,
    or one per axis; None finds no board), whatever the installed OpenCV's own convention. The
    session's measured offset is forgotten on each swap and after the test, so later tests
    measure the real detector again.
    """
    real_detector = cv2.aruco.CharucoDetector
    real_shift = charuco_offset_px()

    def emulate(shift):
        class Detector:
            def __init__(self, board):
                self._detector = real_detector(board)

            def detectBoard(self, image):
                corners, ids, marker_corners, marker_ids = self._detector.detectBoard(image)
                if shift is None:
                    return None, None, marker_corners, marker_ids
                if corners is not None:
                    corners = corners - np.float32(real_shift) + np.asarray(shift, np.float32)
                return corners, ids, marker_corners, marker_ids

        monkeypatch.setattr(cv2.aruco, 'CharucoDetector', Detector)
        charuco_offset_px.cache_clear()

    yield emulate
    charuco_offset_px.cache_clear()


# --- settings -------------------------------------------------------------------------------


def test_settings_defaults():
    settings = CalibrationSettings()
    assert settings.board == BoardSpec()
    assert settings.min_corners is None
    assert settings.min_corners_resolved == 8           # 5x5 board: max(6, ceil(16 / 2))
    assert settings.min_views == 12
    assert settings.review_s == 2.0
    assert settings.outlier_factor == 3.0
    assert settings.fix_k3 is False
    assert settings.coverage_grid == (6, 8)
    assert settings.converge_rel_sigma_f == 0.002
    assert settings.converge_sigma_c_px == 1.0


@pytest.mark.parametrize('squares, min_corners, expected', [
    ((5, 5), None, 8),
    ((11, 8), None, 35),        # 70 inner corners
    ((3, 3), None, 6),          # 4 inner corners: the floor of 6 wins
    ((5, 5), 10, 10),
])
def test_min_corners_resolved(squares, min_corners, expected):
    settings = CalibrationSettings(board=BoardSpec(squares=squares), min_corners=min_corners)
    assert settings.min_corners_resolved == expected


@pytest.mark.parametrize('cfg', [None, {}, {'CALIBRATION': None}, {'CALIBRATION': {}},
                                 {'CALIBRATION': {'min_corners': None, 'board': None}},
                                 yaml.safe_load(DOCUMENTED_BLOCK)])
def test_settings_from_cfg_defaults(cfg):
    assert settings_from_cfg(cfg) == CalibrationSettings()


def test_settings_from_cfg_maps_every_key():
    cfg = {'MODEL': {'detector': 'x.onnx'}, 'CALIBRATION': {
        'board': {'squares': [7, 5], 'square_length_m': 0.03, 'dictionary': 'DICT_4X4_50'},
        'min_corners': 9, 'min_views': 20, 'review_s': 1.5, 'outlier_factor': 2.5,
        'fix_k3': True, 'coverage_grid': [4, 5], 'converge_rel_sigma_f': 0.001,
        'converge_sigma_c_px': 0.5}}
    assert settings_from_cfg(cfg) == CalibrationSettings(
        board=BoardSpec(squares=(7, 5), square_length=0.03, dictionary='DICT_4X4_50'),
        min_corners=9, min_views=20, review_s=1.5, outlier_factor=2.5, fix_k3=True,
        coverage_grid=(4, 5), converge_rel_sigma_f=0.001, converge_sigma_c_px=0.5)


@pytest.mark.parametrize('block, match', [
    ({'min_view': 10}, 'unknown CALIBRATION keys'),
    ({'coverage_grid': [6, 8, 2]}, r'\[rows, cols\]'),
    ({'board': {'colour': 'red'}}, 'unknown CALIBRATION.board keys'),
])
def test_bad_settings_config_is_rejected(block, match):
    with pytest.raises(ValueError, match=match):
        settings_from_cfg({'CALIBRATION': block})


@pytest.mark.parametrize('settings, match', [
    (CalibrationSettings(board=BoardSpec(squares=(1, 5))), 'at least 2x2'),
    (CalibrationSettings(min_corners=3), 'at least 4'),
    (CalibrationSettings(min_views=0), 'at least 1'),
    (CalibrationSettings(coverage_grid=(0, 8)), 'coverage_grid'),
])
def test_session_rejects_bad_settings(settings, match):
    with pytest.raises(ValueError, match=match):
        CalibrationSession(settings)


# --- evaluate -------------------------------------------------------------------------------


def test_evaluate_keeps_a_full_board_in_pixel_centre_coordinates():
    session = CalibrationSession(CalibrationSettings())
    det = session.evaluate(render_board(BoardSpec(), RVEC, TVEC))
    assert det.kept and det.reason == ''
    np.testing.assert_array_equal(det.ids, np.arange(16))
    assert det.corners.shape == (16, 2) and det.corners.dtype == np.float64
    assert det.image_size == FULL
    truth, _ = cv2.projectPoints(CHESS, RVEC, TVEC, K_TRUE, None)
    # Half a pixel of convention error would fail this by far.
    assert np.abs(det.corners - truth.reshape(-1, 2)[det.ids]).max() < 0.2
    assert det.obj_points.shape == (16, 3) and det.obj_points.dtype == np.float32
    np.testing.assert_array_equal(det.obj_points, CHESS[det.ids])
    assert det.img_points.shape == (16, 2) and det.img_points.dtype == np.float32
    np.testing.assert_allclose(det.img_points, det.corners, atol=1e-4)


def test_charuco_offset_matches_the_opencv_version():
    # OpenCV 4.7-4.13 refine ChArUco corners at p - 0.5 and add the 0.5 back; 4.14 and 5.x no
    # longer do (opencv/opencv#28380). evaluate() takes out whichever offset is measured.
    version = tuple(int(part) for part in cv2.__version__.split('.')[:2])
    assert charuco_offset_px() == (0.5 if (4, 7) <= version < (4, 14) else 0.0)


@pytest.mark.parametrize('shift', [0.5, 0.0])     # the detector of OpenCV 4.7-4.13, of 4.14+
def test_evaluate_is_pixel_centred_under_either_detector_convention(emulate_detector, shift):
    emulate_detector(shift)
    assert charuco_offset_px() == shift
    det = CalibrationSession(CalibrationSettings()).evaluate(render_board(BoardSpec(), RVEC, TVEC))
    assert det.kept
    truth, _ = cv2.projectPoints(CHESS, RVEC, TVEC, K_TRUE, None)
    assert np.abs(det.corners - truth.reshape(-1, 2)[det.ids]).max() < 0.2


@pytest.mark.parametrize('shift, match', [
    (0.25, r'sit \(\+0\.2\d+, \+0\.2\d+\) px off'),             # neither known convention
    ((0.5, 0.0), r'sit \(\+0\.[45]\d+, [+-]0\.0\d+\) px off'),  # the axes disagree
    (None, 'found 0 of the 16 corners'),
])
def test_session_refuses_an_unknown_detector_convention(emulate_detector, shift, match):
    # A wrong correction would bias cx and cy without a trace, so the session must not start.
    emulate_detector(shift)
    with pytest.raises(RuntimeError, match=match):
        CalibrationSession(CalibrationSettings())


def test_evaluate_converts_bgr_to_grey():
    session = CalibrationSession(CalibrationSettings())
    frame = render_board(BoardSpec(), RVEC, TVEC)
    mono = session.evaluate(frame)
    colour = session.evaluate(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    assert colour.kept
    np.testing.assert_array_equal(colour.ids, mono.ids)
    np.testing.assert_allclose(colour.corners, mono.corners, atol=1e-6)


def test_evaluate_without_a_board():
    session = CalibrationSession(CalibrationSettings())
    frame = np.random.default_rng(0).normal(BACKGROUND, 3.0, (FULL[1], FULL[0]))
    det = session.evaluate(np.clip(frame, 0, 255).astype(np.uint8))
    assert not det.kept and det.reason == REJECT_NO_BOARD == 'no board found'
    assert det.corners.shape == (0, 2) and det.ids.shape == (0,)
    assert det.image_size == FULL
    assert det.obj_points is None and det.img_points is None


def test_evaluate_too_few_corners():
    # Only the top two rows of squares: inner corners 0..3; a 5x5 board needs 8.
    det = CalibrationSession(CalibrationSettings()).evaluate(
        render_board(BoardSpec(), RVEC, TVEC, visible=(0, 2, 0, 5)))
    assert not det.kept and det.reason == 'too few corners (4/8)'
    np.testing.assert_array_equal(det.ids, [0, 1, 2, 3])
    assert det.corners.shape == (4, 2)          # still reported, for the review overlay
    assert det.obj_points is None and det.img_points is None
    # The whole board, against a minimum above its 16 corners.
    det = CalibrationSession(CalibrationSettings(min_corners=17)).evaluate(
        render_board(BoardSpec(), RVEC, TVEC))
    assert det.reason == 'too few corners (16/17)'


@pytest.mark.parametrize('visible, ids, kept', [
    ((0, 2, 0, 5), [0, 1, 2, 3], False),        # one row of inner corners
    ((0, 5, 0, 2), [0, 4, 8, 12], False),       # one column
    ((0, 3, 0, 3), [0, 1, 4, 5], True),         # a 2x2 block spans two rows and two columns
])
def test_evaluate_collinear_corners(visible, ids, kept):
    session = CalibrationSession(CalibrationSettings(min_corners=4))
    det = session.evaluate(render_board(BoardSpec(), RVEC, TVEC, visible=visible))
    np.testing.assert_array_equal(det.ids, ids)
    assert det.kept is kept
    assert det.reason == ('' if kept else REJECT_COLLINEAR)
    assert REJECT_COLLINEAR == 'corners in a line'


def test_evaluate_rectangular_board_rows_use_the_column_count():
    # 7x5 squares: 6 inner corners per row, marker ids from 10.
    spec = BoardSpec(squares=(7, 5), first_marker_id=10)
    session = CalibrationSession(CalibrationSettings(board=spec, min_corners=4))
    tvec = np.array([-3.5, -2.5, 14.0])
    one_row = session.evaluate(render_board(spec, RVEC, tvec, visible=(0, 2, 0, 7)))
    np.testing.assert_array_equal(one_row.ids, np.arange(6))
    assert one_row.reason == REJECT_COLLINEAR
    two_rows = session.evaluate(render_board(spec, RVEC, tvec, visible=(0, 3, 0, 7)))
    np.testing.assert_array_equal(two_rows.ids, np.arange(12))
    assert two_rows.kept
    np.testing.assert_array_equal(two_rows.obj_points,
                                  make_board(spec).getChessboardCorners()[:12])


def test_evaluate_records_nothing():
    session = CalibrationSession(CalibrationSettings())
    session.evaluate(render_board(BoardSpec(), RVEC, TVEC))
    session.evaluate(np.full((FULL[1], FULL[0]), BACKGROUND, np.uint8))
    assert (session.kept_count, session.rejected_count) == (0, 0)


# --- submit, undo, counts -------------------------------------------------------------------


def test_submit_undo_and_counts():
    session = CalibrationSession(CalibrationSettings(min_views=3))
    views = synthetic_views(4, seed=7)
    assert session.submit(views[0]) == 0
    assert session.submit(rejected()) is None
    assert session.submit(views[1]) == 1
    assert session.submit(rejected('too few corners (5/8)')) is None
    assert (session.kept_count, session.rejected_count) == (2, 2)
    assert (session.active_count, session.dropped_count) == (2, 0)
    assert not session.ready
    with pytest.raises(RuntimeError):
        session.solve()
    assert session.submit(views[2]) == 2
    assert session.ready
    assert session.undo() == 2
    assert not session.ready and session.kept_count == 2
    assert session.submit(views[3]) == 3        # an index is never reused, even after undo
    assert [session.undo() for _ in range(4)] == [3, 1, 0, None]
    assert (session.kept_count, session.rejected_count, session.active_count) == (0, 2, 0)


def test_submit_rejects_a_second_image_size():
    session = CalibrationSession(CalibrationSettings())
    session.submit(synthetic_views(1, seed=7)[0])
    with pytest.raises(ValueError, match='image size'):
        session.submit(kept_view(CHESS[:, :2] * 50, size=(800, 600)))
    assert session.kept_count == 1


# --- solve ----------------------------------------------------------------------------------


def test_solve_recovers_a_known_camera():
    session, solution = solved_session(synthetic_views(30, seed=1, noise_px=0.02))
    assert_recovers(solution)
    # About 4 sigma at this noise and view count: f to 0.07%, the principal point to 1.5 px.
    tolerance = {'fx': 1.0, 'fy': 1.0, 'cx': 1.5, 'cy': 1.5, 'k1': 3e-3, 'k2': 1.5e-2,
                 'p1': 3e-4, 'p2': 3e-4, 'k3': 2.5e-2}
    estimate = parameters(solution.K, solution.dist)
    truth = parameters(K_TRUE, DIST_TRUE)
    for name in STD_KEYS:
        assert abs(estimate[name] - truth[name]) < tolerance[name], \
            f'{name} = {estimate[name]:.6g}, truth {truth[name]:.6g}'
    assert solution.K.shape == (3, 3) and solution.dist.shape == (5,)
    assert solution.K[0, 1] == solution.K[1, 0] == 0.0
    np.testing.assert_array_equal(solution.K[2], [0.0, 0.0, 1.0])
    assert solution.rms == pytest.approx(0.02 * np.sqrt(2), rel=0.3)   # 0.02 px noise per axis
    assert list(solution.std) == STD_KEYS
    assert solution.n_views == 30 and solution.dropped == []
    assert sorted(solution.per_view_errors) == list(range(30))
    assert solution.image_size == FULL
    assert session.solution is solution
    assert session.history == [{'n_views': 30, 'rms': solution.rms,
                                'fx': solution.K[0, 0], 'fy': solution.K[1, 1],
                                'cx': solution.K[0, 2], 'cy': solution.K[1, 2]}]


def test_uncertainty_shrinks_with_more_views():
    views = synthetic_views(36, seed=3)
    session, few = solved_session(views[:12])
    for view in views[12:]:
        session.submit(view)
    many = session.solve()
    for name in ['fx', 'fy', 'cx', 'cy', 'k1', 'k2']:
        assert many.std[name] < 0.75 * few.std[name], name
    assert_recovers(many)
    assert [entry['n_views'] for entry in session.history] == [12, 36]


def test_outlier_view_is_dropped_and_the_solve_repeated():
    views = synthetic_views(20, seed=8)
    session = CalibrationSession(CalibrationSettings())
    for view in views:
        session.submit(view)
    bad = session.submit(noisy(views[7], 3.0))
    solution = session.solve()
    assert solution.dropped == [bad]
    assert solution.n_views == 20
    assert sorted(solution.per_view_errors) == list(range(20))
    assert solution.rms < 0.1           # the repeated solve is back at the 0.05 px noise level
    assert_recovers(solution)
    assert (session.kept_count, session.active_count, session.dropped_count) == (21, 20, 1)
    assert session.coverage().sum() == 16 * 20      # dropped views leave the coverage map
    again = session.solve()                         # and stay out of later solves
    assert again.dropped == [] and again.n_views == 20


def test_outliers_never_leave_fewer_than_min_views_active():
    views = synthetic_views(13, seed=9)
    session = CalibrationSession(CalibrationSettings(min_views=12))
    for view in views[:11]:
        session.submit(view)
    mild = session.submit(noisy(views[11], 2.0, seed=1))
    severe = session.submit(noisy(views[12], 6.0, seed=2))
    first = session.solve()
    assert first.dropped == [severe]                # 13 active: room to drop the worst only
    assert session.active_count == 12
    second = session.solve()
    errors = second.per_view_errors
    assert errors[mild] > 3.0 * np.median(list(errors.values()))   # still an outlier ...
    assert second.dropped == [] and second.n_views == 12            # ... but min_views holds it


def test_solve_warm_starts_from_the_previous_solution(monkeypatch):
    calls = record_calibrations(monkeypatch)
    views = synthetic_views(16, seed=4)
    session, first = solved_session(views[:14])
    first_K, first_dist = first.K.copy(), first.dist.copy()
    assert len(calls) == 1 and not calls[0]['flags'] & cv2.CALIB_USE_INTRINSIC_GUESS
    for view in views[14:]:
        session.submit(view)
    calls.clear()
    second = session.solve()
    assert calls[0]['flags'] & cv2.CALIB_USE_INTRINSIC_GUESS
    np.testing.assert_array_equal(calls[0]['K0'], first_K)
    np.testing.assert_array_equal(np.ravel(calls[0]['dist0']), first_dist)
    np.testing.assert_array_equal(first.K, first_K)     # the guess was a copy
    assert second.n_views == 16
    assert_recovers(second)


def test_no_warm_start_across_image_sizes(monkeypatch):
    calls = record_calibrations(monkeypatch)
    session, _ = solved_session(synthetic_views(12, seed=5))
    while session.undo() is not None:
        pass
    # Twice the size: the old principal point is inside the new image, so only the size differs.
    double = (3200, 2400)
    k_double = scale_intrinsics(K_TRUE, FULL, double)
    for view in synthetic_views(12, seed=6, k=k_double, size=double):
        session.submit(view)
    calls.clear()
    solution = session.solve()
    assert calls[0]['size'] == double
    assert not calls[0]['flags'] & cv2.CALIB_USE_INTRINSIC_GUESS
    assert solution.image_size == double
    assert_recovers(solution, k=k_double)


def test_no_warm_start_from_a_principal_point_outside_the_image(monkeypatch):
    # OpenCV refuses such a guess outright, so using it would fail every later solve.
    views = synthetic_views(13, seed=15)
    session = CalibrationSession(CalibrationSettings())
    for view in views[:12]:
        session.submit(view)
    real = cv2.calibrateCameraExtended

    def off_image(*args, **kwargs):
        rms, K, *rest = real(*args, **kwargs)
        K = K.copy()
        K[0, 2] = -5.0
        return (rms, K, *rest)

    monkeypatch.setattr(cv2, 'calibrateCameraExtended', off_image)
    assert session.solve().K[0, 2] == -5.0
    monkeypatch.undo()
    calls = record_calibrations(monkeypatch)
    session.submit(views[12])
    solution = session.solve()
    assert not calls[0]['flags'] & cv2.CALIB_USE_INTRINSIC_GUESS
    assert_recovers(solution)


def test_fix_k3_holds_k3_at_zero():
    _, solution = solved_session(synthetic_views(14, seed=10), CalibrationSettings(fix_k3=True))
    assert solution.dist[4] == 0.0
    assert solution.std['k3'] == 0.0
    assert_recovers(solution)


def test_converged_needs_both_focal_and_centre_thresholds():
    views = synthetic_views(14, seed=11)
    _, reference = solved_session(views)
    rel_f = max(reference.std['fx'] / reference.K[0, 0], reference.std['fy'] / reference.K[1, 1])
    sigma_c = max(reference.std['cx'], reference.std['cy'])
    for f_factor, c_factor, expected in [(1.1, 1.1, True), (0.9, 1.1, False), (1.1, 0.9, False)]:
        settings = CalibrationSettings(converge_rel_sigma_f=f_factor * rel_f,
                                       converge_sigma_c_px=c_factor * sigma_c)
        assert solved_session(views, settings)[1].converged is expected


def test_solve_releases_the_lock_while_calibrating(monkeypatch):
    views = synthetic_views(14, seed=12)
    session = CalibrationSession(CalibrationSettings())
    for view in views[:13]:
        session.submit(view)
    real = cv2.calibrateCameraExtended
    during = {}

    def capture_and_undo():
        during['undone'] = session.undo()               # view 12, part of the solve's snapshot
        during['added'] = session.submit(views[13])     # view 13, too late for this solve
        during['kept'] = session.kept_count

    def calibrate_meanwhile(*args, **kwargs):
        if not during:
            worker = threading.Thread(target=capture_and_undo, daemon=True)
            worker.start()
            worker.join(timeout=10.0)
            assert not worker.is_alive(), 'solve() held the session lock while calibrating'
        return real(*args, **kwargs)

    monkeypatch.setattr(cv2, 'calibrateCameraExtended', calibrate_meanwhile)
    solution = session.solve()
    assert during == {'undone': 12, 'added': 13, 'kept': 13}
    assert sorted(solution.per_view_errors) == list(range(13))     # the snapshot
    assert session.kept_count == 13 and session.active_count == 13
    log_views = build_log(session, CalibrationSettings(), {})['views']
    assert [view['index'] for view in log_views] == list(range(12)) + [13]
    assert log_views[-1]['error_px'] is None


# --- coverage -------------------------------------------------------------------------------


def test_coverage_counts_corners_per_cell():
    session = CalibrationSession(CalibrationSettings(coverage_grid=(2, 4)))   # 400 x 600 px cells
    assert session.coverage().shape == (2, 4)
    assert session.coverage_fraction() == 0.0
    session.submit(kept_view([[10, 10], [390, 590], [410, 20], [1590, 1190], [-0.5, 1199.5],
                              [1599.5, -0.5]]))
    session.submit(kept_view([[20, 20], [810, 700], [820, 710], [830, 720]]))
    np.testing.assert_array_equal(session.coverage(), [[3, 1, 0, 1],
                                                       [1, 0, 3, 1]])
    assert session.coverage_fraction() == 6 / 8
    session.submit(rejected())
    assert session.coverage().sum() == 10
    session.undo()
    np.testing.assert_array_equal(session.coverage(), [[2, 1, 0, 1],
                                                       [1, 0, 0, 1]])
    assert session.coverage_fraction() == 5 / 8


# --- scale_intrinsics -----------------------------------------------------------------------


def test_scale_intrinsics_maps_the_frame_centre_to_the_frame_centre():
    np.testing.assert_allclose(scale_intrinsics(K_TRUE, FULL, STREAM),
                               [[560.0, 0.0, 319.5], [0.0, 560.0, 239.5], [0.0, 0.0, 1.0]])


def test_scale_intrinsics_is_exact_against_projection():
    # A uniform resize maps pixel-centre coordinates p -> s * (p + 0.5) - 0.5. Projecting with the
    # scaled K, and the same dist, must land exactly there.
    k_full = np.array([[1402.3, 0.0, 803.7], [0.0, 1398.1, 596.2], [0.0, 0.0, 1.0]])
    points = np.random.default_rng(0).uniform([-2.0, -1.5, 4.0], [2.0, 1.5, 9.0], (200, 3))
    rvec = tvec = np.zeros(3)
    full, _ = cv2.projectPoints(points, rvec, tvec, k_full, DIST_TRUE)
    for size in [STREAM, (800, 600), (3200, 2400)]:
        s = size[0] / FULL[0]
        scaled, _ = cv2.projectPoints(points, rvec, tvec, scale_intrinsics(k_full, FULL, size),
                                      DIST_TRUE)
        np.testing.assert_allclose(scaled.reshape(-1, 2), s * (full.reshape(-1, 2) + 0.5) - 0.5,
                                   rtol=0, atol=1e-9)


@pytest.mark.parametrize('full, target', [((1600, 1300), (640, 480)), ((1600, 1200), (640, 512))])
def test_scale_intrinsics_rejects_a_non_uniform_scale(full, target):
    with pytest.raises(ValueError, match='uniform'):
        scale_intrinsics(K_TRUE, full, target)


# --- config write ---------------------------------------------------------------------------


def test_write_camera_to_cfg_keeps_other_keys_and_backs_up_once(tmp_path):
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text(CFG_TEXT)
    cfg_path.chmod(0o640)
    backup = tmp_path / 'backup' / 'cfg.yaml.orig'
    original = yaml.safe_load(CFG_TEXT)
    k_stream = scale_intrinsics(K_TRUE, FULL, STREAM)
    k_stream[0, 0] += 0.123456789

    write_camera_to_cfg(cfg_path, k_stream, DIST_TRUE, STREAM, backup_path=backup)
    written = yaml.safe_load(cfg_path.read_text())
    assert list(written) == list(original)
    for key in ['MODEL', 'PIPELINE', 'BOARD', 'CALIBRATION']:
        assert written[key] == original[key]
    assert list(written['CAMERA']) == ['K', 'dist', 'image_size']
    assert written['CAMERA']['K'] == k_stream.tolist()           # exact floats, 3x3 nested
    assert written['CAMERA']['dist'] == DIST_TRUE.tolist()
    assert written['CAMERA']['image_size'] == [640, 480]
    assert '  K: [[' in cfg_path.read_text()
    assert backup.read_text() == CFG_TEXT
    assert stat.S_IMODE(cfg_path.stat().st_mode) == 0o640

    write_camera_to_cfg(cfg_path, K_TRUE, np.zeros(5), FULL, backup_path=backup)
    assert backup.read_text() == CFG_TEXT                       # still the very first original
    rewritten = yaml.safe_load(cfg_path.read_text())
    assert rewritten['CAMERA'] == {'K': K_TRUE.tolist(), 'dist': [0.0] * 5,
                                   'image_size': [1600, 1200]}
    assert list(rewritten) == list(original)
    assert sorted(p.name for p in tmp_path.rglob('*')) == ['backup', 'cfg.yaml', 'cfg.yaml.orig']


def test_write_camera_to_cfg_without_camera_block_or_backup(tmp_path):
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text('MODEL:\n  detector: d.onnx\nEXTRA: [1, 2]\n')
    write_camera_to_cfg(cfg_path, K_TRUE, DIST_TRUE, FULL)
    written = yaml.safe_load(cfg_path.read_text())
    assert list(written) == ['MODEL', 'EXTRA', 'CAMERA']
    assert written['EXTRA'] == [1, 2]
    assert written['CAMERA']['image_size'] == [1600, 1200]
    assert sorted(p.name for p in tmp_path.iterdir()) == ['cfg.yaml']


def test_write_camera_to_cfg_through_a_symlink(tmp_path):
    target = tmp_path / 'configs' / 'cfg.yaml'
    target.parent.mkdir()
    target.write_text(CFG_TEXT)
    link = tmp_path / 'cfg.yaml'
    link.symlink_to(target)
    write_camera_to_cfg(link, K_TRUE, DIST_TRUE, FULL, backup_path=tmp_path / 'cfg.yaml.bak')
    assert link.is_symlink()
    assert yaml.safe_load(target.read_text())['CAMERA']['image_size'] == [1600, 1200]
    assert (tmp_path / 'cfg.yaml.bak').read_text() == CFG_TEXT


def test_write_camera_to_cfg_keeps_other_camera_keys_in_place(tmp_path):
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text('CAMERA:\n  frame_id: pi_camera\n  K: null\n  rate_hz: 30\n'
                        'PIPELINE:\n  tau_hm: 0.3\n')
    write_camera_to_cfg(cfg_path, K_TRUE, DIST_TRUE, FULL)
    written = yaml.safe_load(cfg_path.read_text())
    assert list(written) == ['CAMERA', 'PIPELINE']
    assert list(written['CAMERA']) == ['frame_id', 'K', 'rate_hz', 'dist', 'image_size']
    assert written['CAMERA']['frame_id'] == 'pi_camera' and written['CAMERA']['rate_hz'] == 30


# --- log ------------------------------------------------------------------------------------


def test_build_log_reports_the_session(tmp_path):
    views = synthetic_views(14, seed=13)
    settings = CalibrationSettings()
    session = CalibrationSession(settings)
    for view in views:
        session.submit(view)
    bad = session.submit(noisy(views[3], 4.0))
    session.submit(rejected())
    session.submit(rejected('corners in a line'))
    solution = session.solve()
    assert solution.dropped == [bad]
    meta = {'started': '2026-10-05T16:19:00', 'config': '/cfg/cfg.yaml', 'auto_replace': True,
            'config_written': True, 'config_backup': '/cfg/cfg.yaml.bak', 'stream_size': STREAM}

    log = build_log(session, settings, meta)
    assert list(log) == ['session', 'board', 'images', 'counts', 'coverage_fraction', 'solution',
                         'views', 'history']
    assert log['session'] == {'started': '2026-10-05T16:19:00', 'config': '/cfg/cfg.yaml',
                              'auto_replace': True, 'config_written': True,
                              'config_backup': '/cfg/cfg.yaml.bak'}
    assert log['board'] == {'squares': [5, 5], 'square_length_m': 1.0, 'marker_length_m': 0.7,
                            'dictionary': 'DICT_5X5_50', 'first_marker_id': 0,
                            'legacy_pattern': False}
    assert log['images'] == {'full_size': [1600, 1200], 'stream_size': [640, 480]}
    assert log['counts'] == {'kept': 15, 'rejected': 2, 'dropped': 1, 'active': 14}
    assert log['coverage_fraction'] == session.coverage_fraction()
    assert 0.0 < log['coverage_fraction'] <= 1.0

    sol = log['solution']
    assert list(sol) == ['rms_px', 'converged', 'K_full', 'dist', 'std', 'K_stream',
                         'stream_size']
    assert sol['rms_px'] == solution.rms and sol['converged'] == solution.converged
    assert sol['K_full'] == solution.K.tolist()
    assert sol['dist'] == solution.dist.tolist()
    assert list(sol['std']) == STD_KEYS
    assert sol['K_stream'] == scale_intrinsics(solution.K, FULL, STREAM).tolist()
    assert sol['stream_size'] == [640, 480]

    assert [view['index'] for view in log['views']] == list(range(15))
    for view in log['views']:
        assert list(view) == ['index', 'file', 'n_corners', 'error_px', 'dropped']
        assert view['file'] == f'frames/view_{view["index"]:03d}.png'
        assert view['n_corners'] == 16
        assert view['dropped'] == (view['index'] == bad)
        if view['index'] != bad:
            assert view['error_px'] == solution.per_view_errors[view['index']]
    median = np.median(list(solution.per_view_errors.values()))
    assert log['views'][bad]['error_px'] > 3.0 * median     # why it was dropped
    assert log['history'] == [{'n_views': 14, 'rms_px': solution.rms, 'fx': solution.K[0, 0],
                               'fy': solution.K[1, 1], 'cx': solution.K[0, 2],
                               'cy': solution.K[1, 2]}]

    path = tmp_path / 'session' / 'calibration_log.yaml'
    write_log(path, log)
    assert yaml.safe_load(path.read_text()) == log
    write_log(path, build_log(session, settings, {}))              # replaces it
    assert yaml.safe_load(path.read_text())['session']['config'] is None
    assert sorted(p.name for p in path.parent.iterdir()) == ['calibration_log.yaml']


def test_build_log_before_any_solve_and_without_a_stream():
    settings = CalibrationSettings(board=BoardSpec(marker_length=0.6))
    session = CalibrationSession(settings)
    log = build_log(session, settings, None)
    assert log['solution'] is None and log['views'] == [] and log['history'] == []
    assert log['images'] == {'full_size': None, 'stream_size': None}
    assert log['counts'] == {'kept': 0, 'rejected': 0, 'dropped': 0, 'active': 0}
    assert log['coverage_fraction'] == 0.0
    assert log['board']['marker_length_m'] == 0.6
    assert log['session'] == {'started': None, 'config': None, 'auto_replace': False,
                              'config_written': False, 'config_backup': None}
    session.submit(rejected())
    assert build_log(session, settings, {})['images']['full_size'] == [1600, 1200]

    for view in synthetic_views(12, seed=14):
        session.submit(view)
    session.solve()
    solved = build_log(session, settings, {'view_files': {0: 'frames/first.png'}})
    assert solved['solution']['K_stream'] is None and solved['solution']['stream_size'] is None
    assert solved['views'][0]['file'] == 'frames/first.png'
    assert solved['views'][1]['file'] == 'frames/view_001.png'
    # A stream that is not a uniform scale of the full frame gets no K_stream.
    skewed = build_log(session, settings, {'stream_size': [640, 512]})
    assert skewed['solution']['K_stream'] is None
    assert skewed['images']['stream_size'] == [640, 512]
