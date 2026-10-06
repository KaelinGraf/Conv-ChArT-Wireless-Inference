"""
Tests for the synthetic ChArUco renderer (calib_synth.py).

OpenCV's ChArUco detector runs on the renders: the corners it finds must sit on the ground truth
that render_view projects with cv2.projectPoints, which checks the render geometry end to end
(board image to board mapping, pose homography, distortion remap). Pose properties are checked
against independent references (scipy rotations, cv2.projectPoints), never the module's helpers.
"""

from convchart_ros.calib_utils.calib_board import BoardSpec, make_board
from convchart_ros.calib_utils.calib_session import charuco_offset_px
from convchart_ros.calib_utils.calib_synth import (
    blank_frame, DEFAULT_DIST, DEFAULT_FULL_SIZE, DEFAULT_K_FULL, DEFAULT_STREAM_SIZE,
    diverse_poses, render_view)
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

K = np.array(DEFAULT_K_FULL)
DIST = np.array(DEFAULT_DIST)
SIZE = DEFAULT_FULL_SIZE
# The same camera at the stream resolution (uniform scale, pixel-centre convention).
K_STREAM = np.array([[560.0, 0.0, 319.5], [0.0, 560.0, 239.5], [0.0, 0.0, 1.0]])
SPEC = BoardSpec()
OTHER_BOARDS = [
    BoardSpec(squares=(7, 5), square_length=0.03, marker_length=0.022),
    BoardSpec(squares=(11, 8), square_length=0.03, marker_length=0.022,
              dictionary='DICT_4X4_100', first_marker_id=10),
    BoardSpec(squares=(6, 6), legacy_pattern=True),
]
# How far the installed cv2.aruco.CharucoDetector reports corners off the pixel-centre convention
# of cv2.projectPoints: 0.5 px on OpenCV 4.7-4.13, 0 from 4.14.
DETECTOR_SHIFT = charuco_offset_px()


@pytest.fixture(scope='module')
def poses():
    return diverse_poses(SPEC, K, SIZE)


def detect(board, frame):
    """Detect ChArUco corners; return their ids and pixel-centre coordinates."""
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(frame)
    if ids is None:
        return np.empty(0, dtype=int), np.empty((0, 2))
    return ids.ravel(), corners.reshape(-1, 2).astype(np.float64) - DETECTOR_SHIFT


def detection_errors(spec, pose, dist=DIST, seed=0):
    """Render a pose, detect it; return (ids found, distance of each to its ground truth in px)."""
    board = make_board(spec)
    frame, truth, _ = render_view(board, spec, K, dist, *pose, SIZE, seed=seed)
    ids, found = detect(board, frame)
    return ids, np.linalg.norm(found - truth[ids], axis=1)


def chessboard_corners(spec):
    return np.asarray(make_board(spec).getChessboardCorners(), dtype=np.float64)


def board_outline(spec, per_edge=25):
    """Sample the board's outer edge (lens distortion bends it), in board coordinates."""
    width, height = np.array(spec.squares) * spec.square_length
    t = np.linspace(0.0, 1.0, per_edge, endpoint=False)
    zero, one = np.zeros_like(t), np.ones_like(t)
    edges = [(t, zero), (one, t), (1 - t, one), (zero, 1 - t)]
    xy = np.concatenate([np.column_stack([x * width, y * height]) for x, y in edges])
    return np.column_stack([xy, np.zeros(len(xy))])


def board_centre(spec):
    return np.append(np.array(spec.squares) * spec.square_length / 2, 0.0)


def project(points, pose, k=K, dist=None):
    return cv2.projectPoints(points, pose[0], pose[1], k, dist)[0].reshape(-1, 2)


def test_charuco_detector_reports_corners_half_a_pixel_off_pixel_centres():
    # The shift detect() undoes: an axis-aligned board pasted at (100, 100) has its first inner
    # corner on the boundary between pixels 179 and 180, which is x = y = 179.5 pixel-centred.
    board = make_board(SPEC)
    canvas = np.full((600, 600), 140, np.uint8)
    canvas[100:500, 100:500] = board.generateImage((400, 400))
    ids, found = detect(board, cv2.GaussianBlur(canvas, (0, 0), 1.0))
    assert len(ids) == SPEC.n_inner_corners
    expected = 100 + 80 * (1 + np.column_stack([ids % 4, ids // 4])) - 0.5
    np.testing.assert_allclose(found, expected, atol=0.05)


def test_render_view_returns_the_frame_and_its_projected_corners(poses):
    board = make_board(SPEC)
    frame, corners, ids = render_view(board, SPEC, K, DIST, *poses[0], SIZE)
    assert frame.shape == (SIZE[1], SIZE[0])
    assert frame.dtype == np.uint8
    assert corners.shape == (SPEC.n_inner_corners, 2)
    assert corners.dtype == np.float64
    np.testing.assert_array_equal(ids, np.arange(SPEC.n_inner_corners))
    np.testing.assert_allclose(corners, project(chessboard_corners(SPEC), poses[0], dist=DIST))


@pytest.mark.parametrize('index', [0, 2, 4, 6, 8, 13, 19, 23])
def test_detected_corners_sit_on_the_projected_ground_truth(poses, index):
    ids, errors = detection_errors(SPEC, poses[index], seed=index)
    assert len(ids) == SPEC.n_inner_corners
    assert np.median(errors) < 0.3, f'pose {index}: median error {np.median(errors):.3f} px'
    assert errors.max() < 1.0, f'pose {index}: max error {errors.max():.3f} px'


def test_distortion_is_rendered_not_only_projected(poses):
    # Pose 9 sits in the top-left cell, far: the lens moves its corners by many pixels, so a render
    # that skipped or inverted the distortion would miss the ground truth by about as much.
    pose = poses[9]
    _, truth, _ = render_view(make_board(SPEC), SPEC, K, DIST, *pose, SIZE)
    pinhole = project(chessboard_corners(SPEC), pose)
    assert np.median(np.linalg.norm(truth - pinhole, axis=1)) > 5.0
    ids, errors = detection_errors(SPEC, pose)
    assert len(ids) == SPEC.n_inner_corners
    assert np.median(errors) < 0.3


def test_zero_distortion_renders_the_pinhole_view(poses):
    _, truth, _ = render_view(make_board(SPEC), SPEC, K, np.zeros(5), *poses[4], SIZE)
    np.testing.assert_allclose(truth, project(chessboard_corners(SPEC), poses[4]))
    ids, errors = detection_errors(SPEC, poses[4], dist=np.zeros(5))
    assert len(ids) == SPEC.n_inner_corners
    assert np.median(errors) < 0.3


def test_renders_are_deterministic_for_a_seed(poses):
    board = make_board(SPEC)
    first, corners, _ = render_view(board, SPEC, K, DIST, *poses[4], SIZE, seed=3)
    again, _, _ = render_view(board, SPEC, K, DIST, *poses[4], SIZE, seed=3)
    other, other_corners, _ = render_view(board, SPEC, K, DIST, *poses[4], SIZE, seed=4)
    np.testing.assert_array_equal(first, again)
    np.testing.assert_array_equal(corners, other_corners)
    # Pose 4 is centred, so the top-left corner of the frame is plain background: there the two
    # seeds differ by independent noise of noise_std = 2 grey levels each.
    difference = first[:100, :100].astype(float) - other[:100, :100]
    assert 2.5 < difference.std() < 3.2


def test_noise_free_render_shows_the_board_on_the_background(poses):
    frame, _, _ = render_view(make_board(SPEC), SPEC, K, DIST, *poses[4], SIZE, background=90,
                              noise_std=0.0, blur_sigma=0.0)
    assert frame[0, 0] == 90
    assert frame[-1, -1] == 90
    assert frame.min() == 0
    assert frame.max() == 255


def test_blank_frame_is_board_free_noise_on_the_background():
    frame = blank_frame(SIZE, seed=5)
    assert frame.shape == (SIZE[1], SIZE[0])
    assert frame.dtype == np.uint8
    np.testing.assert_array_equal(frame, blank_frame(SIZE, seed=5))
    assert np.any(frame != blank_frame(SIZE, seed=6))
    assert abs(frame.mean() - 140) < 0.5
    assert 1.5 < frame.std() < 2.5
    ids, _ = detect(make_board(SPEC), frame)
    assert len(ids) == 0


@pytest.mark.parametrize('spec', OTHER_BOARDS, ids=['7x5', '11x8_4x4_from_10', '6x6_legacy'])
@pytest.mark.parametrize('index', [1, 4])
def test_other_boards_render_and_detect(spec, index):
    pose = diverse_poses(spec, K, SIZE)[index]
    ids, errors = detection_errors(spec, pose, seed=index)
    assert len(ids) >= 0.9 * spec.n_inner_corners
    assert np.median(errors) < 0.3, f'median error {np.median(errors):.3f} px'


def test_render_view_rejects_a_board_of_another_spec(poses):
    with pytest.raises(ValueError, match='squares'):
        render_view(make_board(BoardSpec(squares=(7, 5))), SPEC, K, DIST, *poses[0], SIZE)


def test_diverse_poses_are_deterministic(poses):
    longer = diverse_poses(SPEC, DEFAULT_K_FULL, SIZE, n=30)
    assert len(poses) == 24
    assert len(longer) == 30
    for (rvec, tvec), (rvec_again, tvec_again) in zip(poses, longer):
        assert rvec.shape == (3,)
        assert tvec.shape == (3,)
        np.testing.assert_array_equal(rvec, rvec_again)
        np.testing.assert_array_equal(tvec, tvec_again)


@pytest.mark.parametrize('spec, k, size', [
    (SPEC, K, SIZE),
    (OTHER_BOARDS[0], K, SIZE),
    (OTHER_BOARDS[1], K, SIZE),
    (SPEC, K_STREAM, DEFAULT_STREAM_SIZE),
], ids=['5x5', '7x5', '11x8', '5x5_stream'])
def test_every_diverse_pose_keeps_the_whole_board_in_the_frame(spec, k, size):
    outline = board_outline(spec)
    margin = 0.02 * min(size)
    for index, pose in enumerate(diverse_poses(spec, k, size)):
        depth = (Rotation.from_rotvec(pose[0]).apply(outline) + pose[1])[:, 2]
        assert np.all(depth > 0), f'pose {index} is behind the camera'
        for dist in (None, DIST):
            uv = project(outline, pose, k, dist)
            assert np.all(uv >= margin), f'pose {index} leaves the frame (dist={dist})'
            assert np.all(uv <= np.array(size) - 1 - margin), \
                f'pose {index} leaves the frame (dist={dist})'


def test_diverse_poses_cover_the_frame_near_and_far(poses):
    cells, distances = set(), []
    for pose in poses:
        u, v = project(board_centre(SPEC)[None], pose)[0]
        cells.add((int(3 * u / SIZE[0]), int(3 * v / SIZE[1])))
        distances.append(np.linalg.norm(Rotation.from_rotvec(pose[0]).apply(board_centre(SPEC))
                                        + pose[1]))
    assert cells == {(col, row) for col in range(3) for row in range(3)}
    assert max(distances) > 1.5 * min(distances)


def test_diverse_poses_tilt_about_both_board_axes_and_roll_in_plane(poses):
    # Relative to the board facing the camera along its line of sight (the shortest-arc rotation of
    # the optical axis onto it), each pose is R = Rz(roll) Rx(tilt_x) Ry(tilt_y): scipy's 'ZXY'.
    angles = []
    for rvec, tvec in poses:
        rotation = Rotation.from_rotvec(rvec)
        sight = rotation.apply(board_centre(SPEC)) + tvec
        sight /= np.linalg.norm(sight)
        axis = np.cross([0.0, 0.0, 1.0], sight)
        facing = Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.arccos(sight[2]))
        angles.append((facing.inv() * rotation).as_euler('ZXY', degrees=True))
    roll, tilt_x, tilt_y = np.array(angles).T
    for tilt in (tilt_x, tilt_y):
        assert np.all(np.abs(tilt) >= 15.0 - 1e-6)
        assert np.all(np.abs(tilt) <= 40.0 + 1e-6)
        assert np.any(tilt > 0)
        assert np.any(tilt < 0)
    assert np.all(np.abs(roll) <= 20.0 + 1e-6)
    assert np.ptp(roll) > 20.0
