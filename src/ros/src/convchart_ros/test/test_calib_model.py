"""
Tests for the model as the calibration's corner detector and distance readout (calib_model.py).

The real inference pipeline runs on the GPU with the repo's MODEL block, so these are skipped
without onnxruntime or the checkpoints (not in git). The boards are calib_synth renders from a
camera like the Pi's: K is the repo's stream K scaled to 1600x1200 (fx about 892), distortion
k1 0.05, k2 -0.10. The detector's config deliberately holds a stale CAMERA block (another K, the
opposite k1), which corners() must not depend on. No ROS.

    python3 -m pytest src/ros/src/convchart_ros/test/test_calib_model.py -q
"""

from pathlib import Path

from convchart_ros.calib_utils.calib_board import BoardSpec, make_board
from convchart_ros.calib_utils.calib_session import (
    CalibrationSession, CalibrationSettings, charuco_offset_px, REJECT_NO_BOARD, scale_intrinsics)
from convchart_ros.calib_utils.calib_synth import blank_frame, diverse_poses, render_view
from convchart_ros.launch_plan import find_repo
import cv2
import numpy as np
import pytest
import yaml

pytest.importorskip('onnxruntime')
calib_model = pytest.importorskip('convchart_ros.calib_utils.calib_model')

REPO = find_repo(Path(__file__))
FULL = (1600, 1200)
STREAM = (640, 480)
K_STREAM = np.array([[356.66, 0.0, 327.55], [0.0, 356.63, 234.19], [0.0, 0.0, 1.0]])
K_FULL = scale_intrinsics(K_STREAM, STREAM, FULL)
DIST = np.array([0.0535, -0.1003, 0.0003, 0.0001, 0.0317])
SQUARE_M = 0.0405
SPEC = BoardSpec()
BOARD = make_board(SPEC)


def model_files():
    if REPO is None:
        return None
    model = (yaml.safe_load((REPO / 'cfg' / 'cfg.yaml').read_text()) or {}).get('MODEL') or {}
    files = {key: REPO / str(model.get(key, '')) for key in ('detector', 'refiner')}
    return files if all(path.is_file() for path in files.values()) else None


@pytest.fixture(scope='module')
def detector(tmp_path_factory):
    files = model_files()
    if files is None:
        pytest.skip("needs the repo's MODEL checkpoints")
    cfg = {'MODEL': {key: str(path) for key, path in files.items()},
           'BOARD': {'squares': [5, 5], 'square_length_m': SQUARE_M},
           'CAMERA': {'K': K_STREAM.tolist(), 'dist': [-0.05, 0.1, 0.0, 0.0, 0.0],
                      'image_size': list(STREAM)}}
    path = tmp_path_factory.mktemp('model') / 'cfg' / 'cfg.yaml'
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(cfg))
    return calib_model.ModelDetector(path)


@pytest.fixture(scope='module')
def stills():
    """Full-resolution renders over diverse poses: [(frame, true corners, true ids)]."""
    poses = diverse_poses(SPEC, K_FULL, FULL, n=16)
    return [render_view(BOARD, SPEC, K_FULL, DIST, rvec, tvec, FULL, seed=i)
            for i, (rvec, tvec) in enumerate(poses)]


def test_model_ids_are_opencv_charuco_ids_and_its_corners_are_pixel_centred(detector, stills):
    found, errors = 0, []
    for frame, truth, truth_ids in stills:
        corners, ids = detector.corners(frame)
        assert list(ids) == sorted(set(ids)), 'sorted, each id once'
        errors.append(corners - truth[ids])
        found += len(ids)
        ocv, ocv_ids, _, _ = cv2.aruco.CharucoDetector(BOARD).detectBoard(frame)
        common, mine, theirs = np.intersect1d(ids, ocv_ids.ravel(), return_indices=True)
        assert len(common) >= 12
        theirs_xy = ocv.reshape(-1, 2)[theirs] - charuco_offset_px()
        assert np.abs(corners[mine] - theirs_xy).max() < 0.5
    errors = np.concatenate(errors)
    assert found >= 0.97 * 16 * len(stills), f'{found} corners found'
    assert np.abs(errors.mean(axis=0)).max() < 0.05, 'no offset: pixel-centre convention'
    assert np.median(np.linalg.norm(errors, axis=1)) < 0.1
    assert np.linalg.norm(errors, axis=1).max() < 0.5


def test_calibrating_on_the_models_corners_recovers_the_camera(detector, stills):
    session = CalibrationSession(CalibrationSettings(), detect=detector.corners)
    for frame, _, _ in stills:
        assert session.submit(session.evaluate(frame)) is not None
    solution = session.solve()
    K = solution.K
    assert solution.rms < 0.15
    np.testing.assert_allclose([K[0, 0], K[1, 1]], [K_FULL[0, 0], K_FULL[1, 1]], rtol=1e-3)
    np.testing.assert_allclose([K[0, 2], K[1, 2]], [K_FULL[0, 2], K_FULL[1, 2]], atol=1.0)
    np.testing.assert_allclose(solution.dist[:2], DIST[:2], atol=0.01)


def test_a_frame_without_a_board_has_no_corners_and_no_range(detector):
    corners, ids = detector.corners(blank_frame(FULL))
    assert corners.shape == (0, 2) and ids.shape == (0,)
    session = CalibrationSession(CalibrationSettings(), detect=detector.corners)
    assert session.evaluate(blank_frame(FULL)).reason == REJECT_NO_BOARD
    assert detector.board_range(blank_frame(STREAM), K_STREAM, DIST) is None


@pytest.mark.parametrize('z', [0.19, 0.5, 0.8, 1.2, 1.5])
@pytest.mark.parametrize('tilt_deg, offset', [(0.0, (0.0, 0.0)), (25.0, (0.15, -0.1))])
def test_board_range_reads_the_board_centres_distance(detector, z, tilt_deg, offset):
    spec = BoardSpec(square_length=SQUARE_M)
    rotation = cv2.Rodrigues(np.radians([tilt_deg, -0.6 * tilt_deg, 0.0]))[0]
    centre = np.array([offset[0] * z, offset[1] * z, z])
    tvec = centre - rotation @ np.array([2.5 * SQUARE_M, 2.5 * SQUARE_M, 0.0])
    frame, _, _ = render_view(make_board(spec), spec, K_STREAM, DIST, cv2.Rodrigues(rotation)[0],
                              tvec, STREAM, seed=1)
    reading = detector.board_range(frame, K_STREAM, DIST)
    assert reading is not None
    assert reading.z == pytest.approx(z, rel=0.02)
    assert reading.range == pytest.approx(np.linalg.norm(centre), rel=0.02)


@pytest.mark.parametrize('spec', [BoardSpec(), BoardSpec(square_length=SQUARE_M),
                                  BoardSpec(square_length=0.024, marker_length=0.0168)])
def test_the_models_board_at_any_square_size_is_accepted(spec):
    calib_model.check_model_board(spec)


@pytest.mark.parametrize('spec, named', [
    (BoardSpec(squares=(7, 5)), 'squares [7, 5]'),
    (BoardSpec(marker_ratio=0.75), 'markers 0.75 of a square'),
    (BoardSpec(dictionary='DICT_4X4_50'), 'dictionary DICT_4X4_50'),
    (BoardSpec(first_marker_id=10), 'first marker id 10'),
    (BoardSpec(legacy_pattern=True), 'legacy pattern'),
])
def test_any_other_board_is_refused_with_a_pointer_to_opencv(spec, named):
    with pytest.raises(ValueError, match='use detector:=opencv') as error:
        calib_model.check_model_board(spec)
    assert named in str(error.value)
