"""Tests for the configurable calibration board (calib_board.py)."""

from convchart_ros.calib_utils.calib_board import board_spec_from_cfg, BoardSpec, make_board
import numpy as np
import pytest


def test_default_board_is_the_dcc_board():
    dcc_board = pytest.importorskip('dcc.board', reason='needs the Conv-ChArT checkout (dcc)')
    ours = make_board(BoardSpec())
    theirs, _ = dcc_board.get_board()
    np.testing.assert_array_equal(ours.generateImage((500, 500)), theirs.generateImage((500, 500)))
    np.testing.assert_array_equal(np.ravel(ours.getIds()), np.ravel(theirs.getIds()))
    np.testing.assert_allclose(ours.getChessboardCorners(), theirs.getChessboardCorners())


def test_empty_config_gives_the_default_spec():
    assert board_spec_from_cfg(None) == BoardSpec()
    assert board_spec_from_cfg({}) == BoardSpec()
    assert board_spec_from_cfg({'marker_length_m': None, 'squares': None}) == BoardSpec()


def test_config_keys_map_onto_the_spec():
    spec = board_spec_from_cfg({'squares': [11, 8], 'square_length_m': 0.03,
                                'marker_length_m': 0.022, 'dictionary': 'DICT_4X4_100',
                                'first_marker_id': 10, 'legacy_pattern': True})
    assert spec == BoardSpec(squares=(11, 8), square_length=0.03, marker_length=0.022,
                             dictionary='DICT_4X4_100', first_marker_id=10, legacy_pattern=True)
    assert spec.marker_side == 0.022
    assert spec.n_inner_corners == 70


def test_marker_ratio_used_without_marker_length():
    spec = board_spec_from_cfg({'square_length_m': 0.04, 'marker_ratio': 0.75})
    assert spec.marker_side == pytest.approx(0.03)


def test_first_marker_id_offsets_the_ids():
    board = make_board(BoardSpec(first_marker_id=20))
    ids = np.ravel(board.getIds())
    np.testing.assert_array_equal(ids, np.arange(20, 20 + len(ids)))


def test_rectangular_board():
    board = make_board(BoardSpec(squares=(7, 5), square_length=0.03, marker_length=0.02))
    assert len(board.getChessboardCorners()) == 6 * 4


def test_legacy_pattern_flag_is_applied():
    assert make_board(BoardSpec(squares=(6, 6), legacy_pattern=True)).getLegacyPattern()
    assert not make_board(BoardSpec(squares=(6, 6))).getLegacyPattern()


@pytest.mark.parametrize('cfg, match', [
    ({'colour': 'red'}, 'unknown CALIBRATION.board keys'),
    ({'squares': [5]}, r'\[cols, rows\]'),
])
def test_bad_config_is_rejected(cfg, match):
    with pytest.raises(ValueError, match=match):
        board_spec_from_cfg(cfg)


@pytest.mark.parametrize('spec, match', [
    (BoardSpec(squares=(1, 5)), 'at least 2x2'),
    (BoardSpec(marker_length=1.5), 'smaller than the square'),
    (BoardSpec(dictionary='DICT_NOPE'), 'unknown cv2.aruco dictionary'),
    (BoardSpec(first_marker_id=45), 'do not fit'),
])
def test_bad_spec_is_rejected(spec, match):
    with pytest.raises(ValueError, match=match):
        make_board(spec)
