"""
Tests for the calibration window canvas (calib_view.py): pure rendering, no ROS and no window.

Frames are uniform grey (noise where only the shape matters), so whatever the overlays draw stands
out. scale_points is checked against the pixel-centre geometry (image edges map onto image edges),
coverage cells against the 80 px grid of a 640x480 canvas, and the review markers against
scale_points.
"""

import dataclasses

from convchart_ros.calib_utils.calib_view import (
    compose_live, compose_review, HudState, scale_points)
import cv2
import numpy as np
import pytest

FULL = (1600, 1200)                     # capture size (W, H)
STREAM = (640, 480)                     # stream size and default canvas
GRID = (6, 8)                           # coverage (rows, cols): 80 px cells on a 640x480 canvas
CELL = 80
GREY = 128
MIDDLE = slice(100, 440)                # canvas rows clear of the HUD bands and verdict banner
GREEN, RED = 1, 2                       # BGR channel indices
LONG_STATUS = ('rms 0.214 px | fx 1402.31 +- 0.93 | fy 1401.87 +- 0.91 | cx 799.12 +- 1.21 | '
               'cy 600.20 +- 1.13 | converged')


def make_hud(**overrides):
    fields = {'kept': 3, 'rejected': 1, 'min_views': 12, 'status': 'collecting 3/12',
              'coverage_fraction': 0.25}
    fields.update(overrides)
    return HudState(**fields)


def grey(size, value=GREY, channels=None):
    w, h = size
    return np.full((h, w) if channels is None else (h, w, channels), value, np.uint8)


def noise(size, channels=None, seed=0):
    w, h = size
    shape = (h, w) if channels is None else (h, w, channels)
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def excess(canvas, channel):
    """Return how far one channel exceeds both others, per pixel: 0 on grey, black or white."""
    planes = canvas.astype(int)
    others = np.delete(planes, channel, axis=2).max(axis=2)
    return np.clip(planes[..., channel] - others, 0, None)


def centroid(weight, x, y, half=10):
    """Return the weighted centroid of weight in a (2 * half + 1) px window around (x, y)."""
    x0, y0 = int(round(x)) - half, int(round(y)) - half
    window = weight[y0:y0 + 2 * half + 1, x0:x0 + 2 * half + 1].astype(float)
    ys, xs = np.indices(window.shape)
    return x0 + (xs * window).sum() / window.sum(), y0 + (ys * window).sum() / window.sum()


def band_rows(canvas):
    """Return the row ranges of the top and bottom HUD bands of a canvas over a bright frame."""
    dark = np.median(canvas, axis=(1, 2)) < 100
    top_end = int(np.argmin(dark))
    bottom_start = len(dark) - int(np.argmin(dark[::-1]))
    return slice(0, top_end), slice(bottom_start, len(dark))


def test_hud_state_fields_and_defaults():
    names = [field.name for field in dataclasses.fields(HudState)]
    assert names == ['kept', 'rejected', 'min_views', 'status', 'coverage_fraction', 'converged',
                     'verdict', 'verdict_ok', 'message', 'help']
    hud = HudState(5, 2, 12, 'collecting 5/12')
    defaults = (hud.coverage_fraction, hud.converged, hud.verdict, hud.verdict_ok, hud.message)
    assert defaults == (None, False, None, None, None)
    assert hud.help == 'SPACE capture   U undo   M coverage   Q quit'


@pytest.mark.parametrize('canvas_size', [STREAM, (800, 600), (320, 240)])
@pytest.mark.parametrize('frame_size', [STREAM, FULL, (800, 600)])
@pytest.mark.parametrize('channels', [None, 1, 3])
def test_every_canvas_has_the_canvas_size(frame_size, channels, canvas_size):
    frame = noise(frame_size, channels)
    corners = np.array([[0.3, 0.4], [0.6, 0.5]]) * frame_size
    hud = make_hud(status=LONG_STATUS, message='capture timed out')
    kept = make_hud(verdict='KEPT', verdict_ok=True)
    rejected = make_hud(verdict='REJECTED: corners in a line', verdict_ok=False)
    canvases = [
        compose_live(frame, hud, np.ones(GRID, int), canvas_size),
        compose_live(frame, hud, None, canvas_size, show_coverage=False),
        compose_review(frame, corners, True, kept, canvas_size),
        compose_review(frame, corners, False, rejected, canvas_size),
    ]
    w, h = canvas_size
    for canvas in canvases:
        assert canvas.shape == (h, w, 3)
        assert canvas.dtype == np.uint8


def test_default_canvas_is_the_stream_size():
    assert compose_live(grey(FULL), make_hud()).shape == (480, 640, 3)
    assert compose_review(grey(FULL), [], True, make_hud()).shape == (480, 640, 3)


def test_mono_becomes_grey_bgr_and_bgr_keeps_its_channel_order():
    assert (compose_live(grey(STREAM, 77), make_hud())[MIDDLE] == 77).all()
    bgr = np.empty((480, 640, 3), np.uint8)
    bgr[...] = (10, 120, 230)
    assert (compose_live(bgr, make_hud())[MIDDLE] == (10, 120, 230)).all()


def test_review_resizes_the_capture_with_inter_area():
    frame = noise(FULL)
    canvas = compose_review(frame, [], True, make_hud())
    expected = cv2.resize(frame, STREAM, interpolation=cv2.INTER_AREA)
    for channel in range(3):
        np.testing.assert_array_equal(canvas[MIDDLE, :, channel], expected[MIDDLE])


def test_inputs_are_not_modified():
    frame = noise(STREAM, 3)            # already canvas-sized BGR: the path that needs no resize
    corners = np.array([[320.25, 240.75]])
    coverage = np.full(GRID, 2)
    originals = [frame.copy(), corners.copy(), coverage.copy()]
    compose_live(frame, make_hud(message='capture timed out'), coverage)
    compose_review(frame, corners, False, make_hud())
    for array, original in zip([frame, corners, coverage], originals):
        np.testing.assert_array_equal(array, original)


def test_scale_points_maps_pixel_centres_and_edges():
    # Pixel-centre convention: a W x H image spans [-0.5, W - 0.5] x [-0.5, H - 0.5].
    full = [[-0.5, -0.5], [1599.5, 1199.5], [799.5, 599.5], [0.0, 0.0], [1599.0, 1199.0]]
    expected = [[-0.5, -0.5], [639.5, 479.5], [319.5, 239.5], [-0.3, -0.3], [639.3, 479.3]]
    np.testing.assert_allclose(scale_points(full, FULL, STREAM), expected, rtol=0, atol=1e-12)


def test_scale_points_scales_each_axis_on_its_own():
    got = scale_points([[99.5, 99.5], [-0.5, 1199.5]], FULL, (800, 300))
    np.testing.assert_allclose(got, [[49.5, 24.5], [-0.5, 299.5]], rtol=0, atol=1e-12)


def test_scale_points_upscaling_lands_on_the_block_centre():
    # pixel (0, 0) of a 160x120 image becomes the 4x4 block of canvas pixels 0..3, centred on 1.5
    got = scale_points([[0.0, 0.0], [159.0, 119.0]], (160, 120), STREAM)
    np.testing.assert_allclose(got, [[1.5, 1.5], [637.5, 477.5]], rtol=0, atol=1e-12)


def test_scale_points_round_trip():
    pts = np.random.default_rng(3).uniform([-0.5, -0.5], [1599.5, 1199.5], (500, 2))
    back = scale_points(scale_points(pts, FULL, STREAM), STREAM, FULL)
    np.testing.assert_allclose(back, pts, rtol=0, atol=1e-9)


def test_scale_points_keeps_the_opencv_point_layout():
    pts = np.array([[[10.0, 20.0]], [[30.0, 40.0]]])            # (N, 1, 2), as cv2 returns
    out = scale_points(pts, FULL, STREAM)
    assert out.shape == (2, 1, 2)
    np.testing.assert_allclose(out[:, 0], scale_points(pts[:, 0], FULL, STREAM))


@pytest.mark.parametrize('points', [[], np.empty((0, 2))])
def test_scale_points_of_nothing_is_empty(points):
    assert scale_points(points, FULL, STREAM).shape == (0, 2)


@pytest.mark.parametrize('src, dst', [((0, 1200), STREAM), (FULL, (640, -480))])
def test_scale_points_rejects_degenerate_sizes(src, dst):
    with pytest.raises(ValueError, match='positive'):
        scale_points([[1.0, 2.0]], src, dst)


@pytest.mark.parametrize('frame_size', [FULL, (160, 120)])     # shrunk capture, grown small frame
@pytest.mark.parametrize('kept', [True, False])
def test_review_corners_are_drawn_where_scale_points_puts_them(frame_size, kept):
    corners = np.array([[0.437, 0.471], [0.571, 0.533]]) * frame_size
    canvas = compose_review(grey(frame_size), corners, kept, make_hud())
    expected = scale_points(corners, frame_size, STREAM)
    weight = excess(canvas, GREEN if kept else RED)
    for x, y in expected:
        b, g, r = canvas[int(round(y)), int(round(x))].astype(int)
        if kept:
            assert g > 200 and max(b, r) < 60, f'no green marker at ({x:.2f}, {y:.2f})'
        else:
            assert r > 200 and max(b, g) < 60, f'no red marker at ({x:.2f}, {y:.2f})'
        # centred to a fraction of a pixel; a convention slip (missing half-pixel shift,
        # truncation) would move it by 0.25-1.5 px here
        cx, cy = centroid(weight, x, y)
        assert abs(cx - x) < 0.15 and abs(cy - y) < 0.15, (cx, cy, x, y)
    ys, xs = np.nonzero(weight[MIDDLE])
    distance = np.hypot(xs[:, None] - expected[:, 0], ys[:, None] + MIDDLE.start - expected[:, 1])
    assert distance.min(axis=1).max() < 8, 'marker colour away from every corner'


@pytest.mark.parametrize('kept', [True, False])
def test_review_markers_are_drawn_over_the_hud(kept):
    # corners near the top or bottom edge lie under the HUD bands and the verdict banner; their
    # markers must look the same there as in the clear middle, over shade, banner and text alike
    verdict = 'KEPT' if kept else 'REJECTED: too few corners (5/8)'
    hud = make_hud(kept=14, status=LONG_STATUS, converged=True, verdict=verdict, verdict_ok=kept)
    bare = compose_review(grey(FULL), [], kept, hud)
    shaded = np.flatnonzero((bare != GREY).any(axis=2).mean(axis=1) > 0.9)  # rows under the HUD
    assert shaded[0] == 0 and shaded[-1] == 479, 'expected HUD bands along the top and bottom'
    lattice = np.stack(np.meshgrid(np.arange(8, 640, 40), shaded[2::12]), axis=-1).reshape(-1, 2)
    points = np.vstack([lattice, [320, 240]]) + 0.3        # the last one in the clear middle
    canvas = compose_review(grey(FULL), scale_points(points, STREAM, FULL), kept, hud)
    x, y = np.rint(points).astype(int).T
    hidden = np.any(canvas[y, x] != canvas[y[-1], x[-1]], axis=1)
    assert not hidden.any(), f'{hidden.sum()} markers dimmed or covered: {points[hidden][:4]}'
    # and the HUD stays drawn around them: only the marker discs differ from the bare canvas
    ys, xs = np.nonzero(np.any(canvas != bare, axis=2))
    distance = np.hypot(xs[:, None] - points[:, 0], ys[:, None] - points[:, 1])
    assert distance.min(axis=1).max() < 6, 'canvas changed away from every corner'


@pytest.mark.parametrize('corners', [np.empty((0, 2)), [], None])
def test_review_without_corners(corners):
    hud = make_hud(verdict='REJECTED: no board found', verdict_ok=False)
    canvas = compose_review(grey(FULL), corners, False, hud)
    assert canvas.shape == (480, 640, 3)
    assert (canvas[MIDDLE] == GREY).all()


@pytest.mark.parametrize('kept, verdict, verdict_ok', [
    (True, 'KEPT', True),
    (False, 'REJECTED: too few corners (5/8)', False),
    (True, None, None),                 # no verdict in the HUD: banner from kept
    (False, None, None),
])
def test_review_shows_a_verdict_banner(kept, verdict, verdict_ok):
    hud = make_hud(verdict=verdict, verdict_ok=verdict_ok)
    canvas = compose_review(grey(FULL), [], kept, hud)
    tint = excess(canvas, GREEN if kept else RED) > 40
    assert tint.sum() > 20 * 640, 'expected a full-width banner at least 20 px tall'
    assert np.flatnonzero(tint.any(axis=1)).max() < MIDDLE.start
    assert not (excess(canvas, RED if kept else GREEN) > 40).any()
    live = compose_live(grey(FULL), hud)                # live mode never shows a verdict
    assert not (excess(live, GREEN) > 40).any() and not (excess(live, RED) > 40).any()


def test_review_banner_shows_the_verdict_text():
    # same kept and verdict_ok, two reasons: only the text inside the banner may differ
    def review(verdict):
        return compose_review(grey(FULL), [], False, make_hud(verdict=verdict, verdict_ok=False))

    no_board = review('REJECTED: no board found')
    too_few = review('REJECTED: too few corners (5/8)')
    rows = np.flatnonzero(np.any(no_board != too_few, axis=(1, 2)))
    banner = np.flatnonzero((excess(no_board, RED) > 40).any(axis=1))
    assert rows.size, 'the banner shows the reason, not only REJECTED'
    assert banner.min() <= rows.min() and rows.max() <= banner.max(), 'only the banner changes'


@pytest.mark.parametrize('kept, default', [(True, 'KEPT'), (False, 'REJECTED')])
def test_review_banner_without_a_verdict_follows_kept(kept, default):
    def review(verdict):
        return compose_review(grey(FULL), [], kept, make_hud(verdict=verdict))

    np.testing.assert_array_equal(review(None), review(default))


def test_coverage_fill_changes_only_covered_cells():
    frame, hud = grey(STREAM), make_hud()
    empty = np.zeros(GRID, int)
    coverage = empty.copy()
    coverage[2, 3] = 1
    coverage[3, 6] = 5
    changed = np.any(compose_live(frame, hud, coverage) != compose_live(frame, hud, empty), axis=2)
    covered = np.zeros(changed.shape, bool)
    for r, c in [(2, 3), (3, 6)]:
        covered[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = True
        # the whole cell turns green except the grid lines on its top and left edges
        assert changed[r * CELL + 1:(r + 1) * CELL, c * CELL + 1:(c + 1) * CELL].all()
    assert not (changed & ~covered).any()


def test_coverage_fill_strengthens_with_count_and_saturates_at_3():
    coverage = np.zeros(GRID, int)
    coverage[3, [0, 2, 4, 6]] = [1, 2, 3, 9]
    green = excess(compose_live(grey(STREAM), make_hud(), coverage), GREEN)

    def strength(c):                    # mean green excess inside cell (3, c), off its grid lines
        return green[3 * CELL + 5:4 * CELL - 5, c * CELL + 5:(c + 1) * CELL - 5].mean()

    one, two, three, nine = (strength(c) for c in (0, 2, 4, 6))
    assert 0 < one < two < three
    assert nine == three
    assert strength(1) == 0


def test_empty_cells_get_only_grid_lines():
    frame, hud = grey(STREAM), make_hud()
    changed = np.any(compose_live(frame, hud, np.zeros(GRID, int)) != compose_live(frame, hud),
                     axis=2)
    cols = np.flatnonzero(changed[MIDDLE].all(axis=0))
    rows = np.flatnonzero(changed.all(axis=1))
    assert len(cols) == 7 and np.abs(cols - CELL * np.arange(1, 8)).max() <= 1
    assert len(rows) == 5 and np.abs(rows - CELL * np.arange(1, 6)).max() <= 1
    lines = np.zeros(changed.shape, bool)
    lines[:, cols] = True
    lines[rows, :] = True
    assert not (changed & ~lines).any()


def test_coverage_grid_spans_the_whole_canvas():
    # 5x7 cells do not divide 640x480 evenly; every pixel still belongs to a cell
    green = excess(compose_live(grey(STREAM), make_hud(), np.full((5, 7), 3)), GREEN)[MIDDLE] > 0
    assert green.mean() > 0.97                          # all but the grid lines
    assert green[:, [0, -1]].mean() > 0.97              # out to the canvas edges


def test_hidden_coverage_draws_no_fill_and_no_grid():
    frame, hud = grey(STREAM), make_hud()
    coverage = np.full(GRID, 3)
    hidden = compose_live(frame, hud, coverage, show_coverage=False)
    np.testing.assert_array_equal(hidden, compose_live(frame, hud, None))
    assert (hidden[MIDDLE] == GREY).all()
    assert not np.array_equal(compose_live(frame, hud, coverage), hidden)


def test_hud_bands_darken_the_frame_and_carry_text():
    canvas = compose_live(grey(STREAM, 200), make_hud())
    top, bottom = band_rows(canvas)
    assert 0 < top.stop < MIDDLE.start and MIDDLE.stop < bottom.start < 480
    assert canvas[0].max() < 100 and canvas[-1].max() < 100     # band padding: shade, no text
    assert canvas[top].max() > 230                              # white counters and status
    assert canvas[bottom].max() > 150                           # light grey key help
    assert (canvas[MIDDLE] == 200).all()


@pytest.mark.parametrize('change, band', [
    ({'kept': 4}, 'top'),
    ({'rejected': 2}, 'top'),
    ({'coverage_fraction': 0.5}, 'top'),
    ({'status': 'solving...'}, 'top'),
    ({'help': 'Q quit'}, 'bottom'),
])
def test_hud_text_follows_the_state(change, band):
    frame = grey(STREAM, 200)
    base = compose_live(frame, make_hud())
    rows = np.flatnonzero(np.any(compose_live(frame, make_hud(**change)) != base, axis=(1, 2)))
    top, bottom = band_rows(base)
    assert rows.size
    if band == 'top':
        assert rows.max() < top.stop
    else:
        assert rows.min() >= bottom.start


def test_transient_message_is_red_in_the_bottom_band():
    frame = grey(STREAM)
    assert not (excess(compose_live(frame, make_hud()), RED) > 100).any()
    canvas = compose_live(frame, make_hud(message='capture timed out'))
    red = excess(canvas, RED) > 100
    _, bottom = band_rows(canvas)
    assert red.sum() > 50
    assert np.flatnonzero(red.any(axis=1)).min() >= bottom.start


def test_converged_status_is_green():
    frame = grey(STREAM)
    assert not (excess(compose_live(frame, make_hud(status=LONG_STATUS)), GREEN) > 100).any()
    canvas = compose_live(frame, make_hud(status=LONG_STATUS, converged=True))
    green = excess(canvas, GREEN) > 100
    assert green.sum() > 50
    assert np.flatnonzero(green.any(axis=1)).max() < MIDDLE.start


def test_long_status_wraps_inside_the_canvas():
    canvas = compose_live(grey(STREAM, 200), make_hud(status=LONG_STATUS))
    top, _ = band_rows(canvas)
    band = canvas[top]
    assert top.stop < MIDDLE.start
    # wrapped lines: text reaches the right half but stops short of the right edge
    assert band[:, 320:-4].max() > 230
    assert (band[:, -4:] == band[0, -1]).all()


def test_overlong_text_never_covers_the_centre():
    # status, message and verdict are cut to a few lines, so the bands stay out of the centre
    hud = make_hud(status=LONG_STATUS * 4, message='timeout ' * 200,
                   verdict='REJECTED: ' + 'reason ' * 200, verdict_ok=False)
    for canvas in (compose_live(grey(STREAM), hud), compose_review(grey(FULL), [], False, hud)):
        assert (canvas[180:400] == GREY).all()


def test_live_without_a_frame_is_black_under_the_hud():
    canvas = compose_live(None, make_hud(status='waiting for the image stream'))
    assert canvas.shape == (480, 640, 3)
    assert (canvas[MIDDLE] == 0).all()
    assert canvas[:MIDDLE.start].max() > 200


def test_malformed_coverage_is_rejected():
    with pytest.raises(ValueError, match='rows, cols'):
        compose_live(grey(STREAM), make_hud(), np.zeros(8))
