"""
Window canvas for the calibration GUI: live and review modes, coverage map and HUD.

Pure rendering, no ROS and no window calls: the node's GUI loop shows the canvases returned here.
Every canvas is BGR uint8 of exactly canvas_size (W, H), whatever the size or channel count of the
input frame, so the window never changes size between modes. Frames are stretched to the canvas per
axis. Coordinates use the pixel-centre convention: pixel (0, 0) is centred on (0, 0), so a W x H
image spans [-0.5, W - 0.5] x [-0.5, H - 0.5].
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import ArrayLike

_Colour = tuple[int, int, int]

_REF_SIZE = (640, 480)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TEXT_SCALE = 0.5
_VERDICT_SCALE = 0.8
_DISTANCE_SCALE = 1.6
_RANGE_SCALE = 0.8
_PAD = 6
_BAND_ALPHA = 0.7
_VERDICT_ALPHA = 0.8
_MAX_STATUS_LINES = 3
_MAX_MESSAGE_LINES = 2
_MAX_VERDICT_LINES = 2
_MARKER_RADIUS = 3.0
_SHIFT = 4
_COVERAGE_SATURATION = 3
_COVERAGE_MAX_ALPHA = 0.45

_WHITE: _Colour = (255, 255, 255)
_LIGHT_GREY: _Colour = (200, 200, 200)
_BLACK: _Colour = (0, 0, 0)
_GREEN: _Colour = (0, 255, 0)
_RED: _Colour = (0, 0, 255)
_CONVERGED: _Colour = (110, 255, 110)
_MESSAGE: _Colour = (70, 70, 255)
_NO_BOARD: _Colour = (0, 200, 255)
_VERDICT_KEPT: _Colour = (0, 140, 0)
_VERDICT_REJECTED: _Colour = (0, 0, 190)


@dataclass
class HudState:
    """What the HUD shows; the node builds one per displayed frame. @args: kept: kept views so far.
    rejected: rejected captures so far. min_views: kept views needed before the first solve.
    status: status line, e.g. "collecting 5/12", "solving...", "rms 0.21 px | fx 1402.3 +- 0.9 |
    ..."; long lines wrap at " | ". coverage_fraction: share of coverage cells holding a kept
    corner, 0-1; None hides it. converged: the solution meets the convergence thresholds; the
    status line turns green. verdict: review only, "KEPT" or "REJECTED: <reason>"; None shows
    "KEPT" or "REJECTED" according to kept. verdict_ok: verdict banner colour, True green and
    False red; None follows kept. message: transient error shown in red, e.g. "capture timed
    out"; None shows nothing. help: key help line. distance: live only, the board's distance
    from the camera for setting the focus: (z, range) in metres, drawn large at the top right
    under the top band; a str such as "no board" stands in for the numbers; None shows no
    readout.
    """
    kept: int
    rejected: int
    min_views: int
    status: str
    coverage_fraction: float | None = None
    converged: bool = False
    verdict: str | None = None
    verdict_ok: bool | None = None
    message: str | None = None
    help: str = "SPACE capture   U undo   M coverage   Q quit"
    distance: tuple[float, float] | str | None = None


def scale_points(points: ArrayLike, src_size: tuple[int, int],
                 dst_size: tuple[int, int]) -> np.ndarray:
    """Map pixel coordinates from an image of src_size to the same image resized to dst_size."""
    ratio = np.divide(_size(dst_size, "dst_size"), _size(src_size, "src_size"), dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return np.empty((0, 2))
    if pts.shape[-1] != 2:
        raise ValueError(f"points must have shape (..., 2), got {pts.shape}")
    return (pts + 0.5) * ratio - 0.5


def compose_live(frame: np.ndarray | None, hud: HudState, coverage: np.ndarray | None = None,
                 canvas_size: tuple[int, int] = (640, 480),
                 show_coverage: bool = True) -> np.ndarray:
    """Live-mode canvas: the latest stream frame under the coverage map, the HUD and the distance.
    @args: frame: stream frame, mono (H, W) or (H, W, 1), or BGR (H, W, 3), any size; it is
    resized to the canvas (INTER_AREA).
    """
    canvas = _frame_canvas(frame, _size(canvas_size, "canvas_size"))
    if show_coverage and coverage is not None:
        _draw_coverage(canvas, coverage)
    below_top = _draw_hud(canvas, hud)
    if hud.distance is not None:
        _draw_distance(canvas, hud.distance, below_top)
    return canvas


def compose_review(frame_full: np.ndarray | None, corners_full: ArrayLike | None, kept: bool,
                   hud: HudState, canvas_size: tuple[int, int] = (640, 480)) -> np.ndarray:
    """Review-mode canvas: a captured frame scaled into the canvas, its corners and a verdict
    banner.
    """
    size = _size(canvas_size, "canvas_size")
    image = None if frame_full is None else np.asarray(frame_full)
    canvas = _frame_canvas(image, size)
    verdict = hud.verdict or ("KEPT" if kept else "REJECTED")
    verdict_ok = kept if hud.verdict_ok is None else hud.verdict_ok
    _draw_hud(canvas, hud, banner=(verdict, verdict_ok))
    if image is not None and image.size and corners_full is not None:
        full_size = (image.shape[1], image.shape[0])
        corners = scale_points(corners_full, full_size, size)
        _draw_markers(canvas, corners, _GREEN if kept else _RED)
    return canvas


@dataclass(frozen=True)
class _Font:
    """FONT_HERSHEY_SIMPLEX at one scale and thickness, with its line metrics in px."""
    scale: float
    thickness: int

    def width(self, text: str) -> int:
        return cv2.getTextSize(text, _FONT, self.scale, self.thickness)[0][0]

    def metrics(self) -> tuple[int, int, int]:
        """Return (ascent, descent, gap between lines) in px."""
        (_, ascent), descent = cv2.getTextSize("Ag", _FONT, self.scale, self.thickness)
        return ascent, descent, max(1, ascent // 3)


def _size(size: tuple[int, int], name: str) -> tuple[int, int]:
    """Validate a (W, H) size and return it as ints."""
    w, h = (int(v) for v in size)
    if w <= 0 or h <= 0:
        raise ValueError(f"{name} must be a positive (W, H), got {size}")
    return w, h


def _ui_scale(canvas: np.ndarray) -> float:
    """Return the layout scale of canvas relative to the reference size (1.0 at 640x480)."""
    h, w = canvas.shape[:2]
    return min(w / _REF_SIZE[0], h / _REF_SIZE[1])


def _frame_canvas(frame: np.ndarray | None, size: tuple[int, int]) -> np.ndarray:
    """Return a new BGR uint8 canvas of size (W, H) holding frame."""
    w, h = size
    if frame is None or np.size(frame) == 0:
        return np.zeros((h, w, 3), np.uint8)
    source = np.asarray(frame)
    image = source
    if image.ndim == 3 and image.shape[2] in (1, 3, 4):
        image = image[:, :, 0] if image.shape[2] == 1 else image[:, :, :3]
    elif image.ndim != 2:
        raise ValueError(f"frame must be mono (H, W) or BGR (H, W, 3), got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[:2] != (h, w):
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy() if np.may_share_memory(image, source) else image


def _cell_edges(n_px: int, n_cells: int) -> np.ndarray:
    """
    Return the first pixel of each of n_cells equal cells spanning n_px pixels, then n_px.

    A pixel belongs to the cell holding its centre (pixel-centre convention).
    """
    cell_of = np.minimum(((np.arange(n_px) + 0.5) * (n_cells / n_px)).astype(int), n_cells - 1)
    return np.searchsorted(cell_of, np.arange(n_cells + 1))


def _draw_coverage(canvas: np.ndarray, coverage: np.ndarray) -> None:
    """Draw the coverage map onto canvas in place."""
    counts = np.asarray(coverage)
    if counts.ndim != 2 or 0 in counts.shape:
        raise ValueError(f"coverage must be a (rows, cols) array, got shape {counts.shape}")
    h, w = canvas.shape[:2]
    y_edges, x_edges = _cell_edges(h, counts.shape[0]), _cell_edges(w, counts.shape[1])
    for r, c in zip(*np.nonzero(counts > 0)):
        alpha = min(float(counts[r, c]), _COVERAGE_SATURATION) / _COVERAGE_SATURATION
        cell = canvas[y_edges[r]:y_edges[r + 1], x_edges[c]:x_edges[c + 1]]
        _blend(cell, _GREEN, alpha * _COVERAGE_MAX_ALPHA)
    xs, ys = x_edges[1:-1], y_edges[1:-1]
    canvas[:, xs[xs < w]] = _LIGHT_GREY
    canvas[ys[ys < h], :] = _LIGHT_GREY


def _draw_markers(canvas: np.ndarray, points: np.ndarray, colour: _Colour) -> None:
    """Draw filled, black-outlined dots centred on points (canvas pixels, sub-pixel accurate)."""
    h, w = canvas.shape[:2]
    ui = _ui_scale(canvas)
    radius = max(2.0, _MARKER_RADIUS * ui)
    outline = radius + max(1.0, ui)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    on_canvas = ((pts[:, 0] > -outline) & (pts[:, 0] < w - 1 + outline)
                 & (pts[:, 1] > -outline) & (pts[:, 1] < h - 1 + outline))
    one = 1 << _SHIFT
    centres = [(int(round(x * one)), int(round(y * one))) for x, y in pts[on_canvas]]
    for r, c in ((outline, _BLACK), (radius, colour)):
        for centre in centres:
            cv2.circle(canvas, centre, int(round(r * one)), c, -1, cv2.LINE_AA, _SHIFT)


def _draw_hud(canvas: np.ndarray, hud: HudState, banner: tuple[str, bool] | None = None) -> int:
    """Draw the HUD onto canvas in place; return the row just below the top band."""
    h, w = canvas.shape[:2]
    ui = _ui_scale(canvas)
    pad = max(1, round(_PAD * ui))
    width = w - 2 * pad
    body = _Font(_TEXT_SCALE * ui, max(1, round(ui)))

    counters = f"kept {hud.kept}" + (f"/{hud.min_views}" if hud.kept < hud.min_views else "")
    counters += f"   rejected {hud.rejected}"
    if hud.coverage_fraction is not None:
        counters += f"   coverage {100.0 * hud.coverage_fraction:.0f}%"
    status_colour = _CONVERGED if hud.converged else _WHITE
    top = [(line, _WHITE) for line in _wrap(counters, width, body, 2)]
    top += [(line, status_colour) for line in _wrap(hud.status, width, body, _MAX_STATUS_LINES)]
    y = _text_band(canvas, 0, top, body, pad, _BLACK, _BAND_ALPHA)

    if banner is not None:
        verdict, ok = banner
        big = _Font(_VERDICT_SCALE * ui, max(1, round(2 * ui)))
        lines = [(line, _WHITE) for line in _wrap(verdict, width, big, _MAX_VERDICT_LINES)]
        colour = _VERDICT_KEPT if ok else _VERDICT_REJECTED
        _text_band(canvas, y, lines, big, pad, colour, _VERDICT_ALPHA, centred=True)

    message = _wrap(hud.message or "", width, body, _MAX_MESSAGE_LINES)
    bottom = [(line, _MESSAGE) for line in message]
    bottom += [(line, _LIGHT_GREY) for line in _wrap(hud.help, width, body, 1)]
    _text_band(canvas, h - _band_height(len(bottom), body, pad), bottom, body, pad, _BLACK,
               _BAND_ALPHA)
    return y


def _draw_distance(canvas: np.ndarray, distance: tuple[float, float] | str, y0: int) -> None:
    """Draw the distance readout onto canvas in place: a dark box at the top right from row y0."""
    w = canvas.shape[1]
    ui = _ui_scale(canvas)
    pad = max(1, round(_PAD * ui))
    big = _Font(_DISTANCE_SCALE * ui, max(1, round(3 * ui)))
    small = _Font(_RANGE_SCALE * ui, max(1, round(2 * ui)))
    if isinstance(distance, str):
        lines = [(distance, big, _NO_BOARD)]
    else:
        z, straight = distance
        lines = [(f"z {z:.2f} m", big, _WHITE), (f"range {straight:.2f} m", small, _LIGHT_GREY)]
    text_w = max(font.width(text) for text, font, _ in lines)
    box_w = min(w, text_w + 2 * pad)
    x0 = w - box_w
    heights = [sum(font.metrics()[:2]) for _, font, _ in lines]
    gap = big.metrics()[2]
    y1 = y0 + 2 * pad + sum(heights) + gap * (len(lines) - 1)
    _blend(canvas[max(0, y0):y1, x0:], _BLACK, _BAND_ALPHA)
    y = y0 + pad
    for (text, font, colour), height in zip(lines, heights):
        ascent = font.metrics()[0]
        x = w - pad - font.width(text)
        cv2.putText(canvas, text, (max(0, x), y + ascent), _FONT, font.scale, colour,
                    font.thickness, cv2.LINE_AA)
        y += height + gap


def _band_height(n_lines: int, font: _Font, pad: int) -> int:
    """Return the height of a band holding n_lines of font text; 0 for no lines."""
    if n_lines <= 0:
        return 0
    ascent, descent, gap = font.metrics()
    return 2 * pad + n_lines * (ascent + descent) + (n_lines - 1) * gap


def _text_band(canvas: np.ndarray, y0: int, lines: list[tuple[str, _Colour]], font: _Font,
               pad: int, colour: _Colour, alpha: float, centred: bool = False) -> int:
    """
    Shade a full-width band from row y0 and write lines into it, left-aligned or centred.

    Returns the row just below the band (y0 when there are no lines).
    """
    if not lines:
        return y0
    w = canvas.shape[1]
    ascent, descent, gap = font.metrics()
    y1 = y0 + _band_height(len(lines), font, pad)
    _blend(canvas[max(0, y0):y1], colour, alpha)
    for i, (text, text_colour) in enumerate(lines):
        x = (w - font.width(text)) // 2 if centred else pad
        baseline = y0 + pad + ascent + i * (ascent + descent + gap)
        cv2.putText(canvas, text, (max(0, x), baseline), _FONT, font.scale, text_colour,
                    font.thickness, cv2.LINE_AA)
    return y1


def _blend(region: np.ndarray, colour: _Colour, alpha: float) -> None:
    """Blend colour over region, a view into the canvas, in place at opacity alpha."""
    if region.size:
        overlay = np.full_like(region, colour)
        region[...] = cv2.addWeighted(region, 1.0 - alpha, overlay, alpha, 0.0)


def _wrap(text: str, max_width: int, font: _Font, max_lines: int) -> list[str]:
    """Wrap text greedily so every line fits max_width px."""
    lines: list[str] = []
    for paragraph in text.splitlines():
        current = ""
        for group in paragraph.split(" | "):
            joined = f"{current} | {group}" if current else group
            if font.width(joined) <= max_width:
                current = joined
                continue
            if current:
                lines.append(current)
            current = ""
            for word in group.split(" "):
                joined = f"{current} {word}" if current else word
                if current and font.width(joined) > max_width:
                    lines.append(current)
                    current = word
                else:
                    current = joined
        if current:
            lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines:
            last = lines[-1].rstrip()
            while last and font.width(last + "...") > max_width:
                last = last[:-1].rstrip()
            lines[-1] = last + "..."
    return lines
