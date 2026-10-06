"""
Synthetic ChArUco views from a known camera, for the fake camera node and the calibration tests.

A view is the ideal pinhole render of the board (the board image warped by the homography K [r1 r2 t] S),
distorted by sampling, for every output pixel, the ideal render at that pixel's undistorted location, then
blurred and given Gaussian noise. Each pixel is the mean of 2x2 samples (the sensor integrates over the
pixel), which keeps far, steeply tilted boards from aliasing. The ground truth is cv2.projectPoints of the
board's chessboard corners with the same K and dist, so corners detected on a frame can be checked against
it. Sizes are (W, H) in pixels; pixel coordinates follow the pixel-centre convention of cv2.projectPoints.
cv2.aruco.CharucoDetector reports corners off that convention on some OpenCV versions (+0.5 px on both
axes in 4.7-4.13, 0 from 4.14): subtract calib_session.charuco_offset_px() before comparing its
detections with the ground truth.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from numpy.typing import ArrayLike

from .calib_board import BoardSpec

DEFAULT_FULL_SIZE = (1600, 1200)
DEFAULT_STREAM_SIZE = (640, 480)
DEFAULT_K_FULL = [[1400.0, 0.0, 799.5], [0.0, 1400.0, 599.5], [0.0, 0.0, 1.0]]
DEFAULT_DIST = [-0.12, 0.05, 0.0008, -0.0005, 0.0]

_BACKGROUND = 140           # grey level around the board
_NOISE_STD = 2.0            # grey levels
_SUPERSAMPLE = 2            # samples per pixel along each axis
_MAP_CHUNK_ROWS = 256       # rows of the sample grid undistorted at a time (bounds peak memory)

# diverse_poses
_MARGIN = 0.05              # the whole board stays this fraction of min(W, H) inside every frame edge
_FILL_NEAR = 0.5            # board bounding box / frame size, in the tighter dimension
_FILL_FAR = 0.3
_TILT_DEG = (15.0, 40.0)    # magnitude of the tilt about each board axis
_ROLL_DEG = 20.0            # in-plane rotation, +-


def diverse_poses(spec: BoardSpec, K: ArrayLike, size: tuple[int, int],
                  n: int = 24) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Deterministic board poses that make a well-conditioned calibration, as [(rvec, tvec)] (float64, (3,)).
    Pose i sits in cell i % 9 of a 3x3 grid over the frame (outer cells flush with the margin), near for even
    i and far for odd i. Like a hand-held board, it faces the camera along its line of sight and is then
    tilted 15-40 degrees about each of its axes (signs alternating) and rolled up to +-20 degrees in-plane:
    R = F Rz(roll) Rx(tilt_x) Ry(tilt_y), F turning the optical axis onto the line of sight to the board
    centre by the shortest arc. The whole board stays inside the frame with a margin of 5% of min(W, H)
    under the pinhole model; the margin absorbs moderate lens distortion.
      @args:
        spec: board to place; tvec comes out in spec.square_length units.
        K: (3, 3) camera matrix.
        size: (W, H) of the frame.
        n: number of poses.
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    extent = np.array(spec.squares, dtype=np.float64) * spec.square_length
    centre = np.append(extent / 2.0, 0.0)
    outline = np.array([[0.0, 0.0, 0.0], [extent[0], 0.0, 0.0], [extent[0], extent[1], 0.0],
                        [0.0, extent[1], 0.0]]) - centre
    poses = []
    for i in range(n):
        cell = i % 9
        anchor = np.array([cell % 3, cell // 3], dtype=np.float64) / 2.0
        fill = _FILL_NEAR if i % 2 == 0 else _FILL_FAR
        tilt_x = (-1) ** (i // 2) * _spread(i, 0.7548776662466927, *_TILT_DEG)
        tilt_y = (-1) ** (i // 4) * _spread(i, 0.5698402909980532, *_TILT_DEG)
        roll = _spread(i, 0.6180339887498949, -_ROLL_DEG, _ROLL_DEG)
        rotation, position = _place(_tilt(tilt_x, tilt_y, roll), outline, K, size, anchor, fill)
        rvec = cv2.Rodrigues(rotation)[0].ravel()
        poses.append((rvec, position - rotation @ centre))
    return poses


def render_view(board: cv2.aruco.CharucoBoard, spec: BoardSpec, K: ArrayLike, dist: ArrayLike,
                rvec: ArrayLike, tvec: ArrayLike, size: tuple[int, int], px_per_square: int = 80,
                background: int = _BACKGROUND, noise_std: float = _NOISE_STD, blur_sigma: float = 0.7,
                seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Render the board under one pose, as (frame uint8 (H, W), corners (n_inner, 2) float64, ids (n_inner,)).
    corners are the chessboard corners projected by cv2.projectPoints with dist (pixel-centre convention); ids
    are 0..n_inner-1, the ChArUco corner ids, so corners[found_ids] is the ground truth of a detection.
      @args:
        board: the cv2 board of spec (make_board(spec)).
        spec: board geometry.
        K: (3, 3) camera matrix.
        dist: distortion coefficients in OpenCV order; all zero skips the distortion remap.
        rvec, tvec: board-to-camera pose, tvec in spec.square_length units.
        size: (W, H) of the frame.
        px_per_square: resolution of the board image that is warped into the frame.
        background: grey level around the board.
        noise_std: standard deviation of the additive Gaussian noise, in grey levels.
        blur_sigma: Gaussian blur (optics) in pixels, applied before the noise.
        seed: noise seed; the same seed gives the same frame.
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float64).ravel()
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
    width, height = int(size[0]), int(size[1])
    cols, rows = spec.squares
    if tuple(board.getChessboardSize()) != (cols, rows):
        raise ValueError(f"board has {tuple(board.getChessboardSize())} squares, spec says {spec.squares}")

    board_img = board.generateImage((cols * px_per_square, rows * px_per_square)).astype(np.float32)
    rotation = cv2.Rodrigues(rvec)[0]
    pixel_to_board = _pixel_to_board(spec.square_length / px_per_square)
    homography = K @ np.column_stack([rotation[:, 0], rotation[:, 1], tvec]) @ pixel_to_board
    s = _SUPERSAMPLE
    if np.any(dist):
        map_x, map_y, to_canvas, canvas_size = _undistort_maps(tuple(K.ravel()), tuple(dist), (width, height))
        ideal = _warp(board_img, to_canvas @ homography, canvas_size, background)
        samples = cv2.remap(ideal, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=float(background))
    else:
        samples = _warp(board_img, _to_samples(s) @ homography, (width * s, height * s), background)
    frame = cv2.resize(samples, (width, height), interpolation=cv2.INTER_AREA)
    if blur_sigma > 0:
        frame = cv2.GaussianBlur(frame, (0, 0), blur_sigma)
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        frame += rng.standard_normal(frame.shape, dtype=np.float32) * np.float32(noise_std)

    object_points = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    corners = cv2.projectPoints(object_points, rvec, tvec, K, dist)[0].reshape(-1, 2)
    ids = np.arange(len(object_points), dtype=np.int32)
    return _to_uint8(frame), corners, ids


def blank_frame(size: tuple[int, int], seed: int = 0) -> np.ndarray:
    """
    A frame without a board: render_view's background and noise only, uint8 (H, W).
      @args:
        size: (W, H) of the frame.
        seed: noise seed; the same seed gives the same frame.
    """
    width, height = int(size[0]), int(size[1])
    noise = np.random.default_rng(seed).standard_normal((height, width), dtype=np.float32)
    return _to_uint8(np.float32(_BACKGROUND) + noise * np.float32(_NOISE_STD))


def _pixel_to_board(scale: float) -> np.ndarray:
    """
    S: board-image pixel (u, v) -> board coordinates ((u + 0.5) * scale, (v + 0.5) * scale), pixel-centre
    convention, same origin and axes as board.getChessboardCorners().
    """
    return np.array([[scale, 0.0, 0.5 * scale], [0.0, scale, 0.5 * scale], [0.0, 0.0, 1.0]])


def _to_samples(s: int) -> np.ndarray:
    """
    Frame pixel coordinates -> sample-grid coordinates (s x s samples per pixel, pixel-centre convention).
    """
    offset = (s - 1) / 2.0
    return np.array([[s, 0.0, offset], [0.0, s, offset], [0.0, 0.0, 1.0]])


def _warp(board_img: np.ndarray, homography: np.ndarray, size: tuple[int, int],
          background: float) -> np.ndarray:
    """
    Warp the board image onto a float32 canvas of background, leaving the canvas outside the board as is.
    """
    canvas = np.full((size[1], size[0]), float(background), dtype=np.float32)
    cv2.warpPerspective(board_img, homography, size, dst=canvas, flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_TRANSPARENT)
    return canvas


@lru_cache(maxsize=2)
def _undistort_maps(k_key: tuple[float, ...], dist_key: tuple[float, ...],
                    size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """
    For every sample of the supersampled frame, its undistorted (ideal pinhole) location, cached per camera.
    Returns (map_x, map_y, to_canvas, canvas_size): float32 maps into an ideal canvas sampled like the frame,
    the transform from ideal pixel coordinates to that canvas, and its (W, H). The canvas covers every mapped
    location, which can lie outside the frame (barrel distortion maps the frame's corners outwards).
    """
    width, height = size
    s = _SUPERSAMPLE
    K = np.array(k_key).reshape(3, 3)
    dist = np.array(dist_key)
    from_samples = np.linalg.inv(_to_samples(s))
    xs = np.arange(width * s) * from_samples[0, 0] + from_samples[0, 2]
    ideal = np.empty((height * s, width * s, 2), dtype=np.float32)
    for top in range(0, height * s, _MAP_CHUNK_ROWS):
        rows = np.arange(top, min(top + _MAP_CHUNK_ROWS, height * s))
        u, v = np.meshgrid(xs, rows * from_samples[1, 1] + from_samples[1, 2])
        pixels = np.stack([u.ravel(), v.ravel()], axis=1).reshape(-1, 1, 2)
        ideal[top:top + len(rows)] = cv2.undistortPoints(pixels, K, dist, P=K).reshape(len(rows), -1, 2)
    np.nan_to_num(ideal, copy=False, nan=-1e9, posinf=1e9, neginf=-1e9)
    # pad by 2 px for the bilinear taps; clip so a pathological model cannot ask for a huge canvas
    lo = np.maximum(np.floor(ideal.min(axis=(0, 1))) - 2, [-width, -height])
    hi = np.minimum(np.ceil(ideal.max(axis=(0, 1))) + 2, [2 * width, 2 * height])
    to_canvas = _to_samples(s) @ np.array([[1.0, 0.0, -lo[0]], [0.0, 1.0, -lo[1]], [0.0, 0.0, 1.0]])
    maps = []
    for axis in (0, 1):
        m = ideal[..., axis] * np.float32(s) + np.float32(to_canvas[axis, 2])
        m.setflags(write=False)
        maps.append(m)
    canvas_size = (int(hi[0] - lo[0] + 1) * s, int(hi[1] - lo[1] + 1) * s)
    return maps[0], maps[1], to_canvas, canvas_size


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(frame), 0, 255).astype(np.uint8)


def _spread(i: int, step: float, lo: float, hi: float) -> float:
    """
    Element i of a low-discrepancy (Weyl) sequence, scaled to [lo, hi).
    """
    return lo + (hi - lo) * ((0.5 + i * step) % 1.0)


def _tilt(tilt_x_deg: float, tilt_y_deg: float, roll_deg: float) -> np.ndarray:
    """
    Rz(roll) Rx(tilt_x) Ry(tilt_y): the board's orientation relative to facing the camera.
    """
    a, b, c = np.radians([tilt_x_deg, tilt_y_deg, roll_deg])
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(a), -np.sin(a)], [0.0, np.sin(a), np.cos(a)]])
    rot_y = np.array([[np.cos(b), 0.0, np.sin(b)], [0.0, 1.0, 0.0], [-np.sin(b), 0.0, np.cos(b)]])
    rot_z = np.array([[np.cos(c), -np.sin(c), 0.0], [np.sin(c), np.cos(c), 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_x @ rot_y


def _facing(direction: np.ndarray) -> np.ndarray:
    """
    Rotation turning the optical axis (0, 0, 1) onto direction by the shortest arc.
    """
    unit = direction / np.linalg.norm(direction)
    axis = np.cross([0.0, 0.0, 1.0], unit)
    sin = np.linalg.norm(axis)
    if sin < 1e-15:
        return np.eye(3)
    return cv2.Rodrigues(axis / sin * np.arctan2(sin, unit[2]))[0]


def _bbox(points: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Pixel bounding box (lo, hi) of camera-frame points under the pinhole model.
    """
    projected = points @ K.T
    uv = projected[:, :2] / projected[:, 2:]
    return uv.min(axis=0), uv.max(axis=0)


def _place(tilt: np.ndarray, outline: np.ndarray, K: np.ndarray, size: tuple[int, int], anchor: np.ndarray,
           fill: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Orientation and camera-frame centre of a board tilted by tilt relative to facing the camera, such that its
    projected bounding box has its larger side, relative to the frame, equal to fill and sits at anchor
    between the margins (per axis: 0 = flush with the low margin, 0.5 = centred, 1 = flush with the high one).
      @args:
        tilt: board orientation relative to facing the camera (_tilt).
        outline: (4, 3) board corners about the board centre, in board axes.
    """
    frame = np.array(size, dtype=np.float64)
    margin = _MARGIN * frame.min()
    focal = np.array([K[0, 0], K[1, 1]])
    span = np.ptp(outline[:, :2], axis=0)
    position = np.array([0.0, 0.0, np.max(focal * span / (fill * frame))])
    for _ in range(500):
        points = outline @ (_facing(position) @ tilt).T
        lo, hi = _bbox(points + position, K)
        # moving along the centre's line of sight rescales the board's image without shifting its centre
        scale = np.max((hi - lo) / frame) / fill
        position *= scale
        lo, hi = _bbox(points + position, K)
        shift = margin + anchor * (frame - 1.0 - 2.0 * margin - (hi - lo)) - lo
        position[:2] += shift * position[2] / focal
        if abs(scale - 1.0) < 1e-12 and np.all(np.abs(shift) < 1e-9):
            break
    rotation = _facing(position) @ tilt
    lo, hi = _bbox(outline @ rotation.T + position, K)
    if np.any(lo < margin - 1e-6) or np.any(hi > frame - 1.0 - margin + 1e-6):
        raise RuntimeError(f"could not fit the board in a {size} frame (fill {fill}, anchor {anchor})")
    return rotation, position
