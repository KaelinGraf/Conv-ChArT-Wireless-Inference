"""
Camera calibration session: detection checks, kept views, the solve, coverage and the file writes.

Pure: no ROS, no GUI. CalibrationSession is thread-safe: capture callbacks evaluate and submit
frames while a worker thread solves, and solve() copies the active views under the session's lock
and calibrates without holding it.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import os
from pathlib import Path
import shutil
import threading
from typing import Any
import uuid

import cv2
import numpy as np
import yaml

from .calib_board import board_spec_from_cfg, BoardSpec, make_board

REJECT_NO_BOARD = "no board found"
REJECT_COLLINEAR = "corners in a line"      # ids span < 2 rows or < 2 cols of the corner grid
_REJECT_TOO_FEW = "too few corners ({}/{})"

_CFG_KEYS = {"board", "min_corners", "min_views", "review_s", "outlier_factor", "fix_k3",
             "coverage_grid", "converge_rel_sigma_f", "converge_sigma_c_px"}
_STD_KEYS = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")
_VIEW_FILE = "frames/view_{:03d}.png"


@dataclass(frozen=True)
class CalibrationSettings:
    """
    Settings of one calibration session: the CALIBRATION block of the config.
      @args:
        board: the physical ChArUco board.
        min_corners: fewest ChArUco corners a kept view needs; None means
                     max(6, ceil(0.5 * inner corners)), see min_corners_resolved.
        min_views: active views needed before the first solve.
        review_s: seconds a capture's verdict stays on screen.
        outlier_factor: a solve drops views whose reprojection error exceeds this times the median.
        fix_k3: hold k3 at 0 instead of estimating it.
        coverage_grid: (rows, cols) of the coverage map over the full frame.
        converge_rel_sigma_f: converged when sigma(fx)/fx and sigma(fy)/fy are below this ...
        converge_sigma_c_px: ... and sigma(cx), sigma(cy) are below this (full-res pixels).
    """
    board: BoardSpec = BoardSpec()
    min_corners: int | None = None
    min_views: int = 12
    review_s: float = 2.0
    outlier_factor: float = 3.0
    fix_k3: bool = False
    coverage_grid: tuple[int, int] = (6, 8)
    converge_rel_sigma_f: float = 0.002
    converge_sigma_c_px: float = 1.0

    @property
    def min_corners_resolved(self) -> int:
        if self.min_corners is not None:
            return self.min_corners
        return max(6, math.ceil(0.5 * self.board.n_inner_corners))


def settings_from_cfg(cfg: dict[str, Any] | None) -> CalibrationSettings:
    """
    Build CalibrationSettings from the CALIBRATION block of a loaded config.

    A missing block, and missing or null keys in it, keep the defaults; unknown keys raise
    ValueError.
      @args:
        cfg: the whole config mapping (yaml.safe_load of cfg.yaml); only CALIBRATION is read.
    """
    block = (cfg or {}).get("CALIBRATION") or {}
    if not isinstance(block, dict):
        raise ValueError(f"CALIBRATION must be a mapping, got {type(block).__name__}")
    values = {k: v for k, v in block.items() if v is not None}
    unknown = set(values) - _CFG_KEYS
    if unknown:
        raise ValueError(f"unknown CALIBRATION keys: {sorted(unknown)}")
    default = CalibrationSettings()
    grid = tuple(int(v) for v in values.get("coverage_grid", default.coverage_grid))
    if len(grid) != 2:
        raise ValueError(f"CALIBRATION.coverage_grid must be [rows, cols], "
                         f"got {values['coverage_grid']}")
    return CalibrationSettings(
        board=board_spec_from_cfg(values.get("board")),
        min_corners=int(values["min_corners"]) if "min_corners" in values else None,
        min_views=int(values.get("min_views", default.min_views)),
        review_s=float(values.get("review_s", default.review_s)),
        outlier_factor=float(values.get("outlier_factor", default.outlier_factor)),
        fix_k3=bool(values.get("fix_k3", default.fix_k3)),
        coverage_grid=grid,
        converge_rel_sigma_f=float(values.get("converge_rel_sigma_f",
                                              default.converge_rel_sigma_f)),
        converge_sigma_c_px=float(values.get("converge_sigma_c_px", default.converge_sigma_c_px)),
    )


@dataclass
class Detection:
    """
    The board found in one capture, and whether the view is usable for calibration.
      @args:
        kept: the view passed every check.
        reason: "" when kept; else REJECT_NO_BOARD, "too few corners (n/min)" or REJECT_COLLINEAR.
        corners: (N, 2) float64 ChArUco corners in full-res pixels (pixel-centre convention); empty
                 when no board was found.
        ids: (N,) ChArUco corner ids, row-major over the inner-corner grid.
        image_size: (W, H) of the frame.
        obj_points: (N, 3) float32 board coordinates of the corners when kept, else None.
        img_points: (N, 2) float32 image coordinates of the corners when kept, else None.
    """
    kept: bool
    reason: str
    corners: np.ndarray
    ids: np.ndarray
    image_size: tuple[int, int]
    obj_points: np.ndarray | None
    img_points: np.ndarray | None


@dataclass
class Solution:
    """
    Result of one solve, at full resolution.
      @args:
        rms: overall RMS reprojection error in pixels.
        K: (3, 3) camera matrix.
        dist: (5,) distortion coefficients (k1, k2, p1, p2, k3).
        std: standard deviations of fx fy cx cy k1 k2 p1 p2 k3 (0 for a fixed parameter).
        per_view_errors: view index -> RMS reprojection error in pixels, for the views solved over.
        n_views: number of views solved over.
        dropped: indices of the views this solve dropped as outliers.
        image_size: (W, H) the solution belongs to.
        converged: the standard deviations meet the settings' convergence thresholds.
    """
    rms: float
    K: np.ndarray
    dist: np.ndarray
    std: dict[str, float]
    per_view_errors: dict[int, float]
    n_views: int
    dropped: list[int]
    image_size: tuple[int, int]
    converged: bool


@dataclass
class _View:
    """A kept view. The point arrays never change after submit(); the rest only under the lock."""
    index: int
    obj_points: np.ndarray              # (N, 3) float32
    img_points: np.ndarray              # (N, 2) float32
    corners: np.ndarray                 # (N, 2) float64
    image_size: tuple[int, int]
    dropped: bool = False
    error_px: float | None = None       # from the latest solve that used or dropped it


class CalibrationSession:
    """
    One calibration run: checks captures, keeps the usable ones and solves over them.

    View indices count kept views from 0 and are never reused, so an index names the same view (and
    file) for the whole session, across undo. Thread-safe: one internal lock guards the views,
    counters, solution and history; solve() holds it only to copy the active views and to store
    its result, so captures and undo keep working during a solve. Construction also measures the
    installed OpenCV's ChArUco corner convention (charuco_offset_px) and lets its RuntimeError
    through, so an unknown convention stops the session before the first capture.
      @args:
        settings: the session settings; the board and the counts are validated here (ValueError).
    """

    def __init__(self, settings: CalibrationSettings):
        make_board(settings.board)
        if settings.min_corners_resolved < 4:
            raise ValueError(f"min_corners must be at least 4 (calibrating needs 4 points per "
                             f"view), got {settings.min_corners_resolved}")
        if settings.min_views < 1:
            raise ValueError(f"min_views must be at least 1, got {settings.min_views}")
        if len(settings.coverage_grid) != 2 or min(settings.coverage_grid) < 1:
            raise ValueError(f"coverage_grid must be two positive counts, "
                             f"got {settings.coverage_grid}")
        charuco_offset_px()     # an unknown corner convention fails here, not at the first capture
        self._settings = settings
        self._lock = threading.Lock()
        self._views: list[_View] = []
        self._next_index = 0
        self._rejected = 0
        self._last_size: tuple[int, int] | None = None
        self._solution: Solution | None = None
        self._history: list[dict[str, Any]] = []

    def evaluate(self, frame: np.ndarray) -> Detection:
        """
        Detect the board in a full-resolution frame and check the view is usable. Records nothing.
          @args:
            frame: mono (H, W) or BGR (H, W, 3) uint8 image; colour is converted to grey.
        """
        gray = _to_gray(frame)
        size = (int(gray.shape[1]), int(gray.shape[0]))
        board = make_board(self._settings.board)
        charuco_corners, charuco_ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
        if charuco_ids is None or len(charuco_ids) == 0:
            return Detection(False, REJECT_NO_BOARD, np.zeros((0, 2)), np.zeros(0, dtype=int),
                             size, None, None)
        # The detector's own convention depends on the OpenCV version; see charuco_offset_px.
        corners = charuco_corners.reshape(-1, 2).astype(np.float64) - charuco_offset_px()
        ids = charuco_ids.reshape(-1).astype(int)
        minimum = self._settings.min_corners_resolved
        if len(ids) < minimum:
            return Detection(False, _REJECT_TOO_FEW.format(len(ids), minimum), corners, ids, size,
                             None, None)
        per_row = self._settings.board.squares[0] - 1
        if len(np.unique(ids // per_row)) < 2 or len(np.unique(ids % per_row)) < 2:
            return Detection(False, REJECT_COLLINEAR, corners, ids, size, None, None)
        obj_points, img_points = board.matchImagePoints(
            corners.astype(np.float32).reshape(-1, 1, 2), charuco_ids)
        return Detection(True, "", corners, ids, size,
                         obj_points.reshape(-1, 3).astype(np.float32),
                         img_points.reshape(-1, 2).astype(np.float32))

    def submit(self, det: Detection) -> int | None:
        """
        Record an evaluated capture; return the new view's index, or None for a rejected capture.

        A rejected capture is only counted. Kept views must all share one image size (ValueError).
          @args:
            det: the result of evaluate().
        """
        size = (int(det.image_size[0]), int(det.image_size[1]))
        if not det.kept:
            with self._lock:
                self._rejected += 1
                self._last_size = size
            return None
        if det.obj_points is None or det.img_points is None:
            raise ValueError("a kept detection needs obj_points and img_points")
        obj_points = np.array(det.obj_points, dtype=np.float32).reshape(-1, 3)
        img_points = np.array(det.img_points, dtype=np.float32).reshape(-1, 2)
        if len(obj_points) != len(img_points):
            raise ValueError(f"{len(obj_points)} object points for {len(img_points)} image points")
        corners = np.array(det.corners, dtype=np.float64).reshape(-1, 2)
        with self._lock:
            if self._views and self._views[0].image_size != size:
                raise ValueError(f"view image size {size} differs from the session's "
                                 f"{self._views[0].image_size}")
            view = _View(self._next_index, obj_points, img_points, corners, size)
            self._views.append(view)
            self._next_index += 1
            self._last_size = size
            return view.index

    def undo(self) -> int | None:
        """
        Remove the last kept view, active or dropped, and return its index (None if there is none).
        """
        with self._lock:
            return self._views.pop().index if self._views else None

    @property
    def kept_count(self) -> int:
        with self._lock:
            return len(self._views)

    @property
    def rejected_count(self) -> int:
        with self._lock:
            return self._rejected

    @property
    def active_count(self) -> int:
        with self._lock:
            return sum(not v.dropped for v in self._views)

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return sum(v.dropped for v in self._views)

    @property
    def ready(self) -> bool:
        return self.active_count >= self._settings.min_views

    @property
    def solution(self) -> Solution | None:
        with self._lock:
            return self._solution

    @property
    def history(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(entry) for entry in self._history]

    def solve(self) -> Solution:
        """
        Calibrate over the active views, then drop outlier views and solve again once.

        Warm-started from the previous solution when the image size matches (and OpenCV accepts it
        as a guess: positive focal lengths, principal point inside the image). A view is an outlier
        when its error exceeds outlier_factor * median; the worst go first, and never more than
        would leave fewer than min_views active. Dropped views stay out of later solves. Raises
        RuntimeError when not ready.
        """
        settings = self._settings
        with self._lock:
            active = [(v.index, v.obj_points, v.img_points) for v in self._views if not v.dropped]
            if len(active) < settings.min_views:
                raise RuntimeError(f"solve needs {settings.min_views} active views, "
                                   f"has {len(active)}")
            size = self._views[0].image_size
            previous = self._solution

        flags = cv2.CALIB_FIX_K3 if settings.fix_k3 else 0
        guess = None
        if previous is not None and _usable_guess(previous, size):
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS
            guess = (previous.K, previous.dist)
        rms, K, dist, std, errors = _calibrate(active, size, guess, flags)
        first_errors = errors
        limit = settings.outlier_factor * float(np.median(errors))
        outliers = [int(i) for i in np.argsort(-errors, kind="stable") if errors[i] > limit]
        drop = set(outliers[:max(len(active) - settings.min_views, 0)])
        used = active
        if drop:
            used = [view for i, view in enumerate(active) if i not in drop]
            rms, K, dist, std, errors = _calibrate(used, size, guess, flags)

        std_by_name = {name: float(s) for name, s in zip(_STD_KEYS, std)}
        solution = Solution(
            rms=rms, K=K, dist=dist, std=std_by_name,
            per_view_errors={index: float(e) for (index, _, _), e in zip(used, errors)},
            n_views=len(used),
            dropped=sorted(active[i][0] for i in drop),
            image_size=size,
            converged=_converged(K, std_by_name, settings),
        )
        with self._lock:
            by_index = {v.index: v for v in self._views}     # undo may have removed some meanwhile
            for i in drop:
                view = by_index.get(active[i][0])
                if view is not None:
                    view.dropped = True
                    view.error_px = float(first_errors[i])
            for index, error in solution.per_view_errors.items():
                if index in by_index:
                    by_index[index].error_px = error
            self._solution = solution
            self._history.append({"n_views": solution.n_views, "rms": solution.rms,
                                  "fx": float(K[0, 0]), "fy": float(K[1, 1]),
                                  "cx": float(K[0, 2]), "cy": float(K[1, 2])})
        return solution

    def coverage(self) -> np.ndarray:
        """
        Count the active views' corners per cell of a (rows, cols) grid spanning the full frame.
        """
        with self._lock:
            active = [(v.corners, v.image_size) for v in self._views if not v.dropped]
        return _coverage_grid(active, self._settings.coverage_grid)

    def coverage_fraction(self) -> float:
        """
        Return the fraction of coverage cells holding at least one corner.
        """
        grid = self.coverage()
        return float(np.count_nonzero(grid)) / grid.size

    def _log_snapshot(self) -> dict[str, Any]:
        """
        Take everything build_log reports under one lock, so the parts agree with each other.
        """
        with self._lock:
            return {
                "views": [(v.index, len(v.obj_points), v.error_px, v.dropped)
                          for v in self._views],
                "active": [(v.corners, v.image_size) for v in self._views if not v.dropped],
                "rejected": self._rejected,
                "last_size": self._last_size,
                "solution": self._solution,
                "history": [dict(entry) for entry in self._history],
            }


@lru_cache(maxsize=1)
def charuco_offset_px() -> float:
    """
    Measure how far the installed OpenCV's CharucoDetector reports corners off pixel centres.

    OpenCV 4.7 to 4.13 refine each ChArUco corner with cornerSubPix at p - 0.5 and return the
    result + 0.5, so their corners sit 0.5 px right of and below the pixel-centre position; 4.14
    and 5.x return the cornerSubPix result as it is (opencv/opencv#28380), so 0. Subtracting the
    result on both axes puts detected corners in the pixel-centre convention. Measured once per
    process, on the default board pasted axis-aligned into a grey frame, where every corner is
    known exactly. Raises RuntimeError when that board is not fully found or the offset is neither
    0.5 nor 0 on both axes: an unknown convention must stop a calibration, not bias it.
    """
    spec = BoardSpec()
    board = make_board(spec)
    canvas = np.full((600, 600), 140, np.uint8)
    canvas[100:500, 100:500] = board.generateImage((400, 400))      # 80 px squares
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(
        cv2.GaussianBlur(canvas, (0, 0), 1.0))
    found = 0 if ids is None else len(ids)
    if found != spec.n_inner_corners:
        raise RuntimeError(f"cannot measure the ChArUco corner convention of OpenCV "
                           f"{cv2.__version__}: its detector found {found} of the "
                           f"{spec.n_inner_corners} corners of the test board")
    ids = ids.reshape(-1)
    per_row = spec.squares[0] - 1
    # Inner corner (i, j) lies on the edge between pixels 179 + 80 i and 180 + 80 i, which is
    # x = 179.5 + 80 i pixel-centred; likewise y with j.
    truth = 100 + 80 * (1 + np.column_stack([ids % per_row, ids // per_row])) - 0.5
    shift = np.mean(corners.reshape(-1, 2) - truth, axis=0)
    offset = round(2 * float(np.mean(shift))) / 2
    if offset not in (0.0, 0.5) or np.abs(shift - offset).max() > 0.05:
        raise RuntimeError(f"unknown ChArUco corner convention in OpenCV {cv2.__version__}: "
                           f"corners sit ({shift[0]:+.3f}, {shift[1]:+.3f}) px off pixel "
                           f"centres, neither the 0.5 px of OpenCV 4.7-4.13 nor the 0 of 4.14+")
    return offset


def scale_intrinsics(K: np.ndarray, full_size: tuple[int, int],
                     target_size: tuple[int, int]) -> np.ndarray:
    """
    Scale a camera matrix to a uniformly resized image of the same view.

    Pixel-centre convention: s = target_w / full_w, fx' = s * fx, cx' = s * (cx + 0.5) - 0.5, and
    likewise for y; distortion is unchanged. ValueError unless target_h / full_h equals s.
      @args:
        K: (3, 3) camera matrix at full_size.
        full_size: (W, H) that K belongs to.
        target_size: (W, H) to scale K to.
    """
    full_w, full_h = full_size
    target_w, target_h = target_size
    if min(full_w, full_h, target_w, target_h) <= 0:
        raise ValueError(f"image sizes must be positive, got {full_size} -> {target_size}")
    if target_w * full_h != target_h * full_w:
        raise ValueError(f"{tuple(full_size)} -> {tuple(target_size)} is not a uniform scale")
    s = target_w / full_w
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got shape {K.shape}")
    scaled = K.copy()
    scaled[0, 0] = s * K[0, 0]
    scaled[0, 1] = s * K[0, 1]
    scaled[1, 1] = s * K[1, 1]
    scaled[0, 2] = s * (K[0, 2] + 0.5) - 0.5
    scaled[1, 2] = s * (K[1, 2] + 0.5) - 0.5
    return scaled


def build_log(session: CalibrationSession, settings: CalibrationSettings,
              meta: dict[str, Any] | None) -> dict[str, Any]:
    """
    Build the calibration log document (plain Python types, ready for write_log).

    images.full_size is the solution's size, else the last captured frame's; K_stream is None until
    the stream size is known, and also when it is not a uniform scale of the full size.
      @args:
        session: the session to report.
        settings: the session's settings (for the board block).
        meta: what the session does not know, every key optional: started, config, auto_replace,
              config_written, config_backup (the session block); stream_size ([W, H] of the preview
              stream, None until a frame has been seen); view_files ({view index: file}, default
              frames/view_NNN.png with NNN the zero-padded index).
    """
    meta = meta or {}
    snap = session._log_snapshot()
    sol: Solution | None = snap["solution"]
    stream_size = None if meta.get("stream_size") is None else _size_list(meta["stream_size"])
    full_size = sol.image_size if sol is not None else snap["last_size"]
    spec = settings.board

    solution = None
    if sol is not None:
        K_stream = None
        if stream_size is not None:
            try:
                K_stream = _matrix_list(scale_intrinsics(sol.K, sol.image_size, stream_size))
            except ValueError:
                K_stream = None
        solution = {
            "rms_px": float(sol.rms),
            "converged": bool(sol.converged),
            "K_full": _matrix_list(sol.K),
            "dist": [float(v) for v in np.ravel(sol.dist)],
            "std": {name: float(sol.std[name]) for name in _STD_KEYS},
            "K_stream": K_stream,
            "stream_size": stream_size,
        }

    view_files = meta.get("view_files") or {}
    views = [{"index": int(index),
              "file": str(view_files.get(index, _VIEW_FILE.format(index))),
              "n_corners": int(n_corners),
              "error_px": None if error is None else float(error),
              "dropped": bool(dropped)}
             for index, n_corners, error, dropped in snap["views"]]
    n_dropped = sum(view["dropped"] for view in views)
    grid = _coverage_grid(snap["active"], settings.coverage_grid)

    return {
        "session": {
            "started": _optional_str(meta.get("started")),
            "config": _optional_str(meta.get("config")),
            "auto_replace": bool(meta.get("auto_replace", False)),
            "config_written": bool(meta.get("config_written", False)),
            "config_backup": _optional_str(meta.get("config_backup")),
        },
        "board": {
            "squares": [int(v) for v in spec.squares],
            "square_length_m": float(spec.square_length),
            "marker_length_m": float(spec.marker_side),
            "dictionary": spec.dictionary,
            "first_marker_id": int(spec.first_marker_id),
            "legacy_pattern": bool(spec.legacy_pattern),
        },
        "images": {
            "full_size": None if full_size is None else _size_list(full_size),
            "stream_size": stream_size,
        },
        "counts": {
            "kept": len(views),
            "rejected": int(snap["rejected"]),
            "dropped": n_dropped,
            "active": len(views) - n_dropped,
        },
        "coverage_fraction": float(np.count_nonzero(grid)) / grid.size,
        "solution": solution,
        "views": views,
        "history": [{"n_views": int(entry["n_views"]), "rms_px": float(entry["rms"]),
                     "fx": float(entry["fx"]), "fy": float(entry["fy"]),
                     "cx": float(entry["cx"]), "cy": float(entry["cy"])}
                    for entry in snap["history"]],
    }


def write_log(path: str | os.PathLike, log: dict[str, Any]) -> None:
    """
    Write the log as YAML atomically, so a reader sees the old log or the new one, never half.
      @args:
        path: the log file; its directory is created if missing.
        log: the document from build_log (plain Python types only).
    """
    _atomic_write(Path(path), _dump_yaml(log).encode("utf-8"))


def write_camera_to_cfg(cfg_path: str | os.PathLike, K: np.ndarray, dist: np.ndarray,
                        image_size: tuple[int, int],
                        backup_path: str | os.PathLike | None = None) -> None:
    """
    Write K, dist and image_size into the CAMERA block of a YAML config.

    Every other key and the key order are kept; comments are lost, since the file is rewritten
    through PyYAML. The write is atomic, keeps the file's permissions, and goes through a symlink
    to the file it points at. With backup_path, the original file is copied there first unless that
    file already exists, so repeated writes keep the very first original.
      @args:
        cfg_path: the config to update.
        K: (3, 3) camera matrix belonging to image_size.
        dist: distortion coefficients (k1, k2, p1, p2, k3).
        image_size: (W, H) that K belongs to.
        backup_path: where to keep a copy of the original config.
    """
    cfg_path = Path(os.path.realpath(cfg_path))
    original = cfg_path.read_bytes()
    cfg = yaml.safe_load(original.decode("utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{cfg_path} does not hold a YAML mapping")
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got shape {K.shape}")
    camera = cfg.get("CAMERA")
    if not isinstance(camera, dict):
        camera = {}
    camera["K"] = _matrix_list(K)
    camera["dist"] = [float(v) for v in np.ravel(dist)]
    camera["image_size"] = _size_list(image_size)
    cfg["CAMERA"] = camera                      # an existing key keeps its place
    if backup_path is not None and not Path(backup_path).exists():
        _atomic_write(Path(backup_path), original, mode_from=cfg_path)
    _atomic_write(cfg_path, _dump_yaml(cfg).encode("utf-8"))


def _to_gray(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 1:
        return np.ascontiguousarray(frame[:, :, 0])
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"expected a mono or BGR image, got shape {frame.shape}")


def _calibrate(views: list[tuple[int, np.ndarray, np.ndarray]], size: tuple[int, int],
               guess: tuple[np.ndarray, np.ndarray] | None,
               flags: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run cv2.calibrateCameraExtended over (index, obj_points, img_points) views.

    Returns rms, K (3, 3), dist (5,), the std of fx fy cx cy k1 k2 p1 p2 k3, and per-view errors.
    """
    obj = [o.reshape(-1, 1, 3) for _, o, _ in views]
    img = [p.reshape(-1, 1, 2) for _, _, p in views]
    # Copies: OpenCV writes its result into the guess arrays, which are the previous solution's.
    K0 = None if guess is None else np.array(guess[0], dtype=np.float64)
    dist0 = None if guess is None else np.array(guess[1], dtype=np.float64).reshape(-1)
    rms, K, dist, _, _, std, _, errors = cv2.calibrateCameraExtended(
        obj, img, size, K0, dist0, flags=flags)
    return (float(rms), np.array(K, dtype=np.float64), np.ravel(dist)[:5].astype(np.float64),
            np.ravel(std)[:len(_STD_KEYS)].astype(np.float64), np.ravel(errors).astype(np.float64))


def _usable_guess(solution: Solution, size: tuple[int, int]) -> bool:
    """
    Check calibrateCamera would accept the solution as an intrinsic guess for this image size.
    """
    K = solution.K
    return bool(solution.image_size == size
                and np.isfinite(K).all() and np.isfinite(solution.dist).all()
                and K[0, 0] > 0 and K[1, 1] > 0
                and 0 <= K[0, 2] < size[0] and 0 <= K[1, 2] < size[1])


def _converged(K: np.ndarray, std: dict[str, float], settings: CalibrationSettings) -> bool:
    rel_f = settings.converge_rel_sigma_f
    sigma_c = settings.converge_sigma_c_px
    return bool(std["fx"] / K[0, 0] < rel_f and std["fy"] / K[1, 1] < rel_f
                and std["cx"] < sigma_c and std["cy"] < sigma_c)


def _coverage_grid(views: list[tuple[np.ndarray, tuple[int, int]]],
                   grid: tuple[int, int]) -> np.ndarray:
    """
    Count corners per cell of a (rows, cols) grid spanning each view's frame uniformly.

    Pixel-centre convention: the frame covers [-0.5, W - 0.5] x [-0.5, H - 0.5].
    """
    rows, cols = grid
    counts = np.zeros((rows, cols), dtype=int)
    for corners, (width, height) in views:
        col = np.clip(np.floor((corners[:, 0] + 0.5) * cols / width), 0, cols - 1).astype(int)
        row = np.clip(np.floor((corners[:, 1] + 0.5) * rows / height), 0, rows - 1).astype(int)
        np.add.at(counts, (row, col), 1)
    return counts


def _matrix_list(K: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in np.asarray(K, dtype=np.float64)]


def _size_list(size: Any) -> list[int]:
    width, height = size
    return [int(width), int(height)]


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


class _YamlDumper(yaml.SafeDumper):
    """SafeDumper writing flat lists and lists of flat lists (vectors, matrices) on one line."""

    def ignore_aliases(self, data: Any) -> bool:
        # A list used twice (build_log's stream_size) would otherwise come out as &id001 / *id001.
        return True


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _represent_list(dumper: yaml.SafeDumper, data: list | tuple) -> yaml.Node:
    inline = all(_is_scalar(v) or (isinstance(v, (list, tuple)) and all(_is_scalar(w) for w in v))
                 for v in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=inline)


_YamlDumper.add_representer(list, _represent_list)
_YamlDumper.add_representer(tuple, _represent_list)


def _dump_yaml(doc: Any) -> str:
    return yaml.dump(doc, Dumper=_YamlDumper, sort_keys=False, default_flow_style=False,
                     allow_unicode=True, width=4096)


def _atomic_write(path: Path, data: bytes, mode_from: Path | None = None) -> None:
    """
    Replace path with data atomically: write a temporary file next to it, fsync, then rename.

    The new file takes the permission bits of mode_from (default: the file it replaces), if that
    exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        mode_source = path if mode_from is None else mode_from
        if mode_source.exists():
            shutil.copymode(mode_source, tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
