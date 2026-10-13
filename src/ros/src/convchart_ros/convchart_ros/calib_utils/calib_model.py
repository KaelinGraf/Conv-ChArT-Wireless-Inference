"""
The deployed Conv-ChArT model as the calibration's ChArUco corner detector, and as the live range
finder that helps set the lens focus.

Both go through the inference node's own pipeline (inference.inference_pipeline, the config's
MODEL block), unchanged: the detector finds and labels the corners on the frame downsized to its
input, and the refiner places them at the frame's own resolution. One pipeline serves both, one
call at a time: its ONNX sessions bind fixed input and output buffers.

The model's corner ids are OpenCV's ChArUco ids for its board (row-major over the inner corners, x
fastest), so its corners go into board.matchImagePoints as they are. They follow the pixel-centre
convention of cv2.projectPoints, so, unlike OpenCV's detector's, they need no offset.

Corners for a calibration must not depend on the CAMERA block that calibration is about to
replace. The pipeline undistorts the corners with K and dist before its lattice gate (a RANSAC
homography from the ids' lattice positions, which takes the id off every corner that disagrees)
and before it recovers ids from that homography, so a stale K/dist costs good corners, and the
stream-resolution K applied to a full-resolution still costs most of them. corners() gates on the
raw pixels instead: with dist = 0 the undistortion is the identity whatever K is. The lens
distortion left in the raw pixels fits inside the gate once its tolerance is taken as tuned, on
the detector's input, where the coarse corner positions it gates are quantised: on a still 2.5
times wider than the input, that is 2.5 times as many pixels.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import os

import cv2
import numpy as np

from .calib_board import BoardSpec
from ..inference import inference_pipeline

MODEL_BOARD = BoardSpec()

_NO_DIST = np.zeros(5, dtype=np.float64)


@dataclass(frozen=True)
class BoardRange:
    """The board centre's distance from the camera, in BOARD.square_length_m units (metres). @args:
    z: depth along the optical axis. range: straight-line distance from the camera centre.
    """
    z: float
    range: float


def check_model_board(spec: BoardSpec) -> None:
    """Raise ValueError unless spec has the model's board layout."""
    model = MODEL_BOARD
    differences = []
    if tuple(spec.squares) != tuple(model.squares):
        differences.append(f"squares {list(spec.squares)}")
    ratio = spec.marker_side / spec.square_length
    if abs(ratio - model.marker_side / model.square_length) > 1e-6:
        differences.append(f"markers {ratio:g} of a square")
    if spec.dictionary != model.dictionary:
        differences.append(f"dictionary {spec.dictionary}")
    if spec.first_marker_id != model.first_marker_id:
        differences.append(f"first marker id {spec.first_marker_id}")
    if spec.legacy_pattern != model.legacy_pattern:
        differences.append("the legacy pattern")
    if differences:
        raise ValueError(
            f"detector:=model knows only the {model.squares[0]}x{model.squares[1]} "
            f"{model.dictionary} board (markers {model.marker_ratio:g} of a square, ids from "
            f"{model.first_marker_id}), but the calibration board has {', '.join(differences)}: "
            f"use detector:=opencv")


class ModelDetector:
    """The deployed model on calibration stills (corners) and on the live stream (the board's
    range).
    """

    def __init__(self, cfg_path: str | os.PathLike):
        self._model_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="calib_model")
        try:
            self._pipe = self._model_thread.submit(inference_pipeline, str(cfg_path)).result()
        except AssertionError as error:
            raise RuntimeError(f"cannot load the model of {cfg_path}: {error} (detector:=opencv "
                               f"calibrates without it)") from None
        squares = tuple(self._pipe.board.squares)
        if squares != tuple(MODEL_BOARD.squares):
            raise ValueError(f"BOARD.squares in {cfg_path} is {list(squares)}, but the model's "
                             f"board has {list(MODEL_BOARD.squares)}")
        self._tuned = self._pipe.pipeline

    @property
    def square_length_m(self) -> float | None:
        """Return BOARD.square_length_m, the unit of board_range(); None means board squares."""
        return self._pipe.board.square_length_m

    def corners(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Find the ChArUco corners in a frame of any resolution, without the config's K and dist.
        """
        h, w = frame.shape[:2]
        f = float(max(w, h))
        nominal = np.array([[f, 0.0, (w - 1) / 2.0], [0.0, f, (h - 1) / 2.0], [0.0, 0.0, 1.0]])
        scale = max(1.0, w / self._pipe.W_in)
        gate = replace(self._tuned, lattice_tol_px=self._tuned.lattice_tol_px * scale)

        def run() -> dict:
            self._pipe.pipeline = gate
            try:
                return self._pipe.run_inference(frame, K=nominal, dist=_NO_DIST)
            finally:
                self._pipe.pipeline = self._tuned
        result = self._model_thread.submit(run).result()
        found = [(c["index"], c["x"], c["y"]) for c in result["corners"]
                 if c["index"] is not None]
        ids = np.array([index for index, _, _ in found], dtype=int)
        xy = np.array([(x, y) for _, x, y in found], dtype=np.float64).reshape(-1, 2)
        unique, count = np.unique(ids, return_counts=True)
        keep = np.isin(ids, unique[count == 1])
        order = np.argsort(ids[keep], kind="stable")
        return xy[keep][order], ids[keep][order]

    def board_range(self, frame: np.ndarray, K: np.ndarray,
                    dist: np.ndarray | None) -> BoardRange | None:
        """Return the board centre's distance from the camera by the pipeline's pose, else None.
        @args: frame: uint8 mono or BGR frame at the resolution K belongs to.
        """
        dist = _NO_DIST if dist is None else np.asarray(dist, dtype=np.float64)
        result = self._model_thread.submit(self._pipe.run_inference, frame,
                                           K=np.asarray(K, dtype=np.float64), dist=dist).result()
        if result["rvec"] is None or result["tvec"] is None:
            return None
        rotation = cv2.Rodrigues(np.asarray(result["rvec"], dtype=np.float64).reshape(3, 1))[0]
        half = (self._pipe.board.n + 1) / 2.0 * (self._pipe.board.square_length_m or 1.0)
        centre = (rotation @ np.array([half, half, 0.0])
                  + np.asarray(result["tvec"], dtype=np.float64).reshape(3))
        return BoardRange(z=float(centre[2]), range=float(np.linalg.norm(centre)))
