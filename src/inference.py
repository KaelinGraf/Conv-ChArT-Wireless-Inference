from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple, Sequence, TypeAlias, TypedDict

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as rt
import yaml

rt.preload_dlls()

PREFERRED_CUDA_DEVICE: Final[int] = 0
PINNED_CFG: Final[str] = "cfg/cfg.yaml"
TOP_K: Final[int] = 64
CROP_UPSAMPLE: Final[int] = cv2.INTER_LANCZOS4

F32: TypeAlias = npt.NDArray[np.float32]
F64: TypeAlias = npt.NDArray[np.float64]
I64: TypeAlias = npt.NDArray[np.int64]
Bool: TypeAlias = npt.NDArray[np.bool_]
U8: TypeAlias = npt.NDArray[np.uint8]
AnyArray: TypeAlias = npt.NDArray[Any]
Shape: TypeAlias = list[int]
Reason: TypeAlias = Literal["too_few", "collinear", "vacuous", "no_intrinsics", "vacuous_uncorroborated",
                            "too_few_correspondences", "pnp_solver_failed"]
Source: TypeAlias = Literal["head", "recovered"]

_ORT_TO_NP_DTYPE: Final[dict[str, type[np.generic]]] = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int8)": np.int8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(uint32)": np.uint32,
    "tensor(uint64)": np.uint64,
    "tensor(bool)": np.bool_,
}


# --- configuration -----------------------------------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    tau_hm: float = 0.3
    tau_id: float = 0.5
    lattice_tol_px: float = 3.0
    refine_min_peak: float = 0.3
    refiner_crop_alpha: float = 0.0
    refiner_crop_min_px: int = 12
    id_readout: Literal["coarse", "refined"] = "coarse"


@dataclass(frozen=True)
class BoardConfig:
    squares: tuple[int, int] = (5, 5)
    square_length_m: float | None = None

    @property
    def n(self) -> int:
        nx, ny = self.squares
        assert nx == ny, f"rectangular boards are unsupported, got squares [{nx}, {ny}]"
        return nx - 1


@dataclass(frozen=True)
class CameraConfig:
    K: F64 | None = None
    dist: F64 | None = None

    @staticmethod
    def from_yaml(cam: dict[str, Any] | None) -> CameraConfig:
        cam = cam or {}
        K = None if cam.get("K") is None else np.asarray(cam["K"], dtype=np.float64).reshape(3, 3)
        dist = None if cam.get("dist") is None else np.asarray(cam["dist"], dtype=np.float64).ravel()
        return CameraConfig(K=K, dist=dist)


# --- typed results -------------------------------------------------------------

class Corner(TypedDict):
    x: float
    y: float
    index: int | None
    x_coarse: float
    y_coarse: float
    source: Source | None
    p_hm: float
    p_id: float
    sigma_px: float


class InferenceResult(TypedDict):
    rvec: F64 | None
    tvec: F64 | None
    rms: float | None
    reason: Reason | None
    corners: list[Corner]
    pose_cov: list[list[float]] | None
    ambiguous: bool
    rvec_alt: F64 | None
    tvec_alt: F64 | None
    rms_alt: float | None
    pose_cov_alt: list[list[float]] | None
    demoted: int
    recovered: int


class Peaks(NamedTuple):
    xy: I64
    scores: F64


class Crops(NamedTuple):
    crops: F32
    centres: I64
    kept_mask: Bool
    extents: I64


class SoftArgmax(NamedTuple):
    uv: F64
    spread: F64


class Ids(NamedTuple):
    idx: I64
    conf: F64


class LatticeGate(NamedTuple):
    H: F64 | None
    inlier_mask: Bool
    demoted_mask: Bool
    degenerate: Reason | None


class Recovery(NamedTuple):
    idx: I64
    recovered_mask: Bool
    corroborated: bool


class PnP(NamedTuple):
    rvec: F64 | None
    tvec: F64 | None
    rms: float | None
    ambiguous: bool
    n_used: int
    reason: Reason | None
    cov: F64 | None
    rvec_alt: F64 | None
    tvec_alt: F64 | None
    rms_alt: float | None
    cov_alt: F64 | None


# --- ONNX Runtime session with pre-bound GPU buffers -------------------------------

class onnx_session:
    def __init__(self, model_path: str, batch: int | None = None) -> None:
        assert model_path and os.path.exists(model_path), f"Model not found at {model_path}"
        self.session = rt.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self._i_list = self.session.get_inputs()
        self._o_list = self.session.get_outputs()
        self.input_shapes: list[Shape] = [_fix_shape(i.shape, batch) for i in self._i_list]
        self.output_shapes: list[Shape] = [_fix_shape(o.shape, batch) for o in self._o_list]
        self.input_dtypes: list[type[np.generic]] = [_ORT_TO_NP_DTYPE[i.type] for i in self._i_list]
        assert rt.get_device() == "GPU", "Onnx provider fell back to CPU, CUDA unavailable"
        self._io = self.session.io_binding()

        self._i = [rt.OrtValue.ortvalue_from_numpy(np.zeros(s, d), device_type="cuda", device_id=PREFERRED_CUDA_DEVICE)
                   for s, d in zip(self.input_shapes, self.input_dtypes)]
        for j, _i in enumerate(self._i):
            self._io.bind_ortvalue_input(self._i_list[j].name, _i)

        self._o = [rt.OrtValue.ortvalue_from_shape_and_type(s, _ORT_TO_NP_DTYPE[o.type], device_type="cuda",
                                                            device_id=PREFERRED_CUDA_DEVICE)
                   for s, o in zip(self.output_shapes, self._o_list)]
        for j, _o in enumerate(self._o):
            self._io.bind_ortvalue_output(self._o_list[j].name, _o)

    def __call__(self, inputs: AnyArray | Sequence[AnyArray]) -> list[AnyArray]:
        arrays: Sequence[AnyArray] = [inputs] if isinstance(inputs, np.ndarray) else inputs
        for j, _i in enumerate(self._i):
            _i.update_inplace(np.ascontiguousarray(arrays[j], dtype=self.input_dtypes[j]))
        self.session.run_with_iobinding(self._io)
        return [_o.numpy() for _o in self._o]


def _fix_shape(shape: Sequence[int | str | None], batch: int | None) -> Shape:
    fixed = [d if isinstance(d, int) else (batch if k == 0 and batch is not None else d) for k, d in enumerate(shape)]
    assert all(isinstance(d, int) for d in fixed), f"Dynamic model shape {list(shape)} - pass batch="
    return [int(d) for d in fixed]  # type: ignore[arg-type]


def _sigmoid(x: AnyArray) -> F32:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float32)))


# --- stage 3 decode chain (numpy / OpenCV port of dcc/pipeline.py) ---------------------

def canon_lattice(n: int) -> F64:
    return np.array([[(i % n) + 1.0, (i // n) + 1.0] for i in range(n * n)], dtype=np.float64)


_CANON: Final[F64] = canon_lattice(4)
_NO_DIST: Final[F64] = np.zeros(0, dtype=np.float64)


def peaks(hm_sigmoid: AnyArray, tau_hm: float, top_k: int = TOP_K) -> Peaks:
    hm: F32 = np.ascontiguousarray(hm_sigmoid, dtype=np.float32)
    pooled: F32 = np.asarray(cv2.dilate(hm, np.ones((3, 3), np.uint8)), dtype=np.float32)
    ys, xs = np.nonzero((hm == pooled) & (hm >= tau_hm))
    scores = hm[ys, xs].astype(np.float64)
    xy = np.stack([xs, ys], axis=1).astype(np.int64)
    order = np.lexsort((xy[:, 0], xy[:, 1], -scores))[:top_k]
    return Peaks(xy[order], scores[order])


def merge_close(xy: AnyArray, scores: AnyArray, radius: float = 2.0) -> Peaks:
    xy = np.asarray(xy, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((xy[:, 0], xy[:, 1], -scores))
    kept: list[int] = []
    for i in order:
        if all(np.hypot(*(xy[i] - xy[j])) > radius for j in kept):
            kept.append(int(i))
    keep = np.array(kept, dtype=np.int64)
    return Peaks(xy[keep], scores[keep])


def spacing_estimate(xy: AnyArray) -> float | None:
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return None
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def crop_extent(s_sensor: float | AnyArray, alpha: float = 0.5, e_min: int = 16) -> I64:
    e = 2.0 * np.rint(alpha * np.asarray(s_sensor, dtype=np.float64) / 2.0)
    return np.clip(e, e_min, 24).astype(np.int64)


def refined_offset(u_star: float | AnyArray, E: int | AnyArray) -> F64:
    k = np.asarray(E, dtype=np.float64) / 24.0
    if k.ndim:
        k = k.reshape(-1, 1)
    return ((np.asarray(u_star, dtype=np.float64) + 0.5) / 8.0 + 8.0 - 11.5) * k - 0.5


def refiner_support(E: int) -> tuple[float, float]:
    return float(refined_offset(0.0, E)), float(refined_offset(63.0, E))


def cut_crops(frame_sensor: U8, peaks_input: AnyArray, r: float, extent: int | AnyArray = 24) -> Crops:
    Hs, Ws = frame_sensor.shape
    pts = np.asarray(peaks_input, dtype=np.float64).reshape(-1, 2)
    centre = np.rint((pts + 0.5) / r - 0.5).astype(np.int64)
    E = np.broadcast_to(np.asarray(extent, dtype=np.int64).reshape(-1), (len(centre),))
    h, cx, cy = E // 2, centre[:, 0], centre[:, 1]
    kept_mask = (cx - h >= 0) & (cx + h <= Ws) & (cy - h >= 0) & (cy + h <= Hs)
    idxs = np.nonzero(kept_mask)[0]
    crops = np.zeros((len(idxs), 24, 24), dtype=np.float32)
    for k, i in enumerate(idxs):
        c = frame_sensor[cy[i] - h[i]:cy[i] + h[i], cx[i] - h[i]:cx[i] + h[i]].astype(np.float32)
        crops[k] = c if E[i] == 24 else cv2.resize(c, (24, 24), interpolation=CROP_UPSAMPLE)
    return Crops(crops[:, None] / 255.0, centre[idxs], kept_mask, E[idxs])


def soft_argmax(ref_sigmoid: AnyArray) -> SoftArgmax:
    t = np.asarray(ref_sigmoid, dtype=np.float32).reshape(-1, 64, 64)
    n = t.shape[0]
    if n == 0:
        return SoftArgmax(np.zeros((0, 2), np.float64), np.zeros(0, np.float64))
    flat = t.reshape(n, -1).argmax(axis=1)
    y0 = np.clip(flat // 64 - 2, 0, 59)
    x0 = np.clip(flat % 64 - 2, 0, 59)
    rows = y0[:, None] + np.arange(5)[None, :]
    cols = x0[:, None] + np.arange(5)[None, :]
    w = t[np.arange(n)[:, None, None], rows[:, :, None], cols[:, None, :]].astype(np.float64)
    wsum = np.maximum(w.sum(axis=(1, 2)), 1e-12)
    gy = rows[:, :, None].astype(np.float64)
    gx = cols[:, None, :].astype(np.float64)
    u = (w * gx).sum(axis=(1, 2)) / wsum
    v = (w * gy).sum(axis=(1, 2)) / wsum
    var = ((w * (gx - u[:, None, None]) ** 2).sum(axis=(1, 2))
           + (w * (gy - v[:, None, None]) ** 2).sum(axis=(1, 2))) / wsum
    return SoftArgmax(np.stack([u, v], axis=1), np.sqrt(np.maximum(var, 0.0)) / 8.0)


def read_ids(cls_sigmoid: AnyArray, xy_input: AnyArray) -> Ids:
    cls = np.asarray(cls_sigmoid, dtype=np.float32)
    _, H4, W4 = cls.shape
    xy = np.asarray(xy_input, dtype=np.float32).reshape(-1, 2)
    if len(xy) == 0:
        return Ids(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64))
    cx = np.clip((xy[:, 0] + 0.5) / 4.0 - 0.5, 0.0, W4 - 1.0)
    cy = np.clip((xy[:, 1] + 0.5) / 4.0 - 0.5, 0.0, H4 - 1.0)
    x0 = np.floor(cx).astype(np.int64)
    y0 = np.floor(cy).astype(np.int64)
    x1 = np.minimum(x0 + 1, W4 - 1)
    y1 = np.minimum(y0 + 1, H4 - 1)
    fx = (cx - x0).astype(np.float32)
    fy = (cy - y0).astype(np.float32)
    s = (cls[:, y0, x0] * (1 - fx) * (1 - fy) + cls[:, y0, x1] * fx * (1 - fy)
         + cls[:, y1, x0] * (1 - fx) * fy + cls[:, y1, x1] * fx * fy)
    return Ids(s.argmax(axis=0).astype(np.int64), s.max(axis=0).astype(np.float64))


def undistort(xy_sensor: AnyArray, K: F64, dist: F64 | None) -> F64:
    pts = np.asarray(xy_sensor, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return pts.copy()
    out = cv2.undistortPoints(pts.astype(np.float32).reshape(-1, 1, 2), K, _NO_DIST if dist is None else dist, P=K)
    return out.reshape(-1, 2).astype(np.float64)


def lattice_gate(xy: AnyArray, idx: AnyArray, conf: AnyArray, tol: float = 3.0, n: int = 4) -> LatticeGate:
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    index = np.asarray(idx, dtype=np.int64).reshape(-1)
    lattice = _CANON if n == 4 else canon_lattice(n)
    cnt = len(index)
    inlier_mask, demoted_mask = np.zeros(cnt, dtype=bool), np.zeros(cnt, dtype=bool)
    idd = np.nonzero(index >= 0)[0]
    if len(idd) < 4:
        return LatticeGate(None, inlier_mask, demoted_mask, "too_few")
    canon = lattice[index[idd]]
    if np.linalg.svd(canon - canon.mean(axis=0), compute_uv=False)[1] < 1e-6:
        return LatticeGate(None, inlier_mask, demoted_mask, "collinear")
    H, mask = cv2.findHomography(canon.astype(np.float32), pts[idd].astype(np.float32),
                                 method=cv2.RANSAC, ransacReprojThreshold=tol)
    if H is None:
        return LatticeGate(None, inlier_mask, demoted_mask, "collinear")
    inliers = mask.ravel().astype(bool)
    inlier_mask[idd], demoted_mask[idd] = inliers, ~inliers
    return LatticeGate(np.asarray(H, dtype=np.float64), inlier_mask, demoted_mask,
                       "vacuous" if len(idd) == 4 else None)


def recover(H: F64, xy_all: AnyArray, idx: AnyArray, conf: AnyArray, tol: float, n: int = 4) -> Recovery:
    pts = np.asarray(xy_all, dtype=np.float64).reshape(-1, 2)
    idx_in = np.asarray(idx, dtype=np.int64).reshape(-1)
    idx_out = idx_in.copy()
    recovered_mask = np.zeros(len(idx_in), dtype=bool)
    lattice = _CANON if n == 4 else canon_lattice(n)
    proj = np.hstack([lattice, np.ones((n * n, 1))]) @ H.T
    proj = proj[:, :2] / proj[:, 2:3]
    claimed: set[int] = set(idx_in[idx_in >= 0].tolist())
    for i in np.nonzero(idx_in < 0)[0]:
        d = np.linalg.norm(proj - pts[i], axis=1)
        for j in np.argsort(d):
            j = int(j)
            if d[j] > tol:
                break
            if j not in claimed:
                idx_out[i], recovered_mask[i] = j, True
                claimed.add(j)
                break
    corroborated = bool(int(np.sum(idx_in >= 0)) == 4 and recovered_mask.any())
    return Recovery(idx_out, recovered_mask, corroborated)


def pnp(xy_pinhole: AnyArray, idx: AnyArray, K: F64, square_length_m: float, n: int = 4,
        sigma_px: F64 | None = None) -> PnP:
    pts = np.asarray(xy_pinhole, dtype=np.float64).reshape(-1, 2)
    index = np.asarray(idx, dtype=np.int64).reshape(-1)
    ok = np.nonzero(index >= 0)[0]
    n_used = len(ok)
    if n_used < 4:
        return PnP(None, None, None, False, n_used, "too_few_correspondences", None, None, None, None, None)
    lattice = _CANON if n == 4 else canon_lattice(n)
    obj = (np.hstack([lattice[index[ok]], np.zeros((n_used, 1))]) * square_length_m).reshape(-1, 1, 3)
    img = pts[ok].reshape(-1, 1, 2)
    _, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, K, _NO_DIST, flags=cv2.SOLVEPNP_IPPE)
    if len(rvecs) == 0:
        return PnP(None, None, None, False, n_used, "pnp_solver_failed", None, None, None, None, None)
    err = np.asarray(errs, dtype=np.float64).ravel()
    order = np.argsort(err)
    rms = float(err[order[0]])
    ambiguous = bool(len(order) > 1 and err[order[1]] / max(rms, 1e-12) < 1.5)
    rvec = np.asarray(rvecs[order[0]], dtype=np.float64)
    tvec = np.asarray(tvecs[order[0]], dtype=np.float64)
    sig = sigma_px[ok] if sigma_px is not None else None
    cov = pose_covariance(obj, rvec, tvec, K, sig)
    if len(order) == 1:
        return PnP(rvec, tvec, rms, ambiguous, n_used, None, cov, None, None, None, None)
    alt = int(order[1])
    rvec_alt = np.asarray(rvecs[alt], dtype=np.float64)
    tvec_alt = np.asarray(tvecs[alt], dtype=np.float64)
    return PnP(rvec, tvec, rms, ambiguous, n_used, None, cov, rvec_alt, tvec_alt, float(err[alt]),
               pose_covariance(obj, rvec_alt, tvec_alt, K, sig))


def pose_covariance(obj: F64, rvec: F64, tvec: F64, K: F64, sigma_px: F64 | None = None) -> F64 | None:
    _, jac = cv2.projectPoints(obj, rvec, tvec, K, _NO_DIST)
    J = np.asarray(jac, dtype=np.float64)[:, :6]
    m = J.shape[0] // 2
    if sigma_px is None:
        JtRiJ = J.T @ J
    else:
        var = np.repeat(np.asarray(sigma_px, dtype=np.float64).reshape(m), 2) ** 2
        var = np.where(np.isfinite(var) & (var > 1e-12), var, np.nan)
        if not np.isfinite(var).all():
            return None
        JtRiJ = (J.T * (1.0 / var)) @ J
    try:
        return np.asarray(np.linalg.inv(JtRiJ), dtype=np.float64)
    except np.linalg.LinAlgError:
        return None


# --- the engine ---------------------------------------------------------------------

class inference_pipeline:
    def __init__(self, config: str) -> None:
        with open(config, "r") as file:
            cfg: dict[str, Any] = yaml.safe_load(file) or {}
        root = Path(config).resolve().parent.parent
        model: dict[str, str] = cfg.get("MODEL", {})
        self.detector = onnx_session(_resolve(model.get("detector", ""), root))
        self.refiner = onnx_session(_resolve(model.get("refiner", ""), root), batch=TOP_K)
        self.pipeline = PipelineConfig(**(cfg.get("PIPELINE") or {}))
        board = dict(cfg.get("BOARD") or {})
        if "squares" in board:
            board["squares"] = tuple(int(v) for v in board["squares"])
        self.board = BoardConfig(**board)
        self.camera = CameraConfig.from_yaml(cfg.get("CAMERA"))
        _, _, self.H_in, self.W_in = self.detector.input_shapes[0]

    def run_inference(self, img: U8, K: F64 | None = None, dist: F64 | None = None) -> InferenceResult:
        """
        Full pose pipeline on one frame: detector -> peaks -> refiner -> IDs -> undistort -> lattice gate -> recovery -> PnP.
          @args:
            img: uint8 image (H, W), (H, W, 1) or BGR (H, W, 3); native sensor resolution.
            K, dist: camera intrinsics / distortion, overriding cfg CAMERA. Without K the pose is refused.
        """
        K = self.camera.K if K is None else np.asarray(K, dtype=np.float64).reshape(3, 3)
        dist = self.camera.dist if dist is None else np.asarray(dist, dtype=np.float64).ravel()
        p, n = self.pipeline, self.board.n
        sqlen = self.board.square_length_m or 1.0

        frame_sensor = self._to_mono(img)
        Hs, Ws = frame_sensor.shape
        r = self.W_in / Ws
        frame_input = self._preprocess(frame_sensor, self.H_in, self.W_in)
        hm_logits, cls_logits = self.detector(frame_input)
        hm_sigmoid, cls_sigmoid = _sigmoid(hm_logits[0, 0]), _sigmoid(cls_logits[0])

        xy_pk, p_hm = merge_close(*peaks(hm_sigmoid, p.tau_hm))
        s_est = spacing_estimate(xy_pk) if p.refiner_crop_alpha > 0 else None
        E: int | I64 = 24 if s_est is None else crop_extent(s_est * r, p.refiner_crop_alpha, p.refiner_crop_min_px)
        crops, centres_sensor, kept_mask, E_kept = cut_crops(frame_sensor, xy_pk, r, E)
        xy_sensor = np.zeros((len(xy_pk), 2), dtype=np.float64)
        sigma_px = np.full(len(xy_pk), np.nan, dtype=np.float64)
        if len(crops):
            idxs = np.nonzero(kept_mask)[0]
            rmap = self._refine(crops)
            u_star, u_spread = soft_argmax(rmap)
            peak = rmap.reshape(rmap.shape[0], -1).max(axis=1)
            ok = peak >= p.refine_min_peak
            sel = idxs[ok]
            xy_sensor[sel] = centres_sensor[ok] + refined_offset(u_star[ok], E_kept[ok])
            sigma_px[sel] = u_spread[ok] * (E_kept[ok] / 24.0)
            kept_mask[idxs[~ok]] = False
        xy_coarse: F64 = (xy_pk + 0.5) / r - 0.5
        xy_sensor[~kept_mask] = xy_coarse[~kept_mask]

        xy_id = xy_coarse if p.id_readout == "coarse" else xy_sensor
        idx_raw, p_id = read_ids(cls_sigmoid, (xy_id + 0.5) * r - 0.5)
        idx_thr = np.where(p_id >= p.tau_id, idx_raw, -1)
        K_eff: F64 = K if K is not None else np.array([[max(Ws, Hs), 0, (Ws - 1) / 2],
                                                        [0, max(Ws, Hs), (Hs - 1) / 2], [0, 0, 1]], dtype=np.float64)
        xy_pinhole = undistort(xy_sensor, K_eff, dist)
        xy_pin_id = xy_pinhole if p.id_readout != "coarse" else undistort(xy_coarse, K_eff, dist)

        H, inlier_mask, demoted_mask, degenerate = lattice_gate(xy_pin_id, idx_thr, p_id, p.lattice_tol_px, n)
        idx_final = idx_thr.copy()
        idx_final[demoted_mask] = -1
        recovered_mask = np.zeros(len(idx_final), dtype=bool)
        pose = PnP(None, None, None, False, 0, degenerate, None, None, None, None, None)

        if H is not None:
            idx_final, recovered_mask, corroborated = recover(H, xy_pin_id, idx_final, p_id, p.lattice_tol_px, n)
            if K is None:
                pose = pose._replace(reason="no_intrinsics")
            elif degenerate == "vacuous" and not corroborated:
                pose = pose._replace(reason="vacuous_uncorroborated")
            else:
                pose = pnp(xy_pinhole, idx_final, K, sqlen, n, sigma_px=sigma_px)

        corners: list[Corner] = []
        for i in range(len(idx_final)):
            idx_i: int | None = int(idx_final[i]) if idx_final[i] >= 0 else None
            source: Source | None = None if idx_i is None else ("recovered" if recovered_mask[i] else "head")
            corners.append(Corner(x=float(xy_sensor[i, 0]), y=float(xy_sensor[i, 1]), index=idx_i,
                                  x_coarse=float(xy_coarse[i, 0]), y_coarse=float(xy_coarse[i, 1]),
                                  source=source, p_hm=float(p_hm[i]), p_id=float(p_id[i]),
                                  sigma_px=float(sigma_px[i])))
        return InferenceResult(rvec=pose.rvec, tvec=pose.tvec, rms=pose.rms, reason=pose.reason, corners=corners,
                               pose_cov=None if pose.cov is None else pose.cov.tolist(),
                               ambiguous=pose.ambiguous, rvec_alt=pose.rvec_alt, tvec_alt=pose.tvec_alt,
                               rms_alt=pose.rms_alt,
                               pose_cov_alt=None if pose.cov_alt is None else pose.cov_alt.tolist(),
                               demoted=int(demoted_mask.sum()), recovered=int(recovered_mask.sum()))

    def _refine(self, crops: F32) -> F32:
        B = self.refiner.input_shapes[0][0]
        out = np.zeros((len(crops), 64, 64), dtype=np.float32)
        for s in range(0, len(crops), B):
            chunk = crops[s:s + B]
            buf = np.zeros((B, 1, 24, 24), dtype=np.float32)
            buf[:len(chunk)] = chunk
            out[s:s + len(chunk)] = self.refiner(buf)[0][:len(chunk), 0]
        return _sigmoid(out)

    @staticmethod
    def _to_mono(img: AnyArray) -> U8:
        if img.ndim == 3 and img.shape[2] == 3:
            img = np.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]
        assert img.ndim == 2 and img.dtype == np.uint8, f"expected a uint8 mono frame, got {img.shape} {img.dtype}"
        return img

    def _preprocess(self, img: U8, H: int, W: int) -> F32:
        """
        Sensor frame -> detector input (1, 1, H, W) float32 in [0, 1].
          @args:
            img: uint8 mono frame.
            H: Target height.
            W: Target width.
        """
        if img.shape != (H, W):
            img = np.asarray(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA), dtype=np.uint8)
        return img[None, None].astype(np.float32) / 255.0


def _resolve(path: str, root: Path) -> str:
    return path if os.path.isabs(path) or os.path.exists(path) else str(root / path)


def main() -> None:
    cfg = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith((".yaml", ".yml")) else PINNED_CFG
    images = [a for a in sys.argv[1:] if not a.endswith((".yaml", ".yml"))]
    pipe = inference_pipeline(cfg)
    print(f"detector {pipe.W_in}x{pipe.H_in}, refiner batch {pipe.refiner.input_shapes[0][0]}, "
          f"K={'cfg' if pipe.camera.K is not None else 'none'}")
    for path in images:
        raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        assert raw is not None, f"could not read {path}"
        img = np.asarray(raw, dtype=np.uint8)
        res = pipe.run_inference(img)
        ids = sorted(c["index"] for c in res["corners"] if c["index"] is not None)
        print(f"{path}: {len(res['corners'])} corners, ids {ids}, reason={res['reason']}, "
              f"rms={res['rms']}, ambiguous={res['ambiguous']}")
        if res["rvec"] is not None and res["tvec"] is not None:
            print("  rvec", res["rvec"].ravel(), "tvec", res["tvec"].ravel())


if __name__ == "__main__":
    main()
