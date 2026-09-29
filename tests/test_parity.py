"""Parity of the numpy/ONNX port (src/inference.py) against the torch reference
(Conv-ChArT/dcc/pipeline.py). Needs the GPU container: real .onnx checkpoints run
through CUDA, torch on CPU for the reference decode.

    cd /workspace/Conv-ChArT-Wireless-Inference && python3 -m pytest tests/ -q
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Conv-ChArT"))

import src.inference as port          # noqa: E402
from dcc import pipeline as ref       # noqa: E402
from dcc.board import render_board    # noqa: E402

CFG = ROOT / "cfg" / "cfg.yaml"
SENSOR = (1200, 1600)                 # native OV2311 frame the Pi will send
K_TEST = np.array([[1400.0, 0.0, 799.5], [0.0, 1400.0, 599.5], [0.0, 0.0, 1.0]])
DIST_TEST = np.array([-0.05, 0.01, 0.0005, -0.0003, 0.0])


@pytest.fixture(scope="module")
def pipe():
    return port.inference_pipeline(str(CFG))


@pytest.fixture(scope="module")
def frame():
    """A rendered 5x5 ChArUco board projected into a native-resolution frame under a KNOWN
    rigid pose: H = K [r1 r2 t] S, with S mapping board pixels to the unitless lattice
    (square_length_m is null in cfg, so tvec comes back in units of squares). Returns
    (frame, ground-truth corners in sensor px, rvec_gt, tvec_gt)."""
    rng = np.random.default_rng(0)
    res = 480
    img, corners = render_board(res)
    sq = res // 5
    S = np.array([[1 / sq, 0, 0.5 / sq], [0, 1 / sq, 0.5 / sq], [0, 0, 1]])
    rvec_gt = np.array([0.35, -0.25, 0.15])
    tvec_gt = np.array([-2.6, -2.2, 14.0])
    R, _ = cv2.Rodrigues(rvec_gt)
    Hw = K_TEST @ np.column_stack([R[:, 0], R[:, 1], tvec_gt]) @ S
    canvas = np.full(SENSOR, 140, np.uint8)
    warped = cv2.warpPerspective(img, Hw, (SENSOR[1], SENSOR[0]), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_TRANSPARENT, dst=canvas.copy())
    warped = cv2.GaussianBlur(warped, (0, 0), 1.2)
    warped = np.clip(warped.astype(np.float32) * 0.8 + rng.normal(0, 4, SENSOR), 0, 255).astype(np.uint8)
    gt = cv2.perspectiveTransform(corners.reshape(-1, 1, 2).astype(np.float64), Hw).reshape(-1, 2)
    return warped, gt, rvec_gt, tvec_gt


class _Shim(torch.nn.Module):
    """Wraps an onnx_session so dcc.pipeline.detect() can call it like the torch model."""

    def __init__(self, session, pad_batch=None):
        super().__init__()
        self.session, self.pad_batch = session, pad_batch
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = x.detach().cpu().numpy().astype(np.float32)
        if self.pad_batch is None:
            hm, cls = self.session(x)
            return torch.from_numpy(hm), torch.from_numpy(cls)
        buf = np.zeros((self.pad_batch, *x.shape[1:]), np.float32)
        buf[:len(x)] = x
        return torch.from_numpy(self.session(buf)[0][:len(x)])


def ref_cfg(pipe):
    p = pipe.pipeline
    return {"tau_hm": p.tau_hm, "tau_id": p.tau_id, "lattice_tol_px": p.lattice_tol_px,
            "input_size": [pipe.W_in, pipe.H_in], "board": {"squares": list(pipe.board.squares),
                                                           "square_length_m": pipe.board.square_length_m},
            "refine_min_peak": p.refine_min_peak, "refiner_crop_alpha": p.refiner_crop_alpha,
            "refiner_crop_min_px": p.refiner_crop_min_px}


def test_onnx_io_contract(pipe):
    assert (pipe.H_in, pipe.W_in) == (480, 640)
    assert pipe.refiner.input_shapes[0] == [port.TOP_K, 1, 24, 24]
    assert pipe.detector.output_shapes == [[1, 1, 480, 640], [1, 16, 120, 160]]


def test_peaks_random_heatmaps():
    rng = np.random.default_rng(1)
    for trial in range(20):
        hm = rng.random((60, 80), dtype=np.float32)
        hm[rng.random((60, 80)) < 0.7] = 0.0
        hm[5:8, 5:8] = 0.9          # plateau: all ties must survive peaks() and be deduped by merge_close
        hm[0, 0] = hm[59, 79] = 0.95  # border maxima (padding semantics)
        a_xy, a_s = port.peaks(hm, 0.3)
        b_xy, b_s = ref.peaks(hm, 0.3)
        np.testing.assert_array_equal(a_xy, b_xy)
        np.testing.assert_array_equal(a_s, b_s)
        m_a = port.merge_close(a_xy, a_s)
        m_b = ref.merge_close(b_xy, b_s)
        np.testing.assert_array_equal(m_a[0], m_b[0])


def test_soft_argmax_random_maps():
    rng = np.random.default_rng(2)
    maps = rng.random((40, 64, 64), dtype=np.float32) * 0.2
    for i in range(40):                       # a sharp peak somewhere, including at the borders
        y, x = rng.integers(0, 64, 2) if i > 4 else [(0, 0), (63, 63), (0, 63), (63, 0), (31, 31)][i]
        yy, xx = np.mgrid[0:64, 0:64]
        maps[i] += np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 3.0).astype(np.float32)
    a, a_s = port.soft_argmax(maps)
    b, b_s = ref.soft_argmax(maps, return_spread=True)
    np.testing.assert_allclose(a, b, atol=2e-5)
    np.testing.assert_allclose(a_s, b_s, atol=2e-5)
    assert port.soft_argmax(np.zeros((0, 64, 64), np.float32)).uv.shape == (0, 2)


def test_read_ids_random_maps():
    rng = np.random.default_rng(3)
    cls = rng.random((16, 120, 160), dtype=np.float32)
    xy = np.concatenate([rng.uniform(-3, 643, (200, 1)), rng.uniform(-3, 483, (200, 1))], axis=1)  # incl. off-map
    xy = np.vstack([xy, [[-0.5, -0.5], [639.5, 479.5], [0.0, 0.0], [639.0, 479.0], [1.5, 1.5]]])
    a_idx, a_conf = port.read_ids(cls, xy)
    b_idx, b_conf = ref.read_ids(cls, xy)
    np.testing.assert_allclose(a_conf, b_conf, atol=1e-5)
    agree = a_idx == b_idx
    # only a near-tie between channels can legitimately flip the argmax across float paths
    assert agree.mean() > 0.99, f"{(~agree).sum()} id mismatches"


def test_end_to_end_matches_reference(pipe, frame):
    img, _, _, _ = frame
    res = pipe.run_inference(img, K=K_TEST, dist=DIST_TEST)
    shim_det = _Shim(pipe.detector)
    shim_ref = _Shim(pipe.refiner, pad_batch=port.TOP_K)
    exp = ref.detect(img, shim_det, shim_ref, K=K_TEST, dist=DIST_TEST, cfg=ref_cfg(pipe))

    assert res["reason"] == exp["reason"]
    assert len(res["corners"]) == len(exp["corners"])
    # peaks are score-ordered and saturated peaks can differ by one float ulp between the
    # numpy and torch sigmoids, so compare corners by coarse position, not list order
    key = lambda c: (round(c["y_coarse"], 3), round(c["x_coarse"], 3))
    for a, b in zip(sorted(res["corners"], key=key), sorted(exp["corners"], key=key)):
        assert key(a) == key(b)
        assert a["index"] == b["index"] and a["source"] == b["source"]
        assert abs(a["x"] - b["x"]) < 1e-3 and abs(a["y"] - b["y"]) < 1e-3
        assert abs(a["p_hm"] - b["p_hm"]) < 1e-5 and abs(a["p_id"] - b["p_id"]) < 1e-5
        assert (np.isnan(a["sigma_px"]) and np.isnan(b["sigma_px"])) or abs(a["sigma_px"] - b["sigma_px"]) < 1e-4
    assert res["demoted"] == exp["demoted"] and res["recovered"] == exp["recovered"]
    if exp["rvec"] is not None:
        np.testing.assert_allclose(res["rvec"], exp["rvec"], atol=1e-5)
        np.testing.assert_allclose(res["tvec"], exp["tvec"], atol=1e-5)
        assert abs(res["rms"] - exp["rms"]) < 1e-5 and res["ambiguous"] == exp["ambiguous"]


def test_synthetic_board_detected(pipe, frame):
    img, gt, rvec_gt, tvec_gt = frame
    res = pipe.run_inference(img, K=K_TEST, dist=None)
    found = {c["index"]: (c["x"], c["y"]) for c in res["corners"] if c["index"] is not None}
    errs = [np.hypot(*(np.array(found[i]) - gt[i])) for i in found]
    print(f"\n{len(res['corners'])} corners, {len(found)} identified, reason={res['reason']}, "
          f"median corner err {np.median(errs):.3f} px, PnP rms {res['rms']:.3f} px")
    assert len(found) >= 12 and all(i in range(16) for i in found)
    assert np.median(errs) < 0.5
    assert res["reason"] is None and res["rvec"] is not None and not res["ambiguous"]
    assert res["rms"] < 0.5
    R_gt, _ = cv2.Rodrigues(rvec_gt)
    R_est, _ = cv2.Rodrigues(res["rvec"])
    ang = np.degrees(np.arccos(np.clip((np.trace(R_gt.T @ R_est) - 1) / 2, -1, 1)))
    dt = np.linalg.norm(res["tvec"].ravel() - tvec_gt)
    print(f"pose error: {ang:.3f} deg, {dt:.4f} squares (|t|={np.linalg.norm(tvec_gt):.1f})")
    assert ang < 0.5 and dt < 0.05
    # both IPPE solutions come back, each with its own first-order covariance
    assert res["rvec_alt"] is not None and res["tvec_alt"] is not None and res["rms_alt"] is not None
    assert res["rms_alt"] >= res["rms"]
    assert res["pose_cov"] is not None and res["pose_cov_alt"] is not None
    cov, cov_alt = np.asarray(res["pose_cov"]), np.asarray(res["pose_cov_alt"])
    assert cov.shape == cov_alt.shape == (6, 6)
    assert np.allclose(cov, cov.T) and np.allclose(cov_alt, cov_alt.T)
    assert np.all(np.linalg.eigvalsh(cov) > 0) and np.all(np.linalg.eigvalsh(cov_alt) > 0)
    assert not np.allclose(cov, cov_alt)          # the two tilt solutions have different Jacobians


def test_no_intrinsics_refuses_pose(pipe, frame):
    img, _, _, _ = frame
    res = pipe.run_inference(img, K=None)
    assert res["rvec"] is None and res["reason"] in ("no_intrinsics", "too_few", "collinear")
