"""Online harvester of the live SSL node on a rendered board video with known geometry (ROS-free).

    python3 -m pytest src/ros/src/convchart_ros/test/test_ssl_harvester.py -q
Needs the Conv-ChArT checkout next to this repo (dcc); skipped without it.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ssl_scene import homography, MODEL_CFG, render, result_for, truth

pytest.importorskip("dcc.video", reason="needs the Conv-ChArT checkout (dcc)")
from convchart_ros.ssl_utils.harvester import HarvestParams, OnlineHarvester  # noqa: E402


def records(session):
    out = []
    for d in sorted((Path(session) / "harvest").glob("*/chunk_*")):
        out += json.loads((d / "records.json").read_text())["records"]
    return out


def test_frames_between_anchors_are_labelled_from_geometry(tmp_path):
    rng = np.random.default_rng(0)
    h = OnlineHarvester(MODEL_CFG, tmp_path, HarvestParams(chunk_size=500))
    Hs = [homography(t) for t in range(60)]
    for t, H in enumerate(Hs):
        dark = 25 <= t < 35
        img = render(H, gain=0.06 if dark else 1.0, noise=1.0, rng=rng)
        h.add(t * 0.1, img, result_for(H, anchor=(t % 10 == 0 and not dark), rng=rng))
    h.flush()
    recs = records(tmp_path)
    assert sorted(r["frame"] for r in recs) == list(range(60))
    errs = [np.hypot(c["x"] - x, c["y"] - y) for r in recs
            for c, (x, y) in zip(r["corners"], truth(Hs[r["frame"]])) if c["visible"] is True]
    assert len(errs) > 800 and max(errs) < 0.3
    tracked_dark = [r for r in recs if 25 <= r["frame"] < 35]
    assert tracked_dark and all(r["source"] == "tracked" for r in tracked_dark)


def test_a_static_board_is_banked_once(tmp_path):
    rng = np.random.default_rng(1)
    h = OnlineHarvester(MODEL_CFG, tmp_path, HarvestParams(chunk_size=500))
    H = homography(0, static=True)
    for t in range(30):
        h.add(t * 0.1, render(H, rng=rng), result_for(H, anchor=(t % 5 == 0), rng=rng))
    h.flush()
    snap = h.snapshot()
    assert snap["totals"]["labelled"] == 30 and snap["totals"]["duplicate"] == 29 and len(records(tmp_path)) == 1


def test_chunks_split_into_train_and_heldout(tmp_path):
    rng = np.random.default_rng(2)
    h = OnlineHarvester(MODEL_CFG, tmp_path, HarvestParams(chunk_size=5, heldout_every=3))
    for t in range(31):
        H = homography(t)
        h.add(t * 0.1, render(H, rng=rng), result_for(H, anchor=True, rng=rng))
    h.flush()
    chunks = h.snapshot()["chunks"]
    names = lambda split: [Path(d).name for d in chunks[split]]
    assert names("heldout") == ["chunk_0002", "chunk_0005"]
    assert names("train")[:3] == ["chunk_0000", "chunk_0001", "chunk_0003"]
    meta = json.loads((Path(chunks["train"][0]) / "records.json").read_text())["meta"]
    assert {"shot", "read_var"} <= set(meta["noise"])


def test_a_long_gap_is_closed_by_forward_backward_tracking(tmp_path):
    rng = np.random.default_rng(3)
    h = OnlineHarvester(MODEL_CFG, tmp_path, HarvestParams(max_gap=20, chunk_size=500))
    segs = []
    for t in range(25):
        H = homography(t)
        ev = h.add(t * 0.1, render(H, rng=rng), result_for(H, anchor=(t == 0), rng=rng))
        if ev["segment"]:
            segs.append(ev["segment"])
    first, gap = segs
    assert first["anchor"] == 1 and first["length"] == 1
    assert gap["tracked"] >= 15 and gap["fb"] and gap["fb"][0] < 1.0 and not gap["closures"]
    assert h.snapshot()["buffered"] < 10


def test_a_stamp_gap_breaks_the_stream(tmp_path):
    rng = np.random.default_rng(4)
    h = OnlineHarvester(MODEL_CFG, tmp_path, HarvestParams(chunk_size=500))
    Hs = [homography(t) if t < 20 else homography(t - 20, theta=0.9, tx=260.0, ty=260.0) for t in range(40)]
    breaks = []
    for t, H in enumerate(Hs):
        stamp = t * 0.1 + (5.0 if t >= 20 else 0.0)
        ev = h.add(stamp, render(H, rng=rng), result_for(H, anchor=t in (0, 10, 25, 35), rng=rng))
        if "break" in ev:
            breaks.append((t, ev["break"]))
    h.flush()
    recs = records(tmp_path)
    assert sorted(r["frame"] for r in recs) == list(range(40))
    errs = [np.hypot(c["x"] - x, c["y"] - y) for r in recs
            for c, (x, y) in zip(r["corners"], truth(Hs[r["frame"]])) if c["visible"] is True]
    assert len(errs) > 500 and max(errs) < 0.3
    (t, seg), = breaks
    assert t == 20 and seg["length"] == 10 and seg["tracked"] == 9 and seg["fb"] and not seg["closures"]
    assert h.snapshot()["totals"]["breaks"] == 1


def test_a_calibrated_lens_keeps_labels_exact_in_a_distorted_image(tmp_path):
    from dcc.video import Lens
    K = np.array([[400.0, 0, 319.5], [0, 400.0, 239.5], [0, 0, 1.0]])
    D = np.array([-0.25, 0.08, 0.0, 0.0, 0.0])
    lens = Lens(K, D, 1.0, (640, 480))
    grid = np.stack(np.meshgrid(np.arange(640), np.arange(480)), -1).reshape(-1, 1, 2).astype(np.float64)
    und = cv2.undistortPoints(grid, K, D, P=K).reshape(480, 640, 2).astype(np.float32)

    def frame(H, rng):
        img = cv2.remap(render(H, noise=0.0).astype(np.float32), und[..., 0], und[..., 1], cv2.INTER_LINEAR)
        return np.clip(np.rint(img + rng.normal(0, 1.0, img.shape)), 0, 255).astype(np.uint8)

    def detected(pts, anchor, rng):
        pts = pts + rng.normal(0, 0.05, pts.shape)
        return {"corners": [{"x": float(x), "y": float(y), "index": k, "source": "head", "p_id": 0.95,
                             "sigma_px": 0.1, "p_hm": 0.9} for k, (x, y) in enumerate(pts)] if anchor else []}

    Hs = [homography(t, s=34.0, theta=0.2, tx=150.0, ty=150.0) for t in range(40)]
    errors, labelled = {}, {}
    for name, camera in (("calibrated", {"K": K.tolist(), "dist": D.tolist(), "image_size": [640, 480]}),
                         ("uncalibrated", None)):
        rng = np.random.default_rng(5)
        h = OnlineHarvester(MODEL_CFG, tmp_path / name, HarvestParams(chunk_size=500), camera=camera)
        assert h.calibrated == (camera is not None)
        for t, H in enumerate(Hs):
            h.add(t * 0.1, frame(H, rng), detected(lens.distort_points(truth(H)), t % 5 == 0, rng))
        h.flush()
        recs = records(tmp_path / name)
        labelled[name] = len(recs)
        errors[name] = [np.hypot(c["x"] - x, c["y"] - y) for r in recs
                        for c, (x, y) in zip(r["corners"], lens.distort_points(truth(Hs[r["frame"]])))
                        if c["visible"] is True]
    assert labelled["calibrated"] == 40 and len(errors["calibrated"]) > 500 and max(errors["calibrated"]) < 0.2
    assert labelled["uncalibrated"] < 40 and np.median(errors["uncalibrated"]) > 0.2


def test_consecutive_anchors_are_each_verified_once(tmp_path, monkeypatch):
    import dcc.video as V
    calls = []
    real = V.ecc_align
    monkeypatch.setattr(V, "ecc_align", lambda *a, **k: calls.append(1) or real(*a, **k))
    rng = np.random.default_rng(6)
    h = OnlineHarvester(MODEL_CFG, tmp_path, HarvestParams(chunk_size=500))
    Hs = [homography(t) for t in range(12)]
    for t, H in enumerate(Hs):
        h.add(t * 0.1, render(H, rng=rng), result_for(H, anchor=True, rng=rng))
    h.flush()
    assert len(calls) == 12
    recs = records(tmp_path)
    errs = [np.hypot(c["x"] - x, c["y"] - y) for r in recs
            for c, (x, y) in zip(r["corners"], truth(Hs[r["frame"]])) if c["visible"] is True]
    assert sorted(r["frame"] for r in recs) == list(range(12)) and max(errs) < 0.3


def test_without_cap_or_motion_filter_every_labelled_frame_is_kept(tmp_path):
    rng = np.random.default_rng(1)
    h = OnlineHarvester(MODEL_CFG, tmp_path, HarvestParams(chunk_size=500, min_motion_px=0.0, cap_per_bucket=0))
    H = homography(0, static=True)
    for t in range(30):
        h.add(t * 0.1, render(H, rng=rng), result_for(H, anchor=(t % 5 == 0), rng=rng))
    h.flush()
    totals = h.snapshot()["totals"]
    assert totals["labelled"] == totals["kept"] == 30 and not totals.get("duplicate") and not totals.get("capped")
    assert len(records(tmp_path)) == 30


def test_a_time_block_split_puts_a_frame_in_the_same_split_whoever_labels_it(tmp_path):
    rng = np.random.default_rng(4)
    frames = [(t * 0.1, homography(t)) for t in range(60)]
    images = [render(H, rng=rng) for _, H in frames]
    splits = {}
    for name, every in (("all", 1), ("some", 3)):
        params = HarvestParams(chunk_size=4, min_motion_px=0.0, cap_per_bucket=0, split_block_s=1.0, heldout_every=3)
        h = OnlineHarvester(MODEL_CFG, tmp_path / name, params)
        for t, ((stamp, H), img) in enumerate(zip(frames, images)):
            h.add(stamp, img, result_for(H, anchor=(t % every == 0), rng=rng))
        h.flush()
        for split in ("train", "heldout"):
            for d in h.snapshot()["chunks"][split]:
                for r in json.loads((Path(d) / "records.json").read_text())["records"]:
                    splits.setdefault(r["frame"], set()).add(split)
                    assert (split == "heldout") == (int(r["stamp"] // 1.0) % 3 == 2), r["stamp"]
    assert all(len(s) == 1 for s in splits.values()) and {"train", "heldout"} == set().union(*splits.values())


def test_two_harvesters_share_a_bank_under_their_own_prefixes(tmp_path):
    rng = np.random.default_rng(5)
    params = HarvestParams(chunk_size=5, min_motion_px=0.0, cap_per_bucket=0, split_block_s=1.0, heldout_every=4)
    hs = {p: OnlineHarvester(MODEL_CFG, tmp_path, params, bank="gen2", prefix=p) for p in ("L", "R")}
    for t in range(40):
        H = homography(t)
        hs["L" if t >= 20 else "R"].add(t * 0.1, render(H, rng=rng), result_for(H, anchor=True, rng=rng))
    for h in hs.values():
        h.flush()
    for p, h in hs.items():
        chunks = h.snapshot()["chunks"]
        dirs = [Path(d) for split in ("train", "heldout") for d in chunks[split]]
        assert dirs and all(d.parent.parent == tmp_path / "harvest" / "gen2" and d.name.startswith(f"chunk_{p}")
                            for d in dirs)
    names = [d.name for d in (tmp_path / "harvest" / "gen2").glob("*/chunk_*")]
    assert len(names) == len(set(names)) and sum(h.snapshot()["banked"]["train"] + h.snapshot()["banked"]["heldout"]
                                                 for h in hs.values()) == 40
