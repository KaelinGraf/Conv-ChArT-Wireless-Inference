"""Round runner of the live SSL node against a fake Conv-ChArT: phases, the acceptance gate, failures (ROS-free).

    python3 -m pytest src/ros/src/convchart_ros/test/test_ssl_rounds.py -q
"""
import json
import os
from pathlib import Path
import stat
import sys
import time

import numpy as np

import pytest
import yaml

from convchart_ros.ssl_utils.rounds import evaluate_onnx, round_steps, RoundConfig, RoundRunner

FAKE = r'''#!{python}
import json, os, sys, time
from pathlib import Path
import yaml
args = sys.argv[1:]
opt = lambda k: args[args.index(k) + 1]
fail = os.environ.get("FAKE_FAIL", "")
if Path(args[0]).name == "train_detector.py":
    run = Path("runs") / opt("--name")
    run.mkdir(parents=True, exist_ok=True)
    time.sleep(float(os.environ.get("FAKE_SLEEP", "0")))
    steps = int(yaml.safe_load(open(opt("--config")))["train"]["steps"])
    (run / "metrics.jsonl").write_text("".join(json.dumps({{"step": s, "loss": 1 / s}}) + "\n" for s in range(1, steps + 1)))
    (run / f"ckpt_{{steps:07d}}.pt").write_text("ckpt")
    sys.exit(0)
cmd = args[1]
if cmd == fail:
    sys.exit(3)
if cmd in ("import-onnx", "export-onnx"):
    Path(opt("--out")).write_text(cmd)
elif cmd == "finetune-config":
    steps, frac = opt("--steps"), opt("--real-frac")
    Path(opt("--out")).write_text("train: {{steps: " + steps + "}}\nreal: {{frac: " + frac + "}}\n")
elif cmd == "synth-val":
    m04 = 0.99 if "base" in opt("--detector") else float(os.environ.get("FAKE_CAND_M04", "0.99"))
    Path(opt("--out")).write_text(json.dumps({{"m01": {{"median": 0.4}}, "m02": {{}}, "m04": {{"accuracy": m04}}}}))
'''


def make_runner(tmp_path, recall, calls=None):
    root = tmp_path / "convchart"
    (root / "tools").mkdir(parents=True)
    for f in ("video_bootstrap.py", "train_detector.py"):
        (root / "tools" / f).write_text("")
    fake = tmp_path / "fake_python"
    fake.write_text(FAKE.format(python=sys.executable))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    chunk = tmp_path / "chunk"
    chunk.mkdir()
    (chunk / "records.json").write_text(json.dumps({"meta": {}, "records": [{}] * 7}))
    cfg = RoundConfig(convchart_root=str(root), python=str(fake), base_onnx="base.onnx", model_config="m.yaml",
                      workdir=str(tmp_path / "rounds"), backgrounds="bgs", workers=2)

    def evaluate(onnx, dirs):
        return {"all": {"recall": 0.90 if onnx == "base.onnx" else recall(onnx), "id_precision": 0.99}}
    hooks = {} if calls is None else {"on_start": lambda: calls.append("start"), "on_end": lambda: calls.append("end")}
    return RoundRunner(cfg, evaluate, log=lambda m: None, **hooks), chunk


def run(rr, chunk):
    assert rr.start([str(chunk)], [str(chunk)])
    rr.thread.join(timeout=60)
    return rr.status()["history"][-1]


def test_better_candidate_is_accepted_and_becomes_the_model_to_beat(tmp_path, monkeypatch):
    calls = []
    rr, chunk = make_runner(tmp_path, recall=lambda onnx: 0.95 if "round1" in onnx else 0.93, calls=calls)
    taken = tmp_path / "convchart" / "runs" / f"ssl_{tmp_path.name}_r1"
    taken.mkdir(parents=True)
    r1 = run(rr, chunk)
    assert r1["run"] == taken.name + "_2" and not any(taken.iterdir())
    assert r1["accepted"] and all(r1["gate"].values()) and r1["train_frames"] == 7
    assert calls == ["start", "end"] and r1["t_start"] <= r1["t_end"] and rr.last_failure is None
    assert rr.status()["best_onnx"].endswith("round1/detector.onnx")
    r2 = run(rr, chunk)
    assert not r2["accepted"] and r2["reference_onnx"].endswith("round1/detector.onnx")
    assert not r2["gate"]["heldout_recall"] and rr.status()["best_onnx"].endswith("round1/detector.onnx")
    assert Path(r1["metrics"]).exists() and (tmp_path / "rounds" / "round2" / "result.json").exists()


def test_synthetic_regression_blocks_a_candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CAND_M04", "0.95")
    rr, chunk = make_runner(tmp_path, recall=lambda onnx: 0.99)
    r = run(rr, chunk)
    assert not r["accepted"] and r["gate"]["heldout_recall"] and not r["gate"]["synth_m04"]


def test_a_failing_step_is_reported_not_raised(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "export-onnx")
    calls = []
    rr, chunk = make_runner(tmp_path, recall=lambda onnx: 0.99, calls=calls)
    r = run(rr, chunk)
    assert not r["accepted"] and "CalledProcessError" in r["error"] and rr.status()["phase"] == "idle"
    assert calls == ["start", "end"] and rr.last_failure is not None
    assert os.path.exists(tmp_path / "rounds" / "round1" / "round.log")


def test_stop_terminates_the_running_round(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SLEEP", "30")
    calls = []
    rr, chunk = make_runner(tmp_path, recall=lambda onnx: 0.99, calls=calls)
    assert rr.start([str(chunk)], [str(chunk)])
    deadline = time.time() + 30
    while rr.status()["phase"] != "train" and time.time() < deadline:
        time.sleep(0.05)
    time.sleep(0.5)
    t0 = time.time()
    rr.stop()
    rr.thread.join(timeout=20)
    r = rr.status()["history"][-1]
    assert time.time() - t0 < 10 and not rr.busy and "exited" in r["error"] and calls == ["start", "end"]
    assert not rr.start([str(chunk)], [str(chunk)])


def test_held_out_groups_follow_the_coverage_maps_brightness_buckets(tmp_path):
    levels = [10.0, 30.0, 59.0, 200.0]
    corners = [{"index": k, "x": 10.0 * k, "y": 5.0, "visible": True} for k in range(4)]
    (tmp_path / "records.json").write_text(json.dumps(
        {"meta": {}, "records": [{"board_level": v, "corners": corners} for v in levels]}))
    np.save(tmp_path / "images.npy", np.zeros((len(levels), 4, 4), np.uint8))

    class Perfect:
        def run_inference(self, img):
            return {"corners": [{"index": c["index"], "x": c["x"], "y": c["y"]} for c in corners]}

    out = evaluate_onnx(Perfect(), [str(tmp_path)])
    assert {g: out[g]["frames"] for g in ("all", "dark", "dim", "bright")} == {"all": 4, "dark": 1, "dim": 2, "bright": 1}
    assert out["all"]["recall"] == 1.0 and out["all"]["id_precision"] == 1.0


def test_a_round_uses_every_banked_frame_at_least_passes_times_padded_with_synthetic(tmp_path):
    rr, chunk = make_runner(tmp_path, recall=lambda onnx: 0.99)
    r = run(rr, chunk)
    assert r["accepted"] and r["steps"] == round_steps(7, rr.cfg) and r["passes"] >= rr.cfg.passes
    cfg = yaml.safe_load((tmp_path / "rounds" / "round1" / "finetune.yaml").read_text())
    assert cfg["train"]["steps"] == r["steps"] and cfg["real"] == {"frac": 0.6, "sampling": "epoch"}
    assert (tmp_path / "convchart" / "runs" / r["run"] / f"ckpt_{r['steps']:07d}.pt").exists()


@pytest.mark.parametrize("workers", [1, 3, 8])
def test_round_steps_give_every_worker_share_its_passes_and_grow_with_the_bank(workers):
    c = RoundConfig(convchart_root="", python="", base_onnx="", model_config="", workdir="", backgrounds="",
                    workers=workers)
    assert c.real_frac == 0.6 and c.passes == 8 and c.sampling == "epoch"
    last = 0
    for n in list(range(1, 200)) + [720, 1080, 4321]:
        steps = round_steps(n, c)
        share = -(-n // workers) if n >= workers else n
        least = steps * c.accum // workers
        assert int(c.real_frac * c.batch * least + 1e-9) >= c.passes * share, (n, steps)
        if n >= workers:
            assert steps >= last
            last = steps
    assert round_steps(1080, RoundConfig("", "", "", "", "", "", workers=8)) == 1204
    assert round_steps(10, RoundConfig("", "", "", "", "", "", min_steps=500)) == 500


def test_a_restored_runner_carries_on_the_numbering_and_the_model_to_beat(tmp_path):
    rr, chunk = make_runner(tmp_path, recall=lambda onnx: 0.99)
    rr.restore([{"round": 1, "train_frames": 480}, {"round": 3, "train_frames": 1080}], "best.onnx")
    assert rr.status()["round"] == 3 and rr.status()["best_onnx"] == "best.onnx" and len(rr.status()["history"]) == 2
    r = run(rr, chunk)
    assert r["round"] == 4 and r["reference_onnx"] == "best.onnx" and (tmp_path / "rounds" / "round4").is_dir()


def test_held_out_groups_include_the_board_size_with_far_frames_apart(tmp_path):
    corners = [{"index": k, "x": 10.0 * k, "y": 5.0, "visible": True} for k in range(4)]
    recs = [{"board_level": 40.0, "s_px": s, "corners": corners} for s in (10.0, 15.9, 20.0, 40.0, 80.0)]
    recs.append({"board_level": 40.0, "corners": corners})
    (tmp_path / "records.json").write_text(json.dumps({"meta": {}, "records": recs}))
    np.save(tmp_path / "images.npy", np.zeros((len(recs), 4, 4), np.uint8))

    class Perfect:
        def run_inference(self, img):
            return {"corners": [{"index": c["index"], "x": c["x"], "y": c["y"]} for c in corners]}

    out = evaluate_onnx(Perfect(), [str(tmp_path)])
    assert {g: out[g]["frames"] for g in ("all", "dim", "far <16 px", "16-32 px", "32-64 px", ">=64 px")} == {
        "all": 6, "dim": 6, "far <16 px": 2, "16-32 px": 1, "32-64 px": 1, ">=64 px": 1}
    assert out["far <16 px"]["recall"] == 1.0


def test_held_out_scores_count_the_poses_the_pipeline_finds_ambiguous(tmp_path):
    corners = [{"index": k, "x": 10.0 * k, "y": 5.0, "visible": True} for k in range(4)]
    recs = [{"board_level": 40.0, "s_px": s, "corners": corners}
            for s in (10.0, 12.0, 14.0, 40.0, 40.0)]
    (tmp_path / "records.json").write_text(json.dumps({"meta": {}, "records": recs}))
    np.save(tmp_path / "images.npy", np.arange(len(recs), dtype=np.uint8).reshape(-1, 1, 1))
    poses = {0: True, 1: True, 2: False, 3: False}

    class Posed:
        def run_inference(self, img):
            i = int(img[0, 0])
            return {"corners": [{"index": c["index"], "x": c["x"], "y": c["y"]} for c in corners],
                    "rvec": np.zeros(3) if i in poses else None, "ambiguous": poses.get(i, False)}

    out = evaluate_onnx(Posed(), [str(tmp_path)])
    assert out["all"]["pose_rate"] == 0.8 and out["all"]["ambiguous_rate"] == 0.5
    far = out["far <16 px"]
    assert far["pose_rate"] == 1.0 and far["ambiguous_rate"] == round(2 / 3, 4)
    assert out["32-64 px"]["pose_rate"] == 0.5 and out["32-64 px"]["ambiguous_rate"] == 0.0
