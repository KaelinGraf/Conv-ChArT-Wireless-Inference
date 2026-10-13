"""
The evaluation record of an ssl fine-tune session (ssl_utils/record.py): resumes, bank generations, steps and passes,
far-range recall (ROS-free).

    python3 -m pytest src/ros/src/convchart_ros/test/test_ssl_record.py -q
"""
import json

import numpy as np

from convchart_ros.ssl_utils.record import record

PARAMS = {"backgrounds": "", "model_config": "m.yaml", "max_rate_hz": 7.5, "auto_deploy": True, "max_gap": 150,
          "min_motion_px": 0.0, "cap_per_bucket": 0, "chunk_size": 120, "heldout_every": 5, "split_block_s": 30.0,
          "min_train_frames": 400, "min_heldout_frames": 80, "retrain_growth": 0.5}


def chunk(path, n, s_px, source="anchor"):
    path.mkdir(parents=True)
    corners = [{"index": k, "x": 1.0, "y": 1.0, "visible": True} for k in range(16)]
    recs = [{"stamp": 1000.0 + i, "s_px": s_px, "board_level": 50.0, "anisotropy": 0.0, "source": source,
             "corners": corners} for i in range(n)]
    (path / "records.json").write_text(json.dumps({"meta": {}, "records": recs}))
    np.save(path / "images.npy", np.zeros((n, 2, 2), np.uint8))


def test_the_record_shows_resumes_generations_steps_and_far_recall(tmp_path):
    s = tmp_path / "repo" / "ssl_output" / "20261013-021802"
    chunk(s / "harvest" / "train" / "chunk_0000", 3, 20.0)
    chunk(s / "harvest" / "gen1" / "train" / "chunk_R0000", 4, 10.0)
    chunk(s / "harvest" / "gen1" / "heldout" / "chunk_R0001", 2, 10.0, "tracked")
    meta = {"params": {**PARAMS, "split_block_s": 0.0}, "detector": "release.onnx", "base_detector": "release.onnx",
            "teacher": "release.onnx", "refiner": "refiner.onnx", "lens": "calibrated", "config": "cfg.yaml",
            "resumes": [{"at": "2026-10-13T16:30:00", "teacher": "r3.onnx", "generation": 1, "params": PARAMS}]}
    (s / "session.json").write_text(json.dumps(meta))
    group = lambda r, far: {"all": {"recall": r, "id_precision": 0.99, "fit_rate": 1.0},       # noqa: E731
                            "far <16 px": {"recall": far, "id_precision": 0.99, "fit_rate": 1.0}}
    rounds = [{"round": 4, "t_start": 1.79e9, "seconds": 600.0, "train_frames": 1080, "heldout_frames": 240,
               "steps": 1204, "passes": 8.03, "accepted": True, "gate": {"heldout_recall": True},
               "config": {"passes": 8.0, "min_steps": 0, "sampling": "epoch", "real_frac": 0.6},
               "reference": group(0.98, 0.90), "candidate": group(0.99, 0.95),
               "synth_base": {"m04": {"accuracy": 0.99}, "m01": {"mean": 0.41}},
               "synth_candidate": {"m04": {"accuracy": 0.99}, "m01": {"mean": 0.42}}, "heldout_dirs": []},
              {"round": 5, "t_start": 1.79e9, "seconds": 5.0, "train_frames": 1620, "heldout_frames": 240,
               "accepted": False, "error": "RuntimeError: round stopped", "config": {}}]
    (s / "rounds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rounds))
    (s / "relabel.jsonl").write_text(json.dumps({"gen": 1, "labelled": 6, "anchor": 4, "tracked": 2, "kept": 6})
                                     + "\n")
    (s / "frames.jsonl").write_text(json.dumps({"stamp": 1.79e9, "anchor": True}) + "\n")
    text = record(s, tmp_path / "out").read_text()
    assert "- Resumed 2026-10-13T16:30:00: teacher `r3.onnx`, bank generation 1" in text
    assert "every banked frame drawn at least 8.0 times (epoch sampling)" in text and "real_frac 0.6: 60%" in text
    assert "every 5th block of 30.0 s of frame stamps" in text
    assert "- Generation 0 (`harvest`): banked train 3, held-out 0." in text
    assert ("- Generation 1 (`harvest/gen1`): banked train 4, held-out 2; its pass over the stored frames labelled 6 "
            "(4 anchors, 2 by tracking).") in text
    assert "The tables below are generation 1's bank." in text and "| <16 px | 0 / 0 | 4 / 2 | 0 / 0 |" in text
    row = next(line for line in text.splitlines() if line.startswith("| 4 |"))
    assert "| 1204 (8.0) |" in row and "| 0.9800 -> 0.9900 | 0.9000 -> 0.9500 |" in row and row.endswith("| yes |")
    assert next(line for line in text.splitlines() if line.startswith("| 5 |")).endswith("| no |")
    assert (tmp_path / "out" / "data" / "rounds.jsonl").exists()
