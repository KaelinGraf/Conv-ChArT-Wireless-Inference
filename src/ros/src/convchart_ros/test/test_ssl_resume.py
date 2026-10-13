"""
Resuming an ssl fine-tune session: what a restarted node carries on from, and its analytics (ROS-free).

    python3 -m pytest src/ros/src/convchart_ros/test/test_ssl_resume.py -q
"""
import json

from convchart_ros.ssl_utils.analytics import Analytics
from convchart_ros.ssl_utils.resume import generations, latest_session, read_rounds, resume_state


def make_session(root, name="20261013-021802", rounds=(), meta=None, gens=()):
    s = root / name
    s.mkdir(parents=True)
    (s / "session.json").write_text(json.dumps(meta or {"detector": "release.onnx", "teacher": "deployed.onnx"}))
    (s / "rounds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rounds) + "not json\n")
    (s / "analytics.json").write_text(json.dumps({"t0": 123.0, "window_s": 30.0}))
    for g in gens:
        (s / "harvest" / ("train" if g == 0 else f"gen{g}/train")).mkdir(parents=True)
    return s


def test_a_resumed_session_carries_on_from_its_last_accepted_round(tmp_path):
    s = make_session(tmp_path, rounds=[
        {"round": 1, "accepted": False, "candidate_onnx": "r1.onnx"},
        {"round": 2, "accepted": True, "candidate_onnx": "r2.onnx", "train_frames": 720},
        {"round": 2, "accepted": True, "candidate_onnx": "r2.onnx", "train_frames": 720},
        {"round": 3, "accepted": True, "candidate_onnx": "r3.onnx", "train_frames": 1080},
        {"round": 4, "accepted": False, "error": "RuntimeError: stopped"}], gens=(0, 1, 3))
    st = resume_state(s)
    assert [r["round"] for r in st["history"]] == [1, 2, 3, 4] and st["round"] == 4
    assert st["best_onnx"] == st["teacher"] == "r3.onnx" and st["t0"] == 123.0
    assert generations(s) == [0, 1, 3] and st["next_gen"] == 4


def test_without_an_accepted_round_the_session_keeps_its_starting_teacher(tmp_path):
    s = make_session(tmp_path, rounds=[{"round": 1, "accepted": False}])
    st = resume_state(s)
    assert st["best_onnx"] is None and st["teacher"] == "deployed.onnx" and st["next_gen"] == 1
    meta = {"detector": "release.onnx", "resumes": [{"teacher": "later.onnx"}]}
    assert resume_state(make_session(tmp_path, "b", meta=meta))["teacher"] == "later.onnx"
    assert read_rounds(tmp_path / "nowhere") == []


def test_latest_session_is_the_newest_session_folder(tmp_path):
    out = tmp_path / "ssl_output"
    assert latest_session(out) is None
    make_session(out, "20261010-021909")
    newest = make_session(out, "20261013-021802")
    (out / "20261099-000000").mkdir()
    assert latest_session(out) == newest


def test_resumed_analytics_keep_the_clock_and_history_and_append(tmp_path):
    s = make_session(tmp_path, rounds=[{"round": 1, "accepted": False}, {"round": 1, "accepted": False}])
    (s / "frames.jsonl").write_text(json.dumps({"t": 5.0, "anchor": True, "n_ids": 16, "latency_ms": 38.0}) + "\n")
    (s / "segments.jsonl").write_text(json.dumps({"labelled": 3, "kept": 2, "t": 5.0}) + "\n")
    a = Analytics(s, resume=True)
    assert a.t0 == 123.0 and len(a.frames) == 1 and [r["round"] for r in a.rounds] == [1]
    assert json.loads((s / "analytics.json").read_text())["t0"] == 123.0
    a.frame({"seq": 0, "stamp": 200.0, "anchor": False, "n_ids": 0, "arrival": 200.0})
    a.round_done({"round": 1, "accepted": False})
    a.round_done({"round": 2, "accepted": True})
    a.relabel({"labelled": 4, "kept": 4}, 2, 17)
    a.close()
    assert len((s / "frames.jsonl").read_text().splitlines()) == 2
    assert [json.loads(x)["round"] for x in (s / "rounds.jsonl").read_text().splitlines() if x.startswith("{")] == \
        [1, 1, 2]
    rows = [json.loads(x) for x in (s / "relabel.jsonl").read_text().splitlines()]
    assert rows == [{"labelled": 4, "kept": 4, "t": rows[0]["t"], "seq": 17, "gen": 2, "kind": "relabel"}]
    assert a.summary()["labelled"] == 3
