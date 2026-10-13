"""Analytics of the live SSL node: JSONL logs, rolling summary, dashboard and report figures (ROS-free).

    python3 -m pytest src/ros/src/convchart_ros/test/test_ssl_analytics.py -q
"""
import json
import time

from convchart_ros.ssl_utils.analytics import Analytics, main

import ssl_scene  # noqa: F401


def seg(i):
    return {"length": 3, "labelled": 3, "anchor": 1, "tracked": 2, "kept": 2, "duplicate": 1, "capped": 0,
            "closures": [0.2], "fb": [0.3] if i % 2 else [], "rho": [0.9, 0.8, 0.85]}


def test_logs_summary_dashboard_and_report(tmp_path):
    a = Analytics(tmp_path, window_s=1e6)
    for i in range(60):
        ev = {"seq": i, "stamp": 0.1 * i, "anchor": i % 3 == 0, "n_detections": 16, "n_ids": 12, "rms": 0.1,
              "level": 200.0, "ms": 3.0, "latency_ms": 20.0 + i % 5, "segment": seg(i) if i % 3 == 0 else None}
        if i == 31:
            ev["break"] = seg(i)
        a.frame(ev)
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text("\n".join(json.dumps({"step": s, "loss": 1.0 / (1 + s)}) for s in range(1, 200)) + "\n")
    a.round_done({"round": 1, "accepted": True, "metrics": str(metrics), "t_start": a.t0, "t_end": a.t0 + 0.5,
                  "reference": {"all": {"recall": 0.90, "id_precision": 0.99}, "dark": {"recall": 0.2}},
                  "candidate": {"all": {"recall": 0.95, "id_precision": 0.995}, "dark": {"recall": 0.5}}})
    a.round_done({"round": 1, "accepted": True})
    s = a.summary()
    assert s["frames"] == 60 and s["labelled"] == 63 and s["kept"] == 42 and abs(s["anchor_rate"] - 1 / 3) < 0.01
    png = a.render({"1,2,0": 30, "0,0,1": 5}, {"phase": "train", "round": 2, "progress": 0.4, "history": [],
                                               "started": time.time() - 0.2}, {"r1": str(metrics)}, pdf=True)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and (tmp_path / "dashboard.png").read_bytes() == png
    assert (tmp_path / "report" / "dashboard.pdf").stat().st_size > 1000
    panels = sorted(f.name for f in (tmp_path / "report").glob("panel_*.pdf"))
    assert len(panels) == 6 and panels[0].startswith("panel_1_stream") and panels[-1].startswith("panel_6_held_out")
    a.close()
    rows = [json.loads(line) for line in (tmp_path / "frames.jsonl").read_text().splitlines()]
    assert len(rows) == 60 and [r["seq"] for r in rows if r.get("stream_break")] == [31]
    segs = [json.loads(line) for line in (tmp_path / "segments.jsonl").read_text().splitlines()]
    assert len(segs) == 21 and [s["seq"] for s in segs if s["kind"] == "break"] == [31]
    assert len((tmp_path / "rounds.jsonl").read_text().splitlines()) == 1

    (tmp_path / "session.json").write_text(json.dumps({"convchart_root": ""}))
    (tmp_path / "summary.json").write_text(json.dumps({"coverage": {"1,2,0": 30}}))
    (tmp_path / "dashboard.png").unlink()
    main([str(tmp_path), "--no-pdf"])
    assert (tmp_path / "dashboard.png").stat().st_size > 10000
    b = Analytics(tmp_path, replay=True)
    assert len(b.frames) == 60 and len(b.segments) == 21 and len(b.rounds) == 1 and b.t0 == a.t0
