"""
The live SSL node's bank generations over ROS 2, with a stand-in teacher: resume, the relabel pass, a new teacher.

    python3 -m pytest src/ros/src/convchart_ros/test/test_ssl_relabel.py -q

A session folder with 60 stored frames of a rendered board is resumed: its best model becomes the teacher and
generation 1 relabels every stored frame into R chunks. 20 live frames follow into the generation's L chunks. A
round is then accepted, as the round runner would report it, and generation 2 relabels all 80 stored frames with the
new model. The teacher is a stand-in for src/inference.py that knows each rendered frame's geometry, so this runs
without a GPU; test/test_ssl_node.py covers the real pipeline. Isolated domain, discovery limited to localhost.
Needs the Conv-ChArT checkout (dcc) and cfg/cfg.yaml.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import types

import cv2
import numpy as np
import pytest
import yaml

from ssl_scene import homography, render, result_for

DOMAIN_ID = 141 + os.getpid() % 20


def find_repo():
    for parent in Path(__file__).resolve().parents:
        if (parent / "cfg" / "cfg.yaml").is_file() and (parent / "checkpoints").is_dir():
            return parent
    return None


def key(img):
    return hashlib.sha1(np.ascontiguousarray(img).tobytes()).hexdigest()


class Teacher:
    """Stand-in for src/inference.py's pipeline on one detector, anchored every 5th frame.

    Args:
        table: Frame hash mapped to (frame index, homography).
        detector: Path of the detector ONNX model.
        used: List that records the detector of every inference.
    """

    def __init__(self, table, detector, used):
        self.table, self.detector, self.used = table, detector, used

    def run_inference(self, img):
        t, H = self.table.get(key(img), (None, None))
        self.used.append(self.detector)
        if H is None:
            return {"corners": []}
        return result_for(H, anchor=(t % 5 == 0), rng=np.random.default_rng(t))


class Rounds:
    """Stand-in for the node's round runner, after a round with an accepted candidate.

    Args:
        best: Path of the accepted candidate ONNX model.
    """

    def __init__(self, best):
        self.best_onnx, self.busy, self.last_failure, self.steps, self.thread = best, False, None, 0, None

    def status(self):
        return {"phase": "idle", "round": 1, "progress": 0.0, "best_onnx": self.best_onnx, "history": []}

    def start(self, *a):
        return False

    def stop(self):
        pass


@pytest.fixture()
def ros(monkeypatch):
    pytest.importorskip("dcc.video", reason="needs the Conv-ChArT checkout (dcc)")
    if find_repo() is None:
        pytest.skip("needs cfg/cfg.yaml and checkpoints/")
    import rclpy
    monkeypatch.setenv("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
    rclpy.init(args=["--ros-args", "-r", "__ns:=/relabel_test"], domain_id=DOMAIN_ID)
    yield rclpy
    rclpy.try_shutdown()


def wait(cond, what, timeout=60.0):
    deadline = time.time() + timeout
    while not cond():
        assert time.time() < deadline, what
        time.sleep(0.1)


def test_a_resumed_session_relabels_what_it_stored_and_again_for_each_new_teacher(ros, tmp_path, monkeypatch):
    from convchart_qos.qos import image_qos_pub
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage

    table, used = {}, []
    rng = np.random.default_rng(0)
    frames = []
    for t in range(80):
        img = render(homography(t), noise=1.5, rng=rng)
        table[key(img)] = (t, homography(t))
        frames.append(img)
    fake = types.ModuleType("inference")
    fake.inference_pipeline = lambda cfg: Teacher(table, yaml.safe_load(Path(cfg).read_text())["MODEL"]["detector"],
                                                  used)
    monkeypatch.setitem(sys.modules, "inference", fake)
    from convchart_ros.ssl_node import SSLNode

    session = tmp_path / "out" / "20261013-000000"
    (session / "stream").mkdir(parents=True)
    (session / "session.json").write_text(json.dumps({"detector": "release.onnx", "teacher": "release.onnx"}))
    (session / "rounds.jsonl").write_text(json.dumps({"round": 1, "accepted": True, "candidate_onnx": "/m/best.onnx",
                                                      "train_frames": 480}) + "\n")
    t0 = time.time() - 600.0
    (session / "analytics.json").write_text(json.dumps({"t0": t0, "window_s": 30.0}))
    for t in range(60):
        sec, nsec = 1000 + t // 10, (t % 10) * 100_000_000
        cv2.imwrite(str(session / "stream" / f"{sec:010d}_{nsec:09d}.png"), frames[t])
    model_cfg = tmp_path / "model.yaml"
    model_cfg.write_text("input_size: [640, 480]\nboard: null\n")
    node = SSLNode(config=str(find_repo() / "cfg" / "cfg.yaml"), model_config=str(model_cfg), resume=str(session),
                   training=False, chunk_size=10, split_block_s=1.0, heldout_every=3, stats_period=0.5,
                   dashboard_period=30.0)
    probe = Node("relabel_probe")
    pub = probe.create_publisher(CompressedImage, "image", image_qos_pub())
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    ex.add_node(probe)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()
    try:
        assert node.session == session and node.analytics.t0 == t0 and node.teacher_onnx == "/m/best.onnx"
        wait(lambda: node.gen.finished, "generation 1 never finished relabelling")
        g1 = node.gen.progress()
        bank1 = node._bank()
        wait(lambda: pub.get_subscription_count() > 0, "the node never subscribed")
        for t in range(60, 80):
            msg = CompressedImage(format="png", data=cv2.imencode(".png", frames[t])[1].tobytes())
            msg.header.stamp.sec, msg.header.stamp.nanosec = 2000 + (t - 60) // 10, (t % 10) * 100_000_000
            pub.publish(msg)
            time.sleep(0.1)
        wait(lambda: node.analytics.summary()["frames"] >= 20 and node.saved >= 20, "live frames not taken in")
        live_banked = node._bank()["totals"].get("labelled", 0) - bank1["totals"].get("labelled", 0)
        node.p["auto_deploy"] = True
        node.rounds = Rounds("/m/other.onnx")
        node._round_start()
        node._round_end()
        assert node.gen.g == 2 and node.teacher_onnx == "/m/other.onnx" and node.gen.cut >= 2001.9 - 1e-6
        wait(lambda: node.gen.finished, "generation 2 never finished relabelling")
        g2 = node.gen.progress()
        bank2 = node._bank()
    finally:
        ex.shutdown()
        spin.join(timeout=10)
        node.shutdown()
        node.destroy_node()
        probe.destroy_node()
    assert g1["gen"] == 1 and g1["relabel"]["done"] == g1["relabel"]["total"] == 60
    assert bank1["banked"]["train"] > 0 and bank1["banked"]["heldout"] > 0 and bank1["totals"]["labelled"] > 50
    assert all(Path(p).parent.parent == session / "harvest" / "gen1" and Path(p).name.startswith("chunk_R")
               for split in ("train", "heldout") for p in bank1["chunks"][split])
    assert live_banked > 0 and list((session / "harvest" / "gen1").glob("*/chunk_L*"))
    assert g2["teacher"] == "/m/other.onnx" and g2["relabel"]["done"] == g2["relabel"]["total"] == 80
    assert bank2["totals"]["labelled"] > bank1["totals"]["labelled"] and all(
        "/gen2/" in p for split in ("train", "heldout") for p in bank2["chunks"][split])
    assert used.count("/m/other.onnx") >= 80 and used.count("/m/best.onnx") >= 60
    relabel = [json.loads(x) for x in (session / "relabel.jsonl").read_text().splitlines()]
    assert {r["gen"] for r in relabel} == {1, 2}
    meta = json.loads((session / "session.json").read_text())
    assert meta["teacher"] == "release.onnx" and meta["resumes"][-1]["teacher"] == "/m/best.onnx"
    summary = json.loads((session / "summary.json").read_text())
    assert summary["generation"]["gen"] == 2 and summary["resumed"]
