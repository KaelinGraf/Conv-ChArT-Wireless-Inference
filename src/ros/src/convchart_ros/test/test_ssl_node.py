"""End-to-end test of the live SSL node over ROS 2 (harvest and analytics; training off).

    python3 -m pytest src/ros/src/convchart_ros/test/test_ssl_node.py -q
A rendered board video is published as PNG CompressedImage on 'image' at 10 Hz; the node must run the deployed
pipeline on it, label and bank frames, publish ssl/stats and ssl/dashboard, and write its session folder. Midway the
teacher is paused and resumed as a training round would (frames dropped meanwhile, the stream broken there). Runs on an
isolated domain with discovery limited to localhost. Needs CUDA, the ONNX checkpoints and the Conv-ChArT checkout.
"""
import json
import os
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import pytest

from ssl_scene import homography, render

DOMAIN_ID = 97 + os.getpid() % 20


def find_repo():
    for parent in Path(__file__).resolve().parents:
        if (parent / "cfg" / "cfg.yaml").is_file() and (parent / "checkpoints").is_dir():
            return parent
    return None


@pytest.fixture()
def ros():
    ort = pytest.importorskip("onnxruntime")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("the node runs the ONNX pipeline on CUDA")
    pytest.importorskip("dcc.video", reason="needs the Conv-ChArT checkout (dcc)")
    if find_repo() is None:
        pytest.skip("needs cfg/cfg.yaml and checkpoints/")
    import rclpy
    saved = os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE")
    os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    rclpy.init(args=["--ros-args", "-r", "__ns:=/ssl_test"], domain_id=DOMAIN_ID)
    yield rclpy
    rclpy.try_shutdown()
    if saved is None:
        os.environ.pop("ROS_AUTOMATIC_DISCOVERY_RANGE", None)
    else:
        os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = saved


def test_node_labels_a_live_stream_and_reports(ros, tmp_path):
    from convchart_qos.qos import image_qos_pub
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import String

    from convchart_ros.ssl_node import OUT_QOS, SSLNode

    model_cfg = tmp_path / "model.yaml"
    model_cfg.write_text("input_size: [640, 480]\nboard: null\n")
    node = SSLNode(config=str(find_repo() / "cfg" / "cfg.yaml"), model_config=str(model_cfg),
                   output_dir=str(tmp_path / "out"), training=False, stats_period=0.5, dashboard_period=2.0,
                   chunk_size=20)
    probe = Node("ssl_probe")
    pub = probe.create_publisher(CompressedImage, "image", image_qos_pub())
    stats, dashboards = [], []
    probe.create_subscription(String, "ssl/stats", lambda m: stats.append(json.loads(m.data)), OUT_QOS)
    probe.create_subscription(CompressedImage, "ssl/dashboard", lambda m: dashboards.append(m), OUT_QOS)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    ex.add_node(probe)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()
    rng = np.random.default_rng(0)

    def send(ts):
        for t in ts:
            H = homography(t) if t < 70 else homography(t - 70, theta=0.9, tx=260.0, ty=260.0)
            img = render(H, gain=0.5 if 30 <= t < 40 else 1.0, noise=1.5, rng=rng)
            msg = CompressedImage(format="png", data=cv2.imencode(".png", img)[1].tobytes())
            msg.header.stamp = probe.get_clock().now().to_msg()
            pub.publish(msg)
            time.sleep(0.1)

    def wait_for(frames):
        deadline = time.time() + 60
        while node.analytics.summary()["frames"] < frames and time.time() < deadline:
            time.sleep(0.2)

    try:
        deadline = time.time() + 20
        while pub.get_subscription_count() == 0 and time.time() < deadline:
            time.sleep(0.1)
        send(range(60))
        wait_for(58)
        node._round_start()
        send(range(60, 70))
        deadline = time.time() + 30
        while node._drops["paused"] < 10 and time.time() < deadline:
            time.sleep(0.05)
        node._round_end()
        send(range(70, 100))
        wait_for(85)
        time.sleep(2.5)
    finally:
        ex.shutdown()
        spin.join(timeout=10)
        node.shutdown()
        session = node.session
        node.destroy_node()
        probe.destroy_node()
    summary = json.loads((session / "summary.json").read_text())
    n, totals = summary["frames"], summary["harvest"]["totals"]
    assert n >= 85 and summary["drops"]["paused"] >= 10 and not summary["paused"]
    assert totals["breaks"] >= 1 and totals["anchors"] > n // 2 and totals["labelled"] >= n - 10
    assert sum(summary["harvest"]["banked"].values()) > 20
    segs = [json.loads(line) for line in (session / "segments.jsonl").read_text().splitlines()]
    assert sum(s["labelled"] for s in segs) == totals["labelled"] and sum(s["kept"] for s in segs) == totals["kept"]
    assert len((session / "frames.jsonl").read_text().splitlines()) == summary["frames"]
    assert (session / "dashboard.png").stat().st_size > 10000 and (session / "report" / "dashboard.pdf").exists()
    assert stats and stats[-1]["frames"] > 0 and dashboards and bytes(dashboards[-1].data[:8]) == b"\x89PNG\r\n\x1a\n"


def test_max_rate_hz_skips_frames_before_the_teacher(ros, tmp_path):
    from convchart_qos.qos import image_qos_pub
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage

    from convchart_ros.ssl_node import SSLNode

    model_cfg = tmp_path / "model.yaml"
    model_cfg.write_text("input_size: [640, 480]\nboard: null\n")
    node = SSLNode(config=str(find_repo() / "cfg" / "cfg.yaml"), model_config=str(model_cfg),
                   output_dir=str(tmp_path / "out"), training=False, max_rate_hz=5.0)
    probe = Node("ssl_rate_probe")
    pub = probe.create_publisher(CompressedImage, "image", image_qos_pub())
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    ex.add_node(probe)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()
    rng = np.random.default_rng(1)
    try:
        deadline = time.time() + 20
        while pub.get_subscription_count() == 0 and time.time() < deadline:
            time.sleep(0.1)
        for t in range(40):
            msg = CompressedImage(format="png", data=cv2.imencode(".png", render(homography(t), rng=rng))[1].tobytes())
            msg.header.stamp = probe.get_clock().now().to_msg()
            pub.publish(msg)
            time.sleep(0.1)
        time.sleep(1.5)
        processed, skipped = node.analytics.summary()["frames"], node._drops["rate"]
    finally:
        ex.shutdown()
        spin.join(timeout=10)
        node.shutdown()
        node.destroy_node()
        probe.destroy_node()
    assert 15 <= processed <= 25 and processed + skipped >= 38
