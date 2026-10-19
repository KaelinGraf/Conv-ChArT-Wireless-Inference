"""
The replay node (replay.py) over ROS 2: every frame, in order, none lost to a late subscriber; the root
namespace refused.

    python3 -m pytest src/ros/src/convchart_ros/test/test_replay.py -q

Isolated domain, discovery limited to localhost, private namespace; main() runs as a subprocess under the
same isolation.
"""
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import CompressedImage

from convchart_qos.qos import IMAGE_SUB_QOS
from convchart_ros.replay import Replay

DOMAIN_ID = 11 + os.getpid() % 20
NAMESPACE = "/replay_test"
TIMEOUT_S = 20.0


@pytest.fixture(scope="module")
def ros():
    saved = os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE")
    os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    rclpy.init(args=["--ros-args", "-r", f"__ns:={NAMESPACE}"], domain_id=DOMAIN_ID)
    try:
        yield
    finally:
        rclpy.shutdown()
        if saved is None:
            os.environ.pop("ROS_AUTOMATIC_DISCOVERY_RANGE", None)
        else:
            os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = saved


@pytest.fixture
def clip(tmp_path):
    """Return a folder of twelve 640x480 frames, frame i a flat grey of 10 * (i + 1)."""
    folder = tmp_path / "clip"
    folder.mkdir()
    for i in range(12):
        cv2.imwrite(str(folder / f"{i:04d}.png"), np.full((480, 640), 10 * (i + 1), np.uint8))
    return folder


def levels(msgs):
    return [int(round(cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_GRAYSCALE).mean()) // 10 - 1)
            for m in msgs]


def test_every_frame_is_published_in_order_after_the_subscriber_matched(ros, clip):
    node = Replay(source=str(clip), rate=40.0, loop=2)
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    probe = rclpy.create_node("replay_probe")
    try:
        time.sleep(0.5)
        assert node.published == 0, "published before anyone subscribed"
        got = []
        probe.create_subscription(CompressedImage, "image", got.append, IMAGE_SUB_QOS)
        executor.add_node(probe)
        deadline = time.monotonic() + TIMEOUT_S
        while not node.done and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.3)
        assert node.done and node.published == 24
        assert levels(got) == list(range(12)) * 2
        assert {m.format for m in got} == {"mono8; png compressed mono8"} and got[0].header.frame_id == "replay"
    finally:
        executor.shutdown()
        spinner.join(timeout=TIMEOUT_S)
        probe.destroy_node()
        node.destroy_node()


@pytest.mark.parametrize("overrides, message", [
    ({}, "set source"),
    ({"source": "/no/such/clip.mp4"}, "does not exist"),
    ({"source": "CLIP", "rate": 0.0}, "rate > 0"),
    ({"source": "CLIP", "loop": -1}, "loop >= 0"),
])
def test_bad_parameters_are_refused(ros, clip, overrides, message):
    overrides = {k: str(clip) if v == "CLIP" else v for k, v in overrides.items()}
    with pytest.raises(ValueError, match=message):
        Replay(**overrides).destroy_node()


def test_an_empty_folder_is_refused(ros, tmp_path):
    with pytest.raises(ValueError, match="no readable frames"):
        Replay(source=str(tmp_path)).destroy_node()


def test_main_refuses_the_root_namespace(clip):
    env = {**os.environ, "ROS_DOMAIN_ID": str(DOMAIN_ID), "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST"}
    out = subprocess.run([sys.executable, "-c", "from convchart_ros.replay import main; main()",
                          "--ros-args", "-p", f"source:={clip}"], env=env, capture_output=True, text=True,
                         timeout=60)
    assert out.returncode == 1 and "replay: refusing to publish recorded frames on /image" in out.stderr
