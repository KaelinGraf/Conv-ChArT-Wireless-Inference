"""
image_relay over ROS 2: the frames come out on laptop/image byte for byte, and it alone reads the camera's.

    python3 -m pytest src/ros/src/convchart_ros/test/test_image_relay.py -q

A stand-in camera publishes PNG frames on image with the camera's QoS; two readers with the inference node's
QoS, as the inference node and the dashboard, read laptop/image. Isolated domain, discovery limited to
localhost and a private namespace.
"""
import os
import time

from builtin_interfaces.msg import Time
from convchart_qos.qos import IMAGE_SUB_QOS, image_qos_pub
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import CompressedImage

from convchart_ros.image_relay import ImageRelay

DOMAIN_ID = 81 + os.getpid() % 15
NAMESPACE = "/relay_test"
TIMEOUT_S = 10.0


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


def spin_until(executor, done, what):
    deadline = time.monotonic() + TIMEOUT_S
    while not done():
        assert time.monotonic() < deadline, what
        executor.spin_once(timeout_sec=0.05)


def frame(k):
    msg = CompressedImage(format="mono8; png compressed mono8", data=bytes(range(256)) * (400 + k))
    msg.header.stamp = Time(sec=1760000000 + k, nanosec=123456789)
    msg.header.frame_id = "pi_camera"
    return msg


def test_each_frame_is_relayed_unchanged_to_every_reader_and_only_the_relay_reads_the_camera(ros):
    relay, camera, readers = ImageRelay(), rclpy.create_node("camera"), rclpy.create_node("readers")
    executor = SingleThreadedExecutor()
    for node in (relay, camera, readers):
        executor.add_node(node)
    try:
        pub = camera.create_publisher(CompressedImage, "image", image_qos_pub(deadline_s=0.1))
        got = {"inference": [], "dashboard": []}
        for name, inbox in got.items():
            readers.create_subscription(CompressedImage, "laptop/image", inbox.append, IMAGE_SUB_QOS)
        spin_until(executor, lambda: pub.get_subscription_count() == 1
                   and relay.pub.get_subscription_count() == 2, "the relay and its readers never matched")
        sent = [frame(k) for k in range(5)]
        for msg in sent:
            pub.publish(msg)
            n = sent.index(msg) + 1
            spin_until(executor, lambda: all(len(inbox) == n for inbox in got.values()), f"frame {n} not relayed")
        for inbox in got.values():
            assert [(m.header.stamp, m.header.frame_id, m.format, bytes(m.data)) for m in inbox] == \
                   [(m.header.stamp, m.header.frame_id, m.format, bytes(m.data)) for m in sent]
        assert relay.relayed == 5
        assert relay.sub.topic_name == f"{NAMESPACE}/image" and relay.pub.topic_name == f"{NAMESPACE}/laptop/image"
    finally:
        for node in (relay, camera, readers):
            node.destroy_node()
