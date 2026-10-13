"""
Image relay: the Pi's frames cross the WiFi once, however many nodes on the laptop read them.

    ros2 run convchart_ros image_relay [--ros-args -p input_topic:=image -p output_topic:=laptop/image]

Over the unicast DDS link (docker/ros/cyclonedds_peers.xml) the Pi's camera sends every frame once to each
process that subscribes to it. With the inference node and the dashboard both on /image, each frame crossed the
link twice, and the two copies arrived at different times. This node is the laptop's one subscriber to /image.
It republishes each frame on laptop/image as the bytes it received, neither deserialised nor decoded, so the
header stamp and the PNG are unchanged. The inference node and the pose overlay (both remapped) and the
dashboard (its image_topic) read that one copy at the same time. Nothing on the Pi subscribes to laptop/image,
so it never crosses back. task:=infer starts it whenever more than one node there reads the Pi's frames.

It subscribes as the inference node does (IMAGE_SUB_QOS: the newest frame only) and offers the camera's QoS
with the loosest deadline its readers accept, since the frames arrive at the link's pace.
"""
from __future__ import annotations

import signal

from convchart_qos.qos import IMAGE_DEADLINE_CEILING_S, IMAGE_SUB_QOS, image_qos_pub
import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


class ImageRelay(Node):

    def __init__(self):
        super().__init__("image_relay")
        source = str(self.declare_parameter("input_topic", "image").value)
        target = str(self.declare_parameter("output_topic", "laptop/image").value)
        self.pub = self.create_publisher(CompressedImage, target, image_qos_pub(deadline_s=IMAGE_DEADLINE_CEILING_S))
        self.relayed = 0
        self.sub = self.create_subscription(CompressedImage, source, self._relay, IMAGE_SUB_QOS, raw=True)
        self.get_logger().info(f"relaying {self.sub.topic_name} to {self.pub.topic_name}")

    def _relay(self, data: bytes) -> None:
        self.pub.publish(data)
        self.relayed += 1
        if self.relayed == 1:
            self.get_logger().info(f"first frame relayed ({len(data) / 1e3:.0f} kB)")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImageRelay()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
