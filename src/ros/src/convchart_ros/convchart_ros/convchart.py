import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy,QoSLivelinessPolicy
from sensor_msgs.msg import Image
from convchart_interfaces.msg import InferenceResult

class ConvChartROS(Node):
    def __init__(self):
        super().__init__('convchart_ros_node')
        imgQOSProfile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            deadline=rclpy.duration.Duration(seconds=0.2),
            lifespan=rclpy.duration.Duration(seconds=0.15),
            liveliness=QoSLivelinessPolicy.AUTOMATIC,
            liveness_lease_duration=rclpy.duration.Duration(seconds=1.0)
        )
        inferenceQOSProfile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth = 10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        self._img_sub = self.create_subscription(Image, 'image', self._image_callback, imgQOSProfile)
        self._res_pub = self.create_publisher(InferenceResult, 'inference_result', inferenceQOSProfile)


    def _image_callback(self, msg: Image):
        # Process the image message here
        self.get_logger().info('Received an image message')

