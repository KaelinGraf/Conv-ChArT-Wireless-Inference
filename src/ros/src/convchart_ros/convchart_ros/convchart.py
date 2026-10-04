import json
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy,QoSLivelinessPolicy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from convchart_interfaces.msg import InferenceResult
from .....inference import inference_pipeline, InferenceResult
    
class ConvChartROS(Node):
    def __init__(self, cfg_pth: str = ''):
        super().__init__('convchart_ros_node')
        self._cfg_pth:str = cfg_pth
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

        self._img_sub = self.create_subscription(CompressedImage, 'image', self._image_callback, imgQOSProfile)
        self._res_pub = self.create_publisher(InferenceResult, 'inference_result', inferenceQOSProfile)

        self._guard_cfg(cfg_pth)
        self._inference_pipeline:inference_pipeline = inference_pipeline(config=self._cfg_pth)

        self._bridge = CvBridge()

    def _guard_cfg(self, cfg_pth: str):
        if not cfg_pth:
            self.get_logger().error("Configuration path is not set")
            raise ValueError("Configuration path is not set")
        #guard that K and Dist are set in the config file
        with open(cfg_pth, "r") as file:
            cfg = json.load(file)
            if cfg.get("K") is None or cfg.get("Dist") is None:
                self.get_logger().error("Configuration file must contain 'K' and 'Dist' parameters")
                raise ValueError("Configuration file must contain 'K' and 'Dist' parameters")

    def _image_callback(self, msg: CompressedImage):
        self.get_logger().info('Received an image message')
        img:np.ndarray = self._bridge.compressed_imgmsg_to_cv2(msg)

        result:InferenceResult = self._inference_pipeline.run_inference(img)
