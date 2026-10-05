import json
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy,QoSLivelinessPolicy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseWithCovariance, Pose
from cv_bridge import CvBridge
import cv2
import ros2_numpy as rnp


class CalibrateCam(Node):
    """
    Ingest camera frames to calibrate camera intrinsics and distortion coeffients. 
    @args:
        cfg_pth: str - path to the configuration file for the camera calibration.
        auto_replace: bool - if True, automatically replace the existing configuration file with the new calibration
    """
    def __init__(self, cfg_pth: str = '', auto_replace:bool = False):
        super().__init__('calibrate_cam_node')
        self._cfg_pth:str = cfg_pth
        imgQOSProfile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            liveliness=QoSLivelinessPolicy.AUTOMATIC,
        )

        self._img_sub = self.create_subscription(CompressedImage, 'image', self._image_callback, imgQOSProfile)
        self._bridge = CvBridge()

    

    