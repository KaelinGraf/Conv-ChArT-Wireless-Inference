import json
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseWithCovariance, Pose
from cv_bridge import CvBridge
import cv2
import ros2_numpy as rnp

from convchart_qos.qos import IMAGE_SUB_QOS


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
        # Shared with the inference node and the Pi's camera publisher; see
        # convchart_qos.
        # This used to omit deadline/lifespan, which matched only by accident.
        self._img_sub = self.create_subscription(CompressedImage, 'image', self._image_callback, IMAGE_SUB_QOS)
        self._bridge = CvBridge()

    

    