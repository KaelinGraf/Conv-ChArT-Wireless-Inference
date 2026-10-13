"""
Inference node: the Pi's 'image' stream through the conv-chart pipeline, poses out on 'inference_result'.

    ros2 run convchart_ros inference --ros-args -p config:=<cfg.yaml>

Refuses to start until the config's CAMERA block holds K and dist: run the calibrate task first
(ros2 launch convchart_ros convchart.launch.py task:=calibrate auto_replace:=true). Each solved pose
carries the measured, range-dependent measurement noise R (measurement_noise.py) as its covariance, not
the pipeline's own pose_cov; covariance_valid is False outside the range that noise was fitted over.
Set BOARD.square_length_m to the printed square edge in metres, or poses and R come out in squares.
"""
import signal

import yaml
import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseWithCovariance, Pose
from cv_bridge import CvBridge
import cv2
import ros2_numpy as rnp

from convchart_interfaces.msg import RosInferenceResult
from convchart_qos.qos import IMAGE_SUB_QOS, INFERENCE_RESULT_QOS
from .inference import inference_pipeline, InferenceResult
from .measurement_noise import measurement_covariance

    
class ConvChartROS(Node):
    def __init__(self, cfg_pth: str = ''):
        super().__init__('convchart_ros_node')
        self._cfg_pth:str = str(self.declare_parameter('config', cfg_pth).value)
        self._img_sub = self.create_subscription(CompressedImage, 'image', self._image_callback, IMAGE_SUB_QOS)
        self._res_pub = self.create_publisher(RosInferenceResult, 'inference_result', INFERENCE_RESULT_QOS)

        self._guard_cfg(self._cfg_pth)
        self._inference_pipeline:inference_pipeline = inference_pipeline(config=self._cfg_pth)

        self._bridge = CvBridge()

    def _guard_cfg(self, cfg_pth: str):
        if not cfg_pth:
            self.get_logger().error("Configuration path is not set")
            raise ValueError("Configuration path is not set")
        with open(cfg_pth, "r") as file:
            camera = (yaml.safe_load(file) or {}).get("CAMERA") or {}
            if camera.get("K") is None or camera.get("dist") is None:
                self.get_logger().error("Configuration file must set CAMERA.K and CAMERA.dist")
                raise ValueError("Configuration file must set CAMERA.K and CAMERA.dist")

    def _image_callback(self, msg: CompressedImage):
        self.get_logger().info('Received an image message')
        img:np.ndarray = self._bridge.compressed_imgmsg_to_cv2(msg)

        result:InferenceResult = self._inference_pipeline.run_inference(img)
        ros_result = self._package_result(msg, result, self._inference_pipeline.board.square_length_m or 1.0)
        self._res_pub.publish(ros_result)


    @staticmethod
    def _package_result(img_msg: CompressedImage, result: InferenceResult,
                        square_length_m: float) -> RosInferenceResult:
        """Pack one InferenceResult (a TypedDict, so a plain dict at runtime) into the ROS message.
        """
        ros_result = RosInferenceResult()
        ros_result.header = img_msg.header
        ros_result.reason = result["reason"] or ""
        ros_result.ambiguous = result["ambiguous"]
        ros_result.num_used = sum(c["index"] is not None for c in result["corners"])
        ros_result.num_used_alt = ros_result.num_used

        ros_result.pose, ros_result.covariance_valid = _pack_pose(
            result["rvec"], result["tvec"], square_length_m)
        ros_result.pose_alt, ros_result.covariance_valid_alt = _pack_pose(
            result["rvec_alt"], result["tvec_alt"], square_length_m)
        ros_result.rms = result["rms"] if result["rms"] is not None else float("nan")
        ros_result.rms_alt = result["rms_alt"] if result["rms_alt"] is not None else float("nan")
        return ros_result




def to_4x4_matrix(tvec: np.ndarray, rvec: np.ndarray) -> np.ndarray:
    """
    Convert translation and rotation vectors to a 4x4 transformation matrix.
    """
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def _pack_pose(rvec: np.ndarray | None, tvec: np.ndarray | None,
               square_length_m: float) -> tuple[PoseWithCovariance, bool]:
    """Pack one PnP solution, returning the pose and whether its covariance is valid."""
    pose = PoseWithCovariance()
    if rvec is None or tvec is None:
        return pose, False
    pose.pose = rnp.msgify(Pose, to_4x4_matrix(tvec, rvec))
    cov, in_range = measurement_covariance(float(np.ravel(tvec)[2]), square_length_m)
    pose.covariance = cov.ravel().tolist()
    return pose, in_range
    



def main(args=None):
    """
    Run the inference node. ROS parameter: config (cfg.yaml path, required). A missing config or
    an uncalibrated CAMERA block ends it with the line "inference: <reason>" and exit status 1.
    """
    rclpy.init(args=args)
    try:
        convchart_node = ConvChartROS()
    except (OSError, ValueError, yaml.YAMLError) as error:
        rclpy.try_shutdown()
        raise SystemExit(f"inference: {error}") from None
    executor = MultiThreadedExecutor()
    executor.add_node(convchart_node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        convchart_node.destroy_node()
        rclpy.try_shutdown()
