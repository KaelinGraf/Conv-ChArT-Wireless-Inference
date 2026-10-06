import yaml
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy,QoSLivelinessPolicy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseWithCovariance, Pose
from cv_bridge import CvBridge
import cv2
import ros2_numpy as rnp

from convchart_interfaces.msg import RosInferenceResult
from .inference import inference_pipeline, InferenceResult

    
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
            liveliness_lease_duration=rclpy.duration.Duration(seconds=1.0)
        )
        inferenceQOSProfile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth = 10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        self._img_sub = self.create_subscription(CompressedImage, 'image', self._image_callback, imgQOSProfile)
        self._res_pub = self.create_publisher(RosInferenceResult, 'inference_result', inferenceQOSProfile)

        self._guard_cfg(cfg_pth)
        self._inference_pipeline:inference_pipeline = inference_pipeline(config=self._cfg_pth)

        self._bridge = CvBridge()

    def _guard_cfg(self, cfg_pth: str):
        if not cfg_pth:
            self.get_logger().error("Configuration path is not set")
            raise ValueError("Configuration path is not set")
        #guard that K and dist are set in the config file's CAMERA block (where the pipeline reads them)
        with open(cfg_pth, "r") as file:
            camera = (yaml.safe_load(file) or {}).get("CAMERA") or {}
            if camera.get("K") is None or camera.get("dist") is None:
                self.get_logger().error("Configuration file must set CAMERA.K and CAMERA.dist")
                raise ValueError("Configuration file must set CAMERA.K and CAMERA.dist")

    def _image_callback(self, msg: CompressedImage):
        self.get_logger().info('Received an image message')
        img:np.ndarray = self._bridge.compressed_imgmsg_to_cv2(msg)

        result:InferenceResult = self._inference_pipeline.run_inference(img)
        ros_result = self._package_result(msg, result)
        self._res_pub.publish(ros_result)


    @staticmethod
    def _package_result(img_msg: CompressedImage, result: InferenceResult) -> RosInferenceResult:
        """
        Pack one InferenceResult (a TypedDict, so a plain dict at runtime) into the ROS message.

        A refusal (reason set) or a missing second IPPE solution leaves that pose at its default
        with covariance_valid* = False; the receiver must check reason before anything else.
        """
        ros_result = RosInferenceResult()
        #extract image header to return image timestamp (not inference timestamp)
        ros_result.header = img_msg.header
        ros_result.reason = result["reason"] or ""
        ros_result.ambiguous = result["ambiguous"]
        #the identified corners are exactly the ones PnP used; both IPPE solutions share them
        ros_result.num_used = sum(c["index"] is not None for c in result["corners"])
        ros_result.num_used_alt = ros_result.num_used

        ros_result.pose, ros_result.covariance_valid = _pack_pose(
            result["rvec"], result["tvec"], result["pose_cov"])
        ros_result.pose_alt, ros_result.covariance_valid_alt = _pack_pose(
            result["rvec_alt"], result["tvec_alt"], result["pose_cov_alt"])
        #nan, not 0, without a solution: an unchecked refusal must not read as a perfect fit
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


#pose_cov is ordered (rvec, tvec); PoseWithCovariance.covariance is row-major over
#(x, y, z, rot_x, rot_y, rot_z). The rotation block stays the covariance of the rvec components.
ROS_COV_ORDER = [3, 4, 5, 0, 1, 2]


def _pack_pose(rvec: np.ndarray | None, tvec: np.ndarray | None,
               pose_cov: list[list[float]] | None) -> tuple[PoseWithCovariance, bool]:
    """
    Pack one PnP solution, returning the pose and whether its covariance is valid.

    No solution leaves the default pose; no covariance (a used corner fell back to its coarse
    peak) keeps the pose but marks the covariance invalid so the receiver uses its fallback.
    """
    pose = PoseWithCovariance()
    if rvec is None or tvec is None:
        return pose, False
    pose.pose = rnp.msgify(Pose, to_4x4_matrix(tvec, rvec))
    if pose_cov is None:
        return pose, False
    cov = np.asarray(pose_cov, dtype=np.float64)[np.ix_(ROS_COV_ORDER, ROS_COV_ORDER)]
    pose.covariance = cov.ravel().tolist()
    return pose, True
    



def main(args=None):
    rclpy.init(args=args)
    convchart_node = ConvChartROS(cfg_pth='/path/to/config.yaml')
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(convchart_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        convchart_node.destroy_node()
        rclpy.shutdown()
