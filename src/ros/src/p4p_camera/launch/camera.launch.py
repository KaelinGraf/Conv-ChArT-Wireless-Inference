"""Launch the Pi's camera node.

    ros2 launch p4p_camera camera.launch.py
    ros2 launch p4p_camera camera.launch.py camera_backend:=mock

The first form uses the v4l2 backend, which needs the host's CSI pipeline
configured (docker/camera-pipeline-pi.sh; host-setup-pi.sh runs it at boot).
The mock backend needs no hardware, so the second form is the one to use for
bring-up on a bench, and with mock_image set it will replay a recorded
1600x1200 frame to the real inference node over WiFi.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = [
        DeclareLaunchArgument('camera_backend', default_value='v4l2',
                              description='v4l2, picamera2 or mock'),
        DeclareLaunchArgument('frame_rate', default_value='10.0',
                              description='stream rate in Hz; floored at 5.0'),
        DeclareLaunchArgument('frame_id', default_value='pi_camera'),
        DeclareLaunchArgument('device', default_value='/dev/video0',
                              description='the v4l2 backend only'),
        DeclareLaunchArgument('camera_info_url', default_value='',
                              description='file://... calibration, which must '
                                          'declare 640x480'),
        DeclareLaunchArgument('mock_image', default_value='',
                              description='1600x1200 PNG for the mock to replay'),
    ]

    # A LaunchConfiguration is a string, and launch type-infers it: pass
    # frame_rate:=10 and it becomes an int, which rclpy rejects against a
    # double-typed parameter and the node dies on startup. The value_type pins it
    # so both 10 and 10.0 work. (Same footgun as bridge.launch.py.)
    camera = Node(
        package='p4p_camera',
        executable='camera_node',
        name='camera_node',
        output='screen',
        parameters=[{
            'camera_backend': LaunchConfiguration('camera_backend'),
            'frame_rate': ParameterValue(LaunchConfiguration('frame_rate'),
                                         value_type=float),
            'frame_id': LaunchConfiguration('frame_id'),
            'device': LaunchConfiguration('device'),
            'camera_info_url': LaunchConfiguration('camera_info_url'),
            'mock_image': LaunchConfiguration('mock_image'),
        }],
    )

    return LaunchDescription([*args, camera])
