"""Bring up the BNO085 IMU node.

    ros2 launch p4p_imu imu.launch.py
    ros2 launch p4p_imu imu.launch.py imu_backend:=mock
    ros2 launch p4p_imu imu.launch.py i2c_bus:=3 sample_rate_hz:=100
    ros2 launch p4p_imu imu.launch.py sample_rate_hz:=200     # see the README first

With no sensor attached, use imu_backend:=mock -- it publishes a deterministic
synthetic stream on the same topic. There is deliberately no automatic fallback
to it: a synthetic heading that looks real is worse than none.

The host has to have I2C enabled first (dtparam=i2c_arm=on in
/boot/firmware/config.txt, then a reboot) or /dev/i2c-1 will not exist, and the
container will not start at all. See docker/README.md.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """Declare the arguments worth overriding from the command line."""
    args = [
        DeclareLaunchArgument('imu_backend', default_value='bno08x',
                              description='bno08x or mock'),
        DeclareLaunchArgument('i2c_bus', default_value='1',
                              description='/dev/i2c-N; a software i2c-gpio bus is not 1'),
        DeclareLaunchArgument('i2c_address', default_value='74',
                              description='decimal; 74 = 0x4A, 75 = 0x4B'),
        DeclareLaunchArgument('sample_rate_hz', default_value='150.0'),
        DeclareLaunchArgument('oversample', default_value='2.0',
                              description='poll this much faster than the sensor reports'),
        DeclareLaunchArgument('frame_id', default_value='base_link'),
    ]
    imu = Node(
        package='p4p_imu',
        executable='imu_node',
        name='imu_node',
        output='screen',
        parameters=[{
            'imu_backend': LaunchConfiguration('imu_backend'),
            # A LaunchConfiguration is a string, and launch type-infers it: pass
            # sample_rate_hz:=100 and it becomes an int, which rclpy rejects
            # against a double-typed parameter and the node dies on startup. The
            # value_type pins it so both 100 and 100.0 work. (Same footgun as
            # bridge.launch.py and camera.launch.py.)
            'i2c_bus': ParameterValue(LaunchConfiguration('i2c_bus'), value_type=int),
            'i2c_address': ParameterValue(LaunchConfiguration('i2c_address'),
                                          value_type=int),
            'sample_rate_hz': ParameterValue(LaunchConfiguration('sample_rate_hz'),
                                             value_type=float),
            'oversample': ParameterValue(LaunchConfiguration('oversample'),
                                         value_type=float),
            'frame_id': LaunchConfiguration('frame_id'),
        }],
    )
    return LaunchDescription([*args, imu])
