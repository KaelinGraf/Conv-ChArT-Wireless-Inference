"""Bring up the serial bridge.

    ros2 launch p4p_serial_bridge bridge.launch.py
    ros2 launch p4p_serial_bridge bridge.launch.py port:=/tmp/ttyFakeMega auto_arm:=true

With no Mega attached, run tools/fake_mega.py first and point `port` at the
pseudo-terminal it prints. Both have to be on the same side of the container
boundary: Docker gives a container its own devpts, so a pty created on the host
cannot be passed in through `devices:`.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Declare the arguments worth overriding from the command line."""
    args = [
        DeclareLaunchArgument('port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('command_rate_hz', default_value='20.0'),
        DeclareLaunchArgument('cmd_timeout', default_value='0.25'),
        DeclareLaunchArgument('auto_arm', default_value='false'),
        DeclareLaunchArgument('frame_id', default_value='base_link'),
    ]
    bridge = Node(
        package='p4p_serial_bridge',
        executable='serial_bridge',
        name='serial_bridge',
        output='screen',
        parameters=[{
            'port': LaunchConfiguration('port'),
            'command_rate_hz': LaunchConfiguration('command_rate_hz'),
            'cmd_timeout': LaunchConfiguration('cmd_timeout'),
            'auto_arm': LaunchConfiguration('auto_arm'),
            'frame_id': LaunchConfiguration('frame_id'),
        }],
    )
    return LaunchDescription([*args, bridge])
