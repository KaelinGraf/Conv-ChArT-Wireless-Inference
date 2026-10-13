"""Launch one task of the Conv-ChArT wireless pipeline on the laptop, picked with task:=.

    ros2 launch convchart_ros convchart.launch.py task:=calibrate [auto_replace:=true]   # Pi camera, GUI
    ros2 launch convchart_ros convchart.launch.py task:=calibrate detector:=opencv        # OpenCV's corners
    ros2 launch convchart_ros convchart.launch.py task:=calibrate source:=fake            # synthetic camera
    ros2 launch convchart_ros convchart.launch.py task:=demo video:=clip.mp4 [record:=out.mp4] [headless:=true]
    ros2 launch convchart_ros convchart.launch.py task:=demo                              # live, Pi stream
    ros2 launch convchart_ros convchart.launch.py task:=finetune [video:=clip.mp4] [backgrounds:=<dir>]
    ros2 launch convchart_ros convchart.launch.py task:=finetune resume:=latest [base_detector:=<onnx>]   # carry on
    ros2 launch convchart_ros convchart.launch.py task:=infer [view:=true]   # poses to the robot, and the dashboard
    ros2 launch convchart_ros convchart.launch.py task:=infer dashboard:=false            # no dashboard
    ros2 launch convchart_ros convchart.launch.py task:=telemetry [snapshot:=dash.png] [headless:=true]  # dashboard
    ros2 launch convchart_ros convchart.launch.py --show-args                             # every argument

source:= is pi (default: the camera node must be running on the Pi), video (replays video:=, the default
whenever video:= is given) or fake (the synthetic camera). Frames that do not come from the Pi stay in
the namespace /local, so a replay can never feed the robot. The launch ends when the task's own node
exits (closing the calibration, overlay or dashboard window, the end of a replayed demo video, Ctrl-C).
A live infer brings the telemetry dashboard up with it when there is a display (dashboard:=auto), and with the
overlay or the dashboard it starts image_relay, so the Pi's frames cross the WiFi once.
telemetry needs the controller workspace sourced too (source controller_ws/install/setup.bash) for its
EKF, stage and MPC panels. The rules live in convchart_ros/launch_plan.py.
"""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, Shutdown
from launch_ros.actions import Node

from convchart_ros.launch_plan import ARGUMENTS, find_repo, plan, PlanError, SOURCES, TASKS


def _launch(context):
    args = {name: context.launch_configurations.get(name) for name in ARGUMENTS}
    try:
        p = plan(args, find_repo(Path(__file__).resolve()))
    except PlanError as error:
        raise RuntimeError(f"convchart.launch.py: {error}") from None
    actions = [LogInfo(msg=f"[convchart] {line}") for line in p.notes]
    for spec in p.nodes:
        actions.append(Node(
            package=spec.package, executable=spec.executable, namespace=p.namespace or None,
            output="screen", parameters=[spec.parameters], remappings=list(spec.remappings),
            sigterm_timeout=str(spec.stop_timeout_s),
            on_exit=[Shutdown(reason=f"{spec.executable} exited")] if spec.ends_launch else None))
    return actions


def generate_launch_description():
    declared = []
    for name, (default, description) in ARGUMENTS.items():
        choices = {"task": list(TASKS), "source": ["auto", *SOURCES]}.get(name)
        if default is None:
            declared.append(DeclareLaunchArgument(name, description=description, choices=choices))
        else:
            declared.append(DeclareLaunchArgument(name, default_value=default, description=description,
                                                  choices=choices))
    return LaunchDescription([*declared, OpaqueFunction(function=_launch)])
