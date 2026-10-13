"""
The launch system end to end: `ros2 launch convchart_ros convchart.launch.py task:=...` on a replayed video.

    python3 -m pytest src/ros/src/convchart_ros/test/test_launch.py -q

A synthetic board video (calib_synth renders at 1600x1200, which the replay downsizes like the Pi does)
drives each task as a user would run it: demo records every frame headless and ends with the video;
infer publishes poses under /local only, never on the robot's /inference_result; finetune labels the
replay and writes its session. Each launch runs on an isolated domain with discovery limited to
localhost. Needs the built workspace (colcon build --symlink-install) and the checkpoints; the window
tasks (calibrate, a demo on screen) are covered by test_launch_plan.py and the node tests.
"""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import cv2
import numpy as np
import pytest
import yaml

from convchart_ros.calib_utils.calib_board import BoardSpec, make_board
from convchart_ros.calib_utils.calib_session import scale_intrinsics
from convchart_ros.calib_utils.calib_synth import (
    DEFAULT_DIST, DEFAULT_FULL_SIZE, DEFAULT_K_FULL, DEFAULT_STREAM_SIZE, diverse_poses, render_view)
from convchart_ros.launch_plan import find_repo

DOMAIN_ID = 61 + os.getpid() % 20
N_FRAMES = 30
TIMEOUT_S = 120.0
REPO = find_repo(Path(__file__))


def _installed():
    try:
        from ament_index_python.packages import get_package_share_directory
        return (Path(get_package_share_directory("convchart_ros")) / "launch" / "convchart.launch.py").exists()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _installed() or shutil.which("ros2") is None or REPO is None,
                                reason="needs the built workspace, the ros2 CLI and the checkpoints")


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    folder = tmp_path_factory.mktemp("clip")
    spec = BoardSpec()
    board = make_board(spec)
    poses = diverse_poses(spec, DEFAULT_K_FULL, DEFAULT_FULL_SIZE)
    for i in range(N_FRAMES):
        rvec, tvec = poses[i % len(poses)]
        frame, _, _ = render_view(board, spec, DEFAULT_K_FULL, DEFAULT_DIST, rvec, tvec, DEFAULT_FULL_SIZE, seed=i)
        cv2.imwrite(str(folder / f"{i:04d}.png"), frame)
    return folder


@pytest.fixture(scope="module")
def ros():
    import rclpy
    saved = os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE")
    os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    rclpy.init(domain_id=DOMAIN_ID)
    try:
        yield rclpy
    finally:
        rclpy.try_shutdown()
        if saved is None:
            os.environ.pop("ROS_AUTOMATIC_DISCOVERY_RANGE", None)
        else:
            os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = saved


def launch(tmp_path, *args):
    env = {**os.environ, "ROS_DOMAIN_ID": str(DOMAIN_ID), "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST"}
    log = open(tmp_path / "launch.log", "w")
    return subprocess.Popen(["ros2", "launch", "convchart_ros", "convchart.launch.py", *args], env=env,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
    try:
        return proc.wait(60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        pytest.fail("the launch did not shut down on SIGINT")


def spin_until(rclpy, node, predicate, proc, what):
    deadline = time.monotonic() + TIMEOUT_S
    while not predicate():
        if proc.poll() is not None:
            pytest.fail(f"the launch exited ({proc.returncode}) before {what}")
        if time.monotonic() > deadline:
            pytest.fail(f"timed out waiting for {what}")
        rclpy.spin_once(node, timeout_sec=0.1)


def test_demo_records_every_replayed_frame_headless_and_ends_with_the_video(clip, tmp_path):
    out = tmp_path / "demo.mp4"
    proc = launch(tmp_path, "task:=demo", f"video:={clip}", "rate:=20", f"record:={out}", "headless:=true")
    try:
        assert proc.wait(TIMEOUT_S) == 0, (tmp_path / "launch.log").read_text()[-3000:]
    finally:
        stop(proc)
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == N_FRAMES
    ok, frame = cap.read()
    assert ok and frame.shape == (int(1.5 * DEFAULT_STREAM_SIZE[1]), int(1.5 * DEFAULT_STREAM_SIZE[0]), 3)


def test_infer_on_a_replay_publishes_poses_under_local_and_never_to_the_robot(ros, clip, tmp_path):
    from convchart_interfaces.msg import RosInferenceResult
    from convchart_qos.qos import INFERENCE_RESULT_QOS

    cfg = yaml.safe_load((REPO / "cfg" / "cfg.yaml").read_text())
    for key in ("detector", "refiner"):
        cfg["MODEL"][key] = str(REPO / cfg["MODEL"][key])
    K = scale_intrinsics(np.array(DEFAULT_K_FULL), DEFAULT_FULL_SIZE, DEFAULT_STREAM_SIZE)
    cfg["CAMERA"] = {**(cfg.get("CAMERA") or {}), "K": K.tolist(), "dist": list(DEFAULT_DIST)}
    config = tmp_path / "cfg" / "cfg.yaml"
    config.parent.mkdir()
    config.write_text(yaml.safe_dump(cfg))
    probe = ros.create_node("launch_probe")
    local, robot = [], []
    probe.create_subscription(RosInferenceResult, "/local/inference_result", local.append, INFERENCE_RESULT_QOS)
    probe.create_subscription(RosInferenceResult, "/inference_result", robot.append, INFERENCE_RESULT_QOS)
    proc = launch(tmp_path, "task:=infer", f"config:={config}", f"video:={clip}", "rate:=10", "loop:=0")
    try:
        spin_until(ros, probe, lambda: len(local) >= 15, proc, "15 poses on /local/inference_result")
        assert stop(proc) == 0
        for _ in range(10):
            ros.spin_once(probe, timeout_sec=0.05)
        assert not robot, "a replayed pose reached the robot's /inference_result"
        posed = [r for r in local if not r.reason and r.covariance_valid]
        assert len(posed) >= 10 and all(np.isfinite(r.rms) and r.rms < 1.0 for r in posed)
    finally:
        stop(proc)
        probe.destroy_node()


@pytest.mark.skipif(REPO is None or not (REPO.parent / "Conv-ChArT" / "configs" / "rel_w882_c2.yaml").is_file(),
                    reason="needs the Conv-ChArT checkout beside this repo, with its release config")
def test_finetune_on_a_replay_labels_frames_and_writes_its_session(ros, clip, tmp_path):
    from std_msgs.msg import String

    from convchart_ros.ssl_node import OUT_QOS

    probe = ros.create_node("launch_probe")
    stats = []
    probe.create_subscription(String, "/local/ssl/stats", lambda m: stats.append(json.loads(m.data)), OUT_QOS)
    out = tmp_path / "ssl"
    proc = launch(tmp_path, "task:=finetune", f"video:={clip}", "rate:=10", "loop:=0", "training:=false",
                  f"output_dir:={out}")
    try:
        spin_until(ros, probe, lambda: stats and stats[-1]["harvest"]["totals"].get("labelled", 0) >= 10, proc,
                   "10 labelled frames on /local/ssl/stats")
        assert stop(proc) == 0
        (session,) = out.iterdir()
        summary = json.loads((session / "summary.json").read_text())
        assert summary["frames"] >= 10 and summary["harvest"]["totals"]["anchors"] >= 10
        assert (session / "report" / "dashboard.pdf").exists()
    finally:
        stop(proc)
        probe.destroy_node()
