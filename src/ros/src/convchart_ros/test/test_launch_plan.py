"""
What convchart.launch.py starts for each task and source, and what it refuses (launch_plan.plan; ROS-free).

    python3 -m pytest src/ros/src/convchart_ros/test/test_launch_plan.py -q

A throwaway repo layout (cfg/cfg.yaml, checkpoints/, a sibling Conv-ChArT/configs/) stands in for the
real one, so the defaults are exercised without touching the real config.
"""
import os
from pathlib import Path

import pytest
import yaml

from convchart_ros.launch_plan import find_repo, plan, PlanError

K = [[448.0, 0.0, 319.5], [0.0, 448.0, 239.5], [0.0, 0.0, 1.0]]


def write_cfg(path: Path, K=None, detector="checkpoints/detector_882k.onnx"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"MODEL": {"detector": detector, "refiner": "checkpoints/refiner.onnx"},
                                    "CAMERA": {"K": K, "dist": None if K is None else [0.0] * 5}}))
    return path


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "Conv-ChArT-Wireless-Inference"
    (root / "checkpoints").mkdir(parents=True)
    write_cfg(root / "cfg" / "cfg.yaml")
    write_cfg(root / "cfg" / "calibrated.yaml", K=K)
    configs = tmp_path / "Conv-ChArT" / "configs"
    configs.mkdir(parents=True)
    (configs / "rel_w882_c2.yaml").write_text(yaml.safe_dump({"synth": {"backgrounds": str(tmp_path / "coco")}}))
    (root / "clips" / "walk").mkdir(parents=True)
    (root / "clips" / "walk" / "0000.png").write_bytes(b"")
    return root


def run(repo, env=None, **args):
    return plan(args, repo, {"DISPLAY": ":0"} if env is None else env)


def names(p):
    return [n.executable for n in p.nodes]


def test_find_repo_walks_up_to_the_config(repo):
    assert find_repo(repo / "src" / "ros" / "install" / "x" / "launch" / "convchart.launch.py") == repo
    assert find_repo(repo.parent / "elsewhere" / "f.py") is None


@pytest.mark.parametrize("task, source, expected, namespace", [
    ("calibrate", "pi", ["calibrate"], ""),
    ("calibrate", "fake", ["fake_camera", "calibrate"], "local"),
    ("demo", "pi", ["pose_view"], ""),
    ("demo", "video", ["replay", "pose_view"], "local"),
    ("demo", "fake", ["fake_camera", "pose_view"], "local"),
    ("finetune", "pi", ["ssl"], ""),
    ("finetune", "video", ["replay", "ssl"], "local"),
    ("infer", "pi", ["image_relay", "inference", "telemetry"], ""),
    ("infer", "video", ["replay", "inference"], "local"),
])
def test_each_task_and_source_starts_its_nodes_in_the_right_namespace(repo, task, source, expected, namespace):
    args = {"task": task, "source": source}
    if source == "video":
        args["video"] = str(repo / "clips" / "walk")
    if task == "infer":
        args["config"] = str(repo / "cfg" / "calibrated.yaml")
    p = run(repo, **args)
    assert names(p) == expected and p.namespace == namespace and p.source == source
    ends = ("calibrate", "pose_view", "ssl", "inference", "image_relay", "telemetry")
    assert [n.ends_launch for n in p.nodes] == [n in ends for n in expected]


def test_source_auto_follows_video(repo):
    assert run(repo, task="demo").source == "pi"
    p = run(repo, task="demo", video=str(repo / "clips" / "walk"))
    assert p.source == "video" and p.namespace == "local"


def test_a_replayed_demo_ends_with_its_video_and_waits_for_its_viewer(repo):
    p = run(repo, task="demo", video=str(repo / "clips" / "walk"), rate="20", loop="2", record="out.mp4",
            headless="true")
    replay, view = p.nodes
    assert replay.parameters == {"source": str(repo / "clips" / "walk"), "rate": 20.0, "loop": 2,
                                 "wait_for_subscribers": 1, "allow_root_namespace": False}
    assert view.parameters["exit_after_idle_s"] == 3.0 and view.parameters["record_fps"] == 20.0
    assert view.parameters["headless"] and Path(view.parameters["record"]).is_absolute()
    live = run(repo, task="demo").nodes[0]
    assert live.parameters["exit_after_idle_s"] == 0.0


def test_infer_with_view_waits_for_both_consumers_and_records_headless_without_a_window(repo):
    cfg = str(repo / "cfg" / "calibrated.yaml")
    p = run(repo, task="infer", config=cfg, video=str(repo / "clips" / "walk"), view="true")
    assert names(p) == ["replay", "inference", "pose_view"] and p.nodes[0].parameters["wait_for_subscribers"] == 2
    assert not p.nodes[2].ends_launch and not p.nodes[2].parameters["headless"]
    relay, _, recorded = run(repo, task="infer", config=cfg, record="poses.mp4", dashboard="false").nodes
    assert recorded.executable == "pose_view" and recorded.parameters["headless"]
    assert relay.executable == "image_relay" and recorded.remappings == (("image", "laptop/image"),)


def test_live_infer_with_the_overlay_and_the_dashboard_shares_one_relayed_copy_of_the_frames(repo):
    cfg = str(repo / "cfg" / "calibrated.yaml")
    p = run(repo, task="infer", config=cfg, view="true", dashboard="true")
    assert names(p) == ["image_relay", "inference", "pose_view", "telemetry"] and p.namespace == ""
    relay, inference, view, dashboard = p.nodes
    assert relay.parameters == {"output_topic": "laptop/image"} and relay.ends_launch and not relay.remappings
    assert inference.remappings == view.remappings == (("image", "laptop/image"),)
    assert dashboard.parameters == {"headless": False, "record": "", "snapshot": "", "image_topic": "laptop/image",
                                    "config": cfg}
    assert dashboard.ends_launch and not dashboard.remappings
    assert "frames: one copy over the WiFi, relayed on /laptop/image to inference, pose_view, telemetry" in p.notes
    assert names(run(repo, task="infer", config=cfg, dashboard="true")) == ["image_relay", "inference", "telemetry"]


def test_a_single_reader_or_local_frames_need_no_relay(repo):
    cfg = str(repo / "cfg" / "calibrated.yaml")
    (alone,) = run(repo, task="infer", config=cfg, dashboard="false").nodes
    assert alone.executable == "inference" and not alone.remappings
    local = run(repo, task="infer", config=cfg, video=str(repo / "clips" / "walk"), view="true")
    assert names(local) == ["replay", "inference", "pose_view"] and not any(n.remappings for n in local.nodes)


@pytest.mark.parametrize("args, message", [
    ({"task": "demo"}, "dashboard:=true is for task:=infer"),
    ({"task": "telemetry"}, "dashboard:=true is for task:=infer"),
    ({"task": "infer", "source": "fake"}, "needs the Pi's frames"),
    ({"task": "infer", "dashboard": "maybe"}, "dashboard:= must be auto, true or false"),
])
def test_the_dashboard_rides_only_on_live_inference(repo, args, message):
    with pytest.raises(PlanError, match=message):
        run(repo, **{"config": str(repo / "cfg" / "calibrated.yaml"), "dashboard": "true", **args})


def test_a_live_infer_brings_the_dashboard_up_when_there_is_a_display(repo):
    cfg = str(repo / "cfg" / "calibrated.yaml")
    assert names(run(repo, task="infer", config=cfg)) == ["image_relay", "inference", "telemetry"]
    assert names(run(repo, env={"WAYLAND_DISPLAY": "wayland-0"}, task="infer", config=cfg))[-1] == "telemetry"
    headless = run(repo, env={}, task="infer", config=cfg)
    assert names(headless) == ["inference"] and any(n.startswith("no display: the telemetry") for n in headless.notes)
    quiet = run(repo, env={}, task="infer", config=cfg, dashboard="false")
    assert not any(n.startswith("no display") for n in quiet.notes)
    assert names(run(repo, task="demo")) == ["pose_view"]
    assert names(run(repo, task="infer", config=cfg, video=str(repo / "clips" / "walk"))) == ["replay", "inference"]


def test_the_dashboard_of_a_live_infer_needs_a_display(repo):
    with pytest.raises(PlanError, match=r"telemetry opens a window but DISPLAY is not set \(.*on the host\)$"):
        run(repo, env={}, task="infer", config=str(repo / "cfg" / "calibrated.yaml"), dashboard="true")


def test_an_explicit_root_namespace_lets_the_replay_through_with_a_warning(repo):
    p = run(repo, task="demo", video=str(repo / "clips" / "walk"), namespace="/")
    assert p.namespace == "" and p.nodes[0].parameters["allow_root_namespace"]
    assert any("WARNING" in n for n in p.notes)


def test_finetune_finds_the_model_config_and_reports_missing_backgrounds(repo):
    p = run(repo, task="finetune")
    params = p.nodes[0].parameters
    assert params["model_config"] == str(repo.parent / "Conv-ChArT" / "configs" / "rel_w882_c2.yaml")
    assert params["backgrounds"] == "" and params["convchart_root"] == str(repo.parent / "Conv-ChArT")
    assert any("not trained on" in n for n in p.notes)
    (repo.parent / "coco").mkdir()
    assert run(repo, task="finetune").nodes[0].parameters["backgrounds"] == str(repo.parent / "coco")
    assert not any("not trained on" in n for n in run(repo, task="finetune", training="false").notes)


@pytest.mark.parametrize("args, message", [
    ({}, "task:= must be one of"),
    ({"task": "fly"}, "task:= must be one of"),
    ({"task": "demo", "source": "usb"}, "source:= must be one of"),
    ({"task": "demo", "source": "video"}, "needs video:="),
    ({"task": "demo", "video": "/no/such/clip.mp4"}, "does not exist"),
    ({"task": "demo", "source": "fake", "video": "x"}, "drop one of them"),
    ({"task": "calibrate", "video": "CLIP"}, "a video cannot serve"),
    ({"task": "calibrate", "source": "fake", "auto_replace": "true"}, "synthetic camera's K"),
    ({"task": "infer"}, "no CAMERA.K"),
    ({"task": "demo", "headless": "true"}, "needs record:="),
    ({"task": "demo", "headless": "maybe"}, "true or false"),
    ({"task": "demo", "video": "CLIP", "rate": "fast"}, "must be a number"),
    ({"task": "demo", "video": "CLIP", "rate": "0"}, "rate:= must be > 0"),
    ({"task": "finetune", "model_config": "/no/such.yaml"}, "does not exist"),
    ({"task": "finetune", "backgrounds": "/no/such/dir"}, "is not a folder"),
    ({"task": "demo", "config": "/no/such/cfg.yaml"}, "does not exist"),
])
def test_bad_arguments_are_refused_with_a_reason(repo, args, message):
    args = {k: str(repo / "clips" / "walk") if v == "CLIP" else v for k, v in args.items()}
    with pytest.raises(PlanError, match=message):
        run(repo, **args)


def test_live_finetune_caps_the_labelling_rate_and_keeps_the_frames_and_a_replay_does_neither(repo):
    live = run(repo, task="finetune").nodes[0].parameters
    assert live["max_rate_hz"] == 7.5 and live["save_frames"] is True
    assert run(repo, task="finetune", max_rate="0").nodes[0].parameters["max_rate_hz"] == 0.0
    replay = run(repo, task="finetune", video=str(repo / "clips" / "walk")).nodes[1].parameters
    assert replay["max_rate_hz"] == 0.0 and replay["save_frames"] is False
    with pytest.raises(PlanError, match="max_rate:= must be >= 0"):
        run(repo, task="finetune", max_rate="-1")


def test_window_tasks_need_a_display_and_headless_recording_does_not(repo):
    with pytest.raises(PlanError, match="calibrate opens a window but DISPLAY is not set"):
        run(repo, env={}, task="calibrate")
    with pytest.raises(PlanError, match="headless"):
        run(repo, env={}, task="demo")
    assert run(repo, env={}, task="demo", record="out.mp4", headless="true").nodes
    assert run(repo, env={"WAYLAND_DISPLAY": "wayland-0"}, task="calibrate").nodes
    assert run(repo, env={}, task="finetune").nodes


def test_an_unknown_base_detector_needs_its_model_config(repo):
    other = repo / "checkpoints" / "detector_9k.onnx"
    other.write_bytes(b"")
    with pytest.raises(PlanError, match="model_config:="):
        run(repo, task="finetune", base_detector=str(other))


def test_finetune_trains_from_the_release_weights_whatever_is_deployed(repo):
    deployed = write_cfg(repo / "cfg" / "deployed.yaml", detector="checkpoints/detector_882k_ssl_20261013_r3.onnx")
    params = run(repo, task="finetune", config=str(deployed)).nodes[0].parameters
    assert params["base_detector"] == str(repo / "checkpoints" / "detector_882k.onnx") and params["resume"] == ""
    assert params["model_config"] == str(repo.parent / "Conv-ChArT" / "configs" / "rel_w882_c2.yaml")
    tuned = repo / "checkpoints" / "detector_882k_ssl_r2.onnx"
    tuned.write_bytes(b"")
    params = run(repo, task="finetune", base_detector=str(tuned)).nodes[0].parameters
    assert params["base_detector"] == str(tuned) and params["model_config"].endswith("rel_w882_c2.yaml")
    with pytest.raises(PlanError, match="base_detector .* does not exist"):
        run(repo, task="finetune", base_detector=str(repo / "checkpoints" / "nope.onnx"))


def test_finetune_resumes_a_session_folder_or_the_latest(repo, tmp_path):
    assert run(repo, task="finetune", resume="latest").nodes[0].parameters["resume"] == "latest"
    session = tmp_path / "ssl_output" / "20261013-021802"
    session.mkdir(parents=True)
    with pytest.raises(PlanError, match="not an ssl session folder"):
        run(repo, task="finetune", resume=str(session))
    (session / "session.json").write_text("{}")
    p = run(repo, task="finetune", resume=str(session))
    assert p.nodes[0].parameters["resume"] == str(session) and any("resuming" in n for n in p.notes)


def test_without_a_repo_the_config_must_be_given(repo):
    with pytest.raises(PlanError, match="config:="):
        plan({"task": "demo"}, None, {"DISPLAY": ":0"})
    assert plan({"task": "demo", "config": str(repo / "cfg" / "cfg.yaml")}, None, {"DISPLAY": ":0"}).nodes


def test_telemetry_starts_only_the_dashboard_in_the_root_namespace(repo):
    p = run(repo, task="telemetry")
    (node,) = p.nodes
    assert node.executable == "telemetry" and node.ends_launch and p.namespace == "" and p.source == "pi"
    assert node.parameters == {"headless": False, "record": "", "snapshot": "", "image_topic": "image",
                               "config": str(repo / "cfg" / "cfg.yaml")}
    assert any("telemetry dashboard of /, subscribing only" in n for n in p.notes)
    assert any(n.startswith("frames: the Pi's camera node on /image") for n in p.notes)
    assert run(repo, task="telemetry", namespace="/sim").namespace == "sim"


def test_telemetry_headless_needs_a_snapshot_or_a_recording(repo, tmp_path):
    with pytest.raises(PlanError, match="needs record:=<out.mp4> or snapshot:=<out.png>"):
        run(repo, env={}, task="telemetry", headless="true")
    with pytest.raises(PlanError, match="telemetry opens a window but DISPLAY is not set.*snapshot:=out.png"):
        run(repo, env={}, task="telemetry")
    p = run(repo, env={}, task="telemetry", headless="true", snapshot=str(tmp_path / "dash.png"))
    assert p.nodes[0].parameters["headless"] and p.nodes[0].parameters["snapshot"] == str(tmp_path / "dash.png")
    recorded = run(repo, env={}, task="telemetry", headless="true", record="dash.mp4").nodes[0]
    assert Path(recorded.parameters["record"]).is_absolute() and not recorded.parameters["snapshot"]
    with pytest.raises(PlanError, match="must name a .png file"):
        run(repo, task="telemetry", snapshot="dash.jpg")


@pytest.mark.parametrize("args", [{"video": "CLIP"}, {"source": "video", "video": "CLIP"}, {"source": "fake"}])
def test_telemetry_starts_no_frame_source_of_its_own(repo, args):
    args = {k: str(repo / "clips" / "walk") if v == "CLIP" else v for k, v in args.items()}
    with pytest.raises(PlanError, match="starts no frame source"):
        run(repo, task="telemetry", **args)


def test_telemetry_without_the_camera_or_without_dock_msgs(repo, tmp_path):
    p = run(repo, task="telemetry", camera="false")
    assert p.nodes[0].parameters["image_topic"] == "" and not any(n.startswith("frames:") for n in p.notes)
    assert any(n.startswith("WARNING: dock_msgs is not on AMENT_PREFIX_PATH") for n in p.notes)
    prefix = tmp_path / "controller_ws" / "install" / "dock_msgs"
    marker = prefix / "share" / "ament_index" / "resource_index" / "packages" / "dock_msgs"
    marker.parent.mkdir(parents=True)
    marker.write_text("")
    env = {"DISPLAY": ":0", "AMENT_PREFIX_PATH": os.pathsep.join([str(tmp_path / "elsewhere"), str(prefix)])}
    assert not any("dock_msgs" in n for n in run(repo, env=env, task="telemetry").notes)
    with pytest.raises(PlanError, match="camera:= must be true or false"):
        run(repo, task="telemetry", camera="sometimes")
