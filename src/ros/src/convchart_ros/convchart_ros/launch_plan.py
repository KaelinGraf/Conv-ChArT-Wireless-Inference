"""
The rules behind convchart.launch.py: launch arguments in, the nodes to start out (ROS-free, so testable).

    from convchart_ros.launch_plan import plan, find_repo
    p = plan({"task": "demo", "video": "clip.mp4"}, find_repo(path))   # Plan; PlanError if the args are wrong

Tasks: calibrate (the calibration GUI, full-resolution stills from the camera's image_full_res),
demo (the pose overlay window, optionally recorded), finetune (live self-supervised fine-tuning; it also keeps
every frame from the Pi, in its session's stream/; resume:= carries on in a session, base_detector:= is what every
round trains from), infer (the inference node; its poses go to the robot; with a display the telemetry dashboard
comes up with it, dashboard:=false leaves it out),
telemetry (the robot's live dashboard: camera, EKF, MPC and drive; it only subscribes, and needs dock_msgs
on AMENT_PREFIX_PATH for all but the camera, cmd_vel, drive status and poses). Sources: pi (the Pi's camera
node; nothing is started here), video (replay a file or image folder; the default when video is given) or fake (the
synthetic camera from the calibration tests). Frames that do not come from the Pi are kept in the
namespace 'local' unless namespace says otherwise, so they never reach the robot's /image and their
poses never reach its /inference_result. When more than one node of a live infer reads the Pi's frames
(the overlay, the dashboard), image_relay is the only one that subscribes to them, so each frame crosses
the WiFi once, and the others read its copy on laptop/image.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import os
from pathlib import Path

import yaml

TASKS = ("calibrate", "demo", "finetune", "infer", "telemetry")
SOURCES = ("pi", "video", "fake")
LOCAL_NAMESPACE = "local"
RELEASE_DETECTOR = Path("checkpoints") / "detector_882k.onnx"
RELAY_TOPIC = "laptop/image"
MODEL_CONFIGS = {"detector_882k": "rel_w882_c2.yaml", "detector_222k": "rel_w25_lam2.yaml"}

ARGUMENTS = {
    "task": (None, "what to run: calibrate, demo, finetune, infer or telemetry"),
    "source": ("auto", "where frames come from: pi, video or fake; auto picks video when video:= is set, "
                       "else pi"),
    "video": ("", "video file or image folder to replay (source video)"),
    "config": ("", "pipeline config; default <repo>/cfg/cfg.yaml"),
    "namespace": ("auto", "namespace for every node; auto is / for the Pi and /local for video or fake"),
    "rate": ("10.0", "replay rate in Hz (source video)"),
    "loop": ("1", "replay passes, 0 for ever (source video)"),
    "record": ("", "demo / infer / telemetry: write the overlay (telemetry: the dashboard) to this mp4 (infer "
                   "without view:=true records headless)"),
    "headless": ("false", "demo / telemetry: no window, record only (telemetry: or snapshot)"),
    "snapshot": ("", "telemetry: rewrite this PNG with the dashboard every second"),
    "camera": ("true", "telemetry: show the Pi's image too (a second copy of the stream over the WiFi; "
                       "task:=infer dashboard:=true shares the inference node's instead)"),
    "view": ("false", "infer: also open the pose overlay window"),
    "dashboard": ("auto", "infer: also open the telemetry dashboard, on the inference node's copy of the frames "
                          "(closing it ends the launch): auto opens it when there is a display, true insists, false "
                          "leaves it out"),
    "auto_replace": ("false", "calibrate: write K and dist into the config after every solve"),
    "output_dir": ("", "calibrate / finetune: session folder root; default beside the config"),
    "model_config": ("", "finetune: the Conv-ChArT config the deployed detector was trained with; "
                         "default picked from MODEL.detector"),
    "backgrounds": ("", "finetune: image folder for the synthetic replay; default the model config's "
                        "synth.backgrounds"),
    "training": ("true", "finetune: run fine-tuning rounds (false only banks labelled frames)"),
    "auto_deploy": ("false", "finetune: switch the fine-tune node's own teacher to accepted models"),
    "max_rate": ("7.5", "finetune on the Pi: label at most this many frames per second, 0 for every frame"),
    "resume": ("", "finetune: carry on in an ssl_output session folder (or latest): its stored frames, rounds and "
                   "best model, which relabels everything stored"),
    "base_detector": ("", "finetune: the weights every round trains from; default the release "
                          "checkpoints/detector_882k.onnx"),
}


class PlanError(ValueError):
    pass


@dataclass(frozen=True)
class NodeSpec:
    executable: str
    parameters: dict
    ends_launch: bool = False
    package: str = "convchart_ros"
    stop_timeout_s: float = 5.0
    remappings: tuple = ()


@dataclass(frozen=True)
class Plan:
    task: str
    source: str
    namespace: str
    nodes: tuple
    notes: tuple = field(default=())


def find_repo(start: Path) -> Path | None:
    for parent in Path(start).resolve().parents:
        if (parent / "cfg" / "cfg.yaml").is_file() and (parent / "checkpoints").is_dir():
            return parent
    return None


def _bool(name: str, value: str, choices: str = "true or false") -> bool:
    v = str(value).strip().lower()
    if v in ("true", "1", "yes", "on"):
        return True
    if v in ("false", "0", "no", "off", ""):
        return False
    raise PlanError(f"{name}:= must be {choices}, got {value!r}")


def _number(name: str, value: str, kind):
    try:
        return kind(str(value).strip())
    except ValueError:
        raise PlanError(f"{name}:= must be a number, got {value!r}") from None


def _path(value: str) -> Path:
    return Path(str(value).strip()).expanduser().resolve()


def _relayed(spec: NodeSpec) -> NodeSpec:
    """The same node, reading image_relay's copy of the frames instead of the Pi's."""
    if spec.executable == "telemetry":
        return replace(spec, parameters={**spec.parameters, "image_topic": RELAY_TOPIC})
    return replace(spec, remappings=(("image", RELAY_TOPIC),))


def _has_package(name: str, env) -> bool:
    return any((Path(prefix) / "share" / "ament_index" / "resource_index" / "packages" / name).is_file()
               for prefix in str(env.get("AMENT_PREFIX_PATH", "")).split(os.pathsep) if prefix)


def plan(args: dict, repo: Path | None, env=None) -> Plan:
    env = os.environ if env is None else env
    a = {k: str(v) for k, (v, _) in ARGUMENTS.items() if v is not None}
    a.update({k: str(v) for k, v in args.items() if v is not None})
    task = a.get("task", "").strip().lower()
    if task not in TASKS:
        raise PlanError(f"task:= must be one of {', '.join(TASKS)} (got {a.get('task', '')!r})")
    if a["config"].strip():
        config = _path(a["config"])
    elif repo is not None:
        config = Path(repo) / "cfg" / "cfg.yaml"
    else:
        raise PlanError("cannot find the repo (cfg/cfg.yaml) above the launch file: pass config:=<cfg.yaml>")
    if not config.is_file():
        raise PlanError(f"config {config} does not exist")
    cfg = yaml.safe_load(config.read_text()) or {}

    video = a["video"].strip()
    source = a["source"].strip().lower()
    if source == "auto":
        source = "video" if video else "pi"
    if source not in SOURCES:
        raise PlanError(f"source:= must be one of auto, {', '.join(SOURCES)} (got {a['source']!r})")
    if source == "video":
        if not video:
            raise PlanError("source:=video needs video:=<video file or image folder>")
        if not _path(video).exists():
            raise PlanError(f"video {_path(video)} does not exist")
    elif video:
        raise PlanError(f"video:= is set but source:={source}: drop one of them")
    if task == "calibrate" and source == "video":
        raise PlanError("calibration takes full-resolution stills from a camera's image_full_res service, which "
                        "a video cannot serve: use source:=pi (the default) or source:=fake")
    if task == "telemetry" and source != "pi":
        raise PlanError("telemetry shows the robot's own topics and starts no frame source: drop source:= and "
                        "video:=")

    ns = a["namespace"].strip()
    ns = ("" if source == "pi" else LOCAL_NAMESPACE) if ns == "auto" else ns.strip("/")
    shown = f"/{ns}" if ns else "/"
    notes = []
    camera = _bool("camera", a["camera"])
    if source == "pi":
        if task != "telemetry" or camera:
            notes.append(f"frames: the Pi's camera node on {shown.rstrip('/')}/image (on the Pi: "
                         "ros2 launch p4p_camera camera.launch.py)")
    elif source == "video":
        notes.append(f"frames: replaying {_path(video)} on {shown.rstrip('/')}/image")
    else:
        notes.append(f"frames: the synthetic camera on {shown.rstrip('/')}/image")
    if source != "pi" and not ns:
        notes.append("WARNING: local frames in the root namespace reach the robot's inference node")

    record = str(_path(a["record"])) if a["record"].strip() else ""
    snapshot = str(_path(a["snapshot"])) if a["snapshot"].strip() else ""
    headless, view = _bool("headless", a["headless"]), _bool("view", a["view"])
    has_display = bool(env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"))
    if a["dashboard"].strip().lower() == "auto":
        dashboard = task == "infer" and source == "pi" and has_display
        if task == "infer" and source == "pi" and not has_display:
            notes.append("no display: the telemetry dashboard is left out (dashboard:=false hides this note)")
    else:
        dashboard = _bool("dashboard", a["dashboard"], "auto, true or false")
    if dashboard and task != "infer":
        raise PlanError("dashboard:=true is for task:=infer (task:=telemetry is the dashboard on its own)")
    if dashboard and source != "pi":
        raise PlanError("dashboard:=true shows the robot's own topics, so it needs the Pi's frames: drop source:= "
                        "and video:=")
    if headless and not record and not (task == "telemetry" and snapshot):
        raise PlanError("headless:=true needs record:=<out.mp4>" + " or snapshot:=<out.png>" * (task == "telemetry"))
    if snapshot and Path(snapshot).suffix.lower() != ".png":
        raise PlanError(f"snapshot:= must name a .png file, got {snapshot}")
    rate = _number("rate", a["rate"], float)
    loop = _number("loop", a["loop"], int)
    if not rate > 0 or loop < 0:
        raise PlanError(f"rate:= must be > 0 and loop:= >= 0 (got {rate}, {loop})")

    def pose_view(ends_launch, headless):
        return NodeSpec("pose_view", {"config": str(config), "record": record, "headless": headless,
                                      "record_fps": rate if source == "video" else 10.0,
                                      "exit_after_idle_s": 3.0 if source == "video" else 0.0}, ends_launch,
                        stop_timeout_s=15.0)

    def telemetry(headless, record, snapshot, image_topic):
        return NodeSpec("telemetry", {"headless": headless, "record": record, "snapshot": snapshot,
                                      "image_topic": image_topic, "config": str(config)}, True, stop_timeout_s=10.0)

    nodes = []
    if task == "calibrate":
        auto_replace = _bool("auto_replace", a["auto_replace"])
        if auto_replace and source == "fake":
            raise PlanError(f"auto_replace:=true with source:=fake would write the synthetic camera's K into "
                            f"{config}")
        nodes.append(NodeSpec("calibrate", {"config": str(config), "auto_replace": auto_replace,
                                            "output_dir": a["output_dir"].strip()}, True))
        notes.append("task: the calibration window (SPACE capture, U undo, M coverage, Q quit); stills come from "
                     f"{shown.rstrip('/')}/image_full_res" + (", written into the config" if auto_replace else ""))
    elif task == "demo":
        nodes.append(pose_view(True, headless))
        notes.append("task: the pose overlay" + (f", recorded to {record}" if record else "")
                     + (" (headless)" if headless else " (q or Esc quits)")
                     + ("; ends with the video" if source == "video" and loop else ""))
    elif task == "infer":
        camera = cfg.get("CAMERA") or {}
        if camera.get("K") is None or camera.get("dist") is None:
            raise PlanError(f"{config} has no CAMERA.K / CAMERA.dist yet: calibrate first (task:=calibrate "
                            "auto_replace:=true), or use task:=demo, which runs with a nominal K")
        nodes.append(NodeSpec("inference", {"config": str(config)}, True))
        if view or record:
            nodes.append(pose_view(False, headless or not view))
        if dashboard:
            nodes.append(telemetry(False, "", "", "image"))
        notes.append(f"task: inference, poses on {shown.rstrip('/')}/inference_result"
                     + (" (the robot consumes these)" if not ns else "")
                     + ("; the telemetry dashboard (q or Esc quits, and ends the launch)" if dashboard else ""))
        if source == "pi" and len(nodes) > 1:
            nodes = [NodeSpec("image_relay", {"output_topic": RELAY_TOPIC}, True), *map(_relayed, nodes)]
            notes.append(f"frames: one copy over the WiFi, relayed on {shown.rstrip('/')}/{RELAY_TOPIC} to "
                         + ", ".join(n.executable for n in nodes[1:]))
    elif task == "telemetry":
        nodes.append(telemetry(headless, record, snapshot, "image" if camera else ""))
        notes.append(f"task: the telemetry dashboard of {shown}, subscribing only"
                     + (f", recorded to {record}" if record else "")
                     + (f", snapshots to {snapshot}" if snapshot else "")
                     + (" (headless)" if headless else " (q or Esc quits)") + ("" if camera else "; no camera"))
    else:
        convchart = (Path(repo).parent / "Conv-ChArT") if repo is not None else None
        if a["base_detector"].strip():
            base = _path(a["base_detector"])
            if not base.is_file():
                raise PlanError(f"base_detector {base} does not exist")
        else:
            base = (Path(repo) if repo is not None else config.parent.parent) / RELEASE_DETECTOR
        resume = a["resume"].strip()
        if resume and resume != "latest":
            resume = str(_path(resume))
            if not (Path(resume) / "session.json").is_file():
                raise PlanError(f"resume:= {resume} is not an ssl session folder (no session.json)")
        if a["model_config"].strip():
            model_config = _path(a["model_config"])
        else:
            stem = base.stem
            name = MODEL_CONFIGS.get(stem) or next((v for k, v in MODEL_CONFIGS.items() if stem.startswith(k + "_")),
                                                   None)
            if name is None or convchart is None:
                raise PlanError(f"no known training config for base detector {stem!r}: pass model_config:=<Conv-ChArT "
                                "config the detector was trained with>")
            model_config = convchart / "configs" / name
        if not model_config.is_file():
            raise PlanError(f"model config {model_config} does not exist: pass model_config:=")
        training = _bool("training", a["training"])
        if a["backgrounds"].strip():
            backgrounds = _path(a["backgrounds"])
            if not backgrounds.is_dir():
                raise PlanError(f"backgrounds {backgrounds} is not a folder")
        else:
            synth = (yaml.safe_load(model_config.read_text()) or {}).get("synth") or {}
            backgrounds = Path(str(synth.get("backgrounds") or "")).expanduser()
            if not str(synth.get("backgrounds") or "") or not backgrounds.is_dir():
                if training:
                    notes.append(f"no backgrounds folder ({synth.get('backgrounds') or 'none set'} is missing): "
                                 "frames are labelled and banked but not trained on; pass backgrounds:=<image dir>")
                backgrounds = None
        max_rate = _number("max_rate", a["max_rate"], float)
        if max_rate < 0:
            raise PlanError(f"max_rate:= must be >= 0 (got {max_rate})")
        params = {"config": str(config), "model_config": str(model_config), "training": training,
                  "max_rate_hz": max_rate if source == "pi" else 0.0, "save_frames": source == "pi",
                  "auto_deploy": _bool("auto_deploy", a["auto_deploy"]), "base_detector": str(base),
                  "resume": resume,
                  "backgrounds": str(backgrounds) if backgrounds else "", "output_dir": a["output_dir"].strip()}
        if convchart is not None:
            params["convchart_root"] = str(convchart)
        nodes.append(NodeSpec("ssl", params, True, stop_timeout_s=30.0))
        notes.append(f"task: live fine-tuning of {base.name} with {model_config.name}"
                     + (f", resuming {resume}" if resume else "")
                     + f"; stats on {shown.rstrip('/')}/ssl/stats, dashboard on {shown.rstrip('/')}/ssl/dashboard")

    if any(n.executable == "telemetry" for n in nodes) and not _has_package("dock_msgs", env):
        notes.append("WARNING: dock_msgs is not on AMENT_PREFIX_PATH, so the dashboard has no EKF, stage or MPC: "
                     "build it (colcon build --packages-select dock_msgs in controller_ws) and source "
                     "controller_ws/install/setup.bash")
    windowed = [n.executable for n in nodes if n.executable == "calibrate"
                or (n.executable in ("pose_view", "telemetry") and not n.parameters["headless"])]
    if windowed and not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")):
        raise PlanError(f"{windowed[0]} opens a window but DISPLAY is not set (for the container, run `make xhost` "
                        "on the host)" + {"pose_view": "; or record headless: record:=out.mp4 headless:=true",
                                          "telemetry": "; or headless: snapshot:=out.png headless:=true"}
                        .get(windowed[0] if task != "infer" or windowed[0] != "telemetry" else "", ""))
    if source == "video":
        consumers = sum(1 for n in nodes if n.executable in ("pose_view", "inference", "ssl"))
        nodes.insert(0, NodeSpec("replay", {"source": str(_path(video)), "rate": rate, "loop": loop,
                                            "wait_for_subscribers": consumers, "allow_root_namespace": not ns}))
    elif source == "fake":
        nodes.insert(0, NodeSpec("fake_camera", {"config": str(config)}))
    return Plan(task, source, ns, tuple(nodes), tuple(notes))
