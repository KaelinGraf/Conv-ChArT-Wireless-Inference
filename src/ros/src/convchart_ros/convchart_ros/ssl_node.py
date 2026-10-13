"""Live self-supervised fine-tuning from the Pi's camera stream.

    ros2 run convchart_ros ssl --ros-args -p config:=<cfg.yaml> -p model_config:=<Conv-ChArT model config.yaml> \
        -p backgrounds:=<image dir> [-p training:=false] [-p auto_deploy:=true] [-p resume:=latest]

Subscribes to 'image' (the Pi's CompressedImage stream). A teacher (src/inference.py's pipeline with a detector ONNX)
runs on each frame; frames are labelled from the board's own geometry as anchors arrive and banked in chunks
(ssl_utils.harvester). Every labelled frame is kept: by default there is no diversity cap and no near-duplicate filter
(cap_per_bucket 0, min_motion_px 0). Once min_train_frames are banked, and again each time the bank grows by
retrain_growth, a background round fine-tunes base_detector (default the release weights,
<repo>/checkpoints/detector_882k.onnx) on every banked train frame, each drawn at least `passes` times and padded
with synthetic boards to real_frac against forgetting (ssl_utils.rounds), and accepts the candidate only if it beats
the current model on the held-out frames without regressing on synthetic validation.

The teacher starts as the config's MODEL.detector (the deployed model), or on resume as the session's best model.
With auto_deploy it switches to each accepted model, and with relabel (the default) a new bank generation starts:
harvest/gen<g>/ holds what that teacher labels, the live frames as they come (chunks L) and, in the background, every
frame already stored in stream/ (chunks R, at full rate), so the better model grows the dataset from everything
recorded. Rounds wait for a generation's pass over the stored frames, then train on its whole bank; older
generations stay on disk. Frames are split train / held-out by split_block_s blocks of their stamps, so a frame keeps
its split whichever teacher labels it. The deployed inference node is never touched.

resume:=<session folder> (or latest) carries on in an existing session: its stored frames, its finished rounds (their
numbering, history and the model to beat), its analytics clock and logs. The session's best model becomes the teacher
and a new generation relabels everything stored (ssl_utils.resume).

With pause_teacher_during_rounds (default, needed on an 8 GB GPU) the teacher frees the GPU while a round runs and
live frames arriving then are counted as paused; a failed round is retried after retry_after_s without waiting for
the bank to grow. A gap above max_dt_s between frame stamps ends the tracked stretch there. max_rate_hz > 0 skips live
frames arriving faster than that, before the teacher; the next generation's pass over the stored frames labels them.

With save_frames (the default) every frame received is written to <session>/stream/ as it came (the camera's PNG, not
re-encoded), named <sec>_<nanosec>.png by its stamp, before any rate limit or pause. A folder of them replays in order
(task:=finetune video:=<session>/stream). At 15 Hz that is about 8 GB an hour.

Publishes ssl/stats (std_msgs/String, JSON, every stats_period s: the live stream, the generation's bank and its
relabel progress, the rounds) and ssl/dashboard (PNG CompressedImage, every dashboard_period s; ssl_view shows both).
Writes <output_dir>/<session>/: frames.jsonl, segments.jsonl, relabel.jsonl, rounds.jsonl, dashboard.png, harvest/,
stream/, rounds/, and on shutdown report/*.pdf and summary.json; python3 -m convchart_ros.ssl_utils.analytics
<session> re-renders the figures from those logs. Labels use the config's CAMERA calibration (K, dist) when it is set.
model_config is the Conv-ChArT config the detector was trained with (architecture and synthetic stream); backgrounds
is any image directory for the synthetic replay. Conv-ChArT is found at <repo>/../Conv-ChArT unless convchart_root
says otherwise.
"""
from __future__ import annotations

from collections import Counter, deque
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time

from convchart_qos.qos import IMAGE_DEADLINE_CEILING_S, IMAGE_LEASE_CEILING_S
import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSLivelinessPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import yaml

SSL_IMAGE_QOS = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10,
                           reliability=QoSReliabilityPolicy.BEST_EFFORT, durability=QoSDurabilityPolicy.VOLATILE,
                           deadline=Duration(seconds=IMAGE_DEADLINE_CEILING_S),
                           liveliness=QoSLivelinessPolicy.AUTOMATIC,
                           liveliness_lease_duration=Duration(seconds=IMAGE_LEASE_CEILING_S))
OUT_QOS = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=2, reliability=QoSReliabilityPolicy.RELIABLE,
                     durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

DEFAULTS = {
    "config": "", "model_config": "", "convchart_root": "", "output_dir": "", "backgrounds": "",
    "base_detector": "", "resume": "",
    "training": True, "auto_deploy": False, "train_python": "", "pause_teacher_during_rounds": True,
    "retry_after_s": 300.0,
    "min_train_frames": 400, "min_heldout_frames": 80, "retrain_growth": 0.5,
    "passes": 8.0, "min_steps": 0, "batch": 6, "accum": 2, "lr": 1e-4, "real_frac": 0.6, "darken_synthetic": 0.0,
    "workers": 8, "val_size": 300,
    "max_gap": 150, "max_dt_s": 0.5, "min_motion_px": 0.0, "cap_per_bucket": 0, "chunk_size": 120, "heldout_every": 5,
    "split_block_s": 30.0, "relabel": True,
    "queue_size": 30, "stats_period": 1.0, "dashboard_period": 10.0, "max_rate_hz": 0.0, "save_frames": True,
}
RELEASE_DETECTOR = Path("checkpoints") / "detector_882k.onnx"


def stamp_of(name: str) -> float:
    """The stamp of a frame saved as <sec>_<nanosec>.<ext>."""
    sec, nsec = Path(name).stem.split("_")
    return int(sec) + int(nsec) * 1e-9


def stored_frames(stream_dir: Path, upto: float = float("inf")) -> list:
    """The names of the frames saved in stream_dir stamped at or before upto, oldest first."""
    if stream_dir is None or not stream_dir.is_dir():
        return []
    with os.scandir(stream_dir) as it:
        names = [e.name for e in it if e.name[:1].isdigit() and "_" in e.name]
    return sorted(n for n in names if stamp_of(n) <= upto)


class Generation:
    """
    One bank generation: the teacher that labels it, the harvesters of its live frames (stamped after cut) and of the
    stored ones (at or before cut), and the progress of the pass over the stored frames.
    """

    def __init__(self, g: int, teacher_onnx: str, cut: float, live, relabeler):
        self.g, self.teacher_onnx, self.cut = g, teacher_onnx, cut
        self.live, self.relabeler = live, relabeler
        self.cancel = threading.Event()
        self.queue: queue.Queue = queue.Queue(maxsize=64)
        self.done = self.total = 0
        self.finished = relabeler is None
        self.t_start = time.time()
        self.t_finish = self.t_start if self.finished else None
        self.threads: list = []

    def harvesters(self) -> list:
        return [h for h in (self.live, self.relabeler) if h is not None]

    def progress(self) -> dict:
        t = (self.t_finish or time.time()) - self.t_start
        return {"gen": self.g, "teacher": self.teacher_onnx, "cut": self.cut,
                "relabel": {"done": self.done, "total": self.total, "finished": self.finished,
                            "fps": round(self.done / t, 1) if t > 0 else 0.0}}


class SSLNode(Node):

    def __init__(self, **overrides):
        super().__init__("convchart_ssl")
        p = {k: self.declare_parameter(k, overrides.get(k, v)).value for k, v in DEFAULTS.items()}
        self.p = p
        if not p["config"] or not p["model_config"]:
            raise ValueError("set the config (cfg.yaml) and model_config (Conv-ChArT config) parameters")
        self.cfg_path = Path(p["config"]).resolve()
        self.repo = self.cfg_path.parent.parent
        self.convchart = Path(p["convchart_root"] or self.repo.parent / "Conv-ChArT").resolve()
        for d in (self.repo / "src", self.convchart):
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
        import inference
        from .ssl_utils.analytics import Analytics
        from .ssl_utils.harvester import HarvestParams, OnlineHarvester
        from .ssl_utils.resume import latest_session, resume_state
        from .ssl_utils.rounds import evaluate_onnx, RoundConfig, RoundRunner
        self._inference, self._evaluate_onnx = inference, evaluate_onnx
        self._Harvester, self._HarvestParams = OnlineHarvester, HarvestParams
        self.base_cfg = yaml.safe_load(self.cfg_path.read_text())
        self.model_cfg = yaml.safe_load(Path(p["model_config"]).read_text())
        out_dir = Path(p["output_dir"] or self.repo / "ssl_output")
        state = None
        if p["resume"]:
            self.session = (latest_session(out_dir) if p["resume"] == "latest"
                            else Path(p["resume"]).expanduser().resolve())
            if self.session is None or not (self.session / "session.json").exists():
                raise ValueError(f"resume: no ssl session at {p['resume']!r} (under {out_dir} for latest)")
            state = resume_state(self.session)
        else:
            self.session = out_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
            self.session.mkdir(parents=True, exist_ok=True)
        self.resumed = state is not None
        self.base_onnx = str(Path(p["base_detector"]).expanduser().resolve() if p["base_detector"]
                             else self.repo / RELEASE_DETECTOR)
        if not Path(self.base_onnx).is_file():
            raise ValueError(f"base_detector {self.base_onnx} does not exist")
        self.refiner_onnx = self._model_path("refiner")
        teacher_onnx = (state["teacher"] if state and state["teacher"] else self._model_path("detector"))
        self.teacher = self._pipeline_for(teacher_onnx)
        self.teacher_onnx = teacher_onnx
        self.stream_dir = self.session / "stream" if p["save_frames"] else None
        if self.stream_dir is not None:
            self.stream_dir.mkdir(exist_ok=True)
        self.analytics = Analytics(self.session, resume=self.resumed)
        self.rounds = None
        if p["training"] and p["backgrounds"]:
            self.rounds = RoundRunner(RoundConfig(
                convchart_root=str(self.convchart), python=p["train_python"] or sys.executable,
                base_onnx=self.base_onnx, model_config=str(Path(p["model_config"]).resolve()),
                workdir=str(self.session / "rounds"), backgrounds=str(p["backgrounds"]), passes=float(p["passes"]),
                min_steps=int(p["min_steps"]), batch=p["batch"], accum=p["accum"], lr=p["lr"],
                real_frac=float(p["real_frac"]), darken_synthetic=p["darken_synthetic"], workers=p["workers"],
                val_size=p["val_size"]),
                self._evaluate, log=self.get_logger().info, on_start=self._round_start, on_end=self._round_end)
            self.rounds.restore(state["history"] if state else [], (state or {}).get("best_onnx") or teacher_onnx)
        elif p["training"]:
            self.get_logger().warn("training disabled: no backgrounds directory for the synthetic replay stream")
        self._frames: deque = deque(maxlen=p["queue_size"])
        self._todo: deque = deque(maxlen=600)
        self._cv, self._hcv = threading.Condition(), threading.Condition()
        self._stop = threading.Event()
        self._teacher_lock = threading.Lock()
        self._gen_lock = threading.RLock()
        self._round_lock = threading.Lock()
        self._drops = {"stream": 0, "harvest": 0, "paused": 0, "rate": 0, "save": 0}
        self._last_accept = None
        self._last_stamp = None
        self._paused = False
        self._seen_rounds = len(state["history"]) if state else 0
        self._to_save: deque = deque(maxlen=300)
        self._scv = threading.Condition()
        self.saved = 0
        self.gen = None
        stored = stored_frames(self.stream_dir)
        g = state["next_gen"] if state else 1
        self._start_generation(g, teacher_onnx, stamp_of(stored[-1]) if stored else 0.0)
        info = {"params": p, "config": str(self.cfg_path), "convchart_root": str(self.convchart),
                "detector": self.base_onnx, "base_detector": self.base_onnx, "teacher": teacher_onnx,
                "refiner": self.refiner_onnx, "generation": g,
                "lens": "calibrated" if self.gen.live.calibrated else "uncalibrated"}
        if state:
            meta = state["meta"]
            meta.setdefault("resumes", []).append({"at": datetime.now().isoformat(timespec="seconds"), **info})
            (self.session / "session.json").write_text(json.dumps(meta, indent=1))
        else:
            (self.session / "session.json").write_text(json.dumps(info, indent=1))
        self._threads = [threading.Thread(target=self._teacher_loop, name="ssl-teacher", daemon=True),
                         threading.Thread(target=self._harvest_loop, name="ssl-harvest", daemon=True),
                         threading.Thread(target=self._save_loop, name="ssl-save", daemon=True)]
        for t in self._threads:
            t.start()
        self._img_group, self._pub_group = MutuallyExclusiveCallbackGroup(), MutuallyExclusiveCallbackGroup()
        self.create_subscription(CompressedImage, "image", self._on_image, SSL_IMAGE_QOS,
                                 callback_group=self._img_group)
        self._stats_pub = self.create_publisher(String, "ssl/stats", OUT_QOS)
        self._dash_pub = self.create_publisher(CompressedImage, "ssl/dashboard", OUT_QOS)
        self.create_timer(p["stats_period"], self._publish_stats, callback_group=self._pub_group)
        self.create_timer(p["dashboard_period"], self._publish_dashboard, callback_group=self._pub_group)
        self.get_logger().info(
            f"ssl session {self.session}{' (resumed)' if state else ''}; teacher {teacher_onnx}; base "
            f"{self.base_onnx}; training {'on' if self.rounds else 'off'}; lens "
            f"{'calibrated' if self.gen.live.calibrated else 'uncalibrated'}; generation {g}, "
            f"{len(stored)} stored frames to relabel" + ("" if p["relabel"] else " (relabel off)"))

    @property
    def harvester(self):
        """The current generation's live harvester (its lens, lattice and template serve the evaluations)."""
        return self.gen.live

    def _model_path(self, key: str) -> str:
        path = Path(self.base_cfg["MODEL"][key])
        return str(path if path.is_absolute() else (self.repo / path).resolve())

    def _pipeline_for(self, detector_onnx: str):
        cfg = json.loads(json.dumps(self.base_cfg))
        cfg["MODEL"]["detector"] = str(Path(detector_onnx).resolve())
        cfg["MODEL"]["refiner"] = self.refiner_onnx
        tag = hashlib.sha1(cfg["MODEL"]["detector"].encode()).hexdigest()[:10]
        path = self.session / "cfg" / f"pipeline_{tag}.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(yaml.safe_dump(cfg))
        return self._inference.inference_pipeline(str(path))

    def _evaluate(self, detector_onnx: str, heldout_dirs) -> dict:
        h = self.harvester
        pipe = self._pipeline_for(detector_onnx)
        try:
            return self._evaluate_onnx(pipe, heldout_dirs, anchor_fn=lambda res: h.V.anchor_from_detection(
                res, 1.0, h.n, h.lens, **h.hp) is not None)
        finally:
            del pipe

    def _harvester_for(self, g: int, prefix: str):
        p = self.p
        params = self._HarvestParams(max_gap=p["max_gap"], max_dt_s=p["max_dt_s"], min_motion_px=p["min_motion_px"],
                                     cap_per_bucket=p["cap_per_bucket"], chunk_size=p["chunk_size"],
                                     heldout_every=p["heldout_every"], split_block_s=p["split_block_s"])
        return self._Harvester(self.model_cfg, self.session, params, camera=self.base_cfg.get("CAMERA"),
                               bank=f"gen{g}", prefix=prefix)

    def _start_generation(self, g: int, teacher_onnx: str, cut: float) -> None:
        """
        Start bank generation g, labelled by teacher_onnx: live frames stamped after cut into its L chunks and, with
        relabel, every stored frame stamped at or before cut into its R chunks, by a background pass.
        """
        relabel = self.p["relabel"] and bool(stored_frames(self.stream_dir, cut))
        gen = Generation(g, teacher_onnx, cut, self._harvester_for(g, "L"),
                         self._harvester_for(g, "R") if relabel else None)
        with self._gen_lock:
            old, self.gen = self.gen, gen
            if old is not None:
                old.cancel.set()
                seg = old.live.flush()
                if seg:
                    self.analytics.segment(seg, old.live.seq - 1, "flush")
        if relabel:
            gen.threads = [threading.Thread(target=self._relabel_teacher, args=(gen,), name=f"ssl-relabel-{g}",
                                            daemon=True),
                           threading.Thread(target=self._relabel_harvest, args=(gen,), name=f"ssl-rebank-{g}",
                                            daemon=True)]
            for t in gen.threads:
                t.start()
        self.get_logger().info(f"generation {g}: teacher {teacher_onnx}; live frames after {cut:.3f}"
                               + (", stored frames up to it relabelled in the background" if relabel else ""))

    def _teach(self, gen: Generation, img: np.ndarray):
        """The generation's teacher on img, waiting while it is paused for a round; None once the pass must stop."""
        while not (self._stop.is_set() or gen.cancel.is_set()):
            with self._teacher_lock:
                if self.teacher is not None and self.teacher_onnx == gen.teacher_onnx:
                    return self.teacher.run_inference(img)
            time.sleep(0.05)
        return None

    def _relabel_teacher(self, gen: Generation) -> None:
        """Run the generation's teacher over the stored frames up to its cut, oldest first, for _relabel_harvest."""
        try:
            deadline = time.time() + 5.0
            while self._to_save and time.time() < deadline and not self._stop.is_set():
                time.sleep(0.05)
            names = stored_frames(self.stream_dir, gen.cut)
            gen.total = len(names)
            for name in names:
                while self._frames and not (self._stop.is_set() or gen.cancel.is_set()):
                    time.sleep(0.002)
                if self._stop.is_set() or gen.cancel.is_set():
                    return
                img = cv2.imread(str(self.stream_dir / name), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    gen.done += 1
                    continue
                try:
                    res = self._teach(gen, img)
                except Exception as e:
                    self.get_logger().error(f"relabel teacher failed on {name}: {type(e).__name__}: {e}",
                                            throttle_duration_sec=10.0)
                    gen.done += 1
                    continue
                if res is None:
                    return
                while not (self._stop.is_set() or gen.cancel.is_set()):
                    try:
                        gen.queue.put((stamp_of(name), img, res), timeout=0.2)
                        break
                    except queue.Full:
                        pass
        finally:
            gen.queue.put(None)

    def _relabel_harvest(self, gen: Generation) -> None:
        """Bank what _relabel_teacher labels into the generation's R chunks; when the pass ends, close it."""
        h = gen.relabeler
        while True:
            item = gen.queue.get()
            if item is None:
                break
            stamp, img, res = item
            try:
                ev = h.add(stamp, img, res)
                for kind in ("break", "segment"):
                    if ev.get(kind):
                        self.analytics.relabel(ev[kind], gen.g, ev["seq"])
            except Exception as e:
                self.get_logger().error(f"relabel harvest failed: {type(e).__name__}: {e}", throttle_duration_sec=10.0)
            gen.done += 1
        seg = h.flush()
        if seg:
            self.analytics.relabel(seg, gen.g, h.seq - 1)
        if gen.cancel.is_set() or self._stop.is_set():
            return
        gen.finished, gen.t_finish = True, time.time()
        snap = h.snapshot()
        self.get_logger().info(f"generation {gen.g}: {gen.done} stored frames relabelled, {snap['banked']['train']} "
                               f"train / {snap['banked']['heldout']} held-out banked from them")
        self._maybe_round()

    def _bank(self) -> dict:
        """The current generation's bank: its live and relabelled chunks together."""
        with self._gen_lock:
            snaps = [h.snapshot() for h in self.gen.harvesters()]
        totals: Counter = Counter()
        for s in snaps:
            totals.update(s["totals"])
        return {"totals": dict(totals),
                "chunks": {k: sorted(sum((s["chunks"][k] for s in snaps), [])) for k in ("train", "heldout")},
                "banked": {k: sum(s["banked"][k] for s in snaps) for k in ("train", "heldout")},
                "open_chunk": sum(s["open_chunk"] for s in snaps), "buffered": sum(s["buffered"] for s in snaps)}

    def _coverage(self) -> dict:
        out: Counter = Counter()
        with self._gen_lock:
            for h in self.gen.harvesters():
                out.update(h.coverage())
        return dict(out)

    def _on_image(self, msg: CompressedImage) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._last_stamp = stamp
        if self.stream_dir is not None:
            ext = ".png" if "png" in msg.format.lower() else ".jpg" if "jp" in msg.format.lower() else ".bin"
            with self._scv:
                if len(self._to_save) == self._to_save.maxlen:
                    self._drops["save"] += 1
                self._to_save.append((f"{msg.header.stamp.sec:010d}_{msg.header.stamp.nanosec:09d}{ext}", msg.data))
                self._scv.notify()
        now = time.time()
        rate = self.p["max_rate_hz"]
        if rate > 0 and self._last_accept is not None and now - self._last_accept < 0.9 / rate:
            self._drops["rate"] += 1
            return
        self._last_accept = now
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return
        with self._cv:
            if len(self._frames) == self._frames.maxlen:
                self._drops["stream"] += 1
            self._frames.append((stamp, time.time(), img))
            self._cv.notify()

    def _teacher_loop(self) -> None:
        while not self._stop.is_set():
            with self._cv:
                while not self._frames and not self._stop.is_set():
                    self._cv.wait(0.2)
                if self._stop.is_set():
                    return
                stamp, arrival, img = self._frames.popleft()
            t0 = time.perf_counter()
            try:
                with self._teacher_lock:
                    if self.teacher is None:
                        self._drops["paused"] += 1
                        continue
                    res = self.teacher.run_inference(img)
            except Exception as e:
                self.get_logger().error(f"teacher failed: {type(e).__name__}: {e}")
                continue
            with self._hcv:
                if len(self._todo) == self._todo.maxlen:
                    self._drops["harvest"] += 1
                self._todo.append((stamp, arrival, img, res, (time.perf_counter() - t0) * 1000.0))
                self._hcv.notify()

    def _save_loop(self) -> None:
        """Write the received frames into stream/; at shutdown, write what is still queued, then return."""
        while True:
            with self._scv:
                while not self._to_save and not self._stop.is_set():
                    self._scv.wait(0.2)
                if not self._to_save:
                    return
                name, data = self._to_save.popleft()
            try:
                with open(self.stream_dir / name, "wb") as f:
                    f.write(data)
                self.saved += 1
            except OSError as e:
                self._drops["save"] += 1
                self.get_logger().error(f"saving {name} failed: {e}", throttle_duration_sec=10.0)

    def _harvest_loop(self) -> None:
        while True:
            with self._hcv:
                while not self._todo and not self._stop.is_set():
                    self._hcv.wait(0.2)
                if self._stop.is_set() or not self._todo:
                    self._drops["shutdown"] = len(self._todo)
                    return
                stamp, arrival, img, res, latency = self._todo.popleft()
            try:
                with self._gen_lock:
                    gen = self.gen
                    if stamp <= gen.cut:
                        continue
                    ev = gen.live.add(stamp, img, res)
                ev["latency_ms"], ev["arrival"], ev["gen"] = round(latency, 2), arrival, gen.g
                self.analytics.frame(ev)
                self._maybe_round()
            except Exception as e:
                self.get_logger().error(f"harvest failed: {type(e).__name__}: {e}")

    def _round_start(self) -> None:
        if not self.p["pause_teacher_during_rounds"]:
            return
        import gc
        with self._teacher_lock:
            self.teacher, self._paused = None, True
        gc.collect()
        self.get_logger().info("teacher paused: GPU handed to the round")

    def _round_end(self) -> None:
        if not self.p["pause_teacher_during_rounds"] or self._stop.is_set():
            return
        onnx = self.rounds.best_onnx if self.p["auto_deploy"] and self.rounds else self.gen.teacher_onnx
        try:
            new = self._pipeline_for(onnx)
        except Exception as e:
            self.get_logger().error(f"teacher on {onnx} failed ({type(e).__name__}: {e}); keeping "
                                    f"{self.gen.teacher_onnx}")
            onnx = self.gen.teacher_onnx
            new = self._pipeline_for(onnx)
        if onnx != self.gen.teacher_onnx and self.p["relabel"]:
            self._start_generation(self.gen.g + 1, onnx, self._last_stamp or 0.0)
        with self._teacher_lock:
            self.teacher, self.teacher_onnx, self._paused = new, onnx, False
        if onnx != self.gen.teacher_onnx:
            self.gen.teacher_onnx = onnx
        self.get_logger().info(f"teacher resumed on {onnx}")

    def _maybe_round(self) -> None:
        if self.rounds is None:
            return
        with self._round_lock:
            if self.rounds.busy or not self.gen.finished:
                return
            if self.rounds.last_failure and time.time() - self.rounds.last_failure < self.p["retry_after_s"]:
                return
            snap = self._bank()
            need = self.p["min_train_frames"]
            ran = [r for r in self.rounds.status()["history"] if "error" not in r]
            if ran:
                need = max(need, int(ran[-1]["train_frames"] * (1.0 + self.p["retrain_growth"])))
            if snap["banked"]["train"] >= need and snap["banked"]["heldout"] >= self.p["min_heldout_frames"]:
                if self.rounds.start(snap["chunks"]["train"], snap["chunks"]["heldout"]):
                    self.get_logger().info(f"round {self.rounds.round} started on generation {self.gen.g}: "
                                           f"{snap['banked']['train']} train / {snap['banked']['heldout']} held-out "
                                           "frames")

    def _collect_rounds(self) -> dict | None:
        if self.rounds is None:
            return None
        status = self.rounds.status()
        for r in status["history"][self._seen_rounds:]:
            self.analytics.round_done(r)
            if r.get("accepted") and self.p["auto_deploy"] and not self.p["pause_teacher_during_rounds"]:
                new = self._pipeline_for(r["candidate_onnx"])
                if self.p["relabel"]:
                    self._start_generation(self.gen.g + 1, r["candidate_onnx"], self._last_stamp or 0.0)
                with self._teacher_lock:
                    self.teacher, self.teacher_onnx = new, r["candidate_onnx"]
                self.gen.teacher_onnx = r["candidate_onnx"]
                self.get_logger().info(f"teacher switched to round {r['round']}: {r['candidate_onnx']}")
        self._seen_rounds = len(status["history"])
        return status

    def _metrics_paths(self, status: dict | None) -> dict:
        if not status:
            return {}
        paths = {f"r{r['round']}": r["metrics"] for r in status["history"] if r.get("metrics")}
        if status.get("run") and status.get("phase") not in ("idle", None):
            paths[f"r{status['round']}"] = str(self.rounds.metrics_path(status["run"]))
        return paths

    def _stats(self, status: dict | None) -> dict:
        snap = self._bank()
        short = None
        if status:
            short = {k: status[k] for k in ("phase", "round", "progress", "best_onnx")}
            short["accepted"] = [r["round"] for r in status["history"] if r.get("accepted")]
            short["steps"] = self.rounds.steps if status["phase"] not in ("idle", None) else None
        harvest = {k: snap[k] for k in ("totals", "banked", "open_chunk", "buffered")}
        return {**self.analytics.summary(), "harvest": harvest, "generation": self.gen.progress(),
                "drops": dict(self._drops), "teacher": self.teacher_onnx,
                "paused": self._paused, "rounds": short, "saved_frames": self.saved, "resumed": self.resumed,
                "session": str(self.session)}

    def _publish_stats(self) -> None:
        self._stats_pub.publish(String(data=json.dumps(self._stats(self._collect_rounds()))))

    def _publish_dashboard(self) -> None:
        status = self._collect_rounds()
        try:
            png = self.analytics.render(self._coverage(), status, self._metrics_paths(status))
        except Exception as e:
            self.get_logger().error(f"dashboard failed: {type(e).__name__}: {e}")
            return
        msg = CompressedImage(format="png", data=png)
        msg.header.stamp = self.get_clock().now().to_msg()
        self._dash_pub.publish(msg)

    def shutdown(self) -> None:
        self._stop.set()
        gen = self.gen
        gen.cancel.set()
        with self._cv:
            self._cv.notify_all()
        with self._hcv:
            self._hcv.notify_all()
        with self._scv:
            self._scv.notify_all()
        for t in self._threads + gen.threads:
            t.join(timeout=10.0)
        if self.rounds is not None:
            self.rounds.stop()
            if self.rounds.thread is not None:
                self.rounds.thread.join(timeout=60.0)
        seg = gen.live.flush()
        if seg:
            self.analytics.segment(seg, gen.live.seq - 1, "flush")
        status = self._collect_rounds()
        self.analytics.render(self._coverage(), status, self._metrics_paths(status), pdf=True)
        (self.session / "summary.json").write_text(json.dumps(
            {**self._stats(status), "coverage": self._coverage()}, indent=1, default=str))
        self.analytics.close()
        self.get_logger().info(f"ssl session written to {self.session}")


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SSLNode()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if node is not None:
            node.shutdown()
            node.destroy_node()
        rclpy.try_shutdown()
