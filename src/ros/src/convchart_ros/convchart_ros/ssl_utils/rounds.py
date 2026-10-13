"""Fine-tuning rounds for the live SSL node (ROS-free).

    rr = RoundRunner(RoundConfig(...), evaluate, on_start=f, on_end=g)   # evaluate(onnx, heldout_dirs) -> metrics
    rr.start(train_dirs, heldout_dirs)            # False while a round is running
    rr.status()                                   # phase, round, progress, finished rounds
    rr.stop()                                     # end the running round (its subprocess is terminated)

Each round runs Conv-ChArT's tools as subprocesses with RoundConfig.python: import the base ONNX once
(video_bootstrap.py import-onnx), score it on synthetic validation once (synth-val), write a finetune config from all
the banked train chunks (finetune-config), train from the base weights (train_detector.py), export (export-onnx), score
the candidate on synthetic validation and on the held-out chunks. The candidate is accepted when its held-out recall
beats the current model's by min_recall_gain without losing ID precision, and synthetic validation does not regress.
on_start / on_end run before and after each round (the node frees and rebuilds its teacher there on a small GPU).

Every banked frame is used: a round's length is set by its bank (round_steps), so that each real frame is drawn at
least `passes` times while synthetic samples pad each batch to real_frac. The finetune config asks Conv-ChArT's
MixedStream for real.sampling "epoch", which makes those passes exact rather than an average.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np


@dataclass
class RoundConfig:
    convchart_root: str
    python: str
    base_onnx: str
    model_config: str
    workdir: str
    backgrounds: str
    passes: float = 8.0
    min_steps: int = 0
    batch: int = 6
    accum: int = 2
    lr: float = 1e-4
    real_frac: float = 0.6
    sampling: str = "epoch"
    darken_synthetic: float = 0.0
    workers: int = 8
    val_size: int = 300
    val_subset: int = 150
    min_recall_gain: float = 0.002
    max_precision_drop: float = 0.005
    max_synth_m04_drop: float = 0.003
    max_synth_m01_rise: float = 0.01


def count_records(dirs) -> int:
    return sum(len(json.loads((Path(d) / "records.json").read_text())["records"]) for d in dirs)


def set_sampling(finetune_yaml: Path, sampling: str) -> None:
    """Set real.sampling in a finetune config written by video_bootstrap.py finetune-config."""
    import yaml
    cfg = yaml.safe_load(Path(finetune_yaml).read_text()) or {}
    cfg.setdefault("real", {})["sampling"] = sampling
    Path(finetune_yaml).write_text(yaml.safe_dump(cfg, sort_keys=False))


def round_steps(train_frames: int, c: RoundConfig) -> int:
    """Steps for a round on train_frames banked frames: each frame at least c.passes times, padded
    to c.real_frac.
    """
    if train_frames <= 0 or c.real_frac <= 0:
        return max(1, c.min_steps)
    w = max(1, c.workers)
    share = math.ceil(train_frames / w) if train_frames >= w else train_frames
    per_step = c.real_frac * c.batch * c.accum
    steps = math.ceil(c.passes * w * share / per_step - 1e-9) + math.ceil(w / c.accum)
    return max(c.min_steps, steps)


SCALE_GROUPS = (("far <16 px", 0.0, 16.0), ("16-32 px", 16.0, 32.0), ("32-64 px", 32.0, 64.0),
                (">=64 px", 64.0, math.inf))


def evaluate_onnx(pipeline, heldout_dirs, anchor_fn=None, groups=((0.0, 20.0), (20.0, 60.0), (60.0, 256.0))) -> dict:
    """Score a pipeline (anything with run_inference(image)) against held-out pseudo-labels: an
    identified corner is a hit when it lands within 2 px of a positive label with the same
    index.
    """
    rows = []
    for d in heldout_dirs:
        recs = json.loads((Path(d) / "records.json").read_text())["records"]
        imgs = np.load(Path(d) / "images.npy", mmap_mode="r")
        for img, rec in zip(imgs, recs):
            img = np.ascontiguousarray(img)
            res = pipeline.run_inference(img)
            dets = {}
            for c in res["corners"]:
                if c["index"] is not None:
                    dets.setdefault(c["index"], []).append((c["x"], c["y"]))
            pos = [(c["index"], c["x"], c["y"]) for c in rec["corners"] if c["visible"] is True]
            hits = sum(1 for k, x, y in pos if any(np.hypot(u - x, v - y) <= 2.0 for u, v in dets.get(k, [])))
            ids = sum(len(v) for v in dets.values())
            good = sum(1 for k, v in dets.items() for u, w in v
                       if any(kk == k and np.hypot(u - x, w - y) <= 2.0 for kk, x, y in pos))
            fit = anchor_fn(res) if anchor_fn else None
            pose = res.get("rvec") is not None
            rows.append((rec["board_level"], len(pos), hits, ids, good, fit, rec.get("s_px"), pose,
                         pose and bool(res.get("ambiguous"))))
    out = {}
    names = ["dark", "dim", "bright"]

    def within(lo, hi, i):
        return lambda r: r[i] is not None and lo <= r[i] < hi
    sels = ([("all", lambda r: True)] + [(n, within(lo, hi, 0)) for n, (lo, hi) in zip(names, groups)]
            + [(n, within(lo, hi, 6)) for n, lo, hi in SCALE_GROUPS])
    for name, sel in sels:
        rs = [r for r in rows if sel(r)]
        if not rs:
            continue
        ref, hit, ids, good = (sum(r[i] for r in rs) for i in (1, 2, 3, 4))
        fits = [r[5] for r in rs if r[5] is not None]
        poses = sum(r[7] for r in rs)
        out[name] = {"frames": len(rs), "recall": round(hit / ref, 4) if ref else None,
                     "id_precision": round(good / ids, 4) if ids else None,
                     "fit_rate": round(float(np.mean(fits)), 4) if fits else None,
                     "pose_rate": round(poses / len(rs), 4),
                     "ambiguous_rate": round(sum(r[8] for r in rs) / poses, 4) if poses else None}
    return out


class RoundRunner:

    def __init__(self, cfg: RoundConfig, evaluate, log=print, on_start=None, on_end=None):
        self.cfg, self.evaluate, self.log = cfg, evaluate, log
        self.on_start, self.on_end = on_start, on_end
        self.last_failure = None
        self.root = Path(cfg.convchart_root)
        self.work = Path(cfg.workdir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.vb = str(self.root / "tools" / "video_bootstrap.py")
        self.history: list = []
        self.best_onnx = str(cfg.base_onnx)
        self.round = 0
        self.steps = 0
        self.phase = "idle"
        self.progress = 0.0
        self.run_name = None
        self.started = None
        self.thread = None
        self.proc = None
        self.stopping = threading.Event()
        self.lock = threading.Lock()

    def restore(self, history: list, best_onnx: str | None = None) -> None:
        """Carry on a resumed session: its finished rounds, round numbering and the model to beat."""
        with self.lock:
            self.history = list(history)
            self.round = max((r.get("round", 0) for r in history), default=0)
            if best_onnx:
                self.best_onnx = str(best_onnx)

    @property
    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, train_dirs, heldout_dirs) -> bool:
        if self.busy or self.stopping.is_set():
            return False
        self.round += 1
        self.started = time.time()
        self.thread = threading.Thread(target=self._run, args=(self.round, list(train_dirs), list(heldout_dirs)),
                                       daemon=True, name=f"ssl-round-{self.round}")
        self.thread.start()
        return True

    def stop(self, timeout: float = 15.0) -> None:
        self.stopping.set()
        proc = self.proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout)
            except subprocess.TimeoutExpired:
                proc.kill()

    def status(self) -> dict:
        with self.lock:
            return {"phase": self.phase, "round": self.round, "progress": round(self.progress, 3),
                    "run": self.run_name, "best_onnx": self.best_onnx, "started": self.started,
                    "history": list(self.history)}

    def metrics_path(self, run_name: str) -> Path:
        return self.root / "runs" / run_name / "metrics.jsonl"

    def _run_name(self, k: int) -> str:
        base = f"ssl_{self.work.parent.name}_r{k}"
        name, i = base, 1
        while (self.root / "runs" / name).exists():
            i += 1
            name = f"{base}_{i}"
        return name

    def _set(self, phase: str, progress: float = 0.0) -> None:
        with self.lock:
            self.phase, self.progress = phase, progress
        self.log(f"[ssl round {self.round}] {phase}")

    def _spawn(self, args, f) -> subprocess.Popen:
        if self.stopping.is_set():
            raise RuntimeError("round stopped")
        self.proc = subprocess.Popen([self.cfg.python, *map(str, args)], cwd=self.root,
                                     env={**os.environ, "PYTHONPATH": ""}, stdout=f, stderr=subprocess.STDOUT)
        return self.proc

    def _cmd(self, args, log_path: Path) -> None:
        with open(log_path, "a") as f:
            f.write(f"$ {' '.join(map(str, args))}\n")
            f.flush()
            proc = self._spawn(args, f)
            if proc.wait() != 0:
                raise subprocess.CalledProcessError(proc.returncode, [self.cfg.python, *map(str, args)])

    def _train(self, finetune_yaml: Path, run_name: str, log_path: Path) -> None:
        metrics = self.metrics_path(run_name)
        with open(log_path, "a") as f:
            proc = self._spawn([self.root / "tools" / "train_detector.py", "--config", finetune_yaml,
                                "--name", run_name], f)
            while proc.poll() is None:
                time.sleep(2.0)
                if metrics.exists():
                    steps = [json.loads(line).get("step", 0) for line in metrics.read_text().splitlines()[-5:]
                             if line.strip()]
                    if steps:
                        with self.lock:
                            self.progress = min(1.0, max(steps) / max(1, self.steps))
        if proc.returncode != 0:
            raise RuntimeError(f"train_detector.py exited {proc.returncode}, see {log_path}")

    def _synth_val(self, detector: Path, out: Path, log_path: Path) -> dict:
        if not out.exists():
            self._cmd([self.vb, "synth-val", "--detector", detector, "--config", self.cfg.model_config,
                       "--backgrounds", self.cfg.backgrounds, "--val-size", self.cfg.val_size, "--out", out], log_path)
        return json.loads(out.read_text())

    def _run(self, k: int, train_dirs: list, heldout_dirs: list) -> None:
        c = self.cfg
        rd = self.work / f"round{k}"
        rd.mkdir(parents=True, exist_ok=True)
        log_path = rd / "round.log"
        t0 = time.time()
        n_train = count_records(train_dirs)
        steps = round_steps(n_train, c)
        with self.lock:
            self.steps = steps
        result = {"round": k, "t_start": t0, "train_frames": n_train, "heldout_frames": count_records(heldout_dirs),
                  "steps": steps, "passes": round(steps * c.batch * c.accum * c.real_frac / max(1, n_train), 2),
                  "train_dirs": train_dirs, "heldout_dirs": heldout_dirs, "config": asdict(c)}
        try:
            if self.on_start:
                self.on_start()
            base_pt, base_meta = self.work / "base.pt", self.work / "base.json"
            if base_pt.exists() and base_meta.exists() and json.loads(base_meta.read_text()).get("onnx") != c.base_onnx:
                base_pt.unlink()
                (self.work / "base_synth_val.json").unlink(missing_ok=True)
            if not base_pt.exists():
                self._set("import base")
                self._cmd([self.vb, "import-onnx", "--onnx", c.base_onnx, "--config", c.model_config,
                           "--out", base_pt], log_path)
            if not base_meta.exists() or json.loads(base_meta.read_text()).get("onnx") != c.base_onnx:
                base_meta.write_text(json.dumps({"onnx": c.base_onnx}))
            self._set("base synthetic val")
            synth_base = self._synth_val(base_pt, self.work / "base_synth_val.json", log_path)
            self._set("config")
            finetune_yaml = rd / "finetune.yaml"
            self._cmd([self.vb, "finetune-config", "--config", c.model_config, "--from", base_pt,
                       "--records", *train_dirs, "--out", finetune_yaml, "--steps", steps, "--batch", c.batch,
                       "--accum", c.accum, "--lr", c.lr, "--real-frac", c.real_frac,
                       "--darken-synthetic", c.darken_synthetic, "--backgrounds", c.backgrounds,
                       "--workers", c.workers, "--val-size", c.val_size, "--val-subset", c.val_subset], log_path)
            set_sampling(finetune_yaml, c.sampling)
            run_name = self._run_name(k)
            with self.lock:
                self.run_name = run_name
            result["run"] = run_name
            result["metrics"] = str(self.metrics_path(run_name))
            self._set("train")
            self._train(finetune_yaml, run_name, log_path)
            ckpt = self.root / "runs" / run_name / f"ckpt_{steps:07d}.pt"
            self._set("export")
            onnx_path = rd / "detector.onnx"
            self._cmd([self.vb, "export-onnx", "--ckpt", ckpt, "--out", onnx_path], log_path)
            self._set("candidate synthetic val")
            synth_cand = self._synth_val(ckpt, rd / "synth_val.json", log_path)
            self._set("held-out eval")
            ref = self.evaluate(self.best_onnx, heldout_dirs)
            cand = self.evaluate(str(onnx_path), heldout_dirs)
            result.update(reference_onnx=self.best_onnx, candidate_onnx=str(onnx_path), reference=ref,
                          candidate=cand, synth_base=synth_base, synth_candidate=synth_cand)
            result["gate"] = self._gate(ref, cand, synth_base, synth_cand)
            result["accepted"] = all(result["gate"].values())
            if result["accepted"]:
                with self.lock:
                    self.best_onnx = str(onnx_path)
        except Exception as e:
            result.update(accepted=False, error=f"{type(e).__name__}: {e}")
            self.last_failure = time.time()
        finally:
            if self.on_end:
                try:
                    self.on_end()
                except Exception as e:
                    result["on_end_error"] = f"{type(e).__name__}: {e}"
        result["t_end"] = time.time()
        result["seconds"] = round(result["t_end"] - t0, 1)
        (rd / "result.json").write_text(json.dumps(result, indent=1, default=str))
        with self.lock:
            self.history.append(result)
            self.phase, self.progress = "idle", 0.0
        self.log(f"[ssl round {k}] done: accepted={result.get('accepted')} {result.get('error', '')}")

    def _gate(self, ref: dict, cand: dict, sb: dict, sc: dict) -> dict:
        c = self.cfg
        r0, r1 = ref.get("all", {}), cand.get("all", {})
        m04 = lambda s: (s.get("m04") or {}).get("accuracy") or 0.0
        return {"heldout_recall": (r1.get("recall") or 0.0) >= (r0.get("recall") or 0.0) + c.min_recall_gain,
                "heldout_id_precision": (r1.get("id_precision") or 0.0) >= (r0.get("id_precision") or 0.0)
                - c.max_precision_drop,
                "synth_m04": m04(sc) >= m04(sb) - c.max_synth_m04_drop,
                "synth_m01": (sc["m01"]["median"] or 0.0) <= (sb["m01"]["median"] or 0.0) + c.max_synth_m01_rise}
