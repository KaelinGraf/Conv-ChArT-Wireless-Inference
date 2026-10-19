"""Online pseudo-labelling of a live board stream (ROS-free; needs Conv-ChArT's dcc on sys.path).

    h = OnlineHarvester(model_cfg, session_dir, HarvestParams(), camera=cfg["CAMERA"], bank="gen1", prefix="L")
    event = h.add(stamp, image, result)   # every frame: uint8 mono image and its inference result dict
    h.break_stream()                      # frames stop being consecutive: label what is buffered, start afresh
    h.flush()                             # at shutdown: break_stream and write the open chunk

A frame whose identified corners fit the lattice is an anchor (dcc.video.anchor_from_detection). The frames between
two anchors are labelled when the second one arrives (dcc.video.harvest_labels); an anchor right after the previous
one has nothing to track between them and is verified alone, so each anchor's ECC check runs once; a stretch of
max_gap frames without an anchor is closed by forward-backward tracking, and so is the stretch before a stamp gap
above max_dt_s (dropped frames, a paused teacher, a reconnect), so tracking never bridges frames that are not
consecutive. A labelled frame is skipped as a near-duplicate if the board moved less than min_motion_px since the
last kept frame of its diversity bucket (0 keeps them all), and once its bucket holds cap_per_bucket frames (0: no
cap). Kept frames are banked in chunks of chunk_size under <session_dir>/harvest[/<bank>]/{train,heldout}/, named
chunk_<prefix><n>. Every heldout_every-th chunk is held out; or, with split_block_s > 0, a frame is held out when its
stamp falls in every heldout_every-th block of split_block_s seconds, so the same frame lands in the same split
whichever teacher labelled it (a relabelled bank cannot leak held-out frames into training).
camera is the pipeline config's CAMERA block (K and dist at image_size): with it the lattice is fitted
and tracked on undistorted geometry and the labels are mapped back through the lens, which matters
toward the frame edges; without it the frames are treated as distortion-free.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from pathlib import Path
import threading
import time

import numpy as np


@dataclass
class HarvestParams:
    max_gap: int = 150
    max_dt_s: float = 0.5
    min_motion_px: float = 2.0
    cap_per_bucket: int = 400
    chunk_size: int = 120
    heldout_every: int = 5
    split_block_s: float = 0.0
    overrides: dict = field(default_factory=dict)


class OnlineHarvester:

    def __init__(self, model_cfg: dict, session_dir, params: HarvestParams | None = None,
                 camera: dict | None = None, bank: str = "", prefix: str = ""):
        from dcc import video
        from dcc.board import n_corners
        self.V = video
        self.p = params or HarvestParams()
        self.W, self.H = model_cfg["input_size"]
        self.n = math.isqrt(n_corners(model_cfg.get("board")))
        self.hp = {**video.HARVEST_DEFAULTS, **self.p.overrides}
        self.tmpl = video.BoardTemplate(model_cfg.get("board"), self.hp["template_size"])
        cam = camera or {}
        size = cam.get("image_size") or (self.W, self.H)
        self.lens = video.Lens(cam.get("K"), cam.get("dist"), self.W / float(size[0]), (self.W, self.H))
        self.calibrated = cam.get("K") is not None
        self.root = Path(session_dir) / "harvest" / bank if bank else Path(session_dir) / "harvest"
        self.prefix = prefix
        self.lock = threading.Lock()
        self.buf: list = []
        self.has_anchor = False
        self.last_stamp = None
        self.seq = 0
        self.last_kept: dict = {}
        self.buckets: Counter = Counter()
        self.open = {"train": ([], []), "heldout": ([], [])}
        self.images: list = []
        self.records: list = []
        self.chunk_id = 0
        self.chunks = {"train": [], "heldout": []}
        self.frames_in = {"train": 0, "heldout": 0}
        self.totals: Counter = Counter()

    def add(self, stamp: float, image: np.ndarray, result: dict) -> dict:
        t0 = time.perf_counter()
        r = self.W / image.shape[1]
        frame = self.V.to_input(image, self.W, self.H)
        fit = self.V.anchor_from_detection(result, r, self.n, self.lens, **self.hp)
        corners = result["corners"]
        event = {"seq": self.seq, "stamp": stamp, "anchor": fit is not None,
                 "n_detections": len(corners), "n_ids": sum(c["index"] is not None for c in corners),
                 "rms": None if fit is None else round(fit.rms, 4), "level": float(np.percentile(frame, 99)),
                 "segment": None}
        if self.last_stamp is not None and not 0.0 <= stamp - self.last_stamp <= self.p.max_dt_s:
            event["break"] = self.break_stream()
        self.last_stamp = stamp
        self.buf.append((self.seq, stamp, frame, fit))
        self.seq += 1
        with self.lock:
            self.totals["frames"] += 1
            self.totals["anchors"] += fit is not None
            self.totals["breaks"] += "break" in event
        if fit is not None:
            if self.has_anchor and len(self.buf) == 2:
                event["segment"] = self._segment(self.buf[1:], {0: fit}, 0)
            else:
                anchors = {len(self.buf) - 1: fit}
                if self.has_anchor:
                    anchors[0] = self.buf[0][3]
                event["segment"] = self._segment(self.buf, anchors, 1 if self.has_anchor else 0)
            self.buf, self.has_anchor = [self.buf[-1]], True
        elif len(self.buf) > self.p.max_gap:
            if self.has_anchor:
                event["segment"] = self._segment(self.buf, {0: self.buf[0][3]}, 1)
                self.buf, self.has_anchor = [], False
            else:
                self.buf = self.buf[-self.p.max_gap:]
        event["ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        return event

    def break_stream(self) -> dict | None:
        seg = None
        if self.has_anchor and len(self.buf) > 1:
            seg = self._segment(self.buf, {0: self.buf[0][3]}, 1)
        self.buf, self.has_anchor = [], False
        return seg

    def flush(self) -> dict | None:
        seg = self.break_stream()
        if self.p.split_block_s > 0:
            for split in ("train", "heldout"):
                self._write_chunk(split)
        else:
            self._write_chunk()
        return seg

    def _segment(self, buf: list, anchors: dict, emit_from: int) -> dict:
        frames = [b[2] for b in buf]
        und = [self.lens.undistort_image(f) for f in frames]
        labels, st = self.V.harvest_labels(und, anchors, self.tmpl, self.n, self.hp)
        out = Counter()
        rho = []
        for t in sorted(labels):
            if t < emit_from:
                continue
            rec = self.V.build_record(t, frames[t], und[t], labels[t], self.tmpl, self.n, self.lens, self.hp)
            rec["frame"] = rec["source_frame"] = buf[t][0]
            rec["stamp"] = buf[t][1]
            out["labelled"] += 1
            out[rec["source"]] += 1
            out[self._keep(rec, frames[t])] += 1
            rho.append(round(labels[t]["rho"], 4))
        with self.lock:
            self.totals.update(out)
        return {"length": len(buf), "labelled": out["labelled"], "anchor": out["anchor"], "tracked": out["tracked"],
                "kept": out["kept"], "duplicate": out["duplicate"], "capped": out["capped"],
                "closures": [round(e, 4) for e in st["closures"]], "fb": [round(e, 4) for e in st["fb"]],
                "rho": rho}

    def _keep(self, rec: dict, image: np.ndarray) -> str:
        b = self.V.bucket_of(rec)
        if self.p.cap_per_bucket and self.buckets[b] >= self.p.cap_per_bucket:
            return "capped"
        pts = np.array([[c["x"], c["y"]] for c in rec["corners"]])
        last = self.last_kept.get(b)
        if last is not None:
            moved = float(np.linalg.norm(pts - last[0], axis=1).max())
            if moved < self.p.min_motion_px and abs(rec["board_mean"] - last[1]) < 0.1 * max(last[1], 1.0):
                return "duplicate"
        self.last_kept[b] = (pts, rec["board_mean"])
        with self.lock:
            self.buckets[b] += 1
        if self.p.split_block_s > 0:
            split = self.split_of(rec["stamp"])
            images, records = self.open[split]
            images.append(image)
            records.append(rec)
            if len(records) >= self.p.chunk_size:
                self._write_chunk(split)
            return "kept"
        self.images.append(image)
        self.records.append(rec)
        if len(self.records) >= self.p.chunk_size:
            self._write_chunk()
        return "kept"

    def split_of(self, stamp: float) -> str:
        """Return the split a frame belongs to when splitting by time blocks.

        Args:
            stamp: Frame stamp, in seconds.
        """
        block = int(math.floor(stamp / self.p.split_block_s))
        return "heldout" if block % self.p.heldout_every == self.p.heldout_every - 1 else "train"

    def _write_chunk(self, split: str | None = None) -> None:
        if split is None:
            images, records = self.images, self.records
            split = "heldout" if self.chunk_id % self.p.heldout_every == self.p.heldout_every - 1 else "train"
        else:
            images, records = self.open[split]
        if not records:
            return
        meta = {"input_size": [self.W, self.H], "stamps": [records[0]["stamp"], records[-1]["stamp"]],
                "noise": self.V.noise_model(images, records)}
        d, _ = self.V.save_harvest(self.root / split, f"chunk_{self.prefix}{self.chunk_id:04d}", images, records, meta)
        with self.lock:
            self.chunks[split].append(str(d))
            self.frames_in[split] += len(records)
        self.chunk_id += 1
        images.clear()
        records.clear()

    def coverage(self) -> dict:
        with self.lock:
            return {f"{s},{b},{a}": n for (s, b, a), n in self.buckets.items()}

    def snapshot(self) -> dict:
        with self.lock:
            open_chunk = len(self.records) + sum(len(r) for _, r in self.open.values())
            return {"totals": dict(self.totals), "chunks": {k: list(v) for k, v in self.chunks.items()},
                    "banked": dict(self.frames_in), "open_chunk": open_chunk, "buffered": len(self.buf)}
