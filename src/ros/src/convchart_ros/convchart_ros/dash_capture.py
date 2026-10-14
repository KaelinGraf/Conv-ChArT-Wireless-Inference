"""
Video captures of the telemetry dashboard while the robot docks (ROS-free, so testable).

    cap = Capture(folder, {"dashboard": (1600, 960, "bgr"), "topdown_hud": (640, 416, "bgra")}, fps=30.0,
                  t0=time.monotonic())
    cap.push("dashboard", t, canvas)        # whenever a new picture is drawn; it is held until the next one
    cap.mark(t, "stage APPROACH")           # an event, in capture.json with its time in the video
    cap.stop(t, "STOP")                     # returns at once; the files are finished in the background
    cap.wait(30.0)

Each output is its own ffmpeg process, fed raw frames by its own thread at a constant frame rate: frame i of
every file is the newest picture drawn by t0 + i / fps. So the files of one capture start together and have the
same number of frames, and lie over each other (or over a film of the robot) with one offset. Opaque outputs
("bgr") are H.264 in .mp4; outputs with alpha ("bgra") are ProRes 4444 in .mov, with straight alpha, which video
editors take as an overlay. ffmpeg runs in its own session and at lower priority: a Ctrl-C meant for the dashboard
does not cut a file short, and if the dashboard dies the pipe closes and ffmpeg still finishes the file. A
picture pushed while its encoder is behind is dropped (counted), and the one before is held for longer.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading
import time

import cv2
import numpy as np

KINDS = {"bgr": 3, "bgra": 4}
QUEUE = 8
FINISH_S = 120.0


def matte(on_black: np.ndarray, on_white: np.ndarray) -> np.ndarray:
    """The BGRA picture (straight alpha) of a drawing, from two copies of it: one drawn on black,
    one on white.
    """
    alpha = 255 - cv2.transform(cv2.subtract(on_white, on_black), np.full((1, 3), 1.0 / 3.0))
    colour = cv2.divide(on_black, cv2.merge([alpha, alpha, alpha]), scale=255.0)
    return cv2.merge([*cv2.split(colour), alpha])


def encoder(kind: str, fps: float) -> tuple[str, str, list]:
    """(file suffix, ffmpeg input pixel format, ffmpeg output options) for an output of this kind."""
    if kind == "bgr":
        return ".mp4", "bgr24", ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-g", str(round(fps)),
                                 "-pix_fmt", "yuv420p"]
    if kind == "bgra":
        return ".mov", "bgra", ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le",
                                "-alpha_bits", "16", "-vendor", "apl0"]
    raise ValueError(f"an output is bgr or bgra, got {kind!r}")


def frames_due(t: float, t0: float, fps: float) -> int:
    """How many frames show times before t: frame i shows t0 + i / fps."""
    return max(0, math.ceil((t - t0) * fps - 1e-6))


def frames_total(t_end: float, t0: float, fps: float) -> int:
    """The frames of a capture that ended at t_end: at least one."""
    return max(1, round((t_end - t0) * fps))


class Recorder:
    """One output: a thread that holds each picture until the next and writes it at the frame times."""

    def __init__(self, path: Path, width: int, height: int, kind: str, fps: float, t0: float,
                 ffmpeg: str = "ffmpeg", sink=None):
        self.path, self.kind, self.fps, self.t0 = Path(path), kind, float(fps), float(t0)
        self.shape = (int(height), int(width), KINDS[kind])
        self.ffmpeg, self.sink = ffmpeg, sink
        self.pushed = self.dropped = self.written = 0
        self.error = None
        self.end_t = None
        self._queue = queue.Queue(maxsize=QUEUE)
        self._closing = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"capture_{self.path.stem}", daemon=True)
        self._thread.start()

    def push(self, t: float, frame: np.ndarray) -> None:
        if frame.shape != self.shape:
            raise ValueError(f"{self.path.name} takes {self.shape} pictures, got {frame.shape}")
        if self.error is not None or self._closing.is_set():
            return
        self.pushed += 1
        try:
            self._queue.put_nowait((t, frame))
        except queue.Full:
            self.dropped += 1

    def close(self, t_end: float) -> None:
        if not self._closing.is_set():
            self.end_t = float(t_end)
            self._closing.set()

    def join(self, timeout: float | None = None) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _run(self) -> None:
        proc, log, write = None, None, self.sink
        try:
            if write is None:
                proc, log = self._start()

                def write(frame):
                    proc.stdin.write(memoryview(np.ascontiguousarray(frame)).cast("B"))
            held = None
            while True:
                try:
                    t, frame = self._queue.get(timeout=0.1)
                except queue.Empty:
                    if self._closing.is_set() and self._queue.empty():
                        break
                    continue
                self._write_until(write, frame if held is None else held, frames_due(t, self.t0, self.fps))
                held = frame
            if held is None:
                held = np.zeros(self.shape, np.uint8)
            self._write_until(write, held, frames_total(self.end_t, self.t0, self.fps))
        except Exception as error:          # noqa: BLE001
            self.error = f"{type(error).__name__}: {error}"
        finally:
            if proc is not None:
                self._finish(proc, log)

    def _write_until(self, write, frame, n) -> None:
        while self.written < n:
            write(frame)
            self.written += 1

    def _start(self):
        suffix, pixels, options = encoder(self.kind, self.fps)
        h, w, _ = self.shape
        cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", pixels,
               "-s", f"{w}x{h}", "-framerate", f"{self.fps:g}", "-i", "pipe:0", "-an", *options, str(self.path)]
        if shutil.which("nice"):
            cmd = ["nice", "-n", "10", *cmd]
        log = tempfile.TemporaryFile()
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log,
                                start_new_session=True)
        return proc, log

    def _finish(self, proc, log) -> None:
        try:
            proc.stdin.close()
        except OSError:
            pass
        try:
            code = proc.wait(timeout=FINISH_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = proc.wait()
        if code != 0:
            log.seek(0)
            tail = log.read().decode(errors="replace").strip().splitlines()[-3:]
            self.error = f"ffmpeg exited with {code}" + (f": {' | '.join(tail)}" if tail else "") + (
                f" (after {self.error})" if self.error else "")
        log.close()


class Capture:
    """The outputs of one capture, in one folder, with its record (capture.json)."""

    def __init__(self, folder: Path, outputs: dict, fps: float, t0: float, info: dict | None = None,
                 ffmpeg: str = "ffmpeg", sinks: dict | None = None):
        self.folder, self.fps, self.t0 = Path(folder), float(fps), float(t0)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.started = time.time()
        self.info = dict(info or {})
        self.events = [(0.0, "start")]
        self.reason = self.end_t = None
        self.done = threading.Event()
        self.recorders = {}
        for name, (w, h, kind) in outputs.items():
            suffix = encoder(kind, fps)[0]
            self.recorders[name] = Recorder(self.folder / f"{name}{suffix}", w, h, kind, fps, t0, ffmpeg,
                                            (sinks or {}).get(name))
        self._write_record("recording")

    @property
    def recording(self) -> bool:
        return self.end_t is None

    def elapsed(self, now: float) -> float:
        return (now if self.end_t is None else self.end_t) - self.t0

    def push(self, name: str, t: float, frame: np.ndarray) -> None:
        if self.end_t is None:
            self.recorders[name].push(t, frame)

    def mark(self, t: float, what: str) -> None:
        if self.end_t is None:
            self.events.append((round(t - self.t0, 3), what))

    def stop(self, t: float, reason: str, on_done=None) -> None:
        """End the capture at t; the files are finished on a thread, which then calls on_done(self)."""
        if self.end_t is not None:
            return
        self.end_t, self.reason = float(t), reason
        self.events.append((round(t - self.t0, 3), f"stop ({reason})"))
        for r in self.recorders.values():
            r.close(t)

        def finish():
            for r in self.recorders.values():
                r.join()
            self._write_record("complete" if not self.errors() else "failed")
            self.done.set()
            if on_done is not None:
                on_done(self)
        threading.Thread(target=finish, name="capture_finish", daemon=True).start()

    def wait(self, timeout: float | None = None) -> bool:
        return self.done.wait(timeout)

    def errors(self) -> dict:
        return {name: r.error for name, r in self.recorders.items() if r.error}

    def summary(self) -> str:
        n = frames_total(self.end_t, self.t0, self.fps) if self.end_t is not None else 0
        errors = self.errors()
        if errors:
            return f"{self.folder.name}: FAILED " + "; ".join(f"{k}: {v}" for k, v in errors.items())
        dropped = sum(r.dropped for r in self.recorders.values())
        return (f"{self.folder.name}: {len(self.recorders)} videos, {n / self.fps:.1f} s"
                + (f", {dropped} pictures dropped" if dropped else ""))

    def _write_record(self, status: str) -> None:
        def utc(t):
            return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int(t % 1 * 1000):03d}Z"

        record = {
            "status": status,
            "started_utc": utc(self.started),
            "fps": self.fps,
            "timing": "frame i of every file shows the dashboard at started_utc + i / fps (the newest picture "
                      "drawn by then); events are seconds into the videos",
            **self.info,
            "outputs": {name: {"file": r.path.name, "width": r.shape[1], "height": r.shape[0],
                               "alpha": r.kind == "bgra",
                               "codec": "ProRes 4444, straight alpha" if r.kind == "bgra" else "H.264 yuv420p",
                               "frames": r.written, "pictures_drawn": r.pushed, "pictures_dropped": r.dropped,
                               **({"error": r.error} if r.error else {})}
                        for name, r in self.recorders.items()},
            "events": [{"t_s": t, "event": e} for t, e in self.events],
        }
        if self.end_t is not None:
            record.update(stopped_by=self.reason, duration_s=round(self.end_t - self.t0, 3),
                          frames=frames_total(self.end_t, self.t0, self.fps))
        path = self.folder / "capture.json"
        tmp = path.with_name(".capture.json.tmp")
        tmp.write_text(json.dumps(record, indent=2) + "\n")
        tmp.replace(path)
