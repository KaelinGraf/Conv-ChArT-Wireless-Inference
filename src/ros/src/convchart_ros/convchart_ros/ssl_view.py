"""
Live view of a running ssl fine-tune (the finetune task), for whoever is holding the board.

    python3 -m convchart_ros.ssl_view [--session <ssl_output/session dir>]     # default: the newest session

Left, the camera: the newest frame the node saved to <session>/stream/, read from disk, so the WiFi carries no
second copy of the stream; over it, the teacher's verdict on the newest frame it looked at (frames.jsonl): an
anchor (the board found and fitted to the lattice; the frames between anchors are labelled by tracking), corners
seen but no fit, or no board. Under it (ssl/stats, every second): the bank generation and its teacher, the pass
over the stored frames that each new teacher makes, the live labelling and the round under way, with its steps.
Right, the node's dashboard (ssl/dashboard, every dashboard_period s). The window is scaled to fill the screen; q or
Esc quits.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from convchart_ros.screen import fill_scale, screen_size

WINDOW = "ssl fine-tune"
CAMERA = (960, 720)
STATUS_H = 165
DASH = (1530, 855)
PAGE, INK, MUTED = (24, 24, 24), (235, 235, 235), (150, 150, 150)
GOOD, WARN, CRIT = (90, 200, 90), (40, 180, 240), (70, 70, 230)
OUT_QOS = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=2, reliability=QoSReliabilityPolicy.RELIABLE,
                     durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def put(img, s, x, y, colour=INK, scale=0.7, thick=1):
    cv2.putText(img, s, (int(x), int(y)), FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, (int(x), int(y)), FONT, scale, colour, thick, cv2.LINE_AA)


def newest(stream: Path) -> str | None:
    """Return the newest frame's name, which is the largest, as frames are named by stamp.

    Args:
        stream: Folder of saved frames.
    """
    best = None
    with os.scandir(stream) as it:
        for e in it:
            if best is None or e.name > best:
                best = e.name
    return best


class Viewer:

    def __init__(self, session: Path):
        self.session = session
        self.stream = session / "stream"
        meta = json.loads((session / "session.json").read_text())
        params = ((meta.get("resumes") or [{}])[-1].get("params") or meta.get("params") or {})
        self.need_train, self.need_heldout = params.get("min_train_frames", 400), params.get("min_heldout_frames", 80)
        self.node = rclpy.create_node("ssl_view")
        self.stats, self.dash, self.dash_t = {}, None, None
        self.node.create_subscription(String, "ssl/stats", self._on_stats, OUT_QOS)
        self.node.create_subscription(CompressedImage, "ssl/dashboard", self._on_dash, OUT_QOS)
        self.frames = open(session / "frames.jsonl")
        self.frames.seek(0, os.SEEK_END)
        self.last, self.last_t = None, None
        self.frame_name, self.frame = None, None

    def _on_stats(self, msg):
        try:
            self.stats = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def _on_dash(self, msg):
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            self.dash, self.dash_t = img, time.monotonic()

    def poll(self):
        for line in self.frames.readlines():
            try:
                self.last, self.last_t = json.loads(line), time.monotonic()
            except json.JSONDecodeError:
                pass
        name = newest(self.stream) if self.stream.is_dir() else None
        if name is not None and name != self.frame_name:
            img = cv2.imread(str(self.stream / name), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.frame_name, self.frame = name, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def verdict(self):
        f, r = self.last, self.stats.get("rounds") or {}
        if self.stats.get("paused"):
            progress = r.get("progress")
            return (f"TRAINING round {r.get('round')}: {r.get('phase')}"
                    + (f" {progress:.0%}" if isinstance(progress, float) else "")
                    + " (labelling paused)", WARN)
        if f is None or time.monotonic() - self.last_t > 2.0:
            return "the teacher is not answering (no frames reaching it)", MUTED
        if f.get("anchor"):
            return f"BOARD LOCKED: anchor, {f.get('n_ids', 0)} corners identified", GOOD
        if f.get("n_ids"):
            return f"board seen: {f['n_ids']} corners identified, not enough to anchor", WARN
        if f.get("n_detections"):
            return f"{f['n_detections']} corner-like points, no board identified", WARN
        return "no board in view", CRIT

    def render(self):
        W, H = CAMERA[0] + DASH[0], max(CAMERA[1] + STATUS_H, DASH[1])
        img = np.full((H, W, 3), PAGE, np.uint8)
        if self.frame is not None:
            img[:CAMERA[1], :CAMERA[0]] = cv2.resize(self.frame, CAMERA, interpolation=cv2.INTER_LINEAR)
        else:
            put(img, f"no frames yet in {self.stream}", 20, CAMERA[1] // 2, CRIT)
        text, colour = self.verdict()
        put(img, text, 16, 40, colour, 0.9, 2)
        s = self.stats
        h = s.get("harvest") or {}
        banked, totals = h.get("banked") or {}, h.get("totals") or {}
        r = s.get("rounds") or {}
        g = s.get("generation") or {}
        rl = g.get("relabel") or {}
        teacher = Path(str(g.get("teacher") or s.get("teacher") or "")).name
        relabel = (f"relabelling the stored frames with this teacher: {rl.get('done', 0)} / {rl.get('total', 0)} "
                   f"({rl.get('fps', 0)}/s); rounds wait for it" if not rl.get("finished", True) else
                   f"stored frames relabelled: {rl.get('total', 0)}; rounds from {self.need_train} train / "
                   f"{self.need_heldout} held-out, then each time the bank grows by half")
        steps = r.get("steps")
        lines = [
            (f"generation {g.get('gen', '?')} ({teacher}): banked train {banked.get('train', 0)}   held-out "
             f"{banked.get('heldout', 0)}", INK),
            (relabel, WARN if not rl.get("finished", True) else INK),
            (f"labelled {totals.get('labelled', 0)} (anchors {totals.get('anchor', 0)}, by tracking "
             f"{totals.get('tracked', 0)})   kept {totals.get('kept', 0)}   saved to disk {s.get('saved_frames', 0)}",
             INK),
            ((f"round {r.get('round')}: {r.get('phase')}"
              + (f" {r['progress']:.0%} of {steps} steps" if steps else "")
              + f"   accepted {r.get('accepted') or 'none yet'}") if r else "rounds: training off", INK),
            ("move the board slowly, further and tilted, back to an easy view every few seconds", MUTED),
        ]
        for i, (line, c) in enumerate(lines):
            put(img, line, 12, CAMERA[1] + 30 + 30 * i, c, 0.6)
        if self.dash is not None:
            d = self.dash
            k = min(DASH[0] / d.shape[1], DASH[1] / d.shape[0])
            d = cv2.resize(d, (int(d.shape[1] * k), int(d.shape[0] * k)), interpolation=cv2.INTER_AREA)
            img[:d.shape[0], CAMERA[0]:CAMERA[0] + d.shape[1]] = d
            put(img, f"dashboard, {time.monotonic() - self.dash_t:.0f} s old (every 10 s)", CAMERA[0] + 12,
                DASH[1] + 24, MUTED, 0.55)
        else:
            put(img, "waiting for ssl/dashboard (every 10 s)", CAMERA[0] + 20, DASH[1] // 2, MUTED)
        return img


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--session", help="ssl_output/<session> folder; default the newest under ./ssl_output")
    a, ros_args = ap.parse_known_args()
    session = Path(a.session) if a.session else max(
        (p for p in (Path(__file__).resolve().parents[5] / "ssl_output").iterdir() if (p / "session.json").exists()),
        key=lambda p: p.name)
    rclpy.init(args=ros_args)
    v = Viewer(session)
    scale = None
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    try:
        while rclpy.ok():
            for _ in range(5):
                rclpy.spin_once(v.node, timeout_sec=0.0)
            v.poll()
            img = v.render()
            if scale is None:
                scale = min(1.0, fill_scale(img.shape[1], img.shape[0], screen_size()))
            if scale != 1.0:
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imshow(WINDOW, img)
            if (cv2.waitKey(66) & 0xFF) in (27, ord("q")):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyAllWindows()
        v.node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
