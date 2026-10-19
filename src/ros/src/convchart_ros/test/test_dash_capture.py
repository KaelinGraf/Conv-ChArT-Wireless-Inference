"""
The dashboard's docking captures (dash_capture.py): the alpha from one drawing (unmix), the frame timing, and the
files ffmpeg writes.

    python3 -m pytest src/ros/src/convchart_ros/test/test_dash_capture.py -q

ROS-free. The file tests need ffmpeg and ffprobe on PATH and are skipped without them.
"""
import json
import shutil
import subprocess
import time

import cv2
import numpy as np
import pytest

from convchart_ros.dash_capture import Capture, CLEAR, frames_due, frames_total, Recorder, unmix

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg and ffprobe")


def drawing(bg):
    """Return the dashboard's kinds of drawing, in 3-value colours, over a background.

    The drawing has AA and plain lines, outlined text, a filled dot and a 30 % band.

    Args:
        bg: Background colour, with 3 or 4 values.
    """
    img = np.empty((120, 200, len(bg)), np.uint8)
    img[:] = bg
    cv2.line(img, (0, 50), (199, 50), (42, 44, 44), 1)
    cv2.line(img, (5, 100), (190, 20), (229, 135, 57), 2, cv2.LINE_AA)
    cv2.putText(img, "x +0.123", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(img, (150, 80), 6, (38, 89, 217), -1, cv2.LINE_AA)
    for colour, thick in (((13, 13, 13), 3), ((183, 194, 195), 1)):
        cv2.putText(img, "theta [deg]", (40, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, thick, cv2.LINE_AA)
    over = img.copy()
    cv2.fillPoly(over, [np.array([(20, 60), (120, 50), (120, 90), (20, 95)], np.int32)], (112, 158, 25), cv2.LINE_AA)
    cv2.addWeighted(over, 0.3, img, 0.7, 0, dst=img)
    return img


def test_unmix_lays_the_drawing_over_any_background_as_if_drawn_on_it():
    hud = unmix(drawing(CLEAR))
    assert hud.shape == (120, 200, 4) and hud.dtype == np.uint8
    assert hud[2, 2, 3] == 0 and np.all(hud[2, 2, :3] == 0)
    assert hud[80, 150, 3] == 255
    assert abs(int(hud[75, 30, 3]) - 77) <= 2
    rng = np.random.default_rng(1)
    for bg in ((40, 40, 40), (200, 180, 150), tuple(int(v) for v in rng.integers(0, 256, 3))):
        a = hud[..., 3:].astype(np.float32) / 255.0
        laid = hud[..., :3].astype(np.float32) * a + np.array(bg, np.float32) * (1.0 - a)
        err = np.abs(laid - drawing(bg).astype(np.float32))
        assert err.max() <= 4.5 and err.mean() < 0.3, (bg, err.max(), err.mean())


def test_frame_i_shows_the_newest_picture_drawn_by_its_time():
    assert frames_due(0.0, 0.0, 30) == 0 and frames_due(0.001, 0.0, 30) == 1 and frames_due(1 / 30, 0.0, 30) == 1
    assert frames_total(2.0, 0.0, 30) == 60 and frames_total(0.0, 0.0, 30) == 1
    shown = []
    rec = Recorder("unused.mp4", 4, 2, "bgr", 10.0, 100.0, sink=lambda f: shown.append(int(f[0, 0, 0])))
    for t, value in ((100.02, 1), (100.25, 2), (100.27, 3), (100.61, 4)):
        rec.push(t, np.full((2, 4, 3), value, np.uint8))
        time.sleep(0.02)
    rec.close(101.0)
    assert rec.join(5.0) and rec.error is None
    assert shown == [1, 1, 1, 3, 3, 3, 3, 4, 4, 4] and rec.written == 10


def test_a_picture_pushed_while_the_encoder_is_behind_is_dropped_and_the_length_kept():
    rec = Recorder("unused.mp4", 4, 2, "bgr", 30.0, 0.0, sink=lambda f: time.sleep(0.01))
    for k in range(40):
        rec.push(k / 30, np.zeros((2, 4, 3), np.uint8))
    rec.close(40 / 30)
    assert rec.join(10.0) and rec.dropped > 0 and rec.pushed == 40 and rec.written == 40
    with pytest.raises(ValueError, match="takes"):
        rec.push(0.0, np.zeros((2, 4, 4), np.uint8))


def ffprobe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=codec_name,pix_fmt,width,height,nb_read_frames", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"][0]


def decode(path, index, w, h, pix="rgba"):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{index})", "-vsync", "0",
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", pix, "-"], capture_output=True,
                         check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w, 4 if pix == "rgba" else 3)


@needs_ffmpeg
def test_a_capture_writes_same_length_files_and_the_hud_keeps_its_transparency(tmp_path):
    hud = unmix(drawing(CLEAR))
    t0 = time.monotonic()
    cap = Capture(tmp_path / "dock_20261015_000000", {"window": (200, 120, "bgr"), "hud": (200, 120, "bgra")},
                  30.0, t0, {"note": "test"})
    assert json.loads((cap.folder / "capture.json").read_text())["status"] == "recording"
    for k in range(20):
        t = t0 + k * 0.05
        cap.push("window", t, drawing((30 + 5 * k,) * 3))
        cap.push("hud", t, hud)
        if k == 10:
            cap.mark(t, "stage APPROACH")
        time.sleep(0.01)
    done = []
    cap.stop(t0 + 1.0, "STOP", on_done=done.append)
    assert cap.wait(60.0) and done == [cap] and not cap.errors(), cap.errors()
    record = json.loads((cap.folder / "capture.json").read_text())
    assert record["status"] == "complete" and record["frames"] == 30 and record["stopped_by"] == "STOP"
    assert record["note"] == "test" and {"t_s": 0.5, "event": "stage APPROACH"} in record["events"]
    window, hud_file = ffprobe(cap.folder / "window.mp4"), ffprobe(cap.folder / "hud.mov")
    assert (window["codec_name"], window["pix_fmt"], window["width"], window["height"]) == ("h264", "yuv420p", 200, 120)
    assert hud_file["codec_name"] == "prores" and hud_file["pix_fmt"].startswith("yuva444")
    assert int(window["nb_read_frames"]) == int(hud_file["nb_read_frames"]) == 30
    back = decode(cap.folder / "hud.mov", 12, 200, 120)
    assert np.abs(back[..., 3].astype(int) - hud[..., 3]).max() <= 2
    opaque = hud[..., 3] == 255
    assert np.abs(back[opaque][:, :3].astype(int) - hud[opaque][:, 2::-1]).mean() < 3
    assert cap.summary().startswith("dock_20261015_000000: 2 videos, 1.0 s")


@needs_ffmpeg
def test_an_encoder_that_fails_is_reported_and_the_others_still_finish(tmp_path):
    t0 = time.monotonic()
    cap = Capture(tmp_path / "c", {"ok": (64, 48, "bgr")}, 10.0, t0)
    broken = Recorder(cap.folder / "broken.mp4", 64, 48, "bgr", 10.0, t0, ffmpeg="false")
    cap.recorders["broken"] = broken
    for k in range(5):
        cap.push("ok", t0 + k * 0.1, np.zeros((48, 64, 3), np.uint8))
        cap.push("broken", t0 + k * 0.1, np.zeros((48, 64, 3), np.uint8))
        time.sleep(0.02)
    cap.stop(t0 + 0.5, "CANCEL DOCKING")
    assert cap.wait(30.0)
    assert set(cap.errors()) == {"broken"} and cap.summary().startswith("c: FAILED broken:")
    assert json.loads((cap.folder / "capture.json").read_text())["status"] == "failed"
    assert int(ffprobe(cap.folder / "ok.mp4")["nb_read_frames"]) == 5
