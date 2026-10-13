#!/usr/bin/env python3
"""Quick live view: the Pi's `image` stream through the conv-chart pipeline, drawn with OpenCV.

    python3 tools/live_pose_view.py [cfg.yaml]          # q or Esc quits

Green circle + number: identified corner and its ID. Red circle: detected, not identified.
Axes and the blue outline: the PnP pose (x red, y green, z blue, one square long).
With CAMERA.K still null in cfg.yaml a nominal K is used, so the pose overlays fine but its numbers are rough.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from sensor_msgs.msg import CompressedImage

from convchart_qos.qos import IMAGE_SUB_QOS

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import inference as inf  # noqa: E402

SCALE = 1.5
WINDOW = "conv-chart live"

cfg = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "cfg" / "cfg.yaml")
pipe = inf.inference_pipeline(cfg)
nominal_k = pipe.camera.K is None
K = np.array([[560.0, 0, 319.5], [0, 560.0, 239.5], [0, 0, 1]]) if nominal_k else pipe.camera.K
dist = np.zeros(5) if nominal_k or pipe.camera.dist is None else pipe.camera.dist
sq = pipe.board.square_length_m or 1.0
unit = "m" if pipe.board.square_length_m else "sq"
lattice = inf._CANON if pipe.board.n == 4 else inf.canon_lattice(pipe.board.n)
(x0, y0), (x1, y1) = lattice.min(0), lattice.max(0)
outline = np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]], dtype=np.float64) * sq

K_disp = K.copy()
K_disp[0, 0] *= SCALE
K_disp[1, 1] *= SCALE
K_disp[:2, 2] = SCALE * (K[:2, 2] + 0.5) - 0.5


def text(img, s, row, colour=(255, 255, 255)):
    org = (8, 24 + 24 * row)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA)


def draw(gray, res, infer_ms, hz):
    vis = cv2.cvtColor(cv2.resize(gray, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR),
                       cv2.COLOR_GRAY2BGR)
    for c in res["corners"]:
        p = (int(round(SCALE * (c["x"] + 0.5) - 0.5)), int(round(SCALE * (c["y"] + 0.5) - 0.5)))
        if c["index"] is None:
            cv2.circle(vis, p, 4, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(vis, p, 5, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, str(c["index"]), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 0), 1, cv2.LINE_AA)
    n_id = sum(c["index"] is not None for c in res["corners"])
    text(vis, f"{res['reason'] or 'OK'}   ids {n_id}/{len(res['corners'])}   "
              f"infer {infer_ms:.0f} ms   {hz:.1f} Hz", 0)
    if res["rvec"] is not None:
        rvec, tvec = res["rvec"], res["tvec"]
        img_pts, _ = cv2.projectPoints(outline, rvec, tvec, K_disp, dist)
        cv2.polylines(vis, [np.round(img_pts).astype(np.int32)], True, (255, 128, 0), 2, cv2.LINE_AA)
        cv2.drawFrameAxes(vis, K_disp, dist, rvec, tvec, sq, 2)
        t = np.ravel(tvec)
        ang = np.degrees(np.ravel(rvec))
        text(vis, f"t = ({t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f}) {unit}   range {np.linalg.norm(t):.2f} {unit}"
                  f"   rms {res['rms']:.2f} px" + ("   AMBIGUOUS" if res["ambiguous"] else ""), 1,
             (0, 200, 255) if res["ambiguous"] else (255, 255, 255))
        text(vis, f"rvec = ({ang[0]:+.1f}, {ang[1]:+.1f}, {ang[2]:+.1f}) deg", 2)
    if nominal_k:
        text(vis, "UNCALIBRATED K (nominal f = 560 px): pose numbers approximate", 3, (0, 200, 255))
    return vis


def main():
    rclpy.init()
    node = rclpy.create_node("live_pose_view")
    latest = [None]
    node.create_subscription(CompressedImage, "image", lambda m: latest.__setitem__(0, m), IMAGE_SUB_QOS)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    hz, t_prev, n, t_log = 0.0, None, 0, time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            msg, latest[0] = latest[0], None
            if msg is not None:
                gray = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_GRAYSCALE)
                t0 = time.perf_counter()
                res = pipe.run_inference(gray, K=K, dist=dist)
                infer_ms = (time.perf_counter() - t0) * 1e3
                now = time.monotonic()
                if t_prev is not None:
                    hz = 0.9 * hz + 0.1 / max(now - t_prev, 1e-3) if hz else 1.0 / max(now - t_prev, 1e-3)
                t_prev, n = now, n + 1
                cv2.imshow(WINDOW, draw(gray, res, infer_ms, hz))
                if now - t_log > 3.0:
                    t_log = now
                    print(f"{n} frames  {hz:.1f} Hz  infer {infer_ms:.0f} ms  {res['reason'] or 'OK'}  "
                          f"ids {sum(c['index'] is not None for c in res['corners'])}/{len(res['corners'])}",
                          flush=True)
            key = cv2.waitKey(1 if msg is not None else 5) & 0xFF
            if key in (27, ord("q")) or (n and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1):
                break
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
