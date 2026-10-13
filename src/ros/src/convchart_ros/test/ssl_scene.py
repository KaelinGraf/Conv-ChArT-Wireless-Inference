"""Rendered board video with known geometry, shared by the test_ssl_* tests (needs dcc on sys.path).

    H = homography(t); img = render(H, gain, noise, rng); res = result_for(H, anchor=True, rng=rng)
"""
from pathlib import Path
import sys

import cv2
import numpy as np

for _parent in Path(__file__).resolve().parents:
    if (_parent / "Conv-ChArT" / "dcc" / "video.py").is_file():
        if str(_parent / "Conv-ChArT") not in sys.path:
            sys.path.insert(0, str(_parent / "Conv-ChArT"))
        break

MODEL_CFG = {"input_size": [640, 480], "board": None}


def lattice():
    from dcc.pipeline import canon_lattice
    return canon_lattice(4)


def homography(t, s=40.0, theta=0.3, tx=300.0, ty=240.0, static=False):
    if not static:
        s, theta, tx, ty = s + 0.4 * t, theta + 0.012 * t, tx + 3.0 * t, ty - 1.0 * t
    c, sn = np.cos(theta), np.sin(theta)
    A = np.array([[s * c, -s * sn, tx - s * 2.5 * (c - sn)], [s * sn, s * c, ty - s * 2.5 * (sn + c)], [0, 0, 1.0]])
    return A @ np.array([[1, 0, 0], [0, 1, 0], [2e-4, -1e-4, 1.0]])


def render(H, gain=1.0, noise=1.0, rng=None, bg=100.0):
    from dcc.board import render_board
    img, _ = render_board(480)
    board = 30 + 180 * img.astype(np.float32) / 255.0
    Ar = np.array([[96.0, 0, -0.5], [0, 96.0, -0.5], [0, 0, 1.0]])
    Hr = H @ np.linalg.inv(Ar)
    warped = cv2.warpPerspective(cv2.GaussianBlur(board, (0, 0), 1.0), Hr, (640, 480), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.ones_like(board), Hr, (640, 480), flags=cv2.INTER_LINEAR)
    f = (bg * (1 - mask) + warped * mask) * gain
    if noise and rng is not None:
        f = f + rng.normal(0, noise, f.shape)
    return np.clip(np.rint(f), 0, 255).astype(np.uint8)


def truth(H):
    p = np.hstack([lattice(), np.ones((16, 1))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def result_for(H, anchor, rng):
    if not anchor:
        return {"corners": []}
    pts = truth(H) + rng.normal(0, 0.05, (16, 2))
    return {"corners": [{"x": float(x), "y": float(y), "index": k, "source": "head", "p_id": 0.95, "sigma_px": 0.1,
                         "p_hm": 0.9} for k, (x, y) in enumerate(pts)]}
