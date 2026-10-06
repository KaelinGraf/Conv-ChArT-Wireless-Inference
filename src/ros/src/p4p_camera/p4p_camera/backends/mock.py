"""Synthetic or replayed frames, so the node runs with no camera attached.

This is what makes the rest of the package testable. Every assertion the test
suite makes about the stream -- shape, dtype, losslessness, stamps, QoS matching,
the full-res service -- is made against this backend, and the only code left
untested until the sensor arrives is picam.py and v4l2.py.

Two knobs earn their keep beyond the unit tests:

    mock_image        replay a recorded 1600x1200 board frame, so the laptop's
                      real inference node can be run end to end over the actual
                      WiFi link before the camera exists
    mock_latency_ms   back-date the reported capture clock, so the sensor-stamp
                      mapping in frames.ros_stamp_ns is exercised off-bench
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np

from . import CameraBackend, CameraError
from .. import frames

_MARKERS = ((100, 200), (600, 400), (1100, 800), (1500, 1000))
_MARKER_VALUE = 255
_FIELD = 110


class MockBackend(CameraBackend):
    """Deterministic frames at the requested rate."""

    name = 'mock'

    def __init__(self, params: dict, logger):
        self._log = logger
        self._path = str(params.get('mock_image', '') or '')
        self._latency_ns = int(float(params.get('mock_latency_ms', 0.0)) * 1e6)
        rate = float(params.get('mock_rate', 0.0) or 0.0)
        if rate <= 0.0:
            rate = float(params.get('frame_rate', 15.0))
        self._period = 1.0 / max(rate, 0.1)
        self._started = False
        self._count = 0
        self._next = 0.0
        self._base: np.ndarray | None = None

    def start(self) -> None:
        if self._path:
            if not os.path.exists(self._path):
                raise CameraError(f'mock_image {self._path!r} does not exist')
            img = cv2.imread(self._path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise CameraError(f'mock_image {self._path!r} is not a readable image')
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img.dtype != np.uint8:
                raise CameraError(f'mock_image must be 8-bit, got {img.dtype}')
            if img.shape != frames.FULL:
                raise CameraError(
                    f'mock_image is {img.shape}, expected {frames.FULL}')
            self._base = img
            self._log.info(f'mock backend replaying {self._path}')
        else:
            self._base = self._synthetic()
            self._log.info('mock backend generating synthetic frames')
        self._started = True
        self._next = time.monotonic()

    def read(self, timeout_s: float) -> tuple[np.ndarray, int | None]:
        if not self._started or self._base is None:
            raise CameraError('mock backend read before start')
        now = time.monotonic()
        wait = self._next - now
        if wait > timeout_s:
            raise TimeoutError(
                f'mock frame is {wait:.3f} s away, past the {timeout_s:.3f} s timeout')
        if wait > 0:
            time.sleep(wait)
        self._next = max(self._next + self._period, time.monotonic())

        frame = self._base.copy()
        self._count += 1
        if not self._path:
            self._stamp_counter(frame, self._count)
        stamp = time.clock_gettime_ns(time.CLOCK_BOOTTIME) - self._latency_ns
        return frame, stamp

    def stop(self) -> None:
        self._started = False
        self._base = None

    @staticmethod
    def _synthetic() -> np.ndarray:
        """Build a flat field with fixed noise, a border and known markers."""
        h, w = frames.FULL
        rng = np.random.default_rng(0xC0FFEE)
        noise = rng.integers(-8, 9, size=(h, w), dtype=np.int16)
        frame = np.clip(np.int16(_FIELD) + noise, 0, 255).astype(np.uint8)
        frame[0, :] = frame[-1, :] = frame[:, 0] = frame[:, -1] = 200
        for x, y in _MARKERS:
            frame[y - 2:y + 3, x - 2:x + 3] = _MARKER_VALUE
        return frame

    @staticmethod
    def _stamp_counter(frame: np.ndarray, count: int) -> None:
        """Burn the frame number in, so a human can see the stream is live."""
        cv2.putText(frame, str(count), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, 255, 3, cv2.LINE_AA)


def expected_stream_markers() -> list[tuple[int, int, int]]:
    """Marker positions and values as they should decode off the 640x480 stream."""
    out = []
    for x, y in _MARKERS:
        out.append((int(x / frames.SCALE), int(y / frames.SCALE), _MARKER_VALUE))
    return out
