"""V4L2 backend via OpenCV: the default, and the one that works on the Pi 5.

On a Pi 5 the CSI path runs through rp1-cfe's media-controller graph, which
libcamera would normally configure. This module does not touch the graph:
docker/camera-pipeline-pi.sh sets it up on the host, at every boot once
host-setup-pi.sh has installed it as a service. That puts the sensor in its
native 8-bit mono mode (Y8_1X8, 1600x1300), links it through csi2 to
rp1-cfe-csi2_ch0 (/dev/video0) and paces it to 15 fps through vertical
blanking. Without that setup, /dev/video0 opens but never delivers a frame.

Why not picamera2 on the Pi: it needs Arducam's libcamera for this sensor, which
ships for Raspberry Pi OS only (Debian bookworm / trixie). This image is Ubuntu
noble, and its stock libcamera reports no cameras at all.

The same module serves a USB/UVC mono camera during development, and a
Pi-4-style unicam path, where /dev/video0 is the sensor and needs no setup.

It reuses frames.* for every transformation, so the geometry is identical to the
picamera2 path and is covered by the same tests.
"""
from __future__ import annotations

import numpy as np

from . import CameraBackend, CameraError
from .. import frames


class V4L2Backend(CameraBackend):
    """A single V4L2 capture device delivering 8-bit greyscale."""

    name = 'v4l2'

    def __init__(self, params: dict, logger):
        self._log = logger
        self._device = str(params.get('device', '/dev/video0'))
        self._crop_top = int(params.get('crop_top', frames.CROP_TOP))
        self._frame_rate = float(params.get('frame_rate', 15.0))
        self._cap = None

    def start(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self._device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            raise CameraError(
                f'cannot open {self._device} with the V4L2 backend. On a Pi 5, '
                'check the camera is detected (dmesg | grep -i pivariety) and '
                'that docker/camera-pipeline-pi.sh reported this device.')
        # CONVERT_RGB off keeps OpenCV from forcing a 3-channel BGR conversion,
        # which would triple the data and then need converting straight back.
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'GREY'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frames.FULL[1])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frames.ACTIVE[0])
        cap.set(cv2.CAP_PROP_FPS, self._frame_rate)
        self._cap = cap
        self._log.info(
            f'v4l2: {self._device} at '
            f'{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x'
            f'{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}')

    def read(self, timeout_s: float) -> tuple[np.ndarray, int | None]:
        if self._cap is None:
            raise CameraError('v4l2 read before start')
        ok, frame = self._cap.read()
        if not ok or frame is None:
            # VideoCapture gives us no way to distinguish "not yet" from
            # "broken", so treat it as a stall and let the node reopen. On a Pi 5
            # the usual cause is an unconfigured CSI pipeline (after a reboot
            # without the boot service, or after rpicam-*/libcamera ran).
            raise TimeoutError(
                f'no frame from {self._device}. On a Pi 5, is the CSI pipeline '
                'configured? Run docker/camera-pipeline-pi.sh on the host, and '
                'this node reconnects by itself')
        if frame.ndim == 3:
            # The driver ignored CONVERT_RGB. Take one channel rather than a
            # weighted conversion: the sensor is mono, so the three are equal and
            # a luma weighting would only add rounding.
            frame = frame[:, :, 0]
        if frame.dtype != np.uint8:
            raise CameraError(f'expected 8-bit frames, got {frame.dtype}')
        try:
            full = frames.crop_window(np.ascontiguousarray(frame), self._crop_top)
        except frames.FrameError as e:
            raise CameraError(
                f'{self._device} delivered {frame.shape}, which is not the '
                f'{frames.ACTIVE} active area or the {frames.FULL} window: {e}') from e
        # VideoCapture exposes no capture clock we can trust, so the node stamps
        # with the ROS clock at read.
        return full, None

    def stop(self) -> None:
        cap, self._cap = self._cap, None
        if cap is None:
            return
        try:
            cap.release()
        except Exception as e:      # noqa: BLE001 - teardown must not raise
            self._log.error(f'releasing {self._device} failed: {e}')
