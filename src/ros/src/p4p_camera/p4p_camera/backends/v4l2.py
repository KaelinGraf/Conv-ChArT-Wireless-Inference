"""V4L2 fallback via OpenCV, for when libcamera is not an option.

SCOPE, honestly: on a Raspberry Pi 5 the CSI path runs through rp1-cfe with a
media-controller graph that libcamera sets up, so opening /dev/video0 directly
will generally NOT produce OV2311 frames without media-ctl work that this module
does not do. Do not reach for this expecting a drop-in replacement for picam.py
on this hardware.

What it is actually for:

  * a USB/UVC mono camera during development, where it works as-is;
  * a Pi-4-style unicam path, where /dev/video0 is the sensor;
  * the escape route if Arducam's libcamera build will not import under this
    image's python (their .debs target Debian bookworm / python 3.11, the ROS
    Jazzy base is Ubuntu noble / python 3.12). In that case this module plus a
    media-ctl setup script is the path that needs no userspace libcamera at all.

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
        self._frame_rate = float(params.get('frame_rate', 10.0))
        self._cap = None

    def start(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self._device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            raise CameraError(
                f'cannot open {self._device} with the V4L2 backend. On a Pi 5 the '
                'CSI sensor is behind a media-controller graph that libcamera '
                'configures, so a bare open often will not work -- see this '
                "module's docstring.")
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
            # "broken", so treat it as a stall and let the node reopen.
            raise TimeoutError(f'no frame from {self._device}')
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
