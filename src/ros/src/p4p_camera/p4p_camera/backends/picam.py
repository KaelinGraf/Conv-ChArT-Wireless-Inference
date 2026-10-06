"""picamera2 backend: the raw mono stream, with the ISP bypassed.

Not the default: on the Pi it needs Arducam's libcamera, which ships for
Raspberry Pi OS only and cannot be installed in this Ubuntu image, so v4l2 is
the default there (see backends/v4l2.py). It is also the module that cannot be
tested without the sensor. All of its arithmetic therefore lives in frames.py,
which can; what is left here is configuration and buffer handling.

Why the RAW stream and not the processed one: the OV2311 is monochrome and has no
colour filter array, so there is nothing for a debayering ISP to do. Going
through it would only let the missing arducam-pivariety_mono.json tuning file
(libcamera reports it as absent for the rpi/pisp IPA on a Pi 5, and Arducam
document that as benign) apply colour, gamma or sharpening to the exact pixels
the refiner reads sub-pixel offsets out of.

Recorded sensor modes go here once the camera is attached, as the fixture for
frames.pick_mode's tests:

    # rpicam-hello --list-cameras
    # v4l2-ctl -d /dev/video0 --list-formats-ext
    # python3 -c "from picamera2 import Picamera2; print(Picamera2().sensor_modes)"
    (not yet recorded -- see the bring-up checklist)
"""
from __future__ import annotations

import numpy as np

from . import CameraBackend, CameraError
from .. import frames


class Picamera2Backend(CameraBackend):
    """One Picamera2 instance streaming raw mono frames."""

    name = 'picamera2'

    def __init__(self, params: dict, logger):
        self._log = logger
        self._index = int(params.get('camera_index', 0))
        self._prefer_8bit = bool(params.get('prefer_8bit', True))
        self._crop_top = int(params.get('crop_top', frames.CROP_TOP))
        self._buffer_count = int(params.get('buffer_count', 4))
        self._frame_rate = float(params.get('frame_rate', 15.0))
        self._exposure_us = int(params.get('exposure_time_us', 0))
        self._gain = float(params.get('analogue_gain', 0.0))
        self._cam = None
        self._raw_cfg: dict | None = None
        self._bit_depth = 8
        self._unbounded_wait = False

    def start(self) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise CameraError(
                f'picamera2 is not importable ({e}). On the Pi this needs '
                "Arducam's libcamera build and their picamera2 fork, which ship "
                'for Raspberry Pi OS only and cannot install in this Ubuntu image '
                '(see the WITH_CAMERA block in docker/pi/Dockerfile). Use the '
                'default v4l2 backend.') from e

        try:
            cam = Picamera2(self._index)
        except Exception as e:      # noqa: BLE001
            raise CameraError(f'cannot open camera {self._index}: {e}') from e

        try:
            modes = cam.sensor_modes
            mode = frames.pick_mode(modes, prefer_8bit=self._prefer_8bit)
            self._bit_depth = int(mode.get('bit_depth', 8))
            self._crop_top = int(mode.get('crop_top', self._crop_top))

            cfg = cam.create_video_configuration(
                raw={'format': mode['format_str'], 'size': tuple(mode['size'])},
                main={'size': (320, 240), 'format': 'YUV420'},
                sensor={'output_size': tuple(mode['size']),
                        'bit_depth': self._bit_depth},
                buffer_count=self._buffer_count,
                queue=False,
                controls=self._controls(),
            )
            cam.configure(cfg)
            self._raw_cfg = cam.camera_configuration()['raw']
            cam.start()
        except frames.FrameError:
            cam.close()
            raise
        except Exception as e:      # noqa: BLE001
            cam.close()
            raise CameraError(f'cannot configure camera {self._index}: {e}') from e

        self._cam = cam
        self._log.info(
            f'picamera2: {mode["format_str"]} {tuple(mode["size"])} '
            f'{self._bit_depth}-bit, stride {self._raw_cfg.get("stride")}, '
            f'crop_top {self._crop_top}')

    def _controls(self) -> dict:
        """Pace the sensor, and pin exposure when asked."""
        period_us = int(1e6 / max(self._frame_rate, 0.1))
        controls: dict = {'FrameDurationLimits': (period_us, period_us)}
        if self._exposure_us > 0 or self._gain > 0.0:
            controls['AeEnable'] = False
            controls['AwbEnable'] = False
            if self._exposure_us > 0:
                controls['ExposureTime'] = self._exposure_us
            if self._gain > 0.0:
                controls['AnalogueGain'] = self._gain
        return controls

    def stop(self) -> None:
        cam, self._cam = self._cam, None
        if cam is None:
            return
        for step in (cam.stop, cam.close):
            try:
                step()
            except Exception as e:      # noqa: BLE001
                self._log.error(f'picamera2 {step.__name__} failed: {e}')

    def read(self, timeout_s: float) -> tuple[np.ndarray, int | None]:
        if self._cam is None or self._raw_cfg is None:
            raise CameraError('picamera2 read before start')

        req = self._wait_for_request(timeout_s)
        try:
            width, height = self._raw_cfg['size']
            stride = int(self._raw_cfg['stride'])
            buf = req.make_buffer('raw')
            padded = np.asarray(buf, dtype=np.uint8).reshape(height, stride)
            if self._bit_depth == 8:
                mono = frames.unpack_raw8(padded, width)
            else:
                mono = frames.unpack_raw10p(padded, width)
            full = frames.crop_window(mono, self._crop_top)
            sensor_ns = req.get_metadata().get('SensorTimestamp')
        except frames.FrameError as e:
            raise CameraError(f'unusable raw buffer: {e}') from e
        except Exception as e:      # noqa: BLE001
            raise CameraError(f'raw capture failed: {e}') from e
        finally:
            req.release()
        return full, sensor_ns

    def _wait_for_request(self, timeout_s: float):
        """Fetch one request, with a bounded wait where the build supports it."""
        cam = self._cam
        job = cam.capture_request(wait=False)
        if not self._unbounded_wait:
            try:
                return cam.wait(job, timeout=timeout_s)
            except TypeError:
                self._unbounded_wait = True
                self._log.error(
                    'this picamera2 build does not support wait(timeout=...); '
                    'reads are UNBOUNDED. The watchdog can report a stalled '
                    'camera but cannot recover it; rely on the container restart '
                    'policy.')
            except TimeoutError:
                raise
            except Exception as e:      # noqa: BLE001
                raise CameraError(f'capture failed: {e}') from e
        try:
            return cam.wait(job)
        except Exception as e:      # noqa: BLE001
            raise CameraError(f'capture failed: {e}') from e
