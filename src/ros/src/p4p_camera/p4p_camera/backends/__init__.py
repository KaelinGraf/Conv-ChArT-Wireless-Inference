"""Camera backends: one interface, three implementations, chosen by parameter.

The node talks only to CameraBackend, so the code that cannot be tested without
the sensor is confined to picam.py and v4l2.py. mock.py exists so that the whole
node -- geometry, encoding, the service, QoS negotiation -- is exercised by the
test suite and can be brought up on a bench with no camera attached.
"""
from __future__ import annotations

import abc
import importlib

import numpy as np

BACKENDS = {
    'picamera2': ('.picam', 'Picamera2Backend'),
    'v4l2': ('.v4l2', 'V4L2Backend'),
    'mock': ('.mock', 'MockBackend'),
}


class CameraError(RuntimeError):
    """The camera could not be opened, configured or read.

    Distinct from TimeoutError, which the node treats as a stalled stream rather
    than a broken one -- both lead to a reopen, but only this one is worth an
    error-level log on the first occurrence.
    """


class CameraBackend(abc.ABC):
    """One camera, owned by one thread.

    The node guarantees that only its capture thread ever calls start, read or
    stop. That is not a style preference: libcamera objects are not thread-safe,
    and a stop() racing a blocked read() inside libcamera is a segfault rather
    than an exception, which no amount of locking in Python can catch. The
    watchdog therefore asks the capture thread to restart the backend instead of
    touching it itself.

    Deliberately NOT part of this interface: resolution negotiation, pixel format
    and intrinsics. Every backend returns the same frames.FULL uint8 frame, so
    the node has exactly one shape to handle downstream.
    """

    name: str = 'backend'

    @abc.abstractmethod
    def start(self) -> None:
        """Open and configure the camera, and begin streaming.

        Raises CameraError on any failure. The node logs it, publishes nothing,
        and retries -- it never silently substitutes a different backend.
        """

    @abc.abstractmethod
    def read(self, timeout_s: float) -> tuple[np.ndarray, int | None]:
        """Return one frames.FULL uint8 frame and its capture time.

        The array is OWNED BY THE CALLER and must already be a copy. The camera
        recycles its DMA buffers the moment the request is released, so a view
        would be overwritten underneath the publisher and the full-res service.

        The second element is CLOCK_BOOTTIME nanoseconds as reported by the
        sensor, or None when the backend has no capture clock of its own and the
        node should stamp with the ROS clock instead.

        Raises CameraError on an I/O failure and TimeoutError when no frame
        arrives within timeout_s.
        """

    @abc.abstractmethod
    def stop(self) -> None:
        """Release the camera.

        Idempotent, and must not raise: it is called from the capture thread's
        finally and from the node's shutdown path, where an exception would cost
        us the rest of the teardown.
        """


def make_backend(name: str, params: dict, logger) -> CameraBackend:
    """Instantiate a backend by name, importing its module lazily.

    Lazy so that a machine without picamera2 can still import the node and run
    the test suite. A missing dependency surfaces from start() as a CameraError,
    where the node's retry loop handles it, rather than at import time.

    There is deliberately NO fallback to another backend. Synthetic frames
    feeding a pose estimate that steers a robot are worse than no frames at all,
    because they look exactly like a working camera. An unavailable backend is an
    error to retry, never a reason to invent data.
    """
    try:
        module_name, class_name = BACKENDS[name]
    except KeyError:
        raise CameraError(
            f'unknown camera_backend {name!r}; expected one of '
            f'{sorted(BACKENDS)}') from None
    module = importlib.import_module(module_name, __package__)
    return getattr(module, class_name)(params, logger)
