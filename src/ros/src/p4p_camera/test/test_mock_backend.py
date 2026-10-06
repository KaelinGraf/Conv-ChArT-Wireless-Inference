"""The CameraBackend contract, exercised against the mock.

Every real backend has to honour the same contract, and the one clause that is
easy to break and impossible to see is the ownership of the returned array. That
is tested directly here rather than left to review.
"""
import time

import cv2
import numpy as np
from p4p_camera import frames
from p4p_camera.backends import CameraError, make_backend
from p4p_camera.backends.mock import expected_stream_markers
import pytest


class _Log:
    """Minimal stand-in for a rclpy logger."""

    def __init__(self):
        self.lines = []

    def _rec(self, msg, **_kw):
        self.lines.append(str(msg))

    info = warning = error = debug = _rec


def _backend(**params):
    params.setdefault('frame_rate', 1000.0)      # do not pace the tests
    return make_backend('mock', params, _Log())


# --------------------------------------------------------------------------- #
# the factory                                                                 #
# --------------------------------------------------------------------------- #

def test_unknown_backend_is_a_camera_error_naming_the_options():
    with pytest.raises(CameraError, match='unknown camera_backend'):
        make_backend('webcam', {}, _Log())


def test_factory_never_substitutes_a_backend():
    """A typo must not quietly become the mock, and vice versa."""
    assert _backend().name == 'mock'


# --------------------------------------------------------------------------- #
# the contract                                                                #
# --------------------------------------------------------------------------- #

def test_read_returns_a_full_uint8_frame_and_a_boottime_stamp():
    b = _backend()
    b.start()
    try:
        before = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
        frame, stamp = b.read(1.0)
        after = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    finally:
        b.stop()
    assert frame.shape == frames.FULL
    assert frame.dtype == np.uint8
    assert frame.ndim == 2
    assert before <= stamp <= after


def test_consecutive_frames_differ():
    """Otherwise a stuck stream would look healthy to every other test."""
    b = _backend()
    b.start()
    try:
        a, _ = b.read(1.0)
        c, _ = b.read(1.0)
    finally:
        b.stop()
    assert not np.array_equal(a, c)


def test_returned_frames_are_owned_by_the_caller():
    """The core aliasing clause: a real camera recycles the buffer we read from.

    Mutating one frame must not change the backend's own state or any later
    frame. With a view into a DMA buffer this passes nothing.
    """
    b = _backend()
    b.start()
    try:
        first, _ = b.read(1.0)
        first[:] = 0
        second, _ = b.read(1.0)
    finally:
        b.stop()
    assert second.max() > 0
    assert not np.shares_memory(first, second)


def test_stop_is_idempotent_and_does_not_raise():
    b = _backend()
    b.start()
    b.stop()
    b.stop()            # must not raise
    b.stop()


def test_read_before_start_is_a_camera_error():
    with pytest.raises(CameraError):
        _backend().read(1.0)


def test_read_after_stop_is_a_camera_error():
    b = _backend()
    b.start()
    b.stop()
    with pytest.raises(CameraError):
        b.read(1.0)


def test_read_raises_timeout_when_the_next_frame_is_too_far_off():
    """The node distinguishes a stalled stream from a broken one by this type."""
    b = _backend(frame_rate=1.0)        # one frame per second
    b.start()
    try:
        b.read(1.0)                      # the first is due immediately
        with pytest.raises(TimeoutError):
            b.read(0.05)                 # the second is ~1 s away
    finally:
        b.stop()


# --------------------------------------------------------------------------- #
# mock_latency_ms -- exercises the sensor-stamp mapping with no hardware       #
# --------------------------------------------------------------------------- #

def test_mock_latency_back_dates_the_capture_stamp():
    b = _backend(mock_latency_ms=40.0)
    b.start()
    try:
        _frame, stamp = b.read(1.0)
        now_boot = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    finally:
        b.stop()
    latency_s = (now_boot - stamp) / 1e9
    assert 0.035 < latency_s < 0.10, latency_s

    # and the node's mapping turns that into a back-dated ROS stamp it trusts
    now_ros = 1_700_000_000 * 1_000_000_000
    _stamp, measured, trusted = frames.ros_stamp_ns(stamp, now_ros, now_boot)
    assert trusted
    assert measured == pytest.approx(latency_s, abs=1e-3)


# --------------------------------------------------------------------------- #
# markers: the fixture the graph test asserts on                              #
# --------------------------------------------------------------------------- #

def test_markers_survive_crop_downscale_png_exactly():
    """If this fails, the graph test's pixel assertions are meaningless."""
    b = _backend()
    b.start()
    try:
        full, _ = b.read(1.0)
    finally:
        b.stop()
    stream = frames.downscale(frames.crop_window(full))
    data = frames.encode_png(stream)
    back = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_ANYCOLOR)
    assert back.ndim == 2
    markers = expected_stream_markers()
    assert markers, 'the fixture must assert on at least one marker'
    for x, y, value in markers:
        assert back[y, x] == value, f'marker at ({x}, {y})'


# --------------------------------------------------------------------------- #
# mock_image replay                                                           #
# --------------------------------------------------------------------------- #

def test_mock_image_is_replayed_byte_for_byte(tmp_path):
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, size=frames.FULL, dtype=np.uint8)
    path = tmp_path / 'board.png'
    cv2.imwrite(str(path), img)

    b = _backend(mock_image=str(path))
    b.start()
    try:
        frame, _ = b.read(1.0)
    finally:
        b.stop()
    assert np.array_equal(frame, img)


def test_missing_mock_image_fails_at_start_not_at_import(tmp_path):
    """This is also how the node's 'camera failed to open' branch is tested."""
    b = _backend(mock_image=str(tmp_path / 'nope.png'))
    with pytest.raises(CameraError, match='does not exist'):
        b.start()


def test_mock_image_of_the_wrong_size_is_refused(tmp_path):
    path = tmp_path / 'small.png'
    cv2.imwrite(str(path), np.zeros((480, 640), dtype=np.uint8))
    b = _backend(mock_image=str(path))
    with pytest.raises(CameraError, match='expected'):
        b.start()
