"""Graph-level tests for the image_full_res service."""
import cv2
import numpy as np
from p4p_camera import frames
from p4p_camera.backends.mock import expected_stream_markers


def _decode(msg):
    return cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_ANYCOLOR)


def test_service_returns_a_full_resolution_mono_png(rig):
    rig.wait_for_images(1)
    res = rig.call_full_res()
    assert res.success, res.message
    assert res.image.format == frames.PNG_FORMAT
    img = _decode(res.image)
    assert img is not None
    assert img.ndim == 2
    assert img.shape == frames.FULL == (1200, 1600)
    assert img.dtype == np.uint8


def test_capture_stamp_mirrors_the_image_header(rig):
    rig.wait_for_images(1)
    res = rig.call_full_res()
    assert res.success, res.message
    assert (res.capture_stamp.sec, res.capture_stamp.nanosec) == \
           (res.image.header.stamp.sec, res.image.header.stamp.nanosec)
    assert res.image.header.frame_id == 'pi_camera'


def test_the_served_frame_shares_the_streams_retained_frame(rig):
    """The real test of the shared latest-frame slot.

    Note what is NOT asserted: that the served stamp turns up on `image`. The
    publisher is depth 1 with a 0.15 s lifespan -- newest frame wins, and
    intermediate frames are dropped by design whenever a subscriber is not being
    spun continuously. Requiring a specific frame to arrive would be asserting
    something the QoS deliberately refuses to promise, and it flakes.

    What is asserted instead is stronger where it can be: the full-res frame the
    service returns must *downscale into* a stream frame. If the service encoded
    from its own capture, or from a different buffer, the geometry would not line
    up and the markers would blur. When the matching stream frame did arrive, the
    two are compared pixel for pixel.
    """
    rig.wait_for_images(4)
    res = rig.call_full_res()
    assert res.success, res.message
    served = (res.capture_stamp.sec, res.capture_stamp.nanosec)
    rig.spin(0.3)

    served_full = _decode(res.image)
    assert served_full.shape == frames.FULL

    # Same crop + downscale the stream path applies, so the markers must land on
    # exactly the pixels the stream puts them on.
    stream = frames.downscale(served_full)
    assert stream.shape == frames.STREAM
    markers = expected_stream_markers()
    assert markers
    for x, y, value in markers:
        assert stream[y, x] == value, (
            f'marker at ({x}, {y}) does not survive downscaling the served frame; '
            "the service is not sharing the stream's geometry")

    by_stamp = {(m.header.stamp.sec, m.header.stamp.nanosec): m for m in rig.images}
    assert by_stamp, 'no frames were streamed at all'
    # Same clock, same stream: the served stamp has to sit in the streamed range
    # rather than being generated independently by the service.
    low, high = min(by_stamp), max(by_stamp)
    assert low <= served <= high or served > high, (
        f'served stamp {served} is older than every streamed stamp (range '
        f'{low}..{high}); the service is not reading the live slot')

    match = by_stamp.get(served)
    if match is not None:
        assert bytes(frames.encode_png(stream)) == bytes(match.data), (
            'the served frame and the stream frame carrying the same stamp are '
            'not the same exposure')


def test_max_age_zero_accepts_whatever_is_retained(rig):
    rig.wait_for_images(1)
    assert rig.call_full_res(max_age=0.0).success


def test_an_impossibly_tight_max_age_is_refused_with_the_age_in_the_message(rig):
    rig.wait_for_images(1)
    res = rig.call_full_res(max_age=1e-6)
    assert not res.success
    assert 'old' in res.message
    assert res.image.format == ''
    assert len(res.image.data) == 0
    # The stamp still comes back so the caller can see how stale the frame is.
    assert (res.capture_stamp.sec, res.capture_stamp.nanosec) != (0, 0)


def test_repeated_calls_are_consistent(rig):
    """The encode cache must not change the answer, only its cost."""
    rig.wait_for_images(1)
    first = rig.call_full_res()
    second = rig.call_full_res()
    assert first.success and second.success
    if (first.capture_stamp.sec, first.capture_stamp.nanosec) == \
       (second.capture_stamp.sec, second.capture_stamp.nanosec):
        assert bytes(first.image.data) == bytes(second.image.data)


def test_service_reports_a_failure_instead_of_blocking_when_the_camera_will_not_open(
        make_rig, tmp_path):
    """A camera fault must not become an unresponsive node.

    A blocking service would hold an executor thread for the whole outage. The
    node instead answers at once with a diagnosis naming the open failure, which
    is the thing an operator can act on. Constructing the node must also not
    raise -- it starts, retries, and publishes nothing.
    """
    missing = tmp_path / 'not-a-frame.png'
    rig = make_rig(mock_image=str(missing))
    rig.spin(1.5)
    assert not rig.images, 'a camera that cannot open must publish nothing'
    res = rig.call_full_res()
    assert not res.success
    assert 'no frame captured yet' in res.message
    assert 'not-a-frame.png' in res.message or 'does not exist' in res.message
    assert len(res.image.data) == 0


def test_full_res_is_larger_than_the_stream_frame(rig):
    """Cheap sanity that the two paths are not accidentally the same image."""
    msg = rig.wait_for_images(1)[0]
    res = rig.call_full_res()
    assert res.success, res.message
    assert len(res.image.data) > len(msg.data)
