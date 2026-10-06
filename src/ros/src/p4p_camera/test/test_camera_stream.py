"""Graph-level tests for the 640x480 stream on `image`.

Runs the real node against the mock backend on an isolated domain, so it needs
no camera. The single most valuable assertion here is the QoS one: a publisher
that drifts outside what convchart_qos requests does not error, it just goes
silent, and this is the test that turns that into a hard failure.
"""
import cv2
import numpy as np
from p4p_camera import frames
from p4p_camera.backends.mock import expected_stream_markers


def _decode(msg):
    """Decode as cv_bridge would: IMREAD_ANYCOLOR, no ANYDEPTH."""
    return cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_ANYCOLOR)


def test_publisher_qos_matches_the_inference_node(rig):
    """The one that stops a silent link. Everything else assumes this passes."""
    rig.wait_for_images(1)
    rig.check_qos()


def test_format_string_is_exactly_what_consumers_expect(rig):
    msg = rig.wait_for_images(1)[0]
    assert msg.format == 'mono8; png compressed mono8'
    assert msg.format == frames.PNG_FORMAT


def test_payload_decodes_to_a_two_dimensional_640x480_uint8_frame(rig):
    """Guard inference._to_mono's assert: a 3-channel PNG would break it.

    ndim == 2 is the clause that matters.
    """
    msg = rig.wait_for_images(1)[0]
    img = _decode(msg)
    assert img is not None, 'the payload is not a decodable PNG'
    assert img.ndim == 2
    assert img.shape == frames.STREAM == (480, 640)
    assert img.dtype == np.uint8


def test_stream_is_lossless_end_to_end(rig):
    """Marker pixels must survive crop -> downscale -> PNG -> publish -> decode.

    The refiner reads sub-pixel offsets out of 24x24 crops, so any lossy step in
    this chain would bias the pose rather than merely soften the picture.
    """
    msg = rig.wait_for_images(1)[0]
    img = _decode(msg)
    markers = expected_stream_markers()
    assert markers
    for x, y, value in markers:
        assert img[y, x] == value, f'marker at ({x}, {y}) came back as {img[y, x]}'


def test_header_carries_the_frame_id_and_a_nonzero_capture_stamp(rig):
    msg = rig.wait_for_images(1)[0]
    assert msg.header.frame_id == 'pi_camera'
    assert (msg.header.stamp.sec, msg.header.stamp.nanosec) != (0, 0)


def test_stamps_are_strictly_increasing_and_never_repeat(rig):
    """A repeated stamp would make the inference node process one frame twice."""
    msgs = rig.wait_for_images(8)
    stamps = [m.header.stamp.sec * 10**9 + m.header.stamp.nanosec for m in msgs]
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == len(stamps), f'duplicate stamps: {stamps}'


def test_frames_differ_so_a_stuck_stream_cannot_pass(rig):
    msgs = rig.wait_for_images(4)
    payloads = {bytes(m.data) for m in msgs}
    assert len(payloads) > 1, 'every published frame was byte-identical'


def test_stream_sustains_its_configured_rate(make_rig):
    """Bounded loosely: this has to survive QEMU and a loaded CI box."""
    rig = make_rig(frame_rate=10.0)
    rig.wait_for_images(1)                     # discovery and first frame
    rig.images.clear()
    rig.spin(3.0)
    rig.check_qos()
    assert len(rig.images) >= 15, (
        f'only {len(rig.images)} frames in 3 s at a nominal 10 Hz')


def test_camera_info_is_published_at_the_stream_resolution(rig):
    """K must live at 640x480 now, and this is where that becomes visible."""
    rig.wait_for_images(2)
    rig.spin(0.5)
    assert rig.infos, 'no camera_info published'
    info = rig.infos[-1]
    assert (info.width, info.height) == (frames.STREAM[1], frames.STREAM[0])
    # Uncalibrated is the honest default: the ROS convention for "no calibration"
    # is a zeroed k, which image_pipeline already reads as such. A plausible but
    # wrong K would be worse than none.
    assert float(info.k[0]) == 0.0


def test_camera_info_shares_a_stamp_with_an_image(rig):
    rig.wait_for_images(3)
    rig.spin(0.3)
    image_stamps = {(m.header.stamp.sec, m.header.stamp.nanosec) for m in rig.images}
    info_stamps = {(m.header.stamp.sec, m.header.stamp.nanosec) for m in rig.infos}
    assert image_stamps & info_stamps, 'camera_info is not paired with any frame'


def test_sensor_timestamp_is_back_dated_by_the_capture_latency(make_rig):
    """The no-hardware exercise of frames.ros_stamp_ns in the live node.

    mock_latency_ms makes the mock report a capture clock 40 ms in the past, as a
    real sensor does. The published stamp must therefore sit behind the time the
    probe receives it -- if the node stamped at publish, this gap would vanish
    and the filter would be told the pose is fresher than it is.
    """
    rig = make_rig(mock_latency_ms=40.0, frame_rate=10.0)
    msgs = rig.wait_for_images(3)
    now = rig.probe.get_clock().now().nanoseconds
    ages = [(now - (m.header.stamp.sec * 10**9 + m.header.stamp.nanosec)) / 1e9
            for m in msgs]
    assert min(ages) > 0.030, f'stamps are not back-dated: ages {ages}'
    assert min(ages) < 1.0, f'stamps are implausibly old: ages {ages}'


def test_stamping_with_the_ros_clock_can_be_forced(make_rig):
    """use_sensor_timestamp:=false pins stamps to the clock at read, for debugging."""
    rig = make_rig(mock_latency_ms=200.0, use_sensor_timestamp=False, frame_rate=10.0)
    msgs = rig.wait_for_images(3)
    now = rig.probe.get_clock().now().nanoseconds
    ages = [(now - (m.header.stamp.sec * 10**9 + m.header.stamp.nanosec)) / 1e9
            for m in msgs]
    # The mock claims 200 ms of latency; ignoring it must keep the stamps recent.
    assert min(ages) < 0.150, f'the sensor latency leaked into the stamp: {ages}'


def test_a_sensor_running_faster_than_frame_rate_does_not_raise_the_publish_rate(
        make_rig):
    """The backstop for a sensor that ignores FrameDurationLimits.

    picamera2 is asked to pace the OV2311 at frame_rate, but that is a request,
    not a guarantee -- and the Arducam fork with the ISP bypassed is exactly the
    kind of build that might ignore it. mock_rate stands in for that: the backend
    delivers at 60 Hz while the node was asked for 10, and the published rate must
    still be ~10, because 60 Hz of 640x480 PNG is six times the bandwidth budget
    this node exists to protect.
    """
    rig = make_rig(frame_rate=10.0, mock_rate=60.0)
    rig.spin(2.0)
    rig.check_qos()
    assert len(rig.images) >= 8, (
        f'only {len(rig.images)} frames in 2 s; the guard is dropping too much')
    assert len(rig.images) <= 30, (
        f'{len(rig.images)} frames in 2 s at a nominal 10 Hz: the surplus guard '
        'is not limiting the publish rate, so an unpaced sensor would saturate '
        'the link')
    assert rig.node._surplus > 0, (
        'the node reported no surplus frames, so this test did not actually '
        'present a fast sensor and proves nothing')


def test_the_guard_does_not_throttle_a_correctly_paced_sensor(make_rig):
    """The other half: the guard must be inert when the sensor behaves.

    A guard that also trims normal jitter would quietly halve the frame rate, so
    this pins the fraction below 1.0 rather than at it.
    """
    rig = make_rig(frame_rate=10.0)
    rig.spin(2.0)
    rig.check_qos()
    # The sharp assertion is the surplus count, not the frame count: it is exact
    # and immune to a loaded CI box, and it is what catches the subtle version of
    # this bug. Timing the guard from the end of the previous publish instead of
    # between arrivals subtracts our own encode cost from the period, so every
    # other on-time frame looks early -- the rate halves while staying well
    # inside any tolerance a frame count could reasonably use.
    assert rig.node._surplus <= 2, (
        f'{rig.node._surplus} frames dropped as surplus at the nominal rate; the '
        'guard is tripping on frames that arrived on time')
    assert len(rig.images) >= 12, (
        f'only {len(rig.images)} frames in 2 s at 10 Hz; the surplus guard is '
        'eating frames it should pass')
