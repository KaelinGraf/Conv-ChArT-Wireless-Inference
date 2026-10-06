"""Parameter validation: clamping, the derived deadline, and camera_info guards.

These are the places where a wrong value would otherwise fail silently -- a
publisher whose offered deadline drifts past the ceiling simply never matches,
and intrinsics for the wrong resolution are not detectable downstream.
"""
from p4p_camera import frames
import pytest
import yaml


def _deadline_s(node):
    qos = node._img_pub.qos_profile
    return qos.deadline.nanoseconds / 1e9


def test_ten_hertz_offers_one_and_a_half_periods(make_rig):
    rig = make_rig(frame_rate=10.0)
    assert _deadline_s(rig.node) == pytest.approx(0.15, abs=1e-6)


def test_a_rate_below_the_floor_is_clamped_rather_than_fatal(make_rig):
    """5 Hz is the floor: below it no offered deadline fits inside 0.2 s.

    The serial bridge's philosophy, deliberately reused -- a bot already on the
    floor is better off running with a safe value than not at all.
    """
    rig = make_rig(frame_rate=1.0)
    assert rig.node._frame_rate == pytest.approx(5.0)
    assert _deadline_s(rig.node) == pytest.approx(0.2, abs=1e-6)


def test_the_offered_deadline_never_exceeds_the_requested_ceiling(make_rig):
    """The whole point of the clamp: past 0.2 s the subscription stops matching."""
    from convchart_qos.qos import IMAGE_DEADLINE_CEILING_S
    for rate in (5.0, 6.0, 10.0, 30.0):
        rig = make_rig(frame_rate=rate)
        assert _deadline_s(rig.node) <= IMAGE_DEADLINE_CEILING_S + 1e-9, rate


def test_a_high_rate_still_offers_a_meetable_deadline(make_rig):
    rig = make_rig(frame_rate=30.0)
    offered = _deadline_s(rig.node)
    assert offered == pytest.approx(0.05, abs=1e-6)
    assert offered >= 1.0 / 30.0


def test_an_unknown_backend_is_fatal_rather_than_quietly_substituted(ros):  # noqa: ARG001
    """A typo in a launch file must not silently run the wrong backend."""
    from rclpy.parameter import Parameter

    from p4p_camera.camera_node import CameraNode

    with pytest.raises(ValueError, match='camera_backend'):
        CameraNode(parameter_overrides=[
            Parameter('camera_backend', value='webcam')])


# --------------------------------------------------------------------------- #
# camera_info_url                                                             #
# --------------------------------------------------------------------------- #

def _write_calibration(path, width, height, fx=500.0):
    path.write_text(yaml.safe_dump({
        'image_width': width,
        'image_height': height,
        'camera_name': 'pi_camera',
        'camera_matrix': {'rows': 3, 'cols': 3,
                          'data': [fx, 0.0, width / 2 - 0.5,
                                   0.0, fx, height / 2 - 0.5,
                                   0.0, 0.0, 1.0]},
        'distortion_model': 'plumb_bob',
        'distortion_coefficients': {'rows': 1, 'cols': 5,
                                    'data': [0.01, -0.02, 0.0, 0.0, 0.0]},
    }))
    return path


def test_a_640x480_calibration_is_loaded(make_rig, tmp_path):
    cal = _write_calibration(tmp_path / 'cal.yaml', 640, 480, fx=500.0)
    rig = make_rig(camera_info_url=f'file://{cal}')
    info = rig.node._info
    assert (info.width, info.height) == (640, 480)
    assert float(info.k[0]) == pytest.approx(500.0)
    assert float(info.d[0]) == pytest.approx(0.01)


def test_a_sensor_resolution_calibration_is_refused(make_rig, tmp_path):
    """The likeliest mistake available: calibrating at 1600x1200.

    Loading it would publish a K that is wrong by a factor of 2.5 with nothing
    downstream able to tell, so the file is ignored and the node publishes
    uncalibrated instead.
    """
    cal = _write_calibration(tmp_path / 'sensor.yaml', 1600, 1200, fx=1250.0)
    rig = make_rig(camera_info_url=f'file://{cal}')
    info = rig.node._info
    assert (info.width, info.height) == (640, 480)
    assert float(info.k[0]) == 0.0, 'wrong-resolution intrinsics were accepted'


def test_a_missing_calibration_file_does_not_stop_the_node(make_rig, tmp_path):
    rig = make_rig(camera_info_url=f'file://{tmp_path / "nope.yaml"}')
    assert float(rig.node._info.k[0]) == 0.0
    rig.wait_for_images(1)          # and the stream is unaffected


def test_a_malformed_calibration_file_does_not_stop_the_node(make_rig, tmp_path):
    bad = tmp_path / 'bad.yaml'
    bad.write_text('image_width: 640\nimage_height: 480\ncamera_matrix: oops\n')
    rig = make_rig(camera_info_url=f'file://{bad}')
    assert float(rig.node._info.k[0]) == 0.0
    rig.wait_for_images(1)


def test_a_bare_path_works_as_well_as_a_file_url(make_rig, tmp_path):
    cal = _write_calibration(tmp_path / 'bare.yaml', 640, 480, fx=321.0)
    rig = make_rig(camera_info_url=str(cal))
    assert float(rig.node._info.k[0]) == pytest.approx(321.0)


def test_camera_info_can_be_turned_off(make_rig):
    rig = make_rig(publish_camera_info=False)
    rig.wait_for_images(2)
    rig.spin(0.5)
    assert not rig.infos


# --------------------------------------------------------------------------- #
# the geometry the node reports                                               #
# --------------------------------------------------------------------------- #

def test_the_node_streams_the_detector_input_size(make_rig):
    """640x480 is not arbitrary: it makes inference.py's own resize the identity."""
    rig = make_rig()
    msg = rig.wait_for_images(1)[0]
    assert (rig.node._info.width, rig.node._info.height) == \
           (frames.STREAM[1], frames.STREAM[0])
    assert msg.format == frames.PNG_FORMAT
