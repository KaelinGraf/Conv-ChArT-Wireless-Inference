"""Unit tests for p4p_camera.frames: geometry, raw unpacking, stamp arithmetic.

No ROS and no camera, so this runs under a plain pytest with numpy and cv2.
Expectations are derived independently of the code under test wherever that is
possible -- the intrinsics test projects real 3-D points rather than restating
the formula, and the RAW10 test packs a buffer by hand.
"""
import cv2
import numpy as np
from p4p_camera import frames
import pytest


# --------------------------------------------------------------------------- #
# crop_window                                                                 #
# --------------------------------------------------------------------------- #

def test_crop_takes_the_centre_rows():
    """Active row CROP_TOP must land on full row 0, or cy is wrong by the error."""
    active = np.zeros(frames.ACTIVE, dtype=np.uint8)
    active[frames.CROP_TOP, :] = 200           # first kept row
    active[frames.CROP_TOP + frames.FULL[0] - 1, :] = 100   # last kept row
    out = frames.crop_window(active)
    assert out.shape == frames.FULL
    assert out[0, 0] == 200
    assert out[-1, 0] == 100
    # and the discarded rows really are gone
    assert out[1:-1].max() == 0


def test_crop_is_idempotent_on_a_prewindowed_frame():
    full = np.full(frames.FULL, 7, dtype=np.uint8)
    out = frames.crop_window(full)
    assert out.shape == frames.FULL
    assert np.array_equal(out, full)


def test_crop_does_not_alias_its_input():
    """read() hands the result to another thread; a view of a DMA buffer tears."""
    active = np.zeros(frames.ACTIVE, dtype=np.uint8)
    out = frames.crop_window(active)
    out[0, 0] = 255
    assert active[frames.CROP_TOP, 0] == 0
    assert not np.shares_memory(out, active)


@pytest.mark.parametrize('shape', [(1080, 1600), (1200, 1920), (480, 640)])
def test_crop_rejects_the_wrong_shape(shape):
    with pytest.raises(frames.FrameError):
        frames.crop_window(np.zeros(shape, dtype=np.uint8))


# --------------------------------------------------------------------------- #
# downscale                                                                   #
# --------------------------------------------------------------------------- #

def test_downscale_shape_and_dtype():
    out = frames.downscale(np.zeros(frames.FULL, dtype=np.uint8))
    assert out.shape == frames.STREAM
    assert out.dtype == np.uint8
    assert out.ndim == 2


def test_downscale_is_exactly_2_point_5_and_grid_aligned():
    """A piecewise-constant 10x10 block image must survive exactly.

    10 = 2.5 * 4, so every 10x10 input block maps onto an exact 4x4 output block.
    If the scale were not exactly 2.5, or the sampling grid were offset by even
    half a pixel, blocks would bleed into each other at the seams and this fails.
    A mean-preservation check would pass either way, which is why it is not the
    assertion used here.
    """
    h, w = frames.FULL
    by, bx = h // 10, w // 10
    rng = np.random.default_rng(0)
    blocks = rng.integers(0, 256, size=(by, bx), dtype=np.uint8)
    full = np.repeat(np.repeat(blocks, 10, axis=0), 10, axis=1)
    assert full.shape == frames.FULL

    out = frames.downscale(full)
    expected = np.repeat(np.repeat(blocks, 4, axis=0), 4, axis=1)
    assert np.array_equal(out, expected)


def test_downscale_rejects_the_wrong_input_size():
    with pytest.raises(frames.FrameError):
        frames.downscale(np.zeros(frames.ACTIVE, dtype=np.uint8))


# --------------------------------------------------------------------------- #
# encode_png                                                                  #
# --------------------------------------------------------------------------- #

def test_png_round_trip_is_lossless_and_two_dimensional():
    """The refiner reads sub-pixel offsets; a lossy or 3-channel decode is fatal."""
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, size=frames.STREAM, dtype=np.uint8)
    data = frames.encode_png(frame)

    # IMREAD_ANYCOLOR is the flag cv_bridge actually uses for a passthrough
    # decode, so it is what the inference node will see -- not IMREAD_UNCHANGED.
    back = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_ANYCOLOR)
    assert back.ndim == 2, 'a 3-channel decode would break inference._to_mono'
    assert back.shape == frames.STREAM
    assert back.dtype == np.uint8
    assert np.array_equal(back, frame)


def test_png_compression_level_changes_size_but_not_pixels():
    rng = np.random.default_rng(2)
    frame = rng.integers(0, 64, size=frames.STREAM, dtype=np.uint8)
    cheap = frames.encode_png(frame, level=1)
    dear = frames.encode_png(frame, level=9)
    assert len(dear) <= len(cheap)
    for data in (cheap, dear):
        back = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_ANYCOLOR)
        assert np.array_equal(back, frame)


@pytest.mark.parametrize('bad', [
    np.zeros(frames.STREAM, dtype=np.uint16),
    np.zeros((*frames.STREAM, 3), dtype=np.uint8),
])
def test_encode_png_rejects_non_mono_uint8(bad):
    with pytest.raises(frames.FrameError):
        frames.encode_png(bad)


# --------------------------------------------------------------------------- #
# scale_intrinsics                                                            #
# --------------------------------------------------------------------------- #

def test_scale_intrinsics_matches_projecting_through_the_mapping():
    """Independent reference: project points, then map pixels, and compare.

    The naive K/2.5 form fails this by 0.3 px in cx and cy, which is the whole
    point of the function existing.
    """
    K_active = np.array([[1800.0, 0.0, 812.3],
                         [0.0, 1795.0, 648.7],
                         [0.0, 0.0, 1.0]])
    K_stream = frames.scale_intrinsics(K_active)

    rng = np.random.default_rng(3)
    pts = np.column_stack([
        rng.uniform(-0.4, 0.4, 400),
        rng.uniform(-0.3, 0.3, 400),
        rng.uniform(0.8, 3.0, 400),
    ])

    # Project through the active-area intrinsics, crop, then map to the stream
    # grid with the pixel-centre rule, independently of scale_intrinsics.
    x_a = K_active[0, 0] * pts[:, 0] / pts[:, 2] + K_active[0, 2]
    y_a = K_active[1, 1] * pts[:, 1] / pts[:, 2] + K_active[1, 2]
    y_f = y_a - frames.CROP_TOP
    x_expect = (x_a + 0.5) / frames.SCALE - 0.5
    y_expect = (y_f + 0.5) / frames.SCALE - 0.5

    # Project the same points straight through the scaled intrinsics.
    x_got = K_stream[0, 0] * pts[:, 0] / pts[:, 2] + K_stream[0, 2]
    y_got = K_stream[1, 1] * pts[:, 1] / pts[:, 2] + K_stream[1, 2]

    assert np.allclose(x_got, x_expect, atol=1e-9)
    assert np.allclose(y_got, y_expect, atol=1e-9)


def test_scale_intrinsics_principal_point_is_not_the_naive_quotient():
    """Guard the 0.3 px specifically, so a 'simplification' trips a test."""
    K = np.array([[1000.0, 0.0, 800.0], [0.0, 1000.0, 650.0], [0.0, 0.0, 1.0]])
    out = frames.scale_intrinsics(K)
    assert out[0, 0] == pytest.approx(400.0)
    assert out[1, 1] == pytest.approx(400.0)
    assert out[0, 2] == pytest.approx(800.0 / 2.5 - 0.3)
    assert out[1, 2] == pytest.approx((650.0 - 50) / 2.5 - 0.3)
    assert out[0, 2] != pytest.approx(800.0 / 2.5)


def test_scale_intrinsics_rejects_a_non_3x3():
    with pytest.raises(frames.FrameError):
        frames.scale_intrinsics(np.eye(4))


# --------------------------------------------------------------------------- #
# raw unpacking                                                               #
# --------------------------------------------------------------------------- #

def test_unpack_raw8_strips_padding():
    h, w, stride = 8, 1600, 1664
    padded = np.zeros((h, stride), dtype=np.uint8)
    padded[:, :w] = 42
    padded[:, w:] = 0xFF                       # padding we must not return
    out = frames.unpack_raw8(padded, w)
    assert out.shape == (h, w)
    assert (out == 42).all()
    assert not np.shares_memory(out, padded)


def test_unpack_raw8_copies_even_when_there_is_no_padding():
    """Copy even at stride == width, where ascontiguousarray returns a view.

    An unpadded buffer is already contiguous, so the cheap call would hand back
    the DMA buffer itself and the frame would be overwritten under the consumer.
    """
    h, w = 4, 1600
    padded = np.zeros((h, w), dtype=np.uint8)
    out = frames.unpack_raw8(padded, w)
    out[0, 0] = 255
    assert padded[0, 0] == 0
    assert not np.shares_memory(out, padded)


def test_unpack_raw10p_copies_and_is_contiguous():
    w, h = 16, 3
    groups = (w + 3) // 4
    pixels = np.arange(h * w, dtype=np.uint16).reshape(h, w) % 1024
    buf = _pack_raw10(pixels, groups * 5)
    out = frames.unpack_raw10p(buf, w)
    out[0, 0] = 7
    assert not np.shares_memory(out, buf)
    assert out.flags['C_CONTIGUOUS']


def test_unpack_raw8_rejects_a_stride_narrower_than_the_image():
    with pytest.raises(frames.FrameError):
        frames.unpack_raw8(np.zeros((4, 100), dtype=np.uint8), 200)


def _pack_raw10(pixels: np.ndarray, stride: int) -> np.ndarray:
    """Pack 10-bit pixels into MIPI RAW10, independently of the unpacker."""
    h, w = pixels.shape
    groups = (w + 3) // 4
    padded_px = np.zeros((h, groups * 4), dtype=np.uint16)
    padded_px[:, :w] = pixels
    g = padded_px.reshape(h, groups, 4)
    buf = np.full((h, stride), 0xFF, dtype=np.uint8)     # padding is poison
    for i in range(4):
        buf[:, i::5][:, :groups] = (g[:, :, i] >> 2).astype(np.uint8)
    low = ((g[:, :, 0] & 3)
           | ((g[:, :, 1] & 3) << 2)
           | ((g[:, :, 2] & 3) << 4)
           | ((g[:, :, 3] & 3) << 6)).astype(np.uint8)
    buf[:, 4::5][:, :groups] = low
    return buf


@pytest.mark.parametrize('w', [1600, 1601, 1602, 1603, 8])
def test_unpack_raw10p_is_an_exact_shift(w):
    """Dropping the LSB byte must equal pixel >> 2, for widths off the group."""
    h = 6
    groups = (w + 3) // 4
    stride = groups * 5 + 16                  # deliberately padded
    rng = np.random.default_rng(4)
    pixels = rng.integers(0, 1024, size=(h, w), dtype=np.uint16)

    out = frames.unpack_raw10p(_pack_raw10(pixels, stride), w)

    assert out.shape == (h, w)
    assert out.dtype == np.uint8
    assert np.array_equal(out, (pixels >> 2).astype(np.uint8))
    assert not (out == 0xFF).all(), 'padding leaked into the image'


def test_unpack_raw10p_rejects_a_stride_that_cannot_hold_the_groups():
    with pytest.raises(frames.FrameError):
        frames.unpack_raw10p(np.zeros((4, 100), dtype=np.uint8), 1600)


# --------------------------------------------------------------------------- #
# pick_mode                                                                   #
# --------------------------------------------------------------------------- #

def _mode(fmt, size, bit_depth, fps):
    return {'format': fmt, 'size': size, 'bit_depth': bit_depth, 'fps': fps}


def test_pick_mode_prefers_8bit_then_falls_back_to_10():
    modes = [
        _mode('R10_CSI2P', (1600, 1300), 10, 60.0),
        _mode('R8', (1600, 1300), 8, 60.0),
    ]
    assert frames.pick_mode(modes)['bit_depth'] == 8
    assert frames.pick_mode(modes, prefer_8bit=False)['bit_depth'] == 10
    assert frames.pick_mode([modes[0]])['bit_depth'] == 10


def test_pick_mode_reports_zero_crop_for_a_native_4_3_window():
    native = frames.pick_mode([_mode('R8', (1600, 1200), 8, 60.0)])
    assert native['crop_top'] == 0
    full = frames.pick_mode([_mode('R8', (1600, 1300), 8, 60.0)])
    assert full['crop_top'] == frames.CROP_TOP


def test_pick_mode_rejects_an_all_bayer_camera():
    modes = [_mode('SBGGR10_CSI2P', (1600, 1300), 10, 60.0),
             _mode('SGRBG8', (1600, 1300), 8, 60.0)]
    with pytest.raises(frames.FrameError, match='mono OV2311'):
        frames.pick_mode(modes)


def test_pick_mode_rejects_modes_that_do_not_cover_the_window():
    with pytest.raises(frames.FrameError, match='covers'):
        frames.pick_mode([_mode('R8', (640, 480), 8, 120.0)])


def test_pick_mode_rejects_an_empty_mode_list():
    with pytest.raises(frames.FrameError):
        frames.pick_mode([])


def test_pick_mode_exposes_the_format_as_a_string():
    class SensorFormat:
        def __str__(self):
            return 'R8'
    picked = frames.pick_mode([_mode(SensorFormat(), (1600, 1300), 8, 60.0)])
    assert picked['format_str'] == 'R8'


# --------------------------------------------------------------------------- #
# ros_stamp_ns                                                                #
# --------------------------------------------------------------------------- #

NS = 1_000_000_000


def test_stamp_back_dates_by_the_measured_latency():
    now_ros, now_boot = 1_700_000_000 * NS, 5_000 * NS
    sensor = now_boot - 30_000_000                       # 30 ms ago
    stamp, latency, trusted = frames.ros_stamp_ns(sensor, now_ros, now_boot)
    assert trusted
    assert latency == pytest.approx(0.030)
    assert stamp == now_ros - 30_000_000


def test_stamp_is_distrusted_for_a_future_sensor_time():
    now_ros, now_boot = 1_700_000_000 * NS, 5_000 * NS
    stamp, latency, trusted = frames.ros_stamp_ns(now_boot + NS, now_ros, now_boot)
    assert not trusted
    assert latency < 0
    assert stamp == now_ros


def test_stamp_is_distrusted_past_the_latency_ceiling():
    now_ros, now_boot = 1_700_000_000 * NS, 5_000 * NS
    stamp, latency, trusted = frames.ros_stamp_ns(now_boot - 2 * NS, now_ros, now_boot,
                                                  max_latency_s=0.5)
    assert not trusted
    assert latency == pytest.approx(2.0)
    assert stamp == now_ros


def test_stamp_falls_back_when_the_backend_has_no_capture_clock():
    stamp, latency, trusted = frames.ros_stamp_ns(None, 123 * NS, 5 * NS)
    assert (stamp, latency, trusted) == (123 * NS, 0.0, False)


def test_stamp_offset_cancels_out():
    """A common shift in all three inputs must not move the latency."""
    base = frames.ros_stamp_ns(5_000 * NS - 20_000_000, 1_700_000_000 * NS, 5_000 * NS)
    shifted = frames.ros_stamp_ns(9_999 * NS - 20_000_000, 1_700_000_000 * NS, 9_999 * NS)
    assert base[1] == pytest.approx(shifted[1])
    assert base[0] == shifted[0]
