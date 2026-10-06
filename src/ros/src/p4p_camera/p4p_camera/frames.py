"""Frame geometry, raw-buffer unpacking and stamp arithmetic for the OV2311.

No ROS and no camera in here on purpose: this is the part most likely to be
wrong, and all of it is reachable from a plain pytest with nothing but numpy and
cv2 installed. The driver modules under backends/ are thin wrappers that call
into here, so that the code which cannot be tested without the sensor contains
as little arithmetic as possible.

The geometry, once:

    1600x1300   OV2311 active area, monochrome, global shutter
      -> crop 50 rows off the top and bottom
    1600x1200   the 4:3 window we calibrate and serve on the full-res service
      -> INTER_AREA downscale by exactly 2.5
     640x480    the stream on `image`, and exactly the detector's input size,
                so inference.py's own resize becomes the identity

The 2.5 is load-bearing. It is what makes the intrinsics mapping exact and what
makes `r = W_in / Ws` come out at 1.0 in inference.run_inference, so changing
either resolution independently silently invalidates the calibration.
"""
from __future__ import annotations

import re

import cv2
import numpy as np

# (H, W) throughout, matching numpy's axis order rather than the camera's.
ACTIVE = (1300, 1600)
FULL = (1200, 1600)
STREAM = (480, 640)
SCALE = 2.5
CROP_TOP = (ACTIVE[0] - FULL[0]) // 2      # 50

# The format string on both `image` and the full-res service response. cv_bridge
# ignores it and sniffs the PNG header, but image_transport consumers do not, and
# test_node_pose_output.py already publishes exactly this.
PNG_FORMAT = 'mono8; png compressed mono8'

# Bayer-patterned sensor formats. Seeing one of these from a module that is
# supposed to be the mono OV2311 means the overlay or the camera is wrong, and
# debayering it would produce a plausible-looking image off the wrong pixels.
_BAYER_RE = re.compile(r'^S(BGGR|GBRG|GRBG|RGGB)')


class FrameError(ValueError):
    """A frame, buffer or sensor mode that cannot be used as configured."""


def crop_window(frame: np.ndarray, crop_top: int = CROP_TOP) -> np.ndarray:
    """Centre-crop the active area down to the 4:3 full-res window.

    Idempotent on a frame that is already FULL height, so a backend whose sensor
    did the windowing itself can call this unconditionally.

    The returned array does not share memory with the input: callers hand it
    straight to another thread, and a row slice of a recycled DMA buffer would
    tear. See CameraBackend.read's contract.

    .copy() and not np.ascontiguousarray(): the latter is a no-op that hands back
    the input when it is already contiguous, which a row slice of a contiguous
    frame is. It would satisfy the type and silently break the ownership.
    """
    h, w = frame.shape[:2]
    if w != FULL[1]:
        raise FrameError(f'expected {FULL[1]} columns, got {w}')
    if h == FULL[0]:
        return frame.copy()
    if h != ACTIVE[0]:
        raise FrameError(
            f'expected {ACTIVE[0]} rows (active) or {FULL[0]} rows (pre-windowed), got {h}')
    if not 0 <= crop_top <= h - FULL[0]:
        raise FrameError(f'crop_top {crop_top} does not fit {h} rows into {FULL[0]}')
    return frame[crop_top:crop_top + FULL[0]].copy()


def downscale(full: np.ndarray) -> np.ndarray:
    """FULL -> STREAM by exactly SCALE, with INTER_AREA.

    INTER_AREA and not INTER_LINEAR: at a 2.5x reduction a bilinear kernel only
    reads 2 of every 5 source pixels per axis, which aliases the board's corners
    -- the exact feature the detector is looking for.
    """
    if full.shape[:2] != FULL:
        raise FrameError(f'expected a {FULL} frame, got {full.shape[:2]}')
    return cv2.resize(full, (STREAM[1], STREAM[0]), interpolation=cv2.INTER_AREA)


def encode_png(frame: np.ndarray, level: int = 1) -> bytes:
    """PNG-encode a mono frame losslessly.

    Lossless is not a preference here: the refiner reads sub-pixel offsets out of
    24x24 crops, so JPEG artefacts would bias the pose rather than merely blur it
    (docker/README.md spells this out). `level` therefore only ever trades CPU for
    bytes, never quality, and 1 is the cheap end.
    """
    if frame.dtype != np.uint8:
        raise FrameError(f'expected uint8, got {frame.dtype}')
    if frame.ndim != 2:
        raise FrameError(f'expected a 2-D mono frame, got shape {frame.shape}')
    ok, buf = cv2.imencode('.png', frame, [cv2.IMWRITE_PNG_COMPRESSION, int(level)])
    if not ok:
        raise FrameError('cv2.imencode refused the frame')
    return buf.tobytes()


def scale_intrinsics(K: np.ndarray, scale: float = SCALE,
                     crop_top: int = CROP_TOP) -> np.ndarray:
    """Map intrinsics from the active area to the streamed 640x480 frame.

    NOT simply K / scale. Under OpenCV's pixel-centre convention -- which
    inference.py itself uses, at `xy_coarse = (xy_pk + 0.5) / r - 0.5` -- pixel i
    spans [i, i+1) with its centre at i+0.5, so an area downscale maps

        x_stream = (x_full + 0.5) / scale - 0.5

    which gives fx/scale for the focal lengths but (cx + 0.5)/scale - 0.5 for the
    principal point. At scale 2.5 that is cx/2.5 - 0.3, and the naive cx/2.5 is
    off by 0.3 px. That is under the detector's coarse grid but well inside the
    refiner's sub-pixel budget, and it costs nothing to get right.

    The crop is applied first and only shifts cy. Distortion coefficients are
    dimensionless in normalised coordinates and carry over untouched.
    """
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3):
        raise FrameError(f'expected a 3x3 camera matrix, got {K.shape}')
    out = K.copy()
    out[0, 0] = K[0, 0] / scale
    out[1, 1] = K[1, 1] / scale
    out[0, 2] = (K[0, 2] + 0.5) / scale - 0.5
    out[1, 2] = (K[1, 2] - crop_top + 0.5) / scale - 0.5
    # A skew term scales with the row axis it shears along.
    out[0, 1] = K[0, 1] / scale
    return out


def unpack_raw8(padded: np.ndarray, width: int) -> np.ndarray:
    """Strip line padding from an 8-bit raw buffer.

    Raw CSI buffers are padded to a hardware stride, so the array handed back by
    the driver is wider than the image. Reshaping it to (h, width) instead of
    (h, stride) shears the picture progressively down the frame -- a diagonal
    smear, which is the signature to look for on first light.
    """
    if padded.ndim != 2:
        raise FrameError(f'expected a (h, stride) buffer, got shape {padded.shape}')
    if padded.shape[1] < width:
        raise FrameError(f'stride {padded.shape[1]} is narrower than width {width}')
    return padded[:, :width].copy()


def unpack_raw10p(padded: np.ndarray, width: int) -> np.ndarray:
    """MIPI RAW10 -> uint8, which is an exact >> 2 and costs no arithmetic.

    RAW10 packs four pixels into five bytes: bytes 0-3 hold the HIGH 8 bits of
    each pixel and byte 4 holds the four pairs of low bits. So discarding byte 4
    of every group is exactly `pixel >> 2` -- no shifting, masking or rounding.

    There is deliberately no 16-bit intermediate. The contract is uint8 end to
    end, and cv_bridge decodes with IMREAD_ANYCOLOR and no ANYDEPTH, so a 16-bit
    PNG would be silently truncated to 8 bits on the laptop anyway.
    """
    if padded.ndim != 2:
        raise FrameError(f'expected a (h, stride) buffer, got shape {padded.shape}')
    h, stride = padded.shape
    groups = (width + 3) // 4
    if stride < groups * 5:
        raise FrameError(
            f'stride {stride} cannot hold {groups} RAW10 groups ({groups * 5} bytes) '
            f'for width {width}')
    highs = padded[:, :groups * 5].reshape(h, groups, 5)[:, :, :4]
    return highs.reshape(h, groups * 4)[:, :width].copy()


def pick_mode(sensor_modes, want=FULL, prefer_8bit: bool = True) -> dict:
    """Choose the sensor mode to configure, from Picamera2.sensor_modes.

    Pure over the list of dicts picamera2 hands back, so it can be tested against
    a mode list recorded off the real camera and pasted in as a literal.

    Order of preference: 8-bit before 10-bit when prefer_8bit (an 8-bit mode
    halves CSI bandwidth, and RAW10's leading byte is the same pixel, so neither
    path loses anything we keep), then the smallest frame that still covers the
    window (less to read out and to crop), then the highest advertised fps.

    Returns the mode dict with two keys added: 'format_str' and 'crop_top', the
    latter being 0 when the sensor can window to `want` itself.
    """
    if not sensor_modes:
        raise FrameError('the camera reported no sensor modes at all')

    bayer = [m for m in sensor_modes if _BAYER_RE.match(str(m.get('format', '')))]
    if bayer and len(bayer) == len(sensor_modes):
        raise FrameError(
            'every sensor mode is Bayer '
            f'({sorted({str(m["format"]) for m in bayer})}); this is not the mono '
            'OV2311. Check dtoverlay=arducam-pivariety and that the right module '
            'is on the FPC')

    want_h, want_w = want
    usable = []
    for m in sensor_modes:
        fmt = str(m.get('format', ''))
        if _BAYER_RE.match(fmt):
            continue
        size = tuple(m.get('size', ()))
        if len(size) != 2:
            continue
        w, h = size                      # picamera2 reports (width, height)
        if w != want_w or h < want_h:
            continue
        usable.append((m, fmt, w, h))

    if not usable:
        raise FrameError(
            f'no mono sensor mode covers {want_w}x{want_h}; the camera offers '
            f'{[(str(m.get("format")), tuple(m.get("size", ()))) for m in sensor_modes]}')

    def rank(item):
        m, _fmt, _w, h = item
        depth = int(m.get('bit_depth', 8))
        depth_key = (depth != 8) if prefer_8bit else 0
        return (depth_key, h, -float(m.get('fps', 0.0) or 0.0))

    mode, fmt, _w, h = min(usable, key=rank)
    out = dict(mode)
    out['format_str'] = fmt
    # A sensor-side window that is not vertically centred would move cy, so take
    # the offset the camera reports rather than assuming our own CROP_TOP.
    if h == want_h:
        out['crop_top'] = 0
    else:
        out['crop_top'] = (h - want_h) // 2
    return out


def ros_stamp_ns(sensor_ns, now_ros_ns: int, now_boot_ns: int,
                 max_latency_s: float = 0.5):
    """Back-date a ROS stamp to the sensor's capture time.

    Returns (stamp_ns, latency_s, trusted).

    The stamp is copied verbatim into RosInferenceResult.header and will be fused
    against drive/telemetry, which the serial bridge stamps from the ROS clock.
    A pose's true age is exposure + CSI + encode + WiFi + inference, easily
    150-300 ms, so stamping at publish would claim the measurement is far fresher
    than it is -- exactly the error that makes a predictor overshoot.

    libcamera reports SensorTimestamp on CLOCK_BOOTTIME; the ROS clock is
    CLOCK_REALTIME. The offset is recomputed every frame rather than latched at
    startup because NTP steps and slews REALTIME against BOOTTIME, and a one-off
    offset would rot silently over a long run.

    An implausible latency -- negative, or beyond max_latency_s -- means the two
    clocks are not what we think they are, so the caller is told not to trust it
    and should stamp with now() instead of guessing.
    """
    if sensor_ns is None:
        return now_ros_ns, 0.0, False
    latency_ns = int(now_boot_ns) - int(sensor_ns)
    latency_s = latency_ns / 1e9
    if not 0.0 <= latency_s <= max_latency_s:
        return now_ros_ns, latency_s, False
    return int(now_ros_ns) - latency_ns, latency_s, True
