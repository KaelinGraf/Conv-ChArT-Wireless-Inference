# p4p_camera

Arducam **2MP OV2311 global-shutter mono** (Pivariety, NoIR) on a Raspberry Pi 5.
Owns the camera and nothing else: no filtering, no pose, no control.

| direction | name | type | notes |
|---|---|---|---|
| out | `image` | `sensor_msgs/CompressedImage` | 640×480 mono8 PNG, 10 Hz |
| out | `camera_info` | `sensor_msgs/CameraInfo` | intrinsics **for that 640×480** |
| srv | `image_full_res` | `convchart_interfaces/GetFullResImage` | newest 1600×1200 frame |

The topic is `image`, not `image_compressed`, because that is what
`convchart_ros/convchart.py` and `calibrate.py` already subscribe to. Remap it in
a launch file if you want another name.

## Geometry — the part that is load-bearing

```
1600x1300   OV2311 active area, monochrome, global shutter, up to 60 fps
  |  crop 50 rows off the top and bottom
1600x1200   the 4:3 window we calibrate and serve  -> image_full_res
  |  INTER_AREA downscale by exactly 2.5
 640x480    the stream                             -> image
```

640×480 is not an arbitrary convenience: it is **exactly the detector's input
size**, so `inference.py`'s own downscale (`r = W_in / Ws`) comes out at 1.0 and
does nothing. Keeping the factor at exactly 2.5 is what makes the intrinsics
mapping exact.

Two consequences that bite:

- **Calibrate at 640×480, not at sensor resolution.** `camera_info_url` is
  validated against 640×480 and ignored with an error otherwise, because
  intrinsics for the wrong resolution are wrong by a factor of 2.5 with nothing
  downstream able to notice.
- **`crop_top` is calibration-invalidating.** Freeze the window before
  calibrating and never touch it afterwards.

Converting an existing sensor-resolution calibration is
`frames.scale_intrinsics`. Note it is **not** `K / 2.5`: under OpenCV's
pixel-centre convention the principal point maps as `(cx + 0.5)/2.5 - 0.5`, i.e.
`cx/2.5 - 0.3`. The 0.3 px is inside the refiner's sub-pixel budget.

## Why full-res is a service and not a topic

A 1600×1200 PNG is tens of milliseconds to encode and over a megabyte on the
wire. Publishing that continuously is the ~230 Mbit/s problem `docker/README.md`
exists to avoid. The node retains the newest frame and encodes it **only when
asked**: calibration and single-shot diagnostics want it, the pose pipeline never
does.

`ros2 service call` prints the response as a 2 MB decimal array, so use
`tools/grab_full_res.py -o /tmp/frame.png`.

## How the 10 Hz is held

There is no ROS timer and no `sleep` in the node. `_capture_loop` is a tight loop
that blocks inside `backend.read(timeout)`, so **the sensor sets the cadence**:
`picam` asks for it with `FrameDurationLimits = (period, period)`, which pins the
OV2311 to `frame_rate` rather than our capturing at 60 fps and throwing away five
of every six frames. The mock paces against an absolute `time.monotonic` deadline,
so it cannot accumulate the drift a fixed `sleep(period)` would.

**If our side overruns the period** (a slow encode), nothing waits — we simply
arrive late for the next read. `queue=False` in the picamera2 configuration means
libcamera will not hand back a frame captured before we asked, so we get the next
*fresh* frame rather than a stale queued one: the measured rate sags, stamps stay
honest because each carries its own `SensorTimestamp`, and `buffer_count` rides
out a transient hiccup. The heartbeat prints the measured encode time, which is
the number to check against the ~3–6 ms budget.

**If the sensor overruns instead** — i.e. `FrameDurationLimits` is ignored, which
is plausible for the Arducam fork with the ISP bypassed — the capture loop drops
the surplus itself, *before* the downscale and the PNG encode, and counts it in
the heartbeat as `N surplus`. Without that guard an unpaced 60 fps sensor would
publish at 60 Hz: six times the bandwidth this node exists to keep down. A
non-zero, growing `surplus` count is therefore the signature of a sensor that is
not honouring the pacing request, and `mock_rate` reproduces it with no hardware.

## Backends

`camera_backend` selects one, and **there is deliberately no automatic
fallback** — synthetic frames feeding a pose that steers a robot are worse than
no frames, because they look exactly like a working camera.

| value | what it is |
|---|---|
| `picamera2` | default. Raw mono stream with the **ISP bypassed** |
| `v4l2` | fallback; see the caveat below |
| `mock` | synthetic or replayed frames, no hardware |

The ISP is bypassed because the sensor is monochrome and has no colour filter
array, so there is nothing for a debayering ISP to do. It also sidesteps
libcamera's `Configuration file 'arducam-pivariety_mono.json' not found for IPA
module 'rpi/pisp'` on a Pi 5 — Arducam document that as benign, but there is no
reason to let a missing tuning file apply gamma or sharpening to the pixels the
refiner measures.

**`v4l2` is not a drop-in replacement on a Pi 5.** The CSI path runs through
`rp1-cfe` with a media-controller graph that libcamera configures, so a bare
`/dev/video0` open generally will not produce OV2311 frames. It is there for a
USB/UVC camera during development, a Pi-4-style unicam path, and as the escape
route if Arducam's libcamera will not import under this image's Python.

`mock` earns its keep beyond the tests: with `mock_image` pointing at a recorded
1600×1200 board frame, the whole chain — including the laptop's real inference
node over actual WiFi — runs with no camera in the building.

## Parameters

| name | default | note |
|---|---|---|
| `camera_backend` | `picamera2` | `picamera2`, `v4l2` or `mock`. A typo is fatal, not clamped |
| `camera_index` | `0` | a Pi 5 has two CSI ports |
| `device` | `/dev/video0` | `v4l2` only |
| `frame_rate` | `10.0` | **floored at 5.0**: below that no offered deadline fits inside the 0.2 s the inference node requests |
| `frame_id` | `pi_camera` | what the pose consumer expects |
| `png_level` | `1` | CPU vs bytes only, never quality. **Must stay lossless** — JPEG would bias the refiner |
| `crop_top` | `50` | see the warning above. Overridden if the sensor windows to 1600×1200 itself |
| `prefer_8bit` | `true` | an 8-bit mode halves CSI bandwidth and loses nothing we keep |
| `exposure_time_us` | `0` | 0 leaves AE alone; pin it once the arena lighting is known |
| `analogue_gain` | `0.0` | 0 is auto; set with `exposure_time_us` |
| `frame_timeout` | `1.0` | longest gap before the stream counts as dead |
| `reconnect_period` | `1.0` | retry interval while the camera is missing |
| `buffer_count` | `4` | libcamera buffers |
| `publish_camera_info` | `true` | |
| `camera_info_url` | `''` | `file://...`; **must declare 640×480** |
| `use_sensor_timestamp` | `true` | back-date stamps to capture time |
| `mock_image` | `''` | 1600×1200 PNG for the mock to replay |
| `mock_latency_ms` | `0.0` | makes the mock back-date its capture clock |
| `mock_rate` | `0.0` | rate the mock *delivers* at; 0 means match `frame_rate`. Set it above `frame_rate` to stand in for a sensor that ignores `FrameDurationLimits` -- the only way to exercise the surplus guard without misbehaving hardware |
| `status_period` | `5.0` | heartbeat interval |

Rates are clamped loudly rather than fatally, as in `p4p_serial_bridge`: a bot
already on the floor is better off running with a safe value than not at all.

## Three things that will bite you if you change them

**1. The capture thread owns the backend, exclusively.** libcamera objects are
not thread-safe, and a `stop()` racing a blocked `read()` is a *segfault*, not an
exception — no amount of locking in Python catches that. The watchdog therefore
sets a flag asking the capture thread to restart the camera; it never touches the
backend itself.

**2. Backends return copies, not views.** The camera recycles its DMA buffer the
moment the request is released, so a view would be overwritten underneath the
publisher and the full-res service. `frames.crop_window` uses `.copy()` and
**not** `np.ascontiguousarray`, which is a no-op returning the input when the
array is already contiguous — which a row slice of a contiguous frame is. It
would satisfy the type and silently break the ownership.

**3. `close()` joins the capture thread before `destroy_node()`.** The capture
thread publishes, and `publish()` on a destroyed publisher is a segfault. At
10 Hz the thread is mid-publish a noticeable fraction of the time.

Unlike `p4p_serial_bridge`, this node uses a `MultiThreadedExecutor` with the
service in its own callback group. That is not gratuitous: the bridge's
operations are all sub-millisecond, whereas a full-res encode is 25–60 ms against
a 100 ms stream period.

## Degraded behaviour

The node **never refuses to start for a hardware reason.** No camera, no ribbon,
no driver: it comes up, logs the reason at `error` on a 5 s throttle, publishes
nothing, and retries every `reconnect_period`, so installing the package or
reseating the FPC recovers without a restart.

The service never blocks waiting for a frame — that would hold an executor thread
for the whole outage and turn a camera fault into an unresponsive node. It returns
`success=false` with a diagnosis instead.

**One real limitation:** if the installed picamera2 lacks `wait(timeout=...)`,
reads are unbounded and a wedged driver cannot be interrupted. The watchdog can
then only *report* the stall; `restart: unless-stopped` in `compose.yaml` is the
backstop. The node logs this once, at `error`, when it detects the old API.

## Running

```bash
# no hardware
ros2 run p4p_camera camera_node --ros-args -p camera_backend:=mock
ros2 launch p4p_camera camera.launch.py camera_backend:=mock

# replay a recorded frame to the real inference node
ros2 run p4p_camera camera_node --ros-args \
  -p camera_backend:=mock -p mock_image:=/workspace/board.png

# on the Pi, with the camera attached
ros2 launch p4p_camera camera.launch.py

# check it
ros2 topic hz /image
ros2 topic bw /image
ros2 topic echo /image --field format --once     # mono8; png compressed mono8
python3 tools/grab_full_res.py -o /tmp/frame.png
```

## Tests

```bash
colcon test --packages-select p4p_camera --event-handlers console_direct+
```

`test_frames.py` and `test_mock_backend.py` need only numpy, cv2 and pytest, so
they also run outside ROS entirely:

```bash
PYTHONPATH=src/ros/src/p4p_camera python3 -m pytest \
  src/ros/src/p4p_camera/test/test_frames.py \
  src/ros/src/p4p_camera/test/test_mock_backend.py
```

The graph tests skip cleanly without `rclpy`. The one worth knowing about is
`test_camera_stream.py::test_publisher_qos_matches_the_inference_node`: it
subscribes with the *real* `convchart_qos.qos.IMAGE_SUB_QOS`, imported rather
than restated, and fails on an `incompatible_qos` event. A QoS mismatch raises on
neither side — the topic simply goes silent — so without that test the failure
mode is a mystified timeout on the laptop.

**On an x86 host under QEMU, set `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` first.**
CycloneDDS cannot create a node under QEMU (`set IP_MULTICAST_IF failed:
Unsupported`), which `docker/README.md` already records, and the graph tests are
what it bites.
