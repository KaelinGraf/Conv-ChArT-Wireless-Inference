# p4p_camera

Arducam **2MP OV2311 global-shutter mono** (Pivariety, NoIR) on a Raspberry Pi 5.
Owns the camera and nothing else: no filtering, no pose, no control.

| direction | name | type | notes |
|---|---|---|---|
| out | `image` | `sensor_msgs/CompressedImage` | 640×480 mono8 PNG, 15 Hz |
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

## How the 15 Hz is held

There is no ROS timer and no `sleep` in the node. `_capture_loop` is a tight loop
that blocks inside `backend.read(timeout)`, so **the sensor sets the cadence**.
On `v4l2`, `camera-pipeline-pi.sh` sets the sensor's vertical blanking for
`FPS` (default 15). On `picam`, the backend asks for
`FrameDurationLimits = (period, period)`. Either way the OV2311 is pinned to
`frame_rate`, rather than our capturing at 60 fps and throwing away five of
every six frames.

**Keep the script's `FPS` equal to `frame_rate`.** A slower sensor starves the
node. A faster one is caught by the surplus guard below, at a CPU cost.

The mock paces against an absolute `time.monotonic` deadline, so it cannot
accumulate the drift a fixed `sleep(period)` would.

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
| `v4l2` | default. The sensor's native 8-bit mono mode straight off `rp1-cfe`; needs the host's CSI pipeline configured (below) |
| `picamera2` | raw mono stream with the ISP bypassed; needs Arducam's libcamera, which this image cannot have (below) |
| `mock` | synthetic or replayed frames, no hardware |

**`v4l2` on a Pi 5.** The CSI path runs through `rp1-cfe`'s media-controller
graph, which libcamera would normally configure. Here
`docker/camera-pipeline-pi.sh` does it on the host instead:
- **Mode:** the sensor runs in its native `Y8_1X8` mode at 1600×1300.
- **Route:** the frames go through `csi2` to `rp1-cfe-csi2_ch0`, which is
  `/dev/video0`. They arrive as plain `GREY`, with nothing to unpack.
- **Rate:** the sensor is paced to 15 fps through vertical blanking.

`docker/host-setup-pi.sh` installs the script as `p4p-camera-pipeline.service`,
which runs at every boot. Without that setup `/dev/video0` opens but never
delivers a frame, and the node logs `no frame from /dev/video0` until it is
configured. The same backend serves a USB/UVC camera during development, or a
Pi-4-style unicam path, with no setup.

**Why not `picamera2` on the Pi.** The OV2311 needs Arducam's libcamera fork:
- **Not installable here:** the fork ships for Raspberry Pi OS (bookworm and
  trixie) only. Its installer has nothing for this image's Ubuntu noble.
- **Stock libcamera is no help:** it reports "No cameras available", whether
  the one in the image or the Pi's own.

The backend stays for a host where Arducam's stack does install. It bypasses
the ISP because the sensor is monochrome and has no colour filter array, so
there is nothing for a debayering ISP to do. That also sidesteps libcamera's
`Configuration file 'arducam-pivariety_mono.json' not found for IPA module
'rpi/pisp'`: Arducam document it as benign, but there is no reason to let a
missing tuning file apply gamma or sharpening to the pixels the refiner
measures.

`mock` earns its keep beyond the tests: with `mock_image` pointing at a recorded
1600×1200 board frame, the whole chain — including the laptop's real inference
node over actual WiFi — runs with no camera in the building.

## Parameters

| name | default | note |
|---|---|---|
| `camera_backend` | `v4l2` | `v4l2`, `picamera2` or `mock`. A typo is fatal, not clamped |
| `camera_index` | `0` | a Pi 5 has two CSI ports |
| `device` | `/dev/video0` | `v4l2` only |
| `frame_rate` | `15.0` | **floored at 5.0**: below that no offered deadline fits inside the 0.2 s the inference node requests |
| `frame_id` | `pi_camera` | what the pose consumer expects |
| `png_level` | `1` | CPU vs bytes only, never quality. **Must stay lossless** — JPEG would bias the refiner |
| `crop_top` | `50` | see the warning above. Overridden if the sensor windows to 1600×1200 itself |
| `prefer_8bit` | `true` | an 8-bit mode halves CSI bandwidth and loses nothing we keep |
| `exposure_time_us` | `0` | `picamera2` only. 0 leaves AE alone; pin it once the arena lighting is known. On `v4l2` there is no AE: see the sensor table below |
| `analogue_gain` | `0.0` | `picamera2` only. 0 is auto; set with `exposure_time_us` |
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

## The sensor, as measured on the Pi 5

These values were read off the hardware with `media-ctl -p` and `v4l2-ctl -l` on
the sensor subdev. Where the board sits:
- **Driver:** `arducam-pivariety`, Pivariety firmware `0x10002`, at i2c
  `11-000c`.
- **Port:** CAM/DISP 1. For CAM/DISP 0, append `,cam0` to the overlay.

| | |
|---|---|
| modes | `Y8_1X8` and `Y10_1X10`, both 1600×1300 |
| `pixel_rate` | 160 MHz, read-only |
| `horizontal_blanking` | 208, fixed |
| `vertical_blanking` | 174–16399. 174 gives ~60 fps, 4600 gives 15 fps, 7550 gives 10 fps, 16399 gives ~5 fps |
| `exposure` | 1–65523, default 800 |
| `analogue_gain` | 100–3100, default 100 |

The frame rate follows from
`pixel_rate / ((1600 + hblank) × (1300 + vblank))`.

On `v4l2` nothing runs auto-exposure. Set exposure and gain directly on the
sensor subdev, which `camera-pipeline-pi.sh` prints:
`v4l2-ctl -d /dev/v4l-subdevN -c exposure=…,analogue_gain=…`.

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
15 Hz the thread is mid-publish a noticeable fraction of the time.

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

# on the Pi, with the camera attached and the CSI pipeline configured
# (p4p-camera-pipeline.service does it at boot; by hand: sudo docker/camera-pipeline-pi.sh)
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
