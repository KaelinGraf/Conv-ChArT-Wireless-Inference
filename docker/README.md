# Raspberry Pi container

The Pi side of the wireless inference link: ROS 2 Jazzy in Docker on a 64-bit Raspberry Pi,
talking over WiFi to the machine that runs the inference node in `src/ros`.

| file | what it is |
|------|------------|
| `compose.yaml` (repo root) | the `pi` service: host networking, Arduino passthrough, build args from `.env` |
| `.env.example` (repo root) | per-machine settings: uid/gid, `ROS_DOMAIN_ID`, RMW, `WITH_CAMERA`, Arduino port |
| `docker/pi/Dockerfile` | the image (`linux/arm64`): ROS 2 Jazzy ros-base, numpy/scipy, pyserial, tf2 + tf_transformations, osqp/qpsolvers, ros2_numpy (patched, see the Dockerfile), CycloneDDS + Zenoh RMWs, OpenCV; camera stack opt-in |
| `docker/pi/smoke_test.py` | runtime check: rclpy + RMW, numpy/scipy, serial ports, tf_transformations |
| `docker/common/` | entrypoint + shell environment: sources ROS 2, the `src/ros` overlay once it is built, and the CycloneDDS profile |
| `docker/ros/cyclonedds.xml` | CycloneDDS profile for the WiFi link (use the same file on the laptop) |
| `docker/host-setup-pi.sh` | one-time Pi host setup: Docker Engine, `docker`/`dialout` groups, UDP buffer sysctls |

## 1. Host setup (once per Pi)

64-bit Raspberry Pi OS or Ubuntu 24.04 arm64, 4 GB RAM or more:

```bash
git clone https://github.com/KaelinGraf/Conv-ChArT-Wireless-Inference ~/Conv-ChArT-Wireless-Inference
cd ~/Conv-ChArT-Wireless-Inference
sudo bash docker/host-setup-pi.sh       # then log out and back in
cp .env.example .env                    # optional, every value has a default
```

## 2. Get the image

Build it on the Pi (simplest; 10-20 min on a Pi 4/5):

```bash
docker compose build pi
```

Or cross-build on an x86 machine and copy it over. The build machine needs arm64 emulation
(`sudo apt-get install qemu-user-static binfmt-support` on Ubuntu; if a build dies with
`exec format error`, check that `ls /proc/sys/fs/binfmt_misc/` lists `qemu-aarch64`).
`UID`/`GID` are baked in at build time, so set them to the Pi user's values.

```bash
docker compose build pi
docker save convchart-wireless/pi:jazzy | gzip > convchart-pi.tar.gz
scp convchart-pi.tar.gz <user>@<pi-ip>:
# on the Pi:
gunzip -c convchart-pi.tar.gz | docker load
```

Camera: `WITH_CAMERA=1` in `.env` (then rebuild) adds `v4l-utils`, `v4l2_camera` (USB
cameras), `image_transport_plugins` (the `compressed` and `zstd` transports) and **Arducam's
libcamera fork plus their picamera2**, which is what `p4p_camera` drives the Pivariety OV2311
with. `camera_ros` is deliberately *not* installed: Arducam's packages replace the distro
libcamera, and `camera_ros` is built against the distro ABI.

Four things the image cannot do for you:

1. **The kernel driver and overlay are the host's job.** `/boot/firmware/config.txt` needs
   `camera_auto_detect=0` and `dtoverlay=arducam-pivariety`, then a reboot. Never run the
   Arducam script's `-p kernel_driver` inside the container: it builds against the host kernel
   and writes to `/boot/firmware/config.txt`, i.e. the wrong machine.
2. **Device passthrough is a separate compose service.** The plain `pi` service passes through
   no camera nodes, because a listed-but-absent device stops the container. Use the profile:
   `docker compose --profile camera up -d pi-camera`. It starts `privileged` for first
   bring-up; narrow it to a `device_cgroup_rules` allow-list once `ls -l /dev/video*
   /dev/media* /dev/v4l-subdev* /dev/dma_heap/*` on the real Pi gives you the major numbers.
   libcamera needs more than `/dev/video0`: `/dev/media*` is the media controller it
   *configures the pipeline through* (missing it is the classic "no cameras available" inside
   Docker), and `/dev/dma_heap` is where a Pi 5 allocates frame buffers.
3. **Verify the binding imports before anything else.** Arducam's `.debs` target Raspberry Pi
   OS (Debian bookworm, python 3.11); this base is Ubuntu noble with python 3.12, and
   `python3-libcamera` is a compiled binding. The build warns rather than failing if it will
   not install, so check it explicitly, inside the container:

   ```bash
   python3 -c "from picamera2 import Picamera2; import pprint; pprint.pp(Picamera2().sensor_modes)"
   ```

   This one command proves the userspace imports, that udev enumeration crosses the container
   boundary, and that dma_heap works. Its output is also the fixture for
   `p4p_camera.frames.pick_mode`'s tests. If it fails, the fallbacks in increasing cost are:
   build the fork from source against python 3.12; use `p4p_camera`'s `v4l2` backend with a
   `media-ctl` setup script, which needs no userspace libcamera at all; or run the camera node
   on the host, where Arducam's stack installs natively.

4. **I2C for the BNO085 is the host's job too, and this one stops the container.**
   `/boot/firmware/config.txt` needs the following, then a reboot:

   ```
   dtparam=i2c_arm=on
   dtparam=i2c_arm_baudrate=400000     # Adafruit's figure for the BNO085
   ```

   Without the first line `/dev/i2c-1` does not exist, and because it is listed in the
   **main** `pi` service's `devices` -- the IMU is not optional -- `docker compose up pi`
   fails outright. That is the Arduino footgun again, deliberately accepted: unlike the
   camera, I2C is one `dtparam` away on every Pi, so a profile would be more ceremony than
   the problem is worth.

   Then put the host's i2c group id in `.env`, because `/dev/i2c-1`'s ownership comes from
   the host and compose matches on the **number**, not the name:

   ```bash
   getent group i2c | cut -d: -f3      # -> I2C_GID in .env
   ```

   `docker/host-setup-pi.sh` creates the group, adds you to it, and installs the udev rule
   that Ubuntu arm64 omits. Verify, on the host and then inside the container:

   ```bash
   i2cdetect -y 1                      # expect 4a (0x4b if ADR is tied high)
   ```

   Stop the IMU node before running `i2cdetect`: it probes with writes. If reads fail with
   `Permission denied: '/dev/i2c-1'`, `I2C_GID` is wrong.

   The second `dtparam` is about reliability, not speed. The BNO085 violates the I2C spec
   when it releases a clock stretch, and Pi 1-4 have a matching controller bug that makes
   this chip notoriously painful on a Pi. The Pi 5's RP1 controller reportedly handles
   stretching correctly, which is why the hardware bus is the default -- keep the kernel
   current, since early Pi 5 kernels shipped I2C timing bugs. If it misbehaves anyway, the
   fallback is software I2C on other pins:

   ```
   dtoverlay=i2c-gpio,bus=3,i2c_gpio_sda=23,i2c_gpio_scl=24   # GPIO23/24 = pins 16/18
   ```

   then rewire, set `I2C_DEV=/dev/i2c-3`, and run the node with `-p i2c_bus:=3`. See
   `src/ros/src/p4p_imu/README.md`.

Arducam also warn that a system `apt upgrade` silently replaces their libcamera with
Raspberry Pi's, which breaks a working camera. The image `apt-mark hold`s it; don't run
`apt upgrade` in a live container, rebuild instead.

## 3. Run

```bash
docker compose up -d pi
docker compose exec pi bash -c "python3 docker/pi/smoke_test.py"
docker compose exec pi bash             # ROS 2 sourced, this repo mounted at /workspace
```

Inside the container, build the workspace once. The Pi needs five packages and **not**
`convchart_ros`: that one is the laptop's inference node, and its `cv_bridge` dependency
resolves to `libopencv-dev` plus some fifty further apt packages a camera has no use for.
The only thing the two sides share is `convchart_qos`, which imports `rclpy` and nothing else.

```bash
cd /workspace/src/ros
# Interfaces FIRST and alone: adding a .srv regenerates the interface library, and a stale
# install/ surfaces later as an unhelpful "unknown type" at runtime.
colcon build --symlink-install --packages-select convchart_interfaces
colcon build --symlink-install --packages-select convchart_qos p4p_camera p4p_serial_bridge p4p_imu
```

Both machines must rebuild `convchart_interfaces` after a `.msg`/`.srv` change, and they must
agree. To run the camera tests (they need no camera -- the `mock` backend stands in):

```bash
colcon test --packages-select p4p_camera p4p_imu --event-handlers console_direct+
colcon test-result --verbose
```

`p4p_imu` needs no sensor either -- its `mock` backend stands in, and its pure tier
(`test_orientation.py`, `test_mock_backend.py`) runs with nothing but pytest:

```bash
PYTHONPATH=src/ros/src/p4p_imu python3 -m pytest \
  src/ros/src/p4p_imu/test/test_orientation.py \
  src/ros/src/p4p_imu/test/test_mock_backend.py
```

Under QEMU on an x86 host, export `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` first: CycloneDDS
cannot create a node there (see the note further down), and the graph tests are what it bites.

New shells source `src/ros/install/setup.bash` automatically.

- Use `bash -c "..."` for one-shot commands: `docker compose exec pi python3 ...` skips the ROS environment.
- The Arduino Mega 2560 is passed through as `/dev/ttyACM0` (set `ARDUINO_PORT` if the host
  names it differently). Docker will not start the container when that device is missing, so
  remove the `devices:` entry in `compose.yaml` if no Arduino is attached.
- `cap_add: SYS_NICE` + `ulimits.rtprio` allow `SCHED_FIFO` / negative nice for a fixed-rate loop.
- `command: sleep infinity` is a placeholder; swap it for `ros2 launch ...` once the nodes
  exist. `restart: unless-stopped` brings the container back after a reboot.
- `src/ros/build`, `install` and `log` are architecture-specific: build on the machine that
  runs them, never copy them between the laptop and the Pi.
- A cross-built image can be smoke-tested on x86 with
  `docker run --rm --platform linux/arm64 --network host -v $PWD:/workspace -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp convchart-wireless/pi:jazzy bash -c "python3 docker/pi/smoke_test.py"`.
  The RMW override is needed because CycloneDDS cannot create a node under QEMU
  (`set IP_MULTICAST_IF failed: Unsupported`, an emulation limit, not a Pi problem).

## ROS 2 across the WiFi link

The container uses host networking, `ROS_DOMAIN_ID` and `RMW_IMPLEMENTATION` from `.env`, and
`docker/ros/cyclonedds.xml` (multicast for discovery only, unicast data, bigger receive
buffers). The laptop must use the same domain ID, the same RMW and the same profile
(`export CYCLONEDDS_URI=file://<repo>/docker/ros/cyclonedds.xml`). First test:

```bash
# laptop                                        # pi
ros2 run demo_nodes_py talker                   ros2 run demo_nodes_py listener
```

If discovery does not cross your access point (client isolation, mesh WiFi), add the other
host under `<Peers>` in `cyclonedds.xml`, or pin `NetworkInterface` to the WiFi NIC if
autodetect picks `docker0`.

Zenoh is installed as the alternative for lossy links: set `RMW_IMPLEMENTATION=rmw_zenoh_cpp`
on both ends, run `ros2 run rmw_zenoh_cpp rmw_zenohd` on the laptop, and on the Pi point at
it with `ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/<laptop-ip>:7447"]'`.

Frames: `p4p_camera` publishes `sensor_msgs/CompressedImage` on `image`, as a **640x480
mono8 PNG at 10 Hz** -- roughly 200 kB a frame, about 16 Mbit/s. That is the whole reason the
stream is downscaled on the Pi: a native 1600x1200 frame is 1.92 MB, so 15 Hz raw would be
~230 Mbit/s, far more than WiFi sustains reliably. 640x480 is also *exactly* the detector's
input size, so the inference pipeline's own resize becomes the identity.

The full 1600x1200 frame is still available, but **on demand only**, through the
`image_full_res` service (`convchart_interfaces/GetFullResImage`); the node retains the newest
frame and encodes it when asked. Use `tools/grab_full_res.py -o frame.png` -- `ros2 service
call` prints the response as a ~2 MB decimal array. That response is an order of magnitude
larger than anything else on this link, so test it *across* WiFi, not just in-container; if it
proves unreliable, the fallback is to have the service write the file into the bind-mounted
workspace and return the path.

**Avoid JPEG at any quality**: the refiner reads sub-pixel offsets from 24x24 native crops, so
compression artefacts would bias the pose rather than merely soften the picture. PNG's
compression level trades CPU for bytes only, never quality.

Because the stream is downscaled, **intrinsics must be calibrated at 640x480**, not at sensor
resolution. See `src/ros/src/p4p_camera/README.md`; converting an existing calibration is
`frames.scale_intrinsics`, and it is *not* simply `K / 2.5`.
