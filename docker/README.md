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
| `docker/host-setup-pi.sh` | one-time Pi host setup: Docker Engine, `docker`/`dialout` groups, UDP buffer sysctls, the camera pipeline boot service |
| `docker/camera-pipeline-pi.sh` | configures the Pi 5 CSI pipeline for `p4p_camera`'s `v4l2` backend; run at boot by `p4p-camera-pipeline.service` |

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

Camera: the image needs nothing extra. `p4p_camera`'s default `v4l2` backend reads the
Pivariety OV2311 through V4L2 with OpenCV, which the base image already has. The camera
needs three things set up on the host:

1. **The kernel driver and overlay.** `/boot/firmware/config.txt` needs:
   - `camera_auto_detect=0`;
   - `dtoverlay=arducam-pivariety`. If the ribbon is in CAM/DISP 0, append `,cam0`; the
     default is CAM/DISP 1.

   Then reboot. `dmesg | grep -i pivariety` should report the board's firmware version.

   Never run the Arducam script's `-p kernel_driver` inside the container. It builds
   against the host kernel and writes to `/boot/firmware/config.txt`, i.e. the wrong
   machine.
2. **The CSI pipeline.** On a Pi 5 the sensor sits behind `rp1-cfe`'s media-controller graph,
   which libcamera would normally configure. `docker/camera-pipeline-pi.sh` does it with
   `media-ctl` instead: 8-bit mono 1600x1300 to `/dev/video0`, with the sensor paced to
   15 fps.
   - **At boot:** `host-setup-pi.sh` installs it as `p4p-camera-pipeline.service`, which runs
     at every boot.
   - **Re-run it** with `sudo /usr/local/sbin/p4p-camera-pipeline` if anything else
     reconfigures the graph, for example `rpicam-hello` on the host.
   - **Check it** with `v4l2-ctl -d /dev/video0 --stream-mmap --stream-count=30
     --stream-to=/dev/null`, which should report ~15 fps.
3. **Device passthrough, through a separate compose service.** The plain `pi` service passes
   through no camera nodes, because a listed-but-absent device stops the container.
   - **Use the profile:** `docker compose --profile camera up -d pi-camera`.
   - **Privileges:** it starts `privileged`. Narrow that to a `device_cgroup_rules` allow-list
     once `ls -l /dev/video* /dev/media* /dev/v4l-subdev*` on the real Pi gives you the major
     numbers.
   - **The `v4l2` path uses only `/dev/video0`.** The service's `/run/udev`, `/dev/media*` and
     `/dev/dma_heap` are there for libcamera.

`WITH_CAMERA=1` (rebuild after changing) adds `v4l-utils`, `v4l2_camera` (USB cameras) and
`image_transport_plugins` (the `compressed` and `zstd` transports). It also tries to install
**Arducam's libcamera fork plus their picamera2** for the `picamera2` backend, which cannot
succeed on this base:
- **No packages for this OS:** Arducam publish packages for Raspberry Pi OS (bookworm and
  trixie) only.
- **The installer stops:** it ends with "Unsupported package" on Ubuntu noble.
- **The build carries on:** it warns, so the `picamera2` backend stays unavailable in this
  image.

`camera_ros` is deliberately *not* installed: Arducam's packages would replace the distro
libcamera it is built against.

## 3. Run

```bash
docker compose up -d pi
docker compose exec pi bash -c "python3 docker/pi/smoke_test.py"
docker compose exec pi bash             # ROS 2 sourced, this repo mounted at /workspace
```

Inside the container, build the workspace once. The Pi needs four packages and **not**
`convchart_ros`: that one is the laptop's inference node, and its `cv_bridge` dependency
resolves to `libopencv-dev` plus some fifty further apt packages a camera has no use for.
The only thing the two sides share is `convchart_qos`, which imports `rclpy` and nothing else.

```bash
cd /workspace/src/ros
# Interfaces FIRST and alone: adding a .srv regenerates the interface library, and a stale
# install/ surfaces later as an unhelpful "unknown type" at runtime.
colcon build --symlink-install --packages-select convchart_interfaces
colcon build --symlink-install --packages-select convchart_qos p4p_camera p4p_serial_bridge
```

Both machines must rebuild `convchart_interfaces` after a `.msg`/`.srv` change, and they must
agree. To run the camera tests (they need no camera -- the `mock` backend stands in):

```bash
colcon test --packages-select p4p_camera --event-handlers console_direct+
colcon test-result --verbose
```

Under QEMU on an x86 host, export `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` first: CycloneDDS
cannot create a node there (see the note further down), and the graph tests are what it bites.

New shells source `src/ros/install/setup.bash` automatically.

- Use `bash -c "..."` for one-shot commands: `docker compose exec pi python3 ...` skips the ROS environment.
- The Arduino Mega 2560 is passed through as `/dev/ttyACM0`. Set `ARDUINO_PORT` if the host
  names it differently.
  - Docker will not start the container when that device is missing. While no Arduino is
    attached, set `ARDUINO_PORT=/dev/null` in `.env`.
  - The containers then start, and the serial bridge has nothing to talk to.
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
mono8 PNG at 15 Hz** -- about 110 kB a frame as measured, so ~13 Mbit/s. That is the whole
reason the stream is downscaled on the Pi: a native 1600x1200 frame is 1.92 MB, so 15 Hz raw would be
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
