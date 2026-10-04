# Raspberry Pi container

The Pi side of the wireless inference link: ROS 2 Jazzy in Docker on a 64-bit Raspberry Pi,
talking over WiFi to the machine that runs the inference node in `src/ros`.

| file | what it is |
|------|------------|
| `compose.yaml` (repo root) | the `pi` service: host networking, Arduino passthrough, build args from `.env` |
| `.env.example` (repo root) | per-machine settings: uid/gid, `ROS_DOMAIN_ID`, RMW, `WITH_CAMERA`, Arduino port |
| `docker/pi/Dockerfile` | the image (`linux/arm64`): ROS 2 Jazzy ros-base, numpy/scipy, pyserial, tf2 + tf_transformations, osqp/qpsolvers, CycloneDDS + Zenoh RMWs; camera stack opt-in |
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

Camera: `WITH_CAMERA=1` in `.env` (then rebuild) adds `camera_ros` (libcamera, CSI camera
modules), `v4l2_camera` (USB cameras) and `image_transport_plugins` (the `compressed` and
`zstd` transports). The container then also needs the camera device nodes: the simplest
working set is `privileged: true` plus `/run/udev:/run/udev:ro` (commented in `compose.yaml`).

## 3. Run

```bash
docker compose up -d pi
docker compose exec pi bash -c "python3 docker/pi/smoke_test.py"
docker compose exec pi bash             # ROS 2 sourced, this repo mounted at /workspace
```

Inside the container, build the message package once (the inference node itself runs on the laptop):

```bash
cd /workspace/src/ros
colcon build --symlink-install --packages-select convchart_interfaces
```

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

Frames: the Pi publishes `sensor_msgs/Image` `mono8`. A native 1600x1200 frame is 1.92 MB, so
15 Hz raw is ~230 Mbit/s, more than WiFi sustains reliably. For lossless compression use the
`compressed` transport with `format: png` (decode on the laptop with `cv2.imdecode`) or the
`zstd` transport; both need `WITH_CAMERA=1` on the Pi. Avoid JPEG: the refiner reads
sub-pixel offsets from 24x24 native crops, so compression artefacts would bias it.
