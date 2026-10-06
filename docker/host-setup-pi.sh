#!/usr/bin/env bash
# One-time HOST setup for the Raspberry Pi (64-bit Raspberry Pi OS or Ubuntu 24.04 arm64):
#   * Docker Engine + compose plugin (Docker's convenience script; supports both distros)
#   * docker + dialout group membership for the login user (Arduino serial without sudo)
#   * DDS-friendly UDP buffers
#   * p4p-camera-pipeline.service: configures the Pi 5 CSI pipeline for
#     p4p_camera's v4l2 backend at every boot (see camera-pipeline-pi.sh)
#
#   sudo bash docker/host-setup-pi.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "$(id -u)" -ne 0 ]; then
    echo "run with sudo: sudo bash $0" >&2
    exit 1
fi
TARGET_USER="${SUDO_USER:-$USER}"

if [ "$(uname -m)" != "aarch64" ]; then
    echo "expected a 64-bit OS (uname -m = aarch64); ROS 2 Jazzy has no 32-bit arm builds" >&2
    exit 1
fi

echo "== Docker Engine"
curl -fsSL https://get.docker.com | sh
usermod -aG docker,dialout "${TARGET_USER}"
systemctl enable --now docker

echo "== DDS-friendly UDP buffers"
cat > /etc/sysctl.d/60-ros2-dds.conf <<'SYSCTL'
net.core.rmem_max=2147483647
net.core.rmem_default=8388608
net.ipv4.ipfrag_time=3
net.ipv4.ipfrag_high_thresh=134217728
SYSCTL
sysctl --system >/dev/null

echo "== Camera pipeline at boot (p4p_camera's v4l2 backend)"
# media-ctl / v4l2-ctl. Raspberry Pi OS ships them; Ubuntu server may not.
command -v media-ctl >/dev/null || { apt-get update -qq && apt-get install -y -qq v4l-utils; }
# A copy, not a link into this checkout: the service runs as root at boot.
# Re-run this script (or just this install line) after changing it.
install -m 0755 "${SCRIPT_DIR}/camera-pipeline-pi.sh" /usr/local/sbin/p4p-camera-pipeline
cat > /etc/systemd/system/p4p-camera-pipeline.service <<'UNIT'
[Unit]
Description=p4p: configure the Pi 5 CSI pipeline for p4p_camera's v4l2 backend

[Service]
Type=oneshot
RemainAfterExit=yes
# 10 fps = the camera node's frame_rate; keep the two equal.
ExecStart=/usr/local/sbin/p4p-camera-pipeline 10

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable p4p-camera-pipeline.service
# Configure it now too, unless no camera is attached yet (it then just reports why).
systemctl restart p4p-camera-pipeline.service \
    || echo "camera pipeline not configured yet: see systemctl status p4p-camera-pipeline"

echo
echo "Done. Log out and back in, plug in the Arduino Mega and check it appears as /dev/ttyACM0 (ls -l /dev/ttyACM*)."
echo "Camera: /boot/firmware/config.txt needs camera_auto_detect=0 and dtoverlay=arducam-pivariety"
echo "(dtoverlay=arducam-pivariety,cam0 if the ribbon is in CAM/DISP 0), then a reboot."
