#!/usr/bin/env bash
# One-time HOST setup for the Raspberry Pi (64-bit Raspberry Pi OS or Ubuntu 24.04 arm64):
#   * Docker Engine + compose plugin (Docker's convenience script; supports both distros)
#   * docker + dialout + i2c group membership for the login user (Arduino serial
#     and the BNO085 without sudo)
#   * a udev rule so /dev/i2c-* is group-readable (Ubuntu arm64 does not ship one)
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
groupadd -f i2c
usermod -aG docker,dialout,i2c "${TARGET_USER}"
systemctl enable --now docker

echo "== I2C access for the BNO085"
cat > /etc/udev/rules.d/60-i2c-tools.rules <<'UDEV'
SUBSYSTEM=="i2c-dev", GROUP="i2c", MODE="0660"
UDEV
udevadm control --reload-rules >/dev/null 2>&1 || true

echo "== DDS-friendly UDP buffers"
cat > /etc/sysctl.d/60-ros2-dds.conf <<'SYSCTL'
net.core.rmem_max=2147483647
net.core.rmem_default=8388608
net.ipv4.ipfrag_time=3
net.ipv4.ipfrag_high_thresh=134217728
SYSCTL
sysctl --system >/dev/null

echo "== Camera pipeline at boot (p4p_camera's v4l2 backend)"
command -v media-ctl >/dev/null || { apt-get update -qq && apt-get install -y -qq v4l-utils; }
install -m 0755 "${SCRIPT_DIR}/camera-pipeline-pi.sh" /usr/local/sbin/p4p-camera-pipeline
cat > /etc/systemd/system/p4p-camera-pipeline.service <<'UNIT'
[Unit]
Description=p4p: configure the Pi 5 CSI pipeline for p4p_camera's v4l2 backend

[Service]
Type=oneshot
RemainAfterExit=yes
# 15 fps = the camera node's frame_rate; keep the two equal.
ExecStart=/usr/local/sbin/p4p-camera-pipeline 15

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable p4p-camera-pipeline.service
systemctl restart p4p-camera-pipeline.service \
    || echo "camera pipeline not configured yet: see systemctl status p4p-camera-pipeline"

echo
echo "Done. Log out and back in, plug in the Arduino Mega and check it appears as /dev/ttyACM0 (ls -l /dev/ttyACM*)."
echo
echo "Things this script deliberately does NOT do, because editing the boot"
echo "config is the operator's call:"
echo
echo "  1. Add these to /boot/firmware/config.txt, then reboot:"
echo "         camera_auto_detect=0"
echo "         dtoverlay=arducam-pivariety     (,cam0 if the ribbon is in CAM/DISP 0)"
echo "         dtparam=i2c_arm=on"
echo "         dtparam=i2c_arm_baudrate=400000"
echo "     Without dtparam=i2c_arm=on, /dev/i2c-1 does not exist and the pi container"
echo "     will not start at all."
echo
echo "  2. Put the i2c group's numeric gid in .env as I2C_GID, and the gpio group's"
echo "     as GPIO_GID:"
echo "         getent group i2c | cut -d: -f3"
echo "         getent group gpio | cut -d: -f3"
echo "     The container needs the numbers, not the names."
echo
echo "Then check the sensor answers:  i2cdetect -y 1   # expect 4a"
