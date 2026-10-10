#!/usr/bin/env bash
# One-time HOST setup for the Raspberry Pi (64-bit Raspberry Pi OS or Ubuntu 24.04 arm64):
#   * Docker Engine + compose plugin (Docker's convenience script; supports both distros)
#   * docker + dialout + i2c group membership for the login user (Arduino serial
#     and the BNO085 without sudo)
#   * a udev rule so /dev/i2c-* is group-readable (Ubuntu arm64 does not ship one)
#   * DDS-friendly UDP buffers
#
#   sudo bash docker/host-setup-pi.sh
set -euo pipefail

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
# Raspberry Pi OS ships this rule; Ubuntu arm64 does not, which leaves /dev/i2c-*
# as root-only and every read failing with EACCES.
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

echo
echo "Done. Log out and back in, plug in the Arduino Mega and check it appears as /dev/ttyACM0 (ls -l /dev/ttyACM*)."
echo
echo "Two things this script deliberately does NOT do, because editing the boot"
echo "config is the operator's call (same policy as the camera overlay):"
echo
echo "  1. Add these to /boot/firmware/config.txt, then reboot:"
echo "         dtparam=i2c_arm=on"
echo "         dtparam=i2c_arm_baudrate=400000"
echo "     Without the first, /dev/i2c-1 does not exist and the pi container will"
echo "     not start at all."
echo
echo "  2. Put the i2c group's numeric gid in .env as I2C_GID:"
echo "         getent group i2c | cut -d: -f3"
echo "     The container needs the number, not the name."
echo
echo "Then check the sensor answers:  i2cdetect -y 1   # expect 4a"
