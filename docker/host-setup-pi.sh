#!/usr/bin/env bash
# One-time HOST setup for the Raspberry Pi (64-bit Raspberry Pi OS or Ubuntu 24.04 arm64):
#   * Docker Engine + compose plugin (Docker's convenience script; supports both distros)
#   * docker + dialout group membership for the login user (Arduino serial without sudo)
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

echo
echo "Done. Log out and back in, plug in the Arduino Mega and check it appears as /dev/ttyACM0 (ls -l /dev/ttyACM*)."
