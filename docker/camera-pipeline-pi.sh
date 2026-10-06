#!/usr/bin/env bash
# Configure the Pi 5 CSI pipeline for p4p_camera's v4l2 backend (the default).
#
# On a Pi 5 the OV2311 sits behind rp1-cfe's media-controller graph, which
# libcamera would normally set up. Arducam's libcamera ships for Raspberry Pi OS
# only -- there is none for the container's Ubuntu -- so this does it with
# media-ctl instead:
#
#   sensor --Y8_1X8 1600x1300--> csi2:0 -> csi2:4 -> rp1-cfe-csi2_ch0 (/dev/video0)
#
# Y8 is the sensor's native 8-bit mono mode, so frames reach the node as plain
# GREY with nothing to unpack. The sensor is also paced to FPS through vertical
# blanking, as picamera2's FrameDurationLimits would do; unpaced it runs at
# ~60 fps and the node discards five frames in six.
#
# Runs on the HOST. host-setup-pi.sh installs it as p4p-camera-pipeline.service,
# which runs it at every boot. The kernel forgets this setup on reboot, and
# anything else that configures the graph (rpicam-*, libcamera) overwrites it, so
# re-run it after either -- with the camera node stopped, since a streaming
# pipeline refuses format changes:
#
#   sudo /usr/local/sbin/p4p-camera-pipeline [FPS]    # as installed
#   sudo docker/camera-pipeline-pi.sh [FPS]           # from the repo
#
# FPS defaults to 15, the camera node's frame_rate; keep the two equal.
set -euo pipefail

FPS="${1:-15}"
WIDTH=1600
HEIGHT=1300          # p4p_camera.frames.ACTIVE; the node crops it to 1600x1200
CODE=Y8_1X8
WAIT_S=30            # at boot, rp1-cfe registers only once the sensor has probed

# Media and subdev numbers are not stable across boots: find everything by name.
media=""
for _ in $(seq $((WAIT_S * 2))); do
    for m in /dev/media*; do
        [ -e "$m" ] || continue
        if media-ctl -d "$m" -p 2>/dev/null | grep -q '^driver *rp1-cfe'; then
            media=$m
            break 2
        fi
    done
    sleep 0.5
done
if [ -z "$media" ]; then
    echo "no rp1-cfe media device after ${WAIT_S} s. Is the camera connected, and does" >&2
    echo "/boot/firmware/config.txt have camera_auto_detect=0 and dtoverlay=arducam-pivariety?" >&2
    exit 1
fi

sensor=$(media-ctl -d "$media" -p | sed -n 's/^- entity [0-9]*: \(arducam-pivariety [^ ]*\) (.*/\1/p' | head -n 1)
if [ -z "$sensor" ]; then
    echo "$media has no arducam-pivariety entity: wrong overlay, or the sensor did not" >&2
    echo "probe (dmesg | grep -i pivariety)" >&2
    exit 1
fi
subdev=$(media-ctl -d "$media" -e "$sensor")
video=$(media-ctl -d "$media" -e rp1-cfe-csi2_ch0)

mc() {
    media-ctl -d "$media" "$@" || {
        echo "media-ctl $* failed. If the camera node is streaming, stop it first." >&2
        exit 1
    }
}
fmt="fmt:$CODE/${WIDTH}x$HEIGHT field:none colorspace:raw"
mc -l '"csi2":4 -> "rp1-cfe-csi2_ch0":0 [1]'
mc -V "\"$sensor\":0 [$fmt]"
mc -V "\"csi2\":0 [$fmt]"
mc -V "\"csi2\":4 [$fmt]"

# frame period = (HEIGHT + vblank) * (WIDTH + hblank) / pixel_rate
pixel_rate=$(v4l2-ctl -d "$subdev" -C pixel_rate | awk '{print $2}')
hblank=$(v4l2-ctl -d "$subdev" -C horizontal_blanking | awk '{print $2}')
vblank=$(awk -v r="$pixel_rate" -v w="$WIDTH" -v hb="$hblank" -v h="$HEIGHT" -v f="$FPS" \
    'BEGIN { printf "%d", r / ((w + hb) * f) - h + 0.5 }')
v4l2-ctl -d "$subdev" -c vertical_blanking="$vblank"
# Read back: the driver clamps to its range (174..16399, i.e. ~60 fps down to ~5 fps).
vblank=$(v4l2-ctl -d "$subdev" -C vertical_blanking | awk '{print $2}')
actual=$(awk -v r="$pixel_rate" -v w="$WIDTH" -v hb="$hblank" -v h="$HEIGHT" -v vb="$vblank" \
    'BEGIN { printf "%.2f", r / ((w + hb) * (h + vb)) }')

echo "p4p camera pipeline: $sensor ($subdev) on $media -> $video," \
     "$CODE ${WIDTH}x$HEIGHT at $actual fps (vblank $vblank)"
if [ "$video" != /dev/video0 ]; then
    echo "note: the capture node is $video, not /dev/video0: launch the camera node" \
         "with device:=$video" >&2
fi
