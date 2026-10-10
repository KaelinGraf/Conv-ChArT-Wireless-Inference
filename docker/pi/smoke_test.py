#!/usr/bin/env python3
"""Runtime check for the p4p Raspberry Pi container. Run INSIDE the container:

    python3 docker/pi/smoke_test.py

Verifies ROS 2 (rclpy + the chosen RMW), numpy/scipy (Kalman filter maths), pyserial
(lists the serial ports the container can see -- the Arduino Mega 2560 shows up as
/dev/ttyACM0 when passed through), yaml, cv2, and tf_transformations.

The camera and IMU sections are optional and skip cleanly when the Arducam stack
is absent (WITH_CAMERA=0) or no BNO085 is attached, so this script is useful on a
bare board.
"""
import os
import platform
import sys

import cv2
import numpy as np
import scipy
import scipy.linalg
import serial
import yaml
from serial.tools import list_ports

print(f"{platform.machine()}  python {platform.python_version()}  numpy {np.__version__}  "
      f"scipy {scipy.__version__}  pyserial {serial.__version__}  pyyaml {yaml.__version__}  "
      f"cv2 {cv2.__version__}")

# a Kalman-filter-shaped sanity check: covariance update stays symmetric positive definite
P = np.eye(6) * 0.1
H = np.hstack([np.eye(3), np.zeros((3, 3))])
R = np.eye(3) * 0.01
S = H @ P @ H.T + R
K = P @ H.T @ np.linalg.inv(S)
P2 = (np.eye(6) - K @ H) @ P
assert np.all(np.linalg.eigvalsh((P2 + P2.T) / 2) > 0)
scipy.linalg.cho_factor(S)
print("numpy/scipy KF update OK")

ports = [(p.device, p.description) for p in list_ports.comports()]
print("serial ports visible:", ports or "none (pass the Arduino through: devices: /dev/ttyACM0)")
if ports:
    print("  (opening a port toggles DTR, which resets the Arduino Mega -- expected)")
for dev, _ in ports:
    try:
        with serial.Serial(dev, 115200, timeout=0.1):
            print(f"  opened {dev} OK")
    except serial.SerialException as e:
        print(f"  could not open {dev}: {e}")

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.utilities import get_rmw_implementation_identifier
import tf_transformations

rclpy.init()
node = rclpy.create_node("p4p_pi_smoke")
pub = node.create_publisher(PoseWithCovarianceStamped, "p4p/smoke", 1)
msg = PoseWithCovarianceStamped()
q = tf_transformations.quaternion_from_euler(0.0, 0.0, 0.5)
msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = q
pub.publish(msg)
print(f"rclpy OK  rmw={get_rmw_implementation_identifier()}  ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID')}  "
      f"tf_transformations OK")
node.destroy_node()
rclpy.shutdown()

# p4p_camera's geometry: pure numpy/cv2, so it works with or without a camera.
frame = np.zeros((1200, 1600), dtype=np.uint8)
frame[600, 800] = 255
stream = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
ok, buf = cv2.imencode(".png", stream, [cv2.IMWRITE_PNG_COMPRESSION, 1])
back = cv2.imdecode(np.frombuffer(buf.tobytes(), np.uint8), cv2.IMREAD_ANYCOLOR)
assert ok and back.shape == (480, 640) and back.ndim == 2 and back.dtype == np.uint8, \
    "the 1600x1200 -> 640x480 mono PNG path is broken"
print("cv2 OK  1600x1200 -> 640x480 mono8 PNG round trip, 2-D uint8")

# Camera stack: optional. WITH_CAMERA=0 images have no picamera2, which is fine.
try:
    import libcamera
    from picamera2 import Picamera2
    print(f"camera stack present: libcamera {libcamera.__version__ if hasattr(libcamera, '__version__') else '(no version attr)'}")
except ImportError as e:
    print(f"camera stack: not installed ({e}). Expected unless WITH_CAMERA=1; "
          "p4p_camera's mock backend still works.")
else:
    try:
        cam = Picamera2()
    except Exception as e:
        print(f"camera stack: picamera2 imports but no camera is usable: {e}")
        print("  check: the host has camera_auto_detect=0 + dtoverlay=arducam-pivariety,")
        print("         and this container was started with the `camera` compose profile")
        print("         (/run/udev, /dev/media*, /dev/dma_heap all have to be visible).")
    else:
        try:
            # Read sensor_modes ONCE: the property reconfigures the sensor each access.
            modes = cam.sensor_modes
            print(f"camera stack OK: {len(modes)} sensor mode(s)")
            for m in modes:
                print(f"  {str(m.get('format'))!r:>18}  {tuple(m.get('size', ()))}  "
                      f"{m.get('bit_depth')}-bit  {m.get('fps')} fps")
            print("  (paste the above into p4p_camera/backends/picam.py's docstring and")
            print("   into frames.pick_mode's test fixtures)")
        finally:
            cam.close()

# BNO085 on I2C: optional, and the most common failure is permissions rather than
# wiring, so report which of the three layers is missing instead of just "failed".
try:
    import adafruit_bno08x
    from adafruit_extended_bus import ExtendedI2C
except ImportError as e:
    print(f"imu stack: not installed ({e}). p4p_imu's mock backend still works.")
else:
    import glob
    buses = sorted(glob.glob("/dev/i2c-*"))
    print(f"imu stack present: adafruit_bno08x {adafruit_bno08x.__version__}, "
          f"buses visible: {buses or 'NONE'}")
    if not buses:
        print("  no /dev/i2c-*: the host needs dtparam=i2c_arm=on in")
        print("  /boot/firmware/config.txt plus a reboot, and compose must pass the")
        print("  device through. The `pi` service will not even start without it.")
    for dev in buses:
        bus_n = int(dev.rsplit("-", 1)[1])
        i2c = None
        try:
            i2c = ExtendedI2C(bus_n)
            # Bounded, not `while not try_lock()`: a smoke test must never hang.
            for _ in range(100):
                if i2c.try_lock():
                    break
            else:
                print(f"  {dev}: bus stayed locked; is the IMU node running?")
                continue
            # NOTE: scan() probes with writes, so stop p4p_imu before running this.
            found = set(i2c.scan())
        except PermissionError as e:
            print(f"  {dev}: permission denied ({e}). I2C_GID in .env does not match the")
            print("    host's i2c group -- run `getent group i2c | cut -d: -f3` on the Pi.")
            continue
        except Exception as e:                      # noqa: BLE001 - a smoke test reports
            print(f"  {dev}: could not scan: {e!r}")
            continue
        else:
            addrs = ", ".join(hex(a) for a in found) or "nothing"
            note = ""
            if 0x4A in found:
                note = "  <- BNO085 at 0x4a"
            elif 0x4B in found:
                note = "  <- BNO085 at 0x4b (ADR tied high); set i2c_address:=75"
            print(f"  {dev}: {addrs}{note}")
            if not found & {0x4A, 0x4B}:
                print("    no BNO085 here. Check 5V->Vin, GND->GND, SDA->pin 3, SCL->pin 5.")
        finally:
            if i2c is not None:
                try:
                    i2c.unlock()
                except Exception:                   # noqa: BLE001
                    pass
                try:
                    i2c.deinit()
                except Exception:                   # noqa: BLE001
                    pass

print("ALL OK")
sys.exit(0)
