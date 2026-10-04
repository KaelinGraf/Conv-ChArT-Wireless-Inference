#!/usr/bin/env python3
"""Runtime check for the p4p Raspberry Pi container. Run INSIDE the container:

    python3 docker/pi/smoke_test.py

Verifies ROS 2 (rclpy + the chosen RMW), numpy/scipy (Kalman filter maths), pyserial
(lists the serial ports the container can see -- the Arduino Mega 2560 shows up as
/dev/ttyACM0 when passed through), yaml, and tf_transformations.
"""
import os
import platform
import sys

import numpy as np
import scipy
import scipy.linalg
import serial
import yaml
from serial.tools import list_ports

print(f"{platform.machine()}  python {platform.python_version()}  numpy {np.__version__}  "
      f"scipy {scipy.__version__}  pyserial {serial.__version__}  pyyaml {yaml.__version__}")

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
print("ALL OK")
sys.exit(0)
