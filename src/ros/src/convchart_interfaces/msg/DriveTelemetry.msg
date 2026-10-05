# One decoded CSV telemetry row from the Mega (p4p_arduino, 50 Hz).
#
# Everything here is measured or applied by the firmware. Nothing in it is a
# request -- the request went the other way, over cmd_vel.

std_msgs/Header header

# millis() at the last IMU sample, as the Mega sees it. Quantised to the BNO's
# 100 Hz report rate so it is slightly older than header.stamp, and it wraps
# every 49.7 days. Use it to order samples, not as a ROS time.
uint32 mega_t_ms

# Continuous, unwrapped yaw. NEVER wraps at +-pi: a filter takes differences of
# it, so do not normalise this into (-pi, pi]. Relative rather than absolute --
# the BNO's game rotation vector uses no magnetometer, so it drifts slowly.
float64 heading_rad

# Calibrated gyro Z. The Mega sends deg/s on the wire; this is REP-103 rad/s.
float64 yaw_rate

# The twist the chassis actually acted on, after uniform saturation scaling --
# NOT what was asked for. Zero while disarmed or while the firmware watchdog is
# expired. This, not the cmd_vel request, is the input a predictor should use.
geometry_msgs/Twist applied

bool armed          # E received, S not yet
bool timed_out      # no V arrived within the firmware's 400 ms watchdog
bool saturated      # the last command was scaled down to fit the wheels
uint8 flags         # the raw bitfield, kept so a log can be replayed

uint16 imu_resets   # spontaneous BNO085 resets; should stay 0
uint16 bad_commands # malformed command lines the Mega has rejected so far
