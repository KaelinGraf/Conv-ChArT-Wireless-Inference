# One decoded CSV telemetry row from the Mega (p4p_arduino, 50 Hz).
#
# Everything here is measured or applied by the firmware. Nothing in it is a
# request -- the request went the other way, over cmd_vel.

std_msgs/Header header

uint32 mega_t_ms

float64 heading_rad

float64 yaw_rate

geometry_msgs/Twist applied

bool armed
bool timed_out
bool saturated
uint8 flags

uint16 imu_resets
uint16 bad_commands
