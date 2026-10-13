# One decoded CSV telemetry row from the Mega (p4p_arduino, 50 Hz).
#
# Everything here is measured, applied or received by the firmware. The request
# itself went the other way, over cmd_vel; `requested` is the Mega's copy of it.

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

geometry_msgs/Twist requested

uint16[4] pulse_us
