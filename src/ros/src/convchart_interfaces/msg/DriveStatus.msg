# Bridge-side health of the serial link and of the command stream.
#
# DriveTelemetry reports what the FIRMWARE measured. This reports what the BRIDGE
# knows: whether the port is open, whether the Mega is streaming, and whether
# anyone is actually commanding it. Those are invisible in the telemetry -- a
# stalled controller and a deliberate "hold station" look identical there, since
# both show applied = 0 while armed.
#
# Published at status_rate_hz and again immediately on every state change, with
# transient-local durability so a node that starts late still learns the current
# state instead of waiting for the next tick.

std_msgs/Header header

uint8 NO_LINK=0
uint8 NO_TELEMETRY=1
uint8 DISARMED=2
uint8 STALLED=3
uint8 RUNNING=4
uint8 state

string detail

bool want_armed

string port
float32 cmd_age
float32 telemetry_age
uint32 dropped_rows
uint32 reconnects
