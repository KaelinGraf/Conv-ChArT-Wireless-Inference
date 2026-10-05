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

# The serial port is not open: missing device, no permission, or an I/O error
# dropped it. The bridge retries every reconnect_period.
uint8 NO_LINK=0
# Port open but no telemetry is arriving. Either the Mega is still booting (it
# resets on DTR when the port opens) or it has gone quiet mid-run.
uint8 NO_TELEMETRY=1
# Streaming, but not armed. The firmware parses and stores velocity in this state
# and acts on none of it.
uint8 DISARMED=2
# Armed, but with no fresh cmd_vel, so the bridge is sending zeros on the
# controller's behalf to keep the firmware watchdog fed. The bot holds station.
uint8 STALLED=3
# Armed and acting on a fresh cmd_vel. Note that a commanded zero is RUNNING, not
# STALLED: the controller is alive and asking the bot to hold still.
uint8 RUNNING=4
uint8 state

# The state's name, plus the reason when there is one. Written to be readable
# straight out of `ros2 topic echo`.
string detail

# What the BRIDGE intends, as opposed to what the firmware reports in
# DriveTelemetry.armed. The two should agree within one telemetry period; if they
# disagree for longer, an E or an S did not take effect and nothing else will say
# so. want_armed false with DriveTelemetry.armed true means a disarm was lost.
bool want_armed

string port
float32 cmd_age         # seconds since the last cmd_vel, or -1 if none ever arrived
float32 telemetry_age   # seconds since the last decoded row, or -1 if none ever arrived
uint32 dropped_rows     # telemetry rows that would not parse
uint32 reconnects       # times the port has been reopened since the node started
