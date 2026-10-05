# p4p_serial_bridge

The USB serial link between ROS 2 on the Pi and the `p4p_arduino` firmware on the
Mega. It owns the port and nothing else — no filtering, no fusing, no control.

## Interfaces

| direction | name | type |
|---|---|---|
| in | `cmd_vel` | `geometry_msgs/Twist` |
| out | `drive/telemetry` | `convchart_interfaces/DriveTelemetry` |
| out | `drive/status` | `convchart_interfaces/DriveStatus` |
| out | `imu/data` | `sensor_msgs/Imu` |
| service | `~/arm` | `std_srvs/SetBool` — `true` arms (`E`), `false` is the e-stop (`S`) |

**Why two topics and not one.** A topic is one-directional. A node subscribing to
what it publishes builds a feedback loop, the two directions carry unrelated
types, and the controller would have to filter the bridge's own telemetry back
out of its command stream. Commands in, measurements out, separately.

The firmware frame is already REP-103 — `+x` forward, `+y` left, `+z` up, right
handed — so `Twist` maps straight through with no sign conversion. If the bot
drives the wrong way, flip the trims in the firmware's `config.h`, not here.

## `drive/status` — are we running, stopped or stalled?

`DriveTelemetry` carries what the *firmware* measured. `DriveStatus` carries what
the *bridge* knows, which the telemetry cannot express: a stalled controller and a
deliberate "hold station" look identical there, since both show `applied = 0`
while armed.

| state | meaning | what to do |
|---|---|---|
| `NO_LINK` | port not open: missing device, no permission, or an I/O error dropped it | check the cable and `ARDUINO_PORT`; the bridge retries every `reconnect_period` |
| `NO_TELEMETRY` | port open but nothing arriving — still booting, or the Mega went quiet | normal for a moment after opening (DTR resets the Mega); otherwise the firmware has hung |
| `DISARMED` | streaming, not armed; the firmware stores velocity and acts on none of it | call `~/arm` with `data: true` |
| `STALLED` | armed, but no fresh `cmd_vel` — the bridge is sending zeros to keep the firmware watchdog fed | the controller is late or dead; the bot holds station |
| `RUNNING` | armed and acting on a fresh `cmd_vel` | nothing |

A commanded **zero is `RUNNING`, not `STALLED`** — the controller is alive and
asking the bot to hold still. That distinction is the whole point of the topic.

`detail` carries the state name plus the reason, written to be readable straight
out of `ros2 topic echo`:

```
STALLED: cmd_vel is 6.05 s old
NO_LINK: [Errno 2] could not open port /dev/ttyACM0
```

Also on the message: `want_armed`, the bridge's *intent* as opposed to the
firmware's `DriveTelemetry.armed` — if the two disagree for longer than a
telemetry period, an `E` or an `S` did not take effect and nothing else will say
so. Plus `cmd_age` and `telemetry_age` in seconds (`-1` if that has never
happened), `dropped_rows` for telemetry that would not parse, and `reconnects`. Published at `status_rate_hz` as a heartbeat **and immediately on
every state change**, so a transition is never delayed by the tick. Durability is
transient-local, so a filter or supervisor that starts after the bridge learns the
current state on connection rather than waiting.

> `ros2 topic echo --once` prints "A message was lost!!!" against this topic. That
> is an artifact of a reader that exits after one sample while a transient-local
> backlog is still queued — a persistent subscriber sees none.

## Two things that will bite you if you change them

**The downlink is a fixed-rate timer, not the `cmd_vel` callback.** The firmware
zeroes the motors if no `V` arrives within 400 ms, so a controller that goes
quiet — or just publishes unevenly — stalls the bot. The timer re-sends the newest
command at `command_rate_hz` regardless, and substitutes zeros once that command
is older than `cmd_timeout`. Two watchdogs in series: ours holds station and keeps
feeding theirs; theirs catches us dying altogether.

**`heading_rad` on `drive/telemetry` is unwrapped and must stay that way.** It
passes ±π without discontinuity, because a filter takes differences of it. The
quaternion on `imu/data` necessarily wraps, so `imu/data` is for off-the-shelf
consumers like `robot_localization` — the filter in this project should read
`drive/telemetry`. `applied` there is the twist the chassis actually acted on
after saturation scaling, which is the input a predictor wants, not the request.

## Parameters

| name | default | notes |
|---|---|---|
| `port` | `/dev/ttyACM0` | |
| `baud` | `115200` | fixed by the firmware's `config.h` |
| `command_rate_hz` | `20.0` | 8 lost commands of slack before the 400 ms watchdog |
| `cmd_timeout` | `0.25` | our staleness limit; **clamped below the firmware's 0.4** on startup, because above it our zeros would feed the firmware watchdog a stale velocity forever |
| `read_rate_hz` | `200.0` | |
| `auto_arm` | `false` | arms as soon as the Mega streams, **and again after any reboot or reconnect**. Left false, a reboot requires a fresh `~/arm` |
| `max_linear` | `0.5` | `WHEEL_MAX_MPS` |
| `max_angular` | `2.08` | `WHEEL_MAX_MPS / CHASSIS_L_PLUS_W` |
| `publish_imu` | `true` | |
| `status_rate_hz` | `2.0` | heartbeat only; transitions publish immediately |
| `telemetry_timeout` | `0.5` | silence past this reads as `NO_TELEMETRY` |
| `frame_id` | `base_link` | |
| `reconnect_period` | `1.0` | retry interval while the port is missing |

`max_linear` / `max_angular` come from the firmware's **uncalibrated**
`WHEEL_MAX_MPS` placeholder. Until it is measured, commanded velocities are
proportional but not accurate.

They bound each axis *on its own*, which is **not** the chassis limit. `drive.cpp`
saturates on the worst wheel, so the real envelope is

```
|vx| + |vy| + CHASSIS_L_PLUS_W * |wz|  <=  WHEEL_MAX_MPS      (0.24, 0.5)
```

A command inside both clamps can still come back scaled with `saturated` set —
`(0.3, 0.3, 1.0)` echoes as `(0.179, 0.179, 0.595)`. Every rate parameter is
floored on startup too; `read_rate_hz:=0` used to divide by zero.

### Arming and reboots

The firmware boots disarmed and zeroes its stored velocity on `E`, so that arming
can never make the bot lurch off on a stale command. The bridge preserves that: on
any link loss **or** firmware reboot it forgets the last `cmd_vel` and holds
station until the controller speaks again. A Mega 2560's USB CDC is a separate
ATmega16U2, so the reset button, a brownout or a watchdog reboot restart the
sketch *without* dropping `/dev/ttyACM0` — the bridge detects that from the boot
banner rather than assuming a reset always closes the port.

An arm request that fails to reach the wire does not latch, so an operator told
"failed" cannot have the bridge arm the motors by itself when the cable is next
plugged in. A failed *disarm* does latch: intending to stop is always safe.

## Running it

```bash
colcon build --packages-select convchart_interfaces p4p_serial_bridge
ros2 launch p4p_serial_bridge bridge.launch.py
ros2 service call /serial_bridge/arm std_srvs/srv/SetBool "{data: true}"
ros2 topic pub --rate 20 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}}"
```

With no Mega attached, run the firmware simulator and point `port` at it:

```bash
python3 tools/fake_mega.py --port /tmp/ttyFakeMega     # prints the path
ros2 launch p4p_serial_bridge bridge.launch.py port:=/tmp/ttyFakeMega
```

Both must be on the same side of the container boundary. Docker gives a container
its own `devpts`, so a pty made on the host cannot be passed in through
`devices:` — the node appears with the right major/minor and every read fails
with `EIO`. Run the simulator *inside* the container.

The wire format lives in `protocol.py`, with no ROS and no pyserial imported, so
`tests/test_serial_protocol.py` round-trips it against `tools/fake_mega.py` on a
machine with no ROS installed:

```bash
python3 -m pytest tests/test_serial_protocol.py -q
```
