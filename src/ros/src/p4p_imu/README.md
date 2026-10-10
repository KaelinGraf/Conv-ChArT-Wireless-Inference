# p4p_imu

The BNO085 9-DOF fusion IMU on the Pi's I2C bus. It owns the sensor and nothing
else — no filtering, no fusing, no control.

The chip used to hang off the Arduino Mega, and `p4p_serial_bridge` republished
its heading. It is wired to the Pi now, and this node replaces that path.

## Interfaces

| direction | name | type | notes |
|---|---|---|---|
| out | `imu/data` | `sensor_msgs/Imu` | orientation, angular velocity and linear acceleration, all three axes of each |

`KEEP_LAST`, `RELIABLE`, and a depth derived as `0.1 s × sample_rate_hz` (floor
10) — a filter wants every sample, not just the freshest, which is the same
reasoning as the bridge's telemetry profile. The depth holds a fixed amount of
**time** rather than a fixed count, because a flat depth 10 is 100 ms of slack at
100 Hz but only 50 ms at 200 Hz. Reliable also matches a `BEST_EFFORT` subscriber
(anything using `SensorDataQoS`), where the reverse would not, and a QoS mismatch
makes a topic silent with no error on either side. `test_imu_stream.py` subscribes with `SensorDataQoS` and an
`incompatible_qos` callback so that stays true.

Which SH-2 reports feed which field:

| field | report | note |
|---|---|---|
| `orientation` | `GAME_ROTATION_VECTOR` (0x08) | no magnetometer, so yaw is *relative* and drifts slowly. Roll and pitch are referenced against gravity and are real |
| `angular_velocity` | `GYROSCOPE` (0x02) | rad/s, calibrated, all three axes |
| `linear_acceleration` | `ACCELEROMETER` (0x01) | m/s² **including gravity**, which is what `sensor_msgs/Imu` specifies. `LINEAR_ACCELERATION` (0x04) is the gravity-compensated report; it is deliberately not used here |

## No unwrapped heading any more

The bridge used to carry heading twice: unwrapped on `drive/telemetry` for this
project's filter, and as a quaternion on `imu/data` for off-the-shelf consumers.
This node publishes the standard message only, so **nothing publishes an
unwrapped heading.** A quaternion cannot represent one — it wraps at ±π by
construction — so a consumer that differences heading has to unwrap it itself.

`test_heading_wraps_at_pi_and_nothing_unwraps_it` asserts the wrap rather than
assuming it, so the limitation stays visible in behaviour and not only in prose.
If a filter later needs the unwrapped value, the honest fix is a small custom
message alongside `imu/data`, not a non-standard `imu/data`.

`heading_rad`, `yaw_rate` and `imu_resets` still arrive on `drive/telemetry`
because the firmware still sends those columns. **Nothing is behind them.** Take
heading and yaw rate from here.

## Three things the vendor library does that shape this node

**It hands over placeholders before data flows.** `enable_feature()` returns as
soon as the feature id appears in its readings dict, and the entry it waits for
is seeded from `_INITIAL_REPORTS` as a zero quaternion and zero acceleration. A
node that published what it was handed would emit a fabricated yaw of exactly 0
and claim the bot is in free fall, for its first few cycles. Hence
`orientation.validate`, and hence a `start()` that waits for a sample that passes
it (`warmup_timeout`).

**A reset looks like success.** There is no new-data flag and no exposed sensor
timestamp, and the readings dict holds last-known values. When the chip resets
and stops streaming, every read keeps returning the same tuple with no error, for
as long as you care to ask. Staleness is therefore measured on the **values**
(`orientation.samples_differ` plus `data_timeout`), not on whether a read
succeeded. This is the only path that catches a silent reset.

**A reset raises bare builtins.** The post-reset advertisement packet carries a
report id that is not in the library's length table, so it surfaces as a plain
`KeyError`; a reset-complete resolves to the wrong length and raises
`RuntimeError`. Catching only `RuntimeError` would kill the read thread on
precisely the event being counted, which is why `backends/bno08x.py` catches
`KeyError`, `IndexError`, `ValueError`, `struct.error` and `RuntimeError` and
then classifies them.

## Timestamps are the ROS clock, not the sensor's

The SH-2 protocol carries a report timebase (`0xFB`) and the library parses it,
then drops it without exposing it. So unlike `p4p_camera`, which back-dates to
capture time, there is nothing here to back-date against.

The stamp is taken immediately **before** the read, because the sample the sensor
is about to hand over was generated before we asked. It still carries an
uncharacterised latency of roughly half a poll period plus one report interval
plus the I2C transfer — order 10–15 ms at 100 Hz. **That matters to anything
fusing this against the camera's properly back-dated stamps**, and it is not
something this node can fix.

## Wiring

Adafruit BNO085 breakout (4754), I2C:

| Pi | BNO085 |
|---|---|
| 5V (pin 2 or 4) | Vin |
| GND | GND |
| GPIO2 / SDA (**pin 3**) | SDA |
| GPIO3 / SCL (**pin 5**) | SCL |

5V on Vin is safe: the 4754 has a 3–5V regulator and its SDA/SCL are level
shifted with a 10K pullup, so the Pi's 3.3V logic is not exposed to 5V. **A bare
BNO085 module without that regulator and level shifter is not safe this way** —
check the board before powering it.

Pins 3 and 5 are the hardware bus, `/dev/i2c-1`. The host needs
`dtparam=i2c_arm=on` and `dtparam=i2c_arm_baudrate=400000` in
`/boot/firmware/config.txt` and a reboot; see `docker/README.md`.

## Parameters

| name | default | notes |
|---|---|---|
| `imu_backend` | `bno08x` | or `mock`. A typo is **fatal**, never clamped |
| `i2c_bus` | `1` | `/dev/i2c-N`. A software `i2c-gpio` bus is not 1 |
| `i2c_address` | `74` | `0x4A`. The SparkFun BNO086 is `0x4B`. Decimal because ROS parameters have no hex type |
| `sample_rate_hz` | `150.0` | the sensor's report interval, and so the expected output rate. Clamped to [1, 400]. **Not a datasheet stable request point** — see the rate section |
| `oversample` | `2.0` | how much faster than that we **poll**. Clamped to ≥ 1. See below — this is not a tuning knob, it stops real sample loss |
| `frame_id` | `base_link` | what the bridge used. There is no tf tree in this repo, so this is informational |
| `data_timeout` | `0.5` | longest run with **no change** in any reading before the stream counts as dead. Floored at two poll periods |
| `warmup_timeout` | `2.0` | how long `start()` waits for a sample that is not the library's placeholder |
| `reconnect_period` | `1.0` | retry interval while the sensor is missing. Same name and semantics as the bridge's and the camera's |
| `status_period` | `5.0` | heartbeat interval |
| `jump_tolerance` | `0.35` | largest yaw step, in radians, the gyro need not account for before a reset is suspected |
| `var_yaw` | `0.01` | `(0.1 rad)²` |
| `var_roll_pitch` | `1e6` | **uncharacterised, not unmeasured** — see below |
| `var_gyro` | `1e-4` | `(0.01 rad/s)²`, applied to all three axes |
| `var_accel` | `0.01` | `(0.1 m/s²)²`, applied to all three axes |
| `mock_yaw_rate` | `0.4` | rad/s the mock turns at |
| `mock_noise` | `0.01` | seeded noise amplitude |
| `mock_seed` | `0xB0085` | determinism |
| `mock_reset_after` | `0` | samples after which the mock reports one spontaneous reset and re-zeroes heading |
| `mock_stall_after` | `0` | samples after which it returns the identical sample forever, with no error |
| `mock_fail_open` | `0` | `start()` calls to fail |
| `mock_placeholder_reads` | `0` | reads served as the library's zero placeholder |

Every covariance is a placeholder carried over from the removed bridge code, and
they are parameters so that characterising the BNO is a config change rather than
a patch.

**No covariance carries the `-1.0` sentinel.** The bridge set
`linear_acceleration_covariance[0] = -1.0` — the `sensor_msgs/Imu` marker for
"not measured" — because the Mega never sent acceleration. This node measures
ax/ay/az, so a −1 would now be a false statement that tells `robot_localization`
to discard the very data the node exists to publish. Two regression tests pin
that, because re-adding it while copying from the bridge is the obvious mistake.

`var_roll_pitch` at `1e6` says "do not trust this yet", **not** "not measured":
the game rotation vector does reference roll and pitch against gravity. The
marker for genuinely unmeasured is `-1.0` in element 0, and it would condemn the
whole quaternion.

## Degraded behaviour

The node **never refuses to start for a hardware reason.** No sensor, no I2C, no
Adafruit library: it comes up, logs the reason at `error` on a 5 s throttle,
publishes nothing, and retries every `reconnect_period`.

There is **no fallback to the mock, ever.** A synthetic heading feeding a filter
that steers a robot is worse than no heading, because it looks exactly like a
working IMU. `mock` is a first-class backend you select deliberately.

| condition | what happens |
|---|---|
| sensor absent, or the library not installed | throttled `error`, nothing published, retried every `reconnect_period` |
| placeholder window after enabling | samples dropped, `invalid` counter rises, nothing published |
| sensor resets itself | `resets` counter, throttled `error` saying continuity is broken, reports re-enabled in place |
| sensor stops streaming silently | `stale polls` rise, then the watchdog notices and the read thread reopens it (`reconnects`) |
| heading jumps with no matching gyro | `jumps` counter and a throttled warning. The raw heading is published **uncompensated** |
| I2C I/O error | handle dropped, reopened on the next pass |

**Resets are counted and logged, never compensated.** After a reset the chip's
heading restarts from wherever the bot is pointing, and this node publishes that
raw value. Hiding the step with an offset would make a broken heading look
continuous, which is worse than a visible discontinuity. `test_resets.py` asserts
this, so a later "helpful" compensation fails the suite.

Health is a log heartbeat rather than a topic — `sensor_msgs/Imu` has nowhere to
put it, and adding a status topic would contradict the one-standard-message
decision:

```
streaming: 4812 samples at 149.2 Hz (asked 150.0), polling 300 Hz with 4794 empty,
0 invalid, 1 resets, 0 jumps, 0 reconnects, read 1.4 ms max, yaw +2.983 rad
```

The heartbeat reports the **measured** rate, not the configured one. That is the
instrument for the question below.

## Why we poll faster than the sensor reports

**Polling at exactly the report rate loses samples.** `time.sleep` only ever
overshoots — measured on a dev box at ~100 µs median with 1.5 ms outliers — so an
exactly-matched poll period is in practice a little *longer* than the sensor's
report period. Reports then accumulate between polls, and the vendor library
keeps only the newest per feature, so the older one is silently gone. Simulated
against the measured jitter:

| report rate | `oversample` | delivered | lost | empty polls |
|---|---|---|---|---|
| 100 Hz | 1.0 | 99.0 Hz | 1.0% | 0% |
| 100 Hz | 2.0 | 100.0 Hz | 0% | 49% |
| 200 Hz | 1.0 | 196.1 Hz | 2.0% | 0% |
| 200 Hz | 2.0 | 200.0 Hz | 0% | 48% |

The loss is small but it is *invisible*: it reads exactly like "the CPU cannot
keep up", and it would send you tuning the wrong thing. At `oversample 2.0` it
goes to zero, and the cost is empty polls — about half of them — where an empty
poll is a single 4-byte header read, the cheapest transaction on the bus.

Empty polls are therefore **expected**, and the heartbeat reports the poll rate
separately from the publish rate so the two are never confused. Only a *sustained*
run of empty polls means anything, and that is `data_timeout`'s job.

## Rate: what is actually achievable

**150 Hz is the default, pending verification on the board.** It is above the
100 Hz the Mega's BNO ran at and inside the chip's 400 Hz per-report ceiling, but
it is **not** one of the datasheet's stable request points (below), so the chip
will quantise it. Nothing here is verified on hardware yet, and the heartbeat's
measured rate is the only authority.

```bash
ros2 topic hz /imu/data                                   # at the 150 Hz default
ros2 launch p4p_imu imu.launch.py sample_rate_hz:=200      # a native gyro point
ros2 launch p4p_imu imu.launch.py sample_rate_hz:=100      # fall back
```

If 150 delivers something odd, 200 and 100 are the two nearest native gyro points
and are the first things to compare against.

What is known:

**The node's own Python cost is not the limit.** Measured at **13.5 µs per
cycle** for the whole read-validate-normalise-yaw-jump pipeline — 0.27% of the
5 ms budget at 200 Hz. What is unmeasured is the I2C transfer, the vendor
library's packet parsing, and the DDS publish.

**The chip's per-report maxima are 400 Hz** for the game rotation vector and the
calibrated gyro. But the BNO08x datasheet (§6.9) also says the sensors **cannot
all run at their maximum rate simultaneously**, so asking three reports for 400 Hz
will not get you 400 Hz on any of them.

**Neither rate is a clean request point for all three reports.** The datasheet
lists stable request points of 25/33/50/100/200/400 Hz for the gyro and
15/31/62/125/250/500 Hz for the accelerometer. Those sets do not intersect, so at
*any* configured rate at least one report is quantised to a neighbouring value.
At 200 Hz the gyro and game rotation vector land on a native point and the
accelerometer quantises to 125 or 250.

That is harmless here but worth knowing: `read()` returns the newest of each
report, so a slower accelerometer simply repeats its last value between its own
updates. Output stays at the gyro/quaternion rate because `samples_differ` sees
those change. **It does mean `linear_acceleration` is not sampled at
`sample_rate_hz`** — do not assume otherwise in a filter.

If the measured rate sits below about 0.95 × asked, or `read … ms max` approaches
the report period:

1. Confirm `dtparam=i2c_arm_baudrate=400000` is actually set on the host. At
   100 kHz the bus itself becomes the limit.
2. Raise `oversample` before lowering the rate — the loss may be aliasing, not CPU.
3. Drop `sample_rate_hz` to `100`, or `50` to match the Mega's telemetry row rate.
4. `SYS_NICE` and `rtprio` are already granted to the container in
   `compose.yaml`, so renicing the read thread is available without a config
   change.
5. Past that, the BNO08x's gyro-integrated rotation vector (`0x2A`) streams at up
   to 1 kHz and is the report designed for this, but the Adafruit driver does not
   expose it and a Python poll loop is the wrong shape for that rate anyway.

## Clock stretching, the known hardware risk

The BNO085's I2C violates the spec on clock-stretch release: its SDA-to-SCL setup
time is marginal. Pi 1–4 (BCM283x/BCM2711) have a matching controller bug, and
the combination is the classic reason this chip is painful on a Pi. **The Pi 5's
RP1 controller reportedly handles clock stretching correctly**, which is why the
hardware bus is the default here — but that is Raspberry Pi's claim rather than
an independently verified one, and early Pi 5 kernels shipped I2C timing bugs
since fixed, so keep the kernel current.

If it bites anyway, the symptom is `OSError` storms in the heartbeat's reconnect
count. The remedy is software I2C, which bit-bangs and handles stretching
properly, on the host:

```
dtoverlay=i2c-gpio,bus=3,i2c_gpio_sda=23,i2c_gpio_scl=24   # GPIO23/24 = pins 16/18
```

then rewire to those pins, pass `/dev/i2c-3` into the container, and run with
`-p i2c_bus:=3`. Expect roughly 100 kHz, which means dropping `sample_rate_hz`.
The bus number is a parameter precisely so this costs no code change.

## Backends

`bno08x` is the sensor. `mock` is deterministic, seeded and hardware-free, and
exists so that the validity gate, the staleness watchdog, the reset counter, the
reconnect loop and QoS negotiation are all exercised by the test suite. Its fault
knobs (`mock_reset_after`, `mock_stall_after`, `mock_fail_open`,
`mock_placeholder_reads`) each reach a path that otherwise needs misbehaving
hardware.

The arithmetic lives in `orientation.py`, which imports neither ROS nor adafruit,
so the part most likely to be wrong is reachable from a plain pytest. Same split
as `p4p_camera`'s `frames.py` and `p4p_serial_bridge`'s `protocol.py`.

## Running

```bash
colcon build --symlink-install --packages-select p4p_imu
ros2 launch p4p_imu imu.launch.py
ros2 launch p4p_imu imu.launch.py imu_backend:=mock       # no sensor needed
ros2 launch p4p_imu imu.launch.py sample_rate_hz:=100     # fall back from 150
ros2 launch p4p_imu imu.launch.py sample_rate_hz:=200     # read the rate section
```

Checks:

```bash
i2cdetect -y 1                     # expect 4a. Stop the node first: this probes with writes
ros2 topic hz /imu/data            # against sample_rate_hz
ros2 topic echo /imu/data --once   # confirm no -1.0 in any covariance
ros2 topic echo /imu/data --field orientation
ros2 topic echo /imu/data --field angular_velocity
ros2 topic echo /imu/data --field linear_acceleration
```

Turn the board by hand: `orientation` should change about z, `angular_velocity.z`
should track the turn rate and be **positive counter-clockwise** (REP-103, +z
up), and at rest `linear_acceleration.z` should read about +9.81 with x and y near
zero. If z reads −9.81 the board is upside down; if gravity appears in x or y it
is not mounted flat, and the fix belongs in `orientation.py`, not downstream.

## Tests

```bash
colcon test --packages-select p4p_imu --event-handlers console_direct+
colcon test-result --verbose
```

The pure tier needs neither ROS nor the sensor nor the Adafruit stack:

```bash
PYTHONPATH=src/ros/src/p4p_imu python3 -m pytest \
  src/ros/src/p4p_imu/test/test_orientation.py \
  src/ros/src/p4p_imu/test/test_mock_backend.py
```

The graph tier (`test_imu_stream.py`, `test_params.py`, `test_resets.py`) skips
cleanly without `rclpy`. On an x86 host, `export RMW_IMPLEMENTATION=rmw_fastrtps_cpp`
first — CycloneDDS cannot create a node under QEMU, as `docker/README.md` records.
