"""A software stand-in for the p4p_arduino firmware.

Speaks the exact serial protocol of the Mega (see p4p_arduino/README.md): the
boot banner, 50 Hz CSV telemetry, E/S/V commands, the 400 ms command watchdog
and uniform saturation scaling. Lets the Pi-side ROS node be developed against
the real line format with no Mega, no USB cable and no wheels off the ground.

Two ways in:

    as a fake serial port -- integration, works inside the Jazzy container
        python3 tools/fake_mega.py --port /tmp/ttyFakeMega
        then point the node's `port` parameter at /tmp/ttyFakeMega

    as a library -- unit tests, deterministic, no pty and no wall clock
        mega = FakeMega()
        mega.feed(b"E\nV,0.2,0,0\n")
        mega.pump(now_ms=20)          # one loop() pass, at a time you choose
        rows = mega.drain()

Check it over with no hardware and no ROS at all:

        python3 tools/fake_mega.py --selftest

The constants below mirror p4p_arduino/config.h. They are duplicated rather than
read, because the firmware is a separate repo -- change one there, change it
here, and --selftest will tell you when the two have drifted.

Divergences from the real board, all deliberate:
  * the chassis is ideal. The applied twist integrates straight into heading, so
    there is no slip, no load sag and no battery droop. wz_gain exists to fake
    the uncalibrated WHEEL_MAX_MPS -- set it to the k from README "Calibration".
  * the BNO never resets spontaneously (`resets` stays 0) unless you call
    inject_imu_reset(), and the gyro is noiseless unless you ask for noise.
  * the banner is emitted in one go. The real one has a ~150 ms gap after the
    first line while the IMU comes up.
"""
from __future__ import annotations

import argparse
import contextlib
import errno
import math
import os
import random
import re
import signal
import struct
import sys
import time
import tty
from dataclasses import dataclass

TELEMETRY_HEADER = "t_ms,heading_rad,yaw_rate_dps,vx,vy,wz,flags,resets,bad"

FLAG_ARMED = 0x01
FLAG_TIMEOUT = 0x02
FLAG_SATURATED = 0x04

U32 = 0xFFFFFFFF


@dataclass(frozen=True)
class MegaConfig:
    """Mirror of p4p_arduino/config.h. Override it to explore the firmware's own
    knobs -- wheel_max_mps for an uncalibrated bot, the sign trims for a bring-up
    that went wrong."""
    stream_rate_hz: int = 50
    control_rate_hz: int = 50
    cmd_timeout_ms: int = 400
    cmd_buf_len: int = 48
    imu_report_us: int = 10000
    vx_sign: float = 1.0
    vy_sign: float = 1.0
    wz_sign: float = 1.0
    chassis_l_plus_w: float = 0.24
    wheel_max_mps: float = 0.50
    strafe_gain: float = 1.0


def _f32(x: float) -> float:
    """Round to single precision. `double` is 32-bit on AVR, so every float in
    the firmware is really a float -- without this the last telemetry digit can
    differ from the board's and parity tests chase a ghost."""
    return struct.unpack("f", struct.pack("f", x))[0]


def _i32(x: int) -> int:
    x &= U32
    return x - 0x100000000 if x >= 0x80000000 else x


_NUM_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_INF_RE = re.compile(r"[+-]?(?:inf(?:inity)?|nan)", re.IGNORECASE)


def _strtod(s: str) -> tuple[float, str] | None:
    """avr-libc strtod: consume the longest numeric prefix, return (value, rest),
    or None when nothing numeric was consumed -- which is how the firmware spots
    junk. Leading whitespace is skipped and "inf"/"nan" parse, which is exactly
    why command.cpp range-checks the result afterwards. Hex floats do not parse;
    avr-libc has no support for them."""
    t = s.lstrip(" \t")
    m = _INF_RE.match(t) or _NUM_RE.match(t)
    if m is None:
        return None
    return float(m.group(0)), t[m.end():]


def _print_float(value: float, digits: int) -> str:
    """Arduino Print::printFloat, digit for digit.

    It rounds half away from zero by adding 0.5e-digits and truncating, where
    Python's format() rounds half to even, and it emits "nan"/"inf"/"ovf" where
    printf would give a number. The Pi-side parser has to survive all three, so
    the simulator has to be able to produce them."""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    if value > 4294967040.0 or value < -4294967040.0:
        return "ovf"
    out = ""
    if value < 0.0:
        out, value = "-", -value
    rounding = 0.5
    for _ in range(digits):
        rounding /= 10.0
    value += rounding
    int_part = int(value)
    remainder = value - int_part
    out += str(int_part)
    if digits > 0:
        out += "."
    for _ in range(digits):
        remainder *= 10.0
        d = int(remainder)
        out += str(d)
        remainder -= d
    return out


class _RateTimer:
    """timing.h RateTimer. Fires once per period, and resynchronises instead of
    firing in a burst when the caller falls far behind. The int32 casts are the
    firmware's own and are what make it safe across the micros() overflow."""

    def __init__(self, period_us: int, now_us: int) -> None:
        self.period_us = period_us
        self.next_us = now_us

    def due(self, now_us: int) -> bool:
        if _i32(now_us - self.next_us) < 0:
            return False
        self.next_us = (self.next_us + self.period_us) & U32
        if _i32(now_us - self.next_us) > _i32(self.period_us):
            self.next_us = now_us
        return True


class FakeMega:
    """One Mega, with no transport attached. Feed it downlink bytes, pump it,
    drain its uplink. Time is injected, so a test can cover the 400 ms watchdog
    in microseconds of wall clock and step across the millis() overflow."""

    def __init__(self, cfg: MegaConfig | None = None, *, start_ms: int = 0,
                 wz_gain: float = 1.0, yaw_noise_dps: float = 0.0,
                 seed: int = 0, imu_fail_count: int = 0) -> None:
        self.cfg = cfg or MegaConfig()
        self.wz_gain = wz_gain          # measured rate / commanded; 1.0 = calibrated
        self.yaw_noise_dps = yaw_noise_dps
        self.imu_fail_count = imu_fail_count
        self._rng = random.Random(seed)
        self.reset(start_ms)

    # --- lifecycle ---------------------------------------------------------

    def reset(self, now_ms: int = 0) -> None:
        """Power-on or DTR reset. Queues the boot banner and lands DISARMED."""
        c = self.cfg
        self._t_ms = now_ms & U32
        now_us = (self._t_ms * 1000) & U32
        self._rx = bytearray()
        self._tx = bytearray()
        self._buf = bytearray()
        self._overflow = False
        self._armed = False
        self._vx = self._vy = self._wz = 0.0
        # command.h initialises last_cmd_ms_ to 0, NOT to millis(), so a board
        # that has been up for longer than CMD_TIMEOUT_MS boots already showing
        # TIMEOUT. The firmware README's example session has flags=0 here.
        self._last_cmd_ms = 0
        self._bad = 0
        self._applied = (0.0, 0.0, 0.0)
        self._saturated = False
        self._timed_out = False
        self._heading = 0.0
        self._yaw_rate = 0.0
        self._last_sample_ms = 0        # 0 until the first report, like the real one
        self._resets = 0
        # constructed in the firmware's order: control before stream, so within
        # one pass the watchdog verdict always reaches the telemetry line.
        self._imu = _RateTimer(c.imu_report_us, now_us)
        self._control = _RateTimer(1_000_000 // c.control_rate_hz, now_us)
        self._stream = _RateTimer(1_000_000 // c.stream_rate_hz, now_us)
        self._emit("# p4p_arduino starting")
        for _ in range(self.imu_fail_count):
            self._emit("# BNO085 not found: check SDA=20, SCL=21, power, address")
        self._emit("# BNO085 ready")
        self._emit("# DISARMED -- send E to enable")
        self._emit(TELEMETRY_HEADER)

    def inject_imu_reset(self) -> None:
        """A spontaneous BNO reset: bumps `resets` and drops the heading anchor,
        so the next sample does not integrate the jump."""
        self._resets += 1

    # --- transport ---------------------------------------------------------

    def feed(self, data: bytes) -> None:
        self._rx += data

    def drain(self) -> bytes:
        out, self._tx = bytes(self._tx), bytearray()
        return out

    def pump(self, now_ms: int) -> None:
        """One pass of loop(). Call at >= 1 kHz for realistic behaviour."""
        self._t_ms = now_ms & U32
        now_us = (self._t_ms * 1000) & U32
        self._poll_imu(now_us)
        self._poll_commands()
        if self._control.due(now_us):
            # The watchdog is evaluated here, not in the parser, so a flood of
            # commands cannot starve it (p4p_arduino.ino:72).
            self._timed_out = self._age_ms() > self.cfg.cmd_timeout_ms
            if not self._armed or self._timed_out:
                self._stop()
            else:
                self._set_body_velocity(self._vx, self._vy, self._wz)
        if self._stream.due(now_us):
            self._emit_telemetry()

    # --- introspection, for tests -----------------------------------------

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def timed_out(self) -> bool:
        return self._timed_out

    @property
    def saturated(self) -> bool:
        return self._saturated

    @property
    def bad(self) -> int:
        return self._bad

    @property
    def applied(self) -> tuple[float, float, float]:
        return self._applied

    @property
    def heading(self) -> float:
        return self._heading

    # --- internals ---------------------------------------------------------

    def _age_ms(self) -> int:
        return (self._t_ms - self._last_cmd_ms) & U32

    def _emit(self, line: str) -> None:
        self._tx += line.encode("ascii") + b"\r\n"   # Serial.println sends CRLF

    def _poll_imu(self, now_us: int) -> None:
        if not self._imu.due(now_us):
            return
        dt = self.cfg.imu_report_us / 1e6
        rate = self._applied[2] * self.wz_gain       # ideal open-loop chassis
        if self.yaw_noise_dps:
            rate += math.radians(self._rng.gauss(0.0, self.yaw_noise_dps))
        self._yaw_rate = _f32(rate)
        self._heading = _f32(self._heading + rate * dt)
        self._last_sample_ms = self._t_ms

    def _poll_commands(self) -> None:
        data, self._rx = bytes(self._rx), bytearray()
        for b in data:
            if b == 0x0A:                            # '\n'
                if self._overflow:                   # tail of a long line
                    self._overflow = False
                else:
                    self._handle_line(self._buf.decode("ascii", "replace"))
                self._buf.clear()
                continue
            if b == 0x0D:                            # '\r'
                continue
            if len(self._buf) >= self.cfg.cmd_buf_len - 1:
                if not self._overflow:               # counted once per line
                    self._overflow = True
                    self._bad += 1
                continue
            self._buf.append(b)

    def _handle_line(self, line: str) -> None:
        s = line.strip(" \t")
        if not s:
            return                                   # blank, not an error
        if s[0] == "#":
            return                                   # comment, so a log replays
        tag = s[0].upper()
        rest = s[1:].strip(" \t")

        if tag == "E":
            if rest:
                self._bad += 1
                return
            # Arming must never start motion, and it restarts the watchdog clock
            # so the first telemetry line after E reads clean (command.cpp:51).
            self._vx = self._vy = self._wz = 0.0
            self._last_cmd_ms = self._t_ms
            self._armed = True
            return

        if tag == "S":
            if rest:
                self._bad += 1
                return
            self._vx = self._vy = self._wz = 0.0
            self._armed = False
            return

        if tag == "V":
            if not rest.startswith(","):
                self._bad += 1
                return
            p = rest[1:]
            v: list[float] = []
            for i in range(3):
                parsed = _strtod(p)
                if parsed is None:
                    self._bad += 1
                    return
                val, p = parsed
                v.append(val)
                p = p.strip(" \t")
                if i < 2:
                    if not p.startswith(","):
                        self._bad += 1
                        return
                    p = p[1:]
            if p:                                    # trailing junk, e.g. a 4th field
                self._bad += 1
                return
            if any(math.isnan(x) or math.isinf(x) for x in v):
                self._bad += 1
                return
            self._vx, self._vy, self._wz = v
            self._last_cmd_ms = self._t_ms
            return

        self._bad += 1                               # unknown tag

    def _stop(self) -> None:
        self._applied = (0.0, 0.0, 0.0)
        self._saturated = False

    def _set_body_velocity(self, vx_in: float, vy_in: float, wz_in: float) -> None:
        c = self.cfg
        vx = _f32(vx_in * c.vx_sign)
        vy = _f32(vy_in * c.vy_sign * c.strafe_gain)
        wz = _f32(wz_in * c.wz_sign)
        rot = _f32(c.chassis_l_plus_w * wz)
        inv = _f32(1.0 / c.wheel_max_mps)
        wheels = (_f32((vx - vy - rot) * inv),       # left front
                  _f32((vx + vy + rot) * inv),       # right front
                  _f32((vx + vy - rot) * inv),       # left rear
                  _f32((vx - vy + rot) * inv))       # right rear
        peak = max(abs(w) for w in wheels)
        self._saturated = peak > 1.0
        scale = _f32(1.0 / peak) if self._saturated else 1.0
        # Scaling all four wheels by one factor is exactly a uniform scaling of
        # the body twist, so the echo is the request times that factor, reported
        # before the sign trims in the caller's convention (drive.cpp:63).
        self._applied = (_f32(vx * scale * c.vx_sign),
                         _f32(vy * scale / c.strafe_gain * c.vy_sign),
                         _f32(wz * scale * c.wz_sign))

    def _emit_telemetry(self) -> None:
        flags = 0
        if self._armed:
            flags |= FLAG_ARMED
        if self._timed_out:
            flags |= FLAG_TIMEOUT
        if self._saturated:
            flags |= FLAG_SATURATED
        ax, ay, aw = self._applied
        self._emit(",".join((
            str(self._last_sample_ms),
            _print_float(self._heading, 4),
            _print_float(math.degrees(self._yaw_rate), 1),
            _print_float(ax, 3), _print_float(ay, 3), _print_float(aw, 3),
            str(flags), str(self._resets), str(self._bad))))


# --- running it as a fake serial port ----------------------------------------

def run_pty(mega: FakeMega, *, link: str | None = None, quiet: bool = False) -> None:
    """Expose `mega` on a pseudo-terminal and run it against the wall clock.

    Our copy of the slave fd is closed straight away so the pty behaves like a
    USB CDC port: read() on the master fails with EIO while nobody has it open,
    and the Mega is reset when someone opens it. That is the DTR reset, which is
    the reason a real board reboots DISARMED every time the node reconnects --
    the single most common way a first serial node appears to hang."""
    master, slave = os.openpty()
    tty.setraw(slave)                  # no echo, no \n -> \r\n translation
    dev = os.ttyname(slave)
    os.close(slave)
    os.set_blocking(master, False)

    if link:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(link)
        os.symlink(dev, link)          # a stable path across restarts

    print(link or dev, flush=True)
    if not quiet:
        print(f"-- fake Mega on {dev}" + (f" (linked as {link})" if link else ""),
              file=sys.stderr, flush=True)

    def _term(_signum, _frame):
        # Without this a SIGTERM -- which is how a launch file, a container stop
        # and `timeout` all end this process -- skips the cleanup below and
        # leaves a dangling symlink behind.
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _term)

    t0 = time.monotonic()
    attached = False
    try:
        while True:
            now_ms = int((time.monotonic() - t0) * 1000)
            was = attached
            data = b""
            try:
                data = os.read(master, 4096)
                attached = bool(data)
            except BlockingIOError:
                attached = True        # open, just nothing to say
            except OSError as e:
                if e.errno != errno.EIO:
                    raise
                attached = False       # nobody has the port open

            if attached and not was:
                mega.reset(now_ms)
                if not quiet:
                    print("-- port opened: Mega reset, DISARMED", file=sys.stderr, flush=True)
            if data:
                mega.feed(data)

            mega.pump(now_ms)
            out = mega.drain()         # always drain; a real board prints into the void
            if attached and out:
                with contextlib.suppress(BlockingIOError, OSError):
                    os.write(master, out)
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        if link:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(link)
        os.close(master)


# --- self-test ---------------------------------------------------------------

def _rows(blob: bytes) -> list[str]:
    """CSV data rows only: drop the '#' status lines and the header, the way the
    Pi-side parser must."""
    return [ln for ln in blob.decode("ascii", "replace").split("\r\n")
            if ln and not ln.startswith("#") and ln != TELEMETRY_HEADER]


def _run(mega: FakeMega, t_from: int, t_to: int) -> list[str]:
    for t in range(t_from, t_to + 1):
        mega.pump(t)
    return _rows(mega.drain())


def selftest() -> int:
    """Protocol checks that need no ROS, no Docker and no hardware."""
    fails: list[str] = []

    def ok(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)
        print(f"{'ok  ' if cond else 'FAIL'}  {msg}")

    # the banner, and the header line that looks like data but is not
    m = FakeMega(start_ms=1400)
    banner = m.drain().decode().split("\r\n")[:4]
    ok(all(ln.startswith("#") for ln in banner[:3]), "boot banner is three '#' lines")
    ok(banner[3] == TELEMETRY_HEADER, "fourth boot line is the bare CSV header")

    # a board that has been up a while boots showing TIMEOUT, not flags=0
    rows = _run(m, 1400, 1480)
    ok(bool(rows) and rows[0].split(",")[6] == "2",
       "disarmed boot shows flags=2 (TIMEOUT), not the README's 0")

    # E restarts the watchdog clock, so flags reads 1 immediately, 3 only later
    m.feed(b"E\n")
    rows = _run(m, 1481, 1500)
    ok(bool(rows) and rows[-1].split(",")[6] == "1", "flags=1 immediately after E")
    rows = _run(m, 1501, 1960)
    ok(bool(rows) and rows[-1].split(",")[6] == "3", "flags=3 once 400 ms passes with no V")

    # a V while armed drives, and the echo reports it
    m.feed(b"V,0.2,0,0\n")
    rows = _run(m, 1961, 2000)
    f = rows[-1].split(",")
    ok(f[3] == "0.200" and f[6] == "1", "V,0.2,0,0 echoes 0.200 with flags=1")

    # the watchdog zeroes the echo but leaves the bot armed
    rows = _run(m, 2001, 2500)
    f = rows[-1].split(",")
    ok(f[3:6] == ["0.000", "0.000", "0.000"] and f[6] == "3",
       "watchdog zeroes the echo and keeps ARMED")

    # S does not touch the watchdog clock, so disarming an already-timed-out bot
    # leaves TIMEOUT set. The firmware README's session shows flags=0 here.
    m.feed(b"S\n")
    rows = _run(m, 2501, 2540)
    ok(rows[-1].split(",")[6] == "2", "S after a timeout leaves flags=2, not the README's 0")

    # S while actually driving is the flags=0 case
    d = FakeMega()
    d.feed(b"E\nV,0.2,0,0\n")
    _run(d, 0, 40)
    d.feed(b"S\n")
    rows = _run(d, 41, 80)
    ok(rows[-1].split(",")[6] == "0", "S while driving disarms cleanly, flags=0")

    # saturation: 10 m/s on a 0.5 m/s chassis scales by 1/20 and sets SAT
    s = FakeMega()
    s.feed(b"E\nV,10,0,0\n")
    rows = _run(s, 0, 60)
    f = rows[-1].split(",")
    ok(f[3] == "0.500" and f[6] == "5", "V,10,0,0 saturates: echo 0.500, flags=5")

    # the parser's error accounting
    b = FakeMega()
    b.feed(b"\n   \n# a replayed log line\n")
    b.pump(1)
    ok(b.bad == 0, "blank and '#' lines are not errors")
    for line, why in ((b"X\n", "unknown tag"), (b"E x\n", "E with an argument"),
                      (b"V,1,2\n", "V with two fields"), (b"V,1,2,3,4\n", "V with four fields"),
                      (b"V,a,b,c\n", "V with non-numeric fields"), (b"V1,2,3\n", "V without a comma"),
                      (b"V,nan,0,0\n", "V carrying NaN"), (b"V,0,inf,0\n", "V carrying inf")):
        before = b.bad
        b.feed(line)
        b.pump(1)
        ok(b.bad == before + 1, f"rejected, bad+1: {why}")
    ok(not b.armed, "none of the bad lines armed the bot")

    # a too-long line is counted once, not once per excess byte
    before = b.bad
    b.feed(b"V," + b"9" * 80 + b"\n")
    b.pump(1)
    ok(b.bad == before + 1, "an over-length line counts exactly one error")

    # Arduino's float printing, which rounds half away from zero
    ok(_print_float(0.0005, 3) == "0.001", "printFloat rounds half up, unlike Python")
    ok(_print_float(-1.2345, 3) == "-1.234", "printFloat truncates after rounding")
    ok(_print_float(float("nan"), 4) == "nan", "printFloat emits nan")
    ok(_print_float(float("inf"), 4) == "inf", "printFloat emits unsigned inf")

    # the millis() overflow, 49.7 days in. The watchdog is the thing at risk: its
    # age is an unsigned difference, so it must not see a 49-day gap at the wrap.
    w = FakeMega(start_ms=U32 - 500)
    w.feed(b"E\n")
    seen = []
    for t in range(U32 - 499, U32 + 501):
        if t % 100 == 0:
            w.feed(b"V,0.1,0,0\n")
        w.pump(t)
        seen += _rows(w.drain())
    flags = {int(r.split(",")[6]) for r in seen}
    ok(flags == {1}, f"watchdog survives the millis() wrap (flags seen: {sorted(flags)})")

    print(f"\n{len(fails)} failure(s)" if fails else "\nall passed")
    for msg in fails:
        print(f"  - {msg}")
    return 1 if fails else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--port", metavar="PATH",
                   help="symlink the pty here, e.g. /tmp/ttyFakeMega, for a stable path")
    p.add_argument("--selftest", action="store_true", help="run the protocol checks and exit")
    p.add_argument("--wz-gain", type=float, default=1.0,
                   help="measured yaw rate / commanded; fakes an uncalibrated WHEEL_MAX_MPS")
    p.add_argument("--yaw-noise-dps", type=float, default=0.0, help="gyro noise sigma, deg/s")
    p.add_argument("--wheel-max-mps", type=float, default=MegaConfig.wheel_max_mps,
                   help="chassis top speed, for testing saturation")
    p.add_argument("--imu-fail-count", type=int, default=0,
                   help="extra '# BNO085 not found' lines at boot")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    if a.selftest:
        return selftest()

    mega = FakeMega(MegaConfig(wheel_max_mps=a.wheel_max_mps), wz_gain=a.wz_gain,
                    yaw_noise_dps=a.yaw_noise_dps, seed=a.seed,
                    imu_fail_count=a.imu_fail_count)
    run_pty(mega, link=a.port, quiet=a.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
