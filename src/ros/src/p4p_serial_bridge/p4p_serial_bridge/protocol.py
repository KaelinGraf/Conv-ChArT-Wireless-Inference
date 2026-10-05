"""Wire format of the p4p_arduino serial link.

No ROS and no serial port in here on purpose: the parsing and formatting are the
part most likely to be wrong, and this way they can be tested against
tools/fake_mega.py on a machine with no ROS installed.

The authority for all of it is p4p_arduino/README.md plus command.cpp. Uplink is
one CSV row per line at 50 Hz; downlink is E / S / V,<vx>,<vy>,<wz>.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

TELEMETRY_HEADER = 't_ms,heading_rad,yaw_rate_dps,vx,vy,wz,flags,resets,bad'

BOOT_BANNER = '# p4p_arduino starting'

FLAG_ARMED = 0x01
FLAG_TIMEOUT = 0x02
FLAG_SATURATED = 0x04

FIRMWARE_CMD_TIMEOUT_S = 0.400

MAX_LINE_LEN = 47

ARM = b'E\n'
DISARM = b'S\n'


@dataclass(frozen=True)
class Telemetry:
    """One decoded telemetry row. Fields are exactly what the Mega reported."""

    t_ms: int
    heading_rad: float
    yaw_rate_dps: float
    applied: tuple[float, float, float]
    flags: int
    imu_resets: int
    bad_commands: int

    @property
    def armed(self) -> bool:
        """Report whether the firmware has had an E and no S since."""
        return bool(self.flags & FLAG_ARMED)

    @property
    def timed_out(self) -> bool:
        """Report whether the firmware's 400 ms command watchdog has expired."""
        return bool(self.flags & FLAG_TIMEOUT)

    @property
    def saturated(self) -> bool:
        """Report whether the last command was scaled down to fit the wheels."""
        return bool(self.flags & FLAG_SATURATED)


def parse_line(line: str) -> Telemetry | None:
    """Decode one telemetry row, or return None when the line is not one."""
    s = line.strip()
    if not s or s.startswith('#') or s == TELEMETRY_HEADER:
        return None
    parts = s.split(',')
    if len(parts) != 9:
        return None
    try:
        t_ms = int(parts[0])
        heading, yaw_dps, vx, vy, wz = (float(p) for p in parts[1:6])
        flags, resets, bad = (int(p) for p in parts[6:9])
    except ValueError:
        return None
    if not all(math.isfinite(v) for v in (heading, yaw_dps, vx, vy, wz)):
        return None
    return Telemetry(t_ms=t_ms, heading_rad=heading, yaw_rate_dps=yaw_dps,
                     applied=(vx, vy, wz), flags=flags,
                     imu_resets=resets, bad_commands=bad)


def format_velocity(vx: float, vy: float, wz: float) -> bytes:
    """Encode a body twist as a V line."""
    if not all(math.isfinite(v) for v in (vx, vy, wz)):
        raise ValueError(f'non-finite twist: {vx}, {vy}, {wz}')
    line = f'V,{vx:.3f},{vy:.3f},{wz:.3f}\n'
    if len(line) - 1 > MAX_LINE_LEN:
        raise ValueError(f'V line is {len(line) - 1} bytes, over the firmware limit: {line!r}')
    return line.encode('ascii')


def describe_flags(flags: int) -> str:
    """Render the flags bitfield for a log line."""
    names = [n for bit, n in ((FLAG_ARMED, 'ARMED'), (FLAG_TIMEOUT, 'TIMEOUT'),
                              (FLAG_SATURATED, 'SAT')) if flags & bit]
    return '|'.join(names) if names else 'DISARMED'


class LineReader:
    """Reassemble newline-terminated lines from arbitrary serial chunks."""

    def __init__(self, max_len: int = 1024) -> None:
        self._buf = bytearray()
        self._max_len = max_len
        self.overruns = 0

    def feed(self, data: bytes) -> list[str]:
        """Add received bytes and return whatever complete lines that produced."""
        self._buf += data
        out: list[str] = []
        while True:
            i = self._buf.find(b'\n')
            if i < 0:
                break
            raw = bytes(self._buf[:i])
            del self._buf[:i + 1]
            out.append(raw.decode('ascii', 'replace').rstrip('\r'))
        if len(self._buf) > self._max_len:
            self.overruns += 1
            self._buf.clear()
        return out

    def reset(self) -> None:
        """Discard any partial line, after a reconnect or a firmware reset."""
        self._buf.clear()
