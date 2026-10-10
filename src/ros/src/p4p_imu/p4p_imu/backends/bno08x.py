"""The real BNO085, over I2C, through Adafruit's CircuitPython driver.

This is the file that cannot be tested without the sensor, so it holds as little
arithmetic as possible: it opens the bus, enables three SH-2 reports, hands back
raw tuples, and classifies failures. Everything that can be checked on a laptop
lives in orientation.py.

Why ExtendedI2C and not busio.I2C(board.SCL, board.SDA): it takes a bus NUMBER,
so the hardware bus and the software-I2C fallback differ by a parameter rather
than by code, and it avoids importing `board`, whose pin table runs board
detection and is the most fragile part of Blinka inside a container. Note that
its frequency= argument is accepted and IGNORED on Linux -- the bus speed is a
device-tree dtparam, which is why 400 kHz lives in the host's config.txt and
there is no i2c_frequency parameter here.

Three reports rather than one. The game rotation vector alone would give heading,
but a filter wants the gyro's yaw rate directly rather than differenced from a
quaternion, and the accelerometer is free once the bus is open.
"""
from __future__ import annotations

import struct
import time

from . import ImuBackend, ImuError, ImuResetError
from .. import orientation
from ..orientation import ImuSample

# Anything the SH-2 batch parser can throw on a malformed or unexpected report.
# This list is wider than it looks like it needs to be, and every entry is load
# bearing: on a spontaneous reset the chip sends an unsolicited advertisement
# whose report id is not in the library's length table, so the failure surfaces
# as a bare KeyError. Catching only RuntimeError and PacketError would kill the
# read thread on precisely the event we are trying to count.
_PARSE_ERRORS = (KeyError, IndexError, ValueError, struct.error, RuntimeError)

# Substrings in a library exception that mean the chip reset rather than that the
# bus is broken. "No ... report found" is the giveaway: the features we enabled
# have been cleared, which only a reset does.
_RESET_SIGNS = ('unprocessable batch', 'report found', 'is it enabled')


class Bno08xBackend(ImuBackend):
    """One BNO085 on one I2C bus, owned by the node's read thread."""

    name = 'bno08x'

    def __init__(self, params: dict, logger) -> None:
        self._log = logger
        self._bus = int(params.get('i2c_bus', 1))
        self._address = int(params.get('i2c_address', 0x4A))
        self._interval_us = max(1, int(round(1e6 / float(params.get('sample_rate_hz', 100.0)))))
        self._warmup = float(params.get('warmup_timeout', 2.0))
        self._i2c = None
        self._bno = None
        self._reports: tuple[int, int, int] = (0, 0, 0)
        self._aligned = False

    # --- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Open the bus, enable the three reports, and wait for real data."""
        try:
            from adafruit_bno08x import (BNO_REPORT_ACCELEROMETER,
                                         BNO_REPORT_GAME_ROTATION_VECTOR,
                                         BNO_REPORT_GYROSCOPE)
            from adafruit_bno08x.i2c import BNO08X_I2C
            from adafruit_extended_bus import ExtendedI2C
        except ImportError as e:
            raise ImuError(
                f'the Adafruit BNO08x stack is not installed ({e}); it is pip-only and '
                'lives in docker/pi/Dockerfile. Use imu_backend:=mock off the Pi.') from e

        self._reports = (BNO_REPORT_ACCELEROMETER, BNO_REPORT_GYROSCOPE,
                         BNO_REPORT_GAME_ROTATION_VECTOR)
        try:
            self._i2c = ExtendedI2C(self._bus)
            # Constructing this runs initialize(): a soft reset and an ID check,
            # three attempts, around one to three seconds. reset=None because no
            # reset pin is wired, which makes hard_reset() a no-op -- every
            # recovery path here is a soft reset.
            self._bno = BNO08X_I2C(self._i2c, address=self._address, reset=None)
        except _PARSE_ERRORS as e:
            raise ImuError(f'could not reach a BNO08x at {self._address:#04x} on '
                           f'/dev/i2c-{self._bus}: {e}') from e
        except OSError as e:
            raise ImuError(f'/dev/i2c-{self._bus} I/O error: {e}. Is I2C enabled on the '
                           'host (dtparam=i2c_arm=on) and is the container in the right '
                           'group (I2C_GID)?') from e

        # One probe, not a per-read check. Reading the three public properties
        # works, but each one drains the queue itself, so the three vectors can
        # land from different drains -- acceleration from one instant and rotation
        # from the next. One explicit drain plus three dict lookups is both
        # cheaper and time-aligned. These two names are private, hence the probe:
        # if an upstream release moves them we degrade to the slower public path
        # instead of crashing.
        self._aligned = (hasattr(self._bno, '_process_available_packets')
                         and hasattr(self._bno, '_readings'))
        if not self._aligned:
            self._log.warning(
                'adafruit_bno08x no longer exposes _process_available_packets/_readings; '
                'falling back to the public properties. Samples stay correct but the '
                'three vectors may be up to one read cycle apart.')

        self._enable()
        self._await_data()

    def recover(self) -> None:
        """Soft-reset and re-enable, without rebuilding the bus."""
        if self._bno is None:
            raise ImuError('recover before start')
        try:
            self._bno.soft_reset()
        except (*_PARSE_ERRORS, OSError) as e:
            raise ImuError(f'soft reset failed: {e}') from e
        self._enable()
        self._await_data()

    def stop(self) -> None:
        """Release the sensor and the bus. Idempotent, and never raises."""
        self._bno = None
        i2c, self._i2c = self._i2c, None
        if i2c is not None:
            try:
                i2c.deinit()
            except Exception as e:  # noqa: BLE001 - teardown must not raise
                self._log.warning(f'releasing /dev/i2c-{self._bus} failed: {e}')

    # --- reading ------------------------------------------------------------

    def read(self) -> ImuSample:
        """Return one coherent sample, or classify why that failed."""
        if self._bno is None:
            raise ImuError('read before start or after stop')
        try:
            if self._aligned:
                # Drain once, then take all three from the same drain.
                self._bno._process_available_packets()
                readings = self._bno._readings
                accel = readings.get(self._reports[0])
                gyro = readings.get(self._reports[1])
                quat = readings.get(self._reports[2])
            else:
                accel = self._bno.acceleration
                gyro = self._bno.gyro
                quat = self._bno.game_quaternion
        except _PARSE_ERRORS as e:
            raise self._classify(e) from e
        except OSError as e:
            # EREMOTEIO / EIO: the chip is not acknowledging at all. This is the
            # shape a clock-stretching failure takes, not a reset.
            raise ImuError(f'I2C read failed on /dev/i2c-{self._bus}: {e}') from e

        if accel is None or gyro is None or quat is None:
            # The library drops a feature from its dict when the chip forgets it,
            # which is a reset by another name.
            raise ImuResetError('a report vanished from the sensor; features were cleared')
        return ImuSample(accel=tuple(accel), gyro=tuple(gyro), quat=tuple(quat))

    # --- internals ----------------------------------------------------------

    def _enable(self) -> None:
        """Enable the three reports at the configured interval."""
        for report in self._reports:
            try:
                self._bno.enable_feature(report, self._interval_us)
            except (*_PARSE_ERRORS, OSError) as e:
                raise ImuError(f'could not enable report {report:#04x} at '
                               f'{self._interval_us} us: {e}') from e

    def _await_data(self) -> None:
        """Block until a sample passes orientation.validate, or give up.

        This is what makes start() mean "streaming". enable_feature() returns as
        soon as the feature id appears in the library's readings dict, and the
        entry it waits for is a placeholder -- a zero quaternion and zero
        acceleration. Without this wait, start() would succeed against a sensor
        that never streams and the node would publish a fabricated zero heading.
        """
        deadline = time.monotonic() + self._warmup
        why = 'no sample was read'
        while time.monotonic() < deadline:
            try:
                why = orientation.validate(self.read())
            except ImuResetError as e:
                why = str(e)
            if why is None:
                self._log.info(
                    f'BNO08x streaming at {self._address:#04x} on /dev/i2c-{self._bus}, '
                    f'{self._interval_us} us reports, '
                    f'{"aligned" if self._aligned else "property"} reads')
                return
            time.sleep(0.01)
        raise ImuError(f'no valid sample within {self._warmup} s of enabling reports '
                       f'(last reason: {why})')

    @staticmethod
    def _classify(exc: Exception) -> ImuError:
        """Decide whether a parse failure means a reset or a broken link.

        A reset is recoverable in place; anything else costs us the handle. The
        distinction is a string match because the library raises bare builtins
        with no error codes, and getting it wrong only changes how hard we try.
        """
        text = str(exc).lower()
        if isinstance(exc, KeyError) or any(s in text for s in _RESET_SIGNS):
            return ImuResetError(f'sensor appears to have reset: {exc!r}')
        return ImuError(f'sensor read failed: {exc!r}')
