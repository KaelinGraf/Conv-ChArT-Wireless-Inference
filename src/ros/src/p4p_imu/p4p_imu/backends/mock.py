"""A synthetic BNO085, for the test suite and for a bench with no sensor.

Deterministic: the stream is a function of the sample index and the seed, never
of wall-clock time, so two mocks built with one seed produce identical samples
and a test can assert on exact values. Only the PACING uses the real clock.

The motion is kinematically consistent on purpose -- yaw ramps at mock_yaw_rate,
gyro.z carries that same rate, and gravity sits in +z -- so a test that checks
"does angular_velocity.z match the turn" is checking the node rather than
agreeing with a constant. Yaw ramps fast enough to wrap through +-pi inside a
short test, which is how the unwrapping consequence gets asserted rather than
assumed.

The fault knobs are the reason this file exists. Each reaches a path that
otherwise needs misbehaving hardware, and each is a documented parameter rather
than a test-only hook.
"""
from __future__ import annotations

import math
import random

from . import ImuBackend, ImuError, ImuResetError
from .. import orientation
from ..orientation import ImuSample

_GRAVITY = 9.80665


class MockBackend(ImuBackend):
    """Synthetic samples with injectable faults. Never a fallback for real hardware."""

    name = 'mock'

    def __init__(self, params: dict, logger) -> None:
        self._log = logger
        self._period = 1.0 / float(params.get('sample_rate_hz', 100.0))
        self._yaw_rate = float(params.get('mock_yaw_rate', 0.4))
        self._noise = float(params.get('mock_noise', 0.01))
        self._seed = int(params.get('mock_seed', 0xB0085))
        # Samples after which to report a spontaneous reset, once. 0 disables.
        self._reset_after = int(params.get('mock_reset_after', 0))
        # Samples after which to return the SAME sample forever, with no error.
        # The only way to reach the staleness watchdog, since that is exactly what
        # a real stalled BNO looks like through the vendor library.
        self._stall_after = int(params.get('mock_stall_after', 0))
        # start() calls to fail before succeeding: the retry loop, and the proof
        # that nothing falls back to a different backend.
        self._fail_open = int(params.get('mock_fail_open', 0))
        # Reads to serve as the vendor library's placeholder: a zero quaternion
        # and zero acceleration, with no error. The validity gate.
        self._placeholder_reads = int(params.get('mock_placeholder_reads', 0))

        self._rng = random.Random(self._seed)
        self._running = False
        self._n = 0
        self._opens = 0
        self._yaw_offset = 0.0
        self._reset_done = False
        self._stalled: ImuSample | None = None

    def start(self) -> None:
        """Begin streaming, after mock_fail_open failures."""
        self._opens += 1
        if self._opens <= self._fail_open:
            raise ImuError(
                f'mock_fail_open={self._fail_open}: refusing open {self._opens}')
        self._rng = random.Random(self._seed)
        self._running = True
        self._log.info(f'mock imu streaming at {1.0 / self._period:.1f} Hz, '
                       f'yaw_rate={self._yaw_rate} rad/s, seed={self._seed:#x}')

    def read(self) -> ImuSample:
        """Return the next synthetic sample, or inject the configured fault."""
        if not self._running:
            raise ImuError('mock imu read before start or after stop')

        # No pacing here: the NODE owns the cadence, so one place controls it and
        # a pure test of this class runs at full speed instead of sleeping.
        self._n += 1

        if self._placeholder_reads >= self._n:
            # Exactly what the vendor library hands over between enable_feature()
            # and the first real report.
            return ImuSample(accel=(0.0, 0.0, 0.0), gyro=(0.0, 0.0, 0.0),
                             quat=(0.0, 0.0, 0.0, 0.0))

        if self._stall_after and self._n > self._stall_after:
            if self._stalled is None:
                self._stalled = self._build()
            return self._stalled

        if self._reset_after and self._n > self._reset_after and not self._reset_done:
            self._reset_done = True
            # Re-zero heading, as the real chip does: the game rotation vector
            # restarts from wherever the bot happens to be pointing.
            self._yaw_offset = -self._yaw()
            raise ImuResetError(
                f'mock_reset_after={self._reset_after}: sensor reset itself')

        return self._build()

    def recover(self) -> None:
        """Re-enable reports after a reset. Always succeeds for the mock."""
        if not self._running:
            raise ImuError('mock imu recover before start')
        self._log.warning('mock imu recovered; heading has re-zeroed')

    def stop(self) -> None:
        """Stop streaming. Idempotent, never raises."""
        self._running = False
        self._stalled = None

    # --- the synthetic motion ----------------------------------------------

    def _yaw(self) -> float:
        """Yaw from the sample INDEX, not the clock, so the stream is reproducible."""
        return self._yaw_rate * self._n * self._period

    def _build(self) -> ImuSample:
        yaw = self._yaw() + self._yaw_offset
        n = self._noise
        # A lateral wobble on x so ax is not a constant a test could pass by
        # accident; gravity in +z because the board is mounted flat.
        return ImuSample(
            accel=(0.4 * math.sin(yaw) + self._rng.gauss(0.0, n),
                   self._rng.gauss(0.0, n),
                   _GRAVITY + self._rng.gauss(0.0, n)),
            gyro=(self._rng.gauss(0.0, n),
                  self._rng.gauss(0.0, n),
                  self._yaw_rate + self._rng.gauss(0.0, n)),
            quat=orientation.quaternion_from_yaw(yaw),
        )
