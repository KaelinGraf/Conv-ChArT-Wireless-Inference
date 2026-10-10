"""IMU backends: one interface, two implementations, chosen by parameter.

The node talks only to ImuBackend, so the code that cannot be tested without the
sensor is confined to bno08x.py. mock.py exists so that the whole node -- the
validity gate, the staleness watchdog, the reset counter, the reconnect loop, QoS
negotiation -- is exercised by the test suite and can be brought up on a bench
with no sensor attached.
"""
from __future__ import annotations

import abc
import importlib

from ..orientation import ImuSample

BACKENDS = {
    'bno08x': ('.bno08x', 'Bno08xBackend'),
    'mock': ('.mock', 'MockBackend'),
}


class ImuError(RuntimeError):
    """The sensor could not be opened, configured or read."""


class ImuResetError(ImuError):
    """The sensor reset ITSELF, and is recoverable in place.

    Distinct from a plain ImuError because the remedy differs: a reset means the
    chip is alive but has forgotten its configuration, so recover() re-enables
    the reports without rebuilding the bus. A plain ImuError means the chip is
    not answering, and the node drops the handle and reopens from scratch.

    It also carries the one fact the caller must publish honestly: heading has
    re-zeroed, so continuity across this sample is broken and nothing downstream
    can tell from the data alone.
    """


class ImuBackend(abc.ABC):
    """One sensor, owned by one thread.

    The node guarantees that only its read thread ever calls start, read, recover
    or stop. The watchdog therefore ASKS the read thread to restart the backend
    rather than touching it itself -- the same rule as the camera backends, for a
    weaker but real reason: the recovery paths here sleep for over a second
    inside the vendor library, and a ROS timer that blocked for that long would
    take the heartbeat down with it.

    Deliberately NOT part of this interface: a read timeout. The camera's read
    takes one because a frame may genuinely not have arrived yet; here a read is
    a handful of I2C transactions against the library's cached state and returns
    immediately, so a timeout the backend could not enforce would be a lie. The
    unbounded waits all live in start() and recover(), which is why they run on
    the read thread and why the node's join must allow for them.
    """

    name: str = 'backend'

    @abc.abstractmethod
    def start(self) -> None:
        """Open the bus, configure the sensor, and wait for the first VALID sample.

        Returning means streaming, not merely configured. That distinction is the
        whole point: the vendor library's enable_feature() returns as soon as a
        PLACEHOLDER appears in its readings dict, so a start() that stopped there
        would report success on a sensor that never streams, and the node would
        publish a fabricated zero heading.

        Raises ImuError on any failure. The node logs it, publishes nothing, and
        retries -- it never silently substitutes a different backend.
        """

    @abc.abstractmethod
    def read(self) -> ImuSample:
        """Return one COHERENT sample: all three vectors from a single instant.

        Coherence is the contract. The three reports arrive in separate packets,
        and a sample built from acceleration at one instant and rotation at the
        next is wrong in a way nothing downstream can detect.

        Raises ImuResetError when the sensor has reset itself and ImuError on any
        other I/O failure. It does NOT validate the sample -- orientation.validate
        is the node's job, so that the placeholder window is counted and logged in
        one place rather than two.
        """

    @abc.abstractmethod
    def recover(self) -> None:
        """Re-configure a sensor that reset itself, without rebuilding the bus.

        Raises ImuError if that does not work, which the node escalates to a full
        reopen. Blocking: the vendor library sleeps for roughly a second here.
        """

    @abc.abstractmethod
    def stop(self) -> None:
        """Release the sensor and the bus.

        Idempotent, and must not raise: it is called from the read thread's
        finally and from the node's shutdown path, where an exception would cost
        us the rest of the teardown.
        """


def make_backend(name: str, params: dict, logger) -> ImuBackend:
    """Instantiate a backend by name, importing its module lazily.

    Lazy so that a machine without adafruit_blinka can still import the node and
    run the test suite -- which is most machines, since the Adafruit stack is
    pip-only and installed just in the Pi image. A missing dependency surfaces
    from start() as an ImuError, where the node's retry loop handles it, rather
    than at import time.

    There is deliberately NO fallback to the mock. A synthetic heading feeding a
    filter that steers a robot is worse than no heading at all, because it looks
    exactly like a working IMU. An unavailable backend is an error to retry, never
    a reason to invent data.
    """
    try:
        module_name, class_name = BACKENDS[name]
    except KeyError:
        raise ImuError(
            f'unknown imu_backend {name!r}; expected one of {sorted(BACKENDS)}') from None
    module = importlib.import_module(module_name, __package__)
    return getattr(module, class_name)(params, logger)
