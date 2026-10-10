"""Shared rig for the graph-level IMU tests.

Isolation follows p4p_camera/test/conftest.py: a per-process domain id, discovery
pinned to localhost and restored afterwards, and a dedicated namespace per rig,
so running the suite can never reach a real robot or a colleague's machine. The
domain base is 57 -- 37 is p4p_camera's and 77 is convchart_ros's -- so all three
suites can run concurrently without colliding.

Every ROS import is deferred into the fixtures, so the pure tests
(test_orientation.py, test_mock_backend.py) still collect and run on a machine
with nothing but pytest.
"""
import itertools
import os
import time

import pytest

DOMAIN_ID = 57 + os.getpid() % 20
NAMESPACE = '/imu_test'
TIMEOUT_S = 20.0

# Every Rig gets its own namespace, for the reason p4p_camera's conftest records:
# a module-scoped rig keeps publishing imu/data while a function-scoped one runs,
# and without separate namespaces each probe sees the other node's samples --
# which makes a test asserting "this node publishes nothing" fail against a
# different node's output.
_RIG_SEQ = itertools.count()


@pytest.fixture(scope='session')
def ros():
    """Bring rclpy up on an isolated, localhost-only domain."""
    rclpy = pytest.importorskip('rclpy')
    pytest.importorskip('sensor_msgs.msg', reason='needs sensor_msgs for Imu')
    saved = os.environ.get('ROS_AUTOMATIC_DISCOVERY_RANGE')
    os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'
    # No `-r __ns:=` here: a global namespace remap OVERRIDES the namespace a node
    # is constructed with, which would collapse every Rig back into one namespace.
    rclpy.init(domain_id=DOMAIN_ID)
    try:
        yield rclpy
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        if saved is None:
            os.environ.pop('ROS_AUTOMATIC_DISCOVERY_RANGE', None)
        else:
            os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = saved


def mock_params(**overrides):
    """Parameter overrides for a hardware-free node, as launch would pass them."""
    from rclpy.parameter import Parameter

    params = {
        'imu_backend': 'mock',
        # Faster than production so the tests do not wait: the mock does not
        # sleep, so this only sets the node's own poll cadence.
        'sample_rate_hz': 200.0,
        'frame_id': 'base_link',
        'status_period': 0.5,
        # Turn fast enough that a short test sees heading wrap through +-pi.
        'mock_yaw_rate': 2.0,
    }
    params.update(overrides)
    return [Parameter(name, value=value) for name, value in params.items()]


class Rig:
    """The IMU node plus a probe that subscribes as an off-the-shelf consumer would."""

    def __init__(self, **overrides):
        import rclpy
        from rclpy.event_handler import SubscriptionEventCallbacks
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Imu

        from p4p_imu.imu_node import ImuNode

        self.samples = []
        self.incompatible = []

        self.namespace = f'{NAMESPACE}/rig{next(_RIG_SEQ)}'
        self.node = ImuNode(namespace=self.namespace,
                            parameter_overrides=mock_params(**overrides))
        self.probe = rclpy.create_node('imu_test_probe', namespace=self.namespace)

        # SensorDataQoS, not the node's own profile: this is what a third-party
        # IMU consumer uses, and the point is to prove the node's RELIABLE
        # publisher matches a BEST_EFFORT subscriber. A QoS mismatch raises on
        # neither side -- the topic is simply silent -- so without this event
        # callback the failure mode is a mystified timeout.
        events = SubscriptionEventCallbacks(
            incompatible_qos=lambda e: self.incompatible.append(e.last_policy_kind.name))
        self.probe.create_subscription(
            Imu, 'imu/data', self.samples.append, qos_profile_sensor_data,
            event_callbacks=events)

        # Two threads: the node's heartbeat timer, plus the probe.
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.node)
        self.executor.add_node(self.probe)

    def spin(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.02)

    def check_qos(self):
        assert not self.incompatible, (
            'the IMU publisher is QoS-incompatible with SensorDataQoS: '
            f'{self.incompatible}. An off-the-shelf consumer would receive '
            'nothing, silently.')

    def wait_for_samples(self, count=1, timeout_s=TIMEOUT_S):
        """Spin until `count` samples have arrived, asserting QoS compatibility."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.check_qos()
            if len(self.samples) >= count:
                return list(self.samples[:count])
            self.executor.spin_once(timeout_sec=0.02)
        self.check_qos()
        pytest.fail(f'only {len(self.samples)} of {count} samples arrived on '
                    f'imu/data within {timeout_s:.0f} s')

    def close(self):
        self.executor.shutdown()
        # close() before destroy_node(): the read thread publishes, and a publish
        # on a destroyed publisher is a segfault, not an exception.
        self.node.close()
        self.probe.destroy_node()
        self.node.destroy_node()


@pytest.fixture(scope='module')
def rig(ros):                      # noqa: ARG001 - the fixture is the ROS context
    """Stream from an IMU node on the mock backend, shared across a module."""
    r = Rig()
    try:
        yield r
    finally:
        r.close()


@pytest.fixture
def make_rig(ros):                 # noqa: ARG001
    """Build a rig with custom parameters, torn down after the test."""
    built = []

    def _make(**overrides):
        r = Rig(**overrides)
        built.append(r)
        return r

    try:
        yield _make
    finally:
        for r in built:
            r.close()
