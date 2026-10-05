"""Shared rig for the graph-level camera tests.

Isolation follows convchart_ros/test/test_node_pose_output.py: a per-process
domain id, discovery pinned to localhost and restored afterwards, and a dedicated
namespace, so running the suite can never reach a real robot or a colleague's
machine. The domain base is 37 rather than that file's 77 so the two suites can
run concurrently without colliding.

Every ROS import is deferred into the fixtures, so the pure tests
(test_frames.py, test_mock_backend.py) still collect and run on a machine with
nothing but numpy, cv2 and pytest.
"""
import itertools
import os
import time

import pytest

DOMAIN_ID = 37 + os.getpid() % 20
NAMESPACE = '/camera_test'
TIMEOUT_S = 20.0

# Every Rig gets its own namespace. Without it, a module-scoped rig keeps
# publishing `image` for the whole module while a function-scoped one runs, both
# land on the same topic, and each probe sees the other node's frames -- which
# makes a test asserting "this camera publishes nothing" fail against a different
# camera's output.
_RIG_SEQ = itertools.count()


@pytest.fixture(scope='session')
def ros():
    """Bring rclpy up on an isolated, localhost-only domain.

    Session-scoped on purpose: three test modules want a ROS context, and
    cycling rclpy.init/shutdown between them in one process is needless risk.
    """
    rclpy = pytest.importorskip('rclpy')
    pytest.importorskip('convchart_interfaces.srv',
                        reason='build convchart_interfaces first: it carries '
                               'GetFullResImage')
    pytest.importorskip('convchart_qos.qos',
                        reason='needs convchart_qos for the shared image QoS')
    saved = os.environ.get('ROS_AUTOMATIC_DISCOVERY_RANGE')
    os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'
    # No `-r __ns:=` here: a global namespace remap OVERRIDES the namespace a node
    # is constructed with (verified -- the per-rig namespace had no effect while it
    # was set), which would collapse every Rig back into one namespace and let one
    # rig's frames reach another rig's probe. Each Rig sets its own instead.
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
        'camera_backend': 'mock',
        # Faster than production: these tests wait on frames, and the publisher's
        # offered deadline is derived from this, so it also exercises that path.
        'frame_rate': 20.0,
        'frame_id': 'pi_camera',
        'status_period': 1.0,
    }
    params.update(overrides)
    return [Parameter(name, value=value) for name, value in params.items()]


class Rig:
    """The camera node plus a probe that subscribes exactly as the laptop does."""

    def __init__(self, **overrides):
        import rclpy
        from convchart_interfaces.srv import GetFullResImage
        from convchart_qos.qos import IMAGE_SUB_QOS
        from rclpy.event_handler import SubscriptionEventCallbacks
        from rclpy.executors import MultiThreadedExecutor
        from sensor_msgs.msg import CameraInfo, CompressedImage

        from p4p_camera.camera_node import CameraNode

        self._srv_type = GetFullResImage
        self.images = []
        self.infos = []
        self.incompatible = []

        self.namespace = f'{NAMESPACE}/rig{next(_RIG_SEQ)}'
        self.node = CameraNode(namespace=self.namespace,
                               parameter_overrides=mock_params(**overrides))
        self.probe = rclpy.create_node('camera_test_probe', namespace=self.namespace)

        # The whole point of this rig: subscribe with the REAL profile the
        # inference node uses, imported rather than restated, and notice at once
        # if the publisher is not compatible with it. A QoS mismatch raises on
        # neither side -- the topic is simply silent -- so without this event
        # callback the failure mode is a mystified timeout.
        events = SubscriptionEventCallbacks(
            incompatible_qos=lambda e: self.incompatible.append(e.last_policy_kind.name))
        self.probe.create_subscription(
            CompressedImage, 'image', self.images.append, IMAGE_SUB_QOS,
            event_callbacks=events)
        self.probe.create_subscription(
            CameraInfo, 'camera_info', self.infos.append, IMAGE_SUB_QOS)
        self.client = self.probe.create_client(GetFullResImage, 'image_full_res')

        # Three threads: the node's service and watchdog groups, plus the probe.
        self.executor = MultiThreadedExecutor(num_threads=3)
        self.executor.add_node(self.node)
        self.executor.add_node(self.probe)

    def spin(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.02)

    def check_qos(self):
        assert not self.incompatible, (
            'the camera publisher is QoS-incompatible with IMAGE_SUB_QOS: '
            f'{self.incompatible}. The inference node would receive nothing, '
            'silently.')

    def wait_for_images(self, count=1, timeout_s=TIMEOUT_S):
        """Spin until `count` images have arrived, asserting QoS compatibility."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.check_qos()
            if len(self.images) >= count:
                return list(self.images[:count])
            self.executor.spin_once(timeout_sec=0.02)
        self.check_qos()
        pytest.fail(f'only {len(self.images)} of {count} images arrived on `image` '
                    f'within {timeout_s:.0f} s')

    def call_full_res(self, max_age=0.0, timeout_s=TIMEOUT_S):
        """Call image_full_res and spin until it answers."""
        assert self.client.wait_for_service(timeout_sec=5.0), \
            'image_full_res never appeared on the graph'
        future = self.client.call_async(self._srv_type.Request(max_age=float(max_age)))
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if future.done():
                return future.result()
            self.executor.spin_once(timeout_sec=0.02)
        pytest.fail(f'image_full_res did not answer within {timeout_s:.0f} s')

    def close(self):
        self.executor.shutdown()
        # close() before destroy_node(): the capture thread publishes, and a
        # publish on a destroyed publisher is a segfault, not an exception.
        self.node.close()
        self.probe.destroy_node()
        self.node.destroy_node()


@pytest.fixture(scope='module')
def rig(ros):                      # noqa: ARG001 - the fixture is the ROS context
    """Stream from a camera node on the mock backend, shared across a module."""
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
