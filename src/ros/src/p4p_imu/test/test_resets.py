"""Fault handling, driven through the mock's injectable faults.

Each test here reaches a path that otherwise needs misbehaving hardware, and
three of them exist because of how the vendor library behaves rather than how the
chip does: it hands over placeholders before data flows, it serves its cache
forever once the chip stops streaming, and it raises bare builtins on a reset.
"""


def test_a_spontaneous_reset_is_counted_and_streaming_resumes(make_rig):
    rig = make_rig(mock_reset_after=10)
    rig.wait_for_samples(10)
    rig.spin(1.0)
    assert rig.node._resets >= 1
    before = len(rig.samples)
    rig.spin(1.0)
    assert len(rig.samples) > before, 'publishing did not resume after the reset'


def test_heading_after_a_reset_is_published_uncompensated(make_rig):
    """POLICY. The node counts and logs a reset but must NOT offset heading to
    hide it. If someone later adds "helpful" continuity compensation, this fails.
    """
    from p4p_imu import orientation

    rig = make_rig(mock_reset_after=10, mock_yaw_rate=2.0, mock_noise=0.0)
    rig.wait_for_samples(10)
    rig.spin(1.5)
    assert rig.node._resets >= 1

    after = [orientation.yaw_from_quaternion(
        (m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w))
        for m in rig.samples[-5:]]
    # The mock re-zeroes at the reset and keeps turning from there, so a short
    # while later heading is still small. Compensation would have carried it on
    # from the pre-reset value instead.
    assert min(abs(y) for y in after) < 1.0


def test_a_silently_stalled_sensor_is_noticed_and_restarted(make_rig):
    """The library raises NOTHING here: it serves the same cached tuple forever.
    Only comparing values catches it, which is why staleness is measured on the
    readings rather than on whether a read succeeded."""
    rig = make_rig(mock_stall_after=20, data_timeout=0.3, status_period=0.2)
    rig.wait_for_samples(20)
    rig.spin(2.0)
    assert rig.node._empty_polls > 0
    assert rig.node._reconnects >= 1, 'the watchdog never restarted the sensor'


def test_nothing_is_published_during_the_placeholder_window(make_rig):
    """enable_feature() returns carrying a zero quaternion and zero acceleration.
    Publishing those would fabricate a yaw of 0 and claim free fall."""
    rig = make_rig(mock_placeholder_reads=10000)
    rig.spin(1.5)
    assert rig.samples == []
    assert rig.node._invalid > 0 or rig.node._reconnects > 0


def test_a_sensor_that_will_not_open_is_retried_and_never_faked(make_rig):
    """No fallback to the mock, ever -- here, no fallback to anything. The node
    comes up, publishes nothing, and keeps retrying."""
    rig = make_rig(mock_fail_open=3, reconnect_period=0.1)
    rig.spin(0.2)
    assert rig.samples == []
    assert rig.wait_for_samples(1, timeout_s=10.0)


def test_the_node_starts_even_with_no_sensor_at_all(make_rig):
    """The house rule: never refuse to start for a hardware reason."""
    rig = make_rig(mock_fail_open=100000, reconnect_period=0.1)
    rig.spin(1.0)
    assert rig.samples == []
    assert rig.node._last_open_error is not None
