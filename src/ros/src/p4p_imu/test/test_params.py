"""Parameter validation: what is clamped, what is fatal, and what is derived.

The split matters. A rate out of range is a field condition and the node clamps
it loudly, because a bot already on the floor is better off running. A bad
enumeration is a typo, and running the wrong backend silently is worse than
refusing to start.
"""
import pytest


def test_a_rate_below_the_floor_is_clamped_rather_than_fatal(make_rig):
    rig = make_rig(sample_rate_hz=0.1)
    assert rig.node._sample_rate == pytest.approx(1.0)


def test_an_absurd_rate_is_capped(make_rig):
    """Past the cap we would only be buying empty poll cycles."""
    rig = make_rig(sample_rate_hz=100000.0)
    assert rig.node._sample_rate == pytest.approx(400.0)


def test_the_report_period_follows_the_clamped_rate(make_rig):
    """Not the requested one: the sensor's report interval is derived from the
    same number, so a clamp that did not propagate would silently decimate."""
    rig = make_rig(sample_rate_hz=0.1)
    assert rig.node._report_period == pytest.approx(1.0)
    assert rig.node._params['sample_rate_hz'] == pytest.approx(1.0)


# --- oversampling -------------------------------------------------------------

def test_we_poll_faster_than_the_sensor_reports(make_rig):
    """Polling at exactly the report rate aliases and loses samples, because
    time.sleep only ever overshoots and the library keeps only the newest report
    per feature. Measured at ~2% loss at 200 Hz."""
    rig = make_rig(sample_rate_hz=200.0, oversample=2.0)
    assert rig.node._report_period == pytest.approx(1.0 / 200.0)
    assert rig.node._poll_period == pytest.approx(1.0 / 400.0)


def test_oversampling_below_one_is_clamped(make_rig):
    """Polling SLOWER than the sensor reports throws samples away with no way to
    get them back, so it is clamped rather than honoured."""
    rig = make_rig(sample_rate_hz=100.0, oversample=0.25)
    assert rig.node._oversample == pytest.approx(1.0)
    assert rig.node._poll_period == pytest.approx(rig.node._report_period)


def test_the_staleness_timeout_is_floored_on_report_periods_not_poll_periods(make_rig):
    """With oversampling most polls are legitimately empty, so a poll-based floor
    would let the watchdog fire during normal operation."""
    rig = make_rig(sample_rate_hz=10.0, oversample=4.0, data_timeout=0.001)
    assert rig.node._data_timeout >= 2.0 / 10.0


# --- publisher history --------------------------------------------------------

def test_the_queue_holds_a_fixed_time_rather_than_a_fixed_sample_count(make_rig):
    """A flat depth 10 is 100 ms of slack at 100 Hz but only 50 ms at 200 Hz, so
    the faster the stream the sooner a briefly-stalled subscriber loses data."""
    assert make_rig(sample_rate_hz=200.0).node._pub.qos_profile.depth == 20
    assert make_rig(sample_rate_hz=400.0).node._pub.qos_profile.depth == 40


def test_the_queue_never_drops_below_ten(make_rig):
    """At low rates the time-derived depth would be tiny, and a filter still
    wants a few samples of slack."""
    assert make_rig(sample_rate_hz=20.0).node._pub.qos_profile.depth == 10


def test_an_unknown_backend_refuses_to_start(make_rig):
    """A typo, not a field condition. Clamping would run the mock when the sensor
    was asked for, and synthetic headings look exactly like real ones."""
    with pytest.raises(ValueError) as e:
        make_rig(imu_backend='bno085_typo')
    assert 'bno085_typo' in str(e.value)


def test_the_i2c_address_reaches_the_backend(make_rig):
    """0x4B is the SparkFun board; getting this wrong means no sensor at all."""
    rig = make_rig(i2c_address=0x4B)
    assert rig.node._params['i2c_address'] == 0x4B


def test_the_bus_number_reaches_the_backend(make_rig):
    """The software-I2C fallback is a different bus, and must need no code change."""
    rig = make_rig(i2c_bus=3)
    assert rig.node._params['i2c_bus'] == 3
