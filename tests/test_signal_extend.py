import pytest

from src.signal_extend.state_machine import SignalExtensionStateMachine, SignalState


def make_sm(threshold=10, step=5, cap=15):
    return SignalExtensionStateMachine(
        remaining_time_threshold_sec=threshold,
        extension_step_sec=step,
        max_total_extension_sec=cap,
    )


def test_no_extension_when_remaining_time_sufficient():
    sm = make_sm()
    added = sm.evaluate(remaining_time_sec=20, pedestrian_resident=True)
    assert added == 0
    assert sm.state == SignalState.NORMAL


def test_no_extension_when_no_pedestrian():
    sm = make_sm()
    added = sm.evaluate(remaining_time_sec=5, pedestrian_resident=False)
    assert added == 0


def test_extends_by_step_when_conditions_met():
    sm = make_sm(threshold=10, step=5, cap=15)
    added = sm.evaluate(remaining_time_sec=5, pedestrian_resident=True)
    assert added == 5
    assert sm.accumulated_extension_sec == 5
    assert sm.state == SignalState.EXTENDING


def test_respects_upper_cap():
    sm = make_sm(threshold=10, step=5, cap=8)
    first = sm.evaluate(remaining_time_sec=5, pedestrian_resident=True)
    second = sm.evaluate(remaining_time_sec=5, pedestrian_resident=True)
    assert first == 5
    assert second == 3  # cap(8)까지만 채움
    assert sm.accumulated_extension_sec == 8

    third = sm.evaluate(remaining_time_sec=5, pedestrian_resident=True)
    assert third == 0
    assert sm.state == SignalState.CAPPED


def test_missing_config_raises():
    with pytest.raises(NotImplementedError):
        SignalExtensionStateMachine(
            remaining_time_threshold_sec=None,
            extension_step_sec=5,
            max_total_extension_sec=None,
        )
