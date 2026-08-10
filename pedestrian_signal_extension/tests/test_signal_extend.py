import pytest

from src.signal_extend import SignalExtensionStateMachine, SignalState

# 담당자 예시 규칙을 테스트용 구체값으로: 1/5=0, 2/4=3초, 3=5초.
ZONE_EXT = {1: 0, 2: 3, 3: 5, 4: 3, 5: 0}


def make_sm(threshold=10, cap=15, zone_ext=None):
    return SignalExtensionStateMachine(
        remaining_time_threshold_sec=threshold,
        zone_extension_sec=zone_ext or ZONE_EXT,
        max_total_extension_sec=cap,
    )


def test_no_extension_when_remaining_time_sufficient():
    sm = make_sm()
    assert sm.evaluate(remaining_time_sec=20, occupied_zones=[3]) == 0
    assert sm.state == SignalState.NORMAL


def test_no_extension_when_no_pedestrian():
    sm = make_sm()
    assert sm.evaluate(remaining_time_sec=5, occupied_zones=[]) == 0


def test_center_zone_extends_most():
    sm = make_sm()
    assert sm.evaluate(remaining_time_sec=5, occupied_zones=[3]) == 5
    assert sm.state == SignalState.EXTENDING


def test_between_zone_extends_less():
    sm = make_sm()
    assert sm.evaluate(remaining_time_sec=5, occupied_zones=[2]) == 3


def test_end_zone_no_extension():
    sm = make_sm()
    assert sm.evaluate(remaining_time_sec=5, occupied_zones=[1]) == 0
    assert sm.state == SignalState.NORMAL


def test_multiple_pedestrians_take_max():
    sm = make_sm()
    # 2번(3초)과 3번(5초)에 각각 있으면 더 필요한 5초에 맞춘다.
    assert sm.evaluate(remaining_time_sec=5, occupied_zones=[2, 3]) == 5


def test_respects_upper_cap():
    sm = make_sm(threshold=10, cap=8)
    assert sm.evaluate(remaining_time_sec=5, occupied_zones=[3]) == 5   # 누적 5
    assert sm.evaluate(remaining_time_sec=5, occupied_zones=[3]) == 3   # cap(8)까지만
    assert sm.accumulated_extension_sec == 8
    assert sm.evaluate(remaining_time_sec=5, occupied_zones=[3]) == 0
    assert sm.state == SignalState.CAPPED


def test_undetermined_zone_extension_raises():
    # 3번 구역 연장 시간이 아직 None이면, 그 구역에 사람이 있을 때 임의 동작 대신 실패.
    sm = make_sm(zone_ext={1: 0, 2: 3, 3: None, 4: 3, 5: 0})
    with pytest.raises(NotImplementedError):
        sm.evaluate(remaining_time_sec=5, occupied_zones=[3])


def test_missing_config_raises():
    with pytest.raises(NotImplementedError):
        SignalExtensionStateMachine(
            remaining_time_threshold_sec=None,
            zone_extension_sec=ZONE_EXT,
            max_total_extension_sec=None,
        )
