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
    """상한은 '여러 번의 하강 구간'에 걸쳐 누적된 총합을 막는다.

    연장은 하강 구간당 1회이므로(test_extends_only_once_per_descent_below_threshold),
    상한에 닿으려면 그 사이에 잔여시간이 회복돼 재무장되는 과정을 거쳐야 한다.
    """
    sm = make_sm(threshold=10, cap=8)
    assert sm.evaluate(remaining_time_sec=5, occupied_zones=[3]) == 5   # 누적 5
    sm.evaluate(remaining_time_sec=20, occupied_zones=[3])              # 반영 -> 재무장
    assert sm.evaluate(remaining_time_sec=5, occupied_zones=[3]) == 3   # cap(8)까지만
    assert sm.accumulated_extension_sec == 8
    sm.evaluate(remaining_time_sec=20, occupied_zones=[3])              # 재무장
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


# --- 프레임 루프 안전성: 하강 구간당 1회만 연장 (버그 A) ---

def test_extends_only_once_per_descent_below_threshold():
    """실시간 루프는 evaluate()를 매 프레임 부른다. 그때마다 누적되면 안 된다.

    사람이 3번 구역에 계속 서 있어도, 잔여시간이 임계값 아래로 '내려온 한 번'에 대해
    연장은 한 번만 발급돼야 한다. (수정 전에는 [5, 5, 5, 0, ...] 처럼 프레임마다
    누적돼 0.3초 만에 상한을 소진하고 제어부로 명령을 연타로 보냈다.)
    """
    sm = make_sm(threshold=10, cap=15)
    grants = [sm.evaluate(remaining_time_sec=4, occupied_zones=[3]) for _ in range(5)]

    assert grants == [5, 0, 0, 0, 0]
    assert sm.accumulated_extension_sec == 5
    assert sm.state == SignalState.EXTENDED


def test_rearms_after_remaining_time_recovers():
    """연장이 제어부에 반영돼 잔여시간이 임계값 위로 올라가면 다시 연장할 수 있어야 한다.

    느린 보행자가 한 사이클에서 두 번 연장받는 경로 — 상한이 이를 최종적으로 막는다.
    """
    sm = make_sm(threshold=10, cap=15)

    assert sm.evaluate(remaining_time_sec=4, occupied_zones=[3]) == 5
    assert sm.evaluate(remaining_time_sec=9, occupied_zones=[3]) == 0   # 아직 임계값 아래
    assert sm.evaluate(remaining_time_sec=12, occupied_zones=[3]) == 0  # 반영됨 -> 재무장
    assert sm.state == SignalState.NORMAL
    assert sm.evaluate(remaining_time_sec=4, occupied_zones=[3]) == 5   # 두 번째 연장
    assert sm.accumulated_extension_sec == 10


def test_arming_is_consumed_by_granting_not_by_descending():
    """임계값 아래로 내려온 뒤 뒤늦게 보행자가 들어와도 연장은 발급돼야 한다.

    '하강할 때 무장을 소모'하는 구현이면 이 케이스를 놓친다 — 무장은 '연장 발급'으로만 소모된다.
    """
    sm = make_sm(threshold=10, cap=15)

    assert sm.evaluate(remaining_time_sec=4, occupied_zones=[]) == 0   # 아직 아무도 없음
    assert sm.evaluate(remaining_time_sec=3, occupied_zones=[]) == 0
    assert sm.evaluate(remaining_time_sec=2, occupied_zones=[3]) == 5  # 이제 들어옴


# --- 사이클 경계: reset() (버그 B) ---

def test_new_cycle_allows_extension_after_cap():
    """상한에 도달했어도 다음 보행 신호 사이클에서는 다시 연장할 수 있어야 한다.

    수정 전에는 reset()을 아무도 부르지 않아, 첫 사이클에서 상한을 찍으면 영구히 CAPPED였다.
    """
    sm = make_sm(threshold=10, cap=5)
    assert sm.evaluate(remaining_time_sec=4, occupied_zones=[3]) == 5
    assert sm.evaluate(remaining_time_sec=12, occupied_zones=[3]) == 0
    assert sm.evaluate(remaining_time_sec=4, occupied_zones=[3]) == 0
    assert sm.state == SignalState.CAPPED

    sm.reset()

    assert sm.accumulated_extension_sec == 0
    assert sm.state == SignalState.NORMAL
    assert sm.evaluate(remaining_time_sec=4, occupied_zones=[3]) == 5
