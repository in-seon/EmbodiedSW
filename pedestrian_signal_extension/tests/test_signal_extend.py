import pytest

from config import config
from src.signal_extend import Occupant, SignalExtensionStateMachine, SignalState

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
    assert sm.evaluate(remaining_time_sec=20, occupants=[3]) == 0
    assert sm.state == SignalState.NORMAL


def test_no_extension_when_no_pedestrian():
    sm = make_sm()
    assert sm.evaluate(remaining_time_sec=5, occupants=[]) == 0


def test_center_zone_extends_most():
    sm = make_sm()
    assert sm.evaluate(remaining_time_sec=5, occupants=[3]) == 5
    assert sm.state == SignalState.EXTENDING


def test_between_zone_extends_less():
    sm = make_sm()
    assert sm.evaluate(remaining_time_sec=5, occupants=[2]) == 3


def test_end_zone_no_extension():
    sm = make_sm()
    assert sm.evaluate(remaining_time_sec=5, occupants=[1]) == 0
    assert sm.state == SignalState.NORMAL


def test_multiple_pedestrians_take_max():
    sm = make_sm()
    # 2번(3초)과 3번(5초)에 각각 있으면 더 필요한 5초에 맞춘다.
    assert sm.evaluate(remaining_time_sec=5, occupants=[2, 3]) == 5


def test_respects_upper_cap():
    """상한은 '여러 번의 하강 구간'에 걸쳐 누적된 총합을 막는다.

    연장은 하강 구간당 1회이므로(test_extends_only_once_per_descent_below_threshold),
    상한에 닿으려면 그 사이에 잔여시간이 회복돼 재무장되는 과정을 거쳐야 한다.
    """
    sm = make_sm(threshold=10, cap=8)
    assert sm.evaluate(remaining_time_sec=5, occupants=[3]) == 5   # 누적 5
    sm.evaluate(remaining_time_sec=20, occupants=[3])              # 반영 -> 재무장
    assert sm.evaluate(remaining_time_sec=5, occupants=[3]) == 3   # cap(8)까지만
    assert sm.accumulated_extension_sec == 8
    sm.evaluate(remaining_time_sec=20, occupants=[3])              # 재무장
    assert sm.evaluate(remaining_time_sec=5, occupants=[3]) == 0
    assert sm.state == SignalState.CAPPED


def test_undetermined_zone_extension_raises():
    # 3번 구역 연장 시간이 아직 None이면, 그 구역에 사람이 있을 때 임의 동작 대신 실패.
    sm = make_sm(zone_ext={1: 0, 2: 3, 3: None, 4: 3, 5: 0})
    with pytest.raises(NotImplementedError):
        sm.evaluate(remaining_time_sec=5, occupants=[3])


def test_missing_config_raises(monkeypatch):
    """임계값/상한이 config에도 없으면 임의값으로 동작하지 않고 생성 시점에 실패한다."""
    monkeypatch.setattr(config, "REMAINING_TIME_THRESHOLD_SEC", None)
    monkeypatch.setattr(config, "MAX_TOTAL_EXTENSION_SEC", None)
    with pytest.raises(NotImplementedError):
        SignalExtensionStateMachine(zone_extension_sec=ZONE_EXT)


# --- 프레임 루프 안전성: 하강 구간당 1회만 연장 (버그 A) ---

def test_extends_only_once_per_descent_below_threshold():
    """실시간 루프는 evaluate()를 매 프레임 부른다. 그때마다 누적되면 안 된다.

    사람이 3번 구역에 계속 서 있어도, 잔여시간이 임계값 아래로 '내려온 한 번'에 대해
    연장은 한 번만 발급돼야 한다. (수정 전에는 [5, 5, 5, 0, ...] 처럼 프레임마다
    누적돼 0.3초 만에 상한을 소진하고 제어부로 명령을 연타로 보냈다.)
    """
    sm = make_sm(threshold=10, cap=15)
    grants = [sm.evaluate(remaining_time_sec=4, occupants=[3]) for _ in range(5)]

    assert grants == [5, 0, 0, 0, 0]
    assert sm.accumulated_extension_sec == 5
    assert sm.state == SignalState.EXTENDED


def test_rearms_after_remaining_time_recovers():
    """연장이 제어부에 반영돼 잔여시간이 임계값 위로 올라가면 다시 연장할 수 있어야 한다.

    느린 보행자가 한 사이클에서 두 번 연장받는 경로 — 상한이 이를 최종적으로 막는다.
    """
    sm = make_sm(threshold=10, cap=15)

    assert sm.evaluate(remaining_time_sec=4, occupants=[3]) == 5
    assert sm.evaluate(remaining_time_sec=9, occupants=[3]) == 0   # 아직 임계값 아래
    assert sm.evaluate(remaining_time_sec=12, occupants=[3]) == 0  # 반영됨 -> 재무장
    assert sm.state == SignalState.NORMAL
    assert sm.evaluate(remaining_time_sec=4, occupants=[3]) == 5   # 두 번째 연장
    assert sm.accumulated_extension_sec == 10


def test_arming_is_consumed_by_granting_not_by_descending():
    """임계값 아래로 내려온 뒤 뒤늦게 보행자가 들어와도 연장은 발급돼야 한다.

    '하강할 때 무장을 소모'하는 구현이면 이 케이스를 놓친다 — 무장은 '연장 발급'으로만 소모된다.
    """
    sm = make_sm(threshold=10, cap=15)

    assert sm.evaluate(remaining_time_sec=4, occupants=[]) == 0   # 아직 아무도 없음
    assert sm.evaluate(remaining_time_sec=3, occupants=[]) == 0
    assert sm.evaluate(remaining_time_sec=2, occupants=[3]) == 5  # 이제 들어옴


# --- 사이클 경계: reset() (버그 B) ---

def test_new_cycle_allows_extension_after_cap():
    """상한에 도달했어도 다음 보행 신호 사이클에서는 다시 연장할 수 있어야 한다.

    수정 전에는 reset()을 아무도 부르지 않아, 첫 사이클에서 상한을 찍으면 영구히 CAPPED였다.
    """
    sm = make_sm(threshold=10, cap=5)
    assert sm.evaluate(remaining_time_sec=4, occupants=[3]) == 5
    assert sm.evaluate(remaining_time_sec=12, occupants=[3]) == 0
    assert sm.evaluate(remaining_time_sec=4, occupants=[3]) == 0
    assert sm.state == SignalState.CAPPED

    sm.reset()

    assert sm.accumulated_extension_sec == 0
    assert sm.state == SignalState.NORMAL
    assert sm.evaluate(remaining_time_sec=4, occupants=[3]) == 5


# =====================================================================
# 속도(ETA) 반영 연장 — config.USE_SPEED_FOR_EXTENSION
# =====================================================================
# 규칙: 부족분 = ETA x 안전계수 - 잔여시간,  연장 = min(구역값, ceil(부족분))
# 속도는 연장을 '깎기만' 하고 늘리지 않는다.

def make_speed_sm(threshold=5, cap=10, margin=1.0, zone_ext=None):
    return SignalExtensionStateMachine(
        remaining_time_threshold_sec=threshold,
        zone_extension_sec=zone_ext or ZONE_EXT,
        max_total_extension_sec=cap,
        use_speed=True,
        eta_safety_margin=margin,
    )


def test_eta_reduces_extension_to_the_shortfall():
    """잔여 5초 / 3번 구역(5초) / ETA 7초 -> 모자란 2초만 연장한다."""
    sm = make_speed_sm()
    assert sm.evaluate(remaining_time_sec=5, occupants=[Occupant(zone=3, eta_sec=7.0)]) == 2


def test_no_extension_when_pedestrian_finishes_in_time():
    """ETA가 잔여시간보다 짧으면 연장할 이유가 없다."""
    sm = make_speed_sm()
    assert sm.evaluate(remaining_time_sec=5, occupants=[Occupant(zone=3, eta_sec=3.0)]) == 0
    assert sm.state == SignalState.NORMAL


def test_eta_never_extends_beyond_zone_rule():
    """아주 느려도 구역별 상한을 넘지 않는다 — 구역 규칙이 정책 상한으로 남는다."""
    sm = make_speed_sm()
    assert sm.evaluate(remaining_time_sec=5, occupants=[Occupant(zone=3, eta_sec=60.0)]) == 5


def test_edge_zone_stays_zero_even_with_huge_eta():
    """양 끝 구역은 ETA가 아무리 커도 연장하지 않는다(설계상 확정 규칙)."""
    sm = make_speed_sm()
    assert sm.evaluate(remaining_time_sec=5, occupants=[Occupant(zone=1, eta_sec=60.0)]) == 0


def test_falls_back_to_zone_rule_when_eta_unavailable():
    """ETA가 None이면(호모그래피 없음/정지/샘플 부족) 구역 규칙을 그대로 쓴다 — 안전한 폴백."""
    sm = make_speed_sm()
    assert sm.evaluate(remaining_time_sec=5, occupants=[Occupant(zone=3, eta_sec=None)]) == 5


def test_safety_margin_inflates_eta_and_rounds_up():
    """안전계수 1.25, ETA 5초, 잔여 5초 -> 5*1.25-5 = 1.25 -> 올림하여 2초."""
    sm = make_speed_sm(margin=1.25)
    assert sm.evaluate(remaining_time_sec=5, occupants=[Occupant(zone=3, eta_sec=5.0)]) == 2


def test_speed_disabled_ignores_eta():
    """USE_SPEED_FOR_EXTENSION이 꺼져 있으면 ETA가 있어도 구역 규칙 그대로 — 기존 동작 유지."""
    sm = SignalExtensionStateMachine(
        remaining_time_threshold_sec=5, zone_extension_sec=ZONE_EXT,
        max_total_extension_sec=10, use_speed=False,
    )
    assert sm.evaluate(remaining_time_sec=5, occupants=[Occupant(zone=3, eta_sec=3.0)]) == 5


def test_speed_mode_requires_safety_margin(monkeypatch):
    """속도를 켰는데 안전계수가 미정이면 임의값으로 동작하지 않고 실패한다."""
    monkeypatch.setattr(config, "ETA_SAFETY_MARGIN", None)
    with pytest.raises(NotImplementedError):
        SignalExtensionStateMachine(
            remaining_time_threshold_sec=5, zone_extension_sec=ZONE_EXT,
            max_total_extension_sec=10, use_speed=True,
        )


def test_multiple_pedestrians_use_per_person_need_not_center_only():
    """사람마다 따로 계산한 뒤 최댓값을 쓴다.

    3번 구역 사람이 빨라서 1초만 필요하고, 2번 구역 사람이 느려서 3초가 필요한 상황.
    '가운데 사람만 본다'면 1초만 연장해 2번 구역의 느린 사람을 도로에 가둔다.
    속도를 넣는 순간 "가운데가 가장 오래 걸린다"는 가정이 깨지기 때문이다.
    """
    sm = make_speed_sm()
    fast_center = Occupant(zone=3, eta_sec=5.5)   # 부족분 0.5 -> 1초
    slow_side = Occupant(zone=2, eta_sec=9.0)     # 부족분 4.0 -> 구역 상한 3초

    assert sm.evaluate(remaining_time_sec=5, occupants=[fast_center]) == 1
    sm.reset()
    assert sm.evaluate(remaining_time_sec=5, occupants=[slow_side]) == 3
    sm.reset()
    assert sm.evaluate(remaining_time_sec=5, occupants=[fast_center, slow_side]) == 3


def test_plain_zone_numbers_still_accepted():
    """구역 번호만 넘기면 ETA 없음(None)으로 해석한다 — 기존 호출부/도구 호환."""
    sm = make_speed_sm()
    assert sm.evaluate(remaining_time_sec=5, occupants=[3]) == 5
