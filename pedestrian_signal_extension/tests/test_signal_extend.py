"""ZoneExtensionRule 단위 테스트.

이 규칙은 상태가 없다 — 누적도 무장도 임계값도 여기 없다. 그것들은 제어부(아두이노)가
소유한다(src/signal_extend.py 모듈 docstring, docs/team_interface.md 참고).
그래서 검증할 것은 "구역 목록 -> 필요한 연장 초" 하나뿐이다.
"""

import pytest

from src.signal_extend import Occupant, ZoneExtensionRule

# 테스트가 config 튜닝에 흔들리지 않도록 맵을 직접 소유한다.
ZONES = {1: 0, 2: 3, 3: 5, 4: 3, 5: 0}
PRIORITY_ZONES = {1: 0, 2: 4, 3: 6, 4: 4, 5: 0}


@pytest.fixture
def rule():
    return ZoneExtensionRule(zone_extension_sec=ZONES)


# --- 구역 -> 연장 초 ---

def test_no_occupants_means_no_extension(rule):
    assert rule.required_sec([]) == 0


@pytest.mark.parametrize("zone,expected", [(1, 0), (2, 3), (3, 5), (4, 3), (5, 0)])
def test_each_zone_maps_to_its_value(rule, zone, expected):
    assert rule.required_sec([Occupant(zone=zone)]) == expected


def test_end_zones_alone_mean_no_extension(rule):
    """양 끝 구역은 '거의 다 건넜거나 이제 막 진입' — 연장하지 않는 것이 설계 규칙이다."""
    assert rule.required_sec([Occupant(zone=1), Occupant(zone=5)]) == 0


def test_takes_the_largest_need(rule):
    """여러 명이면 가장 오래 걸리는 사람에 맞춘다 — 느린 쪽을 도로에 가두지 않기 위해."""
    occupants = [Occupant(zone=1), Occupant(zone=2), Occupant(zone=3)]
    assert rule.required_sec(occupants) == 5


def test_accepts_bare_zone_numbers(rule):
    """구역 번호만 넘어와도 받아준다 (ETA 없음으로 해석)."""
    assert rule.required_sec([2, 3, 1]) == 5


def test_ignores_none_entries(rule):
    assert rule.required_sec([None, Occupant(zone=3), Occupant(zone=None)]) == 5


def test_undecided_zone_value_raises():
    """값이 미정(None)인 구역에 사람이 있으면 임의값으로 넘어가지 않고 실패한다."""
    rule = ZoneExtensionRule(zone_extension_sec={1: 0, 2: None, 3: 5, 4: 3, 5: 0})
    with pytest.raises(NotImplementedError):
        rule.required_sec([Occupant(zone=2)])


# --- 교통약자 우선 (숫자만 커진다) ---

def test_priority_uses_bigger_numbers():
    rule = ZoneExtensionRule(zone_extension_sec=ZONES,
                             priority_zone_extension_sec=PRIORITY_ZONES)
    assert rule.required_sec([Occupant(zone=3)]) == 5
    assert rule.required_sec([Occupant(zone=3)], priority_mode=True) == 6


def test_priority_falls_back_when_map_undecided(rule):
    """우선 연장 맵이 미정이면 일반 규칙으로 안전하게 대체한다."""
    assert rule.required_sec([Occupant(zone=3)], priority_mode=True) == 5


# --- ETA 요약 ---

def test_max_eta_picks_slowest(rule):
    occupants = [Occupant(zone=2, eta_sec=4.0), Occupant(zone=3, eta_sec=9.5)]
    assert rule.max_eta_sec(occupants) == 9.5


def test_max_eta_is_none_without_samples(rule):
    assert rule.max_eta_sec([Occupant(zone=3, eta_sec=None)]) is None
    assert rule.max_eta_sec([]) is None


def test_max_eta_ignores_end_zones(rule):
    """양 끝 구역 사람의 ETA는 보내지 않는다 — 어차피 연장을 주지 않기로 한 사람이다.

    보내면 아두이노가 '줄 수 없는 시간'을 기준으로 부족분을 계산하게 된다.
    """
    occupants = [Occupant(zone=1, eta_sec=30.0), Occupant(zone=2, eta_sec=4.0)]
    assert rule.max_eta_sec(occupants) == 4.0

    assert rule.max_eta_sec([Occupant(zone=5, eta_sec=30.0)]) is None
