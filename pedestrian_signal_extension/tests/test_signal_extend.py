"""진척도 요약 단위 테스트.

파이 쪽에 남은 판단은 "가장 덜 건넌 사람이 몇 번째인가" 하나뿐이다.
얼마나 연장할지는 제어부(아두이노)가 기준표와 잔여 시간을 비교해 정한다
(src/signal_extend.py 모듈 docstring, docs/team_interface.md 참고).
"""

from src.signal_extend import Occupant, maximum_eta_sec, minimum_progress


# --- 최솟값 요약 ---

def test_no_occupants_returns_none():
    assert minimum_progress([]) is None


def test_single_occupant():
    assert minimum_progress([Occupant(progress=3)]) == 3


def test_takes_the_least_advanced_person():
    """가장 덜 건넌 사람이 이긴다.

    기준표가 진척도에 대해 단조 감소하므로, 같은 잔여 시간에서는 진척도가 작을수록
    항상 더 부족하다. 최솟값 하나가 나머지를 지배한다.
    """
    occupants = [Occupant(progress=4), Occupant(progress=2), Occupant(progress=5)]
    assert minimum_progress(occupants) == 2


def test_accepts_bare_numbers():
    assert minimum_progress([4, 2, 5]) == 2


def test_ignores_none_entries():
    assert minimum_progress([None, Occupant(progress=3), Occupant(progress=None)]) == 3


def test_all_none_progress_returns_none():
    assert minimum_progress([Occupant(progress=None)]) is None


# --- ETA는 계측용 ---

def test_max_eta_picks_slowest():
    occupants = [Occupant(progress=2, eta_sec=4.0), Occupant(progress=3, eta_sec=9.5)]
    assert maximum_eta_sec(occupants) == 9.5


def test_max_eta_is_none_without_samples():
    assert maximum_eta_sec([Occupant(progress=3)]) is None
    assert maximum_eta_sec([]) is None


def test_eta_does_not_affect_progress():
    """ETA가 있든 없든 전송 값(진척도)은 달라지지 않는다 — 연장 판단은 속도를 쓰지 않는다."""
    with_eta = [Occupant(progress=2, eta_sec=30.0), Occupant(progress=4, eta_sec=1.0)]
    without = [Occupant(progress=2), Occupant(progress=4)]
    assert minimum_progress(with_eta) == minimum_progress(without) == 2
