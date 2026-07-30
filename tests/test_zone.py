import pytest

from src.zone import CrosswalkOccupancy, CrosswalkZones, Zone

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]

# 시작 변 = 왼쪽(x=0), 끝 변 = 오른쪽(x=50). 걷는 방향은 x 증가 방향.
# corners: [시작-왼쪽, 시작-오른쪽, 끝-오른쪽, 끝-왼쪽]
# -> 5등분하면 구역 k는 x in [10*(k-1), 10*k], y in [0,10].
QUAD = [(0, 0), (0, 10), (50, 10), (50, 0)]


# --- Zone (단일 폴리곤) ---

def test_zone_contains_point_inside():
    assert Zone(SQUARE).contains((5, 5)) is True


def test_zone_does_not_contain_point_outside():
    assert Zone(SQUARE).contains((20, 20)) is False


# --- CrosswalkZones (5구역 위치 판정) ---

def test_from_quad_makes_n_zones():
    zones = CrosswalkZones.from_quad(QUAD, n=5)
    assert len(zones) == 5


def test_locate_returns_zone_index_by_position():
    zones = CrosswalkZones.from_quad(QUAD, n=5)
    assert zones.locate((5, 5)) == 1    # 시작 끝 구역
    assert zones.locate((15, 5)) == 2
    assert zones.locate((25, 5)) == 3   # 정중앙
    assert zones.locate((35, 5)) == 4
    assert zones.locate((45, 5)) == 5   # 반대 끝 구역


def test_locate_outside_returns_none():
    zones = CrosswalkZones.from_quad(QUAD, n=5)
    assert zones.locate((100, 100)) is None


# --- CrosswalkOccupancy (확정 보행자/점유 구역) ---

def test_occupancy_requires_consecutive_frames():
    zones = CrosswalkZones.from_quad(QUAD, n=5)
    occ = CrosswalkOccupancy(zones, confirm_frames=3)

    assert occ.update([("p1", (25, 5))]) == {}       # 1프레임
    assert occ.update([("p1", (25, 5))]) == {}       # 2프레임
    assert occ.update([("p1", (25, 5))]) == {"p1": 3}  # 3프레임 -> 확정, 3번 구역


def test_occupancy_tracks_moving_pedestrian_zone():
    zones = CrosswalkZones.from_quad(QUAD, n=5)
    occ = CrosswalkOccupancy(zones, confirm_frames=1)

    occ.update([("p1", (5, 5))])    # 1번 구역
    result = occ.update([("p1", (25, 5))])  # 3번 구역으로 이동
    assert result == {"p1": 3}
    assert occ.occupied_zones() == [3]


def test_occupancy_resets_when_leaving_crosswalk():
    zones = CrosswalkZones.from_quad(QUAD, n=5)
    occ = CrosswalkOccupancy(zones, confirm_frames=2)

    occ.update([("p1", (25, 5))])
    occ.update([("p1", (100, 100))])  # 횡단보도 이탈 -> 리셋
    assert occ.update([("p1", (25, 5))]) == {}  # 다시 처음부터


def test_occupancy_missing_config_raises():
    zones = CrosswalkZones.from_quad(QUAD, n=5)
    with pytest.raises(NotImplementedError):
        CrosswalkOccupancy(zones, confirm_frames=None)
