import json

import pytest

from config import config
from src.ground_plane import GroundPlane
from src.zone import CrosswalkOccupancy, CrosswalkZones, Zone

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]

# 시작 변 = 왼쪽(x=0), 끝 변 = 오른쪽(x=50). 걷는 방향은 x 증가 방향.
# corners: [시작-왼쪽, 시작-오른쪽, 끝-오른쪽, 끝-왼쪽]
# -> 5등분하면 구역 k는 x in [10*(k-1), 10*k], y in [0,10].
QUAD = [(0, 0), (0, 10), (50, 10), (50, 0)]

# 사선 카메라로 본 횡단보도(사다리꼴) — 먼 쪽(끝 변)이 화면에서 좁고 위에 보인다.
# 실제 치수는 폭 400cm x 길이 1000cm 인 직사각형.
OBLIQUE_QUAD = [(80, 460), (560, 460), (400, 150), (240, 150)]
OBLIQUE_W_CM, OBLIQUE_L_CM = 400.0, 1000.0


def _zone_boundaries_cm(zones, ground_plane):
    """각 구역의 '끝 변' 중점이 평면상 몇 cm 지점인지 돌려준다."""
    out = []
    for zone in zones.zones:
        left_end, right_end = zone.points[3], zone.points[2]
        mid = ((left_end[0] + right_end[0]) / 2, (left_end[1] + right_end[1]) / 2)
        out.append(ground_plane.to_ground(mid)[1])
    return out


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


# --- 사선 구도에서의 구역 분할 (원근 보정) ---
# 화면상 균등 분할은 실제 거리로는 균등하지 않다. 실측 치수가 있으면 평면 좌표(cm)에서
# 나눈 뒤 픽셀로 되돌려야 각 구역이 실제로 같은 거리를 담당한다.

def test_zones_are_equal_in_real_distance_with_dimensions():
    zones = CrosswalkZones.from_quad(
        OBLIQUE_QUAD, n=5, width_cm=OBLIQUE_W_CM, length_cm=OBLIQUE_L_CM
    )
    boundaries = _zone_boundaries_cm(zones, zones.ground_plane)

    # 구역 경계가 200, 400, 600, 800, 1000 cm 에 정확히 놓여야 한다.
    assert boundaries == pytest.approx([200.0, 400.0, 600.0, 800.0, 1000.0], abs=1e-3)


def test_true_center_of_crosswalk_lands_in_middle_zone():
    """설계 전제 그 자체 — '정중앙에 있으면 가장 길게 연장'(CLAUDE.md 2.4).

    실제 한가운데(500cm)가 3번 구역에 있어야 그 규칙이 의미를 갖는다.
    화면상 분할로는 4번 구역에 들어가 규칙이 엉뚱한 위치에 적용된다.
    """
    zones = CrosswalkZones.from_quad(
        OBLIQUE_QUAD, n=5, width_cm=OBLIQUE_W_CM, length_cm=OBLIQUE_L_CM
    )
    center_px = zones.ground_plane.to_pixel((OBLIQUE_W_CM / 2, OBLIQUE_L_CM / 2))

    assert zones.locate(center_px) == 3


def test_pixel_split_is_uneven_without_dimensions():
    """치수가 없으면 화면상 분할로 대체된다 — 그때 왜곡이 얼마나 되는지 못박아 둔다.

    이 테스트는 대체 경로가 부정확하다는 사실 자체를 기록한다. 정확도를 원하면
    실측 치수를 넣어야 한다는 근거다.
    """
    zones = CrosswalkZones.from_quad(OBLIQUE_QUAD, n=5)
    assert zones.ground_plane is None  # 치수가 없으니 호모그래피도 없다

    reference = GroundPlane.from_quad(OBLIQUE_QUAD, OBLIQUE_W_CM, OBLIQUE_L_CM)
    boundaries = _zone_boundaries_cm(zones, reference)
    lengths = [b - a for a, b in zip([0.0] + boundaries, boundaries)]

    # 가장 먼 구역이 가장 가까운 구역보다 5배 이상 길다.
    assert max(lengths) > min(lengths) * 5
    # 그리고 실제 한가운데가 3번이 아닌 더 뒤쪽 구역에 들어간다.
    assert zones.locate(reference.to_pixel((OBLIQUE_W_CM / 2, OBLIQUE_L_CM / 2))) == 4


def test_zone_count_and_coverage_unchanged_by_correction():
    """분할 방식이 바뀌어도 구역 개수와 전체 범위는 그대로여야 한다."""
    zones = CrosswalkZones.from_quad(
        OBLIQUE_QUAD, n=5, width_cm=OBLIQUE_W_CM, length_cm=OBLIQUE_L_CM
    )
    assert len(zones) == 5

    gp = zones.ground_plane
    # 양 끝과 중간 지점들이 빠짐없이 어느 구역엔가 속한다.
    for y_cm in [1.0, 199.0, 201.0, 500.0, 999.0]:
        assert zones.locate(gp.to_pixel((OBLIQUE_W_CM / 2, y_cm))) is not None
    # 횡단보도 밖(뒤쪽)은 어느 구역에도 속하지 않는다.
    assert zones.locate(gp.to_pixel((OBLIQUE_W_CM / 2, -50.0))) is None


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


# --- 캘리브레이션 해상도 검증 (버그 C) ---

def _write_zone_config(tmp_path, **extra):
    payload = {
        "name": "crosswalk",
        "corners": [list(p) for p in QUAD],
        "n_zones": 5,
    }
    payload.update(extra)
    path = tmp_path / "zone_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_load_rejects_frame_size_mismatch(tmp_path):
    """캘리브레이션 해상도와 운영 해상도가 다르면 좌표가 통째로 어긋난다 -> 즉시 실패해야 한다.

    수정 전에는 zone_calibrator가 frame_size를 저장해 두는데도 load()가 읽지 않아,
    에러 없이 구역만 엉뚱하게 잡히는 '조용한 오작동'이 났다.
    """
    path = _write_zone_config(tmp_path, frame_size=[1280, 720])

    with pytest.raises(ValueError) as exc:
        CrosswalkZones.load(path, expected_frame_size=(640, 480))

    assert "1280x720" in str(exc.value)
    assert "640x480" in str(exc.value)


def test_load_accepts_matching_frame_size(tmp_path):
    path = _write_zone_config(tmp_path, frame_size=[640, 480])
    zones = CrosswalkZones.load(path, expected_frame_size=(640, 480))
    assert len(zones) == 5


def test_load_allows_config_without_frame_size(tmp_path):
    """frame_size가 없는 옛 설정 파일은 검증할 방법이 없으므로 그대로 통과시킨다."""
    path = _write_zone_config(tmp_path)
    assert len(CrosswalkZones.load(path, expected_frame_size=(640, 480))) == 5


def test_load_uses_config_resolution_by_default(tmp_path, monkeypatch):
    """expected_frame_size를 안 주면 config.CAMERA_RESOLUTION과 비교한다."""
    monkeypatch.setattr(config, "CAMERA_RESOLUTION", (640, 480))
    path = _write_zone_config(tmp_path, frame_size=[320, 240])

    with pytest.raises(ValueError):
        CrosswalkZones.load(path)


# --- 추적 ID가 없는 검출 (버그 D) ---

def test_occupancy_ignores_untracked_detections():
    """track_id=None인 검출은 잔류 판정에서 제외한다 (SpeedEstimator와 동일한 규칙).

    수정 전에는 None을 그대로 딕셔너리 키로 써서, 같은 프레임 안의 여러 사람이
    하나의 카운터를 각각 증가시켰다. 아래는 한 프레임 만에 confirm_frames=3을 채워
    잔류 검증이 무력화되던 케이스다.
    """
    zones = CrosswalkZones.from_quad(QUAD, n=5)
    occ = CrosswalkOccupancy(zones, confirm_frames=3)

    result = occ.update([(None, (5, 5)), (None, (25, 5)), (None, (45, 5))])

    assert result == {}
    assert occ.occupied_zones() == []


def test_occupancy_reports_untracked_count():
    """무시한 검출 수를 노출한다 — 조용히 놓치지 않고 실측에서 눈에 보이게 하기 위함."""
    zones = CrosswalkZones.from_quad(QUAD, n=5)
    occ = CrosswalkOccupancy(zones, confirm_frames=1)

    occ.update([("p1", (25, 5)), (None, (5, 5)), (None, (45, 5))])
    assert occ.untracked_count == 2

    occ.update([("p1", (25, 5))])
    assert occ.untracked_count == 0


def test_untracked_detection_does_not_disturb_tracked_one():
    """ID 없는 검출이 섞여도 정상 추적되는 보행자의 잔류 카운트는 영향받지 않는다."""
    zones = CrosswalkZones.from_quad(QUAD, n=5)
    occ = CrosswalkOccupancy(zones, confirm_frames=2)

    occ.update([("p1", (25, 5)), (None, (25, 5))])
    assert occ.update([("p1", (25, 5)), (None, (25, 5))]) == {"p1": 3}
