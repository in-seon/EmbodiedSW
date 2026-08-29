import json

import pytest

from config import config
from src.ground_plane import GroundPlane
from src.zone import CrossingProgress, CrosswalkOccupancy, CrosswalkZones, Zone

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


def test_occupancy_missing_config_raises(monkeypatch):
    """config가 비어 있으면 임의값으로 동작하지 않고 명시적으로 실패한다.

    config를 직접 비운다 — 인자로 None을 넘기면 "지정 안 함"으로 해석돼 config를 읽으므로,
    config에 값이 채워지는 순간 이 테스트가 무의미해진다(실제로 그렇게 깨진 적이 있다).
    """
    monkeypatch.setattr(config, "ZONE_RESIDENCY_FRAMES", None)
    zones = CrosswalkZones.from_quad(QUAD, n=5)
    with pytest.raises(NotImplementedError):
        CrosswalkOccupancy(zones, confirm_frames=None)


# --- 미검출 유예 (config.TRACK_GRACE_FRAMES) ---
#
# YOLO는 가림·모션블러·저조도에서 한두 프레임씩 사람을 놓친다. 그때마다 잔류 카운트를
# 0으로 되돌리면 확정 보행자가 영영 안 나오고, 에러 없이 조용히 연장만 빠진다.

def test_occupancy_tolerates_brief_detection_gap():
    """유예 안의 깜빡임은 카운트를 잃지 않고 이어간다."""
    zones = CrosswalkZones.from_quad(QUAD, n=5)
    occ = CrosswalkOccupancy(zones, confirm_frames=3, grace_frames=2)

    occ.update([("p1", (25, 5))])                      # count 1
    occ.update([])                                     # 미검출 1
    occ.update([])                                     # 미검출 2 (유예 이내)
    assert occ.update([("p1", (25, 5))]) == {}         # count 2 — 이어서 셈
    assert occ.update([("p1", (25, 5))]) == {"p1": 3}  # count 3 -> 확정


def test_occupancy_freezes_count_during_gap():
    """유예 동안 카운트는 동결된다 — 안 보이는 프레임을 잔류로 쳐주면 안 된다."""
    zones = CrosswalkZones.from_quad(QUAD, n=5)
    occ = CrosswalkOccupancy(zones, confirm_frames=2, grace_frames=2)

    occ.update([("p1", (25, 5))])                      # count 1
    assert occ.update([]) == {}                        # 미검출 — 여기서 확정되면 안 된다
    assert occ.update([("p1", (25, 5))]) == {"p1": 3}  # count 2 -> 확정


def test_occupancy_drops_track_after_grace_expires():
    """유예를 넘긴 미검출은 그때 삭제한다 — 떠난 사람을 영원히 붙들지 않는다."""
    zones = CrosswalkZones.from_quad(QUAD, n=5)
    occ = CrosswalkOccupancy(zones, confirm_frames=2, grace_frames=2)

    occ.update([("p1", (25, 5))])
    occ.update([])
    occ.update([])
    occ.update([])                                     # 미검출 3 > 유예 2 -> 삭제
    assert occ.update([("p1", (25, 5))]) == {}         # 처음부터 다시


def test_occupancy_confirmed_pedestrian_survives_gap():
    """이미 확정된 보행자는 깜빡임 동안 확정 상태를 유지한다 — 연장이 끊기지 않아야 한다."""
    zones = CrosswalkZones.from_quad(QUAD, n=5)
    occ = CrosswalkOccupancy(zones, confirm_frames=1, grace_frames=2)

    assert occ.update([("p1", (25, 5))]) == {"p1": 3}
    assert occ.update([]) == {"p1": 3}                 # 미검출이어도 확정 유지
    assert occ.occupied_zones() == [3]


def test_occupancy_zone_exit_ignores_grace():
    """'안 보임'과 '구역 밖으로 나감'은 다르다 — 후자는 유예 없이 즉시 리셋한다.

    안 보이는 것은 정보가 없는 상태지만, 밖으로 나간 것은 확실한 정보다. 둘을 같이
    취급하면 이미 횡단보도를 벗어난 사람 때문에 차를 세우게 된다.
    """
    zones = CrosswalkZones.from_quad(QUAD, n=5)
    occ = CrosswalkOccupancy(zones, confirm_frames=2, grace_frames=2)

    occ.update([("p1", (25, 5))])
    occ.update([("p1", (100, 100))])                   # 횡단보도 밖 -> 즉시 리셋
    assert occ.update([("p1", (25, 5))]) == {}


def test_occupancy_grace_zero_drops_immediately():
    """grace_frames=0이면 유예 도입 전과 같이 한 프레임 미검출에 바로 리셋된다."""
    zones = CrosswalkZones.from_quad(QUAD, n=5)
    occ = CrosswalkOccupancy(zones, confirm_frames=2, grace_frames=0)

    occ.update([("p1", (25, 5))])
    occ.update([])
    assert occ.update([("p1", (25, 5))]) == {}


def test_occupancy_rejects_negative_grace():
    zones = CrosswalkZones.from_quad(QUAD, n=5)
    with pytest.raises(ValueError):
        CrosswalkOccupancy(zones, confirm_frames=2, grace_frames=-1)


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


# --- 진척도 (CrossingProgress) ---
#
# 물리 구역 번호는 좌표계 기준이라 진입 방향에 따라 의미가 뒤집힌다.
# 반대편에서 들어온 사람의 '2번'은 '거의 다 건넜음'인데, 보정하지 않으면
# '대부분 남았음'으로 읽혀 엉뚱한 사람이 최솟값을 차지한다.

def test_progress_matches_zone_for_forward_entry():
    """1·2번에서 처음 보이면 정방향 — 진척도가 구역 번호와 같다."""
    p = CrossingProgress(zone_count=5)
    assert p.update("a", 1) == 1
    assert p.update("a", 2) == 2
    assert p.update("a", 4) == 4
    assert p.direction("a") == 1


def test_progress_is_mirrored_for_reverse_entry():
    """4·5번에서 처음 보이면 역방향 — 진척도가 뒤집힌다."""
    p = CrossingProgress(zone_count=5)
    assert p.update("b", 5) == 1        # 방금 진입
    assert p.update("b", 4) == 2
    assert p.update("b", 2) == 4        # 거의 다 건넜음
    assert p.direction("b") == -1


def test_opposite_directions_at_same_physical_zone():
    """같은 물리 2번 구역이라도 방향이 다르면 진척도가 다르다 — 이 클래스의 존재 이유."""
    p = CrossingProgress(zone_count=5)
    p.update("fwd", 1)                  # 정방향으로 진입
    p.update("rev", 5)                  # 역방향으로 진입

    assert p.update("fwd", 2) == 2      # 대부분 남음
    assert p.update("rev", 2) == 4      # 거의 다 건넘


def test_middle_start_is_treated_as_middle():
    """중앙에서 처음 보이면 방향을 모른다 -> 중앙값으로 간주한다."""
    p = CrossingProgress(zone_count=5)
    assert p.update("c", 3) == 3
    assert p.direction("c") is None


def test_middle_start_resolves_on_first_move():
    """중앙에서 시작한 뒤 움직이면 그 방향으로 확정된다.

    3->4든 3->2든 '중앙에서 한 칸 더 갔다'로 수렴해 진척도 4가 된다.
    """
    forward = CrossingProgress(zone_count=5)
    forward.update("d", 3)
    assert forward.update("d", 4) == 4
    assert forward.direction("d") == 1

    reverse = CrossingProgress(zone_count=5)
    reverse.update("e", 3)
    assert reverse.update("e", 2) == 4      # 6 - 2
    assert reverse.direction("e") == -1


def test_direction_is_latched_not_recomputed():
    """한 번 정해진 방향은 유지된다 — 검출 흔들림으로 뒤로 밀려도 뒤집히지 않는다."""
    p = CrossingProgress(zone_count=5)
    p.update("f", 1)
    p.update("f", 3)
    assert p.update("f", 2) == 2        # 잠깐 뒤로 밀려도 여전히 정방향
    assert p.direction("f") == 1


def test_forget_clears_direction():
    """ID가 재사용될 때 옛 방향이 따라붙으면 안 된다."""
    p = CrossingProgress(zone_count=5)
    p.update("g", 5)
    assert p.direction("g") == -1

    p.forget("g")
    assert p.direction("g") is None
    assert p.update("g", 1) == 1        # 새 사람으로 다시 시작
    assert p.direction("g") == 1


def test_keep_only_drops_missing_tracks():
    p = CrossingProgress(zone_count=5)
    p.update("h", 1)
    p.update("i", 5)

    p.keep_only(["h"])
    assert p.direction("h") == 1
    assert p.direction("i") is None


# --- track_id 재발급 상속 ---
#
# ByteTrack은 가림·모션블러에서 같은 사람에게 새 ID를 준다. 그러면 잔류 카운트가 0으로
# 돌아갈 뿐 아니라 **CrossingProgress의 진입 방향이 통째로 날아간다.** 방향을 잃으면
# 거의 다 건넌 사람(구역 4)이 "방금 진입"(진척도 2)으로 보고돼 아두이노가 연장을 다시
# 시작한다 — 재확정 지연(0.25초)보다 이쪽이 훨씬 비싸다.
#
# 쓰러짐 감지에는 이미 같은 보정이 있다(FallTracker._resolve_keys). 그쪽은 bbox 겹침으로
# 맞추지만 여기는 발 위치만 받으므로 **거리**로 맞춘다 — 구역 판정 기준과 같은 좌표다.

def _occupancy(confirm_frames=2, grace_frames=2, inherit_distance=5, **kwargs):
    """QUAD는 실측 치수가 없어 호모그래피가 없다 -> 거리 단위가 픽셀이다.

    QUAD가 50px 길이라 구역 하나가 10px다. 임계값 5px = 구역 절반으로 잡아,
    '같은 구역 안의 흔들림'은 잇고 '한 구역 이상 떨어진 것'은 남남이 되게 한다.
    """
    zones = CrosswalkZones.from_quad(QUAD, n=5)
    return CrosswalkOccupancy(zones, confirm_frames=confirm_frames,
                              grace_frames=grace_frames,
                              inherit_distance=inherit_distance, **kwargs)


def test_new_id_near_lost_track_inherits_its_count():
    """★ 이 기능의 본체 — 새 ID가 즉시 '확정 보행자'가 되어 전송이 끊기지 않는다."""
    occ = _occupancy()
    occ.update([(1, (25, 5))])
    occ.update([(1, (26, 5))])
    assert 1 in occ.confirmed()

    occ.update([(7, (27, 5))])          # 같은 자리, 새 ID
    confirmed = occ.confirmed()
    assert 7 in confirmed
    assert 1 not in confirmed, "옛 ID가 남아 유령 보행자가 되면 안 된다"


def test_rekey_is_reported_so_progress_can_follow():
    """occupancy가 이름을 바꿔도 CrossingProgress가 모르면 방향은 그대로 날아간다."""
    occ = _occupancy()
    occ.update([(1, (25, 5))])
    occ.update([(1, (26, 5))])
    assert occ.rekeyed == []

    occ.update([(7, (27, 5))])
    assert occ.rekeyed == [(1, 7)]

    occ.update([(7, (28, 5))])
    assert occ.rekeyed == [], "재발급이 없는 프레임에는 비어 있어야 한다"


def test_distant_new_id_is_a_different_person():
    """임계 거리를 넘으면 남남이다. 아니면 반대편 사람의 방향을 물려받는다."""
    occ = _occupancy()
    occ.update([(1, (5, 5))])
    occ.update([(1, (5, 5))])

    occ.update([(7, (45, 5))])          # 횡단보도 반대쪽 끝
    assert occ.rekeyed == []
    assert 7 not in occ.confirmed()


def test_visible_track_is_not_robbed():
    """★ 살아 있는 사람의 상태를 뺏으면 안 된다.

    두 사람이 나란히 있을 때 한쪽에 새 ID가 붙었다고 옆 사람 상태를 가져가면,
    옆 사람은 확정이 풀리고 새 ID는 남의 방향을 얻는다.
    """
    occ = _occupancy()
    occ.update([(1, (25, 5))])
    occ.update([(1, (25, 5))])

    occ.update([(1, (25, 5)), (7, (26, 5))])   # 1번은 이번 프레임에도 보인다
    assert occ.rekeyed == []
    assert 1 in occ.confirmed()


def test_expired_track_is_not_inherited():
    """유예 창을 넘겨 버려진 트랙은 후보가 아니다."""
    occ = _occupancy(grace_frames=1)
    occ.update([(1, (25, 5))])
    occ.update([(1, (25, 5))])
    occ.update([])                      # miss 1 — 아직 유예 안
    occ.update([])                      # miss 2 > grace -> 폐기

    occ.update([(7, (25, 5))])
    assert occ.rekeyed == []


def test_one_lost_track_is_claimed_by_the_nearest_detection_only():
    """두 검출이 한 트랙을 나눠 가지면 상태가 복제된다."""
    occ = _occupancy()
    occ.update([(1, (25, 5))])
    occ.update([(1, (25, 5))])

    occ.update([(7, (26, 5)), (8, (28, 5))])   # 둘 다 근처
    assert occ.rekeyed == [(1, 7)], "가장 가까운 검출 하나만 물려받는다"
    assert 8 not in occ.confirmed()


def test_id_collision_does_not_clobber_existing_state():
    """이미 쓰이는 번호로 검출돼도 남의 상태를 덮어쓰지 않는다.

    구조적으로 보장된다 — 상속 후보는 '이번 프레임에 처음 보는 ID'뿐이라,
    이미 관리 중인 번호는 애초에 상속 경로를 타지 않는다.
    """
    occ = _occupancy()
    occ.update([(1, (5, 5)), (2, (25, 5))])
    occ.update([(1, (5, 5)), (2, (25, 5))])

    # 1번이 사라지고, 그 자리에 '2'라는 이미 쓰이는 번호로 검출됐다(비정상이지만 방어).
    occ.update([(2, (25, 5))])
    assert 2 in occ.confirmed()
    assert occ.rekeyed == []


def test_inheritance_can_be_disabled():
    """거리 0이면 상속하지 않는다 — 동작을 끄고 비교해볼 수 있어야 한다."""
    occ = _occupancy(inherit_distance=0)
    occ.update([(1, (25, 5))])
    occ.update([(1, (25, 5))])
    occ.update([(7, (25, 5))])
    assert occ.rekeyed == []


# --- CrossingProgress.rekey ---

def test_progress_rekey_preserves_direction():
    """★ 방향이 유지되면 진척도가 뒤집히지 않는다."""
    progress = CrossingProgress(zone_count=5)
    for zone in (1, 2, 3):
        progress.update(1, zone)

    progress.rekey(1, 7)
    assert progress.direction(7) == 1
    assert progress.update(7, 4) == 4      # 상속이 없으면 2가 나온다
    assert progress.update(7, 5) == 5      # 상속이 없으면 1이 나온다


def test_progress_rekey_forgets_the_old_id():
    progress = CrossingProgress(zone_count=5)
    progress.update(1, 1)
    progress.rekey(1, 7)
    assert progress.direction(1) is None


def test_progress_rekey_does_not_clobber_an_existing_id():
    progress = CrossingProgress(zone_count=5)
    progress.update(1, 1)                  # 정방향
    progress.update(2, 5)                  # 역방향
    progress.rekey(1, 2)
    assert progress.direction(2) == -1, "이미 쓰이는 ID의 방향을 덮어쓰면 안 된다"


def test_progress_rekey_of_unknown_id_is_harmless():
    progress = CrossingProgress(zone_count=5)
    progress.rekey(99, 100)
    assert progress.direction(100) is None
