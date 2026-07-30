"""GroundPlane(호모그래피) 단위 테스트.

카메라·모델 없이 좌표 계산만 검증한다.
"""

import pytest

from src.ground_plane import GroundPlane

# corners 순서: [시작-왼쪽, 시작-오른쪽, 끝-오른쪽, 끝-왼쪽]

# (A) 정면에서 본 직사각형 — 픽셀 100x200 이 실제 50cm x 100cm.
RECT_PX = [(0, 0), (100, 0), (100, 200), (0, 200)]
RECT_W_CM, RECT_L_CM = 50.0, 100.0

# (B) 사선 카메라로 본 사다리꼴 — 먼 쪽(끝 변)이 화면에서 좁게 보인다.
#     실제로는 같은 폭인데 픽셀 폭이 다르다는 점이 원근 보정의 핵심.
TRAPEZOID_PX = [(0, 200), (100, 200), (70, 50), (30, 50)]
TRAP_W_CM, TRAP_L_CM = 40.0, 300.0


def approx(point, expected, tol=1e-6):
    assert point[0] == pytest.approx(expected[0], abs=tol)
    assert point[1] == pytest.approx(expected[1], abs=tol)


# --- 생성 ---

def test_from_quad_returns_none_when_dimensions_missing():
    """실측 치수가 없으면 호모그래피를 만들지 않는다(속도가 px/s로 떨어지는 경로)."""
    assert GroundPlane.from_quad(RECT_PX, None, RECT_L_CM) is None
    assert GroundPlane.from_quad(RECT_PX, RECT_W_CM, None) is None
    assert GroundPlane.from_quad(RECT_PX, None, None) is None


def test_from_quad_rejects_wrong_corner_count():
    with pytest.raises(ValueError):
        GroundPlane.from_quad([(0, 0), (1, 0), (1, 1)], RECT_W_CM, RECT_L_CM)


def test_from_quad_rejects_non_positive_dimensions():
    with pytest.raises(ValueError):
        GroundPlane.from_quad(RECT_PX, 0, RECT_L_CM)
    with pytest.raises(ValueError):
        GroundPlane.from_quad(RECT_PX, RECT_W_CM, -10)


# --- 좌표 변환 ---

def test_corners_map_to_rectangle_corners():
    """네 꼭짓점은 정확히 (0,0) (w,0) (w,L) (0,L) 로 간다."""
    plane = GroundPlane.from_quad(RECT_PX, RECT_W_CM, RECT_L_CM)
    approx(plane.to_ground(RECT_PX[0]), (0, 0))
    approx(plane.to_ground(RECT_PX[1]), (RECT_W_CM, 0))
    approx(plane.to_ground(RECT_PX[2]), (RECT_W_CM, RECT_L_CM))
    approx(plane.to_ground(RECT_PX[3]), (0, RECT_L_CM))


def test_rect_center_maps_to_center():
    plane = GroundPlane.from_quad(RECT_PX, RECT_W_CM, RECT_L_CM)
    approx(plane.to_ground((50, 100)), (RECT_W_CM / 2, RECT_L_CM / 2))


def test_trapezoid_corners_map_to_rectangle_corners():
    """사선 구도(사다리꼴)도 네 꼭짓점이 정확히 대응된다."""
    plane = GroundPlane.from_quad(TRAPEZOID_PX, TRAP_W_CM, TRAP_L_CM)
    approx(plane.to_ground(TRAPEZOID_PX[0]), (0, 0), tol=1e-4)
    approx(plane.to_ground(TRAPEZOID_PX[1]), (TRAP_W_CM, 0), tol=1e-4)
    approx(plane.to_ground(TRAPEZOID_PX[2]), (TRAP_W_CM, TRAP_L_CM), tol=1e-4)
    approx(plane.to_ground(TRAPEZOID_PX[3]), (0, TRAP_L_CM), tol=1e-4)


def test_perspective_correction_beats_raw_pixels():
    """이 테스트가 호모그래피를 쓰는 이유 그 자체다.

    사다리꼴에서 '가까운 쪽 절반'과 '먼 쪽 절반'은 화면상 픽셀 높이가 크게 다르지만
    (75px vs 75px 로 같아 보여도 실제 폭 축소 때문에 실거리 비율이 다르다), 평면 좌표로
    펴고 나면 두 구간의 실거리가 같아야 한다.
    """
    plane = GroundPlane.from_quad(TRAPEZOID_PX, TRAP_W_CM, TRAP_L_CM)

    # 화면 중앙선을 따라 픽셀 y를 200(시작) -> 125(중간) -> 50(끝) 으로 3등분.
    near = plane.to_ground((50, 200))
    mid = plane.to_ground((50, 125))
    far = plane.to_ground((50, 50))

    near_half_cm = mid[1] - near[1]
    far_half_cm = far[1] - mid[1]

    # 픽셀상으로는 두 구간이 똑같이 75px이지만, 실거리는 먼 쪽이 훨씬 길다.
    # (= 픽셀 속도를 그대로 쓰면 먼 쪽에서 걷는 사람이 느리게 측정된다.)
    assert far_half_cm > near_half_cm * 1.5
    # 그리고 둘의 합은 전체 길이가 되어야 한다.
    assert near_half_cm + far_half_cm == pytest.approx(TRAP_L_CM, abs=1e-4)


def test_point_outside_crosswalk_extrapolates():
    """횡단보도 밖의 점도 변환된다(범위 밖 값으로). 구역 판정은 zone.py가 따로 한다."""
    plane = GroundPlane.from_quad(RECT_PX, RECT_W_CM, RECT_L_CM)
    x_cm, y_cm = plane.to_ground((-20, 100))
    assert x_cm < 0


# --- 남은 거리 ---

def test_remaining_distance_forward():
    plane = GroundPlane.from_quad(RECT_PX, RECT_W_CM, RECT_L_CM)
    assert plane.remaining_distance_cm((25, 30), direction=1) == pytest.approx(70.0)


def test_remaining_distance_backward():
    plane = GroundPlane.from_quad(RECT_PX, RECT_W_CM, RECT_L_CM)
    assert plane.remaining_distance_cm((25, 30), direction=-1) == pytest.approx(30.0)


def test_remaining_distance_none_when_direction_unknown():
    plane = GroundPlane.from_quad(RECT_PX, RECT_W_CM, RECT_L_CM)
    assert plane.remaining_distance_cm((25, 30), direction=0) is None


def test_remaining_distance_clamped_at_zero():
    """횡단보도를 이미 벗어난 위치에서도 음수 거리가 나오지 않는다."""
    plane = GroundPlane.from_quad(RECT_PX, RECT_W_CM, RECT_L_CM)
    assert plane.remaining_distance_cm((25, 120), direction=1) == 0.0
