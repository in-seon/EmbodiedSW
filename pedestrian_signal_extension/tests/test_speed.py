"""SpeedEstimator 단위 테스트.

시각(timestamp)을 인자로 주입하므로 카메라·모델 없이 전부 검증할 수 있다.
"""

import pytest

from src.ground_plane import GroundPlane
from src.speed import SpeedEstimator

# 픽셀 100x200 사각형이 실제 50cm x 100cm. 즉 픽셀 -> cm 비율은 x, y 모두 0.5.
RECT_PX = [(0, 0), (100, 0), (100, 200), (0, 200)]
WIDTH_CM, LENGTH_CM = 50.0, 100.0


@pytest.fixture
def plane():
    return GroundPlane.from_quad(RECT_PX, WIDTH_CM, LENGTH_CM)


@pytest.fixture
def estimator(plane):
    # 윈도우를 넉넉히 잡아, 테스트에서 넣은 샘플이 잘리지 않게 한다.
    return SpeedEstimator(ground_plane=plane, window_sec=10.0)


# --- 생성 검증 ---

def test_rejects_non_positive_window():
    with pytest.raises(ValueError):
        SpeedEstimator(window_sec=0)


def test_rejects_min_samples_below_two():
    """변위를 재려면 두 점이 필요하다."""
    with pytest.raises(ValueError):
        SpeedEstimator(window_sec=1.0, min_samples=1)


# --- 기본 속도 계산 ---

def test_first_sample_returns_none(estimator):
    """샘플이 하나뿐이면 속도를 낼 수 없다."""
    assert estimator.update("p1", (50, 0), timestamp=0.0) is None


def test_speed_in_cm_per_sec(estimator):
    """픽셀 y가 1초에 20px 이동 = 평면상 10cm 이동 -> 10 cm/s."""
    estimator.update("p1", (50, 0), timestamp=0.0)
    result = estimator.update("p1", (50, 20), timestamp=1.0)

    assert result.unit == "cm/s"
    assert result.is_metric is True
    assert result.crossing_speed == pytest.approx(10.0)
    assert result.speed == pytest.approx(10.0)
    assert result.direction == 1


def test_direction_backward(estimator):
    estimator.update("p1", (50, 100), timestamp=0.0)
    result = estimator.update("p1", (50, 60), timestamp=1.0)

    assert result.direction == -1
    assert result.crossing_speed == pytest.approx(20.0)  # 40px = 20cm, 1초


def test_direction_zero_when_stationary(estimator):
    estimator.update("p1", (50, 100), timestamp=0.0)
    result = estimator.update("p1", (50, 100), timestamp=1.0)

    assert result.direction == 0
    assert result.crossing_speed == pytest.approx(0.0)


def test_crossing_speed_excludes_sideways_motion(estimator):
    """crossing_speed는 걷는 방향(y) 성분만 본다 — 좌우 흔들림에 오염되지 않는다."""
    estimator.update("p1", (20, 0), timestamp=0.0)
    result = estimator.update("p1", (60, 20), timestamp=1.0)  # x로 40px(=20cm), y로 20px(=10cm)

    assert result.crossing_speed == pytest.approx(10.0)          # y 성분만
    assert result.speed == pytest.approx((20**2 + 10**2) ** 0.5)  # 전체 변위
    assert result.speed > result.crossing_speed


def test_zero_elapsed_time_returns_none(estimator):
    """같은 시각의 샘플이 둘이면 속도를 낼 수 없다(0으로 나누지 않는다)."""
    estimator.update("p1", (50, 0), timestamp=1.0)
    assert estimator.update("p1", (50, 20), timestamp=1.0) is None


# --- 호모그래피가 없을 때 (실측 치수 미입력) ---

def test_falls_back_to_pixels_without_ground_plane():
    est = SpeedEstimator(ground_plane=None, window_sec=10.0)
    est.update("p1", (50, 0), timestamp=0.0)
    result = est.update("p1", (50, 20), timestamp=1.0)

    assert result.unit == "px/s"
    assert result.is_metric is False
    assert result.crossing_speed == pytest.approx(20.0)  # 보정 없는 생 픽셀 값


def test_crossing_time_is_none_without_ground_plane():
    """px 단위 속도로는 통과 시간을 낼 수 없다 — cm인 척 하지 않고 None을 준다."""
    est = SpeedEstimator(ground_plane=None, window_sec=10.0)
    est.update("p1", (50, 0), timestamp=0.0)
    est.update("p1", (50, 20), timestamp=1.0)

    assert est.estimated_crossing_time_sec("p1") is None


# --- 예상 통과 시간 ---

def test_estimated_crossing_time(estimator):
    """평면 y=10cm 지점에서 10cm/s로 전진 -> 남은 90cm를 9초에 통과."""
    estimator.update("p1", (50, 0), timestamp=0.0)
    estimator.update("p1", (50, 20), timestamp=1.0)  # 평면 y=10cm, 10cm/s

    assert estimator.estimated_crossing_time_sec("p1") == pytest.approx(9.0)


def test_estimated_crossing_time_backward(estimator):
    """반대 방향으로 걸으면 시작 변(y=0)까지 남은 거리로 계산한다."""
    estimator.update("p1", (50, 100), timestamp=0.0)
    estimator.update("p1", (50, 80), timestamp=1.0)  # 평면 y=40cm, 10cm/s, 방향 -1

    assert estimator.estimated_crossing_time_sec("p1") == pytest.approx(4.0)


def test_estimated_crossing_time_none_when_stationary(estimator):
    estimator.update("p1", (50, 100), timestamp=0.0)
    estimator.update("p1", (50, 100), timestamp=1.0)

    assert estimator.estimated_crossing_time_sec("p1") is None


def test_estimated_crossing_time_none_for_unknown_track(estimator):
    assert estimator.estimated_crossing_time_sec("nobody") is None


# --- 윈도우 동작 ---

def test_uses_window_not_single_frame_diff():
    """한 프레임 지터가 속도를 지배하지 않아야 한다.

    일정 속도로 걷다가 마지막 프레임에서만 검출 박스가 크게 튀는 상황.
    한 프레임 차분이면 속도가 폭증하지만, 윈도우 평균이면 흔들림이 눌린다.
    """
    est = SpeedEstimator(ground_plane=None, window_sec=1.0)
    # 0.0 ~ 1.0초 동안 10px/s 로 등속.
    for i in range(11):
        est.update("p1", (50, i), timestamp=i * 0.1)
    steady = est.latest("p1").crossing_speed

    # 마지막 프레임에서 박스가 20px 튐.
    jittered = est.update("p1", (50, 30), timestamp=1.1).crossing_speed

    single_frame_diff = (30 - 10) / 0.1  # = 200 px/s
    assert steady == pytest.approx(10.0, rel=0.2)
    assert jittered < single_frame_diff / 2


def test_window_trims_old_samples():
    """윈도우보다 오래된 샘플은 버려져 측정 구간이 window_sec 근처로 유지된다."""
    est = SpeedEstimator(ground_plane=None, window_sec=1.0)
    est.update("p1", (50, 0), timestamp=0.0)
    est.update("p1", (50, 100), timestamp=1.0)   # 이 구간은 빠름
    est.update("p1", (50, 110), timestamp=2.0)
    result = est.update("p1", (50, 120), timestamp=3.0)

    # t=0의 빠른 구간이 빠지고 최근 10px/s 구간만 반영돼야 한다.
    assert result.crossing_speed == pytest.approx(10.0)


def test_keeps_two_samples_even_if_both_older_than_window():
    """샘플이 드물게 들어와도(느린 FPS) 속도를 포기하지 않는다."""
    est = SpeedEstimator(ground_plane=None, window_sec=0.1)
    est.update("p1", (50, 0), timestamp=0.0)
    result = est.update("p1", (50, 10), timestamp=5.0)

    assert result is not None
    assert result.crossing_speed == pytest.approx(2.0)


# --- 여러 트랙 ---

def test_update_many_tracks_each_pedestrian(estimator):
    estimator.update_many([("p1", (50, 0)), ("p2", (20, 100))], timestamp=0.0)
    results = estimator.update_many([("p1", (50, 20)), ("p2", (20, 80))], timestamp=1.0)

    assert set(results) == {"p1", "p2"}
    assert results["p1"].direction == 1
    assert results["p2"].direction == -1


def test_update_many_ignores_untracked_detections(estimator):
    """track_id가 None이면 프레임 간 대응을 알 수 없어 속도를 낼 근거가 없다."""
    estimator.update_many([(None, (50, 0))], timestamp=0.0)
    results = estimator.update_many([(None, (50, 20))], timestamp=1.0)

    assert results == {}


def test_update_many_drops_vanished_track(estimator):
    """유예를 넘겨 사라진 트랙의 히스토리는 지운다 — 공백을 가로질러 속도를 재면 값이 왜곡된다.

    긴 공백을 이어붙이면 '끊긴 구간을 계속 이동한 것'으로 쳐서 실제보다 훨씬 느린 속도가
    나오고, 느린 속도는 곧 과대한 ETA -> 불필요한 연장이 된다.
    """
    estimator.update_many([("p1", (50, 0))], timestamp=0.0)
    estimator.update_many([("p1", (50, 20))], timestamp=1.0)
    assert estimator.latest("p1") is not None

    for t in (2.0, 3.0, 4.0):     # 미검출 3프레임 > 유예 2프레임
        estimator.update_many([], timestamp=t)
    assert estimator.latest("p1") is None

    # 다시 나타나면 처음부터 다시 쌓는다.
    assert estimator.update_many([("p1", (50, 90))], timestamp=5.0) == {}


def test_update_many_tolerates_brief_gap(estimator):
    """유예 안의 깜빡임에는 히스토리를 버리지 않는다 (config.TRACK_GRACE_FRAMES).

    버리면 다시 SPEED_WINDOW_SEC만큼 샘플을 쌓아야 하고, 그동안 ETA가 None이 되어
    속도 기반 연장이 조용히 꺼진다.
    """
    estimator.update_many([("p1", (50, 0))], timestamp=0.0)
    estimator.update_many([("p1", (50, 20))], timestamp=1.0)

    estimator.update_many([], timestamp=2.0)      # 미검출 1 (유예 이내)
    assert estimator.latest("p1") is not None     # 직전 속도를 계속 들고 있는다

    # 다시 나타나면 기존 히스토리에 이어붙어 곧바로 속도가 나온다.
    assert "p1" in estimator.update_many([("p1", (50, 60))], timestamp=3.0)


def test_grace_zero_drops_immediately(plane):
    """grace_frames=0이면 유예 도입 전과 같은 동작."""
    estimator = SpeedEstimator(ground_plane=plane, window_sec=10.0, grace_frames=0)
    estimator.update_many([("p1", (50, 0))], timestamp=0.0)
    estimator.update_many([("p1", (50, 20))], timestamp=1.0)

    estimator.update_many([], timestamp=2.0)
    assert estimator.latest("p1") is None


def test_rejects_negative_grace():
    with pytest.raises(ValueError):
        SpeedEstimator(window_sec=1.0, grace_frames=-1)


def test_unit_reflects_ground_plane_presence(plane):
    assert SpeedEstimator(ground_plane=plane).unit == "cm/s"
    assert SpeedEstimator(ground_plane=None).unit == "px/s"


def test_clear_resets_all_state(estimator):
    estimator.update_many([("p1", (50, 0))], timestamp=0.0)
    estimator.update_many([("p1", (50, 20))], timestamp=1.0)
    estimator.clear()

    assert estimator.latest("p1") is None
    assert estimator.estimated_crossing_time_sec("p1") is None
