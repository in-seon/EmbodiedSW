"""SignalExtensionPipeline 배선 테스트.

검출기·카메라·시리얼을 가짜로 주입해, 실제 모델이나 하드웨어 없이
"검출 -> 발 위치 -> 구역 판정 / 속도 추정 -> 연장 결정 -> 전송" 흐름이 맞물리는지 확인한다.
"""

import pytest

from src.detection import BoundingBox
from src.ground_plane import GroundPlane
from src.pipeline import SignalExtensionPipeline
from src.signal_extend import SignalExtensionStateMachine
from src.speed import SpeedEstimator
from src.zone import CrosswalkOccupancy, CrosswalkZones

# 픽셀 100x200 사각형 = 실제 50cm x 100cm. 걷는 방향은 화면 y 증가 방향.
CORNERS = [(0, 0), (100, 0), (100, 200), (0, 200)]
WIDTH_CM, LENGTH_CM = 50.0, 100.0

ZONE_EXT = {1: 0, 2: 3, 3: 5, 4: 3, 5: 0}


def box_at(foot_x, foot_y, track_id, label="person", height=40):
    """발 위치가 (foot_x, foot_y)에 오도록 BoundingBox를 만든다.

    foot_point()는 (하단 모서리 중심) = ((x1+x2)/2, y2) 이므로 그에 맞춰 역산한다.
    """
    return BoundingBox(
        x1=foot_x - 10, y1=foot_y - height, x2=foot_x + 10, y2=foot_y,
        confidence=0.9, label=label, track_id=track_id,
    )


class FakeDetector:
    """프레임 대신 미리 정한 박스 목록을 순서대로 내놓는다."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.calls = 0

    def detect(self, frame):
        boxes = self.frames[min(self.calls, len(self.frames) - 1)]
        self.calls += 1
        return boxes


class FakeSerial:
    def __init__(self):
        self.sent = []

    def send_extend_signal(self, extension_sec, priority=False):
        self.sent.append((extension_sec, priority))


def build_pipeline(frames, confirm_frames=1, width_cm=WIDTH_CM, length_cm=LENGTH_CM,
                   threshold_sec=5, max_total_sec=20):
    zones = CrosswalkZones.from_quad(CORNERS, n=5, width_cm=width_cm, length_cm=length_cm)
    return SignalExtensionPipeline(
        camera=object(),
        detector=FakeDetector(frames),
        zones=zones,
        occupancy=CrosswalkOccupancy(zones, confirm_frames=confirm_frames),
        state_machine=SignalExtensionStateMachine(
            remaining_time_threshold_sec=threshold_sec,
            zone_extension_sec=ZONE_EXT,
            max_total_extension_sec=max_total_sec,
        ),
        serial_comm=FakeSerial(),
        speed_estimator=SpeedEstimator(
            ground_plane=GroundPlane.from_quad(CORNERS, width_cm, length_cm),
            window_sec=10.0,
        ),
    )


# --- 발 위치 기준 구역 판정 ---

def test_uses_foot_point_not_center_for_zone():
    """박스 중심과 발 위치가 서로 다른 구역에 걸치면, 발 위치 쪽이 채택돼야 한다.

    발이 y=100(3번 구역), 중심은 y=70(2번 구역)이 되도록 키 큰 박스를 만든다.
    사선 카메라에서 중심점을 쓰면 실제보다 카메라 쪽으로 당겨져 판정되는 문제(CLAUDE.md 2.1).
    """
    box = box_at(50, 100, track_id=1, height=60)
    assert box.foot_point() == (50, 100)
    assert box.center_point() == (50, 70)

    pipeline = build_pipeline([[box]])
    result = pipeline.process_frame(frame=None, remaining_time_sec=3, timestamp=0.0)

    assert [p.zone for p in result.pedestrians] == [3]   # 발 위치 기준
    assert result.occupied_zones == [3]


def test_pedestrian_outside_crosswalk_has_no_zone():
    pipeline = build_pipeline([[box_at(500, 500, track_id=1)]])
    result = pipeline.process_frame(frame=None, remaining_time_sec=3, timestamp=0.0)

    assert result.pedestrians[0].zone is None
    assert result.occupied_zones == []
    assert result.extension_sec == 0


# --- 속도 추정 연동 ---

def test_speed_needs_two_frames():
    frames = [[box_at(50, 20, 1)], [box_at(50, 40, 1)]]
    pipeline = build_pipeline(frames)

    first = pipeline.process_frame(None, remaining_time_sec=3, timestamp=0.0)
    assert first.pedestrians[0].speed is None  # 샘플 1개 -> 아직 못 냄

    second = pipeline.process_frame(None, remaining_time_sec=3, timestamp=1.0)
    speed = second.pedestrians[0].speed
    assert speed is not None
    assert speed.unit == "cm/s"
    assert speed.crossing_speed == pytest.approx(10.0)  # 20px = 10cm, 1초
    assert speed.direction == 1


def test_estimated_crossing_time_exposed():
    """평면 y=20cm 지점에서 10cm/s -> 남은 80cm를 8초에 통과."""
    frames = [[box_at(50, 20, 1)], [box_at(50, 40, 1)]]
    pipeline = build_pipeline(frames)

    pipeline.process_frame(None, remaining_time_sec=3, timestamp=0.0)
    result = pipeline.process_frame(None, remaining_time_sec=3, timestamp=1.0)

    assert result.pedestrians[0].crossing_time_sec == pytest.approx(8.0)
    assert result.speed_unit == "cm/s"


def test_falls_back_to_pixels_without_dimensions():
    """실측 치수가 없으면 구역 판정은 되고 속도만 px/s로 떨어진다."""
    frames = [[box_at(50, 20, 1)], [box_at(50, 40, 1)]]
    pipeline = build_pipeline(frames, width_cm=None, length_cm=None)

    pipeline.process_frame(None, remaining_time_sec=3, timestamp=0.0)
    result = pipeline.process_frame(None, remaining_time_sec=3, timestamp=1.0)

    assert result.speed_unit == "px/s"
    assert result.pedestrians[0].zone == 1              # 구역 판정은 정상
    assert result.pedestrians[0].speed.crossing_speed == pytest.approx(20.0)
    assert result.pedestrians[0].crossing_time_sec is None  # 실거리로 환산 불가


# --- 연장 결정 / 전송 ---

def test_center_zone_triggers_longest_extension():
    pipeline = build_pipeline([[box_at(50, 100, 1)]])  # 3번 구역
    result = pipeline.process_frame(None, remaining_time_sec=3, timestamp=0.0)

    assert result.extension_sec == ZONE_EXT[3]
    assert pipeline.serial_comm.sent == [(5, False)]


def test_edge_zone_does_not_extend():
    pipeline = build_pipeline([[box_at(50, 20, 1)]])  # 1번 구역
    result = pipeline.process_frame(None, remaining_time_sec=3, timestamp=0.0)

    assert result.extension_sec == 0
    assert pipeline.serial_comm.sent == []


def test_no_extension_when_time_is_plentiful():
    pipeline = build_pipeline([[box_at(50, 100, 1)]])  # 3번 구역이지만
    result = pipeline.process_frame(None, remaining_time_sec=30, timestamp=0.0)  # 시간 충분

    assert result.extension_sec == 0
    assert pipeline.serial_comm.sent == []


def test_multiple_pedestrians_take_the_largest_need():
    """1번 구역(0초)과 3번 구역(5초)에 동시에 있으면 가장 많이 필요한 쪽에 맞춘다."""
    pipeline = build_pipeline([[box_at(50, 20, 1), box_at(50, 100, 2)]])
    result = pipeline.process_frame(None, remaining_time_sec=3, timestamp=0.0)

    assert sorted(result.occupied_zones) == [1, 3]
    assert result.extension_sec == ZONE_EXT[3]


def test_unconfirmed_pedestrian_does_not_extend():
    """ZONE_RESIDENCY_FRAMES를 못 채운 검출은 연장을 발동시키지 않는다."""
    frames = [[box_at(50, 100, 1)]] * 3
    pipeline = build_pipeline(frames, confirm_frames=3)

    first = pipeline.process_frame(None, remaining_time_sec=3, timestamp=0.0)
    assert first.pedestrians[0].confirmed is False
    assert first.extension_sec == 0

    pipeline.process_frame(None, remaining_time_sec=3, timestamp=1.0)
    third = pipeline.process_frame(None, remaining_time_sec=3, timestamp=2.0)
    assert third.pedestrians[0].confirmed is True
    assert third.extension_sec == ZONE_EXT[3]


# --- 교통약자 보류 상태 ---

def test_priority_mode_off_while_mobility_aids_are_shelved():
    """MOBILITY_AID_LABELS가 비어 있는 동안(보류) 우선 연장 모드는 켜지지 않는다."""
    boxes = [box_at(50, 100, 1), box_at(60, 100, 2, label="wheelchair")]
    pipeline = build_pipeline([boxes])
    result = pipeline.process_frame(None, remaining_time_sec=3, timestamp=0.0)

    assert result.priority_mode is False
    # 'wheelchair' 박스는 사람이 아니므로 보행자 목록에도 들어가지 않는다.
    assert [p.track_id for p in result.pedestrians] == [1]
