"""파이프라인 배선 테스트."""

import pytest

from config import config
from src.detection import BoundingBox
from src.ground_plane import GroundPlane
from src.pipeline import CombinedPipeline, SignalExtensionPipeline
from src.serial_comm import STATE_FALL, STATE_NORMAL, STATE_ZONE
from src.speed import SpeedEstimator
from src.zone import CrosswalkOccupancy, CrosswalkZones

CORNERS = [(0, 0), (100, 0), (100, 200), (0, 200)]
WIDTH_CM, LENGTH_CM = 50.0, 100.0

ZONE_EXT = {1: 0, 2: 3, 3: 5, 4: 3, 5: 0}


def box_at(foot_x, foot_y, track_id, label="person", height=40):
    y2 = foot_y + config.FOOT_POINT_OFFSET_RATIO * height
    return BoundingBox(
        x1=foot_x - 10, y1=y2 - height, x2=foot_x + 10, y2=y2,
        confidence=0.9, label=label, track_id=track_id,
    )


class FakeDetector:

    def __init__(self, frames):
        self.frames = list(frames)
        self.calls = 0

    def detect(self, frame):
        boxes = self.frames[min(self.calls, len(self.frames) - 1)]
        self.calls += 1
        return boxes


class FakeSerial:

    def __init__(self):
        self.states = []

    def update_state(self, state, zone=None, now=None):
        self.states.append((state, zone))
        return state

    def send_state(self, state, zone=None, now=None):
        self.states.append((state, zone))
        return state

def build_pipeline(frames, confirm_frames=1, width_cm=WIDTH_CM, length_cm=LENGTH_CM):
    zones = CrosswalkZones.from_quad(CORNERS, n=5, width_cm=width_cm, length_cm=length_cm)
    plane = GroundPlane.from_quad(CORNERS, width_cm, length_cm)
    return SignalExtensionPipeline(
        camera=object(),
        detector=FakeDetector(frames),
        zones=zones,
        occupancy=CrosswalkOccupancy(zones, confirm_frames=confirm_frames,
                                     grace_frames=2),
        serial_comm=FakeSerial(),
        speed_estimator=SpeedEstimator(ground_plane=plane, window_sec=10.0),
    )


def test_uses_foot_point_not_center_for_zone():
    box = box_at(50, 100, track_id=1, height=60)
    assert box.foot_point() == (50, 100)
    assert box.center_point() == (50, 76)

    pipeline = build_pipeline([[box]])
    result = pipeline.process_frame(frame=None, timestamp=0.0)

    assert result.pedestrians[0].zone == 3


@pytest.mark.parametrize("foot_y,zone", [(10, 1), (50, 2), (100, 3), (150, 4), (190, 5)])
def test_physical_zone_is_reported(foot_y, zone):
    pipeline = build_pipeline([[box_at(50, foot_y, track_id=1)]])
    result = pipeline.process_frame(None, timestamp=0.0)
    assert result.pedestrians[0].zone == zone


def test_forward_entry_progress_equals_zone():
    frames = [[box_at(50, 10, 1)], [box_at(50, 100, 1)]]
    pipeline = build_pipeline(frames)
    pipeline.process_frame(None, timestamp=0.0)
    result = pipeline.process_frame(None, timestamp=1.0)
    assert result.progress == 3
    assert result.serial_state() == (STATE_ZONE, 3)


def test_reverse_entry_progress_is_mirrored():
    frames = [[box_at(50, 190, 1)], [box_at(50, 50, 1)]]
    pipeline = build_pipeline(frames)
    pipeline.process_frame(None, timestamp=0.0)
    result = pipeline.process_frame(None, timestamp=1.0)
    assert result.pedestrians[0].zone == 2
    assert result.progress == 4
    assert result.serial_state() == (STATE_ZONE, 4)


def test_opposite_directions_send_the_least_advanced():
    """반대 방향 두 사람이 같은 물리 구역에 있어도, 덜 건넌 쪽이 선택됨.
    보정하지 않으면 둘 다 '2번'이라 구분되지 않음.
    """
    enter = [box_at(50, 10, 1), box_at(60, 190, 2)]
    both_at_zone2 = [box_at(50, 50, 1), box_at(60, 50, 2)]
    pipeline = build_pipeline([enter, both_at_zone2])

    pipeline.process_frame(None, timestamp=0.0)
    result = pipeline.process_frame(None, timestamp=1.0)

    zones = {p.track_id: p.zone for p in result.pedestrians}
    progress = {p.track_id: p.progress for p in result.pedestrians}
    assert zones == {1: 2, 2: 2}
    assert progress == {1: 2, 2: 4}
    assert result.progress == 2


def test_middle_start_is_treated_as_middle():
    """중앙에서 처음 보이면 방향을 모른다 -> 중앙값(3)으로 간주."""
    pipeline = build_pipeline([[box_at(50, 100, 1)]])
    result = pipeline.process_frame(None, timestamp=0.0)
    assert result.progress == 3
    assert result.pedestrians[0].direction is None


def test_no_one_means_normal():
    pipeline = build_pipeline([[]])
    result = pipeline.process_frame(None, timestamp=0.0)
    assert result.progress is None
    assert result.serial_state() == (STATE_NORMAL, None)


def test_outside_crosswalk_is_not_counted():
    pipeline = build_pipeline([[box_at(500, 500, track_id=1)]])
    result = pipeline.process_frame(None, timestamp=0.0)
    assert result.progress is None
    assert result.occupied_zones == []


def test_residency_gate_blocks_until_confirmed():
    """잔류 프레임 수를 못 채운 검출은 진척도를 만들지 않는다.
    방향 래치도 확정 후에만 걸린다 — 스쳐가는 오검출로 방향이 잘못 굳는 것을 막는다.
    """
    frames = [[box_at(50, 100, track_id=1)]] * 3
    pipeline = build_pipeline(frames, confirm_frames=3)

    assert pipeline.process_frame(None, timestamp=0.0).progress is None
    assert pipeline.process_frame(None, timestamp=0.1).progress is None
    assert pipeline.process_frame(None, timestamp=0.2).progress == 3


def test_eta_is_measured_but_does_not_change_the_sent_value():
    frames = [[box_at(50, 10, 1)], [box_at(50, 100, 1)]]
    pipeline = build_pipeline(frames)
    pipeline.process_frame(None, timestamp=0.0)
    result = pipeline.process_frame(None, timestamp=1.0)

    assert result.eta_sec is not None
    assert result.progress == 3


def test_standing_person_still_reports_progress():
    frames = [[box_at(50, 100, 1)]] * 3
    pipeline = build_pipeline(frames)
    for t in (0.0, 1.0, 2.0):
        result = pipeline.process_frame(None, timestamp=t)
    assert result.eta_sec is None
    assert result.progress == 3


def test_process_boxes_does_not_call_detector():
    pipeline = build_pipeline([[box_at(50, 100, 1)]])
    pipeline.process_boxes([box_at(50, 100, track_id=1)], timestamp=0.0)
    assert pipeline.detector.calls == 0


def test_untracked_detection_is_reported():
    pipeline = build_pipeline([[box_at(50, 100, track_id=None)]])
    result = pipeline.process_frame(None, timestamp=0.0)
    assert result.untracked_count == 1
    assert result.progress is None


class StubFall:
    def __init__(self, confirmed):
        self.confirmed = confirmed
        self.roi_px = (0, 0, 100, 200)

    def process_boxes(self, boxes, frame, now=None):
        from src.pipeline import FallAlarmResult

        return FallAlarmResult(fall_confirmed=self.confirmed,
                               people_count=len(boxes))

    def reset_alarm(self):
        self.confirmed = False


def build_combined(fall_confirmed, boxes):
    serial = FakeSerial()
    extension = build_pipeline([boxes])
    combined = CombinedPipeline(
        camera=object(),
        detector=FakeDetector([boxes]),
        serial_comm=serial,
        extension=extension,
        fall=StubFall(fall_confirmed),
        zones=extension.zones,
    )
    return combined, serial


def test_fall_wins_over_zone():
    combined, serial = build_combined(True, [box_at(50, 100, 1)])
    result = combined.process_frame(None, timestamp=0.0)

    assert result.state == STATE_FALL
    assert result.zone is None
    assert serial.states[-1][0] == STATE_FALL
    assert result.extension.progress == 3


def test_zone_resumes_when_fall_clears():
    combined, serial = build_combined(False, [box_at(50, 100, 1)])
    result = combined.process_frame(None, timestamp=0.0)

    assert result.state == STATE_ZONE
    assert result.zone == 3


def test_combined_runs_inference_once_per_frame():
    combined, _ = build_combined(False, [box_at(50, 100, 1)])
    combined.process_frame(None, timestamp=0.0)
    combined.process_frame(None, timestamp=1.0)
    assert combined.detector.calls == 2


def test_zone_resumes_immediately_after_fall_without_normal():
    boxes = [box_at(50, 100, 1)]
    serial = FakeSerial()
    extension = build_pipeline([boxes] * 3)
    fall = StubFall(True)
    combined = CombinedPipeline(
        camera=object(), detector=FakeDetector([boxes] * 3), serial_comm=serial,
        extension=extension, fall=fall, zones=extension.zones,
    )

    assert combined.process_frame(None, timestamp=0.0).state == STATE_FALL
    fall.confirmed = False
    resumed = combined.process_frame(None, timestamp=1.0)

    assert resumed.state == STATE_ZONE
    assert [s for s, _ in serial.states] == [STATE_FALL, STATE_ZONE]
    assert STATE_NORMAL not in [s for s, _ in serial.states]


def test_normal_means_nobody_not_fall_cleared():
    serial = FakeSerial()
    extension = build_pipeline([[]])
    combined = CombinedPipeline(
        camera=object(), detector=FakeDetector([[]]), serial_comm=serial,
        extension=extension, fall=StubFall(False), zones=extension.zones,
    )
    assert combined.process_frame(None, timestamp=0.0).state == STATE_NORMAL


def test_progress_survives_track_id_reissue():
    frames = [
        [box_at(50, 20, 1)],
        [box_at(50, 60, 1)],
        [box_at(50, 100, 1)],
        [box_at(50, 140, 7)],
        [box_at(50, 180, 7)],
    ]
    pipeline = build_pipeline(frames)
    pipeline.occupancy.inherit_distance = 60

    seen = [pipeline.process_frame(None, timestamp=i * 0.1).progress
            for i in range(len(frames))]
    assert seen[-2:] == [4, 5], f"진척도가 뒤집혔다: {seen}"


def test_progress_is_reversed_without_inheritance():
    frames = [
        [box_at(50, 20, 1)],
        [box_at(50, 60, 1)],
        [box_at(50, 100, 1)],
        [box_at(50, 140, 7)],
        [box_at(50, 180, 7)],
    ]
    pipeline = build_pipeline(frames)
    pipeline.occupancy.inherit_distance = 0

    seen = [pipeline.process_frame(None, timestamp=i * 0.1).progress
            for i in range(len(frames))]
    assert seen[-2:] == [2, 1], f"상속을 껐는데 뒤집히지 않았다: {seen}"


def test_reissue_keeps_sending_zone_without_a_normal_gap():
    inside = [box_at(50, 100, 1)]
    frames = [inside, inside, [box_at(50, 105, 7)]]
    pipeline = build_pipeline(frames)
    pipeline.occupancy.inherit_distance = 60

    states = [pipeline.process_frame(None, timestamp=i * 0.1).serial_state()[0]
              for i in range(len(frames))]
    assert states[-1] == STATE_ZONE
    assert STATE_NORMAL not in states[1:]
