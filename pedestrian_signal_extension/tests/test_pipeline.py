"""파이프라인 배선 테스트.

검출기·카메라·시리얼을 가짜로 주입해, 실제 모델이나 하드웨어 없이
"검출 -> 발 위치 -> 구역 판정 / 속도 추정 -> 상태 요약 -> 전송" 흐름을 확인한다.

## 여기서 더 이상 검증하지 않는 것

잔여 시간 임계값, 누적 상한, 엣지 트리거, 사이클 리셋은 **제어부(아두이노)로 옮겨갔다.**
프로토콜을 4상태로 단순화하면서 파이는 "무엇을 보았는가"만 보내게 됐기 때문이다.
그 로직의 근거와 아두이노 쪽 주의사항은 docs/team_interface.md 에 있다.
"""

import pytest

from config import config
from src.detection import BoundingBox
from src.ground_plane import GroundPlane
from src.pipeline import CombinedPipeline, SignalExtensionPipeline
from src.serial_comm import STATE_EXTEND, STATE_FALL, STATE_NORMAL
from src.signal_extend import ZoneExtensionRule
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
    """보낸 줄을 기록한다. 실제 SerialComm의 변화 감지/하트비트까지 흉내내지는 않는다."""

    def __init__(self):
        self.states = []

    def update_state(self, state, extend_sec=None, now=None):
        self.states.append((state, extend_sec))
        return state

    def send_state(self, state, extend_sec=None, now=None):
        self.states.append((state, extend_sec))
        return state


class FakeAidDetector:
    """교통약자 보조 모델 대역. 가중치 없이 '보조기구가 보였는가'만 흉내낸다.

    기본값(빈 목록)은 config.MOBILITY_AID_MODEL_PATH가 None일 때의 실제 동작과 같다.
    """

    def __init__(self, frames=None):
        self.frames = list(frames) if frames is not None else []
        self.calls = 0

    def detect(self, frame):
        if not self.frames:
            return []
        boxes = self.frames[min(self.calls, len(self.frames) - 1)]
        self.calls += 1
        return boxes


def aid_box(label="wheelchair"):
    """보조 모델이 내는 박스. track_id가 없다(predict()라 추적을 안 붙인다)."""
    return BoundingBox(x1=0, y1=0, x2=20, y2=40, confidence=0.5, label=label)


def build_pipeline(frames, confirm_frames=1, width_cm=WIDTH_CM, length_cm=LENGTH_CM,
                   aid_frames=None, priority_zones=None):
    zones = CrosswalkZones.from_quad(CORNERS, n=5, width_cm=width_cm, length_cm=length_cm)
    plane = GroundPlane.from_quad(CORNERS, width_cm, length_cm)
    return SignalExtensionPipeline(
        camera=object(),
        detector=FakeDetector(frames),
        aid_detector=FakeAidDetector(aid_frames),
        zones=zones,
        occupancy=CrosswalkOccupancy(zones, confirm_frames=confirm_frames),
        rule=ZoneExtensionRule(zone_extension_sec=ZONE_EXT,
                               priority_zone_extension_sec=priority_zones),
        serial_comm=FakeSerial(),
        speed_estimator=SpeedEstimator(ground_plane=plane, window_sec=10.0),
    )


# --- 발 위치 기준 구역 판정 ---

def test_uses_foot_point_not_center_for_zone():
    """박스 중심과 발 위치가 서로 다른 구역에 걸치면, 발 위치 쪽이 채택돼야 한다.

    발이 y=100(3번 구역), 중심은 y=70(2번 구역)이 되도록 키 큰 박스를 만든다.
    사선 카메라에서 중심점을 쓰면 실제보다 카메라 쪽으로 당겨져 판정된다(CLAUDE.md 2.1).
    """
    box = box_at(50, 100, track_id=1, height=60)
    assert box.foot_point() == (50, 100)
    assert box.center_point() == (50, 70)

    pipeline = build_pipeline([[box]])
    result = pipeline.process_frame(frame=None, timestamp=0.0)

    assert result.pedestrians[0].zone == 3
    assert result.extension_sec == 5          # 2번(3초)이 아니라 3번(5초)


# --- 구역 -> 연장 요구 ---

@pytest.mark.parametrize("foot_y,zone,expected_sec", [
    (10, 1, 0), (50, 2, 3), (100, 3, 5), (150, 4, 3), (190, 5, 0),
])
def test_zone_maps_to_extension(foot_y, zone, expected_sec):
    pipeline = build_pipeline([[box_at(50, foot_y, track_id=1)]])
    result = pipeline.process_frame(None, timestamp=0.0)
    assert result.pedestrians[0].zone == zone
    assert result.extension_sec == expected_sec


def test_outside_crosswalk_is_not_counted():
    pipeline = build_pipeline([[box_at(500, 500, track_id=1)]])
    result = pipeline.process_frame(None, timestamp=0.0)
    assert result.extension_sec == 0
    assert result.occupied_zones == []


def test_residency_gate_blocks_until_confirmed():
    """잔류 프레임 수를 못 채운 검출은 연장 요구를 만들지 않는다."""
    frames = [[box_at(50, 100, track_id=1)]] * 3
    pipeline = build_pipeline(frames, confirm_frames=3)

    assert pipeline.process_frame(None, timestamp=0.0).extension_sec == 0
    assert pipeline.process_frame(None, timestamp=0.1).extension_sec == 0
    assert pipeline.process_frame(None, timestamp=0.2).extension_sec == 5


def test_takes_largest_need_among_people():
    boxes = [box_at(50, 50, 1), box_at(60, 100, 2), box_at(70, 10, 3)]
    pipeline = build_pipeline([boxes])
    result = pipeline.process_frame(None, timestamp=0.0)
    assert sorted(result.occupied_zones) == [1, 2, 3]
    assert result.extension_sec == 5


# --- 상태 요약 (아두이노로 나가는 형태) ---

def test_serial_state_is_normal_without_need():
    pipeline = build_pipeline([[box_at(50, 10, track_id=1)]])   # 1번 구역 -> 0초
    result = pipeline.process_frame(None, timestamp=0.0)
    assert result.serial_state() == (STATE_NORMAL, None)


def test_serial_state_carries_extension():
    pipeline = build_pipeline([[box_at(50, 100, track_id=1)]])
    result = pipeline.process_frame(None, timestamp=0.0)
    assert result.serial_state() == (STATE_EXTEND, 5)


# --- 보내는 값: ETA 있으면 ETA, 없으면 구역값 ---

def test_uses_zone_value_when_flag_off(monkeypatch):
    """플래그가 꺼져 있으면 걷고 있어도 구역값을 쓴다 (속도 검증 전까지의 안전한 동작)."""
    monkeypatch.setattr(config, "USE_SPEED_FOR_EXTENSION", False)
    frames = [[box_at(50, 100, 1)], [box_at(50, 120, 1)]]
    pipeline = build_pipeline(frames)
    pipeline.process_frame(None, timestamp=0.0)
    result = pipeline.process_frame(None, timestamp=1.0)
    assert result.extension_sec == 5              # 3번 구역값
    assert result.eta_sec is not None             # 계산은 되고 있다(표시용)


def test_uses_eta_when_flag_on(monkeypatch):
    """걷고 있는 사람이 있으면 보내는 값이 ETA가 된다."""
    monkeypatch.setattr(config, "USE_SPEED_FOR_EXTENSION", True)
    # 1초에 20px(=10cm) 이동. 3번 구역(y=120)에서 끝(200px=100cm)까지 40cm 남음 -> ETA 4초
    frames = [[box_at(50, 100, 1)], [box_at(50, 120, 1)]]
    pipeline = build_pipeline(frames)
    pipeline.process_frame(None, timestamp=0.0)
    result = pipeline.process_frame(None, timestamp=1.0)
    assert result.eta_sec is not None
    assert result.extension_sec == int(__import__("math").ceil(result.eta_sec))


def test_falls_back_to_zone_when_standing(monkeypatch):
    """서 있는 사람은 ETA가 없다 -> 구역값으로 폴백한다.

    가운데 서 있는 사람이야말로 연장이 가장 필요하므로, 여기서 0을 보내면 안 된다.
    """
    monkeypatch.setattr(config, "USE_SPEED_FOR_EXTENSION", True)
    frames = [[box_at(50, 100, 1)]] * 3
    pipeline = build_pipeline(frames)
    for t in (0.0, 1.0, 2.0):
        result = pipeline.process_frame(None, timestamp=t)
    assert result.eta_sec is None
    assert result.extension_sec == 5              # 구역값으로 폴백


def test_end_zone_is_gated_even_with_eta(monkeypatch):
    """양 끝 구역은 ETA가 아무리 커도 제외된다 — 구역은 게이트 역할을 유지한다."""
    monkeypatch.setattr(config, "USE_SPEED_FOR_EXTENSION", True)
    frames = [[box_at(50, 10, 1)], [box_at(50, 15, 1)]]     # 1번 구역
    pipeline = build_pipeline(frames)
    pipeline.process_frame(None, timestamp=0.0)
    result = pipeline.process_frame(None, timestamp=1.0)
    assert result.extension_sec == 0
    assert result.serial_state() == (STATE_NORMAL, None)


# --- 교통약자 우선 ---

def test_priority_mode_follows_aid_detector():
    pipeline = build_pipeline([[box_at(50, 100, 1)]], aid_frames=[[aid_box()]])
    result = pipeline.process_frame(None, timestamp=0.0)
    assert result.priority_mode is True
    assert result.mobility_aid_count == 1


def test_priority_mode_off_without_aid_detection():
    pipeline = build_pipeline([[box_at(50, 100, 1)]])
    result = pipeline.process_frame(None, timestamp=0.0)
    assert result.priority_mode is False
    assert result.mobility_aid_count == 0


def test_priority_only_changes_the_number():
    """'우선'은 보내는 숫자가 커지는 것일 뿐, 프로토콜은 그대로다."""
    priority_zones = {1: 0, 2: 4, 3: 6, 4: 4, 5: 0}
    pipeline = build_pipeline([[box_at(50, 100, 1)]], aid_frames=[[aid_box()]],
                              priority_zones=priority_zones)
    result = pipeline.process_frame(None, timestamp=0.0)
    assert result.serial_state() == (STATE_EXTEND, 6)


# --- 추론 1회 공유 ---

def test_process_boxes_does_not_call_detector():
    """CombinedPipeline이 쓰는 진입점 — 여기서 다시 검출하면 추론이 2배가 된다."""
    pipeline = build_pipeline([[box_at(50, 100, 1)]])
    boxes = [box_at(50, 100, track_id=1)]
    pipeline.process_boxes(boxes, timestamp=0.0)
    assert pipeline.detector.calls == 0


def test_untracked_detection_is_reported():
    """track_id가 없으면 잔류·속도 판정에서 제외되고, 그 사실이 보여야 한다."""
    pipeline = build_pipeline([[box_at(50, 100, track_id=None)]])
    result = pipeline.process_frame(None, timestamp=0.0)
    assert result.untracked_count == 1
    assert result.extension_sec == 0


# --- CombinedPipeline: 상태 우선순위 ---

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


def test_fall_wins_over_extend():
    """쓰러진 사람이 있으면 연장 요구보다 그쪽이 급하다."""
    combined, serial = build_combined(True, [box_at(50, 100, 1)])
    result = combined.process_frame(None, timestamp=0.0)

    assert result.state == STATE_FALL
    assert result.extend_sec is None
    assert serial.states[-1][0] == STATE_FALL
    # 연장 계산 자체는 그대로 돌아가고 있다(쓰러짐이 풀리면 곧바로 이어진다).
    assert result.extension.extension_sec == 5


def test_extend_resumes_when_fall_clears():
    combined, serial = build_combined(False, [box_at(50, 100, 1)])
    result = combined.process_frame(None, timestamp=0.0)

    assert result.state == STATE_EXTEND
    assert result.extend_sec == 5


def test_combined_runs_inference_once_per_frame():
    """이 클래스의 존재 이유 — 추론이 두 번 돌면 FPS가 반토막 난다."""
    combined, _ = build_combined(False, [box_at(50, 100, 1)])
    combined.process_frame(None, timestamp=0.0)
    combined.process_frame(None, timestamp=1.0)
    assert combined.detector.calls == 2
