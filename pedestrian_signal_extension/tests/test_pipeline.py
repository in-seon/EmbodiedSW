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
from src.serial_comm import STATE_FALL, STATE_NORMAL, STATE_ZONE
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
        occupancy=CrosswalkOccupancy(zones, confirm_frames=confirm_frames),
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


# --- 진척도 (진입 방향 보정) ---

@pytest.mark.parametrize("foot_y,zone", [(10, 1), (50, 2), (100, 3), (150, 4), (190, 5)])
def test_physical_zone_is_reported(foot_y, zone):
    pipeline = build_pipeline([[box_at(50, foot_y, track_id=1)]])
    result = pipeline.process_frame(None, timestamp=0.0)
    assert result.pedestrians[0].zone == zone


def test_forward_entry_progress_equals_zone():
    """1번에서 진입하면 진척도가 구역 번호와 같다."""
    frames = [[box_at(50, 10, 1)], [box_at(50, 100, 1)]]
    pipeline = build_pipeline(frames)
    pipeline.process_frame(None, timestamp=0.0)
    result = pipeline.process_frame(None, timestamp=1.0)
    assert result.progress == 3
    assert result.serial_state() == (STATE_ZONE, 3)


def test_reverse_entry_progress_is_mirrored():
    """5번에서 진입하면 진척도가 뒤집힌다 — 물리 2번이 진척도 4가 된다."""
    frames = [[box_at(50, 190, 1)], [box_at(50, 50, 1)]]
    pipeline = build_pipeline(frames)
    pipeline.process_frame(None, timestamp=0.0)
    result = pipeline.process_frame(None, timestamp=1.0)
    assert result.pedestrians[0].zone == 2      # 물리 구역
    assert result.progress == 4                  # 거의 다 건넜음
    assert result.serial_state() == (STATE_ZONE, 4)


def test_opposite_directions_send_the_least_advanced():
    """반대 방향 두 사람이 같은 물리 구역에 있어도, 덜 건넌 쪽이 선택된다.

    보정하지 않으면 둘 다 '2번'이라 구분되지 않는다 — 이 파이프라인의 핵심 요구사항.
    """
    enter = [box_at(50, 10, 1), box_at(60, 190, 2)]      # 1번 진입 / 5번 진입
    both_at_zone2 = [box_at(50, 50, 1), box_at(60, 50, 2)]
    pipeline = build_pipeline([enter, both_at_zone2])

    pipeline.process_frame(None, timestamp=0.0)
    result = pipeline.process_frame(None, timestamp=1.0)

    zones = {p.track_id: p.zone for p in result.pedestrians}
    progress = {p.track_id: p.progress for p in result.pedestrians}
    assert zones == {1: 2, 2: 2}                 # 물리적으로 같은 구역
    assert progress == {1: 2, 2: 4}              # 진척도는 다르다
    assert result.progress == 2                  # 덜 건넌 1번 기준


def test_middle_start_is_treated_as_middle():
    """중앙에서 처음 보이면 방향을 모른다 -> 중앙값(3)으로 간주."""
    pipeline = build_pipeline([[box_at(50, 100, 1)]])
    result = pipeline.process_frame(None, timestamp=0.0)
    assert result.progress == 3
    assert result.pedestrians[0].direction is None


# --- 아무도 없을 때 ---

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


# --- 속도는 연장 판단에 쓰이지 않는다 ---

def test_eta_is_measured_but_does_not_change_the_sent_value():
    """ETA는 계측용이다. 있든 없든 보내는 진척도는 같아야 한다."""
    frames = [[box_at(50, 10, 1)], [box_at(50, 100, 1)]]
    pipeline = build_pipeline(frames)
    pipeline.process_frame(None, timestamp=0.0)
    result = pipeline.process_frame(None, timestamp=1.0)

    assert result.eta_sec is not None          # 계측은 되고 있다
    assert result.progress == 3                 # 값은 구역만으로 정해진다


def test_standing_person_still_reports_progress():
    """서 있는 사람은 ETA가 없지만 진척도는 나온다 — 연장 판단이 속도에 의존하지 않는다.

    예전 방식에서는 여기서 ETA가 None이라 폴백이 필요했다. 지금은 폴백 자체가 없다.
    """
    frames = [[box_at(50, 100, 1)]] * 3
    pipeline = build_pipeline(frames)
    for t in (0.0, 1.0, 2.0):
        result = pipeline.process_frame(None, timestamp=t)
    assert result.eta_sec is None
    assert result.progress == 3


# --- 추론 1회 공유 ---

def test_process_boxes_does_not_call_detector():
    """CombinedPipeline이 쓰는 진입점 — 여기서 다시 검출하면 추론이 2배가 된다."""
    pipeline = build_pipeline([[box_at(50, 100, 1)]])
    pipeline.process_boxes([box_at(50, 100, track_id=1)], timestamp=0.0)
    assert pipeline.detector.calls == 0


def test_untracked_detection_is_reported():
    """track_id가 없으면 잔류·진척도 판정에서 제외되고, 그 사실이 보여야 한다."""
    pipeline = build_pipeline([[box_at(50, 100, track_id=None)]])
    result = pipeline.process_frame(None, timestamp=0.0)
    assert result.untracked_count == 1
    assert result.progress is None


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


def test_fall_wins_over_zone():
    """쓰러진 사람이 있으면 연장 요구보다 그쪽이 급하다."""
    combined, serial = build_combined(True, [box_at(50, 100, 1)])
    result = combined.process_frame(None, timestamp=0.0)

    assert result.state == STATE_FALL
    assert result.zone is None
    assert serial.states[-1][0] == STATE_FALL
    # 구역 판정 자체는 그대로 돌아가고 있다(쓰러짐이 풀리면 곧바로 이어진다).
    assert result.extension.progress == 3


def test_zone_resumes_when_fall_clears():
    combined, serial = build_combined(False, [box_at(50, 100, 1)])
    result = combined.process_frame(None, timestamp=0.0)

    assert result.state == STATE_ZONE
    assert result.zone == 3


def test_combined_runs_inference_once_per_frame():
    """이 클래스의 존재 이유 — 추론이 두 번 돌면 FPS가 반토막 난다."""
    combined, _ = build_combined(False, [box_at(50, 100, 1)])
    combined.process_frame(None, timestamp=0.0)
    combined.process_frame(None, timestamp=1.0)
    assert combined.detector.calls == 2


def test_zone_resumes_immediately_after_fall_without_normal():
    """쓰러졌다 일어나 횡단보도에 남으면 FALL -> ZONE 으로 바로 넘어간다 (NORMAL이 끼지 않는다).

    아두이노가 "normal을 받아야 부저를 끈다"로 짜면 이 구간에서 부저가 영영 울린다.
    계약은 `부저 = (마지막 상태 == FALL)` 이다 — docs/team_interface.md 참고.
    """
    boxes = [box_at(50, 100, 1)]
    serial = FakeSerial()
    extension = build_pipeline([boxes] * 3)
    fall = StubFall(True)
    combined = CombinedPipeline(
        camera=object(), detector=FakeDetector([boxes] * 3), serial_comm=serial,
        extension=extension, fall=fall, zones=extension.zones,
    )

    assert combined.process_frame(None, timestamp=0.0).state == STATE_FALL
    fall.confirmed = False                       # 일어났다 (횡단보도에는 그대로 있다)
    resumed = combined.process_frame(None, timestamp=1.0)

    assert resumed.state == STATE_ZONE
    assert [s for s, _ in serial.states] == [STATE_FALL, STATE_ZONE]
    assert STATE_NORMAL not in [s for s, _ in serial.states]


def test_normal_means_nobody_not_fall_cleared():
    """normal은 '쓰러짐 해제'가 아니라 '확정 보행자 없음'이다."""
    serial = FakeSerial()
    extension = build_pipeline([[]])
    combined = CombinedPipeline(
        camera=object(), detector=FakeDetector([[]]), serial_comm=serial,
        extension=extension, fall=StubFall(False), zones=extension.zones,
    )
    assert combined.process_frame(None, timestamp=0.0).state == STATE_NORMAL



# --- track_id 재발급: 카운트와 방향이 함께 이어져야 한다 ---
#
# occupancy가 이름을 바꿔도 CrossingProgress가 모르면 방향이 날아간다. 그러면 거의 다
# 건넌 사람이 "방금 진입"으로 보고돼 아두이노가 연장을 다시 시작한다.

def test_progress_survives_track_id_reissue():
    """★ 이 기능의 본체. 상속이 없으면 진척도가 4,5 대신 2,1로 뒤집힌다."""
    # 구역은 y 방향 5등분(0..200) -> 구역 k는 y in [40*(k-1), 40*k].
    frames = [
        [box_at(50, 20, 1)],    # 구역 1
        [box_at(50, 60, 1)],    # 구역 2
        [box_at(50, 100, 1)],   # 구역 3
        [box_at(50, 140, 7)],   # 구역 4 — 여기서 ID 재발급
        [box_at(50, 180, 7)],   # 구역 5
    ]
    pipeline = build_pipeline(frames)
    pipeline.occupancy.inherit_distance = 60     # 픽셀 (구역 하나가 40px)

    seen = [pipeline.process_frame(None, timestamp=i * 0.1).progress
            for i in range(len(frames))]
    assert seen[-2:] == [4, 5], f"진척도가 뒤집혔다: {seen}"


def test_progress_is_reversed_without_inheritance():
    """상속을 끄면 실제로 뒤집힌다 — 위 테스트가 무엇을 막고 있는지 고정한다."""
    frames = [
        [box_at(50, 20, 1)],
        [box_at(50, 60, 1)],
        [box_at(50, 100, 1)],
        [box_at(50, 140, 7)],
        [box_at(50, 180, 7)],
    ]
    pipeline = build_pipeline(frames)
    pipeline.occupancy.inherit_distance = 0      # 상속 끔

    seen = [pipeline.process_frame(None, timestamp=i * 0.1).progress
            for i in range(len(frames))]
    assert seen[-2:] == [2, 1], f"상속을 껐는데 뒤집히지 않았다: {seen}"


def test_reissue_keeps_sending_zone_without_a_normal_gap():
    """재발급 프레임에도 확정 보행자가 유지되어 normal이 끼어들지 않는다."""
    inside = [box_at(50, 100, 1)]
    frames = [inside, inside, [box_at(50, 105, 7)]]
    pipeline = build_pipeline(frames)
    pipeline.occupancy.inherit_distance = 60

    states = [pipeline.process_frame(None, timestamp=i * 0.1).serial_state()[0]
              for i in range(len(frames))]
    assert states[-1] == STATE_ZONE
    assert STATE_NORMAL not in states[1:]
