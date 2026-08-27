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
    """보낸 줄을 기록한다. 실제 SerialComm의 변화 감지/하트비트까지 흉내내지는 않는다.

    inbox에 넣어 둔 (명령, 인자)는 아두이노가 보낸 것으로 친다 — take_commands가
    실제 SerialComm처럼 **꺼내면서 비운다.**
    """

    def __init__(self, inbox=None):
        self.states = []
        self.inbox = list(inbox or [])

    def update_state(self, state, zone=None, now=None):
        self.states.append((state, zone))
        return state

    def send_state(self, state, zone=None, now=None):
        self.states.append((state, zone))
        return state

    def take_commands(self):
        commands, self.inbox = self.inbox, []
        return commands


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
                   aid_frames=None):
    zones = CrosswalkZones.from_quad(CORNERS, n=5, width_cm=width_cm, length_cm=length_cm)
    plane = GroundPlane.from_quad(CORNERS, width_cm, length_cm)
    return SignalExtensionPipeline(
        camera=object(),
        detector=FakeDetector(frames),
        aid_detector=FakeAidDetector(aid_frames),
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


# --- 교통약자 검출은 계측용으로만 남는다 ---

def test_mobility_aid_is_reported_but_does_not_change_output():
    """느린 사람은 아두이노가 기준 대비 지연으로 잡는다 — 검출 결과로 값을 바꾸지 않는다."""
    pipeline = build_pipeline([[box_at(50, 100, 1)]], aid_frames=[[aid_box()]])
    result = pipeline.process_frame(None, timestamp=0.0)

    assert result.priority_mode is True
    assert result.mobility_aid_count == 1
    assert result.progress == 3                 # 값은 그대로


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


# --- CombinedPipeline: 모형 보행자 모터 ---
#
# 기동은 아두이노(START), 정지는 파이(모형이 다 건넜다). 판단 자체의 단위 테스트는
# tests/test_motor.py에 있고, 여기서는 **파이프라인에 제대로 배선됐는지**만 본다.

class RecordingMotor:
    """StepperMotor 자리에 들어가 호출만 기록한다 (GPIO 없이 배선 확인)."""

    def __init__(self):
        self.calls = []
        self.running = False
        self.mode = None

    def start(self, mode=None):
        self.calls.append(("start", mode))
        self.running = True
        self.mode = mode

    def stop(self):
        self.calls.append(("stop", None))
        self.running = False
        self.mode = None

    def close(self):
        self.calls.append(("close", None))
        self.running = False


def build_combined_with_motor(boxes_per_frame, inbox=None):
    serial = FakeSerial(inbox=inbox)
    extension = build_pipeline(boxes_per_frame)
    motor = RecordingMotor()
    combined = CombinedPipeline(
        camera=object(),
        detector=FakeDetector(boxes_per_frame),
        serial_comm=serial,
        extension=extension,
        fall=StubFall(False),
        zones=extension.zones,
        motor=motor,
    )
    return combined, serial, motor


def test_motor_starts_on_arduino_start():
    """아두이노가 START를 보내면 모터가 돈다."""
    combined, serial, motor = build_combined_with_motor([[]], inbox=[("START", None)])
    result = combined.process_frame(None, timestamp=0.0)
    assert motor.calls == [("start", config.MOTOR_DEFAULT_MODE)]
    assert result.motor_running is True


def test_motor_start_passes_speed_mode():
    combined, serial, motor = build_combined_with_motor([[]], inbox=[("START", 3)])
    combined.process_frame(None, timestamp=0.0)
    assert motor.calls == [("start", 3)]


def test_motor_does_not_run_without_start():
    """START 없이 보행자만 보인다고 모터가 돌면 안 된다 — 신호와 무관하게 모형이 움직인다."""
    boxes = [box_at(50, 100, 1)]
    combined, serial, motor = build_combined_with_motor([boxes])
    combined.process_frame(None, timestamp=0.0)
    assert motor.calls == []


def test_motor_survives_the_frames_before_the_model_enters():
    """★ START 직후에는 아직 아무도 횡단보도 위에 없다 — 여기서 멈추면 출발조차 못 한다."""
    combined, serial, motor = build_combined_with_motor(
        [[], [], []], inbox=[("START", None)])
    for t in (0.0, 0.1, 0.2):
        result = combined.process_frame(None, timestamp=t)
        assert result.motor_running is True, f"{t}초에 멈췄다"
    assert ("stop", None) not in motor.calls


def test_motor_stops_when_model_finishes_crossing():
    """진입 -> 횡단 -> 반대편으로 빠져나감. 나간 프레임에 멈춘다.

    '사라짐'이 아니라 '구역 밖으로 나감'으로 재현한다 — 실제로 모형은 프레임에서
    사라지는 게 아니라 횡단보도 밖에 계속 보인다. 그리고 그 둘은 처리가 다르다:
    구역 이탈은 즉시 리셋이지만 미검출은 유예를 받는다(CrosswalkOccupancy).
    """
    inside = [box_at(50, 100, 1)]
    outside = [box_at(50, 250, 1)]          # y=250 -> 횡단보도(0..200) 밖
    frames = [[], inside, inside, inside, outside]
    combined, serial, motor = build_combined_with_motor(
        frames, inbox=[("START", None)])

    for index, t in enumerate((0.0, 0.1, 0.2, 0.3)):
        result = combined.process_frame(None, timestamp=t)
        assert result.motor_running is True, f"프레임 {index}에서 멈췄다"

    result = combined.process_frame(None, timestamp=0.4)
    assert result.motor_running is False
    assert ("stop", None) in motor.calls
    assert "횡단" in result.motor_event


def test_one_dropped_frame_does_not_stop_the_motor():
    """★ YOLO가 한 프레임 놓쳤다고 모터가 멈추면, 시연 중간에 모형이 멈춰 선다.

    모터 정지 기준은 아두이노로 보내는 ZONE과 **같은 값**(확정 보행자)이라
    잔류 유예(TRACK_GRACE_FRAMES)를 그대로 물려받는다. 이 테스트는 그 연결을 고정한다.
    """
    inside = [box_at(50, 100, 1)]
    frames = [inside, inside, [], inside]   # 세 번째 프레임만 놓침
    combined, serial, motor = build_combined_with_motor(
        frames, inbox=[("START", None)])

    for index, t in enumerate((0.0, 0.1, 0.2, 0.3)):
        result = combined.process_frame(None, timestamp=t)
        assert result.motor_running is True, f"프레임 {index}에서 멈췄다"
    assert ("stop", None) not in motor.calls


def test_motor_stops_on_arduino_stop_command():
    combined, serial, motor = build_combined_with_motor([[], []], inbox=[("START", None)])
    combined.process_frame(None, timestamp=0.0)

    serial.inbox = [("STOP", None)]
    result = combined.process_frame(None, timestamp=0.1)
    assert result.motor_running is False
    assert ("stop", None) in motor.calls


def test_motor_event_only_on_change():
    """매 프레임 사유를 찍으면 로그가 묻힌다 — 바뀐 프레임에만 값이 실린다."""
    combined, serial, motor = build_combined_with_motor(
        [[], [], []], inbox=[("START", None)])
    assert combined.process_frame(None, timestamp=0.0).motor_event is not None
    assert combined.process_frame(None, timestamp=0.1).motor_event is None


def test_motor_is_optional():
    """모형 없이 판정만 확인하는 것이 흔한 사용법이다 — motor=None이어도 죽지 않는다."""
    serial = FakeSerial(inbox=[("START", None)])
    extension = build_pipeline([[]])
    combined = CombinedPipeline(
        camera=object(), detector=FakeDetector([[]]), serial_comm=serial,
        extension=extension, fall=StubFall(False), zones=extension.zones,
    )
    result = combined.process_frame(None, timestamp=0.0)
    assert result.motor_running is True      # 판단은 그대로 돈다


# --- 쓰러짐 연출 중 구동 모터 일시정지 ---

def test_pause_motor_stops_the_hardware_but_keeps_the_run():
    inside = [box_at(50, 100, 1)]
    combined, serial, motor = build_combined_with_motor(
        [inside] * 6, inbox=[("START", None)])
    combined.process_frame(None, timestamp=0.0)
    motor.calls.clear()

    assert combined.pause_motor(now=1.0) is True
    assert motor.calls == [("stop", None)]          # 실제로 코일 전원이 끊긴다

    result = combined.process_frame(None, timestamp=1.5)
    assert result.motor_paused is True
    assert result.motor_running is False            # '움직이는 중'은 아니다


def test_resume_motor_restarts_at_the_same_speed_mode():
    """재개할 때 속도가 기본값으로 되돌아가면 교통약자 시연이 중간에 빨라진다."""
    inside = [box_at(50, 100, 1)]
    combined, serial, motor = build_combined_with_motor(
        [inside] * 6, inbox=[("START", 3)])
    combined.process_frame(None, timestamp=0.0)
    combined.pause_motor(now=1.0)
    motor.calls.clear()

    assert combined.resume_motor(now=5.0) is True
    assert motor.calls == [("start", 3)]


def test_paused_motor_survives_losing_the_person_while_down():
    """★ 누워 있는 동안 검출이 빠져도 모터가 영구 정지되면 안 된다.

    영구 정지되면 서보가 일으켜 세워도 모형이 다시 안 움직여, 데모가 거기서 끝난다.
    """
    inside = [box_at(50, 100, 1)]
    frames = [inside, [], [], [], inside, inside]
    combined, serial, motor = build_combined_with_motor(
        frames, inbox=[("START", None)])
    combined.process_frame(None, timestamp=0.0)
    combined.pause_motor(now=0.1)

    for t in (0.2, 0.3, 0.4):
        result = combined.process_frame(None, timestamp=t)
        assert result.motor_paused is True

    assert combined.resume_motor(now=0.5) is True
    result = combined.process_frame(None, timestamp=0.6)
    assert result.motor_running is True
