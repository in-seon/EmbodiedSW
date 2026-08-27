"""전체 파이프라인: 카메라 -> YOLO 검출 -> 판정 -> 아두이노로 상태 전송.

세 가지 파이프라인이 있다.

    SignalExtensionPipeline  구역 판정 + 속도 -> 연장 요구(zone/normal)
    FallAlarmPipeline        쓰러짐 판정      -> fall/normal
    CombinedPipeline         위 둘을 **추론 1회로** 돌리고 상태를 하나로 합침 (--mode full)

흐름:

    프레임
      -> PersonDetector.detect()            yolov8n-pose, track_id + 키포인트
      ├-> box.foot_point() -> CrosswalkZones.locate() -> CrosswalkOccupancy
      │       -> CrossingProgress -> minimum_progress()  가장 덜 건넌 사람
      ├-> SpeedEstimator.update_many()      평면 좌표 변화율 -> 속도/ETA
      └-> FallDetectionPipeline.update()    키포인트 -> 몸통 각도 -> 쓰러짐 확정
      -> SerialComm.update_state()          normal / zone<진척도> / fall

## '언제 연장할까'는 여기 없다

잔여 녹색 시간·임계값 판단·누적 상한·사이클 리셋은 **제어부(아두이노)가 소유한다.**
파이는 "무엇을 보았는가"만 요약해 보낸다. 이유와 아두이노 쪽 계약은
docs/team_interface.md, 결정 경위는 docs/decisions.md 2026-08-26 항목 참고.

## 추론은 프레임당 한 번뿐이어야 한다

실측상 추론이 프레임 시간의 99.6%다(82.4ms vs 나머지 0.35ms). 두 판정이 각자
detect()를 부르면 FPS가 그대로 반토막 나므로, CombinedPipeline이 한 번 검출한 결과를
process_boxes()로 양쪽에 나눠 준다.
"""

import time
from dataclasses import dataclass, field

from config import config
from src.capture import CameraCapture
from src.detection import MobilityAidDetector, PersonDetector
from src.fall_detection import FallDetectionPipeline, roi_from_ratio, roi_from_zones
from src.motor import MotorGate
from src.serial_comm import SerialComm
from src.serial_comm import CMD_START, CMD_STOP
from src.serial_comm import STATE_FALL, STATE_NORMAL, STATE_ZONE
from src.signal_extend import Occupant, maximum_eta_sec, minimum_progress
from src.speed import SpeedEstimator
from src.zone import CrosswalkOccupancy, CrossingProgress, CrosswalkZones


@dataclass
class PedestrianState:
    """한 프레임 시점에서 본 보행자 한 명의 상태 (화면 표시·로깅용)."""

    track_id: object
    foot_point: tuple
    zone: object = None              # 물리 구역 1..N 또는 None(횡단보도 밖)
    progress: object = None          # 진입 방향으로 보정한 진척도 1..N
    direction: object = None         # +1 / -1 / None(미정)
    confirmed: bool = False          # ZONE_RESIDENCY_FRAMES를 채워 '확정 보행자'가 됐는지
    speed: object = None             # TrackSpeed 또는 None(아직 샘플 부족)
    crossing_time_sec: object = None # 예상 통과 시간(초) 또는 None


@dataclass
class FrameResult:
    """process_frame 한 번의 결과."""

    # 확정 보행자 중 **가장 덜 건넌 사람의 진척도**(1..N). 아무도 없으면 None.
    # 물리 구역 번호가 아니라 진입 방향으로 보정된 값이다 — src/zone.py CrossingProgress 참고.
    progress: object = None
    # 실측 ETA (화면 표시·진단용, **전송하지 않는다**). 연장 판단은 ETA 없이 돌아간다.
    eta_sec: object = None
    occupied_zones: list = field(default_factory=list)
    pedestrians: list = field(default_factory=list)  # PedestrianState 목록
    priority_mode: bool = False
    speed_unit: str = "px/s"         # "cm/s"면 호모그래피 적용됨, "px/s"면 실측 치수 미입력
    # 추적 ID가 붙지 않아 잔류/속도 판정에서 제외한 검출 수. 0이 아닌 값이 계속 나오면
    # 추적이 불안정하다는 뜻이고, 그만큼 연장 조건을 놓치고 있다는 뜻이다.
    untracked_count: int = 0
    # 보조 모델이 이번 프레임에 본 교통약자 보조기구(휠체어/목발 등) 개수.
    # priority_mode가 켜졌는데 이 값이 0이면 배선이 잘못된 것이고, 반대로 이 값만
    # 계속 튀면 보조 모델의 오탐이다 — 둘을 구분할 수 있어야 현장에서 원인을 찾는다.
    mobility_aid_count: int = 0

    def serial_state(self):
        """이 결과를 아두이노로 보낼 (상태, 진척도)로 요약한다.

        확정 보행자가 없으면 NORMAL이다. 쓰러짐은 이 결과에 들어 있지 않으므로 여기서
        판단하지 않는다 — CombinedPipeline이 FALL로 덮어쓴다.
        """
        if self.progress is not None:
            return STATE_ZONE, self.progress
        return STATE_NORMAL, None


class SignalExtensionPipeline:
    def __init__(self, camera=None, detector=None, zones=None, occupancy=None,
                 serial_comm=None, speed_estimator=None, aid_detector=None,
                 progress=None):
        # 인자로 주입하지 않으면 config 기반 기본 객체를 만든다(테스트에선 가짜 객체 주입 가능).
        self.camera = camera or CameraCapture()
        self.detector = detector or PersonDetector()
        # 교통약자(휠체어/목발) 보조 검출. 사람 검출과 **다른 가중치**를 쓴다 — pose 모델에는
        # 해당 클래스가 없어서 PersonDetector로는 원리상 못 잡는다(CLAUDE.md 2.5).
        # config.MOBILITY_AID_MODEL_PATH가 None이면 조용히 비활성되어 항상 빈 목록을 준다.
        self.aid_detector = aid_detector if aid_detector is not None else MobilityAidDetector()
        self.zones = zones or CrosswalkZones.load()
        self.occupancy = occupancy or CrosswalkOccupancy(self.zones)
        # track별 진입 방향을 래치해 물리 구역을 '진척도'로 바꾼다.
        self.progress = progress or CrossingProgress(zone_count=len(self.zones))
        self.serial_comm = serial_comm or SerialComm()
        # 호모그래피는 zone 설정에서 함께 만들어진다. 실측 치수가 없으면 None -> 속도가 px/s로 나온다.
        self.speed = speed_estimator or SpeedEstimator(ground_plane=self.zones.ground_plane)

    def process_frame(self, frame, timestamp=None) -> FrameResult:
        """한 프레임을 처리해 이번 프레임의 연장 요구와 보행자 상태를 반환한다."""
        boxes = self.detector.detect(frame)
        aid_boxes = self.aid_detector.detect(frame)
        return self.process_boxes(boxes, aid_boxes, timestamp)

    def process_boxes(self, boxes, aid_boxes=(), timestamp=None) -> FrameResult:
        """이미 검출된 박스로 처리한다 — **추론을 프레임당 한 번만 돌리기 위한 진입점.**

        --mode full 에서는 쓰러짐 감지와 신호 연장이 같은 프레임을 본다. 각자
        detector.detect()를 부르면 추론이 2배가 되고, 추론이 프레임 시간의 99.6%라
        FPS가 그대로 반토막 난다(포즈 모델을 고른 이유가 무너진다). 그래서 호출자가
        한 번 검출해 결과를 나눠 준다.

        timestamp: 속도 계산용 시각(초). 생략하면 time.monotonic().
        """
        if timestamp is None:
            timestamp = time.monotonic()

        # 사람만 골라 (track_id, 발 위치) 목록을 만든다.
        detections = [
            (box.track_id, box.foot_point())
            for box in boxes
            if box.is_pedestrian()
        ]
        # 교통약자 보조기구 검출 결과. **연장 판단에는 쓰지 않는다.**
        # 기준표 방식에서는 느린 사람이 기준 대비 지연으로 자동 검출되어 더 연장받으므로,
        # "휠체어인가"를 따로 알 필요가 없다. 이 값은 화면 표시·발표 자료용 계측이다.
        priority_mode = bool(aid_boxes)

        confirmed = self.occupancy.update(detections)
        occupied = self.occupancy.occupied_zones()
        speeds = self.speed.update_many(detections, timestamp)

        # 물리 구역 -> 진척도. 확정 보행자만 진척도를 갱신한다(잔류 확정 전에는 방향을
        # 래치하지 않는다 — 스쳐가는 오검출로 방향이 잘못 굳는 것을 막는다).
        progresses = {}
        for track_id, zone in confirmed.items():
            progresses[track_id] = self.progress.update(track_id, zone)
        self.progress.keep_only(confirmed.keys())

        pedestrians = [
            PedestrianState(
                track_id=track_id,
                foot_point=foot_point,
                zone=self.zones.locate(foot_point),
                progress=progresses.get(track_id),
                direction=self.progress.direction(track_id),
                confirmed=track_id in confirmed,
                speed=speeds.get(track_id),
                crossing_time_sec=self.speed.estimated_crossing_time_sec(track_id),
            )
            for track_id, foot_point in detections
        ]

        # 진척도와 (계측용) ETA를 실어 보낸다.
        occupants = [
            Occupant(progress=value,
                     eta_sec=self.speed.estimated_crossing_time_sec(track_id))
            for track_id, value in progresses.items()
        ]

        # 파이는 '가장 덜 건넌 사람'만 알려준다. 얼마나 연장할지는 아두이노가 정한다.
        progress_zone = minimum_progress(occupants)
        eta_sec = maximum_eta_sec(occupants)     # 표시·진단용, 전송하지 않는다

        return FrameResult(
            progress=progress_zone,
            eta_sec=eta_sec,
            occupied_zones=occupied,
            pedestrians=pedestrians,
            priority_mode=priority_mode,
            speed_unit=self.speed.unit,
            untracked_count=self.occupancy.untracked_count,
            mobility_aid_count=len(aid_boxes),
        )

    def run(self, on_result=None):
        """실시간 루프 (신호 연장만). 쓰러짐까지 함께 돌리려면 CombinedPipeline을 쓸 것."""
        with self.camera, self.serial_comm:
            for frame in self.camera.frames():
                result = self.process_frame(frame)
                state, zone = result.serial_state()
                self.serial_comm.update_state(state, zone)
                if on_result is not None and on_result(result, frame) is False:
                    break



@dataclass
class FallAlarmResult:
    """FallAlarmPipeline.process_frame 한 번의 결과."""

    fall_confirmed: bool = False
    confirmed_ids: set = field(default_factory=set)
    people_count: int = 0
    # 이번 프레임에 실제로 아두이노로 보낸 줄("fall"/"normal") 또는 None.
    # 변화 시와 하트비트에만 보내므로 대부분의 프레임에서 None이다(SerialComm.update_state 참고).
    command_sent: object = None


class FallAlarmPipeline:
    """목표 2 전용 파이프라인: 카메라 -> 검출 -> 쓰러짐 판정 -> 부저(아두이노).

    SignalExtensionPipeline과 나눠 둔 이유는 **의존하는 것이 다르기 때문**이다.
    신호 연장은 제어부가 잔여 녹색 시간과 사이클 시작을 알려줘야 판단할 수 있는데
    그 메시지가 아직 팀 합의 전이라 동작하지 못한다. 반면 쓰러짐 알람은 파이 쪽에서
    완결된다 — 카메라와 부저만 있으면 되고, 필요한 조각이 전부 이미 있다.

    둘을 한 클래스에 넣으면 합의되지 않은 쪽 때문에 도는 쪽까지 못 돌게 된다.
    제어부 프로토콜이 확정되면 SignalExtensionPipeline과 나란히 돌리면 된다
    (검출기를 공유해 추론을 한 번만 하는 형태가 될 것이다).

    ROI(어디까지를 '횡단보도 위'로 볼 것인가)는 두 경로가 있다:
      - zones가 있으면 캘리브레이션된 네 꼭짓점을 감싸는 사각형(roi_from_zones) — 권장
      - 없으면 화면 비율(FALL_CONFIG["crosswalk_roi"]) — 눈대중이라 부정확하다
    """

    def __init__(self, camera=None, detector=None, serial_comm=None,
                 zones=None, roi_px=None, fall_detector=None):
        self.camera = camera or CameraCapture()
        self.detector = detector or PersonDetector()
        self.serial_comm = serial_comm or SerialComm()
        self.zones = zones
        self.roi_px = roi_px
        # ROI를 화면 비율로 잡는 경로는 프레임 크기를 알아야 하므로 첫 프레임까지 미룬다.
        self._fall = fall_detector

    def _fall_pipeline(self, frame):
        if self._fall is None:
            roi = self.roi_px
            if roi is None:
                roi = (
                    roi_from_zones(self.zones) if self.zones is not None
                    else roi_from_ratio(frame.shape)
                )
                self.roi_px = roi
            self._fall = FallDetectionPipeline(roi)
        return self._fall

    def process_frame(self, frame, now=None) -> FallAlarmResult:
        """한 프레임을 처리하고, 필요하면 부저 명령을 보낸다.

        now: 판단 기준 시각(초). 생략하면 time.monotonic().
             쓰러짐 판정과 알람 하트비트가 **같은 시계**를 쓰도록 한 값을 둘 다에 넘긴다.
             원문 PoC는 time.time()을 썼지만 둘 다 '차이'만 쓰므로 동작은 같고,
             monotonic은 시스템 시각이 조정돼도 뒤로 가지 않는다는 장점이 있다.
        """
        now = time.monotonic() if now is None else now
        boxes = self.detector.detect(frame)
        return self.process_boxes(boxes, frame, now)

    def process_boxes(self, boxes, frame, now=None) -> FallAlarmResult:
        """이미 검출된 박스로 처리한다 (추론 1회 공유 — SignalExtensionPipeline과 같은 이유).

        frame은 ROI를 화면 비율로 잡을 때 크기를 알아야 해서 받는다(첫 프레임에만 쓰인다).
        """
        now = time.monotonic() if now is None else now

        fall = self._fall_pipeline(frame).update(boxes, now)
        state = STATE_FALL if fall["fall_confirmed"] else STATE_NORMAL
        command = self.serial_comm.update_state(state, now=now)

        return FallAlarmResult(
            fall_confirmed=fall["fall_confirmed"],
            confirmed_ids=fall["confirmed_ids"],
            people_count=len(fall["people"]),
            command_sent=command,
        )

    def reset_alarm(self):
        """오탐으로 부저가 계속 울릴 때의 탈출구 (원문 PoC의 'r' 키와 같다).

        쓰러짐 누적을 지우고 부저도 즉시 끈다. 누적만 지우면 다음 프레임에 다시
        확정될 수 있고, 부저만 끄면 파이 쪽 상태와 어긋난다 — 둘 다 해야 한다.
        """
        if self._fall is not None:
            self._fall.reset()
        self.serial_comm.send_state(STATE_NORMAL)

    def run(self, on_result=None):
        """실시간 루프. on_result(result, frame)이 주어지면 프레임마다 호출한다.

        표시·로깅을 콜백으로 뺀 이유: 헤드리스(파이 SSH)와 창 모드가 필요한데,
        그 차이를 파이프라인이 알 필요가 없다. main.py가 콜백으로 결정한다.
        """
        with self.camera, self.serial_comm:
            for frame in self.camera.frames():
                result = self.process_frame(frame)
                if on_result is not None and on_result(result, frame) is False:
                    break


@dataclass
class CombinedResult:
    """CombinedPipeline.process_frame 한 번의 결과."""

    fall: FallAlarmResult = field(default_factory=FallAlarmResult)
    extension: FrameResult = field(default_factory=FrameResult)
    state: str = STATE_NORMAL          # 실제로 아두이노에 반영된 상태
    zone: object = None                # ZONE 상태일 때의 진척도
    line_sent: object = None           # 이번 프레임에 보낸 줄, 안 보냈으면 None
    motor_running: bool = False        # 모형 보행자 모터가 지금 **움직이고** 있는가
    motor_paused: bool = False         # 기동돼 있으나 쓰러짐 연출로 잠깐 세워 둔 상태
    # 이번 프레임에 모터가 기동/정지했으면 그 사유 한 줄, 아니면 None.
    # 매 프레임 상태를 찍으면 로그가 묻히므로 '바뀐 순간'만 남긴다.
    motor_event: object = None


class CombinedPipeline:
    """목표 1 + 2를 한 루프에서 돌린다 (--mode full).

    ## 추론은 프레임당 한 번뿐이다

    이것이 이 클래스의 존재 이유다. 두 파이프라인을 각각 run() 하면 같은 프레임에
    YOLO가 두 번 돌고, 추론이 프레임 시간의 99.6%라 FPS가 그대로 반토막 난다.
    여기서 한 번 검출해 두 쪽에 나눠 준다(process_boxes).

    ## 상태 우선순위: fall > zone > normal

    시리얼 채널이 하나이고 상태도 하나다. 쓰러진 사람이 있으면 연장 요구보다 그쪽이
    급하므로 fall이 이긴다. 쓰러짐이 풀리면 그 프레임의 연장 요구가 다시 나간다.

    쓰러짐 중에 연장을 못 보내는 것이 손해처럼 보이지만, 쓰러진 사람은 애초에
    연장으로 해결되는 상황이 아니다(스스로 못 건넌다). 제어부가 fall을 받으면
    사이렌과 함께 차량 신호를 어떻게 할지 결정한다 — docs/team_interface.md 참고.

    ## ⚠️ 쓰러짐 해제는 normal이 아니라 '상태가 FALL에서 벗어남'으로 표현된다

    일어난 사람이 횡단보도에 그대로 있으면 여전히 확정 보행자이므로 곧바로 ZONE이 나간다:

        zone1 -> zone3 -> fall -> zone4        <- normal이 끼지 않는다

    그래서 아두이노는 **`부저 = (마지막 상태 == fall)`** 로 짜야 한다. "normal을 받아야
    끈다"로 짜면 이 구간에서 부저가 영영 울린다. normal은 "쓰러짐 해제"가 아니라
    **"확정 보행자 없음"**이다.

    ## 모형 보행자 모터 (선택)

    모터는 파이 GPIO에 붙어 있고, 기동 시점만 아두이노가 안다:

        아두이노 START -> 기동      파이가 '건너갔다'를 판단 -> 정지

    정지를 파이가 내는 이유와 '없음'이 아니라 '있었다가 없어짐'이어야 하는 이유는
    src/motor.py의 MotorGate 참고. motor=None이면 명령만 읽고 구동하지 않는다.

    쓰러짐 연출(서보) 중에는 pause_motor()/resume_motor()로 잠깐 세운다 — 넘어진 모형이
    계속 끌려가면 안 되지만, 일어난 뒤에는 마저 건너야 하므로 정지가 아니라 일시정지다.
    """

    def __init__(self, camera=None, detector=None, serial_comm=None,
                 extension=None, fall=None, zones=None,
                 motor=None, motor_gate=None):
        self.camera = camera or CameraCapture()
        self.detector = detector or PersonDetector()
        self.serial_comm = serial_comm or SerialComm()
        zones = zones if zones is not None else CrosswalkZones.load()

        # 모터는 **선택**이다. None이면 START/STOP을 읽되 아무것도 구동하지 않는다.
        # 모형 없이 판정만 확인하는 것이 흔한 사용법이라 기본을 '없음'으로 뒀다.
        self.motor = motor
        self.motor_gate = motor_gate if motor_gate is not None else MotorGate()

        # 두 파이프라인에 **같은 시리얼 객체**를 주되, 상태 전송은 여기서 한 번만 한다.
        # 각자 보내면 같은 프레임에 NORMAL과 EXTEND가 번갈아 나가 서로를 덮어쓴다.
        self.extension = extension or SignalExtensionPipeline(
            camera=self.camera, detector=self.detector,
            serial_comm=_NoSend(), zones=zones,
        )
        self.fall = fall or FallAlarmPipeline(
            camera=self.camera, detector=self.detector,
            serial_comm=_NoSend(), zones=zones,
        )

    def process_frame(self, frame, timestamp=None) -> CombinedResult:
        now = time.monotonic() if timestamp is None else timestamp

        # ---- 추론 1회 ----
        boxes = self.detector.detect(frame)
        aid_boxes = self.extension.aid_detector.detect(frame)

        fall_result = self.fall.process_boxes(boxes, frame, now)
        ext_result = self.extension.process_boxes(boxes, aid_boxes, now)

        if fall_result.fall_confirmed:
            state, zone = STATE_FALL, None
        else:
            state, zone = ext_result.serial_state()

        line = self.serial_comm.update_state(state, zone, now=now)

        # 모터는 상태 전송 **뒤에** 다룬다. 아두이노로 보내는 것이 이 루프의 본업이고,
        # 모터는 시연용 부수 장치다. 앞에 두면 GPIO 지연이 상태 전송을 밀어낸다.
        motor_event = self._drive_motor(ext_result, now)

        return CombinedResult(
            fall=fall_result, extension=ext_result,
            state=state, zone=zone, line_sent=line,
            motor_running=self.motor_gate.moving,
            motor_paused=self.motor_gate.paused,
            motor_event=motor_event,
        )

    def _drive_motor(self, ext_result, now):
        """아두이노 명령을 반영하고, 정지 조건을 판단해 모터를 구동한다.

        ## 기동은 아두이노가, 정지는 파이가

        "보행 녹색이 시작됐다"는 신호를 소유한 아두이노만 알고, "모형이 횡단보도를
        빠져나갔다"는 영상을 보는 파이만 안다. 그래서 START는 받고 정지는 스스로 낸다.

        ## '확정 보행자'로 판단하는 이유

        occupied 기준을 progress로 잡았다 — 즉 **잔류 확정된 사람이 구역 안에 있는가**다.
        원시 검출로 판단하면 스쳐 지나가는 오검출 한 프레임에 모터가 멈춘다.
        아두이노로 보내는 ZONE의 기준과 같은 값이라 로그를 나란히 읽을 수 있다는
        장점도 있다(ZONE이 사라진 프레임 = 모터가 멈추는 프레임).
        """
        event = None
        for name, mode in self.serial_comm.take_commands():
            if name == CMD_START:
                if self.motor_gate.start(mode, now=now):
                    if self.motor is not None:
                        self.motor.start(self.motor_gate.mode)
                    event = f"모터 기동 (모드 {self.motor_gate.mode})"
            elif name == CMD_STOP and self.motor_gate.stop():
                if self.motor is not None:
                    self.motor.stop()
                event = f"모터 정지 ({self.motor_gate.last_stop_reason})"

        if self.motor_gate.update(ext_result.progress is not None, now=now):
            if self.motor is not None:
                self.motor.stop()
            event = f"모터 정지 ({self.motor_gate.last_stop_reason})"
        return event

    def pause_motor(self, now=None) -> bool:
        """모형 구동을 잠깐 세운다 (쓰러짐 연출용). 정지가 아니라 **일시정지**다.

        모형에는 모터가 둘이다 — 끌고 가는 스텝모터와 발을 넘어뜨리는 서보.
        넘어진 모형이 계속 끌려가면 안 되므로, 서보가 눕히는 순간 이쪽을 세운다.
        stop()이 아닌 이유는 MotorGate.pause() 참고(같은 횡단을 이어서 가야 한다).
        """
        if not self.motor_gate.pause(now=now):
            return False
        if self.motor is not None:
            self.motor.stop()
        return True

    def resume_motor(self, now=None) -> bool:
        """세워 뒀던 모형을 멈춘 지점부터 이어서 움직인다."""
        if not self.motor_gate.resume(now=now):
            return False
        if self.motor is not None:
            self.motor.start(self.motor_gate.mode)
        return True

    def reset_alarm(self):
        """오탐으로 사이렌에 갇혔을 때의 탈출구."""
        self.fall.reset_alarm()

    def run(self, on_result=None):
        try:
            with self.camera, self.serial_comm:
                for frame in self.camera.frames():
                    result = self.process_frame(frame)
                    if on_result is not None and on_result(result, frame) is False:
                        break
        finally:
            # 어떤 경로로 끝나도 모터는 반드시 멈춘다. 예외로 빠져나갈 때 스텝 스레드가
            # 그대로 살아 있으면, 파이썬이 죽은 뒤에도 코일에 전류가 남아 모터가 탄다.
            if self.motor is not None:
                self.motor.close()
            self.motor_gate.stop()


class _NoSend:
    """CombinedPipeline이 하위 파이프라인에 끼워 넣는 '전송하지 않는' 시리얼.

    두 파이프라인은 각자 상태를 보내도록 만들어져 있지만, 합쳐 돌릴 때는 상태가
    하나여야 한다. 하위 쪽 전송을 막아 두고 CombinedPipeline이 우선순위를 정해
    한 번만 보낸다.
    """

    def update_state(self, state, extend_sec=None, now=None):
        return None

    def send_state(self, state, extend_sec=None, now=None):
        return None

    def take_commands(self):
        return []
