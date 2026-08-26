"""전체 파이프라인: 카메라 -> YOLO 검출 -> 구역 판정/속도 추정 -> 신호 연장 결정 -> 제어부 전송.

흐름 (CLAUDE.md 4장):

    프레임
      -> PersonDetector.detect()              yolov8n-pose, track_id 부여
      │  -> box.foot_point()                  bbox 하단 모서리 중심 = 지면 접점
      │  ├-> CrosswalkZones.locate()          몇 번 구역(1~5)인지
      │  │   -> CrosswalkOccupancy            확정 보행자 / 점유 구역
      │  └-> SpeedEstimator.update_many()     평면 좌표 변화율로 속도/방향/예상 통과 시간
      -> MobilityAidDetector.detect()         휠체어/목발 여부 -> priority_mode (저빈도 추론)
      -> SignalExtensionStateMachine.evaluate()  점유 구역 + priority로 연장 시간 결정
      -> SerialComm.send_extend_signal()      제어부로 전송

검출기가 둘인 이유는 가중치가 다르기 때문이다. 사람 검출용 pose 모델에는 휠체어·목발
클래스가 아예 없어서 같은 추론으로는 못 잡는다(CLAUDE.md 2.5). 대신 보조 모델은
'있냐 없냐'만 보면 되므로 매 프레임이 아니라 config.MOBILITY_AID_EVERY_N_FRAMES 마다
한 번만 돌린다 — src/detection.py의 MobilityAidDetector 참고.

속도 반영은 config.USE_SPEED_FOR_EXTENSION으로 켜고 끈다(기본 False, CLAUDE.md 2.3).
켜면 구역별 연장 시간이 '상한'이 되고, 실제 연장은 그 사람이 정말 모자란 만큼으로 줄어든다.
끄면 지금까지처럼 구역 규칙만 쓴다 — 켜고 끄는 것 외에 다른 동작 차이는 없다.

주의: 잔여 녹색 시간(remaining_time_sec)의 소유자는 제어부다. 통신 프로토콜이 확정되면
SerialComm.read_remaining_time()으로 읽어 넣는다. 프로토콜이 미정인 지금은 SerialComm 생성이
NotImplementedError로 멈춰, 임의 값으로 동작하는 대신 "무엇을 먼저 확정해야 하는지" 알려준다.
"""

import time
from dataclasses import dataclass, field

from config import config
from src.capture import CameraCapture
from src.detection import MobilityAidDetector, PersonDetector
from src.fall_detection import FallDetectionPipeline, roi_from_ratio, roi_from_zones
from src.serial_comm import SerialComm
from src.signal_extend import Occupant, SignalExtensionStateMachine
from src.speed import SpeedEstimator
from src.zone import CrosswalkOccupancy, CrosswalkZones


@dataclass
class PedestrianState:
    """한 프레임 시점에서 본 보행자 한 명의 상태 (화면 표시·로깅용)."""

    track_id: object
    foot_point: tuple
    zone: object = None              # 1..N 또는 None(횡단보도 밖)
    confirmed: bool = False          # ZONE_RESIDENCY_FRAMES를 채워 '확정 보행자'가 됐는지
    speed: object = None             # TrackSpeed 또는 None(아직 샘플 부족)
    crossing_time_sec: object = None # 예상 통과 시간(초) 또는 None


@dataclass
class FrameResult:
    """process_frame 한 번의 결과."""

    extension_sec: int = 0
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


class SignalExtensionPipeline:
    def __init__(self, camera=None, detector=None, zones=None, occupancy=None,
                 state_machine=None, serial_comm=None, speed_estimator=None,
                 aid_detector=None):
        # 인자로 주입하지 않으면 config 기반 기본 객체를 만든다(테스트에선 가짜 객체 주입 가능).
        self.camera = camera or CameraCapture()
        self.detector = detector or PersonDetector()
        # 교통약자(휠체어/목발) 보조 검출. 사람 검출과 **다른 가중치**를 쓴다 — pose 모델에는
        # 해당 클래스가 없어서 PersonDetector로는 원리상 못 잡는다(CLAUDE.md 2.5).
        # config.MOBILITY_AID_MODEL_PATH가 None이면 조용히 비활성되어 항상 빈 목록을 준다.
        self.aid_detector = aid_detector if aid_detector is not None else MobilityAidDetector()
        self.zones = zones or CrosswalkZones.load()
        self.occupancy = occupancy or CrosswalkOccupancy(self.zones)
        self.state_machine = state_machine or SignalExtensionStateMachine()
        self.serial_comm = serial_comm or SerialComm()
        # 호모그래피는 zone 설정에서 함께 만들어진다. 실측 치수가 없으면 None -> 속도가 px/s로 나온다.
        self.speed = speed_estimator or SpeedEstimator(ground_plane=self.zones.ground_plane)

    def begin_new_cycle(self):
        """새 보행 신호 사이클이 시작될 때 호출 — 사이클 단위 상태를 전부 초기화한다.

        초기화 대상:
          - 누적 연장(SignalExtensionStateMachine) — 안 지우면 한 번 상한을 찍은 뒤로
            영구히 CAPPED가 되어 그 다음 사이클부터 연장이 아예 안 된다.
          - 잔류 카운트(CrosswalkOccupancy) — 사이클이 바뀌면 보행자도 바뀐다. 이전
            카운트가 남아 있으면 새 사이클 첫 프레임에서 곧바로 '확정'되어 잔류 검증이 무의미해진다.
          - 속도 히스토리(SpeedEstimator) — 사이클 경계를 가로지르는 변위는 속도가 아니다.

        호출 주체는 제어부다. 잔여 녹색 시간과 마찬가지로 신호 사이클의 소유자가 제어부이므로,
        "새 녹색 시작" 이벤트가 시리얼 프로토콜에 있어야 한다(docs/team_interface.md 참고).
        추적 ID(track_id)는 일부러 건드리지 않는다 — 위 세 가지를 지우면 ID가 재사용돼도
        카운트와 히스토리가 처음부터 다시 쌓이므로 문제가 없고, 추적기 리셋은 비용만 든다.
        """
        self.state_machine.reset()
        self.occupancy.clear()
        self.speed.clear()

    def process_frame(self, frame, remaining_time_sec, timestamp=None) -> FrameResult:
        """한 프레임을 처리해 이번 사이클 연장 시간과 보행자 상태를 반환한다.

        remaining_time_sec: 제어부가 알려준 현재 잔여 녹색 시간(초).
        timestamp: 속도 계산용 시각(초). 생략하면 time.monotonic()을 쓴다.
                   (테스트나 녹화 영상 재생 시 프레임 시각을 직접 넣을 수 있게 인자로 뺐다.)
        """
        if timestamp is None:
            timestamp = time.monotonic()

        boxes = self.detector.detect(frame)

        # 사람만 골라 (track_id, 발 위치) 목록을 만든다.
        detections = [
            (box.track_id, box.foot_point())
            for box in boxes
            if box.is_pedestrian()
        ]
        # 교통약자 보조기구가 하나라도 보이면 우선 연장 모드.
        #
        # 판단 근거를 **보조 모델 쪽에서만** 가져오는 것이 중요하다. 예전에는 사람 검출
        # 결과(boxes)에서 is_mobility_aid()를 봤는데, 사람 검출 가중치(yolov8n-pose)에는
        # 휠체어 클래스 자체가 없어서 그 조건은 원리상 절대 참이 되지 않았다. 즉
        # MOBILITY_AID_MODEL_PATH에 가중치를 채워 넣어도 파이프라인은 그걸 쓰지 않았다.
        #
        # 지금은 '사람이 어디 있나'(PersonDetector)와 '보조기구가 있나'(MobilityAidDetector)로
        # 역할이 갈린다. 보조 모델은 매 프레임 돌지 않고(MOBILITY_AID_EVERY_N_FRAMES)
        # 그 사이엔 직전 결과를 재사용하므로, 추론 비용은 크게 늘지 않는다.
        aid_boxes = self.aid_detector.detect(frame)
        priority_mode = bool(aid_boxes)

        confirmed = self.occupancy.update(detections)
        occupied = self.occupancy.occupied_zones()
        speeds = self.speed.update_many(detections, timestamp)

        pedestrians = [
            PedestrianState(
                track_id=track_id,
                foot_point=foot_point,
                zone=self.zones.locate(foot_point),
                confirmed=track_id in confirmed,
                speed=speeds.get(track_id),
                crossing_time_sec=self.speed.estimated_crossing_time_sec(track_id),
            )
            for track_id, foot_point in detections
        ]

        # 확정 보행자마다 (구역, 예상 통과 시간)을 실어 보낸다. 속도 반영이 꺼져 있거나
        # ETA가 None이면 상태 머신이 구역 규칙만으로 판단한다(안전한 폴백).
        occupants = [
            Occupant(zone=zone, eta_sec=self.speed.estimated_crossing_time_sec(track_id))
            for track_id, zone in confirmed.items()
        ]

        extension_sec = self.state_machine.evaluate(
            remaining_time_sec=remaining_time_sec,
            occupants=occupants,
            priority_mode=priority_mode,
        )
        if extension_sec > 0:
            self.serial_comm.send_extend_signal(extension_sec, priority=priority_mode)

        return FrameResult(
            extension_sec=extension_sec,
            occupied_zones=occupied,
            pedestrians=pedestrians,
            priority_mode=priority_mode,
            speed_unit=self.speed.unit,
            untracked_count=self.occupancy.untracked_count,
            mobility_aid_count=len(aid_boxes),
        )

    def run(self):
        """실시간 루프. 제어부 통신 프로토콜 확정 후 사용."""
        with self.camera, self.serial_comm:
            for frame in self.camera.frames():
                # 사이클 경계와 잔여 시간 모두 제어부가 소유하는 값이다.
                if self.serial_comm.read_cycle_started():
                    self.begin_new_cycle()
                remaining = self.serial_comm.read_remaining_time()
                self.process_frame(frame, remaining)


@dataclass
class FallAlarmResult:
    """FallAlarmPipeline.process_frame 한 번의 결과."""

    fall_confirmed: bool = False
    confirmed_ids: set = field(default_factory=set)
    people_count: int = 0
    # 이번 프레임에 실제로 아두이노로 보낸 명령("ALERT"/"STOP") 또는 None.
    # 매 프레임 보내지 않으므로 대부분의 프레임에서 None이다(SerialComm.update_alarm 참고).
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
        fall = self._fall_pipeline(frame).update(boxes, now)
        command = self.serial_comm.update_alarm(fall["fall_confirmed"], now=now)

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
        self.serial_comm.update_alarm(False)

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
