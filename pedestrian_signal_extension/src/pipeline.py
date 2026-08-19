"""전체 파이프라인: 카메라 -> YOLO 검출 -> 구역 판정/속도 추정 -> 신호 연장 결정 -> 제어부 전송.

흐름 (CLAUDE.md 4장):

    프레임
      -> PersonDetector.detect()              yolov8n, track_id 부여
      -> box.foot_point()                     bbox 하단 모서리 중심 = 지면 접점
      ├-> CrosswalkZones.locate()             몇 번 구역(1~5)인지
      │   -> CrosswalkOccupancy               확정 보행자 / 점유 구역
      └-> SpeedEstimator.update_many()        평면 좌표 변화율로 속도/방향/예상 통과 시간
      -> SignalExtensionStateMachine.evaluate()  점유 구역으로 연장 시간 결정
      -> SerialComm.send_extend_signal()      제어부로 전송

속도는 지금 계산·노출만 하고 연장 결정에는 쓰지 않는다(config.USE_SPEED_FOR_EXTENSION = False,
CLAUDE.md 2.3). 실측으로 안정성을 검증한 뒤 반영 여부를 정한다.

주의: 잔여 녹색 시간(remaining_time_sec)의 소유자는 제어부다. 통신 프로토콜이 확정되면
SerialComm.read_remaining_time()으로 읽어 넣는다. 프로토콜이 미정인 지금은 SerialComm 생성이
NotImplementedError로 멈춰, 임의 값으로 동작하는 대신 "무엇을 먼저 확정해야 하는지" 알려준다.
"""

import time
from dataclasses import dataclass, field

from config import config
from src.capture import CameraCapture
from src.detection import PersonDetector
from src.serial_comm import SerialComm
from src.signal_extend import SignalExtensionStateMachine
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


class SignalExtensionPipeline:
    def __init__(self, camera=None, detector=None, zones=None, occupancy=None,
                 state_machine=None, serial_comm=None, speed_estimator=None):
        # 인자로 주입하지 않으면 config 기반 기본 객체를 만든다(테스트에선 가짜 객체 주입 가능).
        self.camera = camera or CameraCapture()
        self.detector = detector or PersonDetector()
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
        # 교통약자가 하나라도 검출되면 우선 연장 모드.
        # 현재는 MOBILITY_AID_LABELS가 비어 있어(보류) 항상 False다 — CLAUDE.md 2.5.
        priority_mode = any(box.is_mobility_aid() for box in boxes)

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

        extension_sec = self.state_machine.evaluate(
            remaining_time_sec=remaining_time_sec,
            occupied_zones=occupied,
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
        )

    def run(self):
        """실시간 루프. 제어부 통신 프로토콜 확정 후 사용."""
        if config.USE_SPEED_FOR_EXTENSION:
            raise NotImplementedError(
                "USE_SPEED_FOR_EXTENSION이 켜져 있지만 속도 기반 연장 규칙은 아직 구현되지 않았습니다. "
                "속도 추정 정확도를 실측 검증하고 팀과 규칙을 확정한 뒤 구현하세요 (CLAUDE.md 2.3, 6)."
            )
        with self.camera, self.serial_comm:
            for frame in self.camera.frames():
                # 사이클 경계와 잔여 시간 모두 제어부가 소유하는 값이다.
                if self.serial_comm.read_cycle_started():
                    self.begin_new_cycle()
                remaining = self.serial_comm.read_remaining_time()
                self.process_frame(frame, remaining)
