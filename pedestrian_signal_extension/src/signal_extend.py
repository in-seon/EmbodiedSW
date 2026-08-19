"""신호 연장 상태 머신 (위치 기반 차등 연장).

CLAUDE.md 2.3, 2.4 + 담당 파트 설계:
- 조건: (남은 시간 <= 임계값) AND (횡단보도 안에 확정 보행자 존재).
- 연장 시간은 보행자가 있는 "구역"에 따라 차등 (config.ZONE_EXTENSION_SEC).
  여러 명이면 가장 많이 필요한 사람(=연장 시간이 가장 큰 구역)에 맞춘다.
- 매 사이클 재평가: 보행자가 사라지거나 남은 시간이 충분해지면 연장하지 않는다.
- 누적 상한(config.MAX_TOTAL_EXTENSION_SEC) 존재 — 차량 신호 충돌 방지.
- 교통약자(휠체어/목발 등) 감지 시 우선 연장(priority_mode).

임계값/상한이 아직 팀 확정 전이면 생성 시점에 명시적으로 실패해 임의 값으로 동작하지 않게 한다.
보행자가 있는 구역의 연장 시간이 아직 None(미확정)이면 evaluate 시점에 실패한다.
"""

from enum import Enum, auto

from config import config


class SignalState(Enum):
    NORMAL = auto()     # 일반 신호 진행, 연장 조건 미충족 (무장 상태)
    EXTENDING = auto()  # 이번 판단에서 연장 발생
    EXTENDED = auto()   # 이번 하강 구간에서 이미 연장을 발급함 — 제어부 반영 대기 중
    CAPPED = auto()     # 상한 도달, 더 이상 연장 불가


class SignalExtensionStateMachine:
    def __init__(
        self,
        remaining_time_threshold_sec=None,
        zone_extension_sec=None,
        max_total_extension_sec=None,
        priority_zone_extension_sec=None,
        priority_max_total_extension_sec=None,
    ):
        self.remaining_time_threshold_sec = (
            remaining_time_threshold_sec
            if remaining_time_threshold_sec is not None
            else config.REMAINING_TIME_THRESHOLD_SEC
        )
        self.zone_extension_sec = (
            zone_extension_sec
            if zone_extension_sec is not None
            else config.ZONE_EXTENSION_SEC
        )
        self.max_total_extension_sec = (
            max_total_extension_sec
            if max_total_extension_sec is not None
            else config.MAX_TOTAL_EXTENSION_SEC
        )
        # 교통약자 우선 연장용(미정이면 None -> evaluate에서 일반 규칙으로 대체).
        self.priority_zone_extension_sec = (
            priority_zone_extension_sec
            if priority_zone_extension_sec is not None
            else config.PRIORITY_ZONE_EXTENSION_SEC
        )
        self.priority_max_total_extension_sec = (
            priority_max_total_extension_sec
            if priority_max_total_extension_sec is not None
            else config.PRIORITY_MAX_TOTAL_EXTENSION_SEC
        )

        if self.remaining_time_threshold_sec is None or self.max_total_extension_sec is None:
            raise NotImplementedError(
                "REMAINING_TIME_THRESHOLD_SEC / MAX_TOTAL_EXTENSION_SEC가 설정되지 않았습니다. "
                "팀 회의로 확정 후 config에 값을 채워주세요."
            )

        self.state = SignalState.NORMAL
        self.accumulated_extension_sec = 0
        # 엣지 트리거용 '무장' 플래그. evaluate()는 실시간 루프에서 매 프레임 호출되므로,
        # 조건이 참인 동안 계속 연장을 누적하면 순식간에 상한을 소진하고 제어부로 같은 명령을
        # 연타로 보내게 된다. 그래서 '잔여시간이 임계값 아래로 내려온 구간'당 한 번만 발급한다.
        # 무장은 '연장을 실제로 발급'할 때만 소모되고, 잔여시간이 임계값 위로 회복되면 되살아난다.
        self._armed = True

    @staticmethod
    def _extension_for_zone(ext_map, zone_index) -> int:
        sec = ext_map.get(zone_index)
        if sec is None:
            raise NotImplementedError(
                f"{zone_index}번 구역의 연장 시간(config.ZONE_EXTENSION_SEC)이 아직 미정(None)입니다. "
                "실측/팀 확정 후 값을 채워주세요."
            )
        return sec

    def evaluate(self, remaining_time_sec, occupied_zones, priority_mode=False) -> int:
        """이번 사이클에 추가할 연장 시간(초)을 반환한다. 연장하지 않으면 0.

        remaining_time_sec: 아두이노가 알려주는 현재 잔여 녹색 시간(초).
        occupied_zones: 확정 보행자들이 점유 중인 구역 번호 목록(1..N). 비어 있으면 보행자 없음.
        priority_mode: 교통약자 감지 시 True. 우선 연장 맵/상한이 설정돼 있으면 그쪽을 쓴다.
        """
        # 우선 연장 규칙이 설정돼 있으면 그것을, 아니면 일반 규칙을 사용 (미정이면 일반 규칙으로 안전하게 대체).
        ext_map = (
            self.priority_zone_extension_sec
            if (priority_mode and self.priority_zone_extension_sec is not None)
            else self.zone_extension_sec
        )
        cap = (
            self.priority_max_total_extension_sec
            if (priority_mode and self.priority_max_total_extension_sec is not None)
            else self.max_total_extension_sec
        )

        # 잔여시간이 넉넉하면 연장 조건 자체가 아니다. 이때 무장을 되살린다 —
        # 직전 연장이 제어부에 반영돼 시간이 회복됐다는 뜻이므로, 다음 하강에서 다시 판단해야 한다.
        if remaining_time_sec > self.remaining_time_threshold_sec:
            self._armed = True
            self.state = SignalState.NORMAL
            return 0

        if self.accumulated_extension_sec >= cap:
            self.state = SignalState.CAPPED
            return 0

        # 이번 하강 구간에서 이미 발급했다면 제어부가 반영할 때까지 더 보내지 않는다.
        if not self._armed:
            self.state = SignalState.EXTENDED
            return 0

        zones = [z for z in occupied_zones if z is not None]
        if not zones:
            self.state = SignalState.NORMAL
            return 0

        # 가장 많이 필요한 사람(연장 시간이 가장 큰 구역)에 맞춘다.
        desired = max(self._extension_for_zone(ext_map, z) for z in zones)
        if desired <= 0:
            # 양 끝 구역(1/5)만 점유 -> 연장 안 함.
            self.state = SignalState.NORMAL
            return 0

        step = min(desired, cap - self.accumulated_extension_sec)
        self.accumulated_extension_sec += step
        self._armed = False   # 이번 하강 구간의 무장 소모
        self.state = SignalState.EXTENDING
        return step

    def reset(self):
        """다음 보행 신호 사이클 시작 시 호출 (누적 연장·무장 초기화).

        직접 부르기보다 SignalExtensionPipeline.begin_new_cycle()을 쓴다. 그쪽이 잔류
        카운트·속도 히스토리·추적 상태까지 같이 정리한다. 이걸 아무도 부르지 않으면
        누적 연장이 사이클을 넘어 남아, 한 번 상한을 찍은 뒤로는 영구히 CAPPED가 된다.
        """
        self.state = SignalState.NORMAL
        self.accumulated_extension_sec = 0
        self._armed = True
