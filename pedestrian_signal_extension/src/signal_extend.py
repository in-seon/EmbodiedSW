"""신호 연장 상태 머신 (위치 기반 차등 연장).

- 조건: (남은 시간 <= 임계값) AND (횡단보도 안에 확정 보행자 존재).
- 연장 시간은 보행자가 있는 "구역"에 따라 차등 (config.ZONE_EXTENSION_SEC).
  여러 명이면 가장 많이 필요한 사람(=연장 시간이 가장 큰 구역)에 맞춘다.
- 매 사이클 재평가: 보행자가 사라지거나 남은 시간이 충분해지면 연장하지 않는다.
- 누적 상한(config.MAX_TOTAL_EXTENSION_SEC) 존재 — 차량 신호 충돌 방지.
- 교통약자(휠체어/목발 등) 감지 시 우선 연장(priority_mode).

임계값/상한이 아직 팀 확정 전이면 생성 시점에 명시적으로 실패해 임의 값으로 동작하지 않게 한다.
보행자가 있는 구역의 연장 시간이 아직 None(미확정)이면 evaluate 시점에 실패한다.
"""

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from config import config


@dataclass(frozen=True)
class Occupant:
    """연장 판단에 넣는 '확정 보행자 한 명'.

    zone     : 점유 중인 구역 번호(1..N).
    eta_sec  : 진행 방향 기준 예상 통과 시간(초). 다음 경우 None이고, 그러면 이 사람은
               구역 규칙만으로 판단한다(안전한 폴백).
                 - 호모그래피 없음(실측 치수 미입력) -> 속도가 px/s라 초로 환산 불가
                 - 정지 중이라 진행 방향을 못 냄
                 - 아직 속도 샘플이 부족함
    """

    zone: int
    eta_sec: Optional[float] = None


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
        use_speed=None,
        eta_safety_margin=None,
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

        # 속도(ETA) 반영 여부와 안전계수 (CLAUDE.md 2.3, config.USE_SPEED_FOR_EXTENSION).
        self.use_speed = (
            use_speed if use_speed is not None else config.USE_SPEED_FOR_EXTENSION
        )
        self.eta_safety_margin = (
            eta_safety_margin
            if eta_safety_margin is not None
            else config.ETA_SAFETY_MARGIN
        )
        if self.use_speed and self.eta_safety_margin is None:
            raise NotImplementedError(
                "USE_SPEED_FOR_EXTENSION이 켜져 있는데 ETA_SAFETY_MARGIN이 설정되지 않았습니다. "
                "속도 추정이 실제보다 빠르면 연장을 덜 주게 되고 그건 보행자가 도로에 갇힌다는 뜻이므로, "
                "안전계수를 임의값으로 두지 않습니다. config에 값을 채워주세요."
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

    def _need_for(self, occupant, remaining_time_sec, ext_map) -> int:
        """이 보행자 한 명에게 필요한 연장 시간(초).

        속도를 쓰지 않거나 ETA가 없으면 구역 규칙 값을 그대로 쓴다.

        속도를 쓸 때는 **구역 값이 상한**이고, 실제로 모자란 만큼으로 깎는다:

            부족분 = ETA x 안전계수 - 남은 신호 시간
            필요량 = min(구역 값, ceil(부족분))

        올림(ceil)과 안전계수를 쓰는 이유: 과대 연장은 차량이 좀 더 기다리는 것이지만
        과소 연장은 보행자가 도로에 갇히는 것이다. 위험이 비대칭이므로 넉넉한 쪽으로 튼다.
        """
        zone_sec = self._extension_for_zone(ext_map, occupant.zone)
        if zone_sec <= 0:
            # 양 끝 구역 — ETA와 무관하게 연장하지 않는다(설계상 확정 규칙).
            return 0
        if not self.use_speed or occupant.eta_sec is None:
            return zone_sec
        shortfall = occupant.eta_sec * self.eta_safety_margin - remaining_time_sec
        if shortfall <= 0:
            return 0
        return min(zone_sec, math.ceil(shortfall))

    @staticmethod
    def _as_occupant(item) -> Occupant:
        """구역 번호(int)만 넘어와도 받아준다 — ETA 없음으로 해석."""
        if isinstance(item, Occupant):
            return item
        return Occupant(zone=item)

    def evaluate(self, remaining_time_sec, occupants, priority_mode=False) -> int:
        """이번 사이클에 추가할 연장 시간(초)을 반환한다. 연장하지 않으면 0.

        remaining_time_sec: 제어부가 알려주는 현재 잔여 녹색 시간(초).
        occupants: 확정 보행자 목록. Occupant(zone, eta_sec) 또는 구역 번호(int)의 iterable.
                   비어 있으면 보행자 없음.
        priority_mode: 교통약자 감지 시 True. 우선 연장 맵/상한이 설정돼 있으면 그쪽을 쓴다.

        여러 명이면 **사람마다 따로 필요량을 계산해 가장 큰 값**에 맞춘다. 속도를 쓰기 시작하면
        "가운데 있는 사람이 가장 오래 걸린다"는 가정이 깨지기 때문이다 — 가운데 사람이 빠르고
        가장자리 사람이 느릴 수 있고, 그때 가운데만 보면 느린 쪽을 도로에 가두게 된다.
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

        people = [self._as_occupant(o) for o in occupants if o is not None]
        people = [p for p in people if p.zone is not None]
        if not people:
            self.state = SignalState.NORMAL
            return 0

        # 가장 많이 필요한 사람에 맞춘다.
        desired = max(self._need_for(p, remaining_time_sec, ext_map) for p in people)
        if desired <= 0:
            # 양 끝 구역(1/5)만 점유했거나, 모두 제시간에 건널 수 있음 -> 연장 안 함.
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
