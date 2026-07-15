"""신호 연장 상태 머신.

CLAUDE.md 2.3, 2.4:
- (남은 시간 <= 임계값) AND (zone 잔류) -> 5초 단위 연장, 누적 상한 존재.
- 매 사이클 잔류 재평가 -> 보행자가 사라지면 추가 연장 중단.
- 교통약자 감지 시 우선 연장(세부 규칙 미정).

상한값과 우선 연장 규칙이 아직 팀 확정 전이므로, config 값이 None이면
생성 시점에 명시적으로 실패해 임의 값으로 동작하지 않도록 한다.
"""

from enum import Enum, auto

from config import config


class SignalState(Enum):
    NORMAL = auto()
    EXTENDING = auto()
    CAPPED = auto()  # 상한 도달, 더 이상 연장 불가


class SignalExtensionStateMachine:
    def __init__(
        self,
        remaining_time_threshold_sec=None,
        extension_step_sec=None,
        max_total_extension_sec=None,
    ):
        self.remaining_time_threshold_sec = (
            remaining_time_threshold_sec
            if remaining_time_threshold_sec is not None
            else config.REMAINING_TIME_THRESHOLD_SEC
        )
        self.extension_step_sec = extension_step_sec or config.EXTENSION_STEP_SEC
        self.max_total_extension_sec = (
            max_total_extension_sec
            if max_total_extension_sec is not None
            else config.MAX_TOTAL_EXTENSION_SEC
        )
        if self.remaining_time_threshold_sec is None or self.max_total_extension_sec is None:
            raise NotImplementedError(
                "REMAINING_TIME_THRESHOLD_SEC / MAX_TOTAL_EXTENSION_SEC가 설정되지 않았습니다. "
                "팀 회의로 확정 후 config에 값을 채워주세요."
            )

        self.state = SignalState.NORMAL
        self.accumulated_extension_sec = 0

    def evaluate(self, remaining_time_sec, pedestrian_resident, priority_mode=False) -> int:
        """이번 사이클에 추가할 연장 시간(초)을 반환한다. 연장하지 않으면 0.

        priority_mode: 교통약자 감지 시 True. 세부 우선 연장 규칙은 미정이므로
        현재는 일반 규칙과 동일하게 동작한다 (CLAUDE.md 2.4 확정 후 분기 추가).
        """
        if self.accumulated_extension_sec >= self.max_total_extension_sec:
            self.state = SignalState.CAPPED
            return 0

        if remaining_time_sec > self.remaining_time_threshold_sec:
            self.state = SignalState.NORMAL
            return 0

        if not pedestrian_resident:
            self.state = SignalState.NORMAL
            return 0

        # TODO(팀 확정 필요): 우선 연장 시 단위/상한 차등 규칙 (config.PRIORITY_EXTENSION_STEP_SEC).
        # 규칙 확정 전까지는 priority_mode 여부와 무관하게 일반 단위를 사용한다.
        step = self.extension_step_sec

        remaining_budget = self.max_total_extension_sec - self.accumulated_extension_sec
        step = min(step, remaining_budget)

        self.accumulated_extension_sec += step
        self.state = SignalState.EXTENDING
        return step

    def reset(self):
        self.state = SignalState.NORMAL
        self.accumulated_extension_sec = 0
