"""구역 기반 연장 요구량 산출 (파이가 담당하는 부분).

## 역할 분담이 바뀌었다 — 이 파일이 작아진 이유

예전에는 파이가 잔여 녹색 시간을 시리얼로 읽어 와서 "지금 연장할까?"까지 판단했다.
그 구조에서는 임계값 비교·누적 상한·엣지 트리거(무장 플래그)가 전부 필요했고,
그래서 이 파일이 200줄이 넘었다.

지금은 **제어부(아두이노)가 카운트다운을 소유하므로 그 판단도 제어부가 한다.**
"남은 시간이 5초 미만인가"는 7세그먼트를 직접 세는 쪽만 답할 수 있는 질문이고,
"이 사람이 몇 번 구역에 있나"는 영상을 보는 쪽만 답할 수 있는 질문이다.
각자 자기만 아는 것을 판단하도록 나눈 결과, 파이 쪽에 남은 것은 아래 한 가지다:

    확정 보행자들의 구역 -> 필요한 연장 초

옮겨간 것들(임계값 5초, 누적 상한 10초, 엣지 트리거, 사이클 리셋)의 근거와 주의사항은
`docs/team_interface.md`의 "아두이노 계약"에 정리해 두었다. 값 자체의 출처는
`docs/decisions.md` 2026-08-19 항목에 그대로 남아 있다.

## ETA(예상 통과 시간)는 여기서 쓰지 않고 그대로 넘긴다

속도 기반 보정은 `연장 = min(구역값, ceil(ETA x 안전계수 - 잔여시간))` 인데,
잔여시간을 파이가 모르므로 뺄셈을 할 수 없다. 대신 **두 숫자를 다 가진 아두이노**가
그 계산을 한다. 파이는 ETA를 실어 보내기만 한다(`config.USE_SPEED_FOR_EXTENSION`).

구역값이 상한으로 남는 성질은 그대로다 — **속도는 연장을 깎기만 하고 늘리지 않는다.**
"양 끝 구역은 연장 안 함" 같은 정책이 ETA와 무관하게 유지된다.
"""

from dataclasses import dataclass
from typing import Optional

from config import config


@dataclass(frozen=True)
class Occupant:
    """연장 판단에 넣는 '확정 보행자 한 명'.

    zone     : 점유 중인 구역 번호(1..N).
    eta_sec  : 진행 방향 기준 예상 통과 시간(초). 다음 경우 None이고, 그러면 아두이노가
               구역 규칙만으로 판단한다(안전한 폴백).
                 - 호모그래피 없음(실측 치수 미입력) -> 속도가 px/s라 초로 환산 불가
                 - 정지 중이라 진행 방향을 못 냄
                 - 아직 속도 샘플이 부족함
    """

    zone: int
    eta_sec: Optional[float] = None


class ZoneExtensionRule:
    """확정 보행자들의 구역으로 '필요한 연장 초'를 낸다.

    상태를 갖지 않는다 — 누적도 무장도 없다. 그것들은 제어부가 소유한다.
    매 프레임 같은 입력에 같은 출력을 내는 순수한 규칙이다.
    """

    def __init__(self, zone_extension_sec=None, priority_zone_extension_sec=None):
        self.zone_extension_sec = (
            zone_extension_sec
            if zone_extension_sec is not None
            else config.ZONE_EXTENSION_SEC
        )
        # 교통약자 우선 연장용. None이면 일반 규칙으로 안전하게 대체된다.
        # 이 구조에서 '우선'은 **보내는 숫자가 커지는 것**일 뿐이고, 프로토콜은 그대로다
        # (아두이노는 받은 숫자를 쓸 뿐 그것이 교통약자인지 알 필요가 없다).
        self.priority_zone_extension_sec = (
            priority_zone_extension_sec
            if priority_zone_extension_sec is not None
            else config.PRIORITY_ZONE_EXTENSION_SEC
        )

    @staticmethod
    def _extension_for_zone(ext_map, zone_index) -> int:
        sec = ext_map.get(zone_index)
        if sec is None:
            raise NotImplementedError(
                f"{zone_index}번 구역의 연장 시간(config.ZONE_EXTENSION_SEC)이 아직 미정(None)입니다. "
                "실측/팀 확정 후 값을 채워주세요."
            )
        return sec

    @staticmethod
    def _as_occupant(item) -> Occupant:
        """구역 번호(int)만 넘어와도 받아준다 — ETA 없음으로 해석."""
        if isinstance(item, Occupant):
            return item
        return Occupant(zone=item)

    def required_sec(self, occupants, priority_mode=False) -> int:
        """이 사람들에게 필요한 연장 초. 아무도 없거나 양 끝 구역만이면 0.

        여러 명이면 **가장 큰 값**에 맞춘다. 가장 오래 걸리는 사람을 도로에 가두지
        않기 위해서다.

        priority_mode: 교통약자가 검출됐으면 True. 우선 연장 맵이 설정돼 있으면 그쪽을
                       쓰고, 미정(None)이면 일반 맵으로 대체한다.
        """
        ext_map = (
            self.priority_zone_extension_sec
            if (priority_mode and self.priority_zone_extension_sec is not None)
            else self.zone_extension_sec
        )

        people = [self._as_occupant(o) for o in occupants if o is not None]
        people = [p for p in people if p.zone is not None]
        if not people:
            return 0
        return max(self._extension_for_zone(ext_map, p.zone) for p in people)

    def max_eta_sec(self, occupants) -> Optional[float]:
        """확정 보행자 중 가장 오래 걸리는 사람의 ETA(초). 아무도 없으면 None.

        연장 요구량과 마찬가지로 '가장 오래 걸리는 사람'에 맞춘다. 여기서 평균이나
        최솟값을 쓰면 느린 사람이 그만큼 시간을 못 받는다.

        **양 끝 구역(연장 0)에 있는 사람은 제외한다.** 그들에게는 연장을 주지 않기로
        했으므로 그 사람의 ETA를 보내면 아두이노가 줄 수 없는 시간을 계산하게 된다.
        """
        candidates = [
            p.eta_sec
            for p in (self._as_occupant(o) for o in occupants if o is not None)
            if p.zone is not None
            and p.eta_sec is not None
            and self.zone_extension_sec.get(p.zone)
        ]
        return max(candidates) if candidates else None
