"""확정 보행자들의 진척도를 하나로 요약한다 (파이가 담당하는 전부).

## 파이는 '얼마나 연장할지'를 계산하지 않는다

    파이   : 이 사람들 중 가장 덜 건넌 사람이 몇 번째 구역에 있는가  -> ZONE <n>
    아두이노: 그 구역에 정상 속도로 도착했다면 남아 있어야 할 시간과
             실제 잔여 시간을 비교해, 뒤처진 만큼 조금씩 연장

"이 구역에 도착했을 때 몇 초 남아 있어야 정상인가"는 카운트다운을 소유한 쪽만 답할 수
있고, "이 사람이 몇 번째 구역에 있는가"는 영상을 보는 쪽만 답할 수 있다. 각자 자기만
아는 것을 판단한다. 전체 계약은 `docs/team_interface.md`, 결정 경위는
`docs/decisions.md` 2026-08-26 항목.

## 이 방식이 없애준 것들

기준표(구역별 '정상 도착 잔여시간')가 **정상 보행 속도를 암묵적으로 담고 있어서**,
파이가 속도를 재서 넘길 필요가 없어졌다. 그 결과 아래가 전부 불필요해졌다:

  - ETA 계산과 그에 필요한 최소 프레임률(4fps) — 파이가 3fps여도 정상 동작한다
  - 안전계수(ETA_SAFETY_MARGIN) — 기준표와의 차이가 곧 부족분이다
  - **교통약자 검출** — 느린 사람은 기준 대비 지연이 크게 잡혀 자동으로 더 연장받는다.
    "휠체어인가"를 알 필요 없이 "느린가"만 보면 되고, 그건 이 방식이 자동으로 잰다.

또한 매 구역에서 지연을 다시 재는 **피드백 제어**라 오차가 누적되지 않는다. 저 프레임률
때문에 구역 하나를 건너뛰어도, 다음 구역에서 누적 지연이 그대로 잡혀 자동으로 만회된다.

속도 추정 코드(src/speed.py)는 남아 있지만 **연장 결정에는 쓰이지 않는다** — 도구 화면과
발표 자료용 계측이다.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Occupant:
    """연장 판단에 넣는 '확정 보행자 한 명'.

    progress : 진척도 1..N. **물리 구역 번호가 아니라 진입 방향 기준**이다.
               1 = 방금 진입, N = 거의 다 건넜다. 변환은 zone.CrossingProgress가 한다.
    eta_sec  : 실측 예상 통과 시간(초). 전송하지 않는다 — 화면 표시·진단용이다.
    """

    progress: int
    eta_sec: Optional[float] = None


def _as_occupant(item) -> Occupant:
    """진척도(int)만 넘어와도 받아준다."""
    if isinstance(item, Occupant):
        return item
    return Occupant(progress=item)


def minimum_progress(occupants) -> Optional[int]:
    """가장 덜 건넌 사람의 진척도. 아무도 없으면 None.

    **최솟값 하나만 보내면 되는 이유**: 아두이노의 기준표는 진척도에 대해 단조 감소한다
    (1번 9초, 2번 7초 ... 5번 1초). 모든 사람이 같은 잔여 시간을 보므로

        부족분 = 기준[진척도] - 잔여시간

    는 진척도가 작을수록 항상 크다. 즉 **가장 작은 진척도의 사람이 항상 가장 부족하다.**
    여러 명의 값을 다 보낼 필요가 없고, 최솟값이 나머지를 지배한다.

    이것이 방향 보정이 필요한 이유이기도 하다. 물리 구역 번호를 그대로 쓰면 반대편에서
    들어온 사람의 '2번'이 '거의 다 건넜음'인데도 '대부분 남았음'으로 읽혀, 엉뚱한 사람이
    최솟값을 차지한다.
    """
    values = [
        person.progress
        for person in (_as_occupant(o) for o in occupants if o is not None)
        if person.progress is not None
    ]
    return min(values) if values else None


def maximum_eta_sec(occupants) -> Optional[float]:
    """확정 보행자 중 가장 큰 ETA. **전송하지 않는다** — 화면 표시·진단용.

    ETA가 계속 None으로 나오면 프레임률이 낮거나 사람이 멈춰 있다는 뜻이다. 연장 판단은
    ETA 없이도 돌아가지만, 속도 계측이 살아 있는지 눈으로 확인할 수 있으면 실측 때 편하다.
    """
    values = [
        person.eta_sec
        for person in (_as_occupant(o) for o in occupants if o is not None)
        if person.eta_sec is not None
    ]
    return max(values) if values else None
