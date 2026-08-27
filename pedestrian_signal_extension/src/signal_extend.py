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

## 보내는 값은 "연장할 초"가 아니라 "앞으로 필요한 시간(초)"이다

    EXTEND <초>      <- 이 보행자가 횡단을 마치는 데 앞으로 필요한 시간

아두이노가 `부족분 = 필요시간 x ETA_SAFETY_MARGIN - 잔여시간` 으로 실제 연장량을 낸다.
파이는 잔여 시간을 모르므로 뺄셈을 할 수 없고, 뺄셈이 저쪽에서 일어나려면 보내는 값이
'연장량'이 아니라 '필요 시간'이어야 한다.

그 값은 사람마다 이렇게 정해진다:

    ETA가 있으면  -> 실측 속도로 계산한 예상 통과 시간 (올림)
    없으면        -> 구역값 (거친 추정. FPS가 낮거나 사람이 멈춰 있으면 자주 이 경로다)

## 구역의 역할이 바뀌었다 — 이제 '게이트'다

예전에는 구역이 연장량을 직접 정했다(3번=5초). 지금은 **"보낼까 말까"를 정하는 역할이
주다.** 값이 0인 구역(양 끝)에 있는 사람은 ETA가 아무리 커도 제외된다 — "이제 막
진입했거나 거의 다 건넜으므로 연장하지 않는다"는 설계 규칙이 ETA와 무관하게 유지된다.

ETA가 없을 때만 구역값이 '얼마나'까지 정한다(폴백).

⚠️ 그래서 **구역값 3초/5초의 의미가 달라졌다.** 원래는 국내 운영 사례에서 가져온
'연장량'이었는데(광주시 5초, 서울시 3~6초), 지금 그 자리는 '필요 시간의 추정치'다.
폴백에서만 쓰이므로 당장 문제는 없지만, 실측 후 모형 기하로 다시 뽑는 것이 정확하다
(docs/decisions.md 2026-08-26 참고).
"""

import math
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

    def required_sec(self, occupants, priority_mode=False, use_eta=None) -> int:
        """가장 오래 걸리는 사람 기준 **"앞으로 필요한 시간(초)"**. 0이면 연장 불필요.

        아두이노로 그대로 나가는 값이며, 저쪽에서
        `부족분 = 필요시간 x ETA_SAFETY_MARGIN - 잔여시간` 으로 실제 연장량을 낸다.
        **"연장할 초"가 아니라 "필요한 시간"이다** — 파이는 잔여 시간을 모르므로
        뺄셈을 할 수 없고, 뺄셈이 저쪽에서 일어나려면 보내는 값이 필요 시간이어야 한다.

        사람마다:
          - ETA가 있으면 그 값 (실측 속도 기반, 올림)
          - 없으면 구역값 (config.ZONE_EXTENSION_SEC) — 거친 추정이지만 없는 것보다 낫다

        **구역은 이제 '보낼지 말지'의 게이트 역할이 주다.** 값이 0인 구역(양 끝)에 있는
        사람은 ETA가 있든 없든 제외된다 — "이제 막 진입했거나 거의 다 건넜으므로 연장하지
        않는다"는 설계 규칙이 ETA와 무관하게 유지된다.

        여러 명이면 가장 큰 값에 맞춘다. 느린 쪽을 도로에 가두지 않기 위해서다.

        use_eta: None이면 config.USE_SPEED_FOR_EXTENSION을 따른다. False면 ETA를 무시하고
                 항상 구역값을 쓴다(속도 검증 전까지의 안전한 동작).
        """
        if use_eta is None:
            use_eta = config.USE_SPEED_FOR_EXTENSION

        ext_map = (
            self.priority_zone_extension_sec
            if (priority_mode and self.priority_zone_extension_sec is not None)
            else self.zone_extension_sec
        )

        needs = []
        for item in occupants:
            if item is None:
                continue
            person = self._as_occupant(item)
            if person.zone is None:
                continue
            zone_sec = self._extension_for_zone(ext_map, person.zone)
            if zone_sec <= 0:
                continue                      # 양 끝 구역 — 게이트에서 걸린다
            if use_eta and person.eta_sec is not None:
                needs.append(int(math.ceil(person.eta_sec)))
            else:
                needs.append(zone_sec)
        return max(needs) if needs else 0

    def max_eta_sec(self, occupants) -> Optional[float]:
        """확정 보행자 중 가장 큰 ETA. 화면 표시·로깅용이며 전송에는 쓰지 않는다.

        전송 값은 required_sec()이 이미 ETA를 반영해 만든다. 이 메서드는 "ETA가 실제로
        나오고 있는가"를 눈으로 확인하기 위한 것이다 — 값이 계속 None이면 FPS가 낮거나
        사람이 멈춰 있다는 뜻이고, 그러면 전송 값이 구역값으로 폴백되고 있다.

        양 끝 구역(연장 0)에 있는 사람은 제외한다 — 어차피 연장 대상이 아니다.
        """
        candidates = [
            p.eta_sec
            for p in (self._as_occupant(o) for o in occupants if o is not None)
            if p.zone is not None
            and p.eta_sec is not None
            and self.zone_extension_sec.get(p.zone)
        ]
        return max(candidates) if candidates else None
