"""교통약자(휠체어/지팡이) 검출 보조 모듈.

CLAUDE.md 2.4: 정확한 접근 방식(별도 클래스 검출 vs bbox 종횡비/속도 패턴 근사)이 아직 미정.
실제 채택 전 팀 내 오탐/미탐 검증이 필요하므로, 이 스텁은 인터페이스만 정의하고
휴리스틱 근사 방식 하나를 기본 구현으로 남겨둔다 (정확도 보장 없음 — 검증 전 사용 금지).
"""

from src.detection.person_detector import BoundingBox


class MobilityAidDetector:
    def is_likely_mobility_aid_user(self, box: BoundingBox) -> bool:
        """bbox 종횡비 기반 근사 판단.

        검증되지 않은 휴리스틱이므로 실제 배치 전 반드시 팀 내 테스트 영상으로 검증할 것.
        """
        raise NotImplementedError(
            "정확도 검증 전이므로 임의로 구현하지 않음. "
            "팀과 판단 기준(종횡비 임계값 등)을 정한 뒤 구현하세요."
        )
