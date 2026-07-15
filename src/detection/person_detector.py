"""사람 검출 모델 래퍼.

CLAUDE.md 3.3: 실제 사용할 모델(YOLO 경량 버전, MediaPipe 등)은 라즈베리파이 실측 벤치마크 후 결정.
이 스텁은 모델 선정 전까지 인터페이스만 고정해, 이후 detection 모듈 교체가 zone/signal_extend에
영향을 주지 않도록 한다.
"""

from dataclasses import dataclass

from config import config


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str  # "person" 등. 교통약자 클래스가 추가되면 값 확장.

    def foot_point(self):
        """탑뷰 카메라에서는 발 위치 추정이 zone 판정에 더 적합할 수 있음 (CLAUDE.md 2.1)."""
        return ((self.x1 + self.x2) / 2, self.y2)

    def center_point(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class PersonDetector:
    def __init__(self, model_path=None, confidence_threshold=None):
        self.model_path = model_path or config.DETECTION_MODEL_PATH
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else config.DETECTION_CONFIDENCE_THRESHOLD
        )
        if self.model_path is None:
            raise NotImplementedError(
                "DETECTION_MODEL_PATH가 설정되지 않았습니다. "
                "모델 선정 및 라즈베리파이 벤치마크 완료 후 config에 값을 채워주세요."
            )
        self._model = self._load_model(self.model_path)

    def _load_model(self, model_path):
        raise NotImplementedError("모델 선정 후 실제 로딩 로직을 구현하세요.")

    def detect(self, frame) -> list[BoundingBox]:
        """frame에서 사람을 검출해 BoundingBox 리스트를 반환한다."""
        raise NotImplementedError("모델 선정 후 실제 추론 로직을 구현하세요.")
