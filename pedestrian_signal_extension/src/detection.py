"""검출 모델 래퍼 (COCO 사전학습 yolov8n).

CLAUDE.md 2.6: 실측 확인 결과 COCO 사전학습 yolov8n이 사람 모형을 "person"으로 검출했고
신뢰도는 80%대였다. 따라서 추가 파인튜닝 없이 사전학습 가중치를 그대로 쓴다.

교통약자(휠체어/목발/지팡이) 검출은 현재 보류다(CLAUDE.md 2.5). COCO에 해당 클래스가 없어
config.MOBILITY_AID_LABELS가 빈 튜플이고, 그 결과 is_mobility_aid()는 항상 False가 된다.
나중에 파인튜닝 모델이 생기면 config의 라벨만 채우면 되도록 인터페이스는 그대로 남겨둔다.

track_id는 YOLO.track(persist=True)가 부여한다. speed.py와 zone.py의 CrosswalkOccupancy가
사람을 프레임 간에 구분하려면 이 ID가 필요하다.
"""

from dataclasses import dataclass
from typing import Optional

from config import config


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str                     # config.PEDESTRIAN_LABEL 또는 config.MOBILITY_AID_LABELS 중 하나(모델 클래스명과 일치).
    track_id: Optional[int] = None  # 추적 ID. YOLO track()이 부여. 없으면 None.

    def foot_point(self):
        """발 위치 = bounding box 하단 모서리의 중심. 위치 판정·속도 추정의 기준점.

        카메라가 횡단보도를 사선으로 비추므로 사람의 전신이 대체로 보이고, 박스 아래쪽이
        지면에 닿는 지점에 대응한다. 반면 center_point()는 사람 키의 절반만큼 위에 떠 있어
        지면 좌표로 쓰면 실제 서 있는 위치보다 카메라 쪽으로 당겨진 지점으로 잘못 판정된다
        (CLAUDE.md 2.1). 그래서 zone 판정과 호모그래피 변환에는 항상 이 점을 쓴다.
        """
        return ((self.x1 + self.x2) / 2, self.y2)

    def center_point(self):
        """박스 중심. 지면 위치 판정에는 쓰지 말 것(위 foot_point 주석 참고).

        목표 2(쓰러짐 감지)처럼 지면 좌표가 필요 없는 용도로만 사용한다.
        """
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def aspect_ratio(self):
        """박스 가로/세로 비. 목표 2(쓰러짐 감지)가 자세 판단에 쓸 수 있도록 노출한다."""
        height = self.y2 - self.y1
        if height <= 0:
            return None
        return (self.x2 - self.x1) / height

    def is_pedestrian(self) -> bool:
        return self.label == config.PEDESTRIAN_LABEL

    def is_mobility_aid(self) -> bool:
        # MOBILITY_AID_LABELS가 빈 튜플인 동안(보류 상태)에는 항상 False.
        return self.label in config.MOBILITY_AID_LABELS


def is_mobility_aid_box(box: BoundingBox) -> bool:
    """검출 결과가 교통약자 클래스인지 확인.

    현재는 보류 상태라 항상 False다(CLAUDE.md 2.5). 파인튜닝 모델 도입 후 재개할 때,
    지팡이(cane)는 사선·저해상도 조건에서 매우 작게 보여 검출률이 낮을 수 있으므로
    별도 검증이 필요하다.
    """
    return box.is_mobility_aid()


class PersonDetector:
    """ultralytics YOLO 래퍼. 프레임당 추론 1회로 사람(+ 후일 교통약자)을 검출한다.

    detect()는 추적 ID가 붙은 BoundingBox 리스트를 반환한다. 모델을 파인튜닝 가중치로
    교체하더라도 이 시그니처가 유지되므로 zone/speed/signal_extend 쪽은 손댈 필요가 없다.
    """

    def __init__(self, model_path=None, confidence_threshold=None, tracker=None):
        self.model_path = model_path or config.DETECTION_MODEL_PATH
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else config.DETECTION_CONFIDENCE_THRESHOLD
        )
        self.tracker = tracker or config.DETECTION_TRACKER
        if self.model_path is None:
            raise NotImplementedError(
                "DETECTION_MODEL_PATH가 설정되지 않았습니다. config에 가중치 경로를 채워주세요."
            )
        self._model = self._load_model(self.model_path)
        # 모델이 아는 클래스 이름 {인덱스: 이름}. detect()에서 라벨 문자열로 되돌리는 데 쓴다.
        self._class_names = self._model.names

        # 관심 있는 클래스만 추론 단계에서 걸러 불필요한 박스를 만들지 않는다.
        wanted = {config.PEDESTRIAN_LABEL, *config.MOBILITY_AID_LABELS}
        self._class_ids = sorted(
            idx for idx, name in self._class_names.items() if name in wanted
        )
        if not self._class_ids:
            raise ValueError(
                f"모델 {self.model_path}에 '{config.PEDESTRIAN_LABEL}' 클래스가 없습니다. "
                f"모델이 아는 클래스: {sorted(self._class_names.values())}. "
                "config.PEDESTRIAN_LABEL이 모델 클래스명과 일치하는지 확인하세요."
            )

    def _load_model(self, model_path):
        # 지연 임포트: ultralytics는 무거워서, 이 클래스를 쓰지 않는 테스트/도구가
        # 임포트 비용을 물지 않게 한다.
        from ultralytics import YOLO

        return YOLO(model_path)

    def detect(self, frame) -> list[BoundingBox]:
        """frame에서 관심 클래스를 검출해 BoundingBox 리스트를 반환한다.

        track(persist=True)를 쓰므로 같은 사람에게는 프레임 간에 같은 track_id가 붙는다.
        추적기가 ID를 붙이지 못한 검출은 track_id=None으로 나가며, 속도 추정에서는 무시된다.
        """
        results = self._model.track(
            frame,
            persist=True,
            conf=self.confidence_threshold,
            classes=self._class_ids,
            tracker=self.tracker,
            verbose=False,
        )[0]

        boxes = []
        if results.boxes is None:
            return boxes

        for box in results.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            label = self._class_names[int(box.cls[0])]
            track_id = int(box.id[0]) if box.id is not None else None
            boxes.append(
                BoundingBox(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=float(box.conf[0]),
                    label=label,
                    track_id=track_id,
                )
            )
        return boxes

    def reset_tracker(self):
        """보행 신호 사이클이 바뀌는 등, 추적 상태를 끊고 싶을 때 호출."""
        if hasattr(self._model, "predictor") and self._model.predictor is not None:
            trackers = getattr(self._model.predictor, "trackers", None)
            if trackers:
                for tracker in trackers:
                    tracker.reset()
