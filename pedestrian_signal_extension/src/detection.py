"""검출 모델 래퍼 (COCO 사전학습 yolov8n).

CLAUDE.md 2.6: 실측 확인 결과 COCO 사전학습 yolov8n이 사람 모형을 "person"으로 검출했고
신뢰도는 80%대였다. 따라서 추가 파인튜닝 없이 사전학습 가중치를 그대로 쓴다.

교통약자를 따로 검출하지 않는다. 구역 기준표가 정상 보행 속도를 담고 있어서, 느린 사람은
기준 대비 지연으로 자동 검출되어 더 연장받는다 — 느린 이유가 무엇이든 결과가 같다
(config.py '교통약자를 따로 검출하지 않는 이유', docs/decisions.md 2026-08-26).

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
    label: str                     # 모델 클래스명. 사람은 config.PEDESTRIAN_LABEL.
    track_id: Optional[int] = None  # 추적 ID. YOLO track()이 부여. 없으면 None.
    # COCO 17개 관절의 (x, y, conf) 배열. 포즈 가중치(yolov8n-pose.pt)일 때만 채워진다.
    # 목표 2(쓰러짐 감지)가 몸통 각도를 재는 데 쓴다 — src/fall_detection.py 참고.
    # 일반 검출 가중치면 None이고, 그때 쓰러짐 판정은 bbox 가로/세로 비율로 폴백한다.
    keypoints: object = None

    def foot_point(self):
        """발 위치 = bounding box 하단 모서리의 중심. 위치 판정·속도 추정의 기준점.

        카메라가 횡단보도를 사선으로 비추므로 사람의 전신이 대체로 보이고, 박스 아래쪽이
        지면에 닿는 지점에 대응한다. 반면 center_point()는 사람 키의 절반만큼 위에 떠 있어
        지면 좌표로 쓰면 실제 서 있는 위치보다 카메라 쪽으로 당겨진 지점으로 잘못 판정된다
        (CLAUDE.md 2.1). 그래서 zone 판정과 호모그래피 변환에는 항상 이 점을 쓴다.
        """
        return ((self.x1 + self.x2) / 2, self.y2)

    def center_point(self):
        """박스 중심. **지면 위치 판정에는 쓰지 말 것**(위 foot_point 주석 참고).

        지면 좌표가 필요 없는 용도로만 쓴다. 현재 실사용처는 없고,
        tests/test_pipeline.py::test_uses_foot_point_not_center_for_zone 이
        "발 위치를 쓰지 중심점을 쓰지 않는다"는 설계 결정을 이 값과 대조해 고정한다.
        """
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def is_pedestrian(self) -> bool:
        return self.label == config.PEDESTRIAN_LABEL


class PersonDetector:
    """ultralytics YOLO 래퍼. 프레임당 추론 1회로 **사람만** 검출한다.

    detect()는 추적 ID가 붙은 BoundingBox 리스트를 반환한다. 모델을 파인튜닝 가중치로
    교체하더라도 이 시그니처가 유지되므로 zone/speed/signal_extend 쪽은 손댈 필요가 없다.
    """

    def __init__(self, model_path=None, confidence_threshold=None, tracker=None, imgsz=None):
        self.model_path = model_path or config.DETECTION_MODEL_PATH
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else config.DETECTION_CONFIDENCE_THRESHOLD
        )
        self.tracker = tracker or config.DETECTION_TRACKER
        # 추론 입력 해상도. 프레임 해상도(CAMERA_RESOLUTION)와 다른 값이며, 이걸 줄여도
        # 박스는 원본 프레임 좌표로 돌아오므로 zone/호모그래피 재캘리브레이션이 필요 없다.
        self.imgsz = imgsz if imgsz is not None else config.DETECTION_IMGSZ
        if self.model_path is None:
            raise NotImplementedError(
                "DETECTION_MODEL_PATH가 설정되지 않았습니다. config에 가중치 경로를 채워주세요."
            )
        self._model = self._load_model(self.model_path)
        # 모델이 아는 클래스 이름 {인덱스: 이름}. detect()에서 라벨 문자열로 되돌리는 데 쓴다.
        self._class_names = self._model.names

        # 관심 있는 클래스만 추론 단계에서 걸러 불필요한 박스를 만들지 않는다.
        self._class_ids = sorted(
            idx for idx, name in self._class_names.items()
            if name == config.PEDESTRIAN_LABEL
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
            imgsz=self.imgsz,
            verbose=False,
        )[0]

        boxes = []
        if results.boxes is None:
            return boxes

        # 포즈 가중치면 같은 추론 결과에 키포인트가 함께 들어 있다. 추론을 한 번 더 돌리지
        # 않고 그대로 실어 보내, 목표 1(신호 연장)과 목표 2(쓰러짐)가 추론을 공유하게 한다.
        keypoints = None
        if getattr(results, "keypoints", None) is not None:
            keypoints = results.keypoints.data
            if hasattr(keypoints, "cpu"):
                keypoints = keypoints.cpu().numpy()

        for index, box in enumerate(results.boxes):
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            label = self._class_names[int(box.cls[0])]
            track_id = int(box.id[0]) if box.id is not None else None
            kp = keypoints[index] if keypoints is not None and index < len(keypoints) else None
            boxes.append(
                BoundingBox(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=float(box.conf[0]),
                    label=label,
                    track_id=track_id,
                    keypoints=kp,
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

