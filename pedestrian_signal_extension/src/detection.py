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
    label: str                     
    track_id: Optional[int] = None 
    keypoints: object = None

    def foot_point(self):
        return ((self.x1 + self.x2) / 2,
                self.y2 - config.FOOT_POINT_OFFSET_RATIO * (self.y2 - self.y1))

    def center_point(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def is_pedestrian(self) -> bool:
        return self.label == config.PEDESTRIAN_LABEL


class PersonDetector:
    def __init__(self, model_path=None, confidence_threshold=None, tracker=None, imgsz=None):
        self.model_path = model_path or config.DETECTION_MODEL_PATH
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else config.DETECTION_CONFIDENCE_THRESHOLD
        )
        self.tracker = tracker or config.DETECTION_TRACKER
        self.imgsz = imgsz if imgsz is not None else config.DETECTION_IMGSZ
        if self.model_path is None:
            raise NotImplementedError(
                "DETECTION_MODEL_PATH가 설정되지 않았습니다. config에 가중치 경로를 채워주세요."
            )
        self._model = self._load_model(self.model_path)
        self._class_names = self._model.names
        self._class_ids = sorted(
            idx for idx, name in self._class_names.items()
            if name == config.PEDESTRIAN_LABEL
        )
        if not self._class_ids:
            raise ValueError(
                f"모델 {self.model_path}에 '{config.PEDESTRIAN_LABEL}' 클래스가 없습니다. ")

    def _load_model(self, model_path):
        from ultralytics import YOLO
        return YOLO(model_path)

    def detect(self, frame) -> list[BoundingBox]:
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
        if hasattr(self._model, "predictor") and self._model.predictor is not None:
            trackers = getattr(self._model.predictor, "trackers", None)
            if trackers:
                for tracker in trackers:
                    tracker.reset()

