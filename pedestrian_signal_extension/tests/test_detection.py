"""PersonDetector 래퍼 테스트 (실제 YOLO 가중치 없이 가짜 모델로)."""

import numpy as np
import pytest

from src.detection import BoundingBox, PersonDetector


class _FakeBox:
    def __init__(self, xyxy, cls_id, conf, track_id):
        self.xyxy = [np.array(xyxy, dtype=float)]
        self.cls = [cls_id]
        self.conf = [conf]
        self.id = None if track_id is None else [track_id]


class _FakeKeypoints:
    def __init__(self, arr):
        self.data = arr


class _FakeResult:
    def __init__(self, boxes, keypoints=None):
        self.boxes = boxes
        self.keypoints = keypoints


class _FakeModel:

    def __init__(self, result):
        self.names = {0: "person"}
        self._result = result
        self.last_kwargs = None

    def track(self, frame, **kwargs):
        self.last_kwargs = kwargs
        return [self._result]

    def predict(self, frame, **kwargs):
        self.last_kwargs = kwargs
        return [self._result]


def _detector_with(result):
    detector = PersonDetector.__new__(PersonDetector)
    detector._model = _FakeModel(result)
    detector._class_names = detector._model.names
    detector._class_ids = [0]
    detector.confidence_threshold = 0.5
    detector.tracker = "bytetrack.yaml"
    detector.imgsz = 640
    return detector


def test_bounding_box_keypoints_default_none():
    box = BoundingBox(0, 0, 10, 20, 0.9, "person")
    assert box.keypoints is None


def test_detector_passes_through_pose_keypoints():
    kpts = np.arange(2 * 17 * 3, dtype=float).reshape(2, 17, 3)
    result = _FakeResult(
        boxes=[_FakeBox((0, 0, 10, 20), 0, 0.9, 1), _FakeBox((30, 0, 40, 20), 0, 0.8, 2)],
        keypoints=_FakeKeypoints(kpts),
    )

    boxes = _detector_with(result).detect(frame=None)

    assert len(boxes) == 2
    assert np.array_equal(boxes[0].keypoints, kpts[0])
    assert np.array_equal(boxes[1].keypoints, kpts[1])


def test_detector_leaves_keypoints_none_for_box_only_model():
    result = _FakeResult(boxes=[_FakeBox((0, 0, 10, 20), 0, 0.9, 1)], keypoints=None)

    boxes = _detector_with(result).detect(frame=None)

    assert boxes[0].keypoints is None
    assert boxes[0].track_id == 1
    assert boxes[0].is_pedestrian() is True


def test_detector_survives_keypoint_count_mismatch():
    kpts = np.zeros((1, 17, 3), dtype=float)
    result = _FakeResult(
        boxes=[_FakeBox((0, 0, 10, 20), 0, 0.9, 1), _FakeBox((30, 0, 40, 20), 0, 0.8, 2)],
        keypoints=_FakeKeypoints(kpts),
    )

    boxes = _detector_with(result).detect(frame=None)

    assert boxes[0].keypoints is not None
    assert boxes[1].keypoints is None


def test_detector_passes_inference_params_to_model():
    result = _FakeResult(boxes=[_FakeBox((0, 0, 10, 20), 0, 0.9, 1)], keypoints=None)
    detector = _detector_with(result)
    detector.imgsz = 320

    detector.detect(frame=None)

    kwargs = detector._model.last_kwargs
    assert kwargs["imgsz"] == 320
    assert kwargs["conf"] == 0.5
    assert kwargs["tracker"] == "bytetrack.yaml"
    assert kwargs["persist"] is True


def test_inference_params_default_to_config(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "DETECTION_IMGSZ", 416)
    monkeypatch.setattr(config, "DETECTION_CONFIDENCE_THRESHOLD", 0.35)
    monkeypatch.setattr(config, "DETECTION_TRACKER", "botsort.yaml")
    result = _FakeResult(boxes=[], keypoints=None)
    monkeypatch.setattr(PersonDetector, "_load_model",
                        lambda self, path: _FakeModel(result))

    detector = PersonDetector()

    assert detector.imgsz == 416
    assert detector.confidence_threshold == 0.35
    assert detector.tracker == "botsort.yaml"

