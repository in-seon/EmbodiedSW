"""PersonDetector 래퍼 테스트 (실제 YOLO 가중치 없이 가짜 모델로).

핵심 관심사: 포즈 가중치(yolov8n-pose.pt)를 쓰면 한 번의 추론으로 사람 박스와
키포인트가 같이 나오므로, 목표 1(신호 연장)과 목표 2(쓰러짐 감지)가 추론을 공유할 수 있다.
그러려면 PersonDetector가 키포인트를 그대로 실어 보내야 한다.
"""

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
    """ultralytics YOLO 흉내. names / track()만 있으면 된다."""

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
    detector = PersonDetector.__new__(PersonDetector)   # __init__ 우회(가중치 로드 회피)
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
    """포즈 모델이면 검출 박스마다 (17,3) 키포인트가 실려 나온다."""
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
    """일반 yolov8n.pt(키포인트 없음)에서는 None 그대로 — 기존 동작이 바뀌지 않는다."""
    result = _FakeResult(boxes=[_FakeBox((0, 0, 10, 20), 0, 0.9, 1)], keypoints=None)

    boxes = _detector_with(result).detect(frame=None)

    assert boxes[0].keypoints is None
    assert boxes[0].track_id == 1
    assert boxes[0].is_pedestrian() is True


def test_detector_survives_keypoint_count_mismatch():
    """키포인트 개수가 박스보다 적어도 죽지 않고 None으로 채운다(방어적)."""
    kpts = np.zeros((1, 17, 3), dtype=float)
    result = _FakeResult(
        boxes=[_FakeBox((0, 0, 10, 20), 0, 0.9, 1), _FakeBox((30, 0, 40, 20), 0, 0.8, 2)],
        keypoints=_FakeKeypoints(kpts),
    )

    boxes = _detector_with(result).detect(frame=None)

    assert boxes[0].keypoints is not None
    assert boxes[1].keypoints is None


def test_detector_passes_inference_params_to_model():
    """추론 파라미터가 전부 config에서 나가야 한다(PoC의 imgsz/conf/tracker와 같은 자리)."""
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
    """인자를 안 주면 config 값을 쓴다 — 파라미터가 config 한 곳에만 있게 하기 위함."""
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


# =====================================================================
# 교통약자(휠체어/목발) 보조 검출기 — CLAUDE.md 2.5
# =====================================================================
# COCO에 해당 클래스가 없어 사람 검출과 같은 모델로는 못 잡는다. 별도 가중치를 쓰되,
# "이 사람이 교통약자인가"는 프레임마다 바뀌는 값이 아니므로 저빈도로만 돌린다.

def _aid_detector_with(result, **kw):
    from src.detection import MobilityAidDetector

    detector = MobilityAidDetector.__new__(MobilityAidDetector)
    detector._model = _FakeModel(result)
    detector._class_names = {0: "wheelchair", 1: "crutches"}
    detector._model.names = detector._class_names
    detector.confidence_threshold = kw.get("conf", 0.3)
    detector.imgsz = kw.get("imgsz", 320)
    detector.every_n_frames = kw.get("every", 10)
    detector._wanted = ()
    detector._class_ids = None
    detector._frame_count = 0
    detector._cached = []
    detector.inference_count = 0
    return detector


def test_mobility_aid_detector_reports_its_classes():
    result = _FakeResult(boxes=[_FakeBox((0, 0, 10, 20), 0, 0.7, None)], keypoints=None)
    boxes = _aid_detector_with(result, every=1).detect(frame=None)

    assert [b.label for b in boxes] == ["wheelchair"]
    assert boxes[0].confidence == pytest.approx(0.7)


def test_mobility_aid_detector_runs_only_every_n_frames():
    """매 프레임 두 번째 모델을 돌리면 FPS가 반토막 난다 — 저빈도로만 추론한다."""
    result = _FakeResult(boxes=[_FakeBox((0, 0, 10, 20), 0, 0.7, None)], keypoints=None)
    detector = _aid_detector_with(result, every=3)

    for _ in range(6):
        detector.detect(frame=None)

    assert detector.inference_count == 2      # 6프레임 / 3 = 2회만 추론


def test_mobility_aid_detector_reuses_last_result_between_inferences():
    """추론을 건너뛴 프레임에서도 직전 결과를 유지한다(깜빡임 방지)."""
    result = _FakeResult(boxes=[_FakeBox((0, 0, 10, 20), 0, 0.7, None)], keypoints=None)
    detector = _aid_detector_with(result, every=3)

    first = detector.detect(frame=None)        # 추론함
    second = detector.detect(frame=None)       # 건너뜀 -> 직전 결과 재사용

    assert [b.label for b in second] == [b.label for b in first]


def test_mobility_aid_detector_disabled_without_model_path(monkeypatch):
    """가중치가 미정이면(config가 None) 조용히 비활성 — 항상 빈 목록."""
    from config import config
    from src.detection import MobilityAidDetector

    monkeypatch.setattr(config, "MOBILITY_AID_MODEL_PATH", None)
    detector = MobilityAidDetector()

    assert detector.enabled is False
    assert detector.detect(frame=None) == []
