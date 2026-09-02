"""쓰러짐 감지(목표 2) 테스트.

두 가지를 확인한다.

1. **배선**: BoundingBox -> Person 어댑터, ROI 생성, FallDetectionPipeline이
   원문 main() 루프와 같은 순서로 로직을 부른다.
2. **이식 충실도**: crosswalk_poc.py에서 그대로 옮겨 온 판단 로직의 핵심 동작이
   그대로인지 고정한다. 이 테스트들이 깨지면 '원문 그대로' 약속이 깨진 것이다.

카메라도 모델도 쓰지 않는다 — 시각(now)을 인자로 주입하므로 시간 판단까지 전부 검증된다.
"""

import numpy as np
import pytest

from config import config
from src.detection import BoundingBox
from src.fall_detection import (
    FallDetectionPipeline,
    FallMonitor,
    FallTracker,
    Person,
    bbox_overlap,
    looks_fallen,
    people_from_boxes,
    person_from_box,
    roi_from_ratio,
    roi_from_zones,
    torso_angle_deg,
)
from src.zone import CrosswalkZones

ROI = (0, 0, 640, 480)

CFG = dict(config.FALL_CONFIG, fall_confirm_sec=3.0, fall_clear_sec=3.0)


def standing(x=100, y=100, w=40, h=120, track_id=1):
    """서 있는 사람 — 세로로 길다."""
    return Person(bbox=(x, y, x + w, y + h), conf=0.9, keypoints=None, track_id=track_id)


def lying(x=100, y=100, w=120, h=40, track_id=1):
    """누운 사람 — 가로로 길다(키포인트 없을 때의 폴백 판정 대상)."""
    return Person(bbox=(x, y, x + w, y + h), conf=0.9, keypoints=None, track_id=track_id)


def test_person_from_box_maps_fields():
    box = BoundingBox(10.4, 20.6, 50.2, 140.9, 0.87, "person", track_id=3)
    person = person_from_box(box)

    assert person.bbox == (10, 20, 50, 140)
    assert person.conf == pytest.approx(0.87)
    assert person.track_id == 3
    assert person.keypoints is None


def test_person_from_box_carries_keypoints():
    """포즈 가중치를 쓰면 키포인트가 그대로 넘어와야 몸통 각도 판정이 산다."""
    kp = np.zeros((17, 3))
    box = BoundingBox(0, 0, 10, 20, 0.9, "person", track_id=1, keypoints=kp)

    assert person_from_box(box).keypoints is kp


def test_people_from_boxes_filters_non_pedestrians():
    boxes = [
        BoundingBox(0, 0, 10, 20, 0.9, "person", track_id=1),
        BoundingBox(0, 0, 10, 20, 0.9, "wheelchair", track_id=2),
    ]
    assert [p.track_id for p in people_from_boxes(boxes)] == [1]


def test_adapter_does_not_mutate_original_box():
    """FallTracker가 Person.track_id를 덮어써도 원본 BoundingBox는 그대로여야 한다.

    같은 객체를 공유하면 목표 2의 ID 보정이 목표 1의 구역/속도 판정을 오염시킨다.
    """
    box = BoundingBox(0, 0, 10, 20, 0.9, "person", track_id=7)
    person = person_from_box(box)
    person.track_id = -1

    assert box.track_id == 7


def test_roi_from_ratio_scales_to_frame():
    roi = roi_from_ratio((480, 640, 3), (0.0, 0.5, 1.0, 1.0))
    assert roi == (0, 240, 640, 480)


def test_roi_from_zones_covers_calibrated_quad():
    """캘리브레이션된 사다리꼴을 감싸는 사각형이 나온다(원문 로직이 축평행 사각형 전제)."""
    quad = [(80, 460), (560, 460), (400, 150), (240, 150)]
    zones = CrosswalkZones.from_quad(quad, n=5, width_cm=400, length_cm=1000)

    x1, y1, x2, y2 = roi_from_zones(zones)

    assert (x1, y1) == (80, 150)
    assert (x2, y2) == (560, 460)


def test_lying_person_is_fall_candidate_by_aspect_ratio():
    """키포인트가 없으면 bbox 가로/세로 비율로 폴백 판정한다."""
    assert looks_fallen(lying(), CFG) is True
    assert looks_fallen(standing(), CFG) is False


def test_torso_angle_needs_confident_keypoints():
    """어깨/엉덩이 신뢰도가 낮으면 각도를 내지 않는다(None -> 비율 폴백)."""
    kp = np.zeros((17, 3))
    kp[[5, 6, 11, 12], 2] = 0.1
    assert torso_angle_deg(kp) is None

    kp[[5, 6], :2] = [100, 100]
    kp[[11, 12], :2] = [200, 100]
    kp[[5, 6, 11, 12], 2] = 0.9
    assert torso_angle_deg(kp) == pytest.approx(90.0)


def test_bbox_overlap_uses_smaller_area_not_iou():
    """넘어질 때 bbox 넓이가 크게 변해도 견디도록 IoU가 아니라 '교집합/작은쪽'을 쓴다."""
    tall = (0, 0, 40, 120)
    wide = (0, 0, 120, 40)
    assert bbox_overlap(tall, wide) == pytest.approx(1600 / 4800)


def test_fall_confirmed_only_after_confirm_sec():
    """쓰러진 자세가 fall_confirm_sec(3초) 유지돼야 사이렌이 확정된다."""
    monitor = FallMonitor(CFG)

    assert monitor.update(True, now=0.0, gap_sec=1.0) is False
    assert monitor.update(True, now=2.9, gap_sec=1.0) is False
    assert monitor.update(True, now=3.0, gap_sec=1.0) is True


def test_getting_up_within_confirm_sec_does_not_trigger():
    """3초 안에 일어나면 오탐으로 보고 사이렌을 울리지 않는다."""
    monitor = FallMonitor(CFG)

    monitor.update(True, now=0.0, gap_sec=1.0)
    monitor.update(True, now=1.5, gap_sec=1.0)
    monitor.update(False, now=3.0, gap_sec=1.0)
    assert monitor.update(True, now=4.0, gap_sec=1.0) is False


def test_short_detection_gap_does_not_reset_candidate():
    """저 fps에서 한 프레임 깜빡이는 미검출은 후보 카운트를 리셋하지 않는다."""
    monitor = FallMonitor(CFG)

    monitor.update(True, now=0.0, gap_sec=1.0)
    monitor.update(False, now=0.5, gap_sec=1.0)
    assert monitor.update(True, now=3.0, gap_sec=1.0) is True


def test_siren_needs_sustained_clear_to_release():
    """확정 후에는 정상 자세가 fall_clear_sec(3초) 연속돼야 사이렌이 꺼진다."""
    monitor = FallMonitor(CFG)
    for t in (0.0, 1.5, 3.0):
        monitor.update(True, now=t, gap_sec=1.0)
    assert monitor.confirmed is True

    assert monitor.update(False, now=4.0, gap_sec=1.0) is True
    assert monitor.update(False, now=7.0, gap_sec=1.0) is False


def test_two_people_do_not_accumulate_together():
    """서로 다른 사람의 짧은 이상 자세가 합산돼 사이렌이 울리면 안 된다."""
    tracker = FallTracker(CFG)

    tracker.update([lying(x=0, track_id=1)], [True], [True], now=0.0)
    tracker.update([lying(x=0, track_id=1)], [True], [True], now=1.6)
    confirmed = tracker.update([lying(x=400, track_id=2)], [True], [True], now=3.2)

    assert confirmed == set()


def test_state_is_inherited_when_tracker_drops_id():
    """낙상 순간 트래커가 ID를 놓쳐도(None) 위치 겹침으로 누적이 이어진다.

    UR Fall 실측에서 확인된 실패 모드 — 이 보정이 없으면 낙상 구간만 쏙 빠져
    영영 확정되지 않는다.
    """
    tracker = FallTracker(CFG)

    tracker.update([lying(track_id=1)], [True], [True], now=0.0)
    tracker.update([lying(track_id=None)], [True], [True], now=1.5)
    confirmed = tracker.update([lying(track_id=None)], [True], [True], now=3.0)

    assert confirmed != set()


def test_pipeline_confirms_fall_from_bounding_boxes():
    """BoundingBox 목록만 넣으면 쓰러짐 확정까지 나온다(원문 main() 루프와 같은 순서)."""
    pipeline = FallDetectionPipeline(roi_px=ROI, cfg=CFG)
    box = BoundingBox(100, 100, 220, 140, 0.9, "person", track_id=1)

    assert pipeline.update([box], now=0.0)["fall_confirmed"] is False
    assert pipeline.update([box], now=1.5)["fall_confirmed"] is False
    result = pipeline.update([box], now=3.0)

    assert result["fall_confirmed"] is True
    assert result["confirmed_ids"] == {1}
    assert result["fallen_flags"] == [True]


def test_pipeline_ignores_standing_person():
    pipeline = FallDetectionPipeline(roi_px=ROI, cfg=CFG)
    box = BoundingBox(100, 100, 140, 220, 0.9, "person", track_id=1)

    for t in (0.0, 1.5, 3.0, 5.0):
        result = pipeline.update([box], now=t)

    assert result["fall_confirmed"] is False
    assert result["fallen_flags"] == [False]


def test_pipeline_reports_foot_in_roi_separately():
    """신호 연장용 '발 위치' 판정과 쓰러짐용 '몸 전체 겹침'은 별개 기준이다."""
    pipeline = FallDetectionPipeline(roi_px=(0, 0, 200, 200), cfg=CFG)
    inside = BoundingBox(50, 50, 90, 170, 0.9, "person", track_id=1)
    outside = BoundingBox(50, 300, 90, 420, 0.9, "person", track_id=2)

    result = pipeline.update([inside, outside], now=0.0)

    assert result["in_roi_flags"] == [True, False]


def test_fallen_person_whose_feet_left_roi_still_triggers_siren():
    """쓰러져 발이 ROI 밖으로 튀어나가도 몸이 겹쳐 있으면 사이렌이 울려야 한다.

    원문이 두 기준을 나눈 이유가 정확히 이것이다. 넘어지면 bbox 하단(=발 위치)이
    자세에 따라 크게 튀는데, 발 한 점으로 쓰러짐까지 판정하면 하필 쓰러진 순간에
    ROI 밖으로 빠져 사이렌을 놓친다. 몸 전체 겹침은 자세가 변해도 안 튄다.
    """
    pipeline = FallDetectionPipeline(roi_px=(0, 0, 200, 200), cfg=CFG)
    straddling = BoundingBox(50, 150, 170, 210, 0.9, "person", track_id=1)

    first = pipeline.update([straddling], now=0.0)
    assert first["in_roi_flags"] == [False]
    assert first["fallen_flags"] == [True]

    pipeline.update([straddling], now=1.5)
    result = pipeline.update([straddling], now=3.0)

    assert result["fall_confirmed"] is True


def test_pipeline_reset_clears_siren():
    """수동 리셋(원문 'r' 키)으로 사이렌 상태를 초기화할 수 있다."""
    pipeline = FallDetectionPipeline(roi_px=ROI, cfg=CFG)
    box = BoundingBox(100, 100, 220, 140, 0.9, "person", track_id=1)
    for t in (0.0, 1.5, 3.0):
        pipeline.update([box], now=t)
    assert pipeline.update([box], now=3.1)["fall_confirmed"] is True

    pipeline.reset()

    assert pipeline.update([box], now=3.2)["fall_confirmed"] is False
