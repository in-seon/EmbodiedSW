"""FallAlarmPipeline 단위 테스트 — 카메라·YOLO·아두이노 없이 배선을 검증한다.

여기서 확인하는 것은 쓰러짐 판정 로직이 아니라(그건 test_fall_detection.py),
**검출 결과가 부저 명령까지 제대로 이어지는가**다.
"""

import numpy as np
import pytest

from src.detection import BoundingBox
from src.pipeline import FallAlarmPipeline

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
# FALL_CONFIG["crosswalk_roi"] = (0.15, 0.30, 0.85, 0.95) -> 640x480에서 (96,144)~(544,456)
ROI = (96, 144, 544, 456)


class FakeDetector:
    """프레임마다 미리 정해둔 박스 목록을 돌려준다."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.calls = 0

    def detect(self, frame):
        boxes = self.frames[min(self.calls, len(self.frames) - 1)]
        self.calls += 1
        return boxes


class SpySerial:
    """update_alarm 호출을 기록한다."""

    def __init__(self):
        self.commands = []
        self.alarm_on = False
        self.closed = False

    def update_alarm(self, active, now=None):
        if active and not self.alarm_on:
            self.alarm_on = True
            self.commands.append("ALERT")
            return "ALERT"
        if not active and self.alarm_on:
            self.alarm_on = False
            self.commands.append("STOP")
            return "STOP"
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True


def standing(track_id=1):
    """ROI 안에 서 있는 사람 — 세로로 긴 박스."""
    return BoundingBox(x1=300, y1=200, x2=340, y2=400,
                       confidence=0.9, label="person", track_id=track_id)


def fallen(track_id=1):
    """ROI 안에 쓰러진 사람 — 가로로 긴 박스(가로/세로 >= fall_aspect_ratio 1.3)."""
    return BoundingBox(x1=250, y1=380, x2=450, y2=440,
                       confidence=0.9, label="person", track_id=track_id)


def build(frames, serial=None):
    spy = serial if serial is not None else SpySerial()
    pipeline = FallAlarmPipeline(
        camera=object(),                 # run()을 쓰지 않으므로 필요 없다
        detector=FakeDetector(frames),
        serial_comm=spy,
        roi_px=ROI,
    )
    return pipeline, spy


def test_standing_person_does_not_trigger_alarm():
    pipeline, spy = build([[standing()]])
    result = pipeline.process_frame(FRAME, now=0.0)

    assert result.fall_confirmed is False
    assert result.people_count == 1
    assert spy.commands == []


def test_fall_must_persist_before_alarm():
    """fall_confirm_sec(3초)를 채우기 전에는 부저를 울리지 않는다 — 오탐 방지."""
    pipeline, spy = build([[fallen()]])

    assert pipeline.process_frame(FRAME, now=0.0).fall_confirmed is False
    assert pipeline.process_frame(FRAME, now=1.0).fall_confirmed is False
    assert spy.commands == []


def test_sustained_fall_sends_alert_once():
    pipeline, spy = build([[fallen()]])

    for t in (0.0, 1.0, 2.0, 3.0, 3.5):
        result = pipeline.process_frame(FRAME, now=t)

    assert result.fall_confirmed is True
    assert spy.commands == ["ALERT"]      # 매 프레임 보내지 않는다


def test_getting_up_sends_stop():
    """확정 후 정상 자세가 fall_clear_sec(3초) 이어지면 해제된다."""
    frames = [[fallen()]] * 5 + [[standing()]] * 6
    pipeline, spy = build(frames)

    times = [0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    for t in times:
        pipeline.process_frame(FRAME, now=t)

    assert spy.commands == ["ALERT", "STOP"]


def test_alarm_survives_brief_detection_gap():
    """쓰러진 순간은 오히려 검출이 잘 안 된다 — 한 프레임 빠졌다고 풀리면 안 된다."""
    frames = [[fallen()]] * 5 + [[]] + [[fallen()]] * 2
    pipeline, spy = build(frames)

    for t in (0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0):
        result = pipeline.process_frame(FRAME, now=t)

    assert result.fall_confirmed is True
    assert spy.commands == ["ALERT"]      # 중간에 STOP이 끼면 안 된다


def test_reset_alarm_clears_state_and_buzzer():
    pipeline, spy = build([[fallen()]])
    for t in (0.0, 1.0, 2.0, 3.0, 3.5):
        pipeline.process_frame(FRAME, now=t)
    assert spy.commands == ["ALERT"]

    pipeline.reset_alarm()
    assert spy.commands == ["ALERT", "STOP"]
    # 누적도 지워졌으므로 다시 3초를 채워야 한다.
    assert pipeline.process_frame(FRAME, now=4.0).fall_confirmed is False


def test_roi_falls_back_to_frame_ratio():
    """roi_px도 zones도 없으면 첫 프레임 크기로 화면 비율 ROI를 만든다."""
    pipeline = FallAlarmPipeline(
        camera=object(), detector=FakeDetector([[standing()]]),
        serial_comm=SpySerial(),
    )
    assert pipeline.roi_px is None
    pipeline.process_frame(FRAME, now=0.0)
    assert pipeline.roi_px == ROI


def test_detector_runs_once_per_frame():
    """추론이 프레임 시간의 사실상 전부다 — 프레임당 한 번만 돌아야 한다."""
    pipeline, _ = build([[standing()]])
    for t in (0.0, 1.0, 2.0):
        pipeline.process_frame(FRAME, now=t)
    assert pipeline.detector.calls == 3
