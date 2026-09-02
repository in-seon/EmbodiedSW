"""FallAlarmPipeline 단위 테스트 — 카메라·YOLO·아두이노 없이 배선을 검증."""

import numpy as np
import pytest

from config import config
from src.detection import BoundingBox
from src.fall_detection import FallDetectionPipeline
from src.pipeline import FallAlarmPipeline
from src.serial_comm import STATE_FALL, STATE_NORMAL

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
ROI = (96, 144, 544, 456)

CFG = dict(config.FALL_CONFIG, fall_confirm_sec=3.0, fall_clear_sec=3.0)


class FakeDetector:
    def __init__(self, frames):
        self.frames = list(frames)
        self.calls = 0

    def detect(self, frame):
        boxes = self.frames[min(self.calls, len(self.frames) - 1)]
        self.calls += 1
        return boxes


class SpySerial:
    def __init__(self):
        self.commands = []
        self.state = None
        self.closed = False

    def update_state(self, state, extend_sec=None, now=None):
        if state == self.state:
            return None
        self.state = state
        self.commands.append(state)
        return state

    def send_state(self, state, extend_sec=None, now=None):
        self.state = state
        self.commands.append(state)
        return state

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True


def standing(track_id=1):
    return BoundingBox(x1=300, y1=200, x2=340, y2=400,
                       confidence=0.9, label="person", track_id=track_id)


def fallen(track_id=1):
    return BoundingBox(x1=250, y1=380, x2=450, y2=440,
                       confidence=0.9, label="person", track_id=track_id)


def build(frames, serial=None):
    spy = serial if serial is not None else SpySerial()
    pipeline = FallAlarmPipeline(
        camera=object(),
        detector=FakeDetector(frames),
        serial_comm=spy,
        roi_px=ROI,
        fall_detector=FallDetectionPipeline(ROI, cfg=CFG),
    )
    return pipeline, spy


def test_standing_person_does_not_trigger_alarm():
    pipeline, spy = build([[standing()]])
    result = pipeline.process_frame(FRAME, now=0.0)

    assert result.fall_confirmed is False
    assert result.people_count == 1
    assert spy.commands == [STATE_NORMAL]


def test_fall_must_persist_before_alarm():
    pipeline, spy = build([[fallen()]])

    assert pipeline.process_frame(FRAME, now=0.0).fall_confirmed is False
    assert pipeline.process_frame(FRAME, now=1.0).fall_confirmed is False
    assert spy.commands == [STATE_NORMAL]


def test_sustained_fall_sends_alert_once():
    pipeline, spy = build([[fallen()]])

    for t in (0.0, 1.0, 2.0, 3.0, 3.5):
        result = pipeline.process_frame(FRAME, now=t)

    assert result.fall_confirmed is True
    assert spy.commands == [STATE_NORMAL, STATE_FALL]


def test_getting_up_sends_stop():
    frames = [[fallen()]] * 5 + [[standing()]] * 6
    pipeline, spy = build(frames)

    times = [0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    for t in times:
        pipeline.process_frame(FRAME, now=t)

    assert spy.commands == [STATE_NORMAL, STATE_FALL, STATE_NORMAL]


def test_alarm_survives_brief_detection_gap():
    frames = [[fallen()]] * 5 + [[]] + [[fallen()]] * 2
    pipeline, spy = build(frames)

    for t in (0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0):
        result = pipeline.process_frame(FRAME, now=t)

    assert result.fall_confirmed is True
    assert spy.commands == [STATE_NORMAL, STATE_FALL]


def test_reset_alarm_clears_state_and_buzzer():
    pipeline, spy = build([[fallen()]])
    for t in (0.0, 1.0, 2.0, 3.0, 3.5):
        pipeline.process_frame(FRAME, now=t)
    assert spy.commands == [STATE_NORMAL, STATE_FALL]

    pipeline.reset_alarm()
    assert spy.commands == [STATE_NORMAL, STATE_FALL, STATE_NORMAL]
    assert pipeline.process_frame(FRAME, now=4.0).fall_confirmed is False


def test_roi_falls_back_to_frame_ratio():
    pipeline = FallAlarmPipeline(
        camera=object(), detector=FakeDetector([[standing()]]),
        serial_comm=SpySerial(),
    )
    assert pipeline.roi_px is None
    pipeline.process_frame(FRAME, now=0.0)
    assert pipeline.roi_px == ROI


def test_detector_runs_once_per_frame():
    pipeline, _ = build([[standing()]])
    for t in (0.0, 1.0, 2.0):
        pipeline.process_frame(FRAME, now=t)
    assert pipeline.detector.calls == 3
