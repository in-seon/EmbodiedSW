"""FallScheduler — '몇 초 뒤에 눕히고, 원하면 다시 세운다'의 타이밍.

GPIO는 다루지 않는다. 서보를 가짜로 주입해 **언제 눕히는지**만 본다.
"""

from src.servo import FELL, STOOD, FallScheduler


class FakeServo:
    """FallServo 자리에 들어가 호출만 기록한다."""

    def __init__(self):
        self.calls = []
        self.fallen = False

    def fall(self):
        self.calls.append("fall")
        self.fallen = True

    def stand(self):
        self.calls.append("stand")
        self.fallen = False

    def close(self):
        self.calls.append("close")


def test_does_nothing_before_the_delay():
    servo = FakeServo()
    scheduler = FallScheduler(servo, fall_after_sec=5.0)

    scheduler.tick(100.0)          # 첫 tick이 기준 시각을 잡는다
    scheduler.tick(102.0)
    scheduler.tick(104.9)
    assert servo.calls == []


def test_falls_after_the_delay():
    servo = FakeServo()
    scheduler = FallScheduler(servo, fall_after_sec=5.0)

    scheduler.tick(100.0)
    event = scheduler.tick(105.0)
    assert servo.calls == ["fall"]
    # 종류로 알린다 — 호출자가 이 시점에 구동 스텝모터를 세워야 한다.
    # 문구를 파싱하게 두면 문구를 고치는 순간 조용히 동작이 바뀐다.
    assert event is not None and event.kind == FELL


def test_first_tick_sets_the_clock_not_the_start_of_the_process():
    """기준은 '프로세스 시작'이 아니라 '첫 프레임'이다.

    모델 로딩에 몇 초가 걸리므로 프로세스 시작을 기준으로 하면, 카메라가 첫 프레임을
    내놓기도 전에 모형이 눕는다. 실제로 YOLO 첫 추론은 수 초가 걸린다.
    """
    servo = FakeServo()
    scheduler = FallScheduler(servo, fall_after_sec=3.0)

    scheduler.tick(1000.0)         # 첫 프레임이 늦게 도착했다
    assert servo.calls == []
    scheduler.tick(1002.9)
    assert servo.calls == []
    scheduler.tick(1003.0)
    assert servo.calls == ["fall"]


def test_falls_only_once():
    """매 프레임 fall()을 다시 부르면 서보가 계속 떨어 검출이 흔들린다."""
    servo = FakeServo()
    scheduler = FallScheduler(servo, fall_after_sec=1.0)

    scheduler.tick(0.0)
    scheduler.tick(1.0)
    scheduler.tick(2.0)
    scheduler.tick(3.0)
    assert servo.calls == ["fall"]


def test_stays_down_without_hold():
    """hold_sec이 없으면 종료할 때까지 누워 있는다."""
    servo = FakeServo()
    scheduler = FallScheduler(servo, fall_after_sec=1.0, hold_sec=None)

    scheduler.tick(0.0)
    scheduler.tick(1.0)
    scheduler.tick(60.0)
    assert servo.calls == ["fall"]
    assert scheduler.done is False


def test_stands_back_up_after_hold():
    """일어나는 것도 시연 대상이다 — 부저가 꺼지는지 확인해야 하니까."""
    servo = FakeServo()
    scheduler = FallScheduler(servo, fall_after_sec=1.0, hold_sec=4.0)

    scheduler.tick(0.0)
    scheduler.tick(1.0)            # 눕힘
    assert scheduler.tick(4.9) is None
    event = scheduler.tick(5.0)    # 눕힌 시각(1.0)으로부터 4초
    assert servo.calls == ["fall", "stand"]
    assert event is not None and event.kind == STOOD
    assert scheduler.done is True


def test_nothing_happens_after_done():
    servo = FakeServo()
    scheduler = FallScheduler(servo, fall_after_sec=1.0, hold_sec=1.0)

    scheduler.tick(0.0)
    scheduler.tick(1.0)
    scheduler.tick(2.0)
    servo.calls.clear()
    scheduler.tick(3.0)
    scheduler.tick(10.0)
    assert servo.calls == []


def test_hold_is_measured_from_the_fall_not_from_the_start():
    """눕힌 뒤 N초여야 한다. 시작 기준이면 hold < fall_after 일 때 즉시 일어난다."""
    servo = FakeServo()
    scheduler = FallScheduler(servo, fall_after_sec=10.0, hold_sec=2.0)

    scheduler.tick(0.0)
    scheduler.tick(10.0)           # 눕힘
    assert servo.calls == ["fall"]
    scheduler.tick(11.0)
    assert servo.calls == ["fall"]  # 아직 1초밖에 안 지났다
    scheduler.tick(12.0)
    assert servo.calls == ["fall", "stand"]
