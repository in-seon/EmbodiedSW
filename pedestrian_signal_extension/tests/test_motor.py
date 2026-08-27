"""MotorGate — '언제 모터를 돌리고 멈출까'의 판단 로직.

GPIO는 여기서 다루지 않는다(하드웨어 확인은 tools/manual_motor_check.py).
이 파일이 지키는 것은 **정지 조건**이다. 그게 이 기능에서 유일하게 틀리기 쉬운 부분이다.
"""

import pytest

from config import config
from src.motor import (
    STOP_COMMAND,
    STOP_CROSSED,
    STOP_TIMEOUT,
    MotorGate,
    NullMotor,
)


def test_idle_until_start():
    """START를 받기 전에는 보행자가 보이든 말든 돌지 않는다."""
    gate = MotorGate()
    assert gate.running is False
    gate.update(occupied=True, now=1.0)
    assert gate.running is False


def test_start_runs_with_default_mode():
    gate = MotorGate()
    assert gate.start(now=0.0) is True
    assert gate.running is True
    assert gate.mode == config.MOTOR_DEFAULT_MODE


def test_start_accepts_explicit_mode():
    """START 3 = 교통약자 속도."""
    gate = MotorGate()
    gate.start(mode=3, now=0.0)
    assert gate.mode == 3


def test_does_not_stop_before_seeing_anyone():
    """★ 이 기능에서 가장 틀리기 쉬운 지점.

    START가 오는 순간 모형은 아직 횡단보도 **밖**이라 확정 보행자가 0명이다.
    "0명이면 정지"로 짜면 출발도 못 하고 멈춘다. 정지 조건은 '없음'이 아니라
    **'있었다가 없어짐'**이어야 한다.
    """
    gate = MotorGate()
    gate.start(now=0.0)
    for t in (0.5, 1.0, 1.5, 2.0):
        assert gate.update(occupied=False, now=t) is False
        assert gate.running is True, f"{t}초에 멈췄다 — 모형이 출발조차 못 한다"


def test_stops_after_pedestrian_leaves():
    gate = MotorGate()
    gate.start(now=0.0)
    gate.update(occupied=False, now=0.5)   # 아직 진입 전
    gate.update(occupied=True, now=1.0)    # 진입
    gate.update(occupied=True, now=1.5)    # 건너는 중
    assert gate.running is True

    assert gate.update(occupied=False, now=2.0) is True   # 빠져나감
    assert gate.running is False
    assert gate.last_stop_reason == STOP_CROSSED


def test_stop_is_reported_only_once():
    """정지 사유를 매 프레임 로그에 찍지 않도록, 실제로 멈춘 그 호출만 True를 돌려준다."""
    gate = MotorGate()
    gate.start(now=0.0)
    gate.update(occupied=True, now=1.0)
    assert gate.update(occupied=False, now=1.5) is True
    assert gate.update(occupied=False, now=2.0) is False


def test_safety_timeout_stops_even_if_never_empty():
    """모형이 탈선하거나 검출이 계속 붙어 있어도 언젠가는 멈춘다.

    이 안전장치가 없으면 정지 조건이 영영 오지 않아 모터가 계속 돌고, 물리 장치라
    조용한 실패로 끝나지 않는다.
    """
    gate = MotorGate(max_run_sec=5.0)
    gate.start(now=0.0)
    for t in (1.0, 2.0, 3.0, 4.0):
        assert gate.update(occupied=True, now=t) is False

    assert gate.update(occupied=True, now=5.0) is True
    assert gate.last_stop_reason == STOP_TIMEOUT


def test_repeated_start_does_not_reset_timeout():
    """START가 중복으로 오거나 재전송돼도 안전 타임아웃 시계는 리셋되지 않는다.

    리셋되면 START가 계속 오는 동안 타임아웃이 영영 발동하지 않아 안전장치가 무력해진다.
    """
    gate = MotorGate(max_run_sec=5.0)
    gate.start(now=0.0)
    assert gate.start(now=3.0) is False       # 이미 돌고 있다 -> 무시
    assert gate.update(occupied=True, now=5.0) is True
    assert gate.last_stop_reason == STOP_TIMEOUT


def test_explicit_stop_command():
    gate = MotorGate()
    gate.start(now=0.0)
    assert gate.stop() is True
    assert gate.running is False
    assert gate.last_stop_reason == STOP_COMMAND
    assert gate.stop() is False               # 이미 멈춰 있다


def test_restart_clears_seen_latch():
    """다음 사이클은 '아무도 못 본' 상태에서 시작해야 한다.

    래치가 남아 있으면 새 녹색이 시작되자마자(아직 모형이 밖일 때) 즉시 멈춘다 —
    첫 사이클만 되고 두 번째부터 안 되는, 시연에서 제일 곤란한 형태의 버그다.
    """
    gate = MotorGate()
    gate.start(now=0.0)
    gate.update(occupied=True, now=1.0)
    gate.update(occupied=False, now=2.0)      # 1차 횡단 완료
    assert gate.running is False

    gate.start(now=10.0)                      # 다음 녹색
    assert gate.seen_pedestrian is False
    assert gate.update(occupied=False, now=10.5) is False
    assert gate.running is True


def test_null_motor_has_the_same_surface():
    """--no-motor 가 AttributeError로 죽지 않도록 표면을 맞춰 둔다."""
    motor = NullMotor()
    motor.start()
    assert motor.mode == config.MOTOR_DEFAULT_MODE
    motor.start(3)
    assert motor.mode == 3
    motor.stop()
    motor.close()
    with NullMotor():
        pass


def test_unknown_mode_is_rejected_by_hardware_class():
    """오타난 모드를 조용히 기본값으로 돌리지 않는다 — 속도가 틀린 채로 시연된다."""
    from src.motor import StepperMotor

    with pytest.raises(ValueError, match="속도 모드"):
        StepperMotor().start(mode=99)


# --- 일시정지: 쓰러짐 연출 중에 모형을 세워 둔다 ---
#
# 모형에는 모터가 둘이다 — 끌고 가는 스텝모터와 발을 넘어뜨리는 서보.
# 넘어진 모형이 계속 끌려가면 안 되지만, 일어난 뒤에는 마저 건너야 한다.

def test_pause_keeps_the_run_alive():
    """★ 일시정지는 정지가 아니다 — START를 다시 받지 않아도 재개된다."""
    gate = MotorGate()
    gate.start(now=0.0)
    gate.update(occupied=True, now=1.0)

    assert gate.pause(now=2.0) is True
    assert gate.paused is True
    assert gate.moving is False
    assert gate.running is True          # 횡단은 아직 끝나지 않았다

    assert gate.resume(now=7.0) is True
    assert gate.moving is True


def test_pause_does_not_lose_the_seen_latch():
    """stop()을 쓰면 래치가 지워져, 일어난 뒤의 이동이 새 사이클로 오해된다."""
    gate = MotorGate()
    gate.start(now=0.0)
    gate.update(occupied=True, now=1.0)
    gate.pause(now=2.0)
    gate.resume(now=7.0)
    assert gate.seen_pedestrian is True

    # 그리고 정지 판단은 그대로 살아 있다.
    assert gate.update(occupied=False, now=8.0) is True
    assert gate.last_stop_reason == STOP_CROSSED


def test_paused_motor_does_not_stop_when_detection_drops():
    """★ 누우면 bbox 모양이 급변해 트래커가 놓치는 것이 정상이다.

    그걸 '건너갔다'로 읽어 영구 정지시키면 일어나도 다시 안 움직인다 —
    데모가 거기서 끝난다.
    """
    gate = MotorGate()
    gate.start(now=0.0)
    gate.update(occupied=True, now=1.0)
    gate.pause(now=2.0)

    for t in (2.5, 3.0, 3.5, 4.0):
        assert gate.update(occupied=False, now=t) is False
        assert gate.running is True

    gate.resume(now=5.0)
    assert gate.moving is True


def test_paused_time_does_not_count_toward_the_safety_timeout():
    """연출로 세워 둔 시간이 주행 시간으로 잡히면, 긴 연출 뒤 곧바로 타임아웃이 터진다."""
    gate = MotorGate(max_run_sec=10.0)
    gate.start(now=0.0)
    gate.update(occupied=True, now=1.0)

    gate.pause(now=2.0)
    gate.resume(now=100.0)               # 98초를 세워 뒀다

    assert gate.update(occupied=True, now=101.0) is False   # 실주행은 3초뿐
    assert gate.running is True


def test_pause_is_idempotent():
    """중복 pause가 재개 시각 계산을 망가뜨리지 않는다."""
    gate = MotorGate(max_run_sec=10.0)
    gate.start(now=0.0)
    gate.pause(now=1.0)
    assert gate.pause(now=5.0) is False   # 두 번째는 무시 — 1.0이 기준으로 남는다
    gate.resume(now=6.0)
    assert gate.update(occupied=True, now=10.0) is False   # 실주행 5초


def test_resume_without_pause_does_nothing():
    gate = MotorGate()
    gate.start(now=0.0)
    assert gate.resume(now=1.0) is False


def test_pause_before_start_does_nothing():
    """START 전에는 세울 것이 없다 — 나중에 START가 와도 멈춘 채로 시작하면 안 된다."""
    gate = MotorGate()
    assert gate.pause(now=1.0) is False
    gate.start(now=2.0)
    assert gate.moving is True


def test_stop_clears_pause():
    """일시정지 상태에서 STOP이 오면 그대로 끝난다 — 다음 START가 멈춘 채로 시작하면 안 된다."""
    gate = MotorGate()
    gate.start(now=0.0)
    gate.pause(now=1.0)
    gate.stop()

    gate.start(now=10.0)
    assert gate.paused is False
    assert gate.moving is True
