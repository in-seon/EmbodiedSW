"""[수동] 모형 보행자 스텝모터 확인 스크립트 (자동 테스트 아님).

카메라도 아두이노도 없이 **모터만** 돌린다. main.py에서 모터가 안 도는 이유가
배선인지, 시리얼인지, 판단 로직인지를 가르는 용도다.

사용법:
    python tools/manual_motor_check.py                # 모드 2로 10초 회전
    python tools/manual_motor_check.py --mode 3       # 교통약자 속도
    python tools/manual_motor_check.py --seconds 12   # 스톱워치용
    python tools/manual_motor_check.py --interactive  # 키로 직접 조작
    python tools/manual_motor_check.py --gate         # 정지 판단 로직만 (GPIO 불필요)

## ⚠️ 가장 중요한 확인: 30cm를 12초에 지나는가

config의 '축척과 모형 보행자 속도' 항목에 따르면 모형은 **2.5 cm/s**로 움직여야
연장 초의 근거 자료(광주시 5초, 서울시 3~6초, 강진군 5~10초)를 그대로 쓸 수 있다.

    30cm / 2.5cm/s = 12초  ==  실제 12m / 1.0m/s = 12초

--seconds 12 로 돌리고 이동 거리를 재서 30cm가 나오는지 본다. 안 맞으면
config.MOTOR_STEP_DELAY_SEC 를 조정한다 (값이 클수록 느리다).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import config
from src.motor import MotorGate, StepperMotor


def run_once(motor, mode, seconds):
    print(f"[1] 모드 {mode}로 기동합니다 (스텝 간격 {config.MOTOR_STEP_DELAY_SEC[mode]*1000:.2f}ms)")
    print(f"    {seconds:.0f}초간 돕니다. 이동 거리를 재세요.\n")
    motor.start(mode)

    start = time.monotonic()
    while time.monotonic() - start < seconds:
        elapsed = time.monotonic() - start
        print(f"\r    {elapsed:5.1f}초 경과", end="", flush=True)
        time.sleep(0.1)

    motor.stop()
    print(f"\n\n[2] 정지했습니다.")
    print(f"    {seconds:.0f}초 동안 이동한 거리가 {seconds * 2.5:.0f}cm 근처면 축척이 맞습니다.")
    print("    (2.5 cm/s 기준 — 자세한 근거는 config.py '축척과 모형 보행자 속도')")


def run_interactive(motor):
    print("명령: 2=일반 속도  3=교통약자 속도  s=정지  q=종료")
    print()
    print("  * 돌고 있는 중에 2/3을 누르면 속도만 바뀝니다(멈추지 않습니다).")
    print("  * 모터가 뜨겁거나 소리만 나고 안 돌면 config.MOTOR_DUTY 를 조정하세요.")
    print("  * ENA/ENB 점퍼가 꽂혀 있으면 PWM이 무시됩니다. 제거했는지 확인하세요.")
    print()
    while True:
        try:
            key = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if key == "q":
            break
        elif key in ("2", "3"):
            mode = int(key)
            motor.start(mode)
            label = "일반" if mode == 2 else "교통약자"
            print(f"    -> 모드 {mode} ({label}), 스텝 간격 "
                  f"{config.MOTOR_STEP_DELAY_SEC[mode]*1000:.2f}ms")
        elif key == "s":
            motor.stop()
            print("    -> 정지 (코일 전원 차단)")
        else:
            print("    2 / 3 / s / q 중에서 입력하세요.")


def run_gate_only():
    """GPIO 없이 정지 판단만 재현한다. 개발 PC에서도 돌아간다.

    실제로 겪었던 함정을 눈으로 보여준다: START 직후에는 확정 보행자가 0명이므로
    '없으면 정지'로 짜면 출발도 못 하고 멈춘다.
    """
    print("정지 판단 로직만 재현합니다 (GPIO를 쓰지 않습니다).\n")
    gate = MotorGate()
    now = 0.0

    # (설명, 확정 보행자 있음?)
    script = [
        ("START 직후 — 모형은 아직 횡단보도 밖", False),
        ("아직 밖", False),
        ("모형이 횡단보도에 진입", True),
        ("건너는 중", True),
        ("건너는 중", True),
        ("반대편으로 빠져나감", False),
        ("계속 비어 있음", False),
    ]

    gate.start(now=now)
    print(f"  {now:5.1f}s  START            -> 모터 기동 (모드 {gate.mode})")
    for label, occupied in script:
        now += 0.5
        stopped = gate.update(occupied, now=now)
        mark = "돌고 있음" if gate.running else "정지"
        note = f"  <- {gate.last_stop_reason}" if stopped else ""
        seen = "본 적 있음" if gate.seen_pedestrian else "아직 못 봄"
        print(f"  {now:5.1f}s  보행자 {'있음' if occupied else '없음'} / {seen}"
              f"  -> {mark}{note}")

    print("\n핵심: 처음 두 프레임은 보행자가 0명인데도 멈추지 않았습니다.")
    print("      '없음'이 아니라 '있었다가 없어짐'을 기다리기 때문입니다.")
    print("      이 래치가 없으면 START 직후 바로 멈춰 모형이 출발조차 못 합니다.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", type=int, default=config.MOTOR_DEFAULT_MODE,
                        choices=sorted(config.MOTOR_STEP_DELAY_SEC),
                        help="속도 모드. 2=일반, 3=교통약자.")
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="회전 시간(초). 축척 확인은 12초를 권장합니다.")
    parser.add_argument("--interactive", action="store_true",
                        help="키로 속도를 바꿔 가며 직접 조작한다.")
    parser.add_argument("--gate", action="store_true",
                        help="정지 판단 로직만 재현한다 (GPIO 불필요 — 개발 PC에서도 동작).")
    args = parser.parse_args()

    if args.gate:
        run_gate_only()
        return

    try:
        motor = StepperMotor()
        motor.start(args.mode)   # 여기서 GPIO를 잡는다 — 실패하면 아래에서 잡힌다
        motor.stop()
    except RuntimeError as exc:
        print(f"모터를 쓸 수 없습니다:\n  {exc}\n")
        print("판단 로직만 보려면 --gate 를 쓰세요.")
        return

    try:
        if args.interactive:
            run_interactive(motor)
        else:
            run_once(motor, args.mode, args.seconds)
    except KeyboardInterrupt:
        print("\n중단됨 (Ctrl+C).")
    finally:
        # close()가 코일 전원까지 끊는다. 켜 둔 채 끝나면 모터가 계속 뜨거워진다.
        motor.close()


if __name__ == "__main__":
    main()
