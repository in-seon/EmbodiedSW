"""
라즈베리파이5 + L298N + NEMA17 회전 테스트

배선:
  IN1 -> GPIO17    IN2 -> GPIO27    IN3 -> GPIO22    IN4 -> GPIO23
  ENA -> GPIO12 (PWM)   ENB -> GPIO13 (PWM)   (ENA/ENB 점퍼 제거 상태여야 함)
  코일 A(검정,초록) -> OUT1, OUT2
  코일 B(빨강,파랑) -> OUT3, OUT4
  GND: 전원(-), 라즈베리파이 GND, L298N GND 모두 브레드보드 공통 GND 레일에 연결

주의:
  - DUTY는 0.3부터 시작. 모터/L298N 방열판이 뜨거우면 낮추고,
    힘이 부족해 스텝을 건너뛰면 조금씩 올릴 것.
  - Ctrl+C로 정지하면 코일 전원이 꺼지도록 처리되어 있음.
"""

from gpiozero import OutputDevice, PWMOutputDevice
from time import sleep

# 방향 제어 핀
in1 = OutputDevice(17)
in2 = OutputDevice(27)
in3 = OutputDevice(22)
in4 = OutputDevice(23)
coils = [in1, in2, in3, in4]

# 속도/전류 제한용 PWM 핀
ena = PWMOutputDevice(12, frequency=1000)
enb = PWMOutputDevice(13, frequency=1000)

DUTY = 0.3          # 12V x 0.3 ≈ 평균 3.6V. 필요시 조정
STEP_DELAY = 0.005  # 값을 늘리면 더 천천히 회전

# 풀스텝 시퀀스 (in1, in2, in3, in4 순서)
SEQUENCE = [
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 0, 1],
]


def stop_all():
    for coil in coils:
        coil.value = False
    ena.value = 0
    enb.value = 0


if __name__ == "__main__":
    ena.value = DUTY
    enb.value = DUTY

    print("회전 시작. 멈추려면 Ctrl+C")
    try:
        while True:
            for step in SEQUENCE:
                for coil, val in zip(coils, step):
                    coil.value = bool(val)
                sleep(STEP_DELAY)
    except KeyboardInterrupt:
        print("정지")
    finally:
        stop_all()