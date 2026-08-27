"""시연용 서보모터 — 모형 사람을 원격으로 눕혀 쓰러짐 감지를 시연한다.

배선:
  신호(주황) -> GPIO18 (config.SERVO_PIN)
  전원(빨강) -> 5V,  GND(갈색) -> 공통 GND
  ⚠️ 서보 전원은 파이 5V 핀이 아니라 **외부 전원**에서 뽑는 편이 안전하다.
     기동 전류가 순간적으로 크면 파이가 리셋된다(모터와 함께 쓰면 특히).

## 이건 '감지 대상'이지 '감지 결과'가 아니다

서보는 쓰러짐이 감지됐을 때 반응하는 장치가 아니라, **감지될 사건을 만드는** 장치다.
그래서 파이프라인 안에 있지 않다. 파이프라인에 넣으면 "쓰러짐을 감지했으니 쓰러뜨린다"는
순환이 되고, 시연 도구가 판정 경로에 섞여 들어간다.

main.py가 타이머로 직접 돌리고, 파이프라인은 그 결과를 **아무것도 모르는 채로** 본다.
그래야 시연이 실제 검출 경로를 그대로 통과하는 진짜 확인이 된다.

## 왜 스레드가 없는가

스텝모터와 달리 서보는 각도만 지정하면 하드웨어 PWM이 알아서 유지한다. 각도 지정은
논블로킹이라 비전 루프를 잡지 않는다. 넘어지는 '속도'는 서보 자체의 기계적 속도로 정해진다.
"""

from dataclasses import dataclass

from config import config

# FallScheduler가 알리는 사건. 문자열 대신 종류를 붙여 보내는 이유: 호출자가 이 시점에
# **스텝모터를 세우고 다시 움직여야** 하기 때문이다. 안내 문구를 파싱하게 두면
# 문구를 고치는 순간 조용히 동작이 바뀐다.
FELL = "fell"
STOOD = "stood"


@dataclass
class FallEvent:
    kind: str        # FELL 또는 STOOD
    message: str     # 화면에 그대로 찍을 안내 문구


class FallServo:
    """모형 사람을 세우고 눕히는 서보.

    GPIO는 **처음 움직일 때** 잡는다(StepperMotor와 같은 이유 — import만으로 죽지 않게).
    """

    def __init__(self, pin=None, stand_angle=None, fall_angle=None):
        self.pin = config.SERVO_PIN if pin is None else pin
        self.stand_angle = (
            config.SERVO_STAND_ANGLE_DEG if stand_angle is None else stand_angle
        )
        self.fall_angle = (
            config.SERVO_FALL_ANGLE_DEG if fall_angle is None else fall_angle
        )
        self._servo = None
        self.fallen = False

    def _ensure_gpio(self):
        if self._servo is not None:
            return
        try:
            from gpiozero import AngularServo
        except ImportError as exc:
            raise RuntimeError(
                "gpiozero를 불러올 수 없습니다 — 서보는 라즈베리파이에서만 동작합니다.\n"
                "  개발 PC에서는 --fall-after 없이 실행하세요.\n"
                f"  ({exc})"
            ) from exc

        # 펄스폭을 넓히는 이유: gpiozero 기본값(1.0~2.0ms)은 SG90 계열에서 가동범위가
        # 90도 정도로 좁게 나온다. 눕히는 각도가 안 나오면 여기부터 의심할 것.
        self._servo = AngularServo(
            self.pin,
            min_angle=min(self.stand_angle, self.fall_angle),
            max_angle=max(self.stand_angle, self.fall_angle),
            min_pulse_width=config.SERVO_MIN_PULSE_WIDTH_SEC,
            max_pulse_width=config.SERVO_MAX_PULSE_WIDTH_SEC,
        )

    def stand(self):
        """세운다."""
        self._ensure_gpio()
        self._servo.angle = self.stand_angle
        self.fallen = False

    def fall(self):
        """눕힌다 — 이 순간부터 카메라에 '쓰러진 사람'이 보여야 한다."""
        self._ensure_gpio()
        self._servo.angle = self.fall_angle
        self.fallen = True

    def relax(self):
        """PWM을 끊어 힘을 뺀다.

        서보는 각도를 유지하려고 계속 미세하게 떨며 전류를 쓴다. 그 진동이 모형을
        흔들면 검출 박스가 같이 떨려 판정에 노이즈가 된다. 자세가 잡힌 뒤에는 놓아준다.
        """
        if self._servo is not None:
            self._servo.detach()

    def close(self):
        """세워 놓고 GPIO를 반납한다.

        눕힌 채로 끝내지 않는 이유: 다음 실행이 '이미 쓰러진 사람'에서 시작하면
        FALL이 즉시 확정돼 무엇을 시연하려던 것인지 알 수 없게 된다.
        """
        if self._servo is None:
            return
        try:
            self.stand()
            self._servo.close()
        except Exception:
            pass    # 종료 경로 — GPIO가 이미 사라졌어도 조용히 넘어간다.
        finally:
            self._servo = None
            self.fallen = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FallScheduler:
    """'N초 뒤에 눕히고, M초 뒤에 다시 세운다'를 매 프레임 호출로 진행한다.

    ## 이 타이밍에 스텝모터도 같이 멈춰야 한다

    모형에는 모터가 **둘** 달려 있다 — 횡단보도를 따라 끌고 가는 스텝모터와, 발에 붙어
    넘어뜨리는 이 서보다. 넘어진 모형을 계속 끌고 가면 데모가 우스워지므로, 눕히는
    순간 스텝모터를 세우고 일으키는 순간 이어서 움직여야 한다.

    그 조작을 여기서 직접 하지 않고 FallEvent로 알리기만 하는 이유: 이 클래스가
    스텝모터까지 쥐면 '연출 타이밍'과 '구동 제어'가 한 덩어리가 되어, 서보 없이
    모터만 확인하는 것도 그 반대도 못 하게 된다.

    타이머 스레드를 쓰지 않는 이유: 스레드로 하면 서보가 비전 루프와 **다른 시계**로
    움직여서, 로그의 "쓰러짐 확정" 시각과 "눕힌" 시각을 맞춰 보기 어렵다. 매 프레임
    tick()을 부르면 둘이 같은 시계를 쓰고, 지연도 프레임 단위로 정직하게 드러난다.
    (해상도는 프레임 간격만큼 거칠지만 시연 지연으로는 충분하다.)
    """

    def __init__(self, servo, fall_after_sec, hold_sec=None):
        self.servo = servo
        self.fall_after_sec = fall_after_sec
        # None이면 눕힌 채로 둔다 (종료 시 close()가 세운다).
        self.hold_sec = hold_sec
        self._started_at = None
        self._fell_at = None
        self.done = False

    def tick(self, now):
        """매 프레임 호출한다. 상태가 바뀌었으면 FallEvent를, 아니면 None을 돌려준다."""
        if self.done:
            return None
        if self._started_at is None:
            self._started_at = now
            return None

        if self._fell_at is None:
            if now - self._started_at >= self.fall_after_sec:
                self.servo.fall()
                self._fell_at = now
                return FallEvent(
                    FELL,
                    f"[시연] 서보로 모형을 눕혔습니다 ({self.fall_after_sec:.1f}초 경과). "
                    "구동 모터를 세웁니다.",
                )
            return None

        if self.hold_sec is not None and now - self._fell_at >= self.hold_sec:
            self.servo.stand()
            self.done = True
            return FallEvent(
                STOOD,
                f"[시연] 모형을 다시 세웠습니다 ({self.hold_sec:.1f}초 유지). "
                "구동 모터가 이어서 움직입니다.",
            )
        return None
