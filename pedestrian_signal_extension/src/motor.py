"""모형 보행자를 움직이는 스텝모터 (라즈베리파이 GPIO 직결).

    라즈베리파이5 + L298N + NEMA17

배선:
  IN1 -> GPIO17    IN2 -> GPIO27    IN3 -> GPIO22    IN4 -> GPIO23
  ENA -> GPIO12 (PWM)   ENB -> GPIO13 (PWM)   (ENA/ENB 점퍼 제거 상태여야 함)
  코일 A(검정,초록) -> OUT1, OUT2
  코일 B(빨강,파랑) -> OUT3, OUT4
  GND: 전원(-), 라즈베리파이 GND, L298N GND 모두 공통 GND 레일에 연결
  핀 번호는 config.MOTOR_COIL_PINS / MOTOR_ENABLE_PINS 에서 바꾼다.

## 이 파일은 두 가지로 나뉜다 — 이유가 있다

    MotorGate     '지금 돌려야 하는가'를 판단하는 순수 로직. GPIO를 모른다.
    StepperMotor  실제로 코일에 전류를 흘리는 쪽. 판단을 하지 않는다.

정지 조건("모형이 횡단보도를 빠져나갔다")은 **틀리기 쉬운 로직**이라 테스트가 필요한데,
GPIO에 묶여 있으면 파이에서만 테스트할 수 있다. 분리해 두면 판단은 노트북에서
초 단위로 검증하고, 하드웨어는 tools/manual_motor_check.py로 따로 확인한다.

## 왜 스텝이 별도 스레드인가

스텝 간격은 5ms인데 비전 루프는 실측 3~10 FPS(100~330ms)다. 루프 안에서 스텝을 밟으면
**한 프레임에 한 스텝**밖에 못 밟아 모터가 20배 느려지거나 아예 안 돈다. 두 주기가
두 자릿수 차이라 같은 스레드에 둘 수 없다.
"""

import threading
import time

from config import config

# 풀스텝 시퀀스 (in1, in2, in3, in4 순서)
SEQUENCE = (
    (1, 0, 0, 1),
    (1, 0, 1, 0),
    (0, 1, 1, 0),
    (0, 1, 0, 1),
)

# MotorGate가 멈춘 이유. 로그에 그대로 찍어 원인을 구분한다.
STOP_CROSSED = "횡단 완료"
STOP_COMMAND = "아두이노 STOP"
STOP_TIMEOUT = "안전 타임아웃"


class MotorGate:
    """모터를 언제 돌리고 언제 멈출지 판단한다. **GPIO를 모른다** (테스트 가능).

    ## 상태

        대기(idle)     START를 기다린다.
        주행(running)  START를 받아 돌고 있다.
        일시정지(paused) 주행 중이되 잠깐 세워 둔 상태 — 쓰러짐 연출 동안.

    **일시정지는 정지와 다르다.** stop()은 '이번 횡단이 끝났다'는 뜻이라 _seen 래치까지
    지우지만, pause()는 같은 횡단 도중의 멈춤이라 상태를 그대로 들고 있다가 이어서 간다.
    쓰러진 사람을 계속 끌고 가면 안 되지만, 일어난 뒤에는 마저 건너야 하기 때문이다.

    ## ⚠️ '보행자가 없으면 정지'를 그대로 쓰면 시작하자마자 멈춘다

    START가 오는 시점은 보행 녹색이 막 켜진 순간이고, 모형은 아직 횡단보도 **밖**에 있다.
    그래서 그 순간의 확정 보행자는 0명이다. "0명이면 정지"로 짜면:

        START -> 0명 -> 즉시 정지        (모형은 출발도 못 했다)

    정지 조건은 "없음"이 아니라 **"있었다가 없어짐"**이어야 한다. 그래서 START 이후
    확정 보행자를 한 번이라도 본 사실(_seen)을 래치해 두고, 그 뒤에 0명이 됐을 때만 멈춘다.

    ## 안전 타임아웃

    모형이 탈선하거나 검출이 계속 실패하면 위 조건이 **영영 오지 않는다.** 물리 장치라
    조용한 실패로 끝나지 않고 모터가 계속 돌아 손상된다. config.MOTOR_MAX_RUN_SEC이
    지나면 무조건 멈춘다 — 정상 동작에서는 발동하지 않아야 하는 값이다.
    """

    def __init__(self, max_run_sec=None):
        self.max_run_sec = (
            max_run_sec if max_run_sec is not None else config.MOTOR_MAX_RUN_SEC
        )
        self.running = False
        self.mode = None
        self._seen = False          # START 이후 확정 보행자를 본 적이 있는가
        self._started_at = None
        self._paused_at = None      # 일시정지 시각 (None이면 정지 중이 아님)
        self.last_stop_reason = None

    def start(self, mode=None, now=None) -> bool:
        """아두이노 START. 이미 돌고 있으면 아무것도 하지 않는다.

        재전송이나 중복 START로 타임아웃 시계가 계속 리셋되면 안전장치가 무력해진다.
        반환: 이번 호출로 새로 기동했으면 True.
        """
        if self.running:
            return False
        self.running = True
        self.mode = config.MOTOR_DEFAULT_MODE if mode is None else mode
        self._seen = False
        self._started_at = time.monotonic() if now is None else now
        self._paused_at = None
        self.last_stop_reason = None
        return True

    def stop(self, reason=STOP_COMMAND) -> bool:
        """정지시킨다. 반환: 이번 호출로 실제로 멈췄으면 True."""
        if not self.running:
            return False
        self.running = False
        self.mode = None
        self._seen = False
        self._started_at = None
        self._paused_at = None
        self.last_stop_reason = reason
        return True

    def update(self, occupied: bool, now=None) -> bool:
        """매 프레임 호출한다.

        occupied: 이번 프레임에 **확정 보행자가 횡단보도 위에 있는가.**
                  (CombinedResult.extension.progress is not None 과 같다)

        반환: 이번 호출로 멈췄으면 True.
        """
        if not self.running:
            return False
        now = time.monotonic() if now is None else now

        if occupied:
            self._seen = True

        # 일시정지 중에는 아무 판단도 하지 않는다.
        #
        # 쓰러진 사람이 검출에서 잠깐 빠지는 것은 정상이다 — 누우면 bbox 모양이 급변해
        # 트래커가 ID를 새로 달거나 놓친다. 그때 '건너갔다'로 읽어 모터를 영구 정지시키면
        # 일어나도 다시 안 움직인다. 멈춰 있는 동안은 진행이 없으니 판단할 것도 없다.
        if self._paused_at is not None:
            return False

        if not occupied and self._seen:
            return self.stop(STOP_CROSSED)

        if now - self._started_at >= self.max_run_sec:
            return self.stop(STOP_TIMEOUT)
        return False

    # ------------------------------------------------------------------
    # 일시정지 — 쓰러짐 연출 중에 모형을 세워 둔다
    # ------------------------------------------------------------------

    def pause(self, now=None) -> bool:
        """잠깐 세운다. **stop()과 다르다** — START를 다시 받지 않아도 재개된다.

        데모에서 모형이 넘어지는 동안 계속 끌려가면 안 된다. 그렇다고 stop()을 쓰면
        '한 번의 횡단'이 거기서 끝난 것이 되어(_seen 래치가 지워져) 일어난 뒤의 이동을
        새 사이클로 오해한다. 그래서 '멈춤'을 두 종류로 나눴다.

        반환: 이번 호출로 실제로 멈췄으면 True.
        """
        if not self.running or self._paused_at is not None:
            return False
        self._paused_at = time.monotonic() if now is None else now
        return True

    def resume(self, now=None) -> bool:
        """멈춰 있던 지점부터 이어서 움직인다.

        멈춰 있던 시간만큼 안전 타임아웃 시계를 뒤로 민다. 그러지 않으면 연출로 세워 둔
        시간이 주행 시간으로 잡혀, 긴 연출 뒤에 곧바로 타임아웃이 터진다.
        """
        if not self.running or self._paused_at is None:
            return False
        now = time.monotonic() if now is None else now
        self._started_at += now - self._paused_at
        self._paused_at = None
        return True

    @property
    def paused(self) -> bool:
        return self._paused_at is not None

    @property
    def moving(self) -> bool:
        """실제로 모형이 움직이고 있는가 (기동했고 일시정지도 아님)."""
        return self.running and self._paused_at is None

    @property
    def seen_pedestrian(self) -> bool:
        """START 이후 확정 보행자를 본 적이 있는가 (진단용)."""
        return self._seen


class StepperMotor:
    """L298N + NEMA17 풀스텝 구동. 스텝은 백그라운드 스레드가 밟는다.

    GPIO는 **start()를 처음 부를 때** 잡는다. import 시점에 잡으면 노트북에서
    `import src.motor` 만 해도 죽어서 테스트도 문서 생성도 못 한다.
    """

    def __init__(self, coil_pins=None, enable_pins=None, duty=None,
                 step_delay_sec=None):
        self.coil_pins = tuple(coil_pins or config.MOTOR_COIL_PINS)
        self.enable_pins = tuple(enable_pins or config.MOTOR_ENABLE_PINS)
        self.duty = config.MOTOR_DUTY if duty is None else duty
        self.step_delay_sec = dict(step_delay_sec or config.MOTOR_STEP_DELAY_SEC)

        self._coils = None
        self._enables = None
        self._thread = None
        self._stop_event = threading.Event()
        self._delay = None
        self.mode = None

    # ------------------------------------------------------------------
    # GPIO
    # ------------------------------------------------------------------

    def _ensure_gpio(self):
        if self._coils is not None:
            return
        try:
            from gpiozero import OutputDevice, PWMOutputDevice
        except ImportError as exc:
            raise RuntimeError(
                "gpiozero를 불러올 수 없습니다 — 모터는 라즈베리파이에서만 동작합니다.\n"
                "  개발 PC에서 확인하려면 main.py에 --no-motor 를 주세요.\n"
                f"  ({exc})"
            ) from exc

        self._coils = [OutputDevice(pin) for pin in self.coil_pins]
        self._enables = [
            PWMOutputDevice(pin, frequency=1000) for pin in self.enable_pins
        ]

    # ------------------------------------------------------------------
    # 구동
    # ------------------------------------------------------------------

    def start(self, mode=None):
        """모터를 돌리기 시작한다. 이미 돌고 있으면 속도만 바꾼다."""
        mode = config.MOTOR_DEFAULT_MODE if mode is None else mode
        if mode not in self.step_delay_sec:
            raise ValueError(
                f"알 수 없는 속도 모드입니다: {mode!r} "
                f"(설정된 모드: {sorted(self.step_delay_sec)})"
            )
        self._ensure_gpio()
        self.mode = mode
        # 스레드가 매 스텝 읽으므로 돌고 있는 중에도 속도가 즉시 반영된다.
        self._delay = self.step_delay_sec[mode]

        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        for enable in self._enables:
            enable.value = self.duty
        # daemon=True: Ctrl+C로 죽을 때 이 스레드가 프로세스를 붙잡지 않게 한다.
        # 코일 전원은 close()가 확실히 끊는다.
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="stepper")
        self._thread.start()

    def _run(self):
        index = 0
        while not self._stop_event.is_set():
            for coil, value in zip(self._coils, SEQUENCE[index]):
                coil.value = bool(value)
            index = (index + 1) % len(SEQUENCE)
            # Event.wait는 sleep과 달리 정지 요청에 즉시 깨어난다.
            self._stop_event.wait(self._delay)

    def stop(self):
        """모터를 멈추고 코일 전원을 끊는다.

        코일을 켜 둔 채 멈추면 정지 토크를 유지하느라 계속 전류가 흘러 모터와 L298N이
        뜨거워진다. 모형이 멈춰 있기만 하면 되는 용도라 전원을 끊는 편이 낫다.
        """
        self._stop_event.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=1.0)
        self.mode = None
        self._de_energize()

    def _de_energize(self):
        if self._coils is None:
            return
        for coil in self._coils:
            coil.value = False
        for enable in self._enables:
            enable.value = 0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def close(self):
        """정지시키고 GPIO를 반납한다. 어떤 경로로 끝나도 반드시 불러야 한다."""
        self.stop()
        for device in list(self._coils or []) + list(self._enables or []):
            try:
                device.close()
            except Exception:
                pass    # 이미 닫혔거나 GPIO가 사라진 경우 — 종료가 우선이다.
        self._coils = None
        self._enables = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class NullMotor:
    """--no-motor 용. StepperMotor 자리에 들어가 아무것도 하지 않는다.

    GPIO가 없는 개발 PC에서 판단 로직만 확인할 때 쓴다. StepperMotor가 노출하는
    표면을 전부 흉내내야 한다 — 하나라도 빠지면 --no-motor가 AttributeError로 죽는다.
    """

    running = False
    mode = None

    def start(self, mode=None):
        self.mode = config.MOTOR_DEFAULT_MODE if mode is None else mode

    def stop(self):
        self.mode = None

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


if __name__ == "__main__":
    # 판단 로직 없이 모터만 돌려 보는 최소 확인. 본격적인 확인은
    # tools/manual_motor_check.py 를 쓸 것(속도 모드 전환·스톱워치 안내 포함).
    motor = StepperMotor()
    print("회전 시작. 멈추려면 Ctrl+C")
    try:
        motor.start()
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("정지")
    finally:
        motor.close()
