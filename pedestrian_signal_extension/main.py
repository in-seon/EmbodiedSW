"""실행 진입점.

    python main.py                      # 전체 (기본값)
    python main.py --mode fall          # 전체 + 서보로 쓰러짐 연출 (시연용)
    python main.py --mode fall-only     # 쓰러짐 감지만 (zone 캘리브레이션 없이)

## 파이가 하는 일 / 아두이노가 하는 일

    파이   : 구역 판정, 잔류 확정, 쓰러짐 확정  -> 아래 세 상태로 요약해 전송
    아두이노: 잔여 녹색 시간, 임계값 판단, 누적 상한, 사이클 리셋, 부저/LED/7세그먼트

        normal         횡단보도에 확정 보행자가 없음
        zone<1..5>     가장 덜 건넌 사람의 **진척도**(1=방금 진입, 5=거의 다 건넘)
        fall           쓰러짐 확정

    (소문자, zone 뒤 숫자는 붙여 쓴다: `zone2`)

"남은 시간이 5초 미만인가"는 7세그먼트를 직접 세는 아두이노만 답할 수 있고, "이 사람이
몇 번 구역에 있나"는 영상을 보는 파이만 답할 수 있다. 각자 자기만 아는 것을 판단한다.
전체 계약은 docs/team_interface.md 참고.

## 모드

    --mode full        전체: 구역 기반 신호 연장 + 쓰러짐 감지 + 모터  (기본값)
    --mode fall        위와 **똑같이 전체를 돌리되**, 서보로 모형을 눕혀 쓰러짐을 연출한다.
                       손으로 넘어뜨리면 프레임에 사람 손이 들어와 검출을 교란하므로
                       원격으로 눕히는 편이 낫다. 눕히는 것은 시연 장치일 뿐이고,
                       판정은 평소와 **완전히 같은 경로**를 통과한다.
    --mode fall-only   쓰러짐 감지만. zone 캘리브레이션 없이 돌아가므로 카메라만 놓고
                       바로 확인할 때 쓴다(신호 연장·모터는 동작하지 않는다).

## 옵션

    --source           카메라 소스. 생략하면 config.CAMERA_SOURCE
    --port             시리얼 포트. 생략하면 자동 탐색
    --display          창을 띄운다 (모니터 없으면 쓰지 말 것). q=종료, r=알람 해제
    --no-serial        아두이노 없이 판정만 확인
    --no-motor         모터 GPIO를 잡지 않는다 (개발 PC / 모형 없이 확인할 때)
    --fall-after <초>  --mode fall 에서 몇 초 뒤에 눕힐지 (기본 config.SERVO_FALL_AFTER_SEC)
    --fall-hold <초>   눕힌 뒤 몇 초 후 다시 세울지 (기본 config.SERVO_FALL_HOLD_SEC)

## 모형에 달린 모터는 둘이다

    스텝모터  횡단보도를 따라 모형을 끌고 간다      (L298N + NEMA17, config.MOTOR_*)
    서보      모형 발에 붙어 쓰러뜨린다             (--mode fall 전용, config.SERVO_*)

--mode fall 에서 둘은 맞물려 움직인다:

    START -> 끌고 감 -> [눕힘 + 구동 정지] -> [일으킴 + 구동 재개] -> 다 건넘 -> 정지

넘어진 모형을 계속 끌고 가면 안 되지만, 일어난 뒤에는 마저 건너야 한다. 그래서 이때의
멈춤은 '정지'가 아니라 **'일시정지'**다 — 같은 횡단을 이어서 간다(src/motor.py).

## 모형 보행자 모터는 아두이노가 켜고 파이가 끈다

모터는 아두이노 핀이 모자라 **파이 GPIO에 붙어 있다.** 그런데 "보행 녹색이 시작됐다"는
신호를 소유한 아두이노만 알고, "모형이 다 건넜다"는 영상을 보는 파이만 안다.

    아두이노 -> 파이 : START [모드]     녹색 시작. 모형을 출발시켜라
    파이 자체 판단   : 확정 보행자가 있었다가 사라짐 -> 정지

'없음'이 아니라 '있었다가 없어짐'인 이유: START 시점에는 모형이 아직 횡단보도 밖이라
확정 보행자가 0명이다. '없으면 정지'로 짜면 출발도 못 하고 멈춘다(src/motor.py).

## 부분만 확인하고 싶을 때

    tools/manual_camera_person_check.py   # 시리얼 없이 비전만 (검출·구역·속도·FPS·--fall)
    tools/manual_buzzer_check.py          # 카메라 없이 시리얼만
"""

import argparse
import sys
import time

from config import config



class _NullSerial:
    """--no-serial 용. SerialComm 자리에 들어가 아무것도 보내지 않는다.

    파이프라인이 시리얼에 기대하는 표면 전체를 흉내내야 한다 — 하나라도 빠지면
    --no-serial 이 AttributeError로 죽는다.
    """

    ready = True
    port = "(없음)"

    def update_state(self, state, zone=None, now=None):
        return None

    def send_state(self, state, zone=None, now=None):
        return None

    def poll(self):
        return []

    def ping(self):
        return True

    def take_commands(self):
        """아두이노가 없으므로 START도 오지 않는다 -> 모터도 돌지 않는다."""
        return []

    def open(self):
        return self

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


def _make_serial(args):
    if args.no_serial:
        print("[안내] --no-serial: 아두이노로 보내지 않고 판정만 합니다.")
        return _NullSerial()
    from src.serial_comm import SerialComm

    print("[안내] 아두이노가 리셋되어 부팅할 때까지 잠시 기다립니다...")
    return SerialComm(port=args.port)


def _make_motor(args):
    """모형 보행자 스텝모터. --no-motor 이거나 GPIO가 없으면 NullMotor로 떨어진다."""
    from src.motor import NullMotor, StepperMotor

    if args.no_motor:
        print("[안내] --no-motor: 모터 GPIO를 잡지 않습니다(판단 로직만 확인).")
        return NullMotor()
    try:
        import gpiozero  # noqa: F401
    except ImportError:
        # 개발 PC에서 흔한 경우다. 여기서 죽이면 노트북에서 아무것도 확인할 수 없고,
        # 조용히 넘어가면 파이에서 "왜 모터가 안 도는지"를 못 찾는다. 그래서 알리고 계속한다.
        print("[안내] gpiozero가 없어 모터를 비활성화합니다(라즈베리파이가 아닌 듯합니다).")
        return NullMotor()
    return StepperMotor()


def _make_fall_scheduler(args):
    """--mode fall 의 서보 연출. 실패해도 본체는 계속 돌린다."""
    from src.servo import FallScheduler, FallServo

    after = config.SERVO_FALL_AFTER_SEC if args.fall_after is None else args.fall_after
    # hold 기본값이 '계속 누워 있음'이 아닌 이유: 데모에서는 일어나서 마저 건너는 것까지
    # 보여야 하고, 그때 구동 모터도 다시 움직여야 한다(config 주석 참고).
    hold_sec = config.SERVO_FALL_HOLD_SEC if args.fall_hold is None else args.fall_hold
    try:
        import gpiozero  # noqa: F401
    except ImportError:
        print("[안내] gpiozero가 없어 서보 연출을 건너뜁니다. 감지 자체는 그대로 동작합니다.")
        print("       (모형을 손으로 눕혀도 되지만, 프레임에 손이 들어가면 검출이 흔들립니다)")
        return None

    servo = FallServo()
    print(f"[시연] {after:.1f}초 뒤 서보가 모형을 눕히고, {hold_sec:.1f}초 뒤 다시 세웁니다.")
    print("[시연] 눕히는 동안 구동 모터는 멈췄다가, 일으키면 이어서 움직입니다.")
    return FallScheduler(servo, fall_after_sec=after, hold_sec=hold_sec)


def _load_zones(required: bool):
    """zone 설정을 읽는다.

    required=False(쓰러짐만)면 **어떤 이유로든** 못 읽어도 None을 돌려주고 계속한다.
    쓰러짐 감지에 zone은 ROI 정확도를 높여주는 선택지일 뿐이라, 신호 연장 쪽 설정 문제로
    쓰러짐까지 못 돌게 만들 이유가 없다.

    required=True(신호 연장)면 그대로 실패시킨다. zone이 없으면 구역 판정 자체가 불가능하다.
    """
    from src.zone import CrosswalkZones

    if required:
        try:
            return CrosswalkZones.load()
        except FileNotFoundError as exc:
            # pathlib의 원래 메시지는 경로만 알려준다. 정작 필요한 것은 "그 파일을
            # 어떻게 만드는가"인데, 그걸 모르면 파이 앞에서 한참 헤맨다.
            raise RuntimeError(
                f"횡단보도 구역 설정이 없습니다: {exc.filename}\n"
                "  캘리브레이션을 먼저 하세요 — 화면에서 횡단보도 네 꼭짓점을 찍습니다:\n"
                "      python tools/zone_calibrator.py\n"
                "  구역 설정 없이 쓰러짐 감지만 확인하려면:\n"
                "      python main.py --mode fall-only"
            ) from exc
    try:
        zones = CrosswalkZones.load()
        print("[안내] zone 설정을 찾았습니다 — 캘리브레이션된 꼭짓점으로 ROI를 잡습니다.")
        return zones
    except FileNotFoundError:
        print("[안내] zone 설정이 없어 화면 비율(FALL_CONFIG['crosswalk_roi'])로 ROI를 잡습니다.")
        print("       tools/zone_calibrator.py 로 만들면 더 정확합니다.")
    except (ValueError, KeyError, OSError) as exc:
        print("[경고] zone 설정을 읽었지만 쓸 수 없습니다. 화면 비율로 ROI를 잡고 계속합니다.")
        print(f"       사유: {exc}")
        print("       쓰러짐 감지는 이대로도 동작하지만, --mode full 을 쓰려면 고쳐야 합니다.")
    return None


class _Reporter:
    """FPS를 재고, 상태가 바뀔 때와 1초마다 한 줄씩 찍는다.

    매 프레임 출력하면 그 출력이 루프를 느리게 만들어 재려는 FPS 자체를 왜곡한다.
    """

    def __init__(self, display):
        self.display = display
        self.t0 = time.monotonic()
        self.frames = 0
        self.fps = 0.0
        self.last_key = None

    def tick(self, key, text, force=False):
        self.frames += 1
        now = time.monotonic()
        second_passed = now - self.t0 >= 1.0
        if second_passed:
            self.fps = self.frames / (now - self.t0)
            self.t0, self.frames = now, 0
        if force or second_passed or key != self.last_key:
            print(f"FPS {self.fps:4.1f} | {text}", flush=True)
            self.last_key = key


def _describe(state, zone, result_line):
    """화면용 문구. **상태 문자열을 여기에 다시 쓰지 말 것** — 상수를 쓴다.

    한때 여기에 "FALL"/"NORMAL"을 직접 박아 뒀다가, 프로토콜을 소문자로 바꿀 때
    화면 표시만 조용히 깨졌다. 전송은 멀쩡한데 로그만 틀리는 형태라 알아채기 어렵다.
    """
    from src.serial_comm import STATE_FALL, STATE_NORMAL, STATE_ZONE

    if state == STATE_FALL:
        mark = "!! FALL !!"
    elif state == STATE_NORMAL:
        mark = "정상"
    elif state == STATE_ZONE:
        mark = f"zone{zone} (진척도)"
    else:
        mark = state
    return f"{mark}{result_line}"


def run_fall_only_mode(args):
    """쓰러짐 감지만. zone 캘리브레이션 없이 돌아가는 것이 이 모드의 존재 이유다."""
    from src.capture import CameraCapture
    from src.detection import PersonDetector
    from src.pipeline import FallAlarmPipeline
    from src.serial_comm import STATE_FALL, STATE_NORMAL

    zones = _load_zones(required=False)
    pipeline = FallAlarmPipeline(
        camera=CameraCapture(source=args.source),
        detector=PersonDetector(),
        serial_comm=_make_serial(args),
        zones=zones,
    )
    print(f"[안내] 카메라 백엔드: {pipeline.camera.backend_name} (source={args.source!r})")
    print("[안내] 종료: " + ("창에서 q" if args.display else "Ctrl+C") + "\n")

    reporter = _Reporter(args.display)

    def on_result(result, frame):
        extra = f" (id={sorted(result.confirmed_ids)})" if result.confirmed_ids else ""
        if result.command_sent:
            extra += f"  -> {result.command_sent}"
        text = f"사람 {result.people_count}명 | " + _describe(
            STATE_FALL if result.fall_confirmed else STATE_NORMAL, None, extra)
        reporter.tick(result.fall_confirmed, text)
        if args.display:
            return _draw(frame, text, result.fall_confirmed, pipeline)
        return True

    _drive(pipeline, on_result)


def run_full_mode(args):
    from src.capture import CameraCapture
    from src.detection import PersonDetector
    from src.pipeline import CombinedPipeline, SignalExtensionPipeline, _NoSend

    from src.speed import SpeedEstimator
    from src.zone import CrosswalkOccupancy

    if args.confirm_frames is None:
        raise NotImplementedError(
            "ZONE_RESIDENCY_FRAMES가 아직 미정입니다.\n"
            "  실측 FPS를 기준으로 값을 정해 config에 넣거나, 확인용으로 --confirm-frames 3 처럼 주세요.\n"
            "  (FPS 실측: python tools/manual_camera_person_check.py --source picamera2 --no-display)"
        )

    zones = _load_zones(required=True)
    camera = CameraCapture(source=args.source)
    detector = PersonDetector()
    extension = SignalExtensionPipeline(
        camera=camera, detector=detector, zones=zones,
        occupancy=CrosswalkOccupancy(zones, confirm_frames=args.confirm_frames),
        serial_comm=_NoSend(),
        speed_estimator=SpeedEstimator(ground_plane=zones.ground_plane),
    )
    from src.servo import FELL, STOOD

    motor = _make_motor(args)
    pipeline = CombinedPipeline(
        camera=camera,
        detector=detector,
        serial_comm=_make_serial(args),
        extension=extension,
        zones=zones,
        motor=motor,
    )
    # 서보는 파이프라인 밖에 둔다 — 감지 결과에 반응하는 장치가 아니라 감지될 사건을
    # 만드는 장치라서다. 파이프라인은 이것이 있는지도 모르는 채로 판정한다.
    scheduler = _make_fall_scheduler(args) if args.mode == "fall" else None

    print(f"[안내] 카메라 백엔드: {pipeline.camera.backend_name} (source={args.source!r})")
    print(f"[안내] 속도 단위: {pipeline.extension.speed.unit}  (계측용 — 연장 판단에는 쓰지 않음)")
    print("[안내] 모터: 아두이노가 START를 보내면 기동, 모형이 다 건너면 자동 정지합니다.")
    print("[안내] 종료: " + ("창에서 q" if args.display else "Ctrl+C") + "\n")

    reporter = _Reporter(args.display)

    def on_result(result, frame):
        if scheduler is not None:
            now = time.monotonic()
            event = scheduler.tick(now)
            if event is not None:
                print(event.message, flush=True)
                # 모형에 달린 두 모터를 맞춰 준다: 넘어지는 동안은 끌고 가지 않고,
                # 일어나면 멈춘 지점부터 이어서 간다. 정지가 아니라 일시정지다.
                if event.kind == FELL:
                    pipeline.pause_motor(now=now)
                elif event.kind == STOOD:
                    pipeline.resume_motor(now=now)
        if result.motor_event:
            print(f"[모터] {result.motor_event}", flush=True)

        ext = result.extension
        extra = ""
        if ext.occupied_zones:
            extra += f" zone={ext.occupied_zones}"
        if ext.eta_sec is not None:
            extra += f" ETA={ext.eta_sec:.1f}s"   # 계측용 — 전송하지 않는다
        if result.motor_paused:
            extra += " [모터 멈춤]"
        elif result.motor_running:
            extra += " [모터]"
        if result.line_sent:
            extra += f"  -> {result.line_sent}"
        text = (f"사람 {len(ext.pedestrians)}명 | "
                + _describe(result.state, result.zone, extra))
        reporter.tick((result.state, result.zone,
                       result.motor_running, result.motor_paused), text)
        if args.display:
            return _draw(frame, text, result.fall.fall_confirmed, pipeline)
        return True

    try:
        _drive(pipeline, on_result)
    finally:
        # 눕힌 채로 끝내지 않는다 — 다음 실행이 '이미 쓰러진 사람'에서 시작하면
        # FALL이 즉시 확정돼 무엇을 시연하려던 것인지 알 수 없게 된다.
        if scheduler is not None:
            scheduler.servo.close()


def _drive(pipeline, on_result):
    try:
        pipeline.run(on_result=on_result)
    except KeyboardInterrupt:
        print("\n[안내] 중단됨 (Ctrl+C).")
    # run()의 with 블록이 카메라를 닫고 시리얼에 NORMAL을 보낸다.


def _draw(frame, text, alarm, pipeline):
    """창 모드. 계속 돌려면 True, 종료하려면 False."""
    import cv2

    roi = getattr(pipeline, "roi_px", None) or getattr(
        getattr(pipeline, "fall", None), "roi_px", None)
    if roi:
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)

    color = (0, 0, 255) if alarm else (0, 255, 0)
    # cv2.putText는 한글을 못 그리므로(Hershey 폰트) 화면에는 ASCII만 남긴다.
    ascii_text = text.encode("ascii", "replace").decode("ascii")
    cv2.putText(frame, ascii_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.imshow("Smart Crosswalk (q: quit, r: reset alarm)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        return False
    if key == ord("r"):
        pipeline.reset_alarm()
        print("[안내] 알람을 수동으로 해제했습니다.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("full", "fall", "fall-only"), default="full",
                        help="full = 전체(기본) / fall = 전체 + 서보로 쓰러짐 연출 / "
                             "fall-only = 쓰러짐 감지만(zone 설정 없이도 동작)")
    parser.add_argument("--source", default=None,
                        help="카메라 소스. 생략하면 config.CAMERA_SOURCE.")
    parser.add_argument("--port", default=None,
                        help="아두이노 시리얼 포트. 생략하면 config.SERIAL_PORT, 그것도 없으면 자동 탐색.")
    parser.add_argument("--display", action="store_true",
                        help="창을 띄운다. 모니터가 없으면 쓰지 말 것(기본은 stdout 출력).")
    parser.add_argument("--no-serial", action="store_true",
                        help="아두이노 없이 판정만 확인한다.")
    parser.add_argument("--no-motor", action="store_true",
                        help="모터 GPIO를 잡지 않는다. 라즈베리파이가 아니면 자동으로 이렇게 된다.")
    parser.add_argument("--fall-after", type=float, default=None,
                        help="--mode fall 에서 서보가 모형을 눕히기까지의 지연(초). "
                             "생략하면 config.SERVO_FALL_AFTER_SEC.")
    parser.add_argument("--fall-hold", type=float, default=None,
                        help="--mode fall 에서 눕힌 뒤 다시 세우기까지의 시간(초). "
                             "생략하면 config.SERVO_FALL_HOLD_SEC. 이 시간 동안 구동 "
                             "모터도 함께 멈춰 있다가, 일으키면 이어서 움직인다.")
    parser.add_argument("--confirm-frames", type=int, default=config.ZONE_RESIDENCY_FRAMES,
                        help="'확정 보행자'로 보기까지 필요한 연속 검출 프레임 수. "
                             "config.ZONE_RESIDENCY_FRAMES가 아직 미정(None)이라, 실측 FPS로 "
                             "값을 정하기 전까지는 이 옵션으로 넣어야 --mode full 이 돌아간다.")
    args = parser.parse_args()

    if args.source is None:
        args.source = config.CAMERA_SOURCE

    # argparse는 --source 0 을 문자열 "0"으로 준다. 그대로 넘기면 cv2가 0번 카메라가 아니라
    # "0"이라는 파일을 열려고 해서 무조건 실패한다(src/capture.py의 normalize_source 참고).
    from src.capture import normalize_source
    args.source = normalize_source(args.source)

    if args.mode != "fall" and (args.fall_after is not None or args.fall_hold is not None):
        # 조용히 무시하면 "왜 서보가 안 도는지"를 찾느라 시간을 버린다.
        print("[경고] --fall-after / --fall-hold 는 --mode fall 에서만 동작합니다. 무시합니다.")

    if args.mode == "fall-only":
        run_fall_only_mode(args)
    else:
        # full 과 fall 은 같은 파이프라인이다 — fall 은 서보 연출이 붙을 뿐이다.
        run_full_mode(args)


if __name__ == "__main__":
    try:
        main()
    except (NotImplementedError, FileNotFoundError, RuntimeError) as exc:
        # 설정이 덜 됐을 때 스택 트레이스 대신 무엇이 막고 있는지 보여준다.
        print(f"\n실행할 수 없습니다:\n  {exc}\n", file=sys.stderr)
        sys.exit(1)
