"""실행 진입점.

두 가지 모드가 있고, **의존하는 것이 달라서** 나뉘어 있다.

    python main.py --mode fall     # 쓰러짐 감지 -> 아두이노 부저.  지금 바로 동작한다.
    python main.py --mode full     # 위 + 신호 연장.               제어부 프로토콜 확정 후.

## --mode fall (목표 2) — 지금 동작함

카메라 -> YOLO(추론 1회) -> 쓰러짐 판정 -> 시리얼로 ALERT/STOP.
필요한 것은 카메라와 아두이노(부저)뿐이다. 잔여 녹색 시간도, 사이클 이벤트도 쓰지 않는다.

    python main.py --mode fall                          # config 기본값으로
    python main.py --mode fall --source 0 --port COM3   # PC 웹캠 + 지정 포트
    python main.py --mode fall --display                # 창을 띄워 눈으로 확인
    python main.py --mode fall --no-serial              # 부저 없이 판정만 (배선 전 확인용)

## --mode full (목표 1 + 2) — 아직 막혀 있음

신호 연장은 제어부가 **잔여 녹색 시간**과 **새 사이클 시작**을 알려줘야 판단할 수 있는데,
그 메시지(`REMAIN <초>` / `CYCLE`)가 아직 팀 합의 전이다. 그래서 이 모드는 실행하면
무엇을 먼저 확정해야 하는지 알려주며 멈춘다(임의값으로 몰래 동작하지 않게 하기 위함).

막고 있는 것:
  - `REMAIN <초>` / `CYCLE` 메시지 합의 (docs/team_interface.md)
  - `EXTEND <초>`의 의미 — 누적인가 절대값인가
  - ZONE_RESIDENCY_FRAMES (실측 FPS 기반)
  - data/zone_config.json (tools/zone_calibrator.py 로 생성)

## 부분만 확인하고 싶을 때

    tools/manual_camera_person_check.py   # 시리얼 없이 비전만 (검출·구역·속도·FPS)
    tools/manual_buzzer_check.py          # 카메라 없이 시리얼·부저만
"""

import argparse
import sys
import time


def _report(result, fps):
    """상태를 한 줄로 만든다."""
    mark = "!! FALL !!" if result.fall_confirmed else "정상"
    line = f"FPS {fps:4.1f} | 사람 {result.people_count}명 | {mark}"
    if result.confirmed_ids:
        line += f" (id={sorted(result.confirmed_ids)})"
    if result.command_sent:
        line += f"  -> 아두이노: {result.command_sent}"
    return line


def run_fall_mode(args):
    from src.capture import CameraCapture
    from src.detection import PersonDetector
    from src.pipeline import FallAlarmPipeline
    from src.serial_comm import SerialComm
    from src.zone import CrosswalkZones

    # zone 설정이 있으면 캘리브레이션된 꼭짓점으로 ROI를 잡는다(눈대중 비율보다 정확).
    #
    # 여기서는 **어떤 이유로든** zone을 못 읽으면 화면 비율로 물러난다. 쓰러짐 감지에
    # zone은 정확도를 높여주는 선택지일 뿐 필수가 아니기 때문이다. 특히 해상도 불일치처럼
    # 설정이 '있는데 유효하지 않은' 경우(ValueError)까지 여기서 죽으면, 쓰러짐 감지가
    # 자기와 상관없는 신호 연장 쪽 설정 문제로 못 돌게 된다.
    #
    # 반대로 --mode full 은 zone이 없으면 구역 판정 자체가 불가능하므로 그대로 실패해야 한다.
    # 그래서 이 관대한 처리는 fall 모드에만 있다.
    zones = None
    try:
        zones = CrosswalkZones.load()
        print("[안내] zone 설정을 찾았습니다 — 캘리브레이션된 꼭짓점으로 ROI를 잡습니다.")
    except FileNotFoundError:
        print("[안내] zone 설정이 없어 화면 비율(FALL_CONFIG['crosswalk_roi'])로 ROI를 잡습니다.")
        print("       tools/zone_calibrator.py 로 만들면 더 정확합니다.")
    except (ValueError, KeyError, OSError) as exc:
        print("[경고] zone 설정을 읽었지만 쓸 수 없습니다. 화면 비율로 ROI를 잡고 계속합니다.")
        print(f"       사유: {exc}")
        print("       쓰러짐 감지는 이대로도 동작하지만, --mode full 을 쓰려면 고쳐야 합니다.")

    serial_comm = None
    if args.no_serial:
        serial_comm = _NullSerial()
        print("[안내] --no-serial: 부저로 보내지 않고 판정만 합니다.")
    else:
        serial_comm = SerialComm(port=args.port)

    pipeline = FallAlarmPipeline(
        camera=CameraCapture(source=args.source),
        detector=PersonDetector(),
        serial_comm=serial_comm,
        zones=zones,
    )

    print(f"[안내] 카메라 백엔드: {pipeline.camera.backend_name} (source={args.source!r})")
    if not args.no_serial:
        print("[안내] 아두이노가 리셋되어 부팅할 때까지 잠시 기다립니다...")
    print("[안내] 종료: " + ("창에서 q" if args.display else "Ctrl+C") + "\n")

    state = {"t0": time.monotonic(), "frames": 0, "fps": 0.0, "last": None}

    def on_result(result, frame):
        state["frames"] += 1
        now = time.monotonic()
        if now - state["t0"] >= 1.0:
            state["fps"] = state["frames"] / (now - state["t0"])
            state["t0"], state["frames"] = now, 0

        # 상태가 바뀌었거나(쓰러짐 발생/해제, 명령 전송) 1초가 지났을 때만 출력한다.
        # 매 프레임 찍으면 그 출력이 루프를 느리게 만든다.
        changed = result.fall_confirmed != state["last"] or result.command_sent
        if changed or state["frames"] == 0:
            print(_report(result, state["fps"]), flush=True)
            state["last"] = result.fall_confirmed

        if args.display:
            return _draw(frame, result, state["fps"], pipeline)
        return True

    try:
        pipeline.run(on_result=on_result)
    except KeyboardInterrupt:
        print("\n[안내] 중단됨 (Ctrl+C).")
    # pipeline.run()의 with 블록이 카메라를 닫고 시리얼에 STOP을 보낸다.


def _draw(frame, result, fps, pipeline):
    """창 모드. 계속 돌려면 True, 종료하려면 False를 반환한다."""
    import cv2

    if pipeline.roi_px:
        x1, y1, x2, y2 = pipeline.roi_px
        cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)

    color = (0, 0, 255) if result.fall_confirmed else (0, 255, 0)
    cv2.putText(frame, _report(result, fps), (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imshow("Fall -> Buzzer (q: quit, r: reset)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        return False
    if key == ord("r"):
        pipeline.reset_alarm()
        print("[안내] 알람을 수동으로 해제했습니다.")
    return True


class _NullSerial:
    """--no-serial 용. SerialComm 자리에 들어가 아무것도 보내지 않는다."""

    def update_alarm(self, active, now=None):
        return None

    def open(self):
        return self

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


def run_full_mode(args):
    from src.pipeline import SignalExtensionPipeline

    SignalExtensionPipeline().run()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("fall", "full"), default="fall",
                        help="fall = 쓰러짐 감지 -> 부저(동작함) / full = 신호 연장 포함(제어부 확정 후)")
    parser.add_argument("--source", default=None,
                        help="카메라 소스. 생략하면 config.CAMERA_SOURCE.")
    parser.add_argument("--port", default=None,
                        help="아두이노 시리얼 포트. 생략하면 config.SERIAL_PORT, 그것도 없으면 자동 탐색.")
    parser.add_argument("--display", action="store_true",
                        help="창을 띄운다. 모니터가 없으면 쓰지 말 것(기본은 stdout 출력).")
    parser.add_argument("--no-serial", action="store_true",
                        help="아두이노 없이 쓰러짐 판정만 확인한다.")
    args = parser.parse_args()

    if args.source is None:
        from config import config
        args.source = config.CAMERA_SOURCE

    if args.mode == "fall":
        run_fall_mode(args)
    else:
        run_full_mode(args)


if __name__ == "__main__":
    try:
        main()
    except (NotImplementedError, FileNotFoundError, RuntimeError) as exc:
        # 설정이 덜 됐을 때 스택 트레이스 대신 무엇이 막고 있는지 보여준다.
        print(f"\n실행할 수 없습니다:\n  {exc}\n", file=sys.stderr)
        sys.exit(1)
