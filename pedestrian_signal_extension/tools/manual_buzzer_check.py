"""[수동] 아두이노 부저 시리얼 통신 확인 스크립트 (자동 테스트 아님).

카메라도 YOLO도 없이 **시리얼과 부저만** 확인한다. 쓰러짐 감지가 부저를 못 울릴 때
원인이 비전 쪽인지 통신 쪽인지 가르는 용도다.

사용법:
    python tools/manual_buzzer_check.py                  # 연결된 포트 목록만 출력
    python tools/manual_buzzer_check.py --port COM3      # PING -> fall -> 5초 -> normal
    python tools/manual_buzzer_check.py --port /dev/ttyACM0 --seconds 3
    python tools/manual_buzzer_check.py --port COM3 --interactive   # 키로 직접 조작
    python tools/manual_buzzer_check.py --port COM3 --watchdog      # 워치독 동작 확인

--port를 생략하면 config.SERIAL_PORT를 쓰고, 그것도 None이면 자동 탐색한다.

## 확인해야 할 것

  1. PONG이 돌아오는가        -> 포트/보드레이트/케이블 OK
  2. fall에 부저가 울리는가    -> 배선(BUZZER_PIN)과 ACTIVE_BUZZER 설정 OK
  3. normal에 멎는가           -> 정지 경로 OK
  4. (--watchdog) 재전송을 멈추면 아두이노가 스스로 normal로 돌아가는가

## ⚠️ 상태 메시지에는 응답이 오지 않는 것이 정상이다

프로토콜상 `normal`/`zone<n>`/`fall`에는 아두이노가 답하지 않는다. 응답 송신 시간이
아두이노 루프를 묶기 때문이다(docs/team_interface.md). **연결 확인은 `PING`/`PONG`으로만**
한다. 상태를 보낸 뒤 아무 줄도 안 오는 것은 고장이 아니다 — 부저가 우는지 귀로 본다.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import config
from src.serial_comm import STATE_FALL, STATE_NORMAL, STATE_ZONE, SerialComm


def print_ports():
    ports = SerialComm.available_ports()
    if not ports:
        print("연결된 시리얼 포트가 없습니다. 아두이노가 USB로 꽂혀 있는지 확인하세요.")
        return
    print("연결된 시리얼 포트:")
    for device, description in ports:
        print(f"  {device}    {description}")
    print("\n--port 로 하나를 골라 다시 실행하세요.")


def drain(comm, label=""):
    """도착한 응답을 읽어 출력한다."""
    for line in comm.poll():
        print(f"    <- {line}{label}")


def run_once(comm, seconds):
    """fall -> seconds초 대기 -> normal."""
    print("[1] PING 으로 연결 확인")
    comm.ping()
    time.sleep(0.3)
    drain(comm)

    print(f"[2] fall 전송 — {seconds}초간 부저가 삑-삑 울려야 합니다")
    comm.send_state(STATE_FALL)
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        comm.send_state(STATE_FALL)      # 하트비트(이 짧은 시간엔 실제로 재전송되진 않는다)
        drain(comm)
        time.sleep(0.1)

    print("[3] normal 전송 — 부저가 멎어야 합니다")
    comm.send_state(STATE_NORMAL)
    time.sleep(0.3)
    drain(comm)
    print("\n부저가 울렸다가 멎었으면 통신·배선 모두 정상입니다.")


def run_watchdog(comm):
    """하트비트를 일부러 끊어 아두이노 30초 워치독이 도는지 본다."""
    print("[1] fall 전송 후 **재전송을 멈춥니다.**")
    comm.send_state(STATE_FALL)
    print("    아두이노 스케치의 TIMEOUT_MS(30초) 뒤에 'TIMEOUT'이 오고 부저가 멎어야 합니다.")
    print("    (파이가 죽거나 USB가 빠진 상황을 흉내내는 것입니다)\n")

    start = time.monotonic()
    seen_timeout = False
    while time.monotonic() - start < 40:
        elapsed = time.monotonic() - start
        for line in comm.poll():
            print(f"    <- {line}   ({elapsed:.1f}초)")
            if line == "TIMEOUT":
                seen_timeout = True
        if seen_timeout:
            break
        time.sleep(0.1)

    if seen_timeout:
        print(f"\n워치독 정상 — {time.monotonic() - start:.1f}초 만에 스스로 멎었습니다.")
    else:
        print("\n40초를 기다렸는데 TIMEOUT이 오지 않았습니다.")
        print("스케치의 TIMEOUT_MS 값과, alarmStart가 FALL마다 갱신되는지 확인하세요.")
    comm.send_state(STATE_NORMAL)


def run_interactive(comm):
    print("명령: a=fall  s=normal  1/3/5=zone n  p=PING  q=종료")
    print()
    print("  * 상태 메시지(a/s/1/3/5)에는 응답이 없는 것이 정상입니다. 부저는 귀로 확인하세요.")
    print("  * 연결이 의심되면 p(PING)로 확인하세요 — PONG이 와야 정상입니다.")
    print("  * PONG은 오는데 부저가 안 울린다면 통신은 정상이고 부저/배선 문제입니다.")
    print("    (arduino/buzzer_only_test/ 스케치로 부저만 따로 확인할 수 있습니다)")
    print()
    while True:
        try:
            key = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if key == "q":
            break
        elif key == "a":
            # send_state는 변화 감지 없이 무조건 보낸다 — 손으로 확인할 때는
            # 누를 때마다 실제로 나가야 진단이 된다(update_state는 엣지 트리거).
            comm.send_state(STATE_FALL)
            print("    -> fall")
        elif key == "s":
            comm.send_state(STATE_NORMAL)
            print("    -> normal")
        elif key == "1":
            comm.send_state(STATE_ZONE, 1)
            print("    -> zone1  (방금 진입 — 가장 많이 남음)")
        elif key == "3":
            comm.send_state(STATE_ZONE, 3)
            print("    -> zone3  (정중앙)")
        elif key == "5":
            comm.send_state(STATE_ZONE, 5)
            print("    -> zone5  (거의 다 건넘)")
        elif key == "p":
            comm.ping()
            print("    -> PING")
        else:
            print("    a / s / 1 / 3 / 5 / p / q 중에서 입력하세요.")
            continue

        # 응답을 기다렸다가 보여준다. 아두이노 왕복은 보통 수 ms지만 넉넉히 준다.
        deadline = time.monotonic() + 0.6
        got = []
        while time.monotonic() < deadline:
            got.extend(comm.poll())
            if got:
                break
            time.sleep(0.02)
        for line in got:
            print(f"    <- {line}")
        if not got and key == "p":
            print("    <- (PONG 없음)  스케치가 안 올라갔거나 보드레이트가 다릅니다.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default=config.SERIAL_PORT,
                        help="시리얼 포트. 생략하면 config.SERIAL_PORT, 그것도 없으면 자동 탐색.")
    parser.add_argument("--baudrate", type=int, default=config.SERIAL_BAUDRATE,
                        help="보드레이트. 스케치의 Serial.begin() 값과 같아야 한다.")
    parser.add_argument("--seconds", type=float, default=5.0,
                        help="FALL 후 부저를 울려 둘 시간(초).")
    parser.add_argument("--interactive", action="store_true",
                        help="키로 FALL/NORMAL을 직접 조작한다.")
    parser.add_argument("--watchdog", action="store_true",
                        help="하트비트를 끊어 아두이노 30초 자동 정지가 도는지 확인한다.")
    parser.add_argument("--list", action="store_true", help="포트 목록만 출력하고 종료.")
    args = parser.parse_args()

    # --port 없이도 SerialComm이 자동 탐색한다. 포트가 아예 없을 때만 미리 걸러
    # "연결된 포트가 없다"를 보여준다(그 상태로 진행해봐야 같은 말을 하게 된다).
    if args.list or (args.port is None and not _has_any_port()):
        print_ports()
        return

    print(f"[안내] 포트 {args.port or '(자동 탐색)'} / {args.baudrate}bps 로 연결합니다.")
    print("[안내] 아두이노가 리셋되어 부팅할 때까지 잠시 기다립니다...\n")

    comm = SerialComm(port=args.port, baudrate=args.baudrate)
    try:
        comm.open()
    except RuntimeError as exc:
        print(f"연결 실패:\n{exc}\n")
        print_ports()
        return

    print(f"연결됨: {comm.port}\n")
    try:
        if args.watchdog:
            run_watchdog(comm)
        elif args.interactive:
            run_interactive(comm)
        else:
            run_once(comm, args.seconds)
    except KeyboardInterrupt:
        print("\n중단됨 (Ctrl+C).")
    finally:
        # close()가 NORMAL을 보내므로 어떤 경로로 끝나도 부저는 멎는다.
        comm.close()


def _has_any_port() -> bool:
    """연결된 시리얼 포트가 하나라도 있는가."""
    try:
        return len(SerialComm.available_ports()) >= 1
    except Exception:
        return False


if __name__ == "__main__":
    main()
