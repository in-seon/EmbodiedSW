"""라즈베리파이 <-> 아두이노(제어부) 시리얼 통신.

부저·신호등 LED·7세그먼트를 **한 보드**가 담당하므로 시리얼 채널도 하나다.
이 파일이 그 채널 전체를 맡는다.

## 프로토콜: 개행으로 끝나는 ASCII 한 줄

    파이 -> 아두이노      아두이노 -> 파이
    ---------------      --------------------------
    ALERT                READY       (부팅 완료)
    STOP                 OK <명령>   (수락함)
    PING                 PONG
                         ERR <명령>  (모르는 명령)
                         TIMEOUT     (워치독이 알람을 껐음)

지금 구현된 것은 쓰러짐 알람(ALERT/STOP)과 연결 확인(PING)뿐이다. 신호 연장·잔여 시간·
사이클 이벤트는 아직 팀 합의 전이라 아래쪽에 스텁으로 남아 있다.

## 나중에 명령을 추가하는 법 — 고칠 곳은 두 군데뿐이다

    파이 -> 아두이노 : 메서드 하나를 만들고 _send("EXTEND 5") 를 호출
    아두이노 -> 파이 : _handle_line() 에 분기 하나를 추가

수신을 poll() 한 곳으로 모아 둔 이유가 이것이다. 한 채널로 여러 종류의 줄이 섞여 오므로,
"필요할 때 readline() 한 번" 방식으로 짜면 명령이 늘어날 때마다 서로의 응답을 잡아먹는다
(예: 잔여 시간을 읽으려다 OK ALERT를 먹어버림). 도착한 줄을 전부 회수해 종류별로
분류하는 지점이 하나면 그런 일이 생기지 않는다.

## 왜 논블로킹인가

이 객체의 메서드는 비전 루프(프레임당 1회) 안에서 호출된다. 응답을 기다리며 블로킹하면
그만큼 FPS가 떨어지고, 그건 곧 검출·속도 추정의 품질 저하다. 그래서 timeout=0으로 열고
"지금 도착해 있는 바이트만" 읽는다. 예외는 open() 하나뿐인데, 그때는 아직 루프가
시작되기 전이라 기다려도 된다.
"""

import time

import serial

from config import config

# 아두이노 정품/호환 보드의 USB VID. 자동 탐색에서 우선 후보로 쓴다.
#   0x2341 Arduino SA, 0x2A03 Arduino org, 0x1A86 CH340 클론, 0x0403 FTDI
_ARDUINO_VIDS = (0x2341, 0x2A03, 0x1A86, 0x0403)


class SerialComm:
    """제어부 아두이노와의 단일 시리얼 채널.

    connection 인자에 객체를 주면 그것을 그대로 쓴다(테스트용 가짜 시리얼 주입).
    """

    def __init__(self, port=None, baudrate=None, ready_timeout_sec=None,
                 heartbeat_sec=None, connection=None):
        self.port = port if port is not None else config.SERIAL_PORT
        self.baudrate = baudrate if baudrate is not None else config.SERIAL_BAUDRATE
        self.ready_timeout_sec = (
            ready_timeout_sec if ready_timeout_sec is not None
            else config.SERIAL_READY_TIMEOUT_SEC
        )
        self.heartbeat_sec = (
            heartbeat_sec if heartbeat_sec is not None else config.ALARM_HEARTBEAT_SEC
        )
        if self.baudrate is None:
            raise NotImplementedError(
                "SERIAL_BAUDRATE가 설정되지 않았습니다. 아두이노 스케치의 Serial.begin() 값과 맞춰주세요."
            )

        self._injected = connection
        self._conn = None
        self._rx = b""              # 아직 개행을 못 만난 수신 바이트
        self._alarm_on = False      # 파이가 아는 알람 상태 (아두이노 실제 상태의 추정)
        self._last_alert_sent = 0.0
        self.ready = False          # READY 또는 PONG을 받아 연결이 확인됐는지
        self.recent_lines = []      # 최근 수신 줄 (진단·도구 표시용, 최대 50줄)

    # ------------------------------------------------------------------
    # 연결
    # ------------------------------------------------------------------

    @staticmethod
    def available_ports():
        """연결된 시리얼 포트 목록 [(device, description)]."""
        from serial.tools import list_ports

        return [(p.device, p.description or "") for p in list_ports.comports()]

    @staticmethod
    def _autodetect_port():
        """아두이노로 보이는 포트를 하나 고른다. 확실하지 않으면 실패시킨다.

        엉뚱한 장치(블루투스 가상 포트 등)를 잘못 잡으면 "열리긴 했는데 아무 반응이 없는"
        상태가 되어 원인을 찾기 어렵다. 후보가 정확히 하나일 때만 자동으로 고르고,
        아니면 목록을 보여주며 명시적으로 지정하게 한다.
        """
        from serial.tools import list_ports

        ports = list(list_ports.comports())
        if not ports:
            raise RuntimeError(
                "시리얼 포트를 찾을 수 없습니다. 아두이노가 USB로 연결돼 있는지 확인하세요."
            )
        candidates = [p for p in ports if p.vid in _ARDUINO_VIDS]
        if len(candidates) != 1:
            # VID로 못 좁혔으면 포트가 딱 하나일 때만 그것을 쓴다.
            candidates = ports if len(ports) == 1 else candidates
        if len(candidates) != 1:
            listing = "\n".join(f"  - {p.device}  {p.description}" for p in ports)
            raise RuntimeError(
                "아두이노 포트를 자동으로 특정할 수 없습니다. config.SERIAL_PORT에 직접 지정하세요.\n"
                f"연결된 포트:\n{listing}"
            )
        return candidates[0].device

    def open(self):
        """포트를 열고 아두이노가 부팅해 READY를 보낼 때까지 기다린다."""
        if self._injected is not None:
            self._conn = self._injected
        else:
            port = self.port or self._autodetect_port()
            try:
                # timeout=0 -> 논블로킹 읽기 (모듈 docstring "왜 논블로킹인가" 참고)
                self._conn = serial.Serial(port, self.baudrate, timeout=0)
            except serial.SerialException as exc:
                # pyserial의 원래 메시지는 원인을 거의 알려주지 않는다("could not open port").
                # 실제로 겪는 원인은 대부분 아래 셋 중 하나다.
                raise RuntimeError(
                    f"시리얼 포트를 열 수 없습니다: {port}\n"
                    f"  ({exc})\n"
                    "  - 권한: 라즈베리파이에서는 사용자가 dialout 그룹에 있어야 합니다.\n"
                    "      sudo usermod -aG dialout $USER   (실행 후 재로그인)\n"
                    "  - 점유: 아두이노 IDE의 시리얼 모니터가 켜져 있으면 포트를 잡고 있습니다. 닫으세요.\n"
                    "  - 포트: 케이블이 빠졌거나 이름이 바뀌었을 수 있습니다.\n"
                    "      python tools/manual_buzzer_check.py --list 로 목록을 확인하세요."
                ) from exc
            self.port = port

        self._rx = b""
        self._alarm_on = False
        self.ready = False
        self._wait_until_ready()
        return self

    def _wait_until_ready(self):
        """READY를 기다리고, 안 오면 PING으로 한 번 더 확인한다.

        보드에 따라(또는 DTR 자동 리셋이 꺼져 있으면) 포트를 열어도 리셋되지 않아 READY가
        오지 않는다. 그때도 이미 돌고 있는 스케치는 PING에 PONG으로 답하므로, 그것으로
        연결을 확인한다. 둘 다 실패하면 여기서 실패시킨다 — 조용히 진행하면 나중에
        "부저가 안 울린다"의 원인이 배선인지 포트인지 알 수 없게 된다.
        """
        deadline = time.monotonic() + self.ready_timeout_sec
        while time.monotonic() < deadline:
            self.poll()
            if self.ready:
                return
            time.sleep(0.05)

        self._send("PING")
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            self.poll()
            if self.ready:
                return
            time.sleep(0.05)

        raise RuntimeError(
            f"아두이노가 응답하지 않습니다 (포트 {self.port}, {self.baudrate}bps).\n"
            "  - 보드레이트가 스케치의 Serial.begin() 값과 같은지\n"
            "  - 시리얼 모니터 등 다른 프로그램이 포트를 잡고 있지 않은지\n"
            "  - 포트가 맞는지 (SerialComm.available_ports()로 목록 확인)\n"
            "확인하세요."
        )

    def close(self):
        """알람을 끄고 포트를 닫는다.

        STOP을 먼저 보내는 이유: 파이 쪽 프로그램이 끝나도 아두이노는 계속 울린다.
        30초 워치독이 결국 끄긴 하지만, 그때까지 부저가 울리는 건 그냥 고장으로 보인다.
        """
        if self._conn is None:
            return
        try:
            if self._alarm_on:
                self._send("STOP")
                if hasattr(self._conn, "flush"):
                    self._conn.flush()
        except Exception:
            pass  # 이미 케이블이 빠진 경우 등 — 닫는 것이 우선이다.
        finally:
            self._conn.close()
            self._conn = None
            self._alarm_on = False
            self.ready = False

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------
    # 전송 계층
    # ------------------------------------------------------------------

    def _send(self, line: str):
        if self._conn is None:
            raise RuntimeError("시리얼이 열려 있지 않습니다. open()을 먼저 호출하세요.")
        self._conn.write((line + "\n").encode("ascii"))

    def poll(self):
        """도착해 있는 줄을 전부 읽어 상태에 반영한다. 매 프레임 호출해도 된다(논블로킹).

        반환: 이번에 읽은 줄 목록(진단용). 보통은 반환값을 쓰지 않는다.
        """
        if self._conn is None:
            return []

        waiting = getattr(self._conn, "in_waiting", 0)
        if waiting:
            self._rx += self._conn.read(waiting)

        lines = []
        while b"\n" in self._rx:
            raw, self._rx = self._rx.split(b"\n", 1)
            # 아두이노 println은 "\r\n"으로 끝나므로 \r도 함께 털어낸다.
            text = raw.decode("ascii", errors="replace").strip()
            if text:
                lines.append(text)
                self._handle_line(text)

        if lines:
            self.recent_lines.extend(lines)
            del self.recent_lines[:-50]
        return lines

    def _handle_line(self, line: str):
        """아두이노가 보낸 줄 하나를 해석한다. **명령을 추가할 때 여기에 분기를 넣는다.**"""
        if line == "READY" or line == "PONG":
            self.ready = True
            if line == "READY":
                # 아두이노가 방금 재부팅했다 -> 알람 상태도 초기화됐다.
                self._alarm_on = False
        elif line == "TIMEOUT":
            # 워치독이 알람을 껐다. 파이 쪽 상태를 맞춰 두어야, 알람이 계속 필요한
            # 상황이면 다음 update_alarm()에서 곧바로 ALERT를 다시 보낸다.
            self._alarm_on = False
        # 앞으로 추가할 것:
        #   elif line.startswith("REMAIN "):  self._remaining_sec = int(line[7:])
        #   elif line == "CYCLE":             self._cycle_started = True

    # ------------------------------------------------------------------
    # 쓰러짐 알람 (구현 완료)
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """PING을 보내고 이미 받아 둔 응답 기준으로 연결 상태를 돌려준다(논블로킹)."""
        self._send("PING")
        self.poll()
        return self.ready

    def update_alarm(self, active: bool, now=None):
        """쓰러짐 알람 상태를 아두이노에 반영한다. 매 프레임 호출하는 것을 전제로 한다.

        active: 쓰러짐이 확정된 상태인가 (FallDetectionPipeline의 fall_confirmed).

        전송 규칙:
          - 꺼짐 -> 켜짐 : ALERT
          - 켜진 동안    : heartbeat_sec마다 ALERT 재전송 (아두이노 30초 워치독 갱신)
          - 켜짐 -> 꺼짐 : STOP

        매 프레임 ALERT를 보내지 않는 이유는 신호 연장의 엣지 트리거와 같다 — 9600bps에
        초당 수십 줄을 밀어 넣으면 버퍼가 밀리고, 정작 필요한 다른 메시지가 뒤로 밀린다.

        반환: 이번에 실제로 보낸 명령("ALERT"/"STOP") 또는 None.
        """
        now = time.monotonic() if now is None else now
        self.poll()

        if active:
            if not self._alarm_on:
                self._send("ALERT")
                self._alarm_on = True
                self._last_alert_sent = now
                return "ALERT"
            if now - self._last_alert_sent >= self.heartbeat_sec:
                self._send("ALERT")           # 워치독 갱신
                self._last_alert_sent = now
                return "ALERT"
            return None

        if self._alarm_on:
            self._send("STOP")
            self._alarm_on = False
            return "STOP"
        return None

    @property
    def alarm_on(self) -> bool:
        """파이가 아는 현재 알람 상태."""
        return self._alarm_on

    # ------------------------------------------------------------------
    # 신호 연장 (팀 합의 전 — 스텁)
    # ------------------------------------------------------------------

    def read_remaining_time(self) -> int:
        """아두이노가 보내는 현재 잔여 녹색 시간(초).

        구현하려면: 아두이노가 `REMAIN <초>` 를 주기적으로 보내게 하고, _handle_line()에
        분기를 추가해 마지막 값을 저장한 뒤 여기서 돌려주면 된다. 전송 주기와 명령 이름만
        팀과 맞추면 되고, 이 파일의 구조는 그대로다.
        """
        raise NotImplementedError(
            "잔여 녹색 시간 메시지(REMAIN <초>)가 팀과 합의되지 않아 구현 보류. "
            "합의되면 _handle_line()에 분기 하나를 추가하면 된다."
        )

    def read_cycle_started(self) -> bool:
        """새 보행 신호 사이클(녹색 시작)이 시작됐는지 여부.

        신호 사이클의 소유자는 제어부이므로 이 이벤트도 제어부가 알려줘야 한다. 파이가
        잔여 시간의 증감만 보고 추측하면(예: "시간이 갑자기 늘면 새 사이클") 우리가 방금
        요청한 연장이 반영된 것과 구분할 수 없다.

        이 값이 없으면 누적 연장이 사이클을 넘어 남아, 한 번 상한을 찍은 뒤로는 영구히
        연장이 안 된다. 구현하려면 아두이노가 `CYCLE` 한 줄을 보내게 하면 된다.
        """
        raise NotImplementedError(
            "제어부 -> 파이 '새 사이클 시작' 이벤트(CYCLE)가 팀과 합의되지 않아 구현 보류. "
            "이 값이 없으면 누적 연장이 사이클을 넘어 남는다(SignalExtensionPipeline.begin_new_cycle)."
        )

    def send_extend_signal(self, extension_sec: int, priority: bool = False):
        """신호 연장 정보를 아두이노로 전달한다.

        구현하려면 `self._send(f"EXTEND {extension_sec}")` 한 줄이면 되지만, 아두이노 쪽이
        이 명령을 어떻게 처리할지(누적인지 절대값인지, priority를 어떻게 반영할지)가
        먼저 합의돼야 한다. 누적/절대값을 잘못 맞추면 연장량이 의도의 몇 배가 된다.
        """
        raise NotImplementedError(
            "연장 명령(EXTEND <초>)의 의미(누적/절대값)와 priority 처리 방식이 "
            "팀과 합의되지 않아 구현 보류."
        )
