import time

import serial

from config import config

_ARDUINO_VIDS = (0x2341, 0x2A03, 0x1A86, 0x0403)

STATE_NORMAL = "normal"
STATE_ZONE = "zone"
STATE_FALL = "fall"


class SerialComm:

    def __init__(self, port=None, baudrate=None, ready_timeout_sec=None,
                 heartbeat_sec=None, connection=None):
        self.port = port if port is not None else config.SERIAL_PORT
        self.baudrate = baudrate if baudrate is not None else config.SERIAL_BAUDRATE
        self.ready_timeout_sec = (
            ready_timeout_sec if ready_timeout_sec is not None
            else config.SERIAL_READY_TIMEOUT_SEC
        )
        self.heartbeat_sec = (
            heartbeat_sec if heartbeat_sec is not None
            else config.SERIAL_STATE_HEARTBEAT_SEC
        )
        if self.baudrate is None:
            raise NotImplementedError(
                "SERIAL_BAUDRATE가 설정되지 않았습니다. 아두이노 스케치의 Serial.begin() 값과 맞춰주세요."
            )

        self._injected = connection
        self._conn = None
        self._rx = b""                    # 아직 개행을 못 만난 수신 바이트
        # 마지막으로 보낸 (상태, 연장초). 변화 감지에 쓴다 — ETA는 일부러 넣지 않는다.
        self._last_state_key = (None, None)
        self._last_state_sent = 0.0
        self.ready = False          # READY 또는 PONG을 받아 연결이 확인됐는지
        self.recent_lines = []      # 최근 수신 줄 (진단·도구 표시용, 최대 50줄)

    # ------------------------------------------------------------------
    # 연결
    # ------------------------------------------------------------------

    @staticmethod
    def available_ports():
        from serial.tools import list_ports

        return [(p.device, p.description or "") for p in list_ports.comports()]

    @staticmethod
    def _autodetect_port():
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
        self._last_state_key = (None, None)
        self.ready = False
        self._wait_until_ready()
        return self

    def _wait_until_ready(self):
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
    
        if self._conn is None:
            return
        try:
            if self._last_state_key[0] not in (None, STATE_NORMAL):
                self._send(STATE_NORMAL)
                if hasattr(self._conn, "flush"):
                    self._conn.flush()
        except Exception:
            pass  # 이미 케이블이 빠진 경우 등 — 닫는 것이 우선이다.
        finally:
            self._conn.close()
            self._conn = None
            self._last_state_key = (None, None)
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
                # 아두이노가 방금 재부팅했다 -> 저쪽 상태가 초기화됐으므로 우리가 아는
                # '마지막으로 보낸 상태'도 무효다. 비워 두면 다음 update_state()가
                # 하트비트를 기다리지 않고 곧바로 현재 상태를 다시 보낸다.
                self._last_state_key = (None, None)
        # 아두이노 -> 파이 방향으로 메시지가 늘어나면 여기에 분기를 추가한다.
        # (잔여 시간·사이클은 제어부가 소유하므로 파이로 올려보낼 필요가 없다.)

    # ------------------------------------------------------------------
    # 상태 전송 — 이 채널의 본체
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """PING을 보내고 이미 받아 둔 응답 기준으로 연결 상태를 돌려준다(논블로킹)."""
        self._send("PING")
        self.poll()
        return self.ready

    @staticmethod
    def format_state(state: str, zone=None) -> str:
        if state != STATE_ZONE:
            return state
        if zone is None:
            raise ValueError("zone 상태에는 진척도가 필요합니다.")
        return f"{STATE_ZONE}{int(zone)}"

    def send_state(self, state: str, zone=None, now=None) -> str:
        line = self.format_state(state, zone)
        self._send(line)
        self._last_state_key = (state, zone)
        self._last_state_sent = time.monotonic() if now is None else now
        return line

    def update_state(self, state: str, zone=None, now=None):
        now = time.monotonic() if now is None else now
        self.poll()

        key = (state, zone)
        if key != self._last_state_key:
            return self.send_state(state, zone, now=now)
        if now - self._last_state_sent >= self.heartbeat_sec:
            return self.send_state(state, zone, now=now)
        return None

    @property
    def last_state(self):
        """파이가 마지막으로 보낸 (상태, 연장초). 아직 아무것도 안 보냈으면 (None, None)."""
        return self._last_state_key
