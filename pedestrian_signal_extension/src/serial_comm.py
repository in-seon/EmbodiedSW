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
        self._rx = b""                  
        self._last_state_key = (None, None)
        self._last_state_sent = 0.0
        self.ready = False         
        self.recent_lines = []    

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
            candidates = ports if len(ports) == 1 else candidates
        if len(candidates) != 1:
            listing = "\n".join(f"  - {p.device}  {p.description}" for p in ports)
            raise RuntimeError(
                "아두이노 포트를 자동으로 특정할 수 없습니다.\n"
                f"연결된 포트:\n{listing}"
            )
        return candidates[0].device

    def open(self):
        if self._injected is not None:
            self._conn = self._injected
        else:
            port = self.port or self._autodetect_port()
            try:
                self._conn = serial.Serial(port, self.baudrate, timeout=0)
            except serial.SerialException as exc:
                raise RuntimeError(f"시리얼 포트를 열 수 없습니다: {port}") from exc
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

        raise RuntimeError(f"아두이노가 응답하지 않습니다 (포트 {self.port}, {self.baudrate}bps).")

    def close(self):
    
        if self._conn is None:
            return
        try:
            if self._last_state_key[0] not in (None, STATE_NORMAL):
                self._send(STATE_NORMAL)
                if hasattr(self._conn, "flush"):
                    self._conn.flush()
        except Exception:
            pass  
        finally:
            self._conn.close()
            self._conn = None
            self._last_state_key = (None, None)
            self.ready = False

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()


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
            text = raw.decode("ascii", errors="replace").strip()
            if text:
                lines.append(text)
                self._handle_line(text)

        if lines:
            self.recent_lines.extend(lines)
            del self.recent_lines[:-50]
        return lines

    def _handle_line(self, line: str):
        if line == "READY" or line == "PONG":
            self.ready = True
            if line == "READY":
                self._last_state_key = (None, None)

    def ping(self) -> bool:
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
        return self._last_state_key
