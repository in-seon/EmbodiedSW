"""라즈베리파이 <-> 아두이노(제어부) 시리얼 통신.

부저·신호등 LED·7세그먼트를 **한 보드**가 담당하므로 시리얼 채널도 하나다.

## 프로토콜: 개행으로 끝나는 ASCII 한 줄

    파이 -> 아두이노       아두이노 -> 파이
    ------------------    --------------------------
    normal                READY   (부팅 완료)
    zone<1..N>            PONG    (PING에 대한 답)
    fall
    PING

예: `normal` / `zone2` / `fall`

⚠️ 파이가 보내는 상태는 **소문자**이고 zone 뒤 숫자는 **붙여 쓴다**(`zone2`, `zone 2` 아님).
아두이노가 보내는 것은 대문자다. 방향에 따라 표기가 다르니 파싱할 때 주의할 것.

**아두이노가 파이에게 지시하는 것은 없다.** 파이는 READY/PONG으로 연결만 확인하고,
그 뒤로는 한 방향으로 상태만 흘려보낸다. 모형 보행자 구동은 별도 보드가 맡는다.

## 파이는 '무엇을 보았는가'만 보내고, '언제 적용할까'는 아두이노가 정한다

    파이   : 구역 판정, 잔류 확정, 쓰러짐 확정  -> 위 세 상태로 요약
    아두이노: 잔여 녹색 시간, 임계값 판단, 누적 상한, 사이클 리셋

"남은 시간이 5초 미만인가"는 7세그먼트를 직접 세는 쪽만 답할 수 있고, "이 사람이 몇 번
구역에 있나"는 영상을 보는 쪽만 답할 수 있다. 각자 자기만 아는 것을 판단하도록 나눴다.

## zone 뒤의 숫자는 '진척도'다 — 물리 구역 번호가 아니다

확정 보행자 중 **가장 덜 건넌 사람**의 진척도(1..N)를 보낸다.

    1 = 방금 진입          N = 거의 다 건넜음

물리 구역 번호를 그대로 보내지 않는 이유: 구역 번호는 좌표계 기준이라 진입 방향에 따라
의미가 뒤집힌다. 물리적으로 같은 2번이라도 한쪽에서 온 사람은 대부분 남았고 반대쪽에서
온 사람은 거의 다 건넜다. 파이가 track별 진입 방향을 보정해 보낸다(src/zone.py).

아두이노는 이 숫자로 이렇게 판단한다:

    기준[n] = 기본녹색 x (1 - (2n-1)/(2N))     # 10초/5구역 -> 9,7,5,3,1
    if (잔여시간 < 기준[n]) 조금 연장           # 뒤처졌다

**기준표가 정상 보행 속도를 담고 있어서** 파이가 속도를 잴 필요가 없다. 느린 사람은
기준 대비 지연이 크게 잡혀 자동으로 더 연장받으므로 교통약자 검출도 필요 없다.

## 전송 정책

상태가 바뀔 때 즉시, 그리고 config.SERIAL_STATE_HEARTBEAT_SEC 마다 한 번 더 보낸다.
자세한 이유는 update_state() 참고. 아두이노는 상태 메시지에 **응답하지 않는 것이 좋다**
(응답 송신 시간이 아두이노 루프를 묶는다). PING에만 PONG으로 답하면 된다.

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

# 파이가 보낼 수 있는 상태. 문자열을 여기 모아 두어 오타로 조용히 어긋나지 않게 한다.
# 우선순위: fall > zone > normal (쓰러진 사람이 있으면 연장 요구보다 그쪽이 급하다).
STATE_NORMAL = "normal"
STATE_ZONE = "zone"
STATE_FALL = "fall"


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
        self._last_state_key = (None, None)
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
        """normal로 되돌리고 포트를 닫는다.

        normal을 먼저 보내는 이유: 파이 쪽 프로그램이 끝나도 아두이노는 마지막 상태를
        그대로 들고 있다. fall로 끝나면 부저가 계속 울리고, zone으로 끝나면 다음 사이클에
        의도치 않은 연장이 붙는다. 아두이노의 무신호 워치독이 결국 되돌리겠지만,
        그때까지의 동작은 그냥 고장으로 보인다.
        """
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
        """보낼 한 줄을 만든다. 전송과 분리해 둔 이유는 테스트·도구가 그대로 쓰기 위함이다.

            normal
            zone2
            fall

        **숫자를 붙여 쓴다**(`zone2`). 공백으로 나누면 아두이노가 토큰을 쪼개야 하는데,
        한 줄이 짧을수록 파싱이 단순하고 수신 버퍼도 덜 찬다.

        zone 뒤의 숫자는 **진척도**(1 = 방금 진입, N = 거의 다 건넜음)이지 물리 구역
        번호가 아니다 — 모듈 docstring 참고.
        """
        if state != STATE_ZONE:
            return state
        if zone is None:
            raise ValueError("zone 상태에는 진척도가 필요합니다.")
        return f"{STATE_ZONE}{int(zone)}"

    def send_state(self, state: str, zone=None, now=None) -> str:
        """상태를 **무조건** 한 줄 보낸다 (수동 조작·진단용).

        평상시 루프에서는 update_state()를 쓸 것 — 그쪽이 변화 감지와 하트비트를 한다.

        now를 받는 이유: 하트비트 시각을 여기서 찍는데, 호출자가 쓰는 시계와 다르면
        주기가 어긋난다. 테스트에서 시각을 주입할 수 있어야 하는 이유이기도 하다.
        """
        line = self.format_state(state, zone)
        self._send(line)
        self._last_state_key = (state, zone)
        self._last_state_sent = time.monotonic() if now is None else now
        return line

    def update_state(self, state: str, zone=None, now=None):
        """상태를 반영한다. 매 프레임 호출하는 것을 전제로 한다.

        전송 규칙 — **상태가 바뀔 때 즉시, 그리고 heartbeat_sec마다 한 번 더.**

          - 매 프레임 보내지 않는 이유: 9600bps에 초당 수십 줄을 밀어 넣으면 아두이노의
            64바이트 수신 버퍼가 넘치고, 아두이노가 응답까지 하면 그 송신 시간에 루프가
            묶인다. 정작 중요한 순간의 메시지가 뒤로 밀린다.
          - 그럼에도 주기적으로 재전송하는 이유: 변화 시에만 보내면 그 한 줄을 놓쳤을 때
            아두이노가 **영영 옛 상태로 남는다.** 1초마다 같은 상태를 다시 보내면 유실이
            자동으로 복구되고, 아두이노는 "N초간 아무것도 안 왔다 -> 파이가 죽었다"를
            워치독으로 쓸 수 있다.

        **변화 판정에 ETA는 넣지 않는다.** ETA는 사람이 걷는 동안 매 프레임 조금씩 변하므로
        그것까지 비교하면 결국 매 프레임 전송이 된다. 대신 보낼 때마다 최신 ETA를 싣는다.
        아두이노는 '잔여 5초 미만'인 순간에만 ETA를 쓰므로 1초 신선도면 충분하다.

        반환: 이번에 실제로 보낸 줄 또는 None.
        """
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
