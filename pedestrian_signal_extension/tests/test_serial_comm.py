"""SerialComm 단위 테스트.

가짜 시리얼 객체를 주입해 아두이노 없이 프로토콜 동작을 검증한다.
시각(now)도 인자로 넣을 수 있어 하트비트 주기를 실시간 대기 없이 확인한다.
"""

import pytest

from src.serial_comm import (
    STATE_ZONE,
    STATE_FALL,
    STATE_NORMAL,
    SerialComm,
)


class FakeSerial:
    """pyserial Serial 중 SerialComm이 쓰는 부분만 흉내낸다.

    feed()로 '아두이노가 보낸 줄'을 넣고, sent로 '파이가 보낸 명령'을 확인한다.
    """

    def __init__(self, *initial_lines):
        self._rx = b""
        self.sent = []
        self.closed = False
        self.flushed = 0
        for line in initial_lines:
            self.feed(line)

    def feed(self, line):
        """아두이노가 한 줄 보낸 것으로 친다 (println이므로 \\r\\n)."""
        self._rx += (line + "\r\n").encode("ascii")

    @property
    def in_waiting(self):
        return len(self._rx)

    def read(self, n):
        chunk, self._rx = self._rx[:n], self._rx[n:]
        return chunk

    def write(self, data):
        self.sent.append(data.decode("ascii").strip())
        return len(data)

    def flush(self):
        self.flushed += 1

    def close(self):
        self.closed = True


def make_comm(*initial_lines, **kwargs):
    fake = FakeSerial(*initial_lines)
    comm = SerialComm(connection=fake, **kwargs)
    return comm, fake


# --- 연결 ---

def test_open_waits_for_ready():
    comm, fake = make_comm("READY")
    comm.open()
    assert comm.ready is True
    assert fake.sent == []          # READY를 받았으므로 PING을 보낼 필요가 없다


def test_open_falls_back_to_ping_when_no_ready():
    """DTR 자동 리셋이 없어 READY가 안 오는 보드도 PING/PONG으로 확인된다."""
    fake = FakeSerial()
    comm = SerialComm(connection=fake, ready_timeout_sec=0.05)

    original_write = fake.write

    def write_and_answer(data):
        result = original_write(data)
        if data.strip() == b"PING":
            fake.feed("PONG")
        return result

    fake.write = write_and_answer

    comm.open()
    assert comm.ready is True
    assert "PING" in fake.sent


def test_open_raises_when_board_silent():
    """응답이 전혀 없으면 조용히 진행하지 않고 실패시킨다."""
    comm, _ = make_comm(ready_timeout_sec=0.05)
    with pytest.raises(RuntimeError, match="응답하지 않습니다"):
        comm.open()


def test_missing_baudrate_raises(monkeypatch):
    """config가 비어 있으면 임의값으로 열지 않고 명시적으로 실패한다."""
    from config import config as cfg

    monkeypatch.setattr(cfg, "SERIAL_BAUDRATE", None)
    with pytest.raises(NotImplementedError):
        SerialComm(connection=FakeSerial())


# --- 상태 전송 규칙 ---

def test_state_sent_on_change_only():
    comm, fake = make_comm("READY", heartbeat_sec=10.0)
    comm.open()

    assert comm.update_state(STATE_NORMAL, now=0.0) == "normal"
    assert comm.update_state(STATE_NORMAL, now=1.0) is None    # 매 프레임 보내지 않는다
    assert comm.update_state(STATE_NORMAL, now=2.0) is None
    assert fake.sent == ["normal"]


def test_state_change_sends_immediately():
    comm, fake = make_comm("READY", heartbeat_sec=10.0)
    comm.open()

    comm.update_state(STATE_NORMAL, now=0.0)
    assert comm.update_state(STATE_ZONE, 5, now=0.1) == "zone5"
    assert comm.update_state(STATE_FALL, now=0.2) == "fall"
    assert fake.sent == ["normal", "zone5", "fall"]


def test_zone_change_counts_as_change():
    """진척도 3 -> 5는 다른 상황이다. 상태 이름만 비교하면 이 변화를 놓친다."""
    comm, fake = make_comm("READY", heartbeat_sec=10.0)
    comm.open()

    comm.update_state(STATE_ZONE, 3, now=0.0)
    assert comm.update_state(STATE_ZONE, 5, now=0.1) == "zone5"
    assert fake.sent == ["zone3", "zone5"]


def test_same_zone_does_not_resend():
    """진척도가 그대로면 재전송하지 않는다 (하트비트 전까지).

    사람이 움직여도 같은 진척도 안에 머무르면 아두이노 입장에서 같은 상황이다.
    """
    comm, fake = make_comm("READY", heartbeat_sec=10.0)
    comm.open()

    comm.update_state(STATE_ZONE, 7, now=0.0)
    assert comm.update_state(STATE_ZONE, 7, now=0.1) is None
    assert comm.update_state(STATE_ZONE, 7, now=0.2) is None
    assert fake.sent == ["zone7"]


def test_heartbeat_resends_current_zone():
    """하트비트로 재전송할 때는 그 시점의 최신 값을 싣는다 (유실 복구 + 값 갱신)."""
    comm, fake = make_comm("READY", heartbeat_sec=1.0)
    comm.open()

    comm.update_state(STATE_ZONE, 7, now=0.0)
    assert comm.update_state(STATE_ZONE, 7, now=0.5) is None
    assert comm.update_state(STATE_ZONE, 7, now=1.0) == "zone7"
    assert fake.sent == ["zone7", "zone7"]


def test_ready_line_forces_resend():
    """아두이노가 재부팅하면 저쪽 상태가 초기화되므로 하트비트를 기다리지 않고 다시 보낸다."""
    comm, fake = make_comm("READY", heartbeat_sec=100.0)
    comm.open()

    comm.update_state(STATE_FALL, now=0.0)
    fake.feed("READY")                                   # 보드 재부팅
    assert comm.update_state(STATE_FALL, now=0.1) == "fall"
    assert fake.sent == ["fall", "fall"]


# --- 줄 포맷 ---

@pytest.mark.parametrize("args,expected", [
    ((STATE_NORMAL, None), "normal"),
    ((STATE_FALL, None), "fall"),
    ((STATE_ZONE, 5), "zone5"),
    ((STATE_ZONE, 3), "zone3"),
    ((STATE_ZONE, 12), "zone12"),
])
def test_format_state(args, expected):
    assert SerialComm.format_state(*args) == expected


def test_zone_without_number_raises():
    with pytest.raises(ValueError):
        SerialComm.format_state(STATE_ZONE)


# --- 수신 파싱 ---

def test_poll_splits_lines_and_strips_crlf():
    comm, fake = make_comm("READY")
    comm.open()

    fake.feed("PONG")
    fake.feed("ERR FOO")
    assert comm.poll() == ["PONG", "ERR FOO"]


def test_poll_handles_partial_line():
    """개행이 아직 안 온 조각은 다음 poll까지 들고 있는다."""
    comm, fake = make_comm("READY")
    comm.open()

    fake._rx += b"PO"
    assert comm.poll() == []
    fake._rx += b"NG\r\n"
    assert comm.poll() == ["PONG"]


def test_recent_lines_are_capped():
    comm, fake = make_comm("READY")
    comm.open()

    for i in range(80):
        fake.feed(f"ERR {i}")
    comm.poll()
    assert len(comm.recent_lines) == 50
    assert comm.recent_lines[-1] == "ERR 79"


# --- 종료 ---

def test_close_returns_to_normal():
    """프로그램이 끝났는데 FALL이나 EXTEND로 남아 있으면 부저가 계속 울거나
    다음 사이클에 의도치 않은 연장이 붙는다."""
    comm, fake = make_comm("READY")
    comm.open()

    comm.update_state(STATE_FALL, now=0.0)
    comm.close()
    assert fake.sent == ["fall", "normal"]
    assert fake.closed is True


def test_close_from_normal_sends_nothing():
    comm, fake = make_comm("READY")
    comm.open()

    comm.update_state(STATE_NORMAL, now=0.0)
    comm.close()
    assert fake.sent == ["normal"]          # 종료 시 추가 전송 없음
    assert fake.closed is True


def test_context_manager_opens_and_closes():
    fake = FakeSerial("READY")
    with SerialComm(connection=fake) as comm:
        comm.update_state(STATE_ZONE, 5, now=0.0)
    assert fake.sent == ["zone5", "normal"]
    assert fake.closed is True


def test_send_before_open_raises():
    comm = SerialComm(connection=FakeSerial("READY"))
    with pytest.raises(RuntimeError, match="열려 있지 않습니다"):
        comm.update_state(STATE_NORMAL, now=0.0)


def test_send_state_always_sends():
    """수동 조작·진단용 — 누를 때마다 실제로 나가야 확인이 된다."""
    comm, fake = make_comm("READY")
    comm.open()

    comm.send_state(STATE_FALL)
    comm.send_state(STATE_FALL)
    assert fake.sent == ["fall", "fall"]


# --- 아두이노 -> 파이 명령 (START/STOP) ---
#
# 모터가 파이 GPIO에 붙어 있어서 생긴 방향이다. 기동 시점은 신호를 소유한 아두이노만
# 알고, 정지 시점은 영상을 보는 파이만 안다 — src/motor.py MotorGate 참고.

def test_start_command_is_queued():
    comm, fake = make_comm("READY")
    comm.open()
    fake.feed("START")
    comm.poll()
    assert comm.take_commands() == [("START", None)]


def test_start_command_carries_speed_mode():
    comm, fake = make_comm("READY")
    comm.open()
    fake.feed("START 3")
    comm.poll()
    assert comm.take_commands() == [("START", 3)]


def test_stop_command_is_queued():
    comm, fake = make_comm("READY")
    comm.open()
    fake.feed("STOP")
    comm.poll()
    assert comm.take_commands() == [("STOP", None)]


def test_take_commands_drains_the_queue():
    """★ 같은 START를 두 번 처리하면 안전 타임아웃 시계가 리셋돼 안전장치가 무력해진다."""
    comm, fake = make_comm("READY")
    comm.open()
    fake.feed("START")
    comm.poll()
    assert comm.take_commands() == [("START", None)]
    assert comm.take_commands() == []


def test_garbled_mode_falls_back_to_default():
    """회선 노이즈로 인자가 깨져도 비전 루프를 죽이지 않는다 — 기본 속도로 돈다."""
    comm, fake = make_comm("READY")
    comm.open()
    fake.feed("START x9")
    comm.poll()
    assert comm.take_commands() == [("START", None)]


def test_unknown_line_is_not_queued():
    """모르는 줄은 명령으로 오해하지 않는다."""
    comm, fake = make_comm("READY")
    comm.open()
    fake.feed("SOMETHING ELSE")
    comm.poll()
    assert comm.take_commands() == []


def test_ready_and_pong_are_not_commands():
    comm, fake = make_comm("READY")
    comm.open()
    fake.feed("PONG")
    comm.poll()
    assert comm.take_commands() == []


def test_reopen_discards_stale_commands():
    """재연결 시 이전 세션의 START가 남아 있으면, 아두이노가 보내지도 않은 시점에 모터가 돈다."""
    fake = FakeSerial("READY")
    comm = SerialComm(connection=fake)
    comm.open()
    fake.feed("START")
    comm.poll()

    fake.feed("READY")          # 보드 재부팅
    comm.open()
    assert comm.take_commands() == []
