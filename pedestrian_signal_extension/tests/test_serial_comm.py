"""SerialComm 단위 테스트.

가짜 시리얼 객체를 주입해 아두이노 없이 프로토콜 동작을 검증한다.
시각(now)도 인자로 넣을 수 있어 하트비트 주기를 실시간 대기 없이 확인한다.
"""

import pytest

from src.serial_comm import (
    STATE_EXTEND,
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

    assert comm.update_state(STATE_NORMAL, now=0.0) == "NORMAL"
    assert comm.update_state(STATE_NORMAL, now=1.0) is None    # 매 프레임 보내지 않는다
    assert comm.update_state(STATE_NORMAL, now=2.0) is None
    assert fake.sent == ["NORMAL"]


def test_state_change_sends_immediately():
    comm, fake = make_comm("READY", heartbeat_sec=10.0)
    comm.open()

    comm.update_state(STATE_NORMAL, now=0.0)
    assert comm.update_state(STATE_EXTEND, 5, now=0.1) == "EXTEND 5 -"
    assert comm.update_state(STATE_FALL, now=0.2) == "FALL"
    assert fake.sent == ["NORMAL", "EXTEND 5 -", "FALL"]


def test_extend_seconds_change_counts_as_change():
    """3초 -> 5초는 다른 요구다. 상태 이름만 비교하면 이 변화를 놓친다."""
    comm, fake = make_comm("READY", heartbeat_sec=10.0)
    comm.open()

    comm.update_state(STATE_EXTEND, 3, now=0.0)
    assert comm.update_state(STATE_EXTEND, 5, now=0.1) == "EXTEND 5 -"
    assert fake.sent == ["EXTEND 3 -", "EXTEND 5 -"]


def test_eta_change_alone_does_not_resend():
    """ETA는 걷는 동안 매 프레임 변한다 — 변화 판정에 넣으면 매 프레임 전송이 된다."""
    comm, fake = make_comm("READY", heartbeat_sec=10.0)
    comm.open()

    comm.update_state(STATE_EXTEND, 5, eta_sec=7.2, now=0.0)
    assert comm.update_state(STATE_EXTEND, 5, eta_sec=7.1, now=0.1) is None
    assert comm.update_state(STATE_EXTEND, 5, eta_sec=6.9, now=0.2) is None
    assert fake.sent == ["EXTEND 5 7.2"]


def test_heartbeat_resends_with_fresh_eta():
    """하트비트로 재전송할 때는 최신 ETA를 싣는다 — 유실 복구 + 값 갱신을 동시에 한다."""
    comm, fake = make_comm("READY", heartbeat_sec=1.0)
    comm.open()

    comm.update_state(STATE_EXTEND, 5, eta_sec=7.2, now=0.0)
    assert comm.update_state(STATE_EXTEND, 5, eta_sec=7.1, now=0.5) is None
    assert comm.update_state(STATE_EXTEND, 5, eta_sec=6.2, now=1.0) == "EXTEND 5 6.2"
    assert fake.sent == ["EXTEND 5 7.2", "EXTEND 5 6.2"]


def test_ready_line_forces_resend():
    """아두이노가 재부팅하면 저쪽 상태가 초기화되므로 하트비트를 기다리지 않고 다시 보낸다."""
    comm, fake = make_comm("READY", heartbeat_sec=100.0)
    comm.open()

    comm.update_state(STATE_FALL, now=0.0)
    fake.feed("READY")                                   # 보드 재부팅
    assert comm.update_state(STATE_FALL, now=0.1) == "FALL"
    assert fake.sent == ["FALL", "FALL"]


# --- 줄 포맷 ---

@pytest.mark.parametrize("args,expected", [
    ((STATE_NORMAL, None, None), "NORMAL"),
    ((STATE_FALL, None, None), "FALL"),
    ((STATE_EXTEND, 5, None), "EXTEND 5 -"),
    ((STATE_EXTEND, 3, 7.25), "EXTEND 3 7.2"),
    ((STATE_EXTEND, 5, 12.0), "EXTEND 5 12.0"),
])
def test_format_state(args, expected):
    assert SerialComm.format_state(*args) == expected


def test_extend_without_seconds_raises():
    with pytest.raises(ValueError):
        SerialComm.format_state(STATE_EXTEND)


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
    assert fake.sent == ["FALL", "NORMAL"]
    assert fake.closed is True


def test_close_from_normal_sends_nothing():
    comm, fake = make_comm("READY")
    comm.open()

    comm.update_state(STATE_NORMAL, now=0.0)
    comm.close()
    assert fake.sent == ["NORMAL"]          # 종료 시 추가 전송 없음
    assert fake.closed is True


def test_context_manager_opens_and_closes():
    fake = FakeSerial("READY")
    with SerialComm(connection=fake) as comm:
        comm.update_state(STATE_EXTEND, 5, now=0.0)
    assert fake.sent == ["EXTEND 5 -", "NORMAL"]
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
    assert fake.sent == ["FALL", "FALL"]
