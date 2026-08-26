"""SerialComm 단위 테스트.

가짜 시리얼 객체를 주입해 아두이노 없이 프로토콜 동작을 검증한다.
시각(now)도 인자로 넣을 수 있어 하트비트 주기를 실시간 대기 없이 확인한다.
"""

import pytest

from src.serial_comm import SerialComm


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


# --- 알람 전송 규칙 ---

def test_alert_sent_on_rising_edge_only():
    comm, fake = make_comm("READY")
    comm.open()

    assert comm.update_alarm(True, now=0.0) == "ALERT"
    assert comm.update_alarm(True, now=1.0) is None    # 매 프레임 보내지 않는다
    assert comm.update_alarm(True, now=2.0) is None
    assert fake.sent == ["ALERT"]


def test_stop_sent_on_falling_edge_only():
    comm, fake = make_comm("READY")
    comm.open()

    comm.update_alarm(True, now=0.0)
    assert comm.update_alarm(False, now=1.0) == "STOP"
    assert comm.update_alarm(False, now=2.0) is None   # 이미 꺼진 상태
    assert fake.sent == ["ALERT", "STOP"]


def test_no_command_when_never_active():
    comm, fake = make_comm("READY")
    comm.open()

    assert comm.update_alarm(False, now=0.0) is None
    assert fake.sent == []


def test_heartbeat_refreshes_watchdog():
    """켜져 있는 동안 주기적으로 ALERT를 재전송해 아두이노 30초 타임아웃을 갱신한다."""
    comm, fake = make_comm("READY", heartbeat_sec=10.0)
    comm.open()

    comm.update_alarm(True, now=0.0)
    assert comm.update_alarm(True, now=9.9) is None     # 아직 주기 전
    assert comm.update_alarm(True, now=10.0) == "ALERT"  # 갱신
    assert comm.update_alarm(True, now=19.9) is None
    assert comm.update_alarm(True, now=20.0) == "ALERT"
    assert fake.sent == ["ALERT", "ALERT", "ALERT"]


def test_timeout_line_resyncs_state():
    """워치독이 알람을 껐으면 파이도 그 사실을 반영해 곧바로 다시 켠다.

    반영하지 않으면 파이는 '이미 켜져 있다'고 믿어 다음 하트비트(최대 10초)까지
    부저가 죽어 있게 된다.
    """
    comm, fake = make_comm("READY", heartbeat_sec=10.0)
    comm.open()

    comm.update_alarm(True, now=0.0)
    assert comm.alarm_on is True

    fake.feed("TIMEOUT")                                # 아두이노가 스스로 껐다
    assert comm.update_alarm(True, now=1.0) == "ALERT"  # 즉시 재전송
    assert fake.sent == ["ALERT", "ALERT"]


def test_ready_line_resyncs_state():
    """아두이노가 재부팅하면 알람도 꺼졌으므로 다시 보내야 한다."""
    comm, fake = make_comm("READY")
    comm.open()

    comm.update_alarm(True, now=0.0)
    fake.feed("READY")                                  # 보드 재부팅
    assert comm.update_alarm(True, now=1.0) == "ALERT"


# --- 수신 파싱 ---

def test_poll_splits_lines_and_strips_crlf():
    comm, fake = make_comm("READY")
    comm.open()

    fake.feed("OK ALERT")
    fake.feed("PONG")
    assert comm.poll() == ["OK ALERT", "PONG"]


def test_poll_handles_partial_line():
    """개행이 아직 안 온 조각은 다음 poll까지 들고 있는다."""
    comm, fake = make_comm("READY")
    comm.open()

    fake._rx += b"OK AL"
    assert comm.poll() == []
    fake._rx += b"ERT\r\n"
    assert comm.poll() == ["OK ALERT"]


def test_recent_lines_are_capped():
    comm, fake = make_comm("READY")
    comm.open()

    for i in range(80):
        fake.feed(f"OK {i}")
    comm.poll()
    assert len(comm.recent_lines) == 50
    assert comm.recent_lines[-1] == "OK 79"


# --- 종료 ---

def test_close_stops_alarm():
    """프로그램이 끝나도 부저가 계속 울리면 그냥 고장으로 보인다."""
    comm, fake = make_comm("READY")
    comm.open()

    comm.update_alarm(True, now=0.0)
    comm.close()
    assert fake.sent == ["ALERT", "STOP"]
    assert fake.closed is True


def test_close_without_alarm_sends_nothing():
    comm, fake = make_comm("READY")
    comm.open()
    comm.close()
    assert fake.sent == []
    assert fake.closed is True


def test_context_manager_opens_and_closes():
    fake = FakeSerial("READY")
    with SerialComm(connection=fake) as comm:
        comm.update_alarm(True, now=0.0)
    assert fake.sent == ["ALERT", "STOP"]
    assert fake.closed is True


def test_send_before_open_raises():
    comm = SerialComm(connection=FakeSerial("READY"))
    with pytest.raises(RuntimeError, match="열려 있지 않습니다"):
        comm.update_alarm(True, now=0.0)


# --- 미확정 명령은 여전히 명시적으로 막혀 있다 ---

@pytest.mark.parametrize("call", [
    lambda c: c.read_remaining_time(),
    lambda c: c.read_cycle_started(),
    lambda c: c.send_extend_signal(5),
])
def test_unagreed_commands_still_raise(call):
    comm, _ = make_comm("READY")
    comm.open()
    with pytest.raises(NotImplementedError):
        call(comm)


# --- 수동 조작용 강제 전송 (진단 도구가 쓴다) ---

def test_alert_always_sends_even_when_already_on():
    """update_alarm과 달리 누를 때마다 실제로 나가야 한다 — 안 그러면 진단이 안 된다."""
    comm, fake = make_comm("READY")
    comm.open()

    comm.alert()
    comm.alert()
    comm.alert()
    assert fake.sent == ["ALERT", "ALERT", "ALERT"]
    assert comm.alarm_on is True


def test_stop_always_sends():
    comm, fake = make_comm("READY")
    comm.open()

    comm.stop()
    comm.stop()
    assert fake.sent == ["STOP", "STOP"]
    assert comm.alarm_on is False


def test_manual_alert_keeps_state_for_update_alarm():
    """alert() 후에는 update_alarm(False)가 STOP을 보낼 수 있어야 한다(상태 동기)."""
    comm, fake = make_comm("READY")
    comm.open()

    comm.alert()
    assert comm.update_alarm(False, now=1.0) == "STOP"
    assert fake.sent == ["ALERT", "STOP"]
