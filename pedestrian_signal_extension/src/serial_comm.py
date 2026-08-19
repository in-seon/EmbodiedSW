"""라즈베리파이 <-> 아두이노 시리얼 통신.

CLAUDE.md 1, 6: 포트/보드레이트/메시지 포맷이 아직 팀과 합의되지 않았으므로 이 파일은
연결 관리 인터페이스만 제공한다. 메시지 포맷 확정 후 send_extend_signal / read_remaining_time을 구현할 것.

역할 분담(가정):
  - 잔여 녹색 시간의 '소유자'는 아두이노다. 아두이노가 현재 남은 시간을 파이로 보내주면
    (read_remaining_time) 파이가 연장 여부를 판단하고, 연장 명령을 다시 아두이노로 보낸다(send_extend_signal).
  - 교통약자 우선(priority) 플래그도 이 통신으로 전달해 "처음부터 넉넉한 기본 시간"을 아두이노가 세팅하게 한다.
"""

import serial

from config import config


class SerialComm:
    def __init__(self, port=None, baudrate=None):
        self.port = port or config.SERIAL_PORT
        self.baudrate = baudrate or config.SERIAL_BAUDRATE
        if self.port is None or self.baudrate is None:
            raise NotImplementedError(
                "SERIAL_PORT / SERIAL_BAUDRATE가 설정되지 않았습니다. "
                "팀과 통신 프로토콜(포트, 보드레이트, 메시지 포맷) 합의 후 config에 값을 채워주세요."
            )
        self._conn = None

    def open(self):
        self._conn = serial.Serial(self.port, self.baudrate)
        return self

    def read_remaining_time(self) -> int:
        """아두이노가 보내는 현재 잔여 녹색 시간(초)을 읽는다.

        메시지 포맷이 미정이므로(config.SERIAL_MESSAGE_FORMAT) 실제 파싱은 구현하지 않는다.
        """
        raise NotImplementedError(
            "메시지 포맷(config.SERIAL_MESSAGE_FORMAT)이 팀과 합의되지 않아 구현 보류."
        )

    def read_cycle_started(self) -> bool:
        """새 보행 신호 사이클(녹색 시작)이 시작됐는지 여부.

        신호 사이클의 소유자는 제어부이므로 이 이벤트도 제어부가 알려줘야 한다. 파이가
        잔여 시간의 증감만 보고 추측하면(예: "시간이 갑자기 늘면 새 사이클") 우리가 방금
        요청한 연장이 반영된 것과 구분할 수 없다.

        이 값이 없으면 누적 연장이 사이클을 넘어 남아, 한 번 상한을 찍은 뒤로는 영구히
        연장이 안 된다. 메시지 포맷과 함께 반드시 팀과 합의할 것(docs/team_interface.md).
        """
        raise NotImplementedError(
            "제어부 -> 파이 '새 사이클 시작' 이벤트가 팀과 합의되지 않아 구현 보류. "
            "이 값이 없으면 누적 연장이 사이클을 넘어 남는다(SignalExtensionPipeline.begin_new_cycle)."
        )

    def send_extend_signal(self, extension_sec: int, priority: bool = False):
        """신호 연장 정보를 아두이노로 전달한다.

        메시지 포맷이 미정이므로(config.SERIAL_MESSAGE_FORMAT) 실제 인코딩은 구현하지 않는다.
        """
        raise NotImplementedError(
            "메시지 포맷(config.SERIAL_MESSAGE_FORMAT)이 팀과 합의되지 않아 구현 보류."
        )

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
