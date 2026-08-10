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
