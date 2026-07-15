"""라즈베리파이 -> 아두이노 시리얼 통신.

CLAUDE.md 1, 6: 포트/보드레이트/메시지 포맷이 아직 팀과 합의되지 않았으므로
이 스텁은 연결 관리 인터페이스만 제공한다. 메시지 포맷 확정 후 encode_* 함수를 구현할 것.
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

    def send_extend_signal(self, extension_sec: int):
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
