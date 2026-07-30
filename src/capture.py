"""카메라 입력 캡처.

목표 1(신호 연장)과 목표 2(쓰러짐 감지)가 공유할 수 있도록 프레임 획득만 담당하고,
판단 로직은 포함하지 않는다.

source(config.CAMERA_SOURCE)로 세 가지 입력을 모두 처리한다 — cv2.VideoCapture가
정수 인덱스/파일 경로/네트워크 URL을 모두 받기 때문이다.
  - 정수 0, 1 ...          : 이 코드를 실행하는 컴퓨터에 직접 연결된 USB 웹캠
  - "video.mp4"           : 테스트용 동영상 파일
  - "http://<파이IP>:8000/" : 라즈베리파이의 MJPEG 스트림 (tools/pi_camera_server.py 참고)
"""

import cv2

from config import config


class CameraCapture:
    def __init__(self, source=None):
        self.source = source if source is not None else config.CAMERA_SOURCE
        self._cap = None

    def open(self):
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"카메라를 열 수 없습니다: {self.source}\n"
                "- USB 웹캠이면 인덱스(0,1,..)를 확인하세요.\n"
                "- 라즈베리파이 스트림이면 파이에서 pi_camera_server.py가 켜져 있고 URL/IP가 맞는지 확인하세요."
            )
        return self

    def read_frame(self):
        if self._cap is None:
            raise RuntimeError("open()을 먼저 호출해야 합니다.")
        ok, frame = self._cap.read()
        if not ok:
            return None
        return frame

    def frames(self):
        """프레임을 순차적으로 반환하는 제너레이터."""
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
