"""카메라 입력 캡처. 목표 1(신호 연장)과 목표 2(쓰러짐 감지)가 공유할 수 있도록
프레임 획득만 담당하고 판단 로직은 포함하지 않는다."""

import cv2


class CameraCapture:
    def __init__(self, source):
        self.source = source
        self._cap = None

    def open(self):
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"카메라를 열 수 없습니다: {self.source}")

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
