"""카메라 입력 캡처.

목표 1(신호 연장)과 목표 2(쓰러짐 감지)가 공유할 수 있도록 프레임 획득만 담당하고,
판단 로직은 포함하지 않는다.

## 백엔드가 두 개인 이유

라즈베리파이 5는 Raspberry Pi OS Bookworm 이상만 지원하고, Bookworm에서는 레거시 카메라
스택(`bcm2835-v4l2`)이 제거됐다. 그래서 **CSI 리본 카메라는 `/dev/video0`으로 잡히지 않고
`cv2.VideoCapture(0)`이 실패한다.** libcamera 기반 `Picamera2`를 써야 한다.

반대로 USB 웹캠·영상 파일·네트워크 스트림은 `cv2.VideoCapture`가 다 처리한다.
그래서 소스 종류에 따라 백엔드를 고른다.

    config.CAMERA_SOURCE = "picamera2"          -> Picamera2  (파이 CSI 카메라)
    config.CAMERA_SOURCE = 0, 1, ...            -> cv2        (USB 웹캠)
    config.CAMERA_SOURCE = "video.mp4"          -> cv2        (테스트 영상)
    config.CAMERA_SOURCE = "http://파이IP:8000/" -> cv2        (파이 MJPEG 스트림)

어느 백엔드든 `read_frame()`이 돌려주는 것은 **BGR 순서의 numpy 배열**로 동일하다
(OpenCV·ultralytics가 기대하는 형식). 따라서 이 아래 단계(detection, zone, speed)는
카메라가 무엇인지 몰라도 된다.

## 해상도 주의

호모그래피와 zone 좌표는 **캘리브레이션 당시 해상도에 종속**된다. 운영 때 해상도가 다르면
좌표가 통째로 어긋난다. 그래서 해상도를 `config.CAMERA_RESOLUTION` 한 곳에서 정하고
두 백엔드에 동일하게 적용한다.
"""

import cv2

from config import config

# config.CAMERA_SOURCE에 이 문자열을 넣으면 Picamera2 백엔드를 쓴다.
PICAMERA2_SOURCE = "picamera2"


class _OpenCVBackend:
    """cv2.VideoCapture 기반. USB 웹캠 / 영상 파일 / 네트워크 스트림."""

    name = "cv2"

    def __init__(self, source, resolution=None):
        self.source = source
        self.resolution = resolution
        self._cap = None

    def open(self):
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(
                f"카메라를 열 수 없습니다 (cv2): {self.source}\n"
                "- USB 웹캠이면 인덱스(0, 1, ...)를 확인하세요.\n"
                "- 라즈베리파이 CSI 리본 카메라라면 cv2로는 열리지 않습니다.\n"
                f"  config.CAMERA_SOURCE = \"{PICAMERA2_SOURCE}\" 로 바꿔 Picamera2를 쓰세요.\n"
                "- 파이 스트림이면 파이에서 pi_camera_server.py가 켜져 있고 URL/IP가 맞는지 확인하세요."
            )
        if self.resolution:
            width, height = self.resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read_frame(self):
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class _Picamera2Backend:
    """Picamera2(libcamera) 기반. 라즈베리파이 CSI 리본 카메라 전용.

    주의: `format="RGB888"`은 이름과 달리 메모리상 **BGR 순서** 배열을 준다(libcamera의
    픽셀 포맷 명명 규칙이 반대다). 결과적으로 cv2.VideoCapture와 같은 BGR 배열이 나오므로
    cv2.imshow와 ultralytics에 그대로 넘길 수 있다. RGB로 바꾸면 색이 뒤집히니 건드리지 말 것.
    """

    name = "picamera2"

    def __init__(self, resolution=None):
        self.resolution = resolution
        self._picam2 = None

    def open(self):
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "Picamera2를 import할 수 없습니다.\n"
                "- 라즈베리파이에서 실행 중인지 확인하세요 (PC에는 설치되지 않습니다).\n"
                "- 파이라면: sudo apt install -y python3-picamera2\n"
                "- venv를 쓴다면 시스템 패키지가 보이도록 --system-site-packages 로 만들어야 합니다."
            ) from exc

        self._picam2 = Picamera2()
        main = {"format": "RGB888"}   # 위 주석 참고: 실제로는 BGR 배열이 나온다.
        if self.resolution:
            main["size"] = tuple(self.resolution)
        self._picam2.configure(self._picam2.create_preview_configuration(main=main))
        self._picam2.start()

    def read_frame(self):
        return self._picam2.capture_array()

    def close(self):
        if self._picam2 is not None:
            self._picam2.stop()
            self._picam2.close()
            self._picam2 = None


def _is_picamera2_source(source) -> bool:
    return isinstance(source, str) and source.strip().lower() == PICAMERA2_SOURCE


class CameraCapture:
    """소스 종류에 맞는 백엔드를 골라 프레임을 공급한다.

    백엔드가 무엇이든 read_frame()은 BGR numpy 배열(또는 프레임이 없으면 None)을 준다.
    """

    def __init__(self, source=None, resolution=None):
        self.source = source if source is not None else config.CAMERA_SOURCE
        self.resolution = (
            resolution if resolution is not None else config.CAMERA_RESOLUTION
        )
        self._backend = self._make_backend()
        self._opened = False

    def _make_backend(self):
        if _is_picamera2_source(self.source):
            return _Picamera2Backend(resolution=self.resolution)
        return _OpenCVBackend(self.source, resolution=self.resolution)

    @property
    def backend_name(self) -> str:
        """"cv2" 또는 "picamera2". 어느 경로로 열렸는지 로그/화면에 찍을 때 쓴다."""
        return self._backend.name

    def open(self):
        self._backend.open()
        self._opened = True
        return self

    def read_frame(self):
        if not self._opened:
            raise RuntimeError("open()을 먼저 호출해야 합니다.")
        return self._backend.read_frame()

    def frames(self):
        """프레임을 순차적으로 반환하는 제너레이터. 소스가 끝나면 멈춘다."""
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame

    def close(self):
        self._backend.close()
        self._opened = False

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()


def grab_one_frame(source=None, resolution=None):
    """프레임 한 장만 얻고 카메라를 닫는다.

    캘리브레이션처럼 정지 화면 한 장만 필요한 곳에서 쓴다. 이미지 파일 경로를 주면
    cv2.imread로 바로 읽는다(카메라를 열 필요가 없다).
    """
    from pathlib import Path

    if isinstance(source, str) and not _is_picamera2_source(source):
        path = Path(source)
        if path.exists() and path.is_file() and path.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        }:
            frame = cv2.imread(source)
            if frame is None:
                raise RuntimeError(f"이미지를 읽을 수 없습니다: {source}")
            return frame

    with CameraCapture(source=source, resolution=resolution) as camera:
        frame = camera.read_frame()
        if frame is None:
            raise RuntimeError(f"프레임을 읽을 수 없습니다: {source}")
        return frame
