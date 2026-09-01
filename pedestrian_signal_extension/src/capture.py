import cv2
from config import config

PICAMERA2_SOURCE = "picamera2"

class _OpenCVBackend:

    name = "cv2"

    def __init__(self, source, resolution=None):
        self.source = source
        self.resolution = resolution
        self._cap = None

    def open(self):
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(f"카메라를 열 수 없습니다 (cv2): {self.source}")
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
    name = "picamera2"

    def __init__(self, resolution=None):
        self.resolution = resolution
        self._picam2 = None

    def open(self):
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "Picamera2를 import할 수 없습니다.") from exc

        self._picam2 = Picamera2()
        main = {"format": "RGB888"}   
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


def normalize_source(source):
    if isinstance(source, str):
        try:
            return int(source)
        except ValueError:
            return source
    return source


class CameraCapture:
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
