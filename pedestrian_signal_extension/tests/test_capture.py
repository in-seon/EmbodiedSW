"""CameraCapture 백엔드 선택 테스트."""

import numpy as np
import pytest

from src.capture import PICAMERA2_SOURCE, CameraCapture, _is_picamera2_source, grab_one_frame


def test_usb_webcam_index_uses_cv2():
    assert CameraCapture(source=0).backend_name == "cv2"
    assert CameraCapture(source=1).backend_name == "cv2"


def test_video_file_uses_cv2():
    assert CameraCapture(source="data/test.mp4").backend_name == "cv2"


def test_stream_url_uses_cv2():
    assert CameraCapture(source="http://192.168.0.15:8000/").backend_name == "cv2"


def test_picamera2_source_uses_picamera2():
    assert CameraCapture(source=PICAMERA2_SOURCE).backend_name == "picamera2"


def test_picamera2_source_is_case_and_space_insensitive():
    for value in ["picamera2", "PiCamera2", "PICAMERA2", "  picamera2  "]:
        assert _is_picamera2_source(value) is True, value


def test_other_strings_are_not_picamera2():
    for value in ["0", "picamera", "camera2", "http://x/", ""]:
        assert _is_picamera2_source(value) is False, value


def test_non_string_sources_are_not_picamera2():
    assert _is_picamera2_source(0) is False
    assert _is_picamera2_source(None) is False


def test_read_before_open_raises():
    camera = CameraCapture(source=0)
    with pytest.raises(RuntimeError, match="open"):
        camera.read_frame()


def test_defaults_come_from_config():
    from config import config

    camera = CameraCapture()
    assert camera.source == config.CAMERA_SOURCE
    assert camera.resolution == config.CAMERA_RESOLUTION


def test_explicit_arguments_override_config():
    camera = CameraCapture(source=PICAMERA2_SOURCE, resolution=(1280, 720))
    assert camera.source == PICAMERA2_SOURCE
    assert camera.resolution == (1280, 720)


def test_grab_one_frame_reads_image_file_without_opening_camera(tmp_path):
    import cv2

    path = tmp_path / "frame.png"
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[10, 20] = (0, 0, 255)
    cv2.imwrite(str(path), image)

    frame = grab_one_frame(str(path))
    assert frame.shape == (48, 64, 3)
    assert tuple(frame[10, 20]) == (0, 0, 255)


def test_grab_one_frame_rejects_unreadable_image(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"not actually a png")

    with pytest.raises(RuntimeError, match="이미지를 읽을 수 없습니다"):
        grab_one_frame(str(path))


def test_missing_picamera2_gives_actionable_error():
    try:
        import picamera2  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("picamera2가 설치된 환경(라즈베리파이)이라 실패 경로를 볼 수 없음")

    camera = CameraCapture(source=PICAMERA2_SOURCE)
    with pytest.raises(RuntimeError, match="Picamera2"):
        camera.open()


def test_normalize_source_converts_webcam_index_to_int():
    from src.capture import normalize_source

    assert normalize_source("0") == 0
    assert normalize_source("1") == 1


def test_normalize_source_leaves_non_numeric_sources_alone():
    from src.capture import normalize_source

    assert normalize_source("picamera2") == "picamera2"
    assert normalize_source("clips/test.mp4") == "clips/test.mp4"
    assert normalize_source("http://raspberrypi.local:8000/") == "http://raspberrypi.local:8000/"
    assert normalize_source(0) == 0
    assert normalize_source(None) is None


def test_main_normalizes_webcam_index_before_opening_camera():
    import main
    from src.capture import CameraCapture, normalize_source

    assert CameraCapture(source=normalize_source("0")).backend_name == "cv2"
    assert (
        "normalize_source" in main.main.__code__.co_names
        or hasattr(main, "normalize_source")
    ), "main()이 --source를 normalize_source로 변환하지 않습니다."
