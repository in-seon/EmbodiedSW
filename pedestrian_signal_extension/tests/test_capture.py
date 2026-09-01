"""CameraCapture 백엔드 선택 테스트.

실제 카메라를 열지 않고 "어느 백엔드가 골라지는가"만 검증한다. picamera2는 라즈베리파이에만
설치되므로 PC에서도 돌아야 하는 이 테스트는 import를 유발하지 않는다
(_Picamera2Backend는 open() 시점에 지연 import 한다).
"""

import numpy as np
import pytest

from src.capture import PICAMERA2_SOURCE, CameraCapture, _is_picamera2_source, grab_one_frame


# --- 백엔드 선택 ---

def test_usb_webcam_index_uses_cv2():
    assert CameraCapture(source=0).backend_name == "cv2"
    assert CameraCapture(source=1).backend_name == "cv2"


def test_video_file_uses_cv2():
    assert CameraCapture(source="data/test.mp4").backend_name == "cv2"


def test_stream_url_uses_cv2():
    assert CameraCapture(source="http://192.168.0.15:8000/").backend_name == "cv2"


def test_picamera2_source_uses_picamera2():
    """라즈베리파이 5 + Bookworm에서 CSI 카메라를 쓰는 경로."""
    assert CameraCapture(source=PICAMERA2_SOURCE).backend_name == "picamera2"


def test_picamera2_source_is_case_and_space_insensitive():
    """config에 손으로 적는 값이라 대소문자/공백에 걸려 넘어지지 않게 한다."""
    for value in ["picamera2", "PiCamera2", "PICAMERA2", "  picamera2  "]:
        assert _is_picamera2_source(value) is True, value


def test_other_strings_are_not_picamera2():
    for value in ["0", "picamera", "camera2", "http://x/", ""]:
        assert _is_picamera2_source(value) is False, value


def test_non_string_sources_are_not_picamera2():
    assert _is_picamera2_source(0) is False
    assert _is_picamera2_source(None) is False


# --- 사용 순서 ---

def test_read_before_open_raises():
    """open()을 건너뛰면 조용히 None을 주는 대신 명시적으로 실패한다."""
    camera = CameraCapture(source=0)
    with pytest.raises(RuntimeError, match="open"):
        camera.read_frame()


# --- 기본값이 config에서 온다 ---

def test_defaults_come_from_config():
    from config import config

    camera = CameraCapture()
    assert camera.source == config.CAMERA_SOURCE
    assert camera.resolution == config.CAMERA_RESOLUTION


def test_explicit_arguments_override_config():
    camera = CameraCapture(source=PICAMERA2_SOURCE, resolution=(1280, 720))
    assert camera.source == PICAMERA2_SOURCE
    assert camera.resolution == (1280, 720)


# --- grab_one_frame (캘리브레이션용 정지 화면 한 장) ---

def test_grab_one_frame_reads_image_file_without_opening_camera(tmp_path):
    """이미지 경로를 주면 카메라를 열지 않고 바로 읽는다.

    zone_calibrator를 저장해둔 사진으로 다시 돌릴 때 쓰는 경로 —
    카메라가 없는 자리에서도 캘리브레이션을 다시 할 수 있어야 한다.
    """
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


# --- picamera2 미설치 환경에서의 에러 메시지 ---

def test_missing_picamera2_gives_actionable_error():
    """PC에서 실수로 --source picamera2 를 주면 무슨 상황인지 알려줘야 한다.

    (라즈베리파이에서 picamera2가 설치돼 있다면 이 테스트는 건너뛴다.)
    """
    try:
        import picamera2  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("picamera2가 설치된 환경(라즈베리파이)이라 실패 경로를 볼 수 없음")

    camera = CameraCapture(source=PICAMERA2_SOURCE)
    with pytest.raises(RuntimeError, match="Picamera2"):
        camera.open()


# --- normalize_source (명령줄 소스 문자열 -> 백엔드가 기대하는 타입) ---

def test_normalize_source_converts_webcam_index_to_int():
    """--source 0 이 문자열로 남으면 cv2가 '0'이라는 파일을 열려다 실패한다."""
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
    """main.py가 변환을 빼먹으면 --source 0 이 통째로 안 돈다 — 그 회귀를 막는다.

    main()은 실행하면 카메라·시리얼을 잡으므로 호출하지 않고, normalize_source를
    거치는지만 본다(함수 안 지연 import든 모듈 상단 import든 통과).
    """
    import main
    from src.capture import CameraCapture, normalize_source

    assert CameraCapture(source=normalize_source("0")).backend_name == "cv2"
    assert (
        "normalize_source" in main.main.__code__.co_names
        or hasattr(main, "normalize_source")
    ), "main()이 --source를 normalize_source로 변환하지 않습니다."
