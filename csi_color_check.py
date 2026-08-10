"""CSI 카메라 채널 순서 확정용 진단 (Pi에서 실행).

목적: picamera2가 주는 배열이 BGR인지 RGB인지를 '추측'이 아니라 '확인'으로 끝낸다.
      camera_test.py와 crosswalk_poc.py가 서로 반대로 처리하고 있어서,
      어느 쪽이 맞는지 정하지 않으면 튜닝한 값이 본 프로그램에서 안 맞는다.

실행:  python csi_color_check.py
준비:  빨간 물체(포스트잇, 테이프, 옷 등)를 카메라 앞 화면 가득 채우고 실행할 것.
       빨강을 쓰는 이유는 R과 B가 바뀌면 결과가 가장 확실하게 갈리기 때문이다.
"""

import sys

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
except ImportError:
    sys.exit("picamera2 없음 — sudo apt install -y python3-picamera2\n"
             "venv라면: python -m venv --system-site-packages venv")

import picamera2

print(f"picamera2 버전: {getattr(picamera2, '__version__', '알 수 없음')}")
print()


def probe(label, config_kwargs):
    """설정 하나를 열어서 실제 포맷과 채널별 평균을 찍는다."""
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(**config_kwargs))
    cam.start()
    for _ in range(8):          # AE/AWB 안정화
        cam.capture_array()
    frame = cam.capture_array()
    actual = cam.camera_configuration()["main"]["format"]
    cam.stop()
    cam.close()

    print(f"[{label}]")
    print(f"  요청 포맷 : {config_kwargs.get('main', {}).get('format', '(미지정 = 라이브러리 기본값)')}")
    print(f"  실제 포맷 : {actual}")
    print(f"  배열      : shape={frame.shape} dtype={frame.dtype}")

    if frame.ndim != 3 or frame.shape[2] < 3:
        print("  ** 3채널이 아님 — cv2.cvtColor(..., COLOR_RGB2BGR)는 여기서 예외가 난다")
        return None

    ch = [float(frame[:, :, i].mean()) for i in range(3)]
    print(f"  채널 평균 : [0]={ch[0]:.1f}  [1]={ch[1]:.1f}  [2]={ch[2]:.1f}")
    # OpenCV는 배열을 BGR로 해석한다. 빨간 물체를 비췄다면:
    #   배열이 BGR이면 -> 마지막 채널[2]이 가장 크다  (변환 불필요)
    #   배열이 RGB이면 -> 첫 채널[0]이 가장 크다      (변환 필요)
    order = "BGR (OpenCV와 동일 — 변환 불필요)" if ch[2] > ch[0] else \
            "RGB (OpenCV와 반대 — 변환 필요)"
    print(f"  => 판정   : {order}")
    print(f"     (빨간 물체를 비춘 경우에만 유효. 차이 {abs(ch[2]-ch[0]):.1f})")

    out = f"csi_check_{label}.jpg"
    cv2.imwrite(out, frame[:, :, :3])   # imwrite는 BGR로 해석해 저장
    print(f"  저장      : {out}  <- 빨간 물체가 '빨갛게' 보이면 변환 불필요")
    print()
    return frame


w, h = 640, 480
probe("explicit_RGB888", {"main": {"size": (w, h), "format": "RGB888"}})
probe("library_default", {"main": {"size": (w, h)}})

print("판정 방법")
print("  1. csi_check_explicit_RGB888.jpg 를 연다")
print("  2. 빨간 물체가 빨갛게 보이면 -> crosswalk_poc.py 그대로 두면 됨")
print("     파랗게 보이면          -> CONFIG['csi_swap_rb'] = True 로 바꿀 것")
print("  3. 어느 쪽이든 camera_test.py 의 cvtColor 한 줄은 여기에 맞춰야 한다")
