"""[파이 전용] Picamera2 + YOLO 최소 동작 확인 스크립트 (자동 테스트 아님).

`src/capture.py`에 Picamera2 백엔드가 들어간 뒤로 이 스크립트의 역할은 좁아졌다. 평소에는
`python tools/manual_camera_person_check.py --source picamera2` 를 쓰면 구역·속도까지 함께 볼 수 있다.

그럼에도 남겨두는 이유: 이 파일은 **config도 src도 import하지 않는 20여 줄**이라,
문제가 생겼을 때 원인을 가르는 데 쓴다.

    manual_camera_person_check.py --source picamera2  실패
    + 이 스크립트                                      성공  -> 우리 코드 문제
    + 이 스크립트                                      실패  -> 카메라/드라이버/Picamera2 설치 문제

사용법 (라즈베리파이에서):
    python tools/manual_rpicam_person_check.py
    'q' : 종료
"""

from picamera2 import Picamera2
from ultralytics import YOLO
import cv2

# 카메라 초기화
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()

# YOLOv8n 모델 로드 (처음 실행 시 자동으로 다운로드됨)
model = YOLO("yolov8n.pt")

while True:
    frame = picam2.capture_array()

    # 사람(person, 클래스 0)만 탐지하도록 필터링
    results = model(frame, classes=[0], verbose=False)

    # 탐지 결과를 프레임 위에 그리기
    annotated_frame = results[0].plot()

    cv2.imshow("Person Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()