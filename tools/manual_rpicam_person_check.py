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