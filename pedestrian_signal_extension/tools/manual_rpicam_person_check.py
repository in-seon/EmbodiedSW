from picamera2 import Picamera2
from ultralytics import YOLO
import cv2

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()

model = YOLO("yolov8n-pose.pt")

while True:
    frame = picam2.capture_array()

    results = model(frame, classes=[0], verbose=False)
    annotated_frame = results[0].plot()
    cv2.imshow("Person Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()