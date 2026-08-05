"""라즈베리파이에서 실행하는 MJPEG 카메라 서버.

용도: 카메라는 라즈베리파이에 두고, 무거운 YOLO 추론은 성능 좋은 PC에서 돌리고 싶을 때.
파이가 이 스크립트로 웹캠 프레임을 MJPEG(HTTP)로 내보내면, PC 쪽 코드는
config.CAMERA_SOURCE = "http://<파이IP>:8000/" 로 지정해 CameraCapture로 그대로 받는다.
(cv2.VideoCapture가 MJPEG URL을 직접 읽을 수 있어서 PC 쪽엔 별도 수신 코드가 필요 없다.)

라즈베리파이에서:
    python tools/pi_camera_server.py --source picamera2 --port 8000   # CSI 리본 카메라
    python tools/pi_camera_server.py --source 0 --port 8000           # USB 웹캠
PC의 config.CAMERA_SOURCE:
    "http://raspberrypi.local:8000/"   또는  "http://192.168.x.x:8000/"

카메라를 여는 일은 src/capture.py의 CameraCapture에 맡긴다. 라즈베리파이 5 + Bookworm에는
레거시 카메라 스택이 없어 CSI 리본 카메라가 /dev/video0으로 잡히지 않으므로,
cv2.VideoCapture로는 열 수 없고 Picamera2 경로가 필요하다. 그 분기가 CameraCapture 안에
이미 있으므로 여기서 다시 구현하지 않는다(카메라 여는 코드가 두 벌이 되면 한쪽만 고치게 된다).

주의: 인증 없는 평문 스트림이므로 같은 로컬 네트워크(공유기 안)에서만 사용할 것.
"""

import argparse
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.capture import CameraCapture

_BOUNDARY = "frame"


def make_handler(camera, jpeg_quality, fps_limit):
    min_interval = 1.0 / fps_limit if fps_limit else 0.0
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]

    class MJPEGHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # 접속 로그 억제 (원하면 제거)

        def do_GET(self):
            self.send_response(200)
            self.send_header(
                "Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}"
            )
            self.end_headers()
            last = 0.0
            try:
                while True:
                    if min_interval:
                        wait = min_interval - (time.time() - last)
                        if wait > 0:
                            time.sleep(wait)
                    last = time.time()

                    frame = camera.read_frame()
                    if frame is None:
                        break
                    ok, buf = cv2.imencode(".jpg", frame, encode_params)
                    if not ok:
                        continue
                    chunk = buf.tobytes()
                    self.wfile.write(f"--{_BOUNDARY}\r\n".encode())
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(chunk)))
                    self.end_headers()
                    self.wfile.write(chunk)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass  # 클라이언트(PC)가 연결을 끊음 — 정상

    return MJPEGHandler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="picamera2",
                        help="'picamera2'(CSI 리본 카메라, 기본) / USB 웹캠 인덱스 / 영상 경로")
    parser.add_argument("--host", default="0.0.0.0", help="바인드 주소")
    parser.add_argument("--port", type=int, default=8000, help="포트(기본 8000)")
    parser.add_argument("--quality", type=int, default=80, help="JPEG 품질 1~100(기본 80)")
    parser.add_argument("--fps", type=float, default=15.0, help="전송 FPS 상한(기본 15)")
    args = parser.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source  # "picamera2" 또는 파일 경로

    camera = CameraCapture(source=source).open()
    print(f"카메라 백엔드: {camera.backend_name}  (source={source!r})")

    handler = make_handler(camera, args.quality, args.fps)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"MJPEG 스트림 시작: http://<이 파이의 IP>:{args.port}/  (종료: Ctrl+C)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n종료합니다.")
        server.shutdown()
    finally:
        camera.close()


if __name__ == "__main__":
    main()
