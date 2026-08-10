# camera_test.py — 카메라 테스트용 글루 (코어 아님)
# 실행: python camera_test.py --conf 0.4
# 종료: 영상 창에서 q (headless: python camera_test.py --headless, 종료는 Ctrl+C)

import argparse
import os
import time
import numpy as np
import cv2
from ultralytics import YOLO

# ── 1. conf를 커맨드라인 인자로 받기 ─────────────────────
# 현장에서 0.4 → 0.25로 바꿀 때 코드 수정 없이
# python camera_test.py --conf 0.25 로 재실행하면 되게 하는 부분
parser = argparse.ArgumentParser()
parser.add_argument("--conf", type=float, default=0.4)
parser.add_argument("--save", type=str, default="capture.npz")  # 키포인트 저장 파일명
parser.add_argument("--device", type=int, default=0)  # 카메라 장치 번호 (0/1)
parser.add_argument("--camera", choices=["usb", "csi"], default="usb",
                     help="usb=cv2.VideoCapture(V4L2), csi=Picamera2 (라즈베리파이 CSI 카메라)")
parser.add_argument("--headless", action="store_true",
                     help="영상 창을 띄우지 않고 실행 (SSH, 모니터 없는 라즈베리파이용). 종료는 Ctrl+C.")
parser.add_argument("--save-frames", type=int, default=0,
                     help="N프레임마다 스켈레톤이 그려진 프레임을 frames/ 폴더에 jpg로 저장 (0=끔)")
args = parser.parse_args()

# 디스플레이가 아예 없는 리눅스 환경이면 --headless를 깜빡해도
# imshow에서 죽지 않도록 자동으로 headless 모드로 전환한다.
if not args.headless and os.name != "nt" and not os.environ.get("DISPLAY"):
    print("DISPLAY 환경변수가 없어 headless 모드로 자동 전환합니다 (영상 창 생략, Ctrl+C로 종료).")
    args.headless = True

# ── 2. 모델 로드 ─────────────────────────────────────────
# 첫 실행 때 yolov8n-pose.pt(약 6MB)를 자동 다운로드한다.
# 그래서 오늘 밤 집에서 미리 한 번 실행해두라는 것.
model = YOLO("yolov8n-pose.pt")

# ── 3. 카메라 열기 ────────────────────────────────────────
# --camera usb (기본): 0 = 첫 번째 카메라. 노트북 내장캠이 잡히면 1로 바꿔서 USB 웹캠 지정.
# 라즈베리파이에서는 backend를 명시하지 않으면 GStreamer로 잡혀
# 열리지 않거나 매우 느려지는 경우가 있어 CAP_V4L2를 명시한다.
# --camera csi: 라즈베리파이 CSI 카메라는 V4L2로 못 잡으므로 Picamera2로 연다.
picam2 = None
if args.camera == "usb":
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없음 — 장치 번호(--device 0/1) 확인")
else:
    from picamera2 import Picamera2
    picam2 = Picamera2()
    # crosswalk_poc.py의 PiCameraSource와 '똑같은' 설정을 쓴다.
    # 포맷을 지정 안 하면 라이브러리 기본값 XBGR8888(4채널)이 나와서,
    # 여기서 튜닝한 conf 값이 본 프로그램(RGB888 3채널)과 다른 입력 위에서
    # 맞춰진 값이 돼버린다. 색 순서 가정은 한 군데에만 있어야 한다.
    picam2.configure(picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}))
    picam2.start()


def read_frame():
    # usb/csi 어느 쪽이든 (ret, BGR 프레임) 형태로 통일해서 돌려준다.
    if args.camera == "usb":
        return cap.read()
    # picamera2의 "RGB888"은 이름과 달리 numpy 배열에 B,G,R 순으로 담긴다
    # (libcamera 명명 규칙). 그래서 OpenCV 기준으로 변환이 필요 없다.
    # 색이 뒤집혀 보이면 csi_color_check.py를 먼저 돌려 확인하고,
    # crosswalk_poc.py의 CONFIG["csi_swap_rb"]와 '같이' 바꿔야 한다.
    return True, picam2.capture_array()

# ── 4. 수집용 변수 ───────────────────────────────────────
all_kpts = []     # 프레임별 키포인트 (17,2)를 쌓을 리스트
frame_count = 0
t_start = time.time()

# --save-frames N: N프레임마다 스켈레톤이 그려진 프레임을 저장할 폴더 준비
frames_dir = "frames"
if args.save_frames > 0:
    os.makedirs(frames_dir, exist_ok=True)

# ── 5. 메인 루프: 프레임 → 추론 → 표시 → 저장 ───────────
try:
    while True:
        ret, frame = read_frame()
        if not ret:
            break

        # YOLO-pose 추론. conf 미만 검출은 버려진다.
        # verbose=False: 매 프레임 콘솔 로그 끄기
        results = model(frame, conf=args.conf, verbose=False)

        # 키포인트 추출: 사람이 잡혔을 때만
        # keypoints.xy 모양 = (검출된 사람 수, 17, 2)
        kpts = results[0].keypoints
        detected = kpts is not None and len(kpts.xy) > 0
        if detected:
            person0 = kpts.xy[0].cpu().numpy()   # 첫 번째 사람만 (17,2)
            all_kpts.append(person0)

        frame_count += 1

        # headless(디스플레이 없음)에서는 화면에 띄우지 않지만,
        # --save-frames가 켜져 있으면 저장을 위해 annotated는 그대로 필요하다.
        need_annotated = (not args.headless) or (args.save_frames > 0)
        if need_annotated:
            # results[0].plot() = 스켈레톤·박스가 그려진 이미지를 돌려줌
            # → 검출됐는지 눈으로 확인하는 용도
            annotated = results[0].plot()
            if detected:
                # 검출 성공 표시 (좌상단 초록 글씨)
                cv2.putText(annotated, "DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # --save-frames N: N프레임마다 스켈레톤 그려진 프레임을 jpg로 저장
        if args.save_frames > 0 and frame_count % args.save_frames == 0:
            frame_path = os.path.join(frames_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, annotated)

        if not args.headless:
            cv2.imshow("camera_test", annotated)

            # q 키로 종료
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
except KeyboardInterrupt:
    # headless에서는 'q' 키를 누를 창이 없어 Ctrl+C가 사실상 유일한 종료 수단이다.
    # 여기서 잡아주지 않으면 아래 fps 출력/npz 저장이 통째로 스킵된다.
    print("\n중단됨 (Ctrl+C) — 지금까지 수집한 데이터를 저장합니다.")
finally:
    # 중간에 예외(Ctrl+C 포함)가 나도 카메라/창은 반드시 해제한다.
    if args.camera == "usb":
        cap.release()
    else:
        picam2.stop()
        picam2.close()   # close()까지 해야 다음 실행에서 카메라가 바로 다시 잡힌다
    if not args.headless:
        cv2.destroyAllWindows()

# ── 6. 종료 처리: fps 출력 + npz 저장 ───────────────────
elapsed = time.time() - t_start
fps = frame_count / elapsed if elapsed > 0 else 0
print(f"프레임 {frame_count}개 / {elapsed:.1f}초 → 평균 {fps:.1f} fps")
print(f"키포인트 잡힌 프레임: {len(all_kpts)}개")

# 자세 3종 저장할 때는 --save 이름을 바꿔가며 3회 실행:
#   python camera_test.py --save standing.npz
#   python camera_test.py --save lying.npz
if all_kpts:
    np.savez(args.save, keypoints=np.array(all_kpts))
    print(f"저장 완료: {args.save}")
else:
    print("저장할 키포인트 없음 (검출 0회)")
