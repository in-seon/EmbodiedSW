"""
스마트 횡단보도 PoC v0.1 — 보드 독립 스켈레톤
================================================
실행: pip install ultralytics opencv-python lap   # lap = ByteTrack ID 추적용
      python crosswalk_poc.py --source csi        # Pi CSI 카메라 (picamera2)
      python crosswalk_poc.py --source usb        # UVC 웹캠
      python crosswalk_poc.py --source 파일.mp4   # 영상 파일

SSH 등 모니터 없는 환경에서는 DISPLAY가 없는 걸 감지해 자동으로 창 없이 돈다
(--no-display를 깜빡해도 안전하다). 창이 없으면 q/r 키 대신 Ctrl+C로 종료한다.

Pi 5 + CSI 준비물 (레거시 카메라 스택이 없어 cv2.VideoCapture로 못 연다):
      sudo apt install -y python3-picamera2 build-essential python3-dev
      python -m venv --system-site-packages venv   # picamera2는 apt 패키지라
                                                   # 일반 venv에서는 안 보인다

설계 원칙:
  1. 보드 독립 — CONFIG만 바꾸면 노트북 → Jetson/D3-G/라파이 포팅.
     로직(신호 연장, 쓰러짐 판단)은 검출기 출력에만 의존하고
     어떤 모델/하드웨어인지 모른다.
  2. 저 fps 전제 — 2~3fps에서도 성립하는 시간 기반 로직.
     (프레임 수 카운트 금지, 항상 '초' 단위로 판단)
  3. 실측 우선 — 매 세션 fps/검출수를 CSV로 남김.
     이 숫자가 그대로 개발완료보고서의 정량 목표 근거가 된다.
"""

import argparse
import csv
import os
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# CONFIG — 보드/환경 바뀌면 여기만 수정
# ============================================================
CONFIG = {
    # --- 모델 계층 (보드 확정 시 여기만 교체) ---
    # 노트북 PoC: yolov8n / NPU 보드 확보 시: yolov8s 등 상향
    # D3-G(CPU only) 포팅 시: NCNN/INT8 export 후 경로 교체
    # 포즈 모델 하나가 사람 박스(res.boxes)+키포인트를 모두 주므로 별도 검출 모델은 쓰지 않는다.
    "pose_model": "yolov8n-pose.pt",  # 사람 박스 + 쓰러짐 자세
    "imgsz": 640,                      # CPU 보드에서 320~416으로 하향 가능
    "conf_thres": 0.4,
    # 쓰러짐은 '사람별'로 누적해야 하므로 ID 추적이 필수다.
    # bytetrack = 외형(ReID) 없이 IoU+칼만만 쓰는 경량 트래커 → 저사양 보드 친화적.
    "tracker": "bytetrack.yaml",

    # --- 신호등 타이밍 (모형 데모 기준, 초) ---
    "green_base_sec": 10.0,       # 기본 보행 신호
    "green_ext_sec": 5.0,         # 1회 연장량
    "green_max_ext": 3,           # 최대 연장 횟수 (무한 연장 방지)
    "red_sec": 8.0,

    # --- 횡단보도 구역 (프레임 내 ROI, 비율 좌표 x1,y1,x2,y2) ---
    # 데모 셋업 후 실측으로 조정. 이 박스 안 = "횡단보도 위"
    "crosswalk_roi": (0.15, 0.30, 0.85, 0.95),

    # --- 쓰러짐 판정 ---
    "fall_angle_deg": 50.0,       # 몸통 축이 수직에서 이만큼 기울면 후보
    "fall_confirm_sec": 3.0,      # 이 시간 유지돼야 사이렌 확정 (3초 안에 일어나면 오탐으로 무시)
    "fall_gap_sec": 1.0,          # 확정 전, 이 시간 이내의 짧은 미검출은 무시하고 카운트 이어감 (저 fps 깜빡임 대응)
    "fall_gap_frames": 2,         # 위 갭의 하한을 '실측 프레임 간격 x N'으로도 잡는다.
                                  # 고정 1초만 쓰면 프레임 간격이 1초를 넘는 보드에서
                                  # 한 프레임 깜빡임에 후보가 취소돼 쓰러짐을 영영 확정 못 한다.
    "fall_gap_max_sec": 4.0,      # 적응 상한 (fps가 병적으로 낮을 때 폭주 방지)
    "fall_clear_sec": 3.0,        # 확정 후, '정상 자세 or ROI 이탈'이 이 시간 연속돼야 사이렌 해제 (깜빡임 방지)
    "fall_aspect_ratio": 1.3,     # bbox 가로/세로 비율 보조 판정
    "fall_roi_overlap": 0.3,      # 쓰러짐 판정용 ROI 겹침 비율 (발 한 점이 아니라 몸 전체 기준)

    # --- 트랙 ID 유지 ---
    # 넘어지는 순간 bbox 모양이 급변해 트래커가 ID를 새로 달면 누적이 리셋된다.
    # 직전에 사라진 트랙과 충분히 겹치는 새 트랙은 그 상태를 물려받는다.
    "track_grace_sec": 1.5,
    "track_grace_frames": 2,      # fall_gap_frames와 같은 이유의 적응 하한.
                                  # 고정 1.5초만 쓰면 프레임 간격이 1.5초를 넘는 순간
                                  # '직전 프레임'조차 유예 밖으로 밀려나, ID가 빠진
                                  # 검출이 매번 새 사람이 되고 누적이 영영 안 쌓인다.
                                  # 유예의 단위는 '초'가 아니라 '프레임 수'여야 한다.
    "track_grace_max_sec": 4.0,   # 적응 상한 (오래전 사람의 상태를 물려받는 것 방지)
    "track_inherit_overlap": 0.3,   # 교집합/작은쪽넓이 기준 (IoU 아님 — bbox_overlap 주석 참고)

    # --- 영상 소스 ---
    # "csi" = Pi CSI 카메라(picamera2) / "usb" = UVC 웹캠 / 그 외 = 영상 파일 경로
    "source": "csi",
    "camera_index": 0,            # usb 소스일 때 /dev/videoN
    "frame_size": (640, 480),
    # picamera2의 "RGB888"은 이름과 달리 numpy 배열에 B,G,R 순으로 담긴다
    # (libcamera 명명 규칙). 그래서 보통은 변환이 필요 없다. 혹시 화면에서
    # 빨강/파랑이 뒤집혀 보이면 이 값을 True로 바꾸면 된다.
    "csi_swap_rb": False,

    # --- 실행/복구 ---
    # 웹캠은 프레임을 종종 흘린다. 한 번 실패했다고 세션을 끝내면 시연 도중
    # 조용히 죽는다. 연속 실패가 이 횟수를 넘을 때만 진짜 끊긴 것으로 본다.
    "max_read_fail": 30,
    "read_retry_sec": 0.05,
    # 카메라는 30fps로 계속 찍는데 추론은 그보다 느려서 큐에 옛 프레임이 쌓인다
    # (실측 3~4장 ≈ 100~133ms 지연). grab()이 이 시간보다 오래 걸리면
    # '갓 찍힌 프레임'으로 보고 거기서 멈춘다.
    "drain_fresh_sec": 0.005,
    "drain_max_skip": 8,

    # --- 로깅 ---
    "log_csv": "poc_metrics.csv",
}


# ============================================================
# 1. 입력 계층 — 카메라/영상 소스. 보드가 바뀌면 여기가 갈린다
# ============================================================
class FrameSource:
    """영상 소스 추상화.

    설계 원칙 1번(보드 독립)이 검출 계층에만 적용돼 있고 영상 소스는 main()에
    cv2.VideoCapture로 하드코딩돼 있었다. 그런데 Pi 5는 레거시 카메라 스택이
    제거돼 CSI 센서를 cv2.VideoCapture로 못 연다 (센서가 raw Bayer로 노출됨).
    pip의 opencv-python 휠은 GStreamer 없이 빌드돼 있어 libcamerasrc 우회도
    막힌다. 그래서 소스를 갈아끼울 수 있게 분리한다.

    계약:
      read() -> (ok, frame)   frame은 BGR numpy 배열
      is_live                 True면 read 실패가 '순간 글리치', False면 '파일 끝'
      nominal_fps             영상 파일의 FPS (라이브 카메라면 0.0)
    """

    is_live = True
    nominal_fps = 0.0

    def read(self):
        raise NotImplementedError

    def release(self):
        pass


class OpenCVSource(FrameSource):
    """UVC(USB) 웹캠과 영상 파일. 노트북 개발용 경로."""

    def __init__(self, spec, cfg, camera_index=None):
        self.cfg = cfg
        if spec == "usb":
            idx = cfg["camera_index"] if camera_index is None else camera_index
            self.cap = cv2.VideoCapture(idx)
            self.is_live = True
            w, h = cfg["frame_size"]
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        else:
            self.cap = cv2.VideoCapture(str(spec))
            self.is_live = False
            self.nominal_fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        if not self.cap.isOpened():
            raise SystemExit(f"영상 소스 열기 실패: {spec}")

    def _drain(self) -> bool:
        """큐에 쌓인 옛 프레임을 버리고 최신 것만 남긴다.

        cap.set(CAP_PROP_BUFFERSIZE, 1)은 쓰지 않는다 — V4L2 백엔드가 True를
        돌려주고 값도 1로 읽히지만 동작은 그대로다(실측 확인). 큐를 직접 비우는
        것만 실제로 듣는다."""
        for _ in range(self.cfg["drain_max_skip"]):
            t0 = time.time()
            if not self.cap.grab():
                return False
            if time.time() - t0 > self.cfg["drain_fresh_sec"]:
                break                      # 기다렸다 = 갓 찍힌 프레임
        return True

    def read(self):
        if not self.is_live:
            return self.cap.read()
        if not self._drain():
            return False, None
        return self.cap.retrieve()

    def release(self):
        self.cap.release()


class PiCameraSource(FrameSource):
    """Raspberry Pi CSI 카메라 (picamera2 / libcamera).

    준비물:
      sudo apt install -y python3-picamera2
      python -m venv --system-site-packages venv   # picamera2는 apt 패키지라
                                                   # 일반 venv에서는 안 보인다
    """

    is_live = True
    nominal_fps = 0.0

    def __init__(self, cfg):
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise SystemExit(
                "picamera2를 찾을 수 없습니다. CSI 카메라에는 필수입니다.\n"
                "  sudo apt install -y python3-picamera2\n"
                "  venv 사용 중이면: python -m venv --system-site-packages venv\n"
                "  (USB 웹캠으로 먼저 보려면 --source usb)"
            ) from e
        self.cam = Picamera2()
        w, h = cfg["frame_size"]
        # format="RGB888"이 numpy에서 BGR로 담기는 게 libcamera 명명 규칙이라
        # OpenCV/YOLO가 기대하는 순서와 그대로 맞는다. 색이 뒤집혀 보이면
        # CONFIG["csi_swap_rb"]로 뒤집을 수 있다 (첫 실행 때 화면으로 확인할 것).
        self.cam.configure(self.cam.create_video_configuration(
            main={"size": (w, h), "format": "RGB888"}))
        self.cam.start()
        self.swap_rb = cfg["csi_swap_rb"]

    def read(self):
        # capture_array()는 카메라가 만든 최신 프레임을 준다. OpenCVSource와 달리
        # 큐를 비울 필요가 없다.
        frame = self.cam.capture_array()
        if frame is None:
            return False, None
        if self.swap_rb:
            frame = frame[:, :, ::-1].copy()
        return True, frame

    def release(self):
        self.cam.stop()
        self.cam.close()


def open_source(spec, cfg, camera_index=None) -> FrameSource:
    """spec: "csi" | "usb" | 영상 파일 경로."""
    if spec == "csi":
        return PiCameraSource(cfg)
    return OpenCVSource(spec, cfg, camera_index)


# ============================================================
# 2. 검출 계층 — 보드/모델 교체 시 이 클래스만 영향
# ============================================================
@dataclass
class Person:
    """검출기 출력의 표준 형식. 로직 계층은 이것만 안다."""
    bbox: tuple          # (x1, y1, x2, y2) 픽셀
    conf: float
    keypoints: np.ndarray | None = None  # (17, 3) COCO 포즈 or None
    track_id: int | None = None          # 사람별 쓰러짐 누적의 키. None이면 쓰러짐 판정에서 제외


class Detector:
    """사람 검출 + 포즈 + ID 추적. 포팅 시 내부 구현만 갈아끼움 (인터페이스 불변)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.pose = YOLO(cfg["pose_model"])

    def detect(self, frame) -> list[Person]:
        # 저사양 보드에서는 det/pose 중 하나만 돌리고 교대 추론하는
        # 전략도 가능 (예: pose는 2프레임에 1번) — 그때 이 메서드만 수정
        # persist=True: 프레임 간 트래커 상태를 유지해 같은 사람에게 같은 ID를 준다.
        res = self.pose.track(frame, imgsz=self.cfg["imgsz"],
                              conf=self.cfg["conf_thres"], persist=True,
                              tracker=self.cfg["tracker"], verbose=False)[0]
        people = []
        if res.boxes is None:
            return people
        kpts = res.keypoints.data.cpu().numpy() if res.keypoints is not None else None
        # 낙상 순간에는 ID가 통째로 빠지는 프레임이 흔하다 (bbox 급변 + conf 하락).
        # 여기서 걸러내지 않고 None 그대로 넘기면 FallTracker가 위치로 이어붙인다.
        ids = res.boxes.id.int().cpu().tolist() if res.boxes.id is not None else None
        for i, box in enumerate(res.boxes):
            xyxy = tuple(int(v) for v in box.xyxy[0].cpu().numpy())
            kp = kpts[i] if kpts is not None and i < len(kpts) else None
            tid = ids[i] if ids is not None and i < len(ids) else None
            people.append(Person(bbox=xyxy, conf=float(box.conf[0]),
                                 keypoints=kp, track_id=tid))
        return people


# ============================================================
# 3. 로직 계층 — 보드와 100% 무관. 여기가 우리 시스템의 '두뇌'
# ============================================================
def foot_in_roi(person: Person, roi_px) -> bool:
    """발 위치(bbox 하단 중앙)가 횡단보도 ROI 안인가 — **신호 연장 판단용**.
    서 있고 걷는 사람의 '어디에 있나'는 발이 가장 정확하다."""
    x1, y1, x2, y2 = person.bbox
    foot_x, foot_y = (x1 + x2) / 2, y2
    rx1, ry1, rx2, ry2 = roi_px
    return rx1 <= foot_x <= rx2 and ry1 <= foot_y <= ry2


def roi_overlap_ratio(person: Person, roi_px) -> float:
    """bbox가 ROI와 겹치는 면적 비율 — **쓰러짐 판단용**.

    사람이 넘어지면 bbox 하단(=발 위치)이 자세에 따라 크게 튄다. 즉 발 한 점
    기준은 하필 쓰러짐을 판정해야 하는 순간에 가장 불안정해서, 실제로 쓰러진
    사람이 ROI 밖으로 튕겨나가 사이렌이 안 울릴 수 있다. 그래서 쓰러짐 쪽은
    한 점이 아니라 몸 전체의 겹침으로 본다 (자세가 변해도 면적은 안 튄다)."""
    x1, y1, x2, y2 = person.bbox
    rx1, ry1, rx2, ry2 = roi_px
    iw = max(0, min(x2, rx2) - max(x1, rx1))
    ih = max(0, min(y2, ry2) - max(y1, ry1))
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return (iw * ih) / area if area > 0 else 0.0


def bbox_overlap(a, b) -> float:
    """트랙 ID 교체 보정용 겹침도 — 교집합 / **작은 쪽 넓이**.

    IoU를 쓰면 안 된다. 사람이 넘어지면 bbox 넓이 자체가 크게 변해서(세로로 길다가
    가로로 납작해짐) 같은 사람인데도 IoU가 뚝 떨어진다. UR Fall 실측에서 낙상
    직전/직후 프레임의 IoU가 0.18까지 내려가 임계값 0.3에 걸려 다른 사람으로
    끊겼다. 같은 두 프레임의 이 지표는 0.36으로, 크기 변화에 훨씬 덜 민감하다."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    smaller = min(max(0, ax2 - ax1) * max(0, ay2 - ay1),
                  max(0, bx2 - bx1) * max(0, by2 - by1))
    return (iw * ih) / smaller if smaller > 0 else 0.0


def torso_angle_deg(kp: np.ndarray) -> float | None:
    """어깨 중점→엉덩이 중점 벡터가 수직에서 기운 각도.
    COCO: 5,6=어깨 / 11,12=엉덩이. conf 낮으면 None."""
    if kp is None:
        return None
    sh = kp[[5, 6]]
    hip = kp[[11, 12]]
    if sh[:, 2].min() < 0.3 or hip[:, 2].min() < 0.3:
        return None
    v = hip[:, :2].mean(axis=0) - sh[:, :2].mean(axis=0)
    if np.linalg.norm(v) < 1e-6:
        return None
    # 수직(0,1)과의 각도
    cos = abs(v[1]) / np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(cos, 0, 1))))


def looks_fallen(person: Person, cfg) -> bool:
    """단일 프레임 쓰러짐 *후보* 판정 (확정은 시간 누적으로)."""
    ang = torso_angle_deg(person.keypoints)
    if ang is not None:
        return ang > cfg["fall_angle_deg"]
    # 포즈 신뢰도 낮으면 bbox 비율로 폴백 (누운 사람은 가로로 길다)
    x1, y1, x2, y2 = person.bbox
    w, h = x2 - x1, y2 - y1
    return h > 0 and (w / h) > cfg["fall_aspect_ratio"]


class FallMonitor:
    """**한 사람**의 쓰러짐 확정/해제를 '비대칭 시간 히스테리시스'로 판단 (저 fps 친화).

    여러 명을 하나로 뭉쳐 판단하면 서로 다른 사람의 짧은 자세 이상이 합산돼
    오발동하므로, 이 객체는 track_id 하나만 담당한다 (관리는 FallTracker).

    - 발동: 쓰러짐 후보가 fall_confirm_sec 유지되면 확정(사이렌). 단, 저 fps에서
            검출이 순간적으로 깜빡이면 카운트가 리셋되는 걸 막기 위해, gap_sec
            이내의 짧은 미검출은 무시하고 카운트를 이어간다. 그보다 긴 공백(=실제로
            일어남)이면 후보를 취소해 3초 안에 일어난 사람은 사이렌이 울리지 않는다.
            (확정은 '실제로 쓰러짐이 보인' 프레임에서만 일어난다 — 공백 중엔 확정 안 함.
             이 성질 덕분에 gap_sec를 키워도 '3초 안에 일어남' 보장은 깨지지 않는다.)
    - 해제: 확정 이후 '쓰러짐 아님'(정상 자세 or 횡단보도 이탈)이 fall_clear_sec
            연속 유지돼야 해제. 덕분에 검출이 깜빡여도 사이렌이 꺼지지 않고,
            실제로 일어나거나 횡단보도 밖으로 나가야 꺼진다.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.candidate_since: float | None = None  # 쓰러짐 후보 run 시작 시각
        self.last_fallen: float | None = None      # 마지막으로 '쓰러짐'이 관측된 시각 (갭 판정용)
        self.clear_since: float | None = None      # 정상 복귀 시작 시각 (확정 이후에만 의미)
        self.confirmed = False

    def update(self, any_fallen: bool, now: float, gap_sec: float) -> bool:
        """gap_sec는 FallTracker가 실측 프레임 간격에 맞춰 넣어준다 (고정 상수 아님)."""
        if any_fallen:
            self.last_fallen = now
            self.clear_since = None
            if self.candidate_since is None:
                self.candidate_since = now
            if now - self.candidate_since >= self.cfg["fall_confirm_sec"]:
                self.confirmed = True
        elif self.confirmed:
            # 이미 사이렌 중 — 해제는 fall_clear_sec 연속 '아님' 이후에만 (깜빡임 무시)
            self.candidate_since = None
            if self.clear_since is None:
                self.clear_since = now
            if now - self.clear_since >= self.cfg["fall_clear_sec"]:
                self.confirmed = False
                self.clear_since = None
        else:
            # 미확정 상태의 짧은 미검출: gap_sec 이내면 후보 카운트를 유지(디바운스),
            # 그보다 길면 실제로 일어난 것으로 보고 후보 취소.
            self.clear_since = None
            if (self.candidate_since is not None and self.last_fallen is not None
                    and now - self.last_fallen > gap_sec):
                self.candidate_since = None
        return self.confirmed


class FallTracker:
    """사람별 FallMonitor 관리 계층. FallMonitor의 상태 기계 자체는 건드리지 않는다.

    세 가지만 책임진다:
      1. 사람별 분리 — track_id마다 독립 FallMonitor. 프레임 전체를 OR로 뭉치면
         A가 1.5초, 뒤이어 B가 1.5초 이상자세만 취해도 합산돼 3초 사이렌이
         울렸다. 이제 누적은 같은 사람 안에서만 일어난다.
      2. 프레임 간격 적응 — 갭이 고정 1초면 프레임 간격이 1초를 넘는 보드에서
         한 프레임만 깜빡여도 후보가 취소돼 쓰러짐을 영영 확정하지 못한다.
         실측 간격 x fall_gap_frames 로 갭의 하한을 올린다. 확정은 여전히
         '쓰러짐이 관측된 프레임'에서만 일어나므로 갭을 키워도 오탐은 안 는다.
         3번의 유예 창(grace)도 같은 이유로 같이 적응시킨다 — 둘 중 하나만
         고정값으로 남으면 저 fps에서 그쪽이 먼저 무너진다 (grace_sec 주석 참고).
      3. ID 공백/교체 보정 — 넘어지는 순간이 하필 트래커가 가장 약한 구간이다.
         UR Fall 실측(fall-01)에서 낙상 프레임의 ID가 None → 2 → None 으로
         끊겼다 (bbox 모양 급변 + conf 하락). 트래커 ID를 그대로 믿으면
         낙상 구간만 쏙 빠져 영영 확정되지 않는다. 그래서 직전 프레임 위치와의
         겹침으로 같은 사람에게 이어붙이는 보정층을 한 겹 둔다.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.monitors: dict[int, FallMonitor] = {}
        self.last_seen: dict[int, float] = {}
        self.last_bbox: dict[int, tuple] = {}
        self.intervals = deque(maxlen=15)   # 실측 프레임 간격 (중앙값으로 갭 산출)
        self.prev_now: float | None = None
        self._synth = 0                     # ID 없는 검출용 임시 키 (음수, 트래커 ID와 안 겹침)

    def _tick(self, now: float) -> float | None:
        """실측 프레임 간격 이력을 갱신하고 중앙값을 돌려준다 (이력 없으면 None).
        튀는 값에 안 흔들리게 평균이 아니라 중앙값을 쓴다.
        update() 당 정확히 한 번만 호출할 것 — 두 번 부르면 간격 0이 섞여 중앙값이 무너진다."""
        if self.prev_now is not None and now > self.prev_now:
            self.intervals.append(now - self.prev_now)
        self.prev_now = now
        if not self.intervals:
            return None
        return sorted(self.intervals)[len(self.intervals) // 2]

    def gap_sec(self, med: float | None) -> float:
        """이번 프레임 기준의 미검출 허용 갭 (쓰러짐 후보 카운트를 이어갈 시간)."""
        if med is None:
            return self.cfg["fall_gap_sec"]
        return min(max(self.cfg["fall_gap_sec"], med * self.cfg["fall_gap_frames"]),
                   self.cfg["fall_gap_max_sec"])

    def grace_sec(self, med: float | None) -> float:
        """ID 유실 보정에서 '직전 트랙'으로 인정해줄 시간 창.

        gap_sec와 정확히 같은 이유로 프레임 간격에 맞춰 늘린다. 고정 1.5초로 두면
        프레임 간격이 1.5초를 넘는 순간(≈0.67fps 미만) 바로 직전 프레임의 트랙조차
        유예 밖으로 밀려난다. 그러면 ID가 빠진 검출마다 새 synth 키가 발급돼
        쓰러짐 누적이 매 프레임 리셋되고, 사이렌이 영영 울리지 않는다."""
        if med is None:
            return self.cfg["track_grace_sec"]
        return min(max(self.cfg["track_grace_sec"], med * self.cfg["track_grace_frames"]),
                   self.cfg["track_grace_max_sec"])

    def _match(self, bbox, now, taken, grace: float) -> int | None:
        """직전 프레임 위치와 가장 많이 겹치는 기존 트랙의 키 (없으면 None).

        위치 이력이 없는 키는 후보에서 제외한다. 기본값으로 얼버무리면
        (예: 없을 때 bbox 자기 자신) 겹침이 1.0으로 나와 아무 상관 없는 사람이
        남의 사이렌 상태를 통째로 물려받는다 — 판단 불가는 '매칭 안 함'이어야 한다."""
        best, best_ov = None, self.cfg["track_inherit_overlap"]
        for key in self.monitors:
            if key in taken:                      # 이번 프레임에 이미 배정된 트랙은 후보 아님
                continue
            if key not in self.last_bbox or key not in self.last_seen:
                continue
            if now - self.last_seen[key] > grace:
                continue
            ov = bbox_overlap(self.last_bbox[key], bbox)
            if ov >= best_ov:
                best, best_ov = key, ov
        return best

    def _rekey(self, old: int, new: int) -> int:
        """트래커가 새 ID를 준 경우 기존 누적 상태를 그 ID로 옮긴다."""
        if new in self.monitors:   # 드문 ID 재사용 충돌 — 남의 상태를 덮어쓰느니 옛 키를 유지
            return old
        self.monitors[new] = self.monitors.pop(old)
        if old in self.last_seen:
            self.last_seen[new] = self.last_seen.pop(old)
        if old in self.last_bbox:
            self.last_bbox[new] = self.last_bbox.pop(old)
        return new

    def _resolve_keys(self, people, now, grace: float) -> list[int]:
        """검출마다 '사람 단위의 안정적인 키'를 정한다.

        트래커 ID가 1순위. 다만 ID가 없거나(저 conf) 처음 보는 ID면 직전 위치와의
        겹침으로 기존 트랙에 이어붙인다. 이 보정이 없으면 낙상 순간의 ID 공백에서
        누적이 리셋돼 쓰러짐을 통째로 놓친다 (UR Fall 실측으로 확인된 실패 모드).
        """
        keys: list[int | None] = [None] * len(people)
        taken: set[int] = set()

        # 1순위: 이미 관리 중인 트랙 ID는 그대로 사용 (빠른 경로)
        for i, p in enumerate(people):
            if p.track_id is not None and p.track_id in self.monitors:
                keys[i] = p.track_id
                taken.add(p.track_id)

        # 2순위: ID 공백이거나 처음 보는 ID → 위치로 이어붙이기
        for i, p in enumerate(people):
            if keys[i] is not None:
                continue
            src = self._match(p.bbox, now, taken, grace)
            if src is None:                       # 이어붙일 곳 없음 = 새로운 사람
                if p.track_id is not None:
                    keys[i] = p.track_id
                else:
                    self._synth -= 1
                    keys[i] = self._synth
            elif p.track_id is None:
                keys[i] = src                     # ID 공백 구간 — 기존 키를 그대로 유지
            else:
                keys[i] = self._rekey(src, p.track_id)   # 트래커가 새 ID를 줌 → 상태 이전
            taken.add(keys[i])
        return keys

    def _retire(self, now: float, gap: float, grace: float):
        """확정(사이렌) 상태는 해제될 때까지 유지. 그 외에는 갭/해제 판정이
        모두 끝날 만큼 오래 안 보이면 정리한다.
        관측 이력이 없는 키는 어차피 매칭 후보도 아니므로 바로 정리한다.

        ttl에 grace가 그대로 들어가야 한다 (gap과 함께 적응값을 써야 하는 이유):
        ttl = fall_clear_sec + gap + grace 이고 앞의 두 항이 양수이므로,
        '아직 물려받을 수 있는(now - last_seen <= grace)' 트랙은 절대 먼저
        정리되지 않는다. 한쪽만 고정값으로 두면 이 불변식이 깨져서, _match가
        찾으려는 트랙을 _retire가 미리 지워버리는 경합이 생긴다."""
        ttl = self.cfg["fall_clear_sec"] + gap + grace
        stale = [t for t, m in self.monitors.items()
                 if not m.confirmed and (t not in self.last_seen
                                         or now - self.last_seen[t] > ttl)]
        for tid in stale:
            del self.monitors[tid]
            self.last_seen.pop(tid, None)
            self.last_bbox.pop(tid, None)

    def update(self, people, fallen_flags, on_cw_flags, now: float) -> set[int]:
        """사이렌이 확정된 키 집합을 돌려준다 (비어 있으면 정상).

        부작용: 각 Person.track_id를 보정된 키로 덮어쓴다. 시각화·로그가
        낙상 전후로 같은 사람을 같은 번호로 보게 하기 위함이다."""
        med = self._tick(now)          # 프레임 간격 이력 갱신은 여기 한 번뿐
        gap, grace = self.gap_sec(med), self.grace_sec(med)
        keys = self._resolve_keys(people, now, grace)

        seen: dict[int, bool] = {}
        for p, key, fallen, on_cw in zip(people, keys, fallen_flags, on_cw_flags):
            p.track_id = key
            # 한 키에 두 검출이 붙는 일은 taken으로 막지만, 방어적으로 OR 병합
            seen[key] = seen.get(key, False) or (fallen and on_cw)
            self.last_bbox[key] = p.bbox

        for key, fallen in seen.items():
            self.monitors.setdefault(key, FallMonitor(self.cfg)).update(fallen, now, gap)
            self.last_seen[key] = now

        # 이번 프레임에 안 보인 트랙도 시간을 흘려보내야 한다. 여기서 미검출 갭
        # 판정과 (확정된 경우) 사이렌 해제 판정이 진행된다. 빼먹으면 사람이
        # 사라져도 사이렌이 영원히 안 꺼진다.
        for key in [k for k in self.monitors if k not in seen]:
            self.monitors[key].update(False, now, gap)

        self._retire(now, gap, grace)
        return {k for k, m in self.monitors.items() if m.confirmed}

    def reset(self):
        """모든 쓰러짐 누적을 버린다 (수동 리셋용).
        프레임 간격 추정치는 보드 성능 측정이므로 유지한다.
        주의: 진짜로 쓰러져 있는 사람이 계속 보이면 다시 3초 뒤 확정된다."""
        self.monitors.clear()
        self.last_seen.clear()
        self.last_bbox.clear()


class SignalState(Enum):
    RED = "RED"
    GREEN = "GREEN"
    GREEN_EXT = "GREEN_EXT"   # 연장된 초록불
    EMERGENCY = "EMERGENCY"   # 쓰러짐 확정 → 사이렌/신고 모의


class SignalController:
    """타이머 베이스 + 비전 개입. 보드의 GPIO/LED 제어로 1:1 치환 예정.
    저 fps 전제: 모든 판단이 '초' 기반, 프레임 기반 아님."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = SignalState.RED
        self.t_state: float | None = None  # 첫 update의 now로 초기화 (웹캠/영상 시간축 공용)
        self.ext_count = 0

    def _elapsed(self, now: float) -> float:
        if self.t_state is None:
            self.t_state = now
        return now - self.t_state

    def _to(self, s: SignalState, now: float):
        self.state = s
        self.t_state = now

    def force_red(self, now: float):
        """수동 리셋 — 오탐으로 EMERGENCY에 갇혔을 때 빠져나오는 유일한 출구."""
        self.ext_count = 0
        self._to(SignalState.RED, now)

    def update(self, n_in_crosswalk: int, fall_confirmed: bool, now: float) -> SignalState:
        c = self.cfg
        if fall_confirmed:
            if self.state != SignalState.EMERGENCY:
                self._to(SignalState.EMERGENCY, now)
                print(f"[{time.strftime('%H:%M:%S')}] !! 쓰러짐 확정 → 사이렌 + 자동신고 (모의)")
            return self.state

        if self.state == SignalState.EMERGENCY:
            # 쓰러짐 해제(일어나서 유지 or 횡단보도 밖 이탈) → 사이렌 종료, 적색 복귀
            print(f"[{time.strftime('%H:%M:%S')}] -- 쓰러짐 해제 → 사이렌 종료, 적색 복귀")
            self._to(SignalState.RED, now)

        elif self.state == SignalState.RED:
            if self._elapsed(now) >= c["red_sec"]:
                self.ext_count = 0
                self._to(SignalState.GREEN, now)

        elif self.state in (SignalState.GREEN, SignalState.GREEN_EXT):
            limit = c["green_base_sec"] if self.state == SignalState.GREEN else c["green_ext_sec"]
            if self._elapsed(now) >= limit:
                # === 핵심 개입 지점: 아직 횡단보도에 사람이 있나? ===
                if n_in_crosswalk > 0 and self.ext_count < c["green_max_ext"]:
                    self.ext_count += 1
                    self._to(SignalState.GREEN_EXT, now)
                    print(f"[{time.strftime('%H:%M:%S')}] >> 보행자 {n_in_crosswalk}명 잔류 → 초록불 연장 {self.ext_count}회차")
                else:
                    self._to(SignalState.RED, now)
        return self.state


# ============================================================
# 4. 계측 — 모든 실행이 보고서 데이터가 된다
# ============================================================
class Metrics:
    """매 프레임을 CSV로 남긴다. 이 파일이 개발완료보고서의 정량 근거가 된다.

    ts는 벽시계, t_logic은 로직이 실제로 쓴 시각이다 (--video면 '영상 속 시각').
    둘이 다르므로 frame_idx와 t_logic이 있어야 로그 행을 영상 프레임에 붙일 수
    있고, 그래야 raw_data/labels의 정답 라벨과 대조해 '낙상 검출률'을 낼 수 있다.
    """

    COLUMNS = ["ts", "t_logic", "frame_idx", "fps", "n_people", "n_in_roi",
               "n_fallen", "n_confirmed", "state", "infer_ms"]

    def __init__(self, path):
        self.path = Path(path)
        # 기존 파일이 다른 스키마면 = 이전 로직으로 기록된 것. 같은 파일에 이어
        # 붙이면 보고서에서 서로 다른 로직의 숫자가 한 표로 보이므로 밀어낸다.
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, newline="") as f:
                head = next(csv.reader(f), [])
            if head != self.COLUMNS:
                backup = self.path.with_name(f"{self.path.stem}_"
                                             f"{time.strftime('%Y%m%d_%H%M%S')}"
                                             f"{self.path.suffix}")
                self.path.rename(backup)
                print(f"이전 스키마 로그를 {backup.name} 으로 보관하고 새로 시작합니다.")
        new = not self.path.exists() or self.path.stat().st_size == 0
        self.f = open(self.path, "a", newline="")
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(self.COLUMNS)
        self.fps_win = deque(maxlen=30)

    def log(self, frame_idx, t_logic, n_people, n_roi, n_fallen,
            n_confirmed, state, infer_ms):
        # fps는 '보드가 실제로 몇 장 처리했나'이므로 t_logic이 아니라 벽시계로 잰다.
        self.fps_win.append(time.time())
        fps = 0.0
        if len(self.fps_win) >= 2:
            span = self.fps_win[-1] - self.fps_win[0]
            fps = (len(self.fps_win) - 1) / span if span > 0 else 0.0
        self.w.writerow([f"{time.time():.2f}", f"{t_logic:.3f}", frame_idx,
                         f"{fps:.2f}", n_people, n_roi, n_fallen, n_confirmed,
                         state.value, f"{infer_ms:.1f}"])
        self.f.flush()  # 비정상 종료(크래시/전원차단)에도 지금까지 행은 디스크에 남게
        return fps

    def close(self):
        self.f.close()


# ============================================================
# 메인 루프
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None,
                    help="csi | usb | 영상파일경로 (기본: CONFIG['source'])")
    ap.add_argument("--video", default=None, help="[구식] --source 영상경로와 동일")
    ap.add_argument("--camera", type=int, default=None,
                    help="usb 소스일 때 /dev/videoN 인덱스")
    ap.add_argument("--no-display", action="store_true",
                    help="창 없이 실행 (디스플레이 없는 보드). 이때는 q/r 키를 못 쓴다")
    args = ap.parse_args()

    # 창을 띄울 수 있는지는 '띄우기 전에' 판별해야 한다.
    # 디스플레이가 없을 때 cv2.imshow는 파이썬 예외를 내지 않는다 — Qt 플러그인이
    # 초기화에 실패하면서 abort()를 부르고, 프로세스가 SIGABRT로 통째로 죽는다.
    # 그래서 try/except cv2.error로는 못 막고(아래 폴백은 headless 빌드 전용이다),
    # 죽는 순간 finally의 카메라 해제·로그 정리까지 건너뛴다.
    # 실측: DISPLAY 없이 실행 시 2프레임 만에 exit code 134.
    show = not args.no_display
    if show and os.name != "nt" and not os.environ.get("DISPLAY"):
        print("DISPLAY가 없어 창 없이 진행합니다 (--no-display와 동일). 종료는 Ctrl+C.")
        show = False

    spec = args.source or args.video or CONFIG["source"]
    src = open_source(spec, CONFIG, args.camera)

    # 시간축 선택: 라이브 카메라는 벽시계(time.time()), 영상 파일은 '영상 속 시각'.
    # 영상을 실시간보다 빠르게/느리게 처리해도 fall_confirm_sec(3초) 같은 시간 판단이
    # '영상 속 3초'로 정확히 계산되게 하기 위함 (오프라인 평가 정합성).
    video_fps = src.nominal_fps
    use_video_clock = (not src.is_live) and video_fps > 0
    if not src.is_live and not use_video_clock:
        print("경고: 영상 FPS를 읽지 못해 벽시계 시간으로 대체합니다.")
    frame_idx = 0

    det = Detector(CONFIG)
    fall_tracker = FallTracker(CONFIG)
    signal = SignalController(CONFIG)
    metrics = Metrics(CONFIG["log_csv"])

    state_color = {
        SignalState.RED: (0, 0, 255), SignalState.GREEN: (0, 200, 0),
        SignalState.GREEN_EXT: (0, 255, 255), SignalState.EMERGENCY: (0, 100, 255),
    }

    read_fail = 0
    print(f"PoC 시작 [{spec}] — {'q 종료 / r 수동리셋' if show else '창 없음(Ctrl+C 종료)'}. "
          f"모든 세션이 {CONFIG['log_csv']}에 기록됨.")
    try:
        while True:
            ok, frame = src.read()
            if not ok:
                # 파일은 ok=False가 곧 끝. 카메라는 순간 글리치일 수 있으므로 버틴다.
                if not src.is_live:
                    break
                read_fail += 1
                if read_fail > CONFIG["max_read_fail"]:
                    print(f"카메라에서 {read_fail}회 연속 읽기 실패 — 종료합니다.")
                    break
                time.sleep(CONFIG["read_retry_sec"])
                continue
            read_fail = 0
            H, W = frame.shape[:2]
            r = CONFIG["crosswalk_roi"]
            roi_px = (int(r[0] * W), int(r[1] * H), int(r[2] * W), int(r[3] * H))

            # 이번 프레임의 '판단 기준 시각' (웹캠=실시간, 영상=영상 속 시간)
            now = frame_idx / video_fps if use_video_clock else time.time()
            frame_idx += 1

            t0 = time.time()
            people = det.detect(frame)
            infer_ms = (time.time() - t0) * 1000

            # 프레임당 사람마다 한 번만 계산해 재사용 (저 fps 보드 연산 절약)
            # ROI 기준이 둘로 나뉜다: 신호 연장은 '발 위치', 쓰러짐은 '몸 전체 겹침'.
            # 쓰러지면 발 위치가 크게 튀어서, 한 기준으로 둘 다 처리하면 하필
            # 쓰러진 순간에 ROI 밖으로 빠져 사이렌을 놓친다.
            roi_flags = [foot_in_roi(p, roi_px) for p in people]
            fallen_flags = [looks_fallen(p, CONFIG) for p in people]
            on_cw_flags = [roi_overlap_ratio(p, roi_px) >= CONFIG["fall_roi_overlap"]
                           for p in people]
            n_roi = sum(roi_flags)
            # 쓰러짐은 사람별로 누적된다 (프레임 전체 OR 아님 — 타인끼리 합산되던 오발동 제거)
            confirmed_ids = fall_tracker.update(people, fallen_flags, on_cw_flags, now)
            fall_confirmed = bool(confirmed_ids)
            state = signal.update(n_roi, fall_confirmed, now)
            fps = metrics.log(frame_idx - 1, now, len(people), n_roi,
                              sum(fallen_flags), len(confirmed_ids), state, infer_ms)

            # ---- 시각화 (데모 연출의 씨앗: 상태가 한눈에 보이게) ----
            # 창을 안 띄우면 그리는 연산 자체를 건너뛴다 (저사양 보드에서 유의미)
            if not show:
                continue
            cv2.rectangle(frame, roi_px[:2], roi_px[2:], (255, 200, 0), 2)
            for p, fallen in zip(people, fallen_flags):
                if p.track_id in confirmed_ids:
                    color, thick = (0, 0, 255), 3      # 사이렌 확정된 사람
                elif fallen:
                    color, thick = (0, 165, 255), 2    # 쓰러짐 후보 (아직 누적 중)
                else:
                    color, thick = (0, 255, 0), 2
                cv2.rectangle(frame, p.bbox[:2], p.bbox[2:], color, thick)
                if p.track_id is not None:
                    cv2.putText(frame, f"#{p.track_id}", (p.bbox[0], p.bbox[1] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(frame, (0, 0), (W, 44), (30, 30, 30), -1)
            cv2.putText(frame,
                        f"{state.value} | ROI:{n_roi} | {fps:.1f}fps | {infer_ms:.0f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        state_color[state], 2)
            try:
                cv2.imshow("crosswalk PoC", frame)
                key = cv2.waitKey(1) & 0xFF
            except cv2.error:
                # GUI 없이 빌드된 opencv(opencv-python-headless)는 여기서 얌전히
                # 예외를 낸다. 디스플레이가 없는 경우는 예외가 아니라 SIGABRT라서
                # 여기까지 못 오고, 그건 위쪽 DISPLAY 사전 검사로 막는다.
                print("opencv에 GUI 기능이 없음 — 창 없이 계속 진행합니다.")
                show, key = False, 255
            if key == ord("q"):
                break
            if key == ord("r"):
                # 오탐으로 EMERGENCY에 갇혔을 때의 탈출구
                fall_tracker.reset()
                signal.force_red(now)
                print(f"[{time.strftime('%H:%M:%S')}] ** 수동 리셋 — 쓰러짐 상태 초기화, 적색 복귀")
    finally:
        # 정상 종료·q·예외·Ctrl+C 어느 경우든 카메라/창/로그파일은 반드시 정리.
        # 앞 단계가 실패해도 뒤 단계는 돌아야 한다 — 예전엔 src.release()가 던지면
        # metrics.close()가 통째로 스킵됐다.
        for name, fn in (("카메라 해제", src.release),
                         ("창 정리", cv2.destroyAllWindows),
                         ("로그 저장", metrics.close)):
            try:
                fn()
            except Exception as e:                     # noqa: BLE001 — 정리 단계는 뭐가 나도 계속
                print(f"정리 중 무시된 오류 ({name}): {e}")
    print(f"세션 종료. 실측 데이터: {CONFIG['log_csv']}")


if __name__ == "__main__":
    main()
