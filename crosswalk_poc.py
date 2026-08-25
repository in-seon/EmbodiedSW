"""
스마트 횡단보도 PoC v0.1 — 보드 독립 스켈레톤

실행: pip install ultralytics opencv-python lap   # lap = ByteTrack ID 추적용
      python crosswalk_poc.py --source csi        # Pi CSI 카메라 (picamera2)
      python crosswalk_poc.py --source usb        # UVC 웹캠
      python crosswalk_poc.py --source 파일.mp4   # 영상 파일

Pi 5 + CSI 준비물:
      sudo apt install -y python3-picamera2
      python -m venv --system-site-packages venv

설계 원칙 (근거는 README 2절):
  1. 보드 독립 — CONFIG만 바꾸면 포팅. 로직은 검출기 출력에만 의존한다.
  2. 저 fps 전제 — 판단은 항상 '초' 단위. 프레임 수를 세지 않는다.
  3. 실측 우선 — 매 세션 fps/검출수를 CSV로 남긴다.

구조: [1] 입력 → [2] 검출 → [3] 로직 → [4] 계측
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
# 튜닝 구역 — 손댈 값은 전부 여기 (설명: README 6절)
#   [A] 설치할 때마다  [B] 데모 감도  [C] 내부
# ============================================================
CONFIG = {
    # ---------- [A] 설치할 때마다 확인 ----------
    "rotate_deg": 0,              # 시계방향 회전 0/90/180/270. 바꾸면 crosswalk_roi 재설정
    "crosswalk_roi": (0.15, 0.30, 0.85, 0.95),   # 횡단보도 영역. 비율 (x1,y1,x2,y2), 0.0~1.0
    "csi_swap_rb": False,         # CSI 색 반전 보정. csi_color_check.py로 확정할 것
    "source": "csi",              # "csi" | "usb" | 영상 파일 경로
    "camera_index": 0,            # usb일 때 /dev/videoN 의 N
    "frame_size": (640, 480),     # 캡처 해상도 (회전 전 기준)

    # ---------- [B] 데모 감도 ----------
    "green_base_sec": 10.0,       # 기본 보행 신호 길이 (초)
    "green_ext_sec": 5.0,         # 1회 연장량
    "green_max_ext": 3,           # 최대 연장 횟수
    "red_sec": 8.0,               # 적색 길이

    "fall_angle_deg": 50.0,       # 몸통이 수직에서 이만큼 기울면 쓰러짐 '후보'
    "fall_confirm_sec": 3.0,      # 후보가 이만큼 유지돼야 사이렌 확정
    "fall_clear_sec": 3.0,        # 확정 후 이만큼 연속 정상이어야 해제
    "kp_conf_min": 0.3,           # 키포인트 신뢰도 하한. 미만이면 bbox 비율로 폴백
    "fall_aspect_ratio": 1.3,     # 폴백 기준: bbox 가로/세로가 이보다 크면 후보
    "fall_roi_overlap": 0.3,      # 몸 전체가 ROI와 이만큼 겹쳐야 '횡단보도 위'

    "pose_model": "yolov8n-pose.pt",   # 박스+키포인트를 한 모델로
    "imgsz": 640,                 # 입력 크기. 느리면 416 → 320
    "conf_thres": 0.4,            # 검출 신뢰도 하한

    # ---------- [C] 내부 동작 ----------
    "tracker": "bytetrack.yaml",  # ReID 없는 경량 트래커 (저사양 보드용)

    # 미검출 갭: 고정값과 '프레임 간격 x N' 중 큰 쪽 (저 fps 깜빡임 방어)
    "fall_gap_sec": 1.0,          # 갭 하한
    "fall_gap_frames": 2,         # 적응 하한 = 실측 프레임 간격 x N
    "fall_gap_max_sec": 4.0,      # 적응 상한

    # ID 유실/교체 시 상태를 물려줄 시간 창. gap과 같은 방식으로 적응한다.
    "track_grace_sec": 1.5,       # 유예 하한
    "track_grace_frames": 2,      # 적응 하한 = 실측 프레임 간격 x N
    "track_grace_max_sec": 4.0,   # 적응 상한
    "track_inherit_overlap": 0.3,  # 상속 판정 겹침도 (IoU 아님 — bbox_overlap 참고)

    "max_read_fail": 30,          # 연속 읽기 실패가 이만큼 넘어야 진짜 끊김으로 본다
    "read_retry_sec": 0.05,       # 읽기 실패 시 재시도 간격
    "drain_fresh_sec": 0.005,     # grab()이 이보다 오래 걸리면 '갓 찍힌 프레임'
    "drain_max_skip": 8,          # 큐 비우기 최대 프레임 수

    "log_csv": "poc_metrics.csv",  # 실측 로그. 실행할 때마다 이어 붙는다
}


# ============================================================
# 1. 입력 계층 — 카메라/영상 소스. 보드가 바뀌면 여기가 갈린다
# ============================================================
class FrameSource:
    """영상 소스 추상화. Pi 5는 CSI를 cv2.VideoCapture로 못 열어 소스를 분리했다.

    read() -> (ok, frame)        BGR 배열, 회전 보정까지 끝난 상태
    _read_raw() -> (ok, frame)   하위 클래스가 구현. 센서가 준 그대로
    is_live                      True면 read 실패가 '글리치', False면 '파일 끝'
    nominal_fps                  영상 파일의 FPS (라이브면 0.0)
    """

    is_live = True
    nominal_fps = 0.0
    rotate_code = None       # cv2.ROTATE_* 상수 or None(회전 없음)

    def read(self):
        """회전은 여기 한 곳에서만 적용한다 — 위층은 똑바로 선 프레임만 본다."""
        ok, frame = self._read_raw()
        if not ok or frame is None or self.rotate_code is None:
            return ok, frame
        return True, cv2.rotate(frame, self.rotate_code)

    def _read_raw(self):
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
        CAP_PROP_BUFFERSIZE가 V4L2에서 안 먹어(실측) 큐를 직접 비운다."""
        for _ in range(self.cfg["drain_max_skip"]):
            t0 = time.time()
            if not self.cap.grab():
                return False
            if time.time() - t0 > self.cfg["drain_fresh_sec"]:
                break                      # 기다렸다 = 갓 찍힌 프레임
        return True

    def _read_raw(self):
        if not self.is_live:
            return self.cap.read()
        if not self._drain():
            return False, None
        return self.cap.retrieve()

    def release(self):
        self.cap.release()


class PiCameraSource(FrameSource):
    """Raspberry Pi CSI 카메라 (picamera2 / libcamera). 준비물은 README 5.2절."""

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
        # "RGB888"은 libcamera 명명 규칙상 numpy에 BGR로 담긴다 (OpenCV와 동일).
        self.cam.configure(self.cam.create_video_configuration(
            main={"size": (w, h), "format": "RGB888"}))
        self.cam.start()
        self.swap_rb = cfg["csi_swap_rb"]

    def _read_raw(self):
        # capture_array()는 항상 최신 프레임 — OpenCVSource와 달리 큐 비우기 불필요.
        frame = self.cam.capture_array()
        if frame is None:
            return False, None
        if self.swap_rb:
            frame = frame[:, :, ::-1].copy()
        return True, frame

    def release(self):
        self.cam.stop()
        self.cam.close()


_ROTATE_CODES = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def rotate_code_for(deg) -> int | None:
    """회전 각도를 cv2 상수로. 90의 배수가 아니면 즉시 실패시킨다.

    임의 각도는 보간이 들어가 검출 성능이 떨어지고, 가장자리 검은 삼각형이 ROI
    비율 좌표의 의미까지 흐린다. 카메라는 브래킷으로 90도 단위로만 단다."""
    try:
        return _ROTATE_CODES[int(deg) % 360]
    except (KeyError, ValueError, TypeError):
        raise SystemExit(
            f"rotate_deg는 0/90/180/270만 됩니다 (받은 값: {deg!r}).\n"
            "  카메라를 어느 쪽으로 눕혔는지 모르면 --rotate 90 과 --rotate 270 을\n"
            "  각각 띄워보고 사람이 똑바로 서 보이는 쪽을 쓰세요."
        ) from None


def open_source(spec, cfg, camera_index=None, rotate_deg=None) -> FrameSource:
    """spec: "csi" | "usb" | 영상 파일 경로."""
    src = PiCameraSource(cfg) if spec == "csi" else OpenCVSource(spec, cfg, camera_index)
    deg = cfg["rotate_deg"] if rotate_deg is None else rotate_deg
    src.rotate_code = rotate_code_for(deg)
    src.rotate_deg = int(deg) % 360
    return src


# ============================================================
# 2. 검출 계층 — 보드/모델 교체 시 이 클래스만 영향
# ============================================================
@dataclass
class Person:
    """검출기 출력의 표준 형식. 로직 계층은 이것만 안다."""
    bbox: tuple          # (x1, y1, x2, y2) 픽셀
    conf: float
    keypoints: np.ndarray | None = None  # (17, 3) COCO 포즈 or None
    track_id: int | None = None          # 트래커 ID. 저 conf 구간에선 None으로 빠지지만
                                         # 쓰러짐 판정에서 제외하지 않는다 (낙상 순간이 하필
                                         # ID가 가장 잘 빠지는 구간). FallTracker가 겹침으로
                                         # 이어붙이고, 안 되면 음수 임시 키를 발급한다
                                         # → update() 이후 이 값은 음수일 수 있다.


class Detector:
    """사람 검출 + 포즈 + ID 추적. 포팅 시 내부 구현만 갈아끼움 (인터페이스 불변)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.pose = YOLO(cfg["pose_model"])

    def detect(self, frame) -> list[Person]:
        # persist=True: 프레임 간 트래커 상태를 유지해 같은 사람에게 같은 ID를 준다.
        # (저사양 보드용 교대 추론 — pose는 2프레임에 1번 — 도 이 메서드만 고치면 된다)
        res = self.pose.track(frame, imgsz=self.cfg["imgsz"],
                              conf=self.cfg["conf_thres"], persist=True,
                              tracker=self.cfg["tracker"], verbose=False)[0]
        people = []
        if res.boxes is None:
            return people
        kpts = res.keypoints.data.cpu().numpy() if res.keypoints is not None else None
        # 낙상 순간엔 ID가 빠지는 프레임이 흔하다 — None 그대로 넘겨 FallTracker가 잇는다.
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

    넘어지면 bbox 하단(=발 위치)이 크게 튄다. 발 한 점 기준은 하필 쓰러짐을
    판정할 순간에 가장 불안정하므로, 여기서는 몸 전체의 겹침 면적으로 본다."""
    x1, y1, x2, y2 = person.bbox
    rx1, ry1, rx2, ry2 = roi_px
    iw = max(0, min(x2, rx2) - max(x1, rx1))
    ih = max(0, min(y2, ry2) - max(y1, ry1))
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return (iw * ih) / area if area > 0 else 0.0


def bbox_overlap(a, b) -> float:
    """트랙 ID 교체 보정용 겹침도 — 교집합 / **작은 쪽 넓이**.

    IoU는 안 된다: 넘어지면 bbox가 세로→가로로 바뀌어 같은 사람인데도 IoU가 뚝
    떨어진다 (UR Fall 실측 낙상 전후 IoU 0.18 < 임계 0.3, 이 지표는 0.36)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    smaller = min(max(0, ax2 - ax1) * max(0, ay2 - ay1),
                  max(0, bx2 - bx1) * max(0, by2 - by1))
    return (iw * ih) / smaller if smaller > 0 else 0.0


def torso_angle_deg(kp: np.ndarray, min_conf: float = 0.3) -> float | None:
    """어깨 중점→엉덩이 중점 벡터가 수직에서 기운 각도.
    COCO: 5,6=어깨 / 11,12=엉덩이. conf가 min_conf 미만이면 None(=폴백으로 넘김)."""
    if kp is None:
        return None
    sh = kp[[5, 6]]
    hip = kp[[11, 12]]
    if sh[:, 2].min() < min_conf or hip[:, 2].min() < min_conf:
        return None
    v = hip[:, :2].mean(axis=0) - sh[:, :2].mean(axis=0)
    if np.linalg.norm(v) < 1e-6:
        return None
    # 수직(0,1)과의 각도
    cos = abs(v[1]) / np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(cos, 0, 1))))


def looks_fallen(person: Person, cfg) -> bool:
    """단일 프레임 쓰러짐 *후보* 판정 (확정은 시간 누적으로)."""
    ang = torso_angle_deg(person.keypoints, cfg["kp_conf_min"])
    if ang is not None:
        return ang > cfg["fall_angle_deg"]
    # 포즈 신뢰도 낮으면 bbox 비율로 폴백 (누운 사람은 가로로 길다)
    x1, y1, x2, y2 = person.bbox
    w, h = x2 - x1, y2 - y1
    return h > 0 and (w / h) > cfg["fall_aspect_ratio"]


class FallMonitor:
    """**한 사람**의 쓰러짐 확정/해제를 비대칭 시간 히스테리시스로 판단 (저 fps 친화).

    여러 명을 OR로 뭉치면 서로 다른 사람의 짧은 자세 이상이 합산돼 오발동하므로
    이 객체는 track_id 하나만 담당한다 (관리는 FallTracker).

    - 발동: 후보가 fall_confirm_sec 유지되면 확정(사이렌). gap_sec 이내의 짧은
            미검출은 무시하고 이어가고, 그보다 긴 공백은 '일어남'으로 보고 취소한다.
            확정은 쓰러짐이 실제로 보인 프레임에서만 일어나므로 gap_sec을 키워도
            '3초 안에 일어나면 사이렌 없음' 보장은 깨지지 않는다.
    - 해제: 확정 이후 '쓰러짐 아님'(정상 자세 or 횡단보도 이탈)이 fall_clear_sec
            연속 유지돼야 해제 — 검출이 깜빡여도 사이렌은 안 꺼진다.
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
            # 미확정 상태의 짧은 미검출: gap_sec 이내면 후보 유지(디바운스),
            # 그보다 길면 실제로 일어난 것으로 보고 취소.
            self.clear_since = None
            if (self.candidate_since is not None and self.last_fallen is not None
                    and now - self.last_fallen > gap_sec):
                self.candidate_since = None
        return self.confirmed


class FallTracker:
    """사람별 FallMonitor 관리 계층. FallMonitor의 상태 기계 자체는 건드리지 않는다.

    1. 사람별 분리 — track_id마다 독립 FallMonitor. 프레임 전체를 OR로 뭉치면
       A가 1.5초 + B가 1.5초만 이상자세여도 합산돼 3초 사이렌이 울렸다.
    2. 프레임 간격 적응 — 갭이 고정 1초면 프레임 간격이 그보다 긴 보드에서 한 번
       깜빡이는 것만으로 후보가 취소된다. 실측 간격 x N으로 갭과 유예 창(grace)을
       함께 올린다 — 하나만 고정으로 남으면 저 fps에서 그쪽이 먼저 무너진다.
    3. ID 공백/교체 보정 — 낙상 순간이 하필 트래커가 가장 약한 구간이다. UR Fall
       실측(fall-01)에서 ID가 None → 2 → None으로 끊겨 낙상 구간만 쏙 빠졌다.
       그래서 직전 프레임 위치와의 겹침으로 같은 사람에게 이어붙인다.
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
        튀는 값에 안 흔들리게 평균이 아니라 중앙값. update()당 정확히 한 번만
        호출할 것 — 두 번 부르면 간격 0이 섞여 중앙값이 무너진다."""
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

        gap_sec와 같은 이유로 프레임 간격에 맞춰 늘린다. 고정 1.5초면 프레임 간격이
        그보다 길어지는 순간(≈0.67fps 미만) 직전 프레임의 트랙조차 유예 밖으로 밀려,
        매 프레임 새 synth 키가 발급돼 누적이 리셋되고 사이렌이 영영 안 울린다."""
        if med is None:
            return self.cfg["track_grace_sec"]
        return min(max(self.cfg["track_grace_sec"], med * self.cfg["track_grace_frames"]),
                   self.cfg["track_grace_max_sec"])

    def _match(self, bbox, now, taken, grace: float) -> int | None:
        """직전 프레임 위치와 가장 많이 겹치는 기존 트랙의 키 (없으면 None).

        위치 이력이 없는 키는 후보에서 제외한다 — 기본값으로 얼버무리면 상관없는
        사람이 남의 사이렌 상태를 물려받는다. 판단 불가는 '매칭 안 함'이어야 한다."""
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

        트래커 ID가 1순위. ID가 없거나(저 conf) 처음 보는 ID면 직전 위치와의 겹침으로
        기존 트랙에 이어붙인다 — 이 보정이 없으면 낙상 순간의 ID 공백에서 누적이
        리셋돼 쓰러짐을 통째로 놓친다 (UR Fall 실측으로 확인된 실패 모드).
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
        """확정(사이렌) 상태는 해제될 때까지 유지. 그 외에는 갭/해제 판정이 모두 끝날
        만큼 오래 안 보이면 정리한다 (관측 이력이 없는 키는 즉시).

        ttl = fall_clear_sec + gap + grace 이므로 '아직 물려받을 수 있는
        (now - last_seen <= grace)' 트랙은 절대 먼저 정리되지 않는다. gap/grace 중
        하나만 고정값이면 이 불변식이 깨져 _match가 찾을 트랙을 _retire가 미리 지운다."""
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

        # 안 보인 트랙도 시간을 흘려보내야 미검출 갭/사이렌 해제 판정이 진행된다.
        # 빼먹으면 사람이 사라져도 사이렌이 영원히 안 꺼진다.
        for key in [k for k in self.monitors if k not in seen]:
            self.monitors[key].update(False, now, gap)

        self._retire(now, gap, grace)
        return {k for k, m in self.monitors.items() if m.confirmed}

    def reset(self):
        """모든 쓰러짐 누적을 버린다 (수동 리셋용). 프레임 간격 추정치는 보드 성능
        측정이라 유지한다. 진짜 쓰러진 사람이 계속 보이면 3초 뒤 다시 확정된다."""
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
                # 핵심 개입 지점: 아직 횡단보도에 사람이 있나?
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

    ts는 벽시계, t_logic은 로직이 실제로 쓴 시각(--video면 '영상 속 시각')이다.
    둘이 다르므로 frame_idx + t_logic이 있어야 로그 행을 영상 프레임에 붙여
    raw_data/labels의 정답 라벨과 대조하고 낙상 검출률을 낼 수 있다.
    """

    COLUMNS = ["ts", "t_logic", "frame_idx", "fps", "n_people", "n_in_roi",
               "n_fallen", "n_confirmed", "state", "infer_ms"]

    def __init__(self, path):
        self.path = Path(path)
        # 스키마가 다르면 이전 로직의 기록이다. 이어 붙이면 보고서에서 서로 다른
        # 로직의 숫자가 한 표로 보이므로 밀어낸다.
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
        self.f.flush()  # 크래시/전원차단에도 지금까지 행은 디스크에 남게
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
    ap.add_argument("--rotate", type=int, default=None, choices=[0, 90, 180, 270],
                    help="프레임을 시계방향으로 회전 (기본: CONFIG['rotate_deg']). "
                         "카메라를 눕혀 단 경우 90 또는 270을 각각 띄워보고 "
                         "사람이 똑바로 서 보이는 쪽으로 확정할 것")
    args = ap.parse_args()

    # 창을 띄울 수 있는지는 '띄우기 전에' 판별해야 한다. 디스플레이가 없으면
    # cv2.imshow가 예외가 아니라 SIGABRT로 프로세스를 통째로 죽여서(실측: 2프레임 만에
    # exit 134) try/except로는 못 막고 finally의 정리까지 건너뛴다.
    # (아래쪽 except cv2.error 폴백은 headless 빌드 전용이다.)
    show = not args.no_display
    if show and os.name != "nt" and not os.environ.get("DISPLAY"):
        print("DISPLAY가 없어 창 없이 진행합니다 (--no-display와 동일). 종료는 Ctrl+C.")
        show = False

    spec = args.source or args.video or CONFIG["source"]
    src = open_source(spec, CONFIG, args.camera, args.rotate)

    # 시간축: 라이브는 벽시계, 영상 파일은 '영상 속 시각'. 재생 속도와 무관하게
    # fall_confirm_sec(3초) 같은 판단이 '영상 속 3초'로 계산되게 한다 (평가 정합성).
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
    rot_msg = f" 회전 {src.rotate_deg}도(시계)." if src.rotate_code is not None else ""
    print(f"PoC 시작 [{spec}]{rot_msg} "
          f"{'q 종료 / r 수동리셋' if show else '창 없음(Ctrl+C 종료)'}. "
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
            if frame_idx == 0:
                # 회전이 먹었는지는 해상도로 보는 게 제일 확실하다 (세로가 길어야 정상)
                print(f"프레임 {W}x{H} — {'세로 김 (의도대로)' if H > W else '가로 김'}"
                      + ("" if H > W or src.rotate_code is not None
                         else " / 카메라를 눕혔다면 --rotate 90 또는 270 필요"))
            r = CONFIG["crosswalk_roi"]
            roi_px = (int(r[0] * W), int(r[1] * H), int(r[2] * W), int(r[3] * H))

            # 이번 프레임의 '판단 기준 시각' (웹캠=실시간, 영상=영상 속 시간)
            now = frame_idx / video_fps if use_video_clock else time.time()
            frame_idx += 1

            t0 = time.time()
            people = det.detect(frame)
            infer_ms = (time.time() - t0) * 1000

            # 사람마다 한 번만 계산해 재사용. ROI 기준은 둘로 갈린다 — 신호 연장은
            # '발 위치', 쓰러짐은 '몸 전체 겹침' (쓰러지면 발 위치가 크게 튄다).
            roi_flags = [foot_in_roi(p, roi_px) for p in people]
            fallen_flags = [looks_fallen(p, CONFIG) for p in people]
            on_cw_flags = [roi_overlap_ratio(p, roi_px) >= CONFIG["fall_roi_overlap"]
                           for p in people]
            n_roi = sum(roi_flags)
            # 쓰러짐 누적은 사람별 (프레임 전체 OR 아님)
            confirmed_ids = fall_tracker.update(people, fallen_flags, on_cw_flags, now)
            fall_confirmed = bool(confirmed_ids)
            state = signal.update(n_roi, fall_confirmed, now)
            fps = metrics.log(frame_idx - 1, now, len(people), n_roi,
                              sum(fallen_flags), len(confirmed_ids), state, infer_ms)

            # ---- 시각화 ---- 창을 안 띄우면 그리는 연산 자체를 건너뛴다
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
                    # 음수는 FallTracker가 발급한 임시 키다. '#-1'로 뜨면 시연 중
                    # 설명거리가 되니 '추적 중, ID 미확정'으로 표시한다.
                    label = f"#{p.track_id}" if p.track_id >= 0 else "#?"
                    cv2.putText(frame, label, (p.bbox[0], p.bbox[1] - 6),
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
                # GUI 없이 빌드된 opencv(headless)는 여기서 얌전히 예외를 낸다.
                # 디스플레이 자체가 없는 경우는 위쪽 DISPLAY 사전 검사로 막는다.
                print("opencv에 GUI 기능이 없음 — 창 없이 계속 진행합니다.")
                show, key = False, 255
            if key == ord("q"):
                break
            if key == ord("r"):
                fall_tracker.reset()
                signal.force_red(now)
                print(f"[{time.strftime('%H:%M:%S')}] ** 수동 리셋 — 쓰러짐 상태 초기화, 적색 복귀")
    finally:
        # 어느 경로로 끝나든 카메라/창/로그는 반드시 정리. 앞 단계가 던져도 뒤 단계는
        # 돌아야 한다 (예전엔 src.release() 실패 시 metrics.close()가 스킵됐다).
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
