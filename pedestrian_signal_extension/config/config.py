from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_WEIGHT_DIRS = (_PROJECT_ROOT, _PROJECT_ROOT.parent)

def _weight_path(name: str) -> str:

    for directory in _WEIGHT_DIRS:
        path = directory / name
        if path.exists():
            return str(path)
    return name


CROSSWALK_ZONE_COUNT = 5
ZONE_RESIDENCY_FRAMES = 2 
ZONE_CONFIG_PATH = str(_PROJECT_ROOT / "data" / "zone_config.json")
TRACK_GRACE_FRAMES = 2

#   하한 — **공백 동안 사람이 움직인 거리보다 커야 한다.** 
#   상한 — **구역 크기(40cm/5 = 8cm)를 넘으면 안 된다.** 넘으면 옆 구역의 남남을 잇는다.
TRACK_INHERIT_DISTANCE_CM = 5.0
TRACK_INHERIT_DISTANCE_PX = 50.0

CROSSWALK_REAL_WIDTH_CM = 30.0   
CROSSWALK_REAL_LENGTH_CM = 36.0  

SPEED_WINDOW_SEC = 0.5
SPEED_MIN_SAMPLES = 2
SPEED_STOPPED_THRESHOLD_CM_S = 0.5


PEDESTRIAN_LABEL = "person"    
DETECTION_MODEL_PATH = _weight_path("yolov8n-pose.pt")  
DETECTION_CONFIDENCE_THRESHOLD = 0.1
DETECTION_TRACKER = "bytetrack.yaml"
DETECTION_IMGSZ = 320

CAMERA_SOURCE = "picamera2"
CAMERA_RESOLUTION = (640, 480)
CAMERA_MOUNT_ANGLE_DEG = None   

SERIAL_PORT = None       # None이면 open() 시 자동 탐색(아두이노 VID 우선). 고정하려면 "/dev/ttyACM0", "COM3" 등.
SERIAL_BAUDRATE = 115200   
SERIAL_MESSAGE_FORMAT = "ascii-lines"
SERIAL_READY_TIMEOUT_SEC = 5.0
SERIAL_STATE_HEARTBEAT_SEC = 1.0



FALL_CONFIG = {
    # 데모 셋업 후 실측으로 조정. 이 박스 안 = "횡단보도 위"
    "crosswalk_roi": (0.15, 0.30, 0.85, 0.95),

    # --- 쓰러짐 판정 ---
    "fall_angle_deg": 30.0,       # 몸통 축이 수직에서 이만큼 기울면 후보
    "fall_confirm_sec": 0.0,      # 이 시간 유지돼야 사이렌 확정 (3초 안에 일어나면 오탐으로 무시)
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
}

