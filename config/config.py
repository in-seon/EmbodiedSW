"""
프로젝트 전역 설정.

팀과 아직 확정되지 않은 값은 None 또는 명확한 placeholder로 두고,
실제 사용 지점에서 값이 None이면 에러를 내도록 한다 (임의 추정치로 동작하지 않게 하기 위함).
CLAUDE.md 6장 "팀과 논의가 필요한 미확정 사항" 참고.
"""

# --- 신호 연장 로직 (CLAUDE.md 2.3) ---
EXTENSION_STEP_SEC = 5          # 확정: 5초씩 연장
MAX_TOTAL_EXTENSION_SEC = None  # TODO(팀 확정 필요): 연장 누적 상한 (초). 도로교통 규정 확인 후 결정.
REMAINING_TIME_THRESHOLD_SEC = None  # TODO(팀 확정 필요): 연장 트리거를 발동할 "남은 신호 시간" 임계값.

# --- 교통약자 우선 연장 (CLAUDE.md 2.4) ---
PRIORITY_EXTENSION_STEP_SEC = None   # TODO(팀 확정 필요): 우선 연장 시 단위 (5초와 다르게 할지 미정).
PRIORITY_MAX_TOTAL_EXTENSION_SEC = None  # TODO(팀 확정 필요): 우선 연장 시 별도 상한 여부 미정.

# --- Zone 잔류 판단 (CLAUDE.md 2.2) ---
ZONE_RESIDENCY_FRAMES = None    # TODO(실측 필요): 잔류로 판단할 연속 검출 프레임 수. 실측 FPS 기반으로 튜닝.
ZONE_CONFIG_PATH = "data/zone_config.json"  # tools/zone_calibrator.py 로 생성되는 zone 좌표 파일 경로.

# --- 사람 검출 모델 (CLAUDE.md 3.3) ---
DETECTION_MODEL_NAME = None     # TODO(팀 결정 필요): 실측 벤치마크 후 선정 (YOLO 경량 버전, MediaPipe 등 후보).
DETECTION_MODEL_PATH = None     # TODO: 선정된 모델 가중치 경로.
DETECTION_CONFIDENCE_THRESHOLD = 0.5  # 임시 기본값 — 실측 후 조정.

# --- 라즈베리파이 - 아두이노 시리얼 통신 (CLAUDE.md 1, 2.5) ---
SERIAL_PORT = None       # TODO(팀 합의 필요): 실제 포트 (예: "/dev/ttyUSB0", "COM3" 등 환경별로 다름).
SERIAL_BAUDRATE = None   # TODO(팀 합의 필요): 보드레이트.
SERIAL_MESSAGE_FORMAT = None  # TODO(팀 합의 필요): 메시지 포맷 스펙 (예: 프레이밍, 필드 정의).

# --- 카메라 (CLAUDE.md 2.1) ---
CAMERA_SOURCE = 0        # 기본 웹캠 인덱스. 실제 배치 시 확인 필요.
CAMERA_MOUNT_ANGLE_DEG = None   # TODO(실측 필요): 자동차 신호등 부착 각도. 캘리브레이션 시 사용.
