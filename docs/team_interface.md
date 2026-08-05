# 팀 인터페이스 합의 필요 사항

CLAUDE.md 6장 기준, 목표 2(쓰러짐 감지) 팀원 및 하드웨어팀과 합의가 필요한 항목.
합의되는 대로 이 문서를 갱신하고 `config/config.py`의 해당 값을 채운다.

## 신호 연장 파라미터 (담당 파트)

- [ ] 구역별 연장 시간 — `ZONE_EXTENSION_SEC` 2·3·4번 (초). 현재 담당자 예시값(3/5/3초)이라 None.
- [ ] 연장 트리거 임계값 — `REMAINING_TIME_THRESHOLD_SEC` (남은 시간이 이 값 이하일 때 연장 검토).
- [ ] 연장 누적 상한 — `MAX_TOTAL_EXTENSION_SEC` (초).
- [ ] 잔류 확정 프레임 수 — `ZONE_RESIDENCY_FRAMES` (실측 FPS 기반).

## 캘리브레이션 / 속도 추정

- [ ] **횡단보도 모형 실측 치수** — `CROSSWALK_REAL_WIDTH_CM`, `CROSSWALK_REAL_LENGTH_CM`.
      속도를 cm/s로 내려면 필수. 없으면 px/s로만 나오고 예상 통과 시간은 계산되지 않는다.
      별도의 체스보드 캘리브레이션은 필요 없다 — `tools/zone_calibrator.py`에서 네 꼭짓점을
      찍을 때 `--width-cm` / `--length-cm` 만 같이 주면 된다.
- [ ] 카메라 최종 설치 위치 고정 후 캘리브레이션 수행 (카메라를 옮기면 zone 좌표·호모그래피 모두 무효).
- [ ] 캘리브레이션 해상도와 운영 해상도 일치 확인 (다르면 좌표가 통째로 어긋난다).
- [ ] 속도 윈도우 `SPEED_WINDOW_SEC` — 사람 모형을 일정 속도로 움직여 실측 오차 확인 후 조정.
- [ ] 속도 추정값을 연장 시간 결정에 반영할지 (`USE_SPEED_FOR_EXTENSION`). 현재 False — 계산·노출만.

## 교통약자 우선 연장 — **보류**

COCO 사전학습 yolov8n에 휠체어/목발/지팡이 클래스가 없어 검출 수단 자체가 없다.
`MOBILITY_AID_LABELS = ()` 로 두어 `priority_mode`가 항상 False다. 파인튜닝 데이터셋을
확보하면 아래를 재개한다.

- [ ] 파인튜닝 데이터셋 확보 여부 결정 (재개 조건)
- [ ] 우선 연장 구역별 차등 — `PRIORITY_ZONE_EXTENSION_SEC`.
- [ ] 우선 연장 별도 상한 — `PRIORITY_MAX_TOTAL_EXTENSION_SEC`.
- [ ] "처음부터 넉넉한 기본 시간"을 제어부가 어떻게 세팅할지 (priority 플래그 처리 방식).
- [ ] 지팡이(cane) 검출 정확도 검증 (사선·저해상도에서 매우 작게 보임).

## 검출 모델

- [x] **모델/가중치 선정 — COCO 사전학습 `yolov8n.pt`, `person` 클래스.**
      근거: 사람 모형이 신뢰도 80%대로 검출됨(담당자 실측). 추가 파인튜닝 불필요.
- [ ] 라즈베리파이 실측 FPS 벤치마크 (아직 미측정 — 추정치를 기록하지 말 것).
      느리면 `ncnn`/`tflite` 변환, 입력 해상도 축소 검토.
- [ ] 추적기 선택 확정 — `DETECTION_TRACKER` (현재 `bytetrack.yaml`).
      track_id가 끊기면 속도 추정과 잔류 판정이 모두 리셋되므로 안정성 확인 필요.

## 하드웨어 / 통신

- [ ] **제어부/구동부 보드 구성 확정** — 현재 ESP32(제어부) + 아두이노 우노(구동부)는 **잠정안**.
- [ ] 라즈베리파이-제어부 통신 프로토콜 — `SERIAL_PORT`, `SERIAL_BAUDRATE`, `SERIAL_MESSAGE_FORMAT`.
      (파이 → 제어부 연장/priority 명령, 제어부 → 파이 잔여 시간 읽기 양방향 포함)
- [ ] 카메라 배치 방식 (배치 A: 파이에서 직접 실행 / 배치 B: 파이 카메라 + PC 추론) — CLAUDE.md 7장.
- [x] **CSI 카메라 접근 경로 — Picamera2로 확정.**
      라즈베리파이 5는 Bookworm 이상만 지원하고 레거시 카메라 스택이 없어
      `cv2.VideoCapture(0)`으로 CSI 카메라가 열리지 않는다. `src/capture.py`에 Picamera2
      백엔드를 두고 `config.CAMERA_SOURCE = "picamera2"` 로 선택한다.
      근거: 담당자가 파이에서 `tools/manual_rpicam_person_check.py`(Picamera2)로 정상 동작 확인.
- [ ] 파이에 `python3-picamera2` 설치 확인 (`sudo apt install -y python3-picamera2`).
      파이에서 venv를 쓴다면 `--system-site-packages`로 만들어야 보인다.
- [ ] 운영 해상도 확정 — `CAMERA_RESOLUTION` (현재 640x480).
      **캘리브레이션 해상도와 반드시 일치해야 한다.** 파이 FPS가 부족해 낮추면 재캘리브레이션 필요.
- [ ] 카메라 설치 각도/높이 실측값 — `CAMERA_MOUNT_ANGLE_DEG` (사선 구도, 탑뷰 아님).

## 목표 2 연계

- [ ] 목표2 파이프라인과 사람 검출 모델/좌표계 공유 여부.
      공유 시 `src/detection.py`의 `BoundingBox`를 그대로 쓰면 된다 — 낙상 판정에 필요한
      `center_point()`, `aspect_ratio()`를 이미 노출하고 있다.
