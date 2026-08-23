# 팀 인터페이스 합의 필요 사항

CLAUDE.md 6장 기준, 목표 2(쓰러짐 감지) 팀원 및 하드웨어팀과 합의가 필요한 항목.
합의되는 대로 이 문서를 갱신하고 `config/config.py`의 해당 값을 채운다.

## 신호 연장 파라미터 (담당 파트)

- [x] **구역별 연장 시간 — `ZONE_EXTENSION_SEC` = {1:0, 2:3, 3:5, 4:3, 5:0}.**
      담당자 예시값이 아니라 국내 스마트횡단보도 운영 사례 기준. 3번 5초는 광주시("최대 5초")·
      음성군("5초 간격") 사례, 2·4번 3초는 서울시 "3~6초 연장"의 하한.
      → `docs/decisions.md` 2026-08-19 항목에 출처 링크. **팀 확인만 남음.**
- [x] **연장 누적 상한 — `MAX_TOTAL_EXTENSION_SEC = 10`.**
      강진군 자동연장 시스템의 "허용된 시간(5~10초) 범위" 상한. **팀 확인만 남음.**
- [ ] 연장 트리거 임계값 — `REMAINING_TIME_THRESHOLD_SEC = 5`.
      ⚠️ **이 값만 외부 근거가 아니라 설계 논리다**(1회 최대 연장량과 동일하게 설정).
      팀 논의로 확정할 것.
- [ ] **모형 스케일 검증** — 위 값들은 실제 도로 기준이다. 축소 모형에서 "5초 연장"이
      의미 있는 양인지 사람 모형을 실제로 움직여 확인하고, 과하면 비율(3 : 5 : 상한 10)을
      유지한 채 축소한다.
- [ ] 잔류 확정 프레임 수 — `ZONE_RESIDENCY_FRAMES` (실측 FPS 기반).

## 캘리브레이션 / 속도 추정

- [ ] **횡단보도 모형 실측 치수** — `CROSSWALK_REAL_WIDTH_CM`, `CROSSWALK_REAL_LENGTH_CM`.
      **속도와 구역 분할 둘 다에 필요하다(우선순위 높음).**
      - 없으면 속도가 px/s로만 나오고 예상 통과 시간은 계산되지 않는다.
      - 없으면 구역도 화면 기준으로 나뉘어 **실제 거리로 균등하지 않다.** 사선 구도 예시에서
        1번 77cm / 5번 429cm로 5.6배 차이가 났고, 실제 한가운데가 3번이 아닌 4번 구역에 들어갔다.
        "정중앙이면 가장 길게 연장" 규칙이 엉뚱한 위치에 적용된다는 뜻이다.
      별도의 체스보드 캘리브레이션은 필요 없다 — `tools/zone_calibrator.py`에서 네 꼭짓점을
      찍을 때 `--width-cm` / `--length-cm` 만 같이 주면 된다.
- [ ] 카메라 최종 설치 위치 고정 후 캘리브레이션 수행 (카메라를 옮기면 zone 좌표·호모그래피 모두 무효).
- [ ] 캘리브레이션 해상도와 운영 해상도 일치 확인 (다르면 좌표가 통째로 어긋난다).
- [ ] 속도 윈도우 `SPEED_WINDOW_SEC` — 사람 모형을 일정 속도로 움직여 실측 오차 확인 후 조정.
- [x] **속도 추정값을 연장 시간 결정에 반영 — 구현 완료** (`USE_SPEED_FOR_EXTENSION`).
      켜면 구역별 연장 시간이 '상한'이 되고 실제 연장은 부족분만큼으로 줄어든다:
      `연장 = min(구역값, ceil(ETA x ETA_SAFETY_MARGIN - 잔여시간))`, 여러 명이면 사람별 최댓값.
      안전계수 1.25 = 경찰청 보행속도 기준의 교통약자 완화 비율(1.0m/s → 0.8m/s).
      → `docs/decisions.md` 2026-08-19 항목.
- [ ] **`USE_SPEED_FOR_EXTENSION`을 True로 켜기** — 기본값은 아직 `False`.
      켜기 전 순서: ① 캘리브레이션 + 실측 치수 입력 → ② `manual_camera_person_check.py`에서
      단위가 `cm/s`인지 확인 → ③ 사람 모형을 일정 속도로 움직여 속도·ETA 정확도 검증 → ④ 켜기.
      **실측 치수가 없으면 ETA가 `None`이라 켜도 구역 규칙만 쓰는 동작으로 되돌아간다.**

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

- [x] **모델/가중치 선정 — `yolov8n-pose.pt`, `person` 클래스.**
      처음엔 `yolov8n.pt`로 확정했으나(사람 모형 신뢰도 80%대, 담당자 실측),
      목표 2 병합 시 **추론 1회를 공유**하기 위해 포즈 가중치로 교체했다. 아래 "목표 2 병합" 참고.
      추가 파인튜닝은 여전히 불필요.
- [ ] 라즈베리파이 실측 FPS 벤치마크 (아직 미측정 — 추정치를 기록하지 말 것).
      느리면 **`DETECTION_IMGSZ` 하향을 먼저** 검토(재캘리브레이션 불필요),
      그다음 `ncnn`/`tflite` 변환, `CAMERA_RESOLUTION` 축소(재캘리브레이션 필요).
- [ ] 추적기 선택 확정 — `DETECTION_TRACKER` (현재 `bytetrack.yaml`).
      track_id가 끊기면 속도 추정과 잔류 판정이 모두 리셋되므로 안정성 확인 필요.

## 목표 2 (쓰러짐 감지) 병합 — `src/fall_detection.py`

팀원의 PoC(`crosswalk_poc.py`)에 있던 쓰러짐 판단 로직을 **한 글자도 바꾸지 않고**
`src/fall_detection.py`로 옮겼다. 원문과 바이트 단위로 동일함을 확인했고
(`Person` / `foot_in_roi` ~ `FallTracker`), 무작위 시나리오 300회 × 40프레임으로
동작이 같은 것도 확인했다. 바꾼 것은 카메라·검출 배선뿐이다.

- [x] **카메라 일원화** — PoC의 `FrameSource`/`OpenCVSource`/`PiCameraSource` 대신
      `src/capture.py`의 `CameraCapture`를 쓴다. 양쪽 다 "파이 5 Bookworm에는 레거시
      카메라 스택이 없어 CSI를 cv2로 못 연다"는 같은 문제를 각자 풀고 있었다.
- [x] **검출 일원화** — PoC의 `Detector` 대신 `src/detection.py`의 `PersonDetector`.
      `BoundingBox.keypoints`가 추가되어 포즈 결과를 그대로 실어 보낸다.
- [x] **검출 모델을 `yolov8n-pose.pt`로 교체** — 포즈 모델 하나가 사람 박스와 키포인트를
      같이 주므로 **프레임당 추론 1회로 두 목표가 다 돌아간다.** 사람 4명 이미지로 전 경로 검증됨
      (박스 + (17,3) 키포인트 + 몸통각도 산출). → `docs/decisions.md` 2026-08-19 항목.
- [ ] **⚠️ 파이에서 포즈 모델 FPS 재측정** — 목표 1의 모델은 실측으로 확정됐던 사항인데
      (CLAUDE.md 2.6) 포즈 모델은 헤드가 하나 더 붙어 느려질 수 있다. 느리면 `DETECTION_IMGSZ`를
      먼저 낮출 것(320이면 약 2.8배 빠르고 **재캘리브레이션 불필요**).
- [ ] **⚠️ `DETECTION_CONFIDENCE_THRESHOLD` 확인** — 추론을 공유하니 임계값도 하나뿐이다.
      PoC 0.4 / 목표 1 0.5 였고 **낮은 쪽 0.4를 택했다** (쓰러진 사람을 놓치는 것이
      연장 오탐보다 위험하므로). 팀 확인 필요.
- [x] **파라미터 config 일원화** — PoC의 `pose_model`/`imgsz`/`conf_thres`/`tracker`와
      쓰러짐 판정값 13개가 전부 `config/config.py`로 모였다
      (`DETECTION_*`, `FALL_CONFIG`). 소스에는 하드코딩된 파라미터가 없다.
- [ ] **⚠️ 신호 제어 충돌 — 팀 합의 필요.** PoC의 `SignalController`/`SignalState`는
      **가져오지 않았다.** 그 안에 자체 녹색 연장 로직이 있어서 목표 1의
      `src/signal_extend.py`(구역·속도·상한·엣지 트리거)와 동시에 돌면 충돌한다.
      `fall_detection.py`는 "쓰러짐이 확정됐는가"까지만 책임진다.
      **정해야 할 것: 쓰러짐 확정(EMERGENCY) 시 신호를 어떻게 할 것인가?**
      (연장 중단 후 적색? 연장 유지? 별도 사이렌만?) 그리고 그 명령을 제어부에
      어떤 메시지로 보낼지.
- [ ] **ROI 기준 통일** — PoC는 화면 비율 사각형(`crosswalk_roi`), 목표 1은 캘리브레이션된
      네 꼭짓점을 쓴다. `roi_from_zones()`로 캘리브레이션 결과를 재사용할 수 있다(권장).
      단 원문 로직이 축평행 사각형을 전제하므로 사다리꼴을 감싸는 바운딩 박스가 되어
      실제 횡단보도보다 조금 넓다. 쓰러짐은 '놓치는 것'이 더 나쁘므로 안전한 방향의 오차다.
- [ ] `crosswalk_poc.py`의 `Metrics`(CSV 로깅)는 옮기지 않았다. 실측 근거 확보에 유용하므로
      필요하면 `tools/`에 수동 확인 스크립트로 두는 것을 검토.

## 하드웨어 / 통신

- [ ] **제어부/구동부 보드 구성 확정** — 현재 ESP32(제어부) + 아두이노 우노(구동부)는 **잠정안**.
- [ ] 라즈베리파이-제어부 통신 프로토콜 — `SERIAL_PORT`, `SERIAL_BAUDRATE`, `SERIAL_MESSAGE_FORMAT`.
      (파이 → 제어부 연장/priority 명령, 제어부 → 파이 잔여 시간 읽기 양방향 포함)
- [ ] **제어부 → 파이 "새 보행 신호 사이클 시작(녹색 시작)" 이벤트** — `SerialComm.read_cycle_started()`.
      신호 사이클의 소유자가 제어부이므로 이 이벤트도 제어부가 알려줘야 한다.
      **이게 없으면 누적 연장이 사이클을 넘어 남아, 한 번 상한을 찍은 뒤로는 영구히 연장이 안 된다.**
      파이가 잔여 시간의 증감만 보고 추측할 수는 없다 — "시간이 갑자기 늘었다"가 새 사이클인지
      우리가 방금 요청한 연장이 반영된 것인지 구분되지 않기 때문이다.
      (파이 쪽 수신부는 `SignalExtensionPipeline.begin_new_cycle()`로 이미 준비돼 있음.
       `docs/bugfix_log.md` B 항목 참고.)
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
