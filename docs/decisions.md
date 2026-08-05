# 파라미터 결정 로그

CLAUDE.md 5장: 파라미터를 정할 때 근거를 남겨 대회 발표 자료에 활용한다.
새 항목은 아래 형식으로 위에 추가.

## 템플릿

### YYYY-MM-DD: 항목명

- **결정값**:
- **근거**:
- **참고 자료/벤치마크**:
- **관련 config 값**:

---

### 2026-08-05: 구역 5등분을 화면(픽셀)이 아니라 실제 거리(cm) 기준으로 수행

- **결정값**: 호모그래피가 있으면 평면 좌표(cm)에서 균등 분할 후 픽셀로 역변환
  (`CrosswalkZones._split_on_ground`). 실측 치수가 없을 때만 화면상 분할로 대체.
- **근거**: 사선 구도에서 화면상 균등 분할은 실제 거리로 균등하지 않다. 먼 쪽 1픽셀이
  가까운 쪽 1픽셀보다 훨씬 긴 실거리를 나타내기 때문이다.
  폭 400cm × 길이 1000cm 횡단보도를 사선으로 본 좌표
  `[(80,460), (560,460), (400,150), (240,150)]` 로 측정한 결과:

  | 분할 방식 | 1번 | 2번 | 3번 | 4번 | 5번 |
  |---|---|---|---|---|---|
  | 화면(픽셀) 기준 | 77cm | 105cm | 152cm | 238cm | 429cm |
  | 실거리(cm) 기준 | 200cm | 200cm | 200cm | 200cm | 200cm |

  화면 기준으로는 1번과 5번이 **5.6배** 차이났고, **횡단보도의 실제 한가운데(500cm)가
  3번이 아니라 4번 구역**에 들어갔다. 이는 "정중앙이면 가장 길게 연장"(CLAUDE.md 2.4)이라는
  설계 전제를 무효화한다.
- **참고 자료/벤치마크**: 사영변환은 직선을 직선으로 보내므로, 평면상 직사각형 띠의 네 꼭짓점만
  역변환하면 화면상 정확한 구역 사각형이 된다(근사가 아님). 회귀 방지 테스트는
  `tests/test_zone.py::test_zones_are_equal_in_real_distance_with_dimensions` 및
  `::test_true_center_of_crosswalk_lands_in_middle_zone`.
- **관련 config 값**: `CROSSWALK_REAL_WIDTH_CM`, `CROSSWALK_REAL_LENGTH_CM`
  (미입력이면 대체 경로로 떨어져 위 왜곡이 발생하므로 실측 우선순위가 높다),
  `CROSSWALK_ZONE_COUNT`.

### 2026-08-05: 라즈베리파이 CSI 카메라 접근 경로를 Picamera2로 확정

- **결정값**: `src/capture.py`에 Picamera2 백엔드 추가, `config.CAMERA_SOURCE = "picamera2"` 로 선택.
- **근거**: 하드웨어가 라즈베리파이 5 + CSI 리본 카메라로 확정됐다. 파이 5는 Raspberry Pi OS
  Bookworm 이상만 지원하고 Bookworm에는 레거시 카메라 스택(`bcm2835-v4l2`)이 없어
  CSI 카메라가 `/dev/video0`으로 잡히지 않는다. `cv2.VideoCapture(0)`은 실패한다.
  담당자가 파이에서 `tools/manual_rpicam_person_check.py`(Picamera2 직접 호출)로 정상 동작을 확인했다.
- **참고 자료/벤치마크**: Picamera2의 `format="RGB888"`은 이름과 달리 메모리상 BGR 순서 배열을
  반환한다(libcamera 명명 규칙). 덕분에 cv2 백엔드와 형식이 일치해 하위 단계가 카메라 종류를
  몰라도 된다. 라즈베리파이 실측 FPS는 아직 미측정.
- **관련 config 값**: `CAMERA_SOURCE`, `CAMERA_RESOLUTION`.

### 2026-08-05: 검출 모델을 COCO 사전학습 yolov8n으로 확정

- **결정값**: `DETECTION_MODEL_PATH = "yolov8n.pt"`, `PEDESTRIAN_LABEL = "person"`,
  `DETECTION_CONFIDENCE_THRESHOLD = 0.5`.
- **근거**: 담당자가 카메라로 실측한 결과 COCO 사전학습 yolov8n이 사람 모형을 `person`으로
  검출했고 신뢰도가 80%대로 관측됐다. 추가 파인튜닝이 필요 없다고 판단. 임계값 0.5는 관측치
  80%대 대비 여유가 있는 값이며, 오탐 발생 시 재조정한다.
- **참고 자료/벤치마크**: 라즈베리파이 실측 FPS는 아직 미측정 —
  `tools/manual_camera_person_check.py` 화면 좌상단 FPS 값으로 확인할 것.
- **관련 config 값**: `DETECTION_MODEL_PATH`, `PEDESTRIAN_LABEL`,
  `DETECTION_CONFIDENCE_THRESHOLD`, `DETECTION_TRACKER`.
