# 구현 상세 — 실행 순서, 조건, 파일별 입출력

이 문서는 **"코드가 실제로 어떤 순서로 무엇을 하는가"**를 파일 단위로 정리한 것이다.

- `CLAUDE.md` — 프로젝트 배경과 설계 의도 (왜 이렇게 만드는가)
- `docs/decisions.md` — 파라미터 값의 근거 (왜 이 숫자인가)
- `docs/bugfix_log.md` — 무엇이 잘못됐고 어떻게 고쳤는가
- `docs/log.md` — 무엇이 언제 왜 바뀌었는가 (시간순)
- **이 문서** — 무엇이 어떤 순서로 실행되고 각 함수가 무엇을 받아 무엇을 내놓는가

> 표기 규칙: `→` 반환값, `⚠️` 실패/예외 조건, `▸` 분기 조건.

---

## 0. 한눈에 보는 전체 그림

```
                          ┌──────────────── config/config.py ────────────────┐
                          │  모든 파라미터가 여기 한 곳에만 있다               │
                          └──────────────────────┬───────────────────────────┘
                                                 │ (모든 모듈이 읽음)
   ┌─────────────┐                               ▼
   │  카메라      │   frame(BGR)      ┌────────────────────────┐
   │ capture.py  │ ─────────────────▶ │  detection.py          │
   └─────────────┘                    │  PersonDetector        │
                                      │  (yolov8n-pose, 1회 추론)│
                                      └───────┬────────────────┘
                                              │ list[BoundingBox]
                                              │ (박스 + track_id + keypoints)
                        ┌─────────────────────┴─────────────────────┐
                        │                                           │
                 [목표 1 신호 연장]                          [목표 2 쓰러짐 감지]
                        │                                           │
          box.foot_point() = (x중앙, y하단)                 person_from_box()
                        │                                           │
        ┌───────────────┴───────────────┐                           ▼
        ▼                               ▼                   fall_detection.py
   zone.py                        ground_plane.py            looks_fallen()
   locate() → 구역번호            to_ground() → (cm,cm)      FallMonitor(사람별)
        │                               │                    FallTracker(관리)
        ▼                               ▼                           │
   CrosswalkOccupancy              speed.py                        ▼
   (N프레임 잔류 확정)             SpeedEstimator              confirmed_ids
        │                          → 속도/방향/ETA                  │
        └───────────┬───────────────────┘                           │
                    ▼                                               │
   CrossingProgress                                                 │
   (진입 방향 보정 → 진척도)                                          │
        │                                                           │
        ▼                                                           │
   signal_extend.py                                                 │
   minimum_progress()  → 가장 덜 건넌 사람                           │
        │                                                           │
        ▼                                                           ▼
   serial_comm.py                                        (미구현: 텔레그램 신고)
   update_state(NORMAL | ZONE <진척도> | FALL)            docs/team_interface.md 참고
   ▸ 변화 시 + 1초 하트비트에만 실제 전송
```

**핵심 구조 두 가지**

1. **추론은 프레임당 딱 1회.** `yolov8n-pose.pt` 하나가 사람 박스(목표 1)와
   키포인트(목표 2)를 함께 준다. 추론이 프레임 시간의 사실상 전부이므로(실측 약 4,400:1)
   모델을 두 개 돌리면 FPS가 반토막 난다.
2. **하위 단계는 카메라 종류를 모른다.** `capture.py`가 무엇이든 BGR numpy 배열로
   통일해 주므로, 검출 이하 단계는 CSI/USB/영상/스트림을 구분하지 않는다.

---

## 1. 사전 준비 — 실행 전에 반드시 끝나야 하는 것

순서가 중요하다. 앞 단계를 건너뛰면 뒤에서 `NotImplementedError`나 `ValueError`로 멈춘다.

| 순서 | 할 일 | 산출물 | 안 하면 |
|---|---|---|---|
| 1 | 카메라를 **최종 위치에 고정** | — | 이후 캘리브레이션이 전부 무효 |
| 2 | `config.CAMERA_SOURCE` 설정 (파이 CSI면 `"picamera2"`) | — | 카메라가 안 열림 |
| 3 | 횡단보도 모형 **실측**(가로/세로 cm) | 숫자 2개 | 속도가 px/s, 구역이 실거리 불균등 |
| 4 | `tools/zone_calibrator.py` 실행 | `data/zone_config.json` | `CrosswalkZones.load()` 실패 |
| 5 | `tools/manual_camera_person_check.py`로 FPS 측정 | FPS 수치 | `ZONE_RESIDENCY_FRAMES` 근거 없음 |
| 6 | `config.ZONE_RESIDENCY_FRAMES` 채우기 | — | `CrosswalkOccupancy` 생성 시 예외 |
| 7 | 시리얼 프로토콜 팀 합의 + `SERIAL_*` 채우기 | — | `SerialComm` 생성 시 예외 |
| 8 | `src/serial_comm.py` 세 메서드 구현 | — | `run()` 실행 시 예외 |

### 1.1 `tools/zone_calibrator.py`

```powershell
python tools/zone_calibrator.py --source picamera2 --width-cm 90 --length-cm 300
```

| 항목 | 내용 |
|---|---|
| **입력** | `--source` 카메라/이미지, `--width-cm` 폭(cm), `--length-cm` 길이(cm), `--n` 구역 수(기본 5), `--out` 저장 경로 |
| **조작** | 네 꼭짓점을 **①시작-왼쪽 ②시작-오른쪽 ③끝-오른쪽 ④끝-왼쪽** 순서로 클릭 → `s` 저장, `r` 초기화, `q` 종료 |
| **출력** | `data/zone_config.json` |

```json
{
  "name": "crosswalk",
  "corners": [[80,460],[560,460],[400,150],[240,150]],
  "n_zones": 5,
  "real_width_cm": 90.0,
  "real_length_cm": 300.0,
  "frame_size": [640, 480]
}
```

▸ `--width-cm` / `--length-cm`를 **생략하면** `real_*_cm`이 저장되지 않는다 →
호모그래피 없음 → 속도 px/s, 구역이 화면 기준 분할(실거리로 불균등).

⚠️ `frame_size`는 **캘리브레이션 당시 해상도**다. 운영 해상도가 다르면
`CrosswalkZones.load()`가 `ValueError`를 낸다(자세한 건 3.4).

---

## 2. 실시간 루프 — `main.py`

```bash
python main.py --mode full     # 쓰러짐 + 신호 연장 (기본)
python main.py --mode fall     # 쓰러짐만 (zone 설정 없이도 돈다)
```

### 2.1 파이가 하는 일 / 아두이노가 하는 일

파이는 **"무엇을 보았는가"**만 보낸다. **"언제 연장할까"는 아두이노가 정한다.**
잔여 녹색 시간을 7세그먼트로 직접 세는 쪽만 그 판단을 할 수 있기 때문이다
(→ `docs/decisions.md` 2026-08-26, 계약은 `docs/team_interface.md`).

| 파이 → 아두이노 | 언제 |
|---|---|
| `NORMAL` | 횡단보도에 확정 보행자가 없음 |
| `ZONE <1..5>` | 가장 덜 건넌 사람의 **진척도**(1=방금 진입, 5=거의 다 건넘) |
| `FALL` | 쓰러짐 확정 |

전송은 **상태가 바뀔 때 + 1초 하트비트**. 아두이노는 구역별 '정상 도착 기준 잔여시간'과
비교해 뒤처진 만큼 2초씩 연장한다 — 그 표가 정상 보행 속도를 담고 있어서 **파이가 속도를
잴 필요가 없다.**

### 2.2 생성 순서와 실패 지점

**하나라도 조건이 안 맞으면 그 자리에서 멈춘다.** 이 예외들은 버그가 아니라
**"이 값을 아직 안 정했다"는 알림**이며, 메시지에 무엇을 채워야 하는지 적혀 있다.

| 만드는 것 | 실패 조건 | 예외 | `--mode fall`도 막힘? |
|---|---|---|---|
| `PersonDetector()` | `DETECTION_MODEL_PATH is None` | `NotImplementedError` | ✅ |
| | 모델에 `PEDESTRIAN_LABEL` 클래스 없음 | `ValueError` | ✅ |
| `CrosswalkZones.load()` | zone 설정 없음 / 해상도 불일치 | `FileNotFoundError` / `ValueError` | ❌ 경고 후 화면 비율로 폴백 |
| `CrosswalkOccupancy()` | `ZONE_RESIDENCY_FRAMES is None` | `NotImplementedError` | ❌ `--confirm-frames`로 우회 가능 |
| `SerialComm()` | `SERIAL_BAUDRATE is None` | `NotImplementedError` | ✅ |
| | 아두이노가 READY/PONG을 안 보냄 | `RuntimeError` | ✅ `--no-serial`이면 건너뜀 |
| `SpeedEstimator()` | `SPEED_WINDOW_SEC <= 0` / `SPEED_MIN_SAMPLES < 2` | `ValueError` | ❌ |

> `--mode fall`이 zone 설정 문제에 관대한 이유: 쓰러짐 감지에 zone은 ROI 정확도를
> 높여주는 선택지일 뿐이다. 신호 연장 쪽 설정 문제로 쓰러짐까지 못 돌게 할 이유가 없다.
> 반대로 `--mode full`은 zone이 없으면 구역 판정 자체가 불가능하므로 그대로 실패한다.

### 2.3 매 프레임 실행 순서 (`CombinedPipeline.process_frame`)

```
반복:
  ① frame = camera.read_frame()                        ▸ None이면 루프 종료
  ② boxes     = detector.detect(frame)                 [추론 1회 — 전체 시간의 99.6%]
     aid_boxes = aid_detector.detect(frame)            [N프레임마다 1회, 없으면 빈 목록]
  ③ fall = fall_pipeline.process_boxes(boxes, frame)   키포인트 → 몸통 각도 → 확정 여부
  ④ ext  = ext_pipeline.process_boxes(boxes, aid_boxes) 발 위치 → 구역 → 진척도
  ⑤ state = FALL if fall.confirmed else ext.serial_state()
  ⑥ serial_comm.update_state(state, 진척도)            ▸ 변화 시 + 1초마다만 실제 전송
```

**②가 이 구조의 핵심이다.** 두 판정이 각자 `detect()`를 부르면 추론이 2배가 되고,
실측상 추론이 프레임 시간의 99.6%(82.4ms vs 나머지 0.35ms)라 FPS가 그대로 반토막 난다.
포즈 모델 하나로 박스와 키포인트를 함께 받는 설계(→ `decisions.md` 2026-08-19)가
여기서 실제로 지켜진다.

**⑤ 우선순위는 FALL > ZONE > NORMAL.** 채널이 하나이므로 상태도 하나다. 쓰러진 사람이
있으면 연장 요구보다 그쪽이 급하다. 쓰러짐이 풀리면 그 프레임의 연장 요구가 다시 나간다.

### 2.4 진척도를 만드는 순서 (`SignalExtensionPipeline.process_boxes`)

```
 1. detections = [(track_id, foot_point) for 사람 박스]
 2. confirmed  = occupancy.update(detections)      {track_id: 구역번호} — 잔류 확정된 사람만
 3. progress   = CrossingProgress.update(...)      물리 구역 → 진입 방향 보정 → 진척도
 4. speeds     = speed.update_many(detections, t)  {track_id: TrackSpeed}  ※ 계측용
 5. progress_zone = minimum_progress(occupants)    가장 작은 진척도 = 가장 덜 건넌 사람
 7. eta_sec = rule.max_eta_sec(occupants)          ▸ USE_SPEED_FOR_EXTENSION이 True일 때만
 8. return FrameResult(...)
```

**중요**: 6번에 들어가는 것은 전체 검출이 아니라 **확정 보행자만**이다.
`ZONE_RESIDENCY_FRAMES`를 못 채운 검출은 연장 요구를 만들지 못한다.

## 3. 파일별 입출력 상세

### 3.1 `src/capture.py` — 카메라 입력

| 클래스/함수 | 입력 | 출력 | 조건 |
|---|---|---|---|
| `CameraCapture(source, resolution)` | `source`: `"picamera2"` \| 정수 \| 파일경로 \| URL<br>`resolution`: `(w, h)` | 인스턴스 | 기본값은 `config.CAMERA_SOURCE`, `config.CAMERA_RESOLUTION` |
| `.open()` | — | `self` | ⚠️ 못 열면 `RuntimeError` (원인별 안내 메시지 포함) |
| `.read_frame()` | — | **BGR numpy 배열** 또는 `None` | ⚠️ `open()` 전 호출 시 `RuntimeError` |
| `.frames()` | — | 제너레이터 (BGR 배열) | ▸ `None`을 만나면 **즉시 중단** |
| `.backend_name` | — | `"cv2"` \| `"picamera2"` | 어느 경로로 열렸는지 |
| `grab_one_frame(source)` | 소스 | BGR 배열 1장 | ▸ 이미지 파일이면 `cv2.imread`로 바로 읽음 |

**백엔드 선택 규칙** (`_is_picamera2_source`)

▸ `source`가 문자열 `"picamera2"`(대소문자·공백 무시) → `Picamera2` 백엔드
▸ 그 외 전부 → `cv2.VideoCapture`

> 라즈베리파이 5 + Bookworm에는 레거시 카메라 스택이 없어 **CSI 카메라가
> `/dev/video0`으로 안 잡힌다.** `cv2.VideoCapture(0)`은 실패한다.
> Picamera2의 `format="RGB888"`은 이름과 달리 메모리에 **BGR 순서**로 담기므로
> (libcamera 명명 규칙) cv2와 형식이 같다 — 뒤집지 말 것.

**알려진 한계** (PoC에는 있으나 아직 안 옮긴 것)
- 프레임 읽기 실패 재시도 없음 → 웹캠이 한 장 흘리면 루프가 끝난다
- 큐에 쌓인 옛 프레임 버리기(drain) 없음 → 지연된 프레임으로 속도를 잴 수 있다

---

### 3.2 `src/detection.py` — 검출

#### `BoundingBox` (데이터 클래스)

| 필드 | 타입 | 설명 |
|---|---|---|
| `x1, y1, x2, y2` | float | 픽셀 좌표 |
| `confidence` | float | 신뢰도 |
| `label` | str | 모델 클래스명 (`"person"` 등) |
| `track_id` | int \| None | 추적 ID. `None`이면 잔류·속도 판정에서 **제외** |
| `keypoints` | (17,3) 배열 \| None | 포즈 가중치일 때만. 쓰러짐 판정용 |

| 메서드 | 출력 | 용도 |
|---|---|---|
| `.foot_point()` | `((x1+x2)/2, y2)` | **구역 판정·속도의 기준점.** 박스 하단 중앙 = 지면 접점 |
| `.center_point()` | `((x1+x2)/2, (y1+y2)/2)` | ⚠️ 지면 판정에 쓰지 말 것. 사람 키의 절반만큼 떠 있어 카메라 쪽으로 당겨진다 |
| `.is_pedestrian()` | bool | `label == config.PEDESTRIAN_LABEL` |
| `.is_mobility_aid()` | bool | `label in config.MOBILITY_AID_LABELS` ▸ 현재 항상 `False` |

#### `PersonDetector`

| 항목 | 내용 |
|---|---|
| **입력** | `frame` (BGR numpy 배열) |
| **출력** | `list[BoundingBox]` |
| **사용 파라미터** | `DETECTION_MODEL_PATH`, `DETECTION_CONFIDENCE_THRESHOLD`, `DETECTION_TRACKER`, `DETECTION_IMGSZ`, `PEDESTRIAN_LABEL`, `MOBILITY_AID_LABELS` |
| **내부 호출** | `model.track(frame, persist=True, conf=..., classes=..., tracker=..., imgsz=..., verbose=False)` |

▸ `persist=True`이므로 같은 사람에게 프레임 간 같은 `track_id`가 붙는다.
▸ 결과에 `keypoints`가 있으면(포즈 가중치) 박스별로 실어 보낸다. 없으면 `None`.
▸ `classes=`로 관심 클래스만 걸러 불필요한 박스를 안 만든다.
⚠️ `DETECTION_IMGSZ`를 낮춰도 **박스는 원본 프레임 좌표로 돌아온다** →
zone 좌표·호모그래피가 그대로 유효하다(재캘리브레이션 불필요).
이것이 `CAMERA_RESOLUTION`을 낮추는 것과 결정적으로 다른 점이다.

#### `MobilityAidDetector` (보류 기능, 파이프라인 미연결)

| 항목 | 내용 |
|---|---|
| **입력** | `frame` |
| **출력** | `list[BoundingBox]` (`track_id`는 항상 `None`) |
| **사용 파라미터** | `MOBILITY_AID_MODEL_PATH`, `_LABELS`, `_EVERY_N_FRAMES`, `_IMGSZ`, `_CONFIDENCE_THRESHOLD` |

▸ `MOBILITY_AID_MODEL_PATH is None` → `enabled=False`, 항상 `[]` 반환 (조용히 비활성)
▸ `every_n_frames` 마다 1회만 추론, 그 사이 프레임은 **직전 결과 재사용**
  (추론이 프레임 비용의 전부라 매 프레임 돌리면 FPS 반토막)
▸ `MOBILITY_AID_LABELS`가 비어 있으면 모델의 **전체 클래스**를 쓴다
  (후보 가중치의 클래스명을 모를 때 바로 시험 가능)
⚠️ 지정한 라벨이 모델에 없으면 `ValueError` + 모델이 아는 클래스 목록 출력

---

### 3.3 `src/ground_plane.py` — 호모그래피 (픽셀 ↔ cm)

**평면 좌표계 규약**: `x` = 폭 방향(0 ~ width_cm), `y` = **걷는 방향**(0 = 시작 변, length_cm = 끝 변)

| 메서드 | 입력 | 출력 | 조건 |
|---|---|---|---|
| `from_quad(corners, width_cm, length_cm)` | 네 꼭짓점(픽셀), 실측 치수 | `GroundPlane` 또는 **`None`** | ▸ 치수 중 하나라도 `None` → `None` 반환<br>⚠️ 꼭짓점 4개 아님 / 치수 ≤ 0 → `ValueError` |
| `.to_ground(point)` | `(px, py)` | `(x_cm, y_cm)` | ⚠️ 소실선 근처 점 → `ValueError` |
| `.to_pixel(ground_point)` | `(x_cm, y_cm)` | `(px, py)` | 위의 역변환 |
| `.remaining_distance_cm(gp, direction)` | 평면 좌표, 방향(+1/-1/0) | 남은 거리(cm) 또는 `None` | ▸ `+1` → `length_cm - y`<br>▸ `-1` → `y`<br>▸ `0` → `None` |

> 체스보드 카메라 캘리브레이션은 **하지 않는다.** 보행자의 발은 지면(평면) 위에 있으므로
> 평면 하나에 대한 사영변환이면 충분하고, 절차가 하나 줄어드는 편이 실제로 쓰일 확률이 높다.

---

### 3.4 `src/zone.py` — 구역 판정과 잔류 추적

#### `CrosswalkZones.load(path, expected_frame_size)`

| 항목 | 내용 |
|---|---|
| **입력** | JSON 경로(기본 `config.ZONE_CONFIG_PATH`), 기대 해상도(기본 `config.CAMERA_RESOLUTION`) |
| **출력** | `CrosswalkZones` (`.zones` 리스트 + `.ground_plane`) |

**해상도 검증** ▸ JSON에 `frame_size`가 있고 `expected_frame_size`와 **다르면 `ValueError`**.
▸ `frame_size` 키가 없는 옛 파일은 검증할 근거가 없으므로 통과.

> 이 검증이 없으면 **에러 없이 구역만 엉뚱하게 잡히는** 조용한 오작동이 난다.
> 현장에서 원인 찾기가 가장 어려운 유형이라 즉시 실패시킨다.

**치수 결정 순서**: JSON의 `real_*_cm` → 없으면 `config.CROSSWALK_REAL_*_CM` → 그것도 없으면 `None`

#### 5등분 방식 — **화면이 아니라 실거리 기준**

▸ 호모그래피 있음 → `_split_on_ground()`: 평면 좌표(cm)에서 균등 분할 후 픽셀로 역변환
▸ 호모그래피 없음 → `_split_on_pixels()`: 화면상 균등 분할 (**왜곡 감수**)

폭 400cm × 길이 1000cm를 사선으로 본 실측 예:

| 분할 방식 | 1번 | 2번 | 3번 | 4번 | 5번 |
|---|---|---|---|---|---|
| 화면(픽셀) 기준 | 77cm | 105cm | 152cm | 238cm | **429cm** |
| 실거리(cm) 기준 | 200cm | 200cm | 200cm | 200cm | 200cm |

화면 기준은 1번과 5번이 **5.6배** 차이나고, **횡단보도의 진짜 한가운데(500cm)가
3번이 아니라 4번**에 들어간다 → "정중앙이면 가장 길게 연장"이라는 전제가 깨진다.

#### `CrosswalkOccupancy.update(detections)`

| 항목 | 내용 |
|---|---|
| **입력** | `[(track_id, (px, py)), ...]` — 보통 `foot_point()` |
| **출력** | `{track_id: 구역번호}` — **확정 보행자만** |
| **부수 효과** | `.untracked_count` 갱신 |

프레임당 처리 순서:
```
각 검출에 대해:
  ▸ track_id is None       → 건너뜀, untracked_count += 1
  ▸ locate(point) is None  → 횡단보도 밖, 상태 삭제(카운트 리셋)
  ▸ 그 외                  → count += 1, zone 갱신
이번 프레임에 안 보인 track → 상태 삭제
반환: count >= confirm_frames 인 것만
```

⚠️ `track_id=None`을 건너뛰는 이유: `None`을 키로 쓰면 **서로 다른 사람이 한 카운터에
합쳐져**, 같은 프레임 안의 여러 명이 각각 카운트를 올린다(잔류 검증 무력화).
`SpeedEstimator`와 같은 규칙이다.

---

### 3.5 `src/speed.py` — 속도·방향·ETA

#### `TrackSpeed` (출력 데이터)

| 필드 | 설명 |
|---|---|
| `speed` | 평면상 전체 속력 (항상 ≥ 0) |
| `crossing_speed` | **걷는 방향(y) 성분만.** 좌우 흔들림 제외 |
| `direction` | `+1` 끝 변 쪽 / `-1` 시작 변 쪽 / `0` 정지·판별 불가 |
| `unit` | `"cm/s"`(호모그래피 있음) 또는 `"px/s"`(없음) |
| `position` | 이번 프레임 위치. 단위에 맞춰 cm 또는 px |

#### `update(track_id, foot_point, timestamp)` → `TrackSpeed` 또는 `None`

```
▸ track_id is None                    → None
발 위치를 측정 좌표로 변환 (호모그래피 있으면 cm, 없으면 px 그대로)
히스토리에 (timestamp, position) 추가
윈도우 밖 오래된 샘플 제거 (단, 최소 2개는 유지)
▸ 샘플 수 < min_samples               → None
▸ dt <= 0 (같은 시각 샘플)             → None
속도 = |끝점 - 시작점| / dt
```

⚠️ **양 끝 샘플만 쓴다.** 중간 샘플은 계산에 안 들어가므로 `SPEED_MIN_SAMPLES`를
늘려도 정확도가 좋아지지 않고, 첫 속도가 나오는 시점만 늦어진다. 게다가
`SPEED_WINDOW_SEC`(0.5초) 안에 N개가 쌓이려면 최소 `(N-1)/0.5` fps가 필요해서,
3으로 두면 **4fps 미만인 보드에서 속도가 영원히 `None`**이 된다.
노이즈를 줄이려면 `SPEED_WINDOW_SEC`를 늘릴 것.

> 한 프레임 차분을 안 쓰는 이유: dt가 짧으면 검출 박스 지터가 그대로 증폭된다
> (30fps에서 3px 지터 → 90px/s).

#### `update_many(detections, timestamp)` → `{track_id: TrackSpeed}`

▸ 이번 프레임에 안 보인 트랙의 히스토리는 **삭제**한다.
  사라졌다 같은 ID로 다시 나타나면 그 공백을 가로질러 변위를 재게 되어 엉뚱한 속도가 나온다.

#### `estimated_crossing_time_sec(track_id)` → 초 또는 `None`

```
ETA = remaining_distance_cm(현재 위치, 진행 방향) / crossing_speed

None이 되는 조건:
  ▸ 호모그래피 없음 (px 단위라 초로 환산 불가)
  ▸ 아직 속도가 계산되지 않음
  ▸ direction == 0 (정지)
  ▸ crossing_speed <= 0
```

---

### 3.6 `src/signal_extend.py` — 연장 결정 (핵심 로직)

#### `minimum_progress(occupants)` → `int | None`

확정 보행자 중 **가장 작은 진척도**. 아무도 없으면 `None`(→ `NORMAL`).

**최솟값 하나면 되는 이유**: 아두이노의 기준표는 진척도에 대해 단조 감소한다
(1번 9초 ... 5번 1초). 모든 사람이 같은 잔여 시간을 보므로

    부족분 = 기준[진척도] - 잔여시간

은 진척도가 작을수록 항상 크다. **가장 작은 진척도의 사람이 항상 가장 부족하다** —
나머지 값은 볼 필요가 없다.

#### `maximum_eta_sec(occupants)` → `float | None`

가장 큰 ETA. **전송하지 않는다** — 화면 표시·진단용이다. 값이 계속 `None`이면 프레임률이
낮거나 사람이 멈춰 있다는 뜻이지만, 연장 판단은 ETA 없이 돌아가므로 문제가 되지 않는다.

> **상태도 없고 속도도 안 쓴다.** 누적·무장·임계값 비교는 제어부가 소유하고, 정상 보행
> 속도는 아두이노의 기준표가 담고 있다. 이전에는 `SignalExtensionStateMachine`이
> 222줄이었다. 이관 경위는 `docs/decisions.md` 2026-08-26 두 항목 참고.

---

### 3.6b `src/zone.py` — `CrossingProgress` (진입 방향 보정)

물리 구역 번호는 좌표계 기준이라 **진입 방향에 따라 의미가 뒤집힌다.** 반대편에서 들어온
사람의 '2번'은 '거의 다 건넜음'인데, 보정하지 않으면 '대부분 남았음'으로 읽혀 엉뚱한
사람이 최솟값을 차지한다.

| 메서드 | 하는 일 |
|---|---|
| `update(track_id, zone)` → `int` | 구역을 반영하고 진척도(1..N)를 돌려준다 |
| `direction(track_id)` | `+1` / `-1` / `None`(미정) — 표시·진단용 |
| `keep_only(track_ids)` | 사라진 트랙 정리 (ID 재사용 시 옛 방향이 따라붙지 않도록) |

**방향 판별은 첫 검출 구역으로 한다** — 속도가 아니라 위치 이력이라 프레임률 요구가 없고,
사람이 멈춰 있어도 유지된다.

```
첫 검출이 중앙보다 앞 → 정방향(+1) → 진척도 = 구역
첫 검출이 중앙보다 뒤 → 역방향(-1) → 진척도 = N+1 - 구역
첫 검출이 중앙        → 미정        → 중앙값으로 간주, 이후 이동으로 확정
```

중앙에서 시작한 뒤 `3→4`든 `3→2`든 진척도 4로 수렴한다.

⚠️ **오판하면 어느 쪽으로 틀리는가**: 중간에 ID가 재발급되면 방향을 반대로 볼 수 있다.
그때 진척도는 **실제보다 작게** 나오고, 작은 진척도는 '덜 왔다'이므로 아두이노가
**더 연장하는** 쪽으로 틀린다 — 보행자가 갇히는 것보다 차가 기다리는 편이 낫다는
이 프로젝트의 기존 판단과 같은 방향이다.

---

### 3.7 `src/fall_detection.py` — 쓰러짐 감지 (목표 2)

> 판단 로직은 팀원 PoC(`crosswalk_poc.py`)에서 **한 글자도 바꾸지 않고** 옮겼다.
> 원문과 바이트 단위로 동일함을 확인했다.

#### 단일 프레임 판정 함수

| 함수 | 입력 | 출력 | 규칙 |
|---|---|---|---|
| `foot_in_roi(person, roi_px)` | Person, ROI 사각형 | bool | 발(하단 중앙)이 ROI 안인가 — **신호 연장용** |
| `roi_overlap_ratio(person, roi_px)` | 〃 | 0.0~1.0 | bbox가 ROI와 겹치는 면적 비 — **쓰러짐용** |
| `torso_angle_deg(kp)` | (17,3) 키포인트 | 각도 또는 `None` | 어깨(5,6)→엉덩이(11,12) 벡터가 수직에서 기운 각도. ▸ 신뢰도 < 0.3이면 `None` |
| `looks_fallen(person, cfg)` | Person, 설정 | bool | ▸ 각도 있음 → `각도 > fall_angle_deg(50°)`<br>▸ 각도 `None` → `w/h > fall_aspect_ratio(1.3)` 폴백 |
| `bbox_overlap(a, b)` | 박스 2개 | 0.0~1.0 | 교집합 / **작은 쪽 넓이** (IoU 아님) |

⚠️ **ROI 기준이 둘로 나뉘는 이유**: 넘어지면 bbox 하단(=발 위치)이 자세에 따라 크게 튄다.
발 한 점으로 쓰러짐까지 판정하면 **하필 쓰러진 순간에 ROI 밖으로 빠져 사이렌을 놓친다.**
몸 전체 겹침은 자세가 변해도 안 튄다.

⚠️ **IoU를 안 쓰는 이유**: 넘어지면 bbox 넓이 자체가 크게 변해(세로로 길다가 가로로 납작)
같은 사람인데도 IoU가 뚝 떨어진다. UR Fall 실측에서 낙상 직전/직후 IoU가 0.18까지 내려가
임계값 0.3에 걸려 다른 사람으로 끊겼다. 같은 두 프레임의 '교집합/작은쪽'은 0.36이었다.

#### `FallMonitor` — **한 사람**의 비대칭 시간 히스테리시스

```
update(any_fallen, now, gap_sec) → confirmed(bool)

▸ any_fallen == True:
     후보 시작 시각 기록 (없으면)
     경과 >= fall_confirm_sec(3초)  → confirmed = True  ★ 사이렌
▸ any_fallen == False 이고 이미 confirmed:
     정상 복귀 시작 시각 기록
     경과 >= fall_clear_sec(3초)    → confirmed = False ★ 해제
▸ any_fallen == False 이고 미확정:
     마지막 관측 후 경과 > gap_sec  → 후보 취소 (실제로 일어난 것)
     그 이내면                      → 후보 유지 (저 fps 깜빡임 무시)
```

**비대칭인 이유**: 발동은 3초 유지가 필요하고(3초 안에 일어나면 오탐으로 무시),
해제도 3초 연속 정상이어야 한다(검출이 깜빡여도 사이렌이 안 꺼진다).

#### `FallTracker` — 사람별 관리 + 프레임 간격 적응 + ID 보정

```
update(people, fallen_flags, on_cw_flags, now) → confirmed_ids(set)

1. _tick(now)        실측 프레임 간격 이력 갱신 → 중앙값
2. gap   = clamp(fall_gap_sec,   중앙값 × fall_gap_frames,   fall_gap_max_sec)
   grace = clamp(track_grace_sec, 중앙값 × track_grace_frames, track_grace_max_sec)
3. _resolve_keys()   사람 단위의 안정적인 키 결정
      1순위: 이미 관리 중인 track_id 그대로
      2순위: ID 없음/처음 보는 ID → 직전 위치와 겹침(>= 0.3)으로 이어붙임
             ▸ 이어붙일 곳 없음 → 새 사람 (ID 없으면 음수 임시 키)
             ▸ ID는 없고 매칭됨 → 기존 키 유지
             ▸ 새 ID로 매칭됨   → _rekey (상태 이전)
4. 사람별 FallMonitor.update(fallen and on_crosswalk, now, gap)
5. 이번 프레임에 안 보인 트랙도 update(False, ...) ★ 빼먹으면 사이렌이 안 꺼짐
6. _retire()  ttl = fall_clear_sec + gap + grace 지난 미확정 트랙 정리
   ▸ confirmed 상태는 해제될 때까지 유지
반환: confirmed인 키 집합
```

⚠️ **프레임 간격에 적응하는 이유**: 갭이 고정 1초면 프레임 간격이 1초를 넘는 보드에서
한 프레임만 깜빡여도 후보가 취소돼 쓰러짐을 **영영 확정하지 못한다.**
grace도 같은 이유로 함께 적응시켜야 한다 — 한쪽만 고정이면 `_match`가 찾으려는 트랙을
`_retire`가 미리 지우는 경합이 생긴다.

⚠️ **ID 보정이 필요한 이유**: 넘어지는 순간이 하필 트래커가 가장 약한 구간이다.
UR Fall 실측에서 낙상 프레임의 ID가 `None → 2 → None`으로 끊겼다(bbox 급변 + conf 하락).
ID를 그대로 믿으면 낙상 구간만 쏙 빠져 영영 확정되지 않는다.

#### 어댑터 (원문에 없던 배선)

| 함수 | 입력 | 출력 |
|---|---|---|
| `person_from_box(box)` | `BoundingBox` | `Person` (bbox는 int 튜플, keypoints 전달) |
| `people_from_boxes(boxes)` | `list[BoundingBox]` | `list[Person]` — 사람 클래스만 |
| `roi_from_ratio(frame_shape, ratio)` | `(H,W,...)`, 비율 사각형 | 픽셀 `(x1,y1,x2,y2)` |
| `roi_from_zones(zones)` | `CrosswalkZones` | 구역 폴리곤을 감싸는 **바운딩 박스** |

▸ `person_from_box`는 **사본을 만든다.** `FallTracker`가 `Person.track_id`를 덮어쓰므로,
사본이 아니면 목표 2의 ID 보정이 목표 1의 구역·속도 판정을 오염시킨다.
▸ `roi_from_zones`는 사선 구도의 사다리꼴을 감싸므로 실제 횡단보도보다 조금 넓다.
쓰러짐은 '놓치는 것'이 더 나쁘므로 안전한 방향의 오차다.

#### `FallDetectionPipeline.update(boxes, now)` → dict

```python
{
  "people":        [Person, ...],
  "fallen_flags":  [bool, ...],      # 자세가 쓰러짐 후보인가
  "in_roi_flags":  [bool, ...],      # 발이 ROI 안인가 (신호 연장용)
  "confirmed_ids": {키, ...},        # 사이렌 확정된 사람
  "fall_confirmed": bool,            # 하나라도 확정됐는가
}
```

---

### 3.8 `src/serial_comm.py` — 제어부 통신 (**구현 완료**)

| 메서드 | 하는 일 |
|---|---|
| `open()` | 포트 열고 `READY`(또는 `PONG`) 대기. 응답 없으면 `RuntimeError` |
| `update_state(state, 초, ETA, now)` | **변화 시 + 하트비트에만** 실제 전송 → 보낸 줄 or `None` |
| `send_state(...)` | 무조건 전송 (수동 조작·진단용) |
| `poll()` | 도착한 줄을 전부 회수해 분류 (논블로킹) |
| `ping()` | `PING` 전송 후 연결 상태 반환 |
| `close()` | `NORMAL`을 보내고 닫는다 (부저·연장이 남지 않도록) |

포트는 `SERIAL_PORT`가 `None`이면 자동 탐색한다 — 아두이노 USB VID로 거르고, **후보가
정확히 하나일 때만** 연결한다. 엉뚱한 장치를 잡으면 "열리긴 했는데 무반응"이 되어
원인을 찾기 어렵기 때문이다.

**수신을 `poll()` 한 곳으로 모은 이유**: 한 채널로 여러 종류의 줄이 섞여 온다.
"필요할 때 `readline()` 한 번" 방식이면 명령이 늘어날 때 서로의 응답을 잡아먹는다.

프로토콜 표와 아두이노가 구현해야 할 것은 `docs/team_interface.md` "아두이노 계약" 참고.

---

## 4. 확인 도구 (`tools/`)

> `tests/`는 "코드가 의도대로 도는가"를 자동 검증하고,
> `tools/`는 "현실이 가정과 맞는가"를 사람이 눈으로 확인한다. 서로를 대체할 수 없다.

| 도구 | 실행 위치 | 용도 |
|---|---|---|
| `zone_calibrator.py` | PC/파이 | 네 꼭짓점 클릭 → `data/zone_config.json`. **가장 먼저 할 일** |
| `manual_camera_person_check.py` | PC/파이 | **주력 확인 도구.** 검출·구역·속도·ETA·FPS·교통약자 |
| `pi_camera_server.py` | 파이 | 프레임을 MJPEG로 송출 (배치 B — PC에서 추론) |
| `manual_rpicam_person_check.py` | 파이 | Picamera2 경로만 확인 |

### `manual_camera_person_check.py` 화면 읽는 법

| 표시 | 정상 | 이상하면 |
|---|---|---|
| 빨간 점 | 사람 발밑 지면에 찍힘 | 박스가 잘리는 중 → 카메라 각도 |
| 박스 색 | 초록=확정, 노랑=구역 안 미확정, 흰색=구역 밖 | 초록이 깜빡이면 track_id 끊김 |
| 자주색 박스 | 교통약자 보조 모델 결과 | `--aid-model` 지정 시에만 |
| 좌상단 `unit` | `cm/s` | `px/s`면 실측 치수 미입력 |
| 좌상단 `FPS` | — | **이 값이 `ZONE_RESIDENCY_FRAMES`의 근거** |
| `aid N (xM)` | N=현재 검출, M=누적 추론 횟수 | 주기가 도는지 확인 |

```powershell
# 후보 가중치가 우리 모형을 잡는지 사진 한 장으로 검증
python tools/manual_camera_person_check.py --source 휠체어모형.jpg --aid-model 후보.pt --aid-conf 0.15

# 실시간
python tools/manual_camera_person_check.py --source picamera2 --aid-model 후보.pt --aid-every 10
```

---

## 5. 성능 — 무엇이 비싼가

실측(개발 PC 기준):

| 단계 | 비용 |
|---|---|
| 구역+호모그래피+속도+ETA+상태머신 **전부** (1명) | **25 µs** |
| 〃 (3명) | 62 µs |
| YOLO `track()` `imgsz=640` | **110.5 ms** |
| YOLO `track()` `imgsz=320` | 39.0 ms |

**비율 약 4,400 : 1.** 판단 로직 전체가 추론 한 프레임의 0.02%다.
즉 **성능 문제는 "파이에서 추론이 몇 fps인가" 하나로 수렴한다.**

느릴 때 손대는 순서:
1. `DETECTION_IMGSZ` 낮추기 — **재캘리브레이션 불필요**, 약 2.8배
2. `ncnn`/`tflite` 변환
3. `CAMERA_RESOLUTION` 낮추기 — ⚠️ **재캘리브레이션 필요**
4. 프레임 스킵 — ⚠️ `ZONE_RESIDENCY_FRAMES`의 의미가 바뀐다(프레임 수 기준이므로).
   속도는 타임스탬프 기반이라 영향 없다.

⚠️ 라즈베리파이 5 실측 FPS는 **아직 미측정**이다. 위 숫자를 파이 값으로 인용하지 말 것.

---

## 6. 아직 안 된 것

| 항목 | 상태 | 막고 있는 것 |
|---|---|---|
| 아두이노 스케치 (신호/연장) | 미구현 | 하드웨어 담당 — 계약은 `team_interface.md`에 확정 |
| `FALL` 수신 시 신호 동작 | 미결정 | 부저만? 차량 적색 유지? 팀 합의 필요 |
| 텔레그램 신고 | 미구현 | 봇 토큰 보관 방식 + 범위 미결정 |
| 교통약자 우선 연장 | 값 미정 | `PRIORITY_ZONE_EXTENSION_SEC` (검출 자체는 배선 완료) |
| `USE_SPEED_FOR_EXTENSION` | 구현됐으나 `False` | 캘리브레이션 + 속도 정확도 실측 |
| `ZONE_RESIDENCY_FRAMES` | `None` | 파이 실측 FPS (`--confirm-frames`로 임시 우회) |

자세한 내용과 담당은 `docs/team_interface.md`.
