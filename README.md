# 스마트 신호등 — 보행자 잔류 감지 기반 신호 연장

임베디드 SW 경진대회 "스마트 신호등" 프로젝트 중 담당 파트(목표 1)의 구현체.
전체 배경, 하드웨어 구성, 판단 로직 설계는 [CLAUDE.md](CLAUDE.md)를 참고.

## 카메라 구도

카메라는 **보행자 신호등에 부착되어 횡단보도를 사선으로 비춘다** (탑뷰 아님).
사람의 전신이 대체로 보이므로, 위치 판정 기준점은 bounding box **중심점이 아니라
하단 모서리의 중심(발 위치)**이다. 중심점은 사람 키만큼 떠 있어 사선 각도에서는
실제 서 있는 지점보다 카메라 쪽으로 당겨진 위치로 잘못 판정된다.

## 핵심 아이디어 1: 5구역 위치 기반 연장

횡단보도를 걷는 방향으로 위치 순서대로 **5개 구역**으로 나눈다.

```
진입 끝 ┃ 1 │ 2 │ 3 │ 4 │ 5 ┃ 반대 끝
        └───┴───┴───┴───┴───┘
```

남은 신호 시간이 임계값 이하일 때, 보행자가 **어느 구역에 있는지**로 연장 시간을 차등한다.
가운데에 있을수록 아직 더 건너야 하므로 더 길게 연장한다.

- 3번(정중앙): 가장 길게 연장
- 2·4번(중앙-끝 사이): 중간 정도 연장
- 1·5번(양 끝): 연장 안 함

구체적인 초 단위 값은 아직 임의값이라 `config/config.py`의 `ZONE_EXTENSION_SEC`에 `None`으로 두었다(2·3·4번). 실측/팀 확정 후 채운다.

## 핵심 아이디어 2: 호모그래피 기반 속도 추정

사선 구도에서는 화면상 1픽셀이 나타내는 실거리가 위치마다 다르다(가까운 쪽은 촘촘, 먼 쪽은 성김).
이 상태로 픽셀 변위를 속도라고 부르면 같은 속도로 걸어도 화면 위/아래에서 값이 달라진다.

보행자의 발은 지면(평면) 위에 있으므로 **평면 하나에 대한 사영변환(호모그래피)**이면 원근을 펼 수 있다.
필요한 건 이미 캘리브레이션에서 찍는 **네 꼭짓점 + 횡단보도 모형의 실제 치수(cm)**뿐이고,
**체스보드 카메라 캘리브레이션은 필요 없다.**

```
픽셀 좌표 --(호모그래피)--> 평면 좌표(cm)  →  최근 0.5초 변위 ÷ 경과 시간  →  cm/s
                                              → 남은 거리 ÷ 속도 = 예상 통과 시간
```

치수를 입력하지 않으면 속도는 `px/s`로 떨어지고 예상 통과 시간은 `None`이 된다 —
원근 보정되지 않은 값을 실거리인 척 변환하지 않는다.

> 속도는 현재 **계산·노출까지만** 하고 연장 시간 결정에는 쓰지 않는다
> (`config.USE_SPEED_FOR_EXTENSION = False`). 실측 검증 후 반영 여부를 결정한다.

## 폴더 구조

```
config/config.py     전역 설정 (미확정 값은 None + 주석)
main.py              실행 진입점
src/
├── capture.py       카메라 입력. Picamera2(파이 CSI) / cv2(웹캠·영상·스트림) 백엔드 자동 선택
├── detection.py     YOLO 검출 래퍼 (yolov8n, person + track_id)
├── ground_plane.py  호모그래피: 픽셀 좌표 → 횡단보도 평면 좌표(cm)
├── zone.py          Zone, 5구역 분할(CrosswalkZones), 점유 추적(CrosswalkOccupancy)
├── speed.py         트랙별 속도/진행 방향/예상 통과 시간 추정
├── signal_extend.py 위치 기반 신호 연장 상태 머신
├── serial_comm.py   라즈베리파이-제어부 시리얼 통신
└── pipeline.py      전체 흐름을 잇는 파이프라인
arduino/             아두이노 펌웨어
tools/               사람이 직접 실행하는 세팅·확인 도구 (자동 테스트 아님)
├── zone_calibrator.py             [PC]   네 꼭짓점 클릭 → zone_config.json 생성
├── manual_camera_person_check.py  [PC]   검출+구역+속도+ETA+FPS 표시 (주력 확인 도구)
├── pi_camera_server.py            [파이] 프레임을 MJPEG로 송출 (배치 B)
└── manual_rpicam_person_check.py  [파이] CSI 카메라가 cv2로 안 잡힐 때 Picamera2 확인
tests/               자동 단위 테스트 (pytest, 하드웨어 불필요)
data/                zone 설정, 테스트 영상 (git 추적 제외)
docs/                설계 결정 로그, 팀 인터페이스 합의 사항
```

```
카메라 프레임
  → YOLO 검출 (yolov8n, person, track_id 부여)
  → foot_point (bbox 하단 모서리 중심 = 지면 접점)
  ├→ 5구역 중 어디인지 판정 (CrosswalkZones) → 점유/확정 추적 (CrosswalkOccupancy)
  └→ 평면 좌표 변환 (GroundPlane) → 속도/방향/예상 통과 시간 (SpeedEstimator)
  → 위치별 연장 시간 결정 (SignalExtensionStateMachine)
  → 제어부에 연장 명령 전송 (SerialComm)
```

## 현재 상태

**비전 파트(검출 → 위치 판정 → 속도 추정)는 동작한다.** 검출 모델은 COCO 사전학습
`yolov8n.pt`로 확정했다 — 사람 모형이 신뢰도 80%대로 검출돼 추가 학습이 필요 없었다.

아직 확정되지 않은 것은 **연장 파라미터**(구역별 초, 임계값, 상한), **횡단보도 실측 치수**,
**시리얼 프로토콜**이다. 값이 없는 상태로 관련 모듈을 실행하면 `NotImplementedError`가
발생하도록 만들어, 미확정 값을 임의로 추정해 쓰지 않게 했다. 목록은
[docs/team_interface.md](docs/team_interface.md) 참고.

교통약자(휠체어·목발) 우선 연장은 **보류** — COCO에 해당 클래스가 없어 검출 수단이 없다.
`MOBILITY_AID_LABELS = ()` 로 두어 우선 연장 모드가 켜지지 않는다. 코드 인터페이스는 남겨뒀다.

## 설치 / 실행 / 테스트

이 프로젝트는 **`.venv`(Python 3.12, uv로 생성)** 를 쓴다. VSCode에서
`Ctrl+Shift+P` → `Python: Select Interpreter` → `.\.venv\Scripts\python.exe` 를 선택할 것.

```bash
uv pip install -r requirements.txt   # .venv 활성화 상태에서
pytest                               # 단위 테스트 (카메라·모델 없이 전부 통과)

# 1) 카메라를 최종 위치에 고정한 뒤 캘리브레이션
python tools/zone_calibrator.py --source 0 --width-cm 90 --length-cm 300

# 2) 검출 + 구역 + 속도를 눈으로 확인 (시리얼 불필요)
python tools/manual_camera_person_check.py

# 3) (연장 파라미터·시리얼 확정 후) 전체 실시간 루프
python main.py
```

> uv로 만든 venv에는 `pip`이 없다. `python -m pip list`가 빈 결과를 내도 패키지가 없는 게 아니니
> `uv pip list`로 확인할 것. 시스템에 별도로 깔린 Python 3.13과 헷갈리지 않게 주의.

## 라즈베리파이 연결

두 가지 배치 중 하나를 고른다. 자세한 설명은 [CLAUDE.md](CLAUDE.md) 7장 참고.

**하드웨어: 라즈베리파이 5 + CSI 리본 카메라.** 파이 5(Bookworm)에는 레거시 카메라 스택이
없어 **CSI 카메라가 `cv2.VideoCapture`로 열리지 않는다.** `config.CAMERA_SOURCE = "picamera2"`
로 두면 `CameraCapture`가 Picamera2(libcamera) 경로를 탄다.

```bash
sudo apt install -y python3-picamera2     # 파이에서 1회
```

- **A. 파이에서 직접 실행**: 파이에서 `main.py` 실행. `config.CAMERA_SOURCE = "picamera2"`.
  (VSCode Remote-SSH로 파이에 접속해 실행 가능.)
- **B. 카메라는 파이, 추론은 PC**: 파이에서 `python tools/pi_camera_server.py --source picamera2`,
  PC의 `config.CAMERA_SOURCE = "http://<파이IP>:8000/"` 로 설정 후 PC에서 실행.

## Zone 캘리브레이션

횡단보도 네 꼭짓점을 순서대로(시작-왼쪽 → 시작-오른쪽 → 끝-오른쪽 → 끝-왼쪽) 클릭하면
5구역으로 자동 분할해 저장한다. `--width-cm` / `--length-cm` 로 횡단보도 모형의 실제 치수를
함께 주면 호모그래피까지 만들어져 속도를 cm/s로 낼 수 있다.

```bash
python tools/zone_calibrator.py --source 0 --width-cm 90 --length-cm 300
```

- `width` = 걷는 방향과 **수직**인 변의 길이, `length` = 걷는 방향 변의 길이
- 치수를 생략하면 구역 판정만 되고 속도는 px/s로만 나온다
- **카메라를 최종 위치에 고정한 뒤** 찍을 것. 카메라를 옮기거나 프레임 해상도를 바꾸면
  zone 좌표와 호모그래피가 모두 무효가 되어 다시 캘리브레이션해야 한다
