# 버그 수정 기록

`docs/decisions.md`가 "왜 이 파라미터 값인가"를 남기는 문서라면, 이 문서는
**"무엇이 잘못돼 있었고 어떻게 고쳤는가"**를 남긴다.
시간순 작업 흐름은 `docs/log.md` 참고. 새 항목은 아래 형식으로 위에 추가한다.

## 템플릿

### YYYY-MM-DD: 제목

- **증상**:
- **원인**:
- **왜 위험한가**:
- **수정**:
- **회귀 테스트**:
- **남은 과제**:

---

## 2026-08-19: 하드웨어 결합 전 코드 점검에서 발견한 4건 (A~D)

하드웨어가 완성되지 않아 실행 검증을 할 수 없는 상태에서, 코드 리뷰로 실시간 루프 결함
4건을 찾아 수정했다. **네 건 모두 각 모듈은 정상인데 모듈이 맞물리는 이음매에서 생긴 문제**다.
기존 테스트 83개가 전부 통과하고 있었는데도 잡히지 않은 이유가 이것이다 — 각 모듈을
단독으로는 검증했지만, "실시간 루프가 매 프레임 호출한다"는 사실을 반영한 테스트가 없었다.

수정 후 테스트: **83개 → 99개, 전부 통과.**

---

### A. 상태 머신이 프레임마다 연장을 누적했다 🔴

- **증상**: 보행자가 3번 구역에 서 있으면 프레임마다 연장이 발급됐다. 10fps 기준 재현 결과:

  ```
  프레임별 연장값: [5, 5, 5, 0, 0, 0, 0, 0, 0, 0]   → 0.3초 만에 상한(15초) 소진
  ```

  0.3초 만에 사이클의 연장 예산을 전부 쓰고, 그동안 제어부로 `send_extend_signal(5)`가
  **3번 따로 전송**됐다.

- **원인**: `SignalExtensionPipeline.process_frame()`은 **매 프레임** 호출되는데,
  `SignalExtensionStateMachine.evaluate()`는 호출될 때마다 조건이 참이면
  `accumulated_extension_sec += step`을 했다. "매 사이클 재평가"라는 설계 의도가
  프레임 루프에 그대로 물리면서 "매 프레임 재연장"이 돼 있었다.

- **왜 위험한가**: 제어부가 "5초 연장" 명령을 연타로 받는다. 제어부가 명령을 누적 처리하면
  의도의 3배로 연장되고, 마지막 값만 반영하면 상한만 헛되이 소진된다. 어느 쪽이든
  **차량 신호와의 충돌 방지를 위해 둔 상한(`MAX_TOTAL_EXTENSION_SEC`)이 안전장치로
  기능하지 못한다.**

- **수정** (`src/signal_extend.py`): **엣지 트리거**로 바꿨다. `_armed` 플래그를 두고,
  "잔여시간이 임계값 아래로 내려온 구간"당 연장을 한 번만 발급한다.

  - 잔여시간 > 임계값 → `_armed = True` (재무장), `NORMAL`
  - 잔여시간 ≤ 임계값이고 `_armed` → 연장 발급, `_armed = False`, `EXTENDING`
  - 잔여시간 ≤ 임계값이고 `not _armed` → 0 반환, 새 상태 `EXTENDED` (제어부 반영 대기)

  **무장은 "하강"이 아니라 "연장 발급"으로만 소모된다.** 임계값 아래로 내려간 뒤에
  뒤늦게 보행자가 들어오는 경우를 놓치지 않기 위해서다.

  연장이 제어부에 반영돼 잔여시간이 임계값 위로 회복되면 자동으로 재무장되므로,
  **느린 보행자가 한 사이클에서 두 번 연장받는 경로는 그대로 살아 있다.** 그 총합을
  막는 것이 상한의 역할이다.

- **회귀 테스트**:
  - `tests/test_signal_extend.py::test_extends_only_once_per_descent_below_threshold`
  - `::test_rearms_after_remaining_time_recovers`
  - `::test_arming_is_consumed_by_granting_not_by_descending`
  - `tests/test_pipeline.py::test_does_not_resend_extension_every_frame`
  - `::test_extends_again_after_controller_applies_extension`

- **기존 테스트 수정**: `tests/test_signal_extend.py::test_respects_upper_cap`이
  **버그 동작을 그대로 인코딩**하고 있었다(`evaluate()`를 연속 두 번 불러 상한에 도달).
  상한이 "여러 하강 구간에 걸친 총합"을 막는다는 올바른 의미로 고쳤다 — 중간에 잔여시간이
  회복돼 재무장되는 과정을 거치게 했다.

  > 교훈: 테스트가 통과한다고 동작이 옳은 것은 아니다. 이 테스트는 잘못된 동작을
  > 충실히 지켜주고 있었다.

---

### B. `reset()`을 아무도 호출하지 않았다 🔴

- **증상**: 첫 보행 신호 사이클에서 누적 연장이 상한에 도달하면, **그 뒤 모든 사이클에서
  영구히 `CAPPED`** 상태가 되어 연장이 아예 동작하지 않는다.

- **원인**: `SignalExtensionStateMachine.reset()`은 정의돼 있었지만 `src/` 어디에서도
  호출되지 않았다(`detector.reset_tracker()`도 마찬가지). 사이클 경계를 알려주는 입력이
  파이프라인에 아예 없었다. 잔류 카운트(`CrosswalkOccupancy`)와 속도 히스토리
  (`SpeedEstimator`)도 같은 이유로 사이클을 넘어 살아남았다.

- **왜 위험한가**: A와 결합하면 특히 나쁘다 — A 때문에 첫 사이클에서 0.3초 만에 상한을
  찍고, B 때문에 그 상태가 영구히 유지된다. **시연 중 첫 번째 보행자 이후로 시스템 전체가
  조용히 죽는다.** 예외도 로그도 없이 "연장이 안 되는 상태"가 된다.

  잔류 카운트가 남는 것도 문제다. 사이클이 바뀌면 보행자도 실제로 바뀌는데, 이전 카운트가
  남아 있으면 새 사이클 첫 프레임에서 곧바로 "확정 보행자"가 되어 `ZONE_RESIDENCY_FRAMES`
  검증이 무의미해진다.

- **수정**:
  - `SignalExtensionPipeline.begin_new_cycle()` 신설 — 누적 연장 + 잔류 카운트 +
    속도 히스토리를 한 번에 초기화한다.
  - `SignalExtensionPipeline.run()`이 매 프레임 `serial_comm.read_cycle_started()`를
    확인해 새 사이클이면 호출한다.
  - `SerialComm.read_cycle_started()` 신설 — 다른 미확정 항목과 같은 방식으로
    `NotImplementedError`를 던져 **"이 프로토콜 항목이 빠져 있다"는 사실을 드러낸다.**

  추적 ID(`track_id`)는 일부러 초기화하지 않았다. 위 세 가지를 지우면 ID가 재사용돼도
  카운트와 히스토리가 처음부터 다시 쌓이므로 문제가 없고, 추적기 리셋은 비용만 든다.

- **회귀 테스트**:
  - `tests/test_pipeline.py::test_begin_new_cycle_resets_extension_budget`
  - `::test_begin_new_cycle_clears_residency_and_speed_state`
  - `tests/test_signal_extend.py::test_new_cycle_allows_extension_after_cap`

- **남은 과제**: 🚩 **팀 합의 필요.** 사이클 시작 이벤트는 코드만으로 해결할 수 없다.
  신호 사이클의 소유자가 제어부(ESP32)이므로 **제어부가 "녹색 시작"을 파이에 알려주는
  메시지가 시리얼 프로토콜에 있어야 한다.** `docs/team_interface.md`에 항목을 추가했다.

  > 파이가 잔여 시간의 증감만 보고 추측하는 방식은 채택하지 않았다. "시간이 갑자기 늘었다"가
  > 새 사이클인지 우리가 방금 요청한 연장이 반영된 것인지 구분할 수 없기 때문이다.
  > 근거 없이 추측하느니 미확정임을 드러내는 편이 낫다(CLAUDE.md 5장 원칙).

---

### C. 캘리브레이션 해상도를 저장만 하고 검증하지 않았다 🟡

- **증상**: 캘리브레이션 당시 해상도와 운영 해상도가 다르면 **에러 없이 구역만 엉뚱하게
  잡힌다.** 사람이 3번 구역에 있는데 1번으로 판정되는 식.

- **원인**: `tools/zone_calibrator.py`는 `frame_size`를 `data/zone_config.json`에
  성실히 저장하고 있었는데, `CrosswalkZones.load()`가 그 값을 **읽지도 비교하지도 않았다.**
  zone 좌표와 호모그래피는 둘 다 캘리브레이션 당시 해상도에 종속된 픽셀 값이다.

- **왜 위험한가**: **조용히 틀리기 때문이다.** 예외도 경고도 없이 판정만 어긋나므로,
  현장에서 "왜 연장이 이상하게 되지"의 원인을 추적하기가 가장 어려운 유형이다.
  CLAUDE.md가 세 군데서 경고하는 시나리오인데 정작 감지 장치가 없었다.

  실제로 발생하기 쉽다 — 배치 A(파이 직접 실행)와 배치 B(PC 추론)를 오가거나,
  파이 FPS가 부족해 `CAMERA_RESOLUTION`을 낮추는 순간 바로 이 상황이 된다.

- **수정** (`src/zone.py`): `CrosswalkZones.load(path, expected_frame_size=None)`에
  검증을 추가했다. `expected_frame_size`를 생략하면 `config.CAMERA_RESOLUTION`과 비교하고,
  다르면 **즉시 `ValueError`**를 낸다. 메시지에 두 해상도와 해결 방법(설정을 되돌리거나
  재캘리브레이션)을 함께 넣었다.

  `frame_size` 키가 없는 옛 설정 파일은 검증할 근거가 없으므로 통과시킨다.

- **회귀 테스트**:
  - `tests/test_zone.py::test_load_rejects_frame_size_mismatch`
  - `::test_load_accepts_matching_frame_size`
  - `::test_load_allows_config_without_frame_size`
  - `::test_load_uses_config_resolution_by_default`

---

### D. `track_id=None`을 speed와 occupancy가 다르게 다뤘다 🟡

- **증상**: 추적 ID가 붙지 않은 검출들이 **하나로 합쳐져**, 같은 프레임 안의 여러 사람이
  같은 카운터를 각각 증가시켰다. 재현 결과:

  ```
  한 프레임에 ID 없는 3명 → confirm_frames=3인데 즉시 확정: {None: 4}
  ```

  한 프레임 만에 3프레임치 잔류를 채웠고, 구역은 마지막 사람 것만 남았다.

- **원인**: `SpeedEstimator.update_many()`는 `track_id is None`이면 무시하는데,
  `CrosswalkOccupancy.update()`는 `None`을 그대로 딕셔너리 키로 썼다. 같은 파이프라인의
  두 소비자가 같은 입력을 다르게 해석하고 있었다.

- **왜 위험한가**: 검출 흔들림을 걸러내려고 둔 `ZONE_RESIDENCY_FRAMES`가 **정확히 추적이
  불안정한 순간에** 무력화된다. 안정적일 때는 잘 동작하고 흔들릴 때 망가지는, 방어 장치로서
  최악의 실패 방식이다.

- **수정** (`src/zone.py`): `CrosswalkOccupancy.update()`가 `track_id is None`인 검출을
  건너뛴다 — `SpeedEstimator`와 같은 규칙. 프레임 간 대응을 알 수 없는 검출로는
  "연속 몇 프레임 잔류했는가"를 셀 수 없기 때문이다.

  다만 **조용히 버리지 않는다.** 무시한 개수를 `CrosswalkOccupancy.untracked_count`와
  `FrameResult.untracked_count`로 노출해, 실측에서 "추적이 자주 끊기고 있다"는 사실이
  보이게 했다. 이 값이 계속 0이 아니면 그만큼 연장 조건을 놓치고 있다는 뜻이므로,
  `DETECTION_TRACKER`를 `botsort.yaml`로 바꾸거나 해상도/FPS를 조정할 근거가 된다.

- **회귀 테스트**:
  - `tests/test_zone.py::test_occupancy_ignores_untracked_detections`
  - `::test_occupancy_reports_untracked_count`
  - `::test_untracked_detection_does_not_disturb_tracked_one`
  - `tests/test_pipeline.py::test_frame_result_exposes_untracked_count`

---

### 이번 수정으로 바뀐 공개 인터페이스

하드웨어/제어부 담당자가 알아야 할 변경:

| 항목 | 변경 |
|---|---|
| `SignalState` | `EXTENDED` 추가 — "이번 하강 구간에서 이미 발급, 제어부 반영 대기" |
| `SignalExtensionPipeline` | `begin_new_cycle()` 추가 |
| `SerialComm` | `read_cycle_started()` 추가 (**프로토콜 합의 필요**) |
| `CrosswalkZones.load()` | `expected_frame_size` 인자 추가, 해상도 불일치 시 `ValueError` |
| `CrosswalkOccupancy` | `untracked_count` 속성 추가, `track_id=None` 검출 무시 |
| `FrameResult` | `untracked_count` 필드 추가 |

### 아직 안 고친 것 (별건)

리뷰에서 함께 발견했지만 이번 수정 범위(A~D) 밖이라 남겨 둔 항목:

- `CameraCapture.frames()`가 프레임 하나만 `None`이어도 루프를 끝낸다. 영상 파일에는
  맞는 동작이지만 라이브 카메라에서는 일시적 실패로 프로그램이 조용히 종료된다.
- `GroundPlane.to_ground()`가 소실선 근처 점에 `ValueError`를 던지는데,
  `pipeline.run()`에 예외 처리가 없어 멀리 있는 사람 한 명에 루프 전체가 죽을 수 있다.
  (횡단보도 밖 검출까지 `to_ground()`에 들어가는 것이 직접 원인)
- `pipeline.process_frame()`이 `zones.locate()`를 사람당 두 번 호출한다(성능, 경미).

이 세 건은 실시간 루프의 **견고성** 문제로, 하드웨어에 처음 붙여 `main.py`를 돌리기
직전에 함께 처리하는 것이 좋다.
