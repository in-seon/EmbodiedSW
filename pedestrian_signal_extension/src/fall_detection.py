"""쓰러짐(이상상황) 감지 — 목표 2.

## 출처와 원칙

이 파일의 판단 로직은 팀원이 작성한 PoC(`crosswalk_poc.py`)에서 그대로 옮겨온 것이다. 
아래 "원문 그대로" 구간(Person / foot_in_roi ~ FallTracker)은
원본과 바이트 단위로 동일하며, `tests/test_fall_detection.py`가 그 동작을 고정한다.

바꾼 것은 **주변 배선뿐**이다:

  - 카메라: PoC의 FrameSource/OpenCVSource/PiCameraSource -> `src/capture.py`의 CameraCapture.
    (양쪽 다 같은 문제를 풀고 있었다 — 파이 5 Bookworm에는 레거시 카메라 스택이 없어
     CSI 카메라를 cv2로 못 연다는 것. 코드가 두 벌이면 한쪽만 고치게 되므로 하나로 합친다.)
  - 검출: PoC의 Detector -> `src/detection.py`의 PersonDetector.
    포즈 가중치(`yolov8n-pose.pt`)를 쓰면 **한 번의 추론**으로 사람 박스와 키포인트가 같이
    나오므로, 목표 1(신호 연장)과 목표 2(쓰러짐)가 추론을 공유할 수 있다. 아래 참고.
  - 신호 제어: PoC의 SignalState/SignalController는 **가져오지 않았다.** 목표 1이 이미
    구역·속도·상한을 갖춘 `src/signal_extend.py`를 쓰고 있어서 둘이 동시에 돌면 충돌한다.
    이 파일은 "쓰러짐이 확정됐는가"까지만 책임지고, 그 신호를 어떻게 쓸지는 호출부가 정한다.

## 추론 공유

`config.DETECTION_MODEL_PATH`는 이미 `yolov8n-pose.pt`라, PersonDetector가 사람 박스와 함께
키포인트를 채워 준다(`BoundingBox.keypoints`). 프레임당 추론 1회로 두 목표가 다 돌아간다.
키포인트가 없는 가중치로 되돌리면 `looks_fallen`이 bbox 가로/세로 비율 폴백으로 동작한다(원문 그대로).

## 판단 파라미터

전부 `config.FALL_CONFIG`에서 온다. 이 파일에는 값이 없다 — 튜닝은 config에서만 한다.

## ROI

원문은 화면 비율 사각형(`crosswalk_roi`)을 쓴다. 목표 1에서 이미 횡단보도 네 꼭짓점을
캘리브레이션하므로 `roi_from_zones()`로 그 결과를 대신 쓸 수 있다(권장). 둘 다 원문
함수가 기대하는 `(x1, y1, x2, y2)` 픽셀 튜플을 만들어 준다.
"""

from dataclasses import dataclass
from collections import deque

import numpy as np

from config import config


# ============================================================
# 설정 — config.FALL_CONFIG 한 곳에서만 온다
# ============================================================
# 예전에는 이 파일에도 같은 dict가 복사돼 있었고, 아래 기본값이 그 사본을 가리켰다.
# 그래서 config/config.py의 값을 아무리 고쳐도 **실행에는 전혀 반영되지 않았다**
# (에러가 나지 않으니 "왜 파라미터가 안 먹지"로만 보였다). 사본을 지우고 config를 직접 쓴다.
# docs/team_interface.md의 "파라미터 config 일원화" 항목이 약속하는 상태가 이것이다.
#
# 값을 바꾸려면 config/config.py의 FALL_CONFIG를 고칠 것. 여기에 다시 복사하지 말 것.

# ============================================================
# ↓↓↓ 여기부터 crosswalk_poc.py 원문 그대로 (수정 금지) ↓↓↓
# 원본: crosswalk_poc.py 244-250행(Person), 286-582행(로직 계층)
# ============================================================
@dataclass
class Person:
    """검출기 출력의 표준 형식. 로직 계층은 이것만 안다."""
    bbox: tuple          # (x1, y1, x2, y2) 픽셀
    conf: float
    keypoints: np.ndarray | None = None  # (17, 3) COCO 포즈 or None
    track_id: int | None = None          # 사람별 쓰러짐 누적의 키. None이면 쓰러짐 판정에서 제외


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

# ============================================================
# ↑↑↑ crosswalk_poc.py 원문 끝 ↑↑↑
# ============================================================


# ============================================================
# 여기부터는 원문에 없던 '배선' 코드 — 기존 src 모듈과 이어 붙이는 어댑터
# ============================================================
def person_from_box(box) -> Person:
    """src.detection.BoundingBox -> 위 로직이 기대하는 Person.

    원문 로직을 건드리지 않으려고 타입을 맞춰 주는 어댑터다. FallTracker.update()가
    Person.track_id를 보정된 키로 덮어쓰므로(원문 주석 참고) 원본 BoundingBox가 아니라
    이 사본이 수정된다 — 목표 1의 구역/속도 판정에 영향이 가지 않는다.
    """
    return Person(
        bbox=(int(box.x1), int(box.y1), int(box.x2), int(box.y2)),
        conf=float(box.confidence),
        keypoints=getattr(box, "keypoints", None),
        track_id=box.track_id,
    )


def people_from_boxes(boxes) -> list:
    """사람 클래스 박스만 골라 Person 목록으로 바꾼다."""
    return [person_from_box(b) for b in boxes if b.is_pedestrian()]


def roi_from_ratio(frame_shape, ratio=None):
    """원문 방식: 화면 비율 사각형 -> 픽셀 (x1, y1, x2, y2).

    frame_shape: numpy 프레임의 .shape (H, W, ...) 또는 (H, W).
    """
    ratio = ratio if ratio is not None else config.FALL_CONFIG["crosswalk_roi"]
    h, w = frame_shape[0], frame_shape[1]
    return (int(ratio[0] * w), int(ratio[1] * h), int(ratio[2] * w), int(ratio[3] * h))


def roi_from_zones(zones):
    """캘리브레이션된 CrosswalkZones -> 픽셀 ROI 사각형 (권장).

    목표 1에서 이미 찍어 둔 네 꼭짓점을 재사용한다. 눈대중 비율보다 정확하다.

    주의: 원문 로직이 축에 평행한 사각형을 전제하므로 여기서도 구역 폴리곤 전체를 감싸는
    **바운딩 박스**를 돌려준다. 사선 구도에서 횡단보도는 사다리꼴이라 이 사각형은 실제
    횡단보도보다 조금 넓다(횡단보도 밖 일부가 ROI 안으로 들어온다). 쓰러짐 감지에서는
    '놓치는 것'이 '조금 넓은 것'보다 훨씬 나쁘므로 이 방향의 오차는 안전한 쪽이다.
    """
    xs, ys = [], []
    for zone in zones.zones:
        for x, y in zone.points:
            xs.append(x)
            ys.append(y)
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


class FallDetectionPipeline:
    """검출 결과 -> 쓰러짐 확정 여부. 원문 main() 루프의 쓰러짐 부분만 떼어낸 것.

    원문 main()이 매 프레임 하던 일과 순서·인자가 같다:

        roi_flags    = [foot_in_roi(p, roi_px) for p in people]          # 신호 연장용(목표 1)
        fallen_flags = [looks_fallen(p, cfg) for p in people]
        on_cw_flags  = [roi_overlap_ratio(p, roi_px) >= cfg[...] ...]
        confirmed    = fall_tracker.update(people, fallen_flags, on_cw_flags, now)

    ROI 기준이 둘로 나뉘는 이유도 원문 그대로다 — 신호 연장은 '발 위치', 쓰러짐은
    '몸 전체 겹침'. 쓰러지면 발 위치가 크게 튀어서 한 기준으로 둘 다 처리하면 하필
    쓰러진 순간에 ROI 밖으로 빠져 사이렌을 놓친다.
    """

    def __init__(self, roi_px, cfg=None):
        # 기본값은 config.FALL_CONFIG. cfg를 직접 주는 것은 테스트에서 특정 파라미터를
        # 고정할 때만 쓴다(운영 값 튜닝이 테스트를 깨지 않게 하기 위함).
        self.cfg = cfg if cfg is not None else config.FALL_CONFIG
        self.roi_px = roi_px
        self.tracker = FallTracker(self.cfg)

    def update(self, boxes, now) -> dict:
        """이번 프레임 검출을 반영한다.

        boxes: src.detection.BoundingBox 목록 (사람 외 클래스가 섞여 있어도 걸러낸다).
        now:   판단 기준 시각(초). 라이브면 time.time(), 영상 평가면 '영상 속 시각'.
               원문과 같은 이유로 인자로 받는다 — 카메라 없이 테스트할 수 있어야 한다.

        반환: {"people": [...], "fallen_flags": [...], "in_roi_flags": [...],
               "confirmed_ids": set, "fall_confirmed": bool}
        """
        people = people_from_boxes(boxes)
        in_roi_flags = [foot_in_roi(p, self.roi_px) for p in people]
        fallen_flags = [looks_fallen(p, self.cfg) for p in people]
        on_cw_flags = [
            roi_overlap_ratio(p, self.roi_px) >= self.cfg["fall_roi_overlap"]
            for p in people
        ]
        confirmed_ids = self.tracker.update(people, fallen_flags, on_cw_flags, now)
        return {
            "people": people,
            "fallen_flags": fallen_flags,
            "in_roi_flags": in_roi_flags,
            "confirmed_ids": confirmed_ids,
            "fall_confirmed": bool(confirmed_ids),
        }

    def reset(self):
        """수동 리셋 — 오탐으로 사이렌에 갇혔을 때의 탈출구(원문 'r' 키와 같다)."""
        self.tracker.reset()
