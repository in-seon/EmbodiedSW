"""보행자 속도 추정 (CLAUDE.md 2.3).

bounding box 하단 모서리 중심(=발 위치)이 시간에 따라 어디로 옮겨가는지로 보행 속도를 낸다.

핵심 두 가지:

1. **평면 좌표로 바꾼 뒤에 잰다.** 사선 카메라의 픽셀 변위는 화면 위/아래 위치에 따라
   같은 실거리가 다른 픽셀 수로 나타나므로 그대로 쓰면 안 된다. GroundPlane(호모그래피)으로
   cm 좌표로 편 다음 재야 의미가 있다. GroundPlane이 없으면(치수 미실측) px/s로 떨어지고,
   그 사실을 TrackSpeed.unit 으로 명시한다 — cm인 척 변환하지 않는다.

2. **한 프레임 차분을 쓰지 않는다.** 검출 박스는 프레임마다 몇 픽셀씩 흔들리는데,
   dt가 짧으면 그 지터가 속도로 증폭된다(예: 30fps에서 3px 지터 -> 90px/s). 그래서 최근
   config.SPEED_WINDOW_SEC 구간의 양 끝 샘플로 변위를 잰다.

시각(timestamp)은 인자로 주입받는다. 실시간에서는 time.monotonic()을, 테스트에서는 원하는 값을
넣어 카메라 없이 검증할 수 있다.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional

from config import config


@dataclass(frozen=True)
class TrackSpeed:
    """한 트랙(사람 한 명)의 이번 프레임 기준 속도 추정 결과."""

    track_id: object
    speed: float            # 평면상 전체 속력 (항상 >= 0)
    crossing_speed: float   # 걷는 방향(y축) 성분의 크기 (항상 >= 0). 좌우 흔들림 제외.
    direction: int          # +1 = 끝 변 쪽, -1 = 시작 변 쪽, 0 = 정지/판별 불가
    unit: str               # "cm/s" (호모그래피 있음) 또는 "px/s" (없음)
    position: tuple         # 이번 프레임 위치. unit에 맞춰 (x_cm, y_cm) 또는 (px, py).

    @property
    def is_metric(self) -> bool:
        """실거리 단위(cm/s)인지. False면 원근 보정이 안 된 픽셀 값이라 절대 비교 불가."""
        return self.unit == "cm/s"


class SpeedEstimator:
    """트랙 ID별로 발 위치 히스토리를 유지하며 속도를 추정한다.

    트랙 ID 부여는 검출기(YOLO.track) 책임이며 여기선 받기만 한다. ID가 None인 검출은
    프레임 간 대응을 알 수 없으므로 무시한다(속도를 낼 근거가 없다).
    """

    def __init__(self, ground_plane=None, window_sec=None, min_samples=None, grace_frames=None,
                 stopped_threshold=None):
        """ground_plane: GroundPlane 또는 None.

        None이면 픽셀 좌표 그대로 계산하고 unit="px/s"로 표시한다.
        (횡단보도 실측 치수 CROSSWALK_REAL_*_CM 가 config에 없을 때의 동작.)

        grace_frames: 이만큼의 연속 미검출까지는 히스토리를 버리지 않는다
        (config.TRACK_GRACE_FRAMES). 0이면 유예 없음.
        """
        self.ground_plane = ground_plane
        self.window_sec = window_sec if window_sec is not None else config.SPEED_WINDOW_SEC
        self.min_samples = min_samples if min_samples is not None else config.SPEED_MIN_SAMPLES
        # 0은 '유예 없음'이라는 유효한 설정이므로 `or`가 아니라 None 검사를 쓴다.
        self.grace_frames = (
            grace_frames if grace_frames is not None else config.TRACK_GRACE_FRAMES
        )
        # 이 속도 미만은 '정지'로 본다. 0이면 예전처럼 0보다 크기만 하면 ETA를 낸다.
        self.stopped_threshold = (
            stopped_threshold if stopped_threshold is not None
            else config.SPEED_STOPPED_THRESHOLD_CM_S
        )
        if self.window_sec is None or self.window_sec <= 0:
            raise ValueError(f"window_sec은 양수여야 합니다: {self.window_sec}")
        if self.min_samples is None or self.min_samples < 2:
            raise ValueError(f"min_samples는 2 이상이어야 합니다(변위를 재려면 두 점이 필요): {self.min_samples}")
        if self.grace_frames is None or self.grace_frames < 0:
            raise ValueError(
                f"grace_frames는 0 이상이어야 합니다(0 = 유예 없음): {self.grace_frames}"
            )

        # track_id -> deque[(timestamp, (x, y))]
        self._history = {}
        # track_id -> TrackSpeed (가장 최근 계산 결과)
        self._latest = {}
        # track_id -> 연속 미검출 프레임 수 (유예 계산용)
        self._misses = {}

    @property
    def unit(self) -> str:
        return "cm/s" if self.ground_plane is not None else "px/s"

    def _to_measure_space(self, foot_point):
        """발 위치를 측정용 좌표로 변환. 호모그래피가 있으면 cm, 없으면 픽셀 그대로."""
        if self.ground_plane is None:
            return (float(foot_point[0]), float(foot_point[1]))
        return self.ground_plane.to_ground(foot_point)

    def update(self, track_id, foot_point, timestamp) -> Optional[TrackSpeed]:
        """한 트랙의 이번 프레임 발 위치를 반영하고 속도를 반환한다.

        아직 윈도우 안에 샘플이 min_samples만큼 쌓이지 않았으면 None.
        track_id가 None이면(추적 실패) None.
        """
        if track_id is None:
            return None

        position = self._to_measure_space(foot_point)

        history = self._history.setdefault(track_id, deque())
        history.append((float(timestamp), position))

        # 윈도우 밖으로 밀려난 오래된 샘플 제거.
        # cutoff보다 오래된 샘플을 '하나만' 남기고 버린다 -> 측정 구간이 window_sec에 가장 가깝게 맞는다.
        # (cutoff 이전 것을 전부 버리면 구간이 window_sec보다 짧아지고, 하나 더 남기면 최대 두 배로 길어진다.)
        # 최소 두 개는 항상 남긴다 — 점이 하나면 변위를 잴 수 없다.
        cutoff = float(timestamp) - self.window_sec
        while len(history) > 2 and history[1][0] <= cutoff:
            history.popleft()

        if len(history) < self.min_samples:
            return None

        t0, p0 = history[0]
        t1, p1 = history[-1]
        dt = t1 - t0
        if dt <= 0:
            # 같은 시각의 샘플이 둘 이상 — 속도를 낼 수 없다.
            return None

        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        speed = (dx * dx + dy * dy) ** 0.5 / dt
        crossing_speed = abs(dy) / dt

        # 진행 방향: y(걷는 방향) 성분의 부호. 노이즈로 인한 미세 변위는 0으로 본다.
        if dy > 0:
            direction = 1
        elif dy < 0:
            direction = -1
        else:
            direction = 0

        result = TrackSpeed(
            track_id=track_id,
            speed=speed,
            crossing_speed=crossing_speed,
            direction=direction,
            unit=self.unit,
            position=p1,
        )
        self._latest[track_id] = result
        return result

    def update_many(self, detections, timestamp) -> dict:
        """여러 트랙을 한 번에 갱신한다.

        detections: (track_id, foot_point) 튜플의 iterable.
        반환: {track_id: TrackSpeed} — 아직 속도를 못 낸 트랙은 빠진다.

        이번 프레임에 보이지 않은 트랙의 히스토리는 **유예(grace_frames)를 넘겼을 때** 지운다.
        사라졌다가 같은 ID로 다시 나타나면 그 공백을 가로질러 변위를 재게 되는데, 그러면
        실제보다 훨씬 느린 속도가 나오고(끊긴 구간을 계속 이동한 것으로 치므로), 느린 속도는
        곧 과대한 ETA -> 불필요한 연장이 된다.

        다만 한두 프레임 깜빡임까지 지우면 다시 window_sec만큼 샘플을 쌓아야 하고, 그동안
        ETA가 None이 되어 속도 기반 연장이 조용히 꺼진다. 그래서 유예 안에서는 히스토리를
        그대로 두고 직전 속도(latest)도 유지한다. CrosswalkOccupancy가 같은 규칙을 쓴다.
        """
        seen = set()
        results = {}
        for track_id, foot_point in detections:
            if track_id is None:
                continue
            seen.add(track_id)
            self._misses[track_id] = 0
            result = self.update(track_id, foot_point, timestamp)
            if result is not None:
                results[track_id] = result

        for track_id in list(self._history):
            if track_id in seen:
                continue
            misses = self._misses.get(track_id, 0) + 1
            self._misses[track_id] = misses
            if misses > self.grace_frames:
                self._history.pop(track_id, None)
                self._latest.pop(track_id, None)
                self._misses.pop(track_id, None)

        return results

    def latest(self, track_id) -> Optional[TrackSpeed]:
        """가장 최근에 계산된 속도(이번 프레임에 갱신 안 됐으면 직전 값)."""
        return self._latest.get(track_id)

    def estimated_crossing_time_sec(self, track_id) -> Optional[float]:
        """진행 방향 기준으로 횡단보도를 다 건너기까지 남은 예상 시간(초).

        다음 경우에 None을 반환한다.
          - 호모그래피가 없음 (px 단위라 실시간으로 환산 불가)
          - 아직 속도가 계산되지 않음
          - 진행 방향을 판별할 수 없음(정지)
          - crossing_speed가 config.SPEED_STOPPED_THRESHOLD_CM_S 미만 (사실상 정지)

        마지막 조건이 중요하다. `> 0` 으로만 거르면 서 있는 사람의 부동소수점 오차가
        거대한 ETA로 증폭된다(실측 관측: ETA=10351225.3s). 그 값이 아두이노로 나가면
        엉뚱한 부족분이 계산되므로, 걷는다고 보기 어려운 속도는 아예 ETA를 내지 않는다.
        """
        if self.ground_plane is None:
            return None
        result = self._latest.get(track_id)
        if result is None or result.direction == 0:
            return None
        if result.crossing_speed <= max(self.stopped_threshold, 0.0):
            return None
        remaining = self.ground_plane.remaining_distance_cm(result.position, result.direction)
        if remaining is None:
            return None
        return remaining / result.crossing_speed

    def clear(self):
        self._history.clear()
        self._latest.clear()
        self._misses.clear()
