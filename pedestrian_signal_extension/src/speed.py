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

    def __init__(self, ground_plane=None, window_sec=None, min_samples=None):
        """ground_plane: GroundPlane 또는 None.

        None이면 픽셀 좌표 그대로 계산하고 unit="px/s"로 표시한다.
        (횡단보도 실측 치수 CROSSWALK_REAL_*_CM 가 config에 없을 때의 동작.)
        """
        self.ground_plane = ground_plane
        self.window_sec = window_sec if window_sec is not None else config.SPEED_WINDOW_SEC
        self.min_samples = min_samples if min_samples is not None else config.SPEED_MIN_SAMPLES
        if self.window_sec is None or self.window_sec <= 0:
            raise ValueError(f"window_sec은 양수여야 합니다: {self.window_sec}")
        if self.min_samples is None or self.min_samples < 2:
            raise ValueError(f"min_samples는 2 이상이어야 합니다(변위를 재려면 두 점이 필요): {self.min_samples}")

        # track_id -> deque[(timestamp, (x, y))]
        self._history = {}
        # track_id -> TrackSpeed (가장 최근 계산 결과)
        self._latest = {}

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

        이번 프레임에 보이지 않은 트랙의 히스토리는 지운다. 사라졌다가 같은 ID로 다시
        나타나면 그 공백을 가로질러 변위를 재게 되는데, 그러면 실제보다 훨씬 느린 속도가
        나온다(먼 거리를 긴 시간으로 나눈 셈이 아니라, 끊긴 구간을 이동한 것으로 치므로).
        CrosswalkOccupancy도 같은 이유로 리셋한다.
        """
        seen = set()
        results = {}
        for track_id, foot_point in detections:
            if track_id is None:
                continue
            seen.add(track_id)
            result = self.update(track_id, foot_point, timestamp)
            if result is not None:
                results[track_id] = result

        for track_id in list(self._history):
            if track_id not in seen:
                self._history.pop(track_id, None)
                self._latest.pop(track_id, None)

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
          - crossing_speed가 0 (걷는 방향으로는 안 움직이는 중)
        """
        if self.ground_plane is None:
            return None
        result = self._latest.get(track_id)
        if result is None or result.direction == 0 or result.crossing_speed <= 0:
            return None
        remaining = self.ground_plane.remaining_distance_cm(result.position, result.direction)
        if remaining is None:
            return None
        return remaining / result.crossing_speed

    def clear(self):
        self._history.clear()
        self._latest.clear()
