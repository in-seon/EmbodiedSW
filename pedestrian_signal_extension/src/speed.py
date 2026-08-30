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
    
    def __init__(self, ground_plane=None, window_sec=None, min_samples=None, grace_frames=None,
                 stopped_threshold=None):

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
