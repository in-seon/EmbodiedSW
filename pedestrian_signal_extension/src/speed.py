from collections import deque
from dataclasses import dataclass
from typing import Optional

from config import config


@dataclass(frozen=True)
class TrackSpeed:

    track_id: object
    speed: float           
    crossing_speed: float  
    direction: int          
    unit: str            
    position: tuple         
    @property
    def is_metric(self) -> bool:
        return self.unit == "cm/s"


class SpeedEstimator:
    
    def __init__(self, ground_plane=None, window_sec=None, min_samples=None, grace_frames=None,
                 stopped_threshold=None):

        self.ground_plane = ground_plane
        self.window_sec = window_sec if window_sec is not None else config.SPEED_WINDOW_SEC
        self.min_samples = min_samples if min_samples is not None else config.SPEED_MIN_SAMPLES
        self.grace_frames = (
            grace_frames if grace_frames is not None else config.TRACK_GRACE_FRAMES
        )
        self.stopped_threshold = (
            stopped_threshold if stopped_threshold is not None
            else config.SPEED_STOPPED_THRESHOLD_CM_S
        )
        if self.window_sec is None or self.window_sec <= 0:
            raise ValueError(f"window_sec은 양수여야 합니다: {self.window_sec}")
        if self.min_samples is None or self.min_samples < 2:
            raise ValueError(f"min_samples는 2 이상이어야 합니다: {self.min_samples}")
        if self.grace_frames is None or self.grace_frames < 0:
            raise ValueError(
                f"grace_frames는 0 이상이어야 합니다: {self.grace_frames}"
            )
        
        self._history = {}
        self._latest = {}
        self._misses = {}

    @property
    def unit(self) -> str:
        return "cm/s" if self.ground_plane is not None else "px/s"

    def _to_measure_space(self, foot_point):
        if self.ground_plane is None:
            return (float(foot_point[0]), float(foot_point[1]))
        return self.ground_plane.to_ground(foot_point)

    def update(self, track_id, foot_point, timestamp) -> Optional[TrackSpeed]:

        if track_id is None:
            return None

        position = self._to_measure_space(foot_point)

        history = self._history.setdefault(track_id, deque())
        history.append((float(timestamp), position))
        cutoff = float(timestamp) - self.window_sec
        while len(history) > 2 and history[1][0] <= cutoff:
            history.popleft()

        if len(history) < self.min_samples:
            return None

        t0, p0 = history[0]
        t1, p1 = history[-1]
        dt = t1 - t0
        if dt <= 0:
            return None

        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        speed = (dx * dx + dy * dy) ** 0.5 / dt
        crossing_speed = abs(dy) / dt

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
