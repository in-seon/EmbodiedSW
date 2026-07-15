"""Zone(횡단보도 ROI) 정의 및 잔류 판단.

CLAUDE.md 2.2: 탑뷰 각도상 좌표->실거리 변환이 어려우므로, 정밀 위치/속도 추정 대신
zone 내 연속 검출 프레임 카운트 기반으로 잔류를 판단한다.
"""

import json

from config import config


class Zone:
    """폴리곤(또는 사각형) ROI. 좌표는 프레임 픽셀 좌표계."""

    def __init__(self, points, name="crosswalk"):
        if len(points) < 3:
            raise ValueError("Zone은 최소 3개의 점으로 이루어진 폴리곤이어야 합니다.")
        self.points = points
        self.name = name

    def contains(self, point) -> bool:
        """point가 zone 내부에 있는지 판정 (ray casting)."""
        x, y = point
        n = len(self.points)
        inside = False
        x1, y1 = self.points[0]
        for i in range(1, n + 1):
            x2, y2 = self.points[i % n]
            if y > min(y1, y2) and y <= max(y1, y2) and x <= max(x1, x2):
                if y1 != y2:
                    x_intersect = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                if x1 == x2 or x <= x_intersect:
                    inside = not inside
            x1, y1 = x2, y2
        return inside

    @classmethod
    def load(cls, path=None):
        path = path or config.ZONE_CONFIG_PATH
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(points=data["points"], name=data.get("name", "crosswalk"))


class ZoneResidencyTracker:
    """트랙(사람 ID)별로 zone 내 연속 검출 프레임 수를 세어 잔류 여부를 판단한다.

    트랙 ID 부여 로직(추적/재식별)은 이 모듈의 책임이 아니며, 검출 파이프라인에서
    제공된다고 가정한다. 트랙 ID가 아직 없다면 단순 프레임 카운트만으로 근사할 수 있다.
    """

    def __init__(self, zone: Zone, required_frames=None):
        self.zone = zone
        self.required_frames = required_frames or config.ZONE_RESIDENCY_FRAMES
        if self.required_frames is None:
            raise NotImplementedError(
                "ZONE_RESIDENCY_FRAMES가 설정되지 않았습니다. 실측 FPS 기반으로 값을 정한 뒤 사용하세요."
            )
        self._counts = {}

    def update(self, track_id, point) -> bool:
        """이번 프레임 검출 결과를 반영하고, 현재 프레임 기준 잔류 여부를 반환한다."""
        if self.zone.contains(point):
            self._counts[track_id] = self._counts.get(track_id, 0) + 1
        else:
            self._counts.pop(track_id, None)
        return self.is_resident(track_id)

    def is_resident(self, track_id) -> bool:
        return self._counts.get(track_id, 0) >= self.required_frames

    def clear(self, track_id):
        self._counts.pop(track_id, None)

    def has_any_resident(self) -> bool:
        return any(count >= self.required_frames for count in self._counts.values())
