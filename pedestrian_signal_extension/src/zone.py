import json

from config import config
from src.ground_plane import GroundPlane


class Zone:

    def __init__(self, points, name="crosswalk"):
        if len(points) < 3:
            raise ValueError("Zone은 최소 3개의 점으로 이루어진 폴리곤이어야 합니다.")
        self.points = [(float(x), float(y)) for x, y in points]
        self.name = name

    def contains(self, point) -> bool:
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


class CrosswalkZones:

    def __init__(self, zones, name="crosswalk", ground_plane=None):
        self.zones = list(zones)
        self.name = name
        self.ground_plane = ground_plane

    @staticmethod
    def _lerp(p, q, t):
        return (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)

    @classmethod
    def _split_on_ground(cls, ground_plane, n, name):
        zones = []
        step = ground_plane.length_cm / n
        width = ground_plane.width_cm
        for k in range(n):
            y0, y1 = k * step, (k + 1) * step
            l0 = ground_plane.to_pixel((0.0, y0))
            r0 = ground_plane.to_pixel((width, y0))
            r1 = ground_plane.to_pixel((width, y1))
            l1 = ground_plane.to_pixel((0.0, y1))
            zones.append(Zone([l0, r0, r1, l1], name=f"{name}_{k + 1}"))
        return zones

    @classmethod
    def _split_on_pixels(cls, corners, n, name):
        start_left, start_right, end_right, end_left = corners
        zones = []
        for k in range(n):
            t0, t1 = k / n, (k + 1) / n
            l0 = cls._lerp(start_left, end_left, t0)
            r0 = cls._lerp(start_right, end_right, t0)
            l1 = cls._lerp(start_left, end_left, t1)
            r1 = cls._lerp(start_right, end_right, t1)
            zones.append(Zone([l0, r0, r1, l1], name=f"{name}_{k + 1}"))
        return zones

    @classmethod
    def from_quad(cls, corners, n=None, name="crosswalk", width_cm=None, length_cm=None):

        n = n or config.CROSSWALK_ZONE_COUNT
        if len(corners) != 4:
            raise ValueError("corners는 [시작-왼쪽, 시작-오른쪽, 끝-오른쪽, 끝-왼쪽] 4개 점이어야 합니다.")

        ground_plane = GroundPlane.from_quad(corners, width_cm, length_cm)
        if ground_plane is not None:
            zones = cls._split_on_ground(ground_plane, n, name)
        else:
            zones = cls._split_on_pixels(corners, n, name)
        return cls(zones, name=name, ground_plane=ground_plane)

    @staticmethod
    def _check_frame_size(data, expected_frame_size):
        saved = data.get("frame_size")
        if saved is None or expected_frame_size is None:
            return
        saved = (int(saved[0]), int(saved[1]))
        expected = (int(expected_frame_size[0]), int(expected_frame_size[1]))
        if saved != expected:
            raise ValueError(
                f"zone 설정의 캘리브레이션 해상도({saved[0]}x{saved[1]})가 "
                f"현재 운영 해상도({expected[0]}x{expected[1]})와 다릅니다. ")

    @classmethod
    def load(cls, path=None, expected_frame_size=None):
      
        path = path or config.ZONE_CONFIG_PATH
        if expected_frame_size is None:
            expected_frame_size = config.CAMERA_RESOLUTION
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cls._check_frame_size(data, expected_frame_size)
        name = data.get("name", "crosswalk")
        if "corners" in data:
            width_cm = data.get("real_width_cm", config.CROSSWALK_REAL_WIDTH_CM)
            length_cm = data.get("real_length_cm", config.CROSSWALK_REAL_LENGTH_CM)
            return cls.from_quad(
                data["corners"], n=data.get("n_zones"), name=name,
                width_cm=width_cm, length_cm=length_cm,
            )
        if "zones" in data:
            zones = [Zone(pts, name=f"{name}_{i + 1}") for i, pts in enumerate(data["zones"])]
            return cls(zones, name=name)
        raise ValueError("zone 설정에 'corners' 또는 'zones' 키가 필요합니다.")

    def locate(self, point):
        for i, zone in enumerate(self.zones, start=1):
            if zone.contains(point):
                return i
        return None

    def __len__(self):
        return len(self.zones)


class CrosswalkOccupancy:
   
    def __init__(self, crosswalk_zones: CrosswalkZones, confirm_frames=None,
                 grace_frames=None, inherit_distance=None):
        self.zones = crosswalk_zones
        self.confirm_frames = confirm_frames or config.ZONE_RESIDENCY_FRAMES
        if self.confirm_frames is None:
            raise NotImplementedError("ZONE_RESIDENCY_FRAMES가 설정되지 않았습니다.")
        self.grace_frames = (
            grace_frames if grace_frames is not None else config.TRACK_GRACE_FRAMES
        )
        if self.grace_frames is None or self.grace_frames < 0:
            raise ValueError(
                f"grace_frames는 0 이상이어야 합니다: {self.grace_frames}"
            )
        if inherit_distance is not None:
            self.inherit_distance = inherit_distance
        elif self.zones.ground_plane is not None:
            self.inherit_distance = config.TRACK_INHERIT_DISTANCE_CM
        else:
            self.inherit_distance = config.TRACK_INHERIT_DISTANCE_PX

        self._state = {}
        self.untracked_count = 0
        self.rekeyed = []

    def update(self, detections) -> dict:
        
        seen = set()
        self.untracked_count = 0
        self.rekeyed = []

        detections = list(detections)
        self._inherit_reissued_ids(detections)

        for track_id, point in detections:
            if track_id is None:
                self.untracked_count += 1
                continue
            zone_index = self.zones.locate(point)
            if zone_index is None:
                self._state.pop(track_id, None)
                continue
            seen.add(track_id)
            st = self._state.get(track_id,
                                 {"count": 0, "zone": None, "misses": 0, "point": None})
            st["count"] += 1
            st["zone"] = zone_index
            st["misses"] = 0
            st["point"] = point
            self._state[track_id] = st

        for track_id in list(self._state):
            if track_id in seen:
                continue
            st = self._state[track_id]
            st["misses"] += 1
            if st["misses"] > self.grace_frames:
                self._state.pop(track_id, None)

        return self.confirmed()

    def _distance(self, a, b) -> float:
       
        plane = self.zones.ground_plane
        if plane is not None:
            a, b = plane.to_ground(a), plane.to_ground(b)
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def _inherit_reissued_ids(self, detections):
        if self.inherit_distance <= 0:
            return

        fresh = [(i, tid, pt) for i, (tid, pt) in enumerate(detections)
                 if tid is not None and tid not in self._state]
        if not fresh:
            return

        visible = {tid for tid, _ in detections if tid is not None}
        lost = [tid for tid, st in self._state.items()
                if tid not in visible and st["point"] is not None]
        if not lost:
            return

        pairs = []
        for index, _, point in fresh:
            for old_id in lost:
                distance = self._distance(point, self._state[old_id]["point"])
                if distance <= self.inherit_distance:
                    pairs.append((distance, index, old_id))
        pairs.sort(key=lambda p: (p[0], p[1]))

        claimed_detections, claimed_tracks = set(), set()
        for _, index, old_id in pairs:
            if index in claimed_detections or old_id in claimed_tracks:
                continue
            new_id = detections[index][0]
            self._state[new_id] = self._state.pop(old_id)
            self.rekeyed.append((old_id, new_id))
            claimed_detections.add(index)
            claimed_tracks.add(old_id)

    def confirmed(self) -> dict:
        return {
            tid: st["zone"]
            for tid, st in self._state.items()
            if st["count"] >= self.confirm_frames
        }

    def occupied_zones(self) -> list:
        return sorted(set(self.confirmed().values()))

    def clear(self):
        self._state.clear()
        self.untracked_count = 0
        self.rekeyed = []


class CrossingProgress:
    
    def __init__(self, zone_count=None):
        self.zone_count = zone_count or config.CROSSWALK_ZONE_COUNT
        self._direction = {}
        self._first_zone = {}

    @property
    def _middle(self) -> float:
        return (self.zone_count + 1) / 2

    def update(self, track_id, zone) -> int:
        if track_id not in self._first_zone:
            self._first_zone[track_id] = zone
            if zone < self._middle:
                self._direction[track_id] = 1
            elif zone > self._middle:
                self._direction[track_id] = -1
            else:
                self._direction[track_id] = None      
        elif self._direction[track_id] is None and zone != self._first_zone[track_id]:
            self._direction[track_id] = 1 if zone > self._first_zone[track_id] else -1

        return self.progress(track_id, zone)

    def progress(self, track_id, zone) -> int:
        direction = self._direction.get(track_id)
        if direction is None:
            return zone                                # 방향 미정 -> 물리 구역 그대로
        return zone if direction > 0 else self.zone_count + 1 - zone

    def direction(self, track_id):
        return self._direction.get(track_id)

    def rekey(self, old_track_id, new_track_id):
        if new_track_id in self._first_zone or old_track_id not in self._first_zone:
            return
        self._first_zone[new_track_id] = self._first_zone.pop(old_track_id)
        self._direction[new_track_id] = self._direction.pop(old_track_id)

    def forget(self, track_id):
        self._direction.pop(track_id, None)
        self._first_zone.pop(track_id, None)

    def keep_only(self, track_ids):
        alive = set(track_ids)
        for track_id in list(self._first_zone):
            if track_id not in alive:
                self.forget(track_id)

    def clear(self):
        self._direction.clear()
        self._first_zone.clear()
