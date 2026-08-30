import json

from config import config
from src.ground_plane import GroundPlane


class Zone:
    """폴리곤(또는 사각형) ROI. 좌표는 프레임 픽셀 좌표계."""

    def __init__(self, points, name="crosswalk"):
        if len(points) < 3:
            raise ValueError("Zone은 최소 3개의 점으로 이루어진 폴리곤이어야 합니다.")
        self.points = [(float(x), float(y)) for x, y in points]
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


class CrosswalkZones:
    """횡단보도를 걷는 방향을 따라 위치 순서대로 N개(기본 5개) 구역으로 나눈 묶음.

    구역 번호는 1..N. 1과 N이 양 끝, 가운데(N=5면 3)가 중앙 구역이다.
    """

    def __init__(self, zones, name="crosswalk", ground_plane=None):
        self.zones = list(zones)
        self.name = name
        # 속도 추정용 픽셀->cm 변환. 실측 치수가 없으면 None (구역 판정에는 영향 없음).
        self.ground_plane = ground_plane

    @staticmethod
    def _lerp(p, q, t):
        return (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)

    @classmethod
    def _split_on_ground(cls, ground_plane, n, name):
        """실제 거리(cm) 기준으로 N등분한 뒤, 경계를 픽셀 좌표로 되돌린다.

        이것이 올바른 분할이다. 사영변환은 직선을 직선으로 보내므로, 평면상 직사각형 띠의
        네 꼭짓점만 역변환하면 화면상 정확한 구역 사각형이 된다(근사가 아니다).
        """
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
        """화면(픽셀) 위에서 균등 분할하는 대체 경로 — 실측 치수가 없을 때만 쓴다.

        원근 때문에 실제 거리로는 균등하지 않다. 아래 from_quad 주석 참고.
        """
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
                f"현재 운영 해상도({expected[0]}x{expected[1]})와 다릅니다. "
                "zone 좌표와 호모그래피가 통째로 어긋나므로 그대로 쓰면 구역 판정이 틀립니다. "
                "config.CAMERA_RESOLUTION을 캘리브레이션 당시 값으로 되돌리거나, "
                "tools/zone_calibrator.py 로 현재 해상도에서 다시 캘리브레이션하세요."
            )

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
        """point가 속한 구역 번호(1..N)를 반환. 어느 구역에도 없으면 None."""
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
            raise NotImplementedError(
                "ZONE_RESIDENCY_FRAMES가 설정되지 않았습니다. 실측 FPS 기반으로 값을 정한 뒤 사용하세요."
            )
        # 0은 '유예 없음'이라는 유효한 설정이므로 `or`가 아니라 None 검사를 쓴다.
        self.grace_frames = (
            grace_frames if grace_frames is not None else config.TRACK_GRACE_FRAMES
        )
        if self.grace_frames is None or self.grace_frames < 0:
            raise ValueError(
                f"grace_frames는 0 이상이어야 합니다(0 = 유예 없음): {self.grace_frames}"
            )
        # 새 track_id를 직전에 사라진 트랙에 이어붙일 거리 임계값. 0이면 상속하지 않는다.
        # 호모그래피가 있으면 cm, 없으면 px 단위다 (_distance 참고).
        if inherit_distance is not None:
            self.inherit_distance = inherit_distance
        elif self.zones.ground_plane is not None:
            self.inherit_distance = config.TRACK_INHERIT_DISTANCE_CM
        else:
            self.inherit_distance = config.TRACK_INHERIT_DISTANCE_PX

        # track_id -> {"count": 연속 프레임 수, "zone": 최근 구역 번호,
        #              "misses": 연속 미검출 수, "point": 마지막으로 본 발 위치}
        self._state = {}
        # 직전 update()에서 track_id가 없어 무시한 검출 수 (아래 update 주석 참고).
        self.untracked_count = 0
        # 직전 update()에서 이어붙인 [(옛 track_id, 새 track_id)].
        # 파이프라인이 CrossingProgress에도 같은 이름 변경을 전달해야 한다 — 그러지 않으면
        # 카운트만 살아남고 **진입 방향은 그대로 날아간다**(이 기능의 핵심 이유).
        self.rekeyed = []

    def update(self, detections) -> dict:
        
        seen = set()
        self.untracked_count = 0
        self.rekeyed = []

        detections = list(detections)
        # 상속을 먼저 처리한다. 여기서 이름이 바뀌면 아래 루프가 새 이름으로 카운트를
        # 이어서 올린다 — 즉 상속된 트랙은 이번 프레임에 곧바로 확정 보행자가 된다.
        self._inherit_reissued_ids(detections)

        for track_id, point in detections:
            if track_id is None:
                self.untracked_count += 1
                continue
            zone_index = self.zones.locate(point)
            if zone_index is None:
                # 횡단보도 밖 -> 유예 없이 즉시 리셋 (클래스 주석 참고: '안 보임'과 다르다)
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

        # 이번 프레임에 검출되지 않은 트랙: 유예 안에서는 상태를 그대로 두고(카운트 동결),
        # 유예를 넘기면 그때 제거한다.
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
        """'처음 보는 ID'를 직전에 사라진 트랙에 이어붙인다 (같은 사람으로 간주).

        ## 후보 규칙

          - **처음 보는 ID만** 이어붙인다. 이미 관리 중인 번호는 빠른 경로로 지나간다.
            덕분에 "남의 상태를 덮어쓴다"가 구조적으로 불가능하다.
          - **이번 프레임에 보인 트랙은 후보가 아니다.** 살아 있는 사람의 상태를 뺏으면
            그 사람은 확정이 풀리고 새 ID는 남의 방향을 얻는다 — 두 번 틀린다.
          - track_id가 None인 검출은 상속하지 않는다. 프레임 간 대응을 모르는 검출이라
            '같은 사람'의 근거가 위치뿐인데, 그것만으로 확정 보행자를 만들면 스쳐가는
            오검출이 연장을 일으킨다.

        ## 왜 전역 최근접인가

        검출마다 순서대로 "가장 가까운 후보"를 집으면 **처리 순서에 결과가 좌우된다.**
        두 사람이 스쳐 지날 때 먼저 처리된 검출이 남의 트랙을 집어가면 나머지도 연쇄로
        어긋난다. 가능한 (검출, 트랙) 쌍을 모두 만들어 거리순으로 배정하면 그 순서
        의존이 없어진다.
        """
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
        """확정 보행자 {track_id: zone_index}."""
        return {
            tid: st["zone"]
            for tid, st in self._state.items()
            if st["count"] >= self.confirm_frames
        }

    def occupied_zones(self) -> list:
        """확정 보행자들이 점유 중인 구역 번호 목록(중복 제거·정렬)."""
        return sorted(set(self.confirmed().values()))

    def clear(self):
        self._state.clear()
        self.untracked_count = 0
        self.rekeyed = []


class CrossingProgress:
    
    def __init__(self, zone_count=None):
        self.zone_count = zone_count or config.CROSSWALK_ZONE_COUNT
        # track_id -> +1(정방향) / -1(역방향) / None(아직 모름)
        self._direction = {}
        # track_id -> 처음 검출된 구역 번호
        self._first_zone = {}

    @property
    def _middle(self) -> float:
        """중앙 구역 번호. 5구역이면 3.0, 짝수면 경계값(예: 4구역이면 2.5)."""
        return (self.zone_count + 1) / 2

    def update(self, track_id, zone) -> int:
        """이번 프레임 구역을 반영하고 진척도(1..N)를 돌려준다."""
        if track_id not in self._first_zone:
            self._first_zone[track_id] = zone
            if zone < self._middle:
                self._direction[track_id] = 1
            elif zone > self._middle:
                self._direction[track_id] = -1
            else:
                self._direction[track_id] = None      # 중앙에서 시작 — 아직 모름
        elif self._direction[track_id] is None and zone != self._first_zone[track_id]:
            # 중앙에서 시작한 사람이 움직였다 -> 그 방향으로 확정
            self._direction[track_id] = 1 if zone > self._first_zone[track_id] else -1

        return self.progress(track_id, zone)

    def progress(self, track_id, zone) -> int:
        """저장된 방향으로 구역 번호를 진척도로 바꾼다(상태를 갱신하지 않는다)."""
        direction = self._direction.get(track_id)
        if direction is None:
            return zone                                # 방향 미정 -> 물리 구역 그대로
        return zone if direction > 0 else self.zone_count + 1 - zone

    def direction(self, track_id):
        """+1 / -1 / None(미정). 화면 표시·진단용."""
        return self._direction.get(track_id)

    def rekey(self, old_track_id, new_track_id):
        """트래커가 같은 사람에게 새 ID를 준 경우, 방향 래치를 그 ID로 옮긴다.

        **이 기능의 존재 이유가 여기 있다.** 카운트만 물려받고 방향을 잃으면, 거의 다
        건넌 사람(구역 4)이 "중앙보다 뒤에서 처음 보였다"로 해석돼 역방향으로 확정되고
        진척도가 4가 아니라 2로 나간다. 그 뒤로는 5번 구역에서 1이 나와 **진척도가
        감소하며** 아두이노가 매번 새로 연장한다.

        이미 쓰이는 번호면 아무것도 하지 않는다 — 남의 방향을 덮어쓰느니 새 사람으로
        보는 편이 낫다(그 경우의 오차는 '덜 왔다'고 보는 쪽 = 안전한 쪽이다).
        """
        if new_track_id in self._first_zone or old_track_id not in self._first_zone:
            return
        self._first_zone[new_track_id] = self._first_zone.pop(old_track_id)
        self._direction[new_track_id] = self._direction.pop(old_track_id)

    def forget(self, track_id):
        """트랙이 사라졌을 때 호출. 안 지우면 ID가 재사용될 때 옛 방향이 따라붙는다."""
        self._direction.pop(track_id, None)
        self._first_zone.pop(track_id, None)

    def keep_only(self, track_ids):
        """살아 있는 트랙만 남긴다."""
        alive = set(track_ids)
        for track_id in list(self._first_zone):
            if track_id not in alive:
                self.forget(track_id)

    def clear(self):
        self._direction.clear()
        self._first_zone.clear()
