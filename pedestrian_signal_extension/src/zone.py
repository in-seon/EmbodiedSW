"""Zone(횡단보도 ROI) 정의 + 5구역 위치 판정 + 잔류/점유 추적.

CLAUDE.md 2.2: 횡단보도를 걷는 방향을 따라 위치 순서대로 5개 구역(1~5)으로 나누고,
보행자가 어느 구역에 있는지에 따라 신호 연장 시간을 차등한다(config.ZONE_EXTENSION_SEC).
연장 시간 결정은 signal_extend 모듈이 담당하고, 여기서는 "누가 몇 번 구역에 있는지"까지만 만든다.

판정 기준점은 bounding box 하단 모서리의 중심(BoundingBox.foot_point)이다. 카메라가 사선으로
비추므로 박스 중심점은 사람 키만큼 떠 있어 실제 서 있는 지점보다 카메라 쪽으로 당겨져 보인다
(CLAUDE.md 2.1).

구역 판정은 픽셀 좌표계의 폴리곤 포함 여부만으로 하며 실거리 변환이 필요 없다. 실거리가 필요한
속도 추정은 별도 모듈(ground_plane.py + speed.py)이 맡는다. 다만 둘 다 같은 네 꼭짓점에서
출발하므로, 여기서 zone 설정을 읽을 때 GroundPlane도 같이 만들어 `ground_plane` 속성으로 들고
있는다(설정 파일을 두 번 읽지 않기 위함). 실측 치수가 없으면 None이고, 그래도 구역 판정은 정상 동작한다.
"""

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
        """횡단보도 사각형의 네 꼭짓점을 받아 걷는 방향으로 N개 구역(스트립)으로 자른다.

        corners 순서: [시작-왼쪽, 시작-오른쪽, 끝-오른쪽, 끝-왼쪽].
        걷는 방향은 '시작 변(0-1)' -> '끝 변(3-2)'. 이 방향을 따라 N등분한다.
        (tools/zone_calibrator.py 가 이 순서대로 클릭받아 저장한다.)

        ## 어디서 5등분하는가 — 화면이 아니라 '실제 거리' 기준이다

        사선 카메라에서는 **화면상 균등 분할이 실제로는 전혀 균등하지 않다.** 원근 때문에
        먼 쪽 1픽셀이 가까운 쪽 1픽셀보다 훨씬 긴 실거리를 나타내기 때문이다.
        폭 400cm x 길이 1000cm 횡단보도를 사선으로 본 실측 예에서, 화면상 5등분은 실제로
        이런 구역을 만든다:

            1번  77cm  |  2번 105cm  |  3번 152cm  |  4번 238cm  |  5번 429cm

        1번과 5번이 5.6배 차이나고, 더 중요하게는 **횡단보도의 진짜 한가운데(500cm)가
        3번이 아니라 4번 구역에 들어간다.** "정중앙에 있으면 가장 길게 연장한다"는
        설계 전제(CLAUDE.md 2.4)가 그대로 깨진다.

        그래서 호모그래피가 있으면 **평면 좌표(cm)에서 균등 분할한 뒤 픽셀로 되돌린다**
        (`_split_on_ground`). 그러면 각 구역이 실제로 200cm씩 담당한다.

        width_cm/length_cm가 없으면 호모그래피를 만들 수 없으므로 화면상 분할로 대체하고
        (`_split_on_pixels`), 위와 같은 왜곡을 감수한다. 실측 치수를 넣는 것이 정확도에
        직결되는 이유가 이것이다.
        """
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
        """캘리브레이션 당시 해상도와 운영 해상도가 같은지 확인한다.

        zone 좌표와 호모그래피는 둘 다 '캘리브레이션 당시 프레임 해상도'에 종속된 픽셀 값이다.
        운영 해상도가 다르면 좌표가 통째로 어긋나는데, **에러 없이 구역만 엉뚱하게 잡히는**
        조용한 오작동이라 현장에서 원인을 찾기가 가장 어렵다. zone_calibrator가 frame_size를
        남겨 두므로, 읽는 쪽에서 비교해 즉시 실패시킨다.

        frame_size가 없는 옛 설정 파일은 검증할 근거가 없으므로 통과시킨다(재캘리브레이션 권장).
        """
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
        """zone 설정 JSON을 읽는다.

        지원 형식:
          {"name": ..., "corners": [[x,y]*4], "n_zones": 5,
           "real_width_cm": 90, "real_length_cm": 300}       # 4코너 자동 분할(권장)
          {"name": ..., "zones": [[[x,y]*>=3], ...]}          # 구역 폴리곤을 직접 나열

        실측 치수(real_width_cm / real_length_cm)는 선택 사항이다. 파일에 없으면 config의
        CROSSWALK_REAL_WIDTH_CM / CROSSWALK_REAL_LENGTH_CM 로 대체하고, 그것도 없으면
        ground_plane은 None이 된다(속도가 px/s로 떨어질 뿐 구역 판정은 정상).

        'zones' 형식(폴리곤 직접 나열)에는 네 꼭짓점이 없으므로 호모그래피를 만들 수 없다.
        속도를 cm/s로 내려면 'corners' 형식을 써야 한다.

        설정에 'frame_size'가 있으면 expected_frame_size(기본: config.CAMERA_RESOLUTION)와
        비교해 다르면 ValueError를 낸다 — 해상도가 다르면 좌표가 통째로 어긋나기 때문이다.
        """
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
    """트랙(사람 ID)별로 현재 위치한 구역을 추적한다.

    검출 흔들림에 대비해, 횡단보도(아무 구역) 안에서 confirm_frames 이상 연속 검출돼야
    '확정 보행자'로 본다. 트랙 ID 부여(추적/재식별)는 검출 파이프라인 책임이며 여기선 받기만 한다.

    참고: 지금은 한 프레임이라도 검출이 끊기면 카운트를 리셋한다(단순·보수적). 추후 실측에서
    깜빡임이 잦으면 '몇 프레임 유예(grace)' 로직을 추가해 완화할 수 있다.
    """

    def __init__(self, crosswalk_zones: CrosswalkZones, confirm_frames=None):
        self.zones = crosswalk_zones
        self.confirm_frames = confirm_frames or config.ZONE_RESIDENCY_FRAMES
        if self.confirm_frames is None:
            raise NotImplementedError(
                "ZONE_RESIDENCY_FRAMES가 설정되지 않았습니다. 실측 FPS 기반으로 값을 정한 뒤 사용하세요."
            )
        # track_id -> {"count": 연속 프레임 수, "zone": 최근 구역 번호}
        self._state = {}
        # 직전 update()에서 track_id가 없어 무시한 검출 수 (아래 update 주석 참고).
        self.untracked_count = 0

    def update(self, detections) -> dict:
        """이번 프레임 검출을 반영하고, 확정 보행자의 {track_id: zone_index} 를 반환한다.

        detections: (track_id, point) 튜플의 iterable. point는 보통 BoundingBox.foot_point().

        track_id가 None인 검출은 무시한다(SpeedEstimator.update_many와 같은 규칙). 프레임 간
        대응을 알 수 없는 검출로는 '연속 몇 프레임 잔류했는가'를 셀 수 없기 때문이다. None을
        키로 쓰면 서로 다른 사람들이 한 카운터에 합쳐져, 같은 프레임 안의 여러 명이 각각
        카운트를 올리고 마지막 사람의 구역만 남는다 — 잔류 검증이 통째로 무력화된다.

        무시한 개수는 untracked_count로 노출한다. 조용히 놓치면 실측에서 "왜 확정이 안 되지"의
        원인을 찾을 수 없으므로, 추적이 자주 끊기고 있다는 사실이 보이게 한다.
        """
        seen = set()
        self.untracked_count = 0
        for track_id, point in detections:
            if track_id is None:
                self.untracked_count += 1
                continue
            zone_index = self.zones.locate(point)
            if zone_index is None:
                self._state.pop(track_id, None)  # 횡단보도 밖 -> 카운트 리셋
                continue
            seen.add(track_id)
            st = self._state.get(track_id, {"count": 0, "zone": None})
            st["count"] += 1
            st["zone"] = zone_index
            self._state[track_id] = st

        # 이번 프레임에 검출되지 않은 트랙 제거
        for track_id in list(self._state):
            if track_id not in seen:
                self._state.pop(track_id, None)

        return self.confirmed()

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
