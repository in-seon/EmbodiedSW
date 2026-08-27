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

    ## 미검출 유예 (grace_frames)

    YOLO는 가림·모션블러·저조도에서 한두 프레임씩 사람을 놓치는 것이 정상이다. 그때마다
    카운트를 0으로 되돌리면 확정 보행자가 좀처럼 나오지 않고, **에러 없이 조용히 연장만
    빠진다.** 그래서 grace_frames 프레임까지의 미검출은 상태를 지우지 않고 넘어간다.

    유예 동안 카운트는 **동결**된다 — 안 보이는 동안은 잔류의 증거가 없으므로 올려주지 않되,
    이미 쌓은 것을 뺏지도 않는다. 구역 번호는 마지막으로 본 값을 유지한다.

    단, **'구역 밖으로 나감'은 유예 대상이 아니다**(즉시 리셋). "안 보임"은 정보가 없는
    상태지만 "밖으로 나감"은 확실한 정보다. 둘을 같이 취급하면 이미 횡단보도를 벗어난
    사람 때문에 차를 세우게 된다.

    유예 단위가 '초'가 아니라 '프레임 수'인 이유는 확정 기준(confirm_frames)이 프레임 수라
    같은 단위여야 섞이지 않기 때문이다. FallTracker가 '초 + 프레임 배수' 적응형을 쓰는 것은
    그쪽 누적(fall_confirm_sec)이 시간 기준이라서다 — config.TRACK_GRACE_FRAMES 주석 참고.
    """

    def __init__(self, crosswalk_zones: CrosswalkZones, confirm_frames=None, grace_frames=None):
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
        # track_id -> {"count": 연속 프레임 수, "zone": 최근 구역 번호, "misses": 연속 미검출 수}
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
                # 횡단보도 밖 -> 유예 없이 즉시 리셋 (클래스 주석 참고: '안 보임'과 다르다)
                self._state.pop(track_id, None)
                continue
            seen.add(track_id)
            st = self._state.get(track_id, {"count": 0, "zone": None, "misses": 0})
            st["count"] += 1
            st["zone"] = zone_index
            st["misses"] = 0
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


class CrossingProgress:
    """track_id별 '얼마나 건넜는가'(진척도 1..N)를 낸다.

    ## 왜 물리 구역 번호를 그대로 쓸 수 없는가

    구역 번호는 **좌표계 기준**이라 진입 방향에 따라 의미가 뒤집힌다. 물리적으로 같은
    2번 구역에 있어도, 1번에서 들어온 사람은 대부분 남았고 5번에서 온 사람은 거의 다
    건넜다. 같은 번호를 보내면 아두이노가 둘을 구분할 수 없다.

    그래서 사람마다 **자기 진입점 기준으로** 번호를 다시 매긴다:

        진척도 1 = 방금 진입      진척도 N = 거의 다 건넘

    ## 방향을 어떻게 아는가 — 속도를 쓰지 않는다

    **첫 검출 구역**으로 판별한다. 중앙보다 앞쪽에서 처음 보였으면 정방향, 뒤쪽이면
    역방향이다. 속도 추정과 달리 프레임률 요구가 없고(위치 이력만 보면 된다), 사람이
    멈춰 있어도 판별이 유지된다.

    중앙에서 처음 보이면 방향을 알 수 없다(track_id 재발급 등). 그때는 **중앙값으로
    간주**하고, 이후 구역이 바뀌면 그 방향으로 확정한다. 3번에서 시작해 4번으로 가면
    정방향이 확정되어 진척도도 4가 되고, 2번으로 가면 역방향이 확정되어 진척도가
    6-2=4가 된다 — 어느 쪽이든 "중앙에서 한 칸 더 갔다"로 수렴한다.

    ## 오판하면 어느 쪽으로 틀리는가

    첫 검출이 실제 진입점이 아니면(중간에 ID가 재발급되면) 방향을 반대로 볼 수 있다.
    그때 진척도는 **실제보다 작게** 나오고, 작은 진척도는 곧 '덜 왔다'이므로 아두이노가
    **더 연장하는** 쪽으로 틀린다. 보행자가 도로에 갇히는 것보다 차가 조금 더 기다리는
    것이 낫다는 이 프로젝트의 기존 판단과 같은 방향의 오차다.
    """

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
