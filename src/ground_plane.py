"""픽셀 좌표 -> 횡단보도 평면 좌표(cm) 변환 (호모그래피).

CLAUDE.md 2.1, 2.2:
카메라가 보행자 신호등에 붙어 횡단보도를 "사선"으로 비추기 때문에, 화면상 1픽셀이 나타내는
실거리가 위치마다 다르다(카메라에 가까운 아래쪽은 촘촘하고, 먼 위쪽은 성기다). 이 상태로
픽셀 변위를 속도라고 부르면 같은 사람이 같은 속도로 걸어도 화면 위/아래에서 값이 달라진다.

보행자의 발은 지면(하나의 평면) 위에 있으므로, 평면 하나에 대한 사영변환(호모그래피)이면
원근을 펼 수 있다. 필요한 입력은 두 가지뿐이다.

  1. 횡단보도 네 꼭짓점의 픽셀 좌표  <- tools/zone_calibrator.py 가 이미 받고 있음
  2. 횡단보도의 실제 가로/세로 치수(cm)  <- 새로 입력받음

체스보드로 렌즈 왜곡까지 잡는 정식 카메라 캘리브레이션은 하지 않는다. 모형 규모에서는
평면 사영변환만으로 충분하고, 절차가 하나 늘지 않는 편이 실제로 쓰일 확률이 높다.

평면 좌표계 규약 (이 규약을 speed.py가 그대로 신뢰한다):

        x = 폭 방향 (0 .. width_cm)
        y = 걷는 방향 (0 = 시작 변, length_cm = 끝 변)

        (0,0) ────────────── (width, 0)      <- 시작 변
          │                      │
          │   y 증가 = 진행 방향  │
          │                      ↓
        (0,length) ───────── (width, length) <- 끝 변

주의: 호모그래피는 카메라 위치와 프레임 해상도에 종속된다. 둘 중 하나라도 바뀌면 재캘리브레이션할 것.
"""

import numpy as np

from config import config


class GroundPlane:
    """횡단보도 네 꼭짓점 <-> 실제 치수(cm) 사이의 사영변환.

    직접 생성하기보다 from_quad()를 쓴다. CrosswalkZones.load()가 zone 설정 JSON에
    치수가 들어 있으면 자동으로 만들어 준다.
    """

    def __init__(self, matrix, inverse_matrix, width_cm, length_cm):
        self.matrix = matrix                  # 픽셀 -> 평면(cm)
        self.inverse_matrix = inverse_matrix  # 평면(cm) -> 픽셀
        self.width_cm = float(width_cm)
        self.length_cm = float(length_cm)

    @staticmethod
    def _apply(matrix, point):
        """3x3 사영변환을 한 점에 적용한다 (동차좌표 나눗셈 포함)."""
        px, py = float(point[0]), float(point[1])
        m = matrix
        denom = m[2][0] * px + m[2][1] * py + m[2][2]
        if abs(denom) < 1e-12:
            # 소실선 위의 점 — 대응하는 상이 무한대라 변환 불가.
            raise ValueError(f"변환할 수 없는 점입니다(소실선 근처): {point}")
        x = (m[0][0] * px + m[0][1] * py + m[0][2]) / denom
        y = (m[1][0] * px + m[1][1] * py + m[1][2]) / denom
        # numpy 스칼라가 아래 단계(speed 계산, JSON 저장)로 새지 않도록 순수 float로 반환.
        return (float(x), float(y))

    @classmethod
    def from_quad(cls, corners, width_cm, length_cm):
        """네 꼭짓점(픽셀)과 실제 치수(cm)로 변환 행렬을 만든다.

        corners 순서는 CrosswalkZones.from_quad와 동일:
            [시작-왼쪽, 시작-오른쪽, 끝-오른쪽, 끝-왼쪽]
        이 순서가 위 도식의 (0,0) / (w,0) / (w,L) / (0,L) 에 각각 대응한다.

        width_cm / length_cm 중 하나라도 None이면 None을 반환한다(치수 미실측 상태).
        호출부는 None을 "속도를 cm로 낼 수 없음"으로 해석하면 된다.
        """
        if width_cm is None or length_cm is None:
            return None
        if len(corners) != 4:
            raise ValueError("corners는 [시작-왼쪽, 시작-오른쪽, 끝-오른쪽, 끝-왼쪽] 4개 점이어야 합니다.")
        if width_cm <= 0 or length_cm <= 0:
            raise ValueError(f"치수는 양수여야 합니다: width_cm={width_cm}, length_cm={length_cm}")

        import cv2  # 지연 임포트: 좌표 계산만 하는 테스트가 OpenCV 없이도 돌게 한다.

        src = np.array(corners, dtype=np.float32) # 영상 좌표
        dst = np.array(                           # 실제 좌표
            [
                [0.0, 0.0],                  # 시작-왼쪽
                [float(width_cm), 0.0],      # 시작-오른쪽
                [float(width_cm), float(length_cm)],  # 끝-오른쪽
                [0.0, float(length_cm)],     # 끝-왼쪽
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        inverse_matrix = cv2.getPerspectiveTransform(dst, src)
        return cls(matrix, inverse_matrix, width_cm, length_cm)

    @classmethod
    def from_config(cls, corners):
        """config의 실측 치수로 만든다. 치수가 미입력이면 None."""
        return cls.from_quad(corners, config.CROSSWALK_REAL_WIDTH_CM, config.CROSSWALK_REAL_LENGTH_CM)

    def to_ground(self, point):
        """픽셀 좌표 (px, py) -> 평면 좌표 (x_cm, y_cm).

        횡단보도 사각형 밖의 점도 변환된다(외삽). 값이 [0,width]x[0,length]를 벗어나면
        횡단보도 밖이라는 뜻이다. 구역 판정은 zone.py가 따로 하므로 여기서 막지 않는다.
        """
        return self._apply(self.matrix, point)

    def to_pixel(self, ground_point):
        """평면 좌표 (x_cm, y_cm) -> 픽셀 좌표 (px, py). to_ground의 역변환.

        "실제 거리 기준으로 정한 위치"를 화면에 그리거나 픽셀 기준 판정에 쓸 때 필요하다.
        구역을 실제 거리로 균등 분할한 뒤 그 경계선을 화면 좌표로 되돌리는 데 쓴다
        (zone.py의 CrosswalkZones.from_quad 참고).

        사영변환은 직선을 직선으로 보내므로, 평면상 직사각형 띠는 픽셀상 사각형이 된다
        — 즉 경계 두 점만 변환하면 정확한 구역 폴리곤이 나온다(근사가 아니다).
        """
        return self._apply(self.inverse_matrix, ground_point)

    def remaining_distance_cm(self, ground_point, direction):
        """진행 방향 기준으로 반대편 끝까지 남은 거리(cm).

        direction: +1 이면 끝 변(y=length_cm) 쪽으로, -1 이면 시작 변(y=0) 쪽으로 가는 중.
        direction이 0(정지/판별 불가)이면 None.
        """
        if direction > 0:
            return max(0.0, self.length_cm - ground_point[1])
        if direction < 0:
            return max(0.0, ground_point[1] - 0.0)
        return None
