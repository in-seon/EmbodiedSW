import numpy as np
from config import config


class GroundPlane:
    def __init__(self, matrix, inverse_matrix, width_cm, length_cm):
        self.matrix = matrix                  
        self.inverse_matrix = inverse_matrix  
        self.width_cm = float(width_cm)
        self.length_cm = float(length_cm)

    @staticmethod
    def _apply(matrix, point):
        px, py = float(point[0]), float(point[1])
        m = matrix
        denom = m[2][0] * px + m[2][1] * py + m[2][2]
        if abs(denom) < 1e-12:
            raise ValueError(f"변환할 수 없는 점입니다(소실선 근처): {point}")
        x = (m[0][0] * px + m[0][1] * py + m[0][2]) / denom
        y = (m[1][0] * px + m[1][1] * py + m[1][2]) / denom
        return (float(x), float(y))

    @classmethod
    def from_quad(cls, corners, width_cm, length_cm):
        if width_cm is None or length_cm is None:
            return None
        if len(corners) != 4:
            raise ValueError("corners는 [시작-왼쪽, 시작-오른쪽, 끝-오른쪽, 끝-왼쪽] 4개 점")
        if width_cm <= 0 or length_cm <= 0:
            raise ValueError(f"치수는 양수여야 합니다: width_cm={width_cm}, length_cm={length_cm}")

        import cv2  
        src = np.array(corners, dtype=np.float32)
        dst = np.array(                         
            [
                [0.0, 0.0],                  
                [float(width_cm), 0.0],      
                [float(width_cm), float(length_cm)],  
                [0.0, float(length_cm)],     
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        inverse_matrix = cv2.getPerspectiveTransform(dst, src)
        return cls(matrix, inverse_matrix, width_cm, length_cm)

    @classmethod
    def from_config(cls, corners):
        return cls.from_quad(corners, config.CROSSWALK_REAL_WIDTH_CM, config.CROSSWALK_REAL_LENGTH_CM)

    def to_ground(self, point):
        return self._apply(self.matrix, point)

    def to_pixel(self, ground_point):
        return self._apply(self.inverse_matrix, ground_point)

    def remaining_distance_cm(self, ground_point, direction):
        if direction > 0:
            return max(0.0, self.length_cm - ground_point[1])
        if direction < 0:
            return max(0.0, ground_point[1] - 0.0)
        return None
