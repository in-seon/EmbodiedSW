"""Zone 좌표 캘리브레이션 도구.

카메라 프레임 한 장을 띄우고 마우스 클릭으로 zone 폴리곤 점을 지정한다.
's' 키로 config.ZONE_CONFIG_PATH에 저장, 'r' 키로 리셋, 'q' 키로 종료.

사용법:
    python tools/zone_calibrator.py --source 0
    python tools/zone_calibrator.py --source path/to/sample.jpg
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import config

_points = []


def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _points.append((x, y))


def _grab_reference_frame(source):
    try:
        source = int(source)
    except ValueError:
        pass  # 파일 경로인 경우

    if isinstance(source, str) and Path(source).exists():
        frame = cv2.imread(source)
        if frame is None:
            raise RuntimeError(f"이미지를 읽을 수 없습니다: {source}")
        return frame

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: {source}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("프레임을 읽을 수 없습니다.")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=config.CAMERA_SOURCE, help="카메라 인덱스 또는 이미지 경로")
    parser.add_argument("--out", default=config.ZONE_CONFIG_PATH, help="저장 경로")
    parser.add_argument("--name", default="crosswalk", help="zone 이름")
    args = parser.parse_args()

    frame = _grab_reference_frame(args.source)

    window = "Zone Calibrator (click: add point, s: save, r: reset, q: quit)"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, _on_mouse)

    while True:
        display = frame.copy()
        for p in _points:
            cv2.circle(display, p, 4, (0, 0, 255), -1)
        if len(_points) > 1:
            cv2.polylines(display, [_as_np(_points)], True, (0, 255, 0), 2)

        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            _points.clear()
        elif key == ord("s"):
            if len(_points) < 3:
                print("최소 3개의 점이 필요합니다.")
                continue
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"name": args.name, "points": _points}, f, ensure_ascii=False, indent=2)
            print(f"저장 완료: {out_path}")

    cv2.destroyAllWindows()


def _as_np(points):
    import numpy as np

    return np.array(points, dtype=int)


if __name__ == "__main__":
    main()
