"""Zone(횡단보도 5구역) 캘리브레이션 도구.

카메라 프레임 한 장을 띄우고, 횡단보도 사각형의 네 꼭짓점을 아래 순서로 클릭한다.

    1) 시작 변 왼쪽   2) 시작 변 오른쪽   3) 끝 변 오른쪽   4) 끝 변 왼쪽

'시작 변 -> 끝 변' 방향이 보행자가 걷는 방향이며, 이 방향을 따라 자동으로 N등분(기본 5구역)한다.
저장된 JSON은 CrosswalkZones.load()가 그대로 읽는다.

--width-cm / --length-cm 로 횡단보도(모형)의 실제 치수를 함께 주면 호모그래피까지 만들어져
속도를 cm/s로 낼 수 있다(CLAUDE.md 2.2, 2.3). 생략하면 구역 판정만 되고 속도는 px/s로 나온다.

    - width  = 걷는 방향과 '수직'인 변의 길이 (시작-왼쪽 <-> 시작-오른쪽 사이)
    - length = 걷는 방향 변의 길이 (시작 변 <-> 끝 변 사이)

    's' : config.ZONE_CONFIG_PATH(또는 --out)에 저장
    'r' : 점 초기화
    'q' : 종료

중요: 카메라를 최종 설치 위치에 '고정한 뒤' 캘리브레이션할 것. 카메라를 움직이거나 프레임
해상도를 바꾸면 zone 좌표와 호모그래피가 모두 무효가 되어 다시 찍어야 한다.

사용법:
    python tools/zone_calibrator.py --source 0
    python tools/zone_calibrator.py --source path/to/sample.jpg --n 5
    python tools/zone_calibrator.py --source 0 --width-cm 90 --length-cm 300
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import config
from src.zone import CrosswalkZones

_points = []
# OpenCV의 putText는 한글을 그리지 못하고 '???'로 표시하므로 화면 라벨은 영문으로 둔다.
# (1) 시작-왼쪽 (2) 시작-오른쪽 (3) 끝-오른쪽 (4) 끝-왼쪽
_LABELS = ["1) START-LEFT", "2) START-RIGHT", "3) END-RIGHT", "4) END-LEFT"]


def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(_points) < 4:
        _points.append((x, y))


def _grab_reference_frame(source):
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass  # 파일 경로 또는 URL인 경우

    if isinstance(source, str) and Path(source).exists():
        frame = cv2.imread(source)
        if frame is None:
            raise RuntimeError(f"이미지를 읽을 수 없습니다: {source}")
        return frame

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"카메라/영상을 열 수 없습니다: {source}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("프레임을 읽을 수 없습니다.")
    return frame


def _draw_preview(frame, n_zones):
    """현재 클릭된 점과, 4점이 다 찍혔으면 N등분 미리보기를 그린다."""
    display = frame.copy()
    for i, p in enumerate(_points):
        cv2.circle(display, p, 5, (0, 0, 255), -1)
        cv2.putText(display, _LABELS[i], (p[0] + 8, p[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if len(_points) == 4:
        zones = CrosswalkZones.from_quad(_points, n=n_zones)
        for idx, zone in enumerate(zones.zones, start=1):
            pts = np.array(zone.points, dtype=np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 0), 2)
            cx = int(sum(x for x, _ in zone.points) / len(zone.points))
            cy = int(sum(y for _, y in zone.points) / len(zone.points))
            cv2.putText(display, str(idx), (cx - 5, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        hint = _LABELS[len(_points)] if len(_points) < 4 else ""
        cv2.putText(display, f"CLICK: {hint}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return display


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=config.CAMERA_SOURCE, help="카메라 인덱스 / 이미지·영상 경로 / 스트림 URL")
    parser.add_argument("--out", default=config.ZONE_CONFIG_PATH, help="저장 경로")
    parser.add_argument("--name", default="crosswalk", help="zone 이름")
    parser.add_argument("--n", type=int, default=config.CROSSWALK_ZONE_COUNT, help="구역 개수(기본 5)")
    parser.add_argument("--width-cm", type=float, default=config.CROSSWALK_REAL_WIDTH_CM,
                        help="횡단보도 실제 폭(cm) — 걷는 방향과 수직인 변. 속도를 cm/s로 내려면 필요.")
    parser.add_argument("--length-cm", type=float, default=config.CROSSWALK_REAL_LENGTH_CM,
                        help="횡단보도 실제 길이(cm) — 걷는 방향 변. 속도를 cm/s로 내려면 필요.")
    args = parser.parse_args()

    if args.width_cm is None or args.length_cm is None:
        print(
            "[안내] --width-cm / --length-cm 가 없어 호모그래피는 저장되지 않습니다.\n"
            "       구역 판정은 정상 동작하지만 속도는 px/s로만 나옵니다(원근 보정 없음).\n"
            "       횡단보도 모형을 실측한 뒤 다시 실행하면 cm/s로 낼 수 있습니다."
        )

    frame = _grab_reference_frame(args.source)

    window = "Zone Calibrator (click 4 corners, s: save, r: reset, q: quit)"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, _on_mouse)

    while True:
        cv2.imshow(window, _draw_preview(frame, args.n))
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            _points.clear()
        elif key == ord("s"):
            if len(_points) != 4:
                print("네 꼭짓점을 모두 클릭해야 저장할 수 있습니다.")
                continue
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"name": args.name, "corners": _points, "n_zones": args.n}
            if args.width_cm is not None and args.length_cm is not None:
                payload["real_width_cm"] = args.width_cm
                payload["real_length_cm"] = args.length_cm
            # 캘리브레이션 당시 해상도를 남긴다 — 운영 해상도가 다르면 좌표가 어긋나므로,
            # 나중에 "왜 안 맞지?" 를 추적할 수 있어야 한다.
            payload["frame_size"] = [int(frame.shape[1]), int(frame.shape[0])]
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            homography = "cm/s 가능" if "real_width_cm" in payload else "치수 없음 -> px/s"
            print(
                f"저장 완료: {out_path} (corners=4, n_zones={args.n}, "
                f"해상도={payload['frame_size'][0]}x{payload['frame_size'][1]}, 속도: {homography})"
            )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
