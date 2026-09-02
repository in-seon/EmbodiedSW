import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import config
from src.capture import grab_one_frame
from src.zone import CrosswalkZones

_points = []
_LABELS = ["1) START-LEFT", "2) START-RIGHT", "3) END-RIGHT", "4) END-LEFT"]


def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(_points) < 4:
        _points.append((x, y))


def _grab_reference_frame(source):
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass  
    return grab_one_frame(source)


def _draw_preview(frame, n_zones, width_cm=None, length_cm=None):
    display = frame.copy()
    for i, p in enumerate(_points):
        cv2.circle(display, p, 5, (0, 0, 255), -1)
        cv2.putText(display, _LABELS[i], (p[0] + 8, p[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if len(_points) == 4:
        zones = CrosswalkZones.from_quad(
            _points, n=n_zones, width_cm=width_cm, length_cm=length_cm
        )
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
            "--width-cm / --length-cm 가 없어 호모그래피는 저장되지 않습니다.\n" )

    frame = _grab_reference_frame(args.source)

    window = "Zone Calibrator (click 4 corners, s: save, r: reset, q: quit)"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, _on_mouse)

    while True:
        cv2.imshow(window, _draw_preview(frame, args.n, args.width_cm, args.length_cm))
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
