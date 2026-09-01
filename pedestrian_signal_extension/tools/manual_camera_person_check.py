import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import config
from src.capture import CameraCapture
from src.detection import PersonDetector
from src.speed import SpeedEstimator
from src.zone import CrosswalkOccupancy, CrosswalkZones

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_still_image(source) -> bool:
    return isinstance(source, str) and Path(source).suffix.lower() in _IMAGE_SUFFIXES


def _load_zones(use_zones):
    """zone 설정을 읽는다. 없거나 --no-zones면 None."""
    if not use_zones:
        return None
    path = Path(config.ZONE_CONFIG_PATH)
    if not path.exists():
        print(f"zone 설정이 없습니다: {path}")
        return None
    zones = CrosswalkZones.load()
    if zones.ground_plane is None:
        print("zone 설정에 실측 치수(real_width_cm / real_length_cm)가 없습니다.")
    else:
        gp = zones.ground_plane
        print(f"호모그래피 적용됨: {gp.width_cm:.0f}cm x {gp.length_cm:.0f}cm -> 속도 단위 cm/s")
    return zones


def _fall_debug_line(box):
    from src.fall_detection import looks_fallen, person_from_box, torso_angle_deg

    person = person_from_box(box)
    angle = torso_angle_deg(person.keypoints)

    kp = person.keypoints
    if kp is not None and len(kp) >= 13:
        shoulder = float(kp[[5, 6]][:, 2].min())
        hip = float(kp[[11, 12]][:, 2].min())
        kp_text = f"{shoulder:.2f}/{hip:.2f}"
    else:
        kp_text = "none"

    width, height = box.x2 - box.x1, box.y2 - box.y1
    ratio = (width / height) if height > 0 else 0.0

    fallen = looks_fallen(person, config.FALL_CONFIG)
    path = "angle" if angle is not None else "ratio"
    verdict = f"FALLEN({path})" if fallen else "ok"

    angle_text = f"{angle:5.1f}d" if angle is not None else "  -  "
    return f"ang={angle_text} kp={kp_text} wh={ratio:.2f} {verdict}"


def _draw_zones(frame, zones):
    for idx, zone in enumerate(zones.zones, start=1):
        pts = np.array(zone.points, dtype=np.int32)
        cv2.polylines(frame, [pts], True, (120, 120, 120), 1)
        cx = int(sum(x for x, _ in zone.points) / len(zone.points))
        cy = int(sum(y for _, y in zone.points) / len(zone.points))
        cv2.putText(frame, str(idx), (cx - 5, cy + 5), FONT, 0.6, (120, 120, 120), 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=config.CAMERA_SOURCE,
                        help="'picamera2'(파이 CSI) or USB 웹캠/영상. 기본값 config.CAMERA_SOURCE")
    parser.add_argument("--no-zones", action="store_true", help="zone 판정 없이 검출만 표시")
    parser.add_argument("--fall", action="store_true",
                        help="쓰러짐 판정에 들어가는 값(몸통 각도, 키포인트 신뢰도, bbox 비율)을 함께 표시")
    parser.add_argument("--no-display", action="store_true",
                        help="창을 띄우지 않고 측정값만 1초마다 stdout으로 출력한다(헤드리스). ")
    parser.add_argument("--confirm-frames", type=int, default=config.ZONE_RESIDENCY_FRAMES or 3,
                        help="'확정 보행자'로 보기까지 필요한 연속 검출 프레임 수")
    args = parser.parse_args()

    try:
        source = int(args.source)
    except (ValueError, TypeError):
        source = args.source

    zones = _load_zones(not args.no_zones)
    ground_plane = zones.ground_plane if zones is not None else None
    occupancy = CrosswalkOccupancy(zones, confirm_frames=args.confirm_frames) if zones else None
    speed = SpeedEstimator(ground_plane=ground_plane)
    detector = PersonDetector()
    print(f"사람 검출 모델: {detector.model_path} (imgsz={detector.imgsz}, "
          f"conf={detector.confidence_threshold})")

    camera = CameraCapture(source=source).open()
    print(f"카메라 백엔드: {camera.backend_name}  (source={source!r})")

    show = not args.no_display
    if show:
        print("카메라 창에서 'q'를 누르면 종료합니다.")
    else:
        print("--no-display: 창 없이 1초마다 측정값을 출력합니다. 종료는 Ctrl+C.")

    fps_t0, fps_frames, fps = time.monotonic(), 0, 0.0

    still = None          
    try:
        while True:
            frame = still if still is not None else camera.read_frame()
            if frame is None:
                print("프레임을 읽을 수 없습니다.")
                break
            if still is None and _is_still_image(source):
                still = frame
            frame = frame.copy() if still is not None else frame

            now = time.monotonic()
            boxes = [b for b in detector.detect(frame) if b.is_pedestrian()]
            detections = [(b.track_id, b.foot_point()) for b in boxes]

            confirmed = occupancy.update(detections) if occupancy else {}
            speeds = speed.update_many(detections, now)

            if show and zones is not None:
                _draw_zones(frame, zones)

            person_lines = []
            for box in boxes:
                fx, fy = box.foot_point()
                zone_index = zones.locate((fx, fy)) if zones else None
                is_confirmed = box.track_id in confirmed
                color = GREEN if is_confirmed else YELLOW if zone_index else WHITE

                tid = box.track_id if box.track_id is not None else "-"
                lines = [f"id={tid} conf={box.confidence:.2f}"]
                lines.append(
                    f"zone={zone_index if zone_index else '-'}"
                    f"{' CONFIRMED' if is_confirmed else ''}"
                )
                ts = speeds.get(box.track_id)
                if ts is not None:
                    arrow = "^" if ts.direction > 0 else "v" if ts.direction < 0 else "-"
                    lines.append(f"speed={ts.crossing_speed:.1f} {ts.unit} {arrow}")
                    eta = speed.estimated_crossing_time_sec(box.track_id)
                    if eta is not None:
                        lines.append(f"ETA={eta:.1f}s")
                else:
                    lines.append("speed=...")
                if args.fall:
                    lines.append(_fall_debug_line(box))
                person_lines.append(" ".join(lines))

                if show:
                    cv2.rectangle(frame, (int(box.x1), int(box.y1)),
                                  (int(box.x2), int(box.y2)), color, 2)
                    cv2.circle(frame, (int(fx), int(fy)), 5, RED, -1)
                    y = max(int(box.y1) - 8, 14)
                    for line in reversed(lines):
                        cv2.putText(frame, line, (int(box.x1), y), FONT, 0.5, color, 2)
                        y -= 16

            fps_frames += 1
            fps_updated = now - fps_t0 >= 1.0
            if fps_updated:
                fps = fps_frames / (now - fps_t0)
                fps_t0, fps_frames = now, 0
            hud = f"FPS {fps:.1f} | {camera.backend_name} | unit {speed.unit}"

            if not show:
                if fps_updated:
                    summary = f"{hud} | 사람 {len(boxes)}명"
                    if occupancy is not None:
                        summary += f" (확정 {len(confirmed)}, ID없음 {occupancy.untracked_count})"
                    print(summary, flush=True)
                    for line in person_lines:
                        print(f"    {line}", flush=True)
                continue

            cv2.putText(frame, hud, (10, 22), FONT, 0.6, WHITE, 2)

            cv2.imshow("Detection + Zone + Speed (q: quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n 중단됨 (Ctrl+C).")
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
