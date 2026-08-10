"""검출 + 구역 판정 + 속도 추정을 눈으로 확인하는 수동 스크립트(자동 테스트 아님).

pytest 자동 수집 대상이 아니도록 tools/ 에 둔다(카메라 창을 띄우는 대화형 스크립트).

시리얼(제어부) 없이 비전 파트만 돌린다. 화면에 표시하는 것:

  - 사람 bounding box 와 신뢰도
  - **발 위치(하단 모서리 중심)** — 위치 판정·속도 계산의 기준점 (CLAUDE.md 2.1)
  - 5구역 폴리곤과 각 사람이 몇 번 구역에 있는지
  - 트랙별 속도. zone 설정에 실측 치수가 있으면 cm/s, 없으면 px/s
  - 예상 통과 시간(cm/s 일 때만)

사용법:
    python tools/manual_camera_person_check.py                    # config.CAMERA_SOURCE 사용
    python tools/manual_camera_person_check.py --source picamera2 # 파이 CSI 카메라
    python tools/manual_camera_person_check.py --source 0         # PC USB 웹캠
    python tools/manual_camera_person_check.py --source path/to/test.mp4
    python tools/manual_camera_person_check.py --no-zones      # zone 설정 없이 검출만

    'q' : 종료

zone 설정(data/zone_config.json)이 없으면 자동으로 검출만 표시한다.
먼저 `python tools/zone_calibrator.py --source 0 --width-cm .. --length-cm ..` 를 돌릴 것.
"""

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


def _load_zones(use_zones):
    """zone 설정을 읽는다. 없거나 --no-zones면 None."""
    if not use_zones:
        return None
    path = Path(config.ZONE_CONFIG_PATH)
    if not path.exists():
        print(
            f"[안내] zone 설정이 없습니다: {path}\n"
            "       구역 판정 없이 검출만 표시합니다.\n"
            "       먼저 tools/zone_calibrator.py 로 횡단보도 네 꼭짓점을 찍어 저장하세요."
        )
        return None
    zones = CrosswalkZones.load()
    if zones.ground_plane is None:
        print(
            "[안내] zone 설정에 실측 치수(real_width_cm / real_length_cm)가 없습니다.\n"
            "       속도가 px/s로만 나오고 예상 통과 시간은 계산되지 않습니다.\n"
            "       zone_calibrator.py 를 --width-cm / --length-cm 와 함께 다시 실행하세요."
        )
    else:
        gp = zones.ground_plane
        print(f"[안내] 호모그래피 적용됨: {gp.width_cm:.0f}cm x {gp.length_cm:.0f}cm -> 속도 단위 cm/s")
    return zones


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
                        help="'picamera2'(파이 CSI) / 카메라 인덱스 / 영상 경로 / 스트림 URL")
    parser.add_argument("--no-zones", action="store_true", help="zone 판정 없이 검출만 표시")
    parser.add_argument("--confirm-frames", type=int, default=config.ZONE_RESIDENCY_FRAMES or 3,
                        help="'확정 보행자'로 보기까지 필요한 연속 검출 프레임 수. "
                             "config.ZONE_RESIDENCY_FRAMES가 아직 미정이라 화면 확인용 기본값 3을 쓴다.")
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

    # 카메라 여는 로직은 src/capture.py 한 곳에 있다.
    # 파이 CSI 카메라는 --source picamera2, 그 외(USB 웹캠/영상/스트림)는 cv2가 처리한다.
    camera = CameraCapture(source=source).open()
    print(f"[안내] 카메라 백엔드: {camera.backend_name}  (source={source!r})")

    print("카메라 창에서 'q'를 누르면 종료합니다.")
    fps_t0, fps_frames, fps = time.monotonic(), 0, 0.0

    while True:
        frame = camera.read_frame()
        if frame is None:
            print("프레임을 읽을 수 없습니다.")
            break

        now = time.monotonic()
        boxes = [b for b in detector.detect(frame) if b.is_pedestrian()]
        detections = [(b.track_id, b.foot_point()) for b in boxes]

        confirmed = occupancy.update(detections) if occupancy else {}
        speeds = speed.update_many(detections, now)

        if zones is not None:
            _draw_zones(frame, zones)

        for box in boxes:
            fx, fy = box.foot_point()
            zone_index = zones.locate((fx, fy)) if zones else None
            is_confirmed = box.track_id in confirmed
            color = GREEN if is_confirmed else YELLOW if zone_index else WHITE

            cv2.rectangle(frame, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color, 2)
            # 위치 판정의 기준점 — 박스 중심이 아니라 하단 모서리 중심임을 눈으로 확인.
            cv2.circle(frame, (int(fx), int(fy)), 5, RED, -1)

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

            y = max(int(box.y1) - 8, 14)
            for line in reversed(lines):
                cv2.putText(frame, line, (int(box.x1), y), FONT, 0.5, color, 2)
                y -= 16

        fps_frames += 1
        if now - fps_t0 >= 1.0:
            fps = fps_frames / (now - fps_t0)
            fps_t0, fps_frames = now, 0
        # 라즈베리파이 실측 FPS는 이 값으로 확인한다 (CLAUDE.md 2.6 — 추정치 기록 금지).
        cv2.putText(frame, f"FPS {fps:.1f} | {camera.backend_name} | unit {speed.unit}",
                    (10, 22), FONT, 0.6, WHITE, 2)

        cv2.imshow("Detection + Zone + Speed (q: quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
