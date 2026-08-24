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
    python tools/manual_camera_person_check.py --no-display    # 창 없이 측정값만 출력(헤드리스)

    'q' : 종료 (--no-display 일 때는 Ctrl+C)

## 실측 FPS를 잴 때는 --no-display 를 쓸 것

모니터 없이 `ssh -X` 로 창을 띄우면 프레임을 네트워크로 보내는 시간이 루프에 그대로
포함된다. 루프가 `추론 -> 그리기 -> imshow -> waitKey` 로 직렬이라, 그때 보이는 FPS는
파이의 실제 처리 성능이 아니라 X11 전송 속도에 눌린 하한값이다.
`--no-display` 는 그리기와 imshow를 건너뛰고 1초에 한 번만 측정값을 stdout으로 찍으므로
(매 프레임 출력하면 그 출력이 다시 루프를 느리게 만든다) 순수 추론 성능을 볼 수 있다.

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
from src.detection import MobilityAidDetector, PersonDetector
from src.speed import SpeedEstimator
from src.zone import CrosswalkOccupancy, CrosswalkZones

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
MAGENTA = (255, 0, 255)   # 교통약자 보조 검출 결과
FONT = cv2.FONT_HERSHEY_SIMPLEX


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_still_image(source) -> bool:
    """정지 이미지 경로인가. 맞으면 한 장을 붙들고 반복 표시한다.

    후보 가중치가 우리 휠체어 모형을 잡는지 사진 한 장으로 확인할 때 쓴다.
    cv2.VideoCapture는 이미지 파일을 한 번만 돌려주고 그다음엔 None이라, 이 처리가 없으면
    창이 뜨자마자 닫힌다.
    """
    return isinstance(source, str) and Path(source).suffix.lower() in _IMAGE_SUFFIXES


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
    parser.add_argument("--no-display", action="store_true",
                        help="창을 띄우지 않고 측정값만 1초마다 stdout으로 출력한다(헤드리스). "
                             "모니터 없이 SSH로 접속한 파이에서 실측 FPS를 잴 때 쓴다. "
                             "ssh -X로 창을 띄우면 프레임을 네트워크로 보내는 시간이 루프에 "
                             "포함돼 FPS가 실제보다 낮게 나오므로, 성능 측정에는 이 옵션을 쓸 것. "
                             "종료는 Ctrl+C.")
    parser.add_argument("--confirm-frames", type=int, default=config.ZONE_RESIDENCY_FRAMES or 3,
                        help="'확정 보행자'로 보기까지 필요한 연속 검출 프레임 수. "
                             "config.ZONE_RESIDENCY_FRAMES가 아직 미정이라 화면 확인용 기본값 3을 쓴다.")
    parser.add_argument("--aid-model", default=config.MOBILITY_AID_MODEL_PATH,
                        help="교통약자(휠체어/목발) 보조 모델 가중치 경로. "
                             "주면 자주색 박스로 함께 표시한다. 후보 가중치가 우리 모형을 "
                             "실제로 잡는지 확인하는 용도.")
    parser.add_argument("--aid-every", type=int, default=config.MOBILITY_AID_EVERY_N_FRAMES,
                        help="보조 모델 추론 주기(프레임). 1이면 매 프레임(비쌈).")
    parser.add_argument("--aid-conf", type=float, default=config.MOBILITY_AID_CONFIDENCE_THRESHOLD,
                        help="보조 모델 신뢰도 임계값. 모형이 안 잡히면 낮춰가며 확인.")
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
    print(f"[안내] 사람 검출 모델: {detector.model_path} (imgsz={detector.imgsz}, "
          f"conf={detector.confidence_threshold})")

    # 교통약자 보조 모델. 경로가 없으면 enabled=False로 조용히 비활성된다.
    aid = MobilityAidDetector(model_path=args.aid_model, every_n_frames=max(1, args.aid_every),
                              confidence_threshold=args.aid_conf)
    if aid.enabled:
        print(f"[안내] 교통약자 보조 모델: {aid.model_path} "
              f"({args.aid_every}프레임마다 1회, imgsz={aid.imgsz}, conf={aid.confidence_threshold})")
        print(f"       이 모델이 아는 클래스: {aid.class_names}")
        print("       * 자주색 박스가 보조 모델 결과다. 휠체어/목발 모형을 화면에 넣고 "
              "잡히는지 확인할 것.")
        if not config.MOBILITY_AID_LABELS:
            print("       * config.MOBILITY_AID_LABELS가 비어 있어 모델의 모든 클래스를 표시한다. "
                  "쓸 클래스명을 확인한 뒤 config에 채우면 된다.")
    else:
        print("[안내] 교통약자 보조 모델 없음 (--aid-model 또는 config.MOBILITY_AID_MODEL_PATH로 지정).")

    # 카메라 여는 로직은 src/capture.py 한 곳에 있다.
    # 파이 CSI 카메라는 --source picamera2, 그 외(USB 웹캠/영상/스트림)는 cv2가 처리한다.
    camera = CameraCapture(source=source).open()
    print(f"[안내] 카메라 백엔드: {camera.backend_name}  (source={source!r})")

    show = not args.no_display
    if show:
        print("카메라 창에서 'q'를 누르면 종료합니다.")
    else:
        print("[안내] --no-display: 창 없이 1초마다 측정값을 출력합니다. 종료는 Ctrl+C.")

    fps_t0, fps_frames, fps = time.monotonic(), 0, 0.0

    still = None          # 정지 이미지 소스면 첫 프레임을 붙들고 반복한다(모형 사진 확인용)
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

            # 교통약자 보조 검출 (저빈도). 비활성이면 빈 목록.
            aid_boxes = aid.detect(frame)

            confirmed = occupancy.update(detections) if occupancy else {}
            speeds = speed.update_many(detections, now)

            if show and zones is not None:
                _draw_zones(frame, zones)

            if show:
                for ab in aid_boxes:
                    cv2.rectangle(frame, (int(ab.x1), int(ab.y1)), (int(ab.x2), int(ab.y2)), MAGENTA, 2)
                    cv2.putText(frame, f"{ab.label} {ab.confidence:.2f}",
                                (int(ab.x1), max(int(ab.y1) - 6, 12)), FONT, 0.5, MAGENTA, 2)

            # 사람별 측정값은 화면 표시 여부와 무관하게 만든다. 헤드리스에서는 이 문자열을
            # 그대로 stdout으로 내보내므로, 창으로 보는 것과 같은 값을 보게 된다.
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
                person_lines.append(" ".join(lines))

                if show:
                    cv2.rectangle(frame, (int(box.x1), int(box.y1)),
                                  (int(box.x2), int(box.y2)), color, 2)
                    # 위치 판정의 기준점 — 박스 중심이 아니라 하단 모서리 중심임을 눈으로 확인.
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
            # 라즈베리파이 실측 FPS는 이 값으로 확인한다 (CLAUDE.md 2.6 — 추정치 기록 금지).
            hud = f"FPS {fps:.1f} | {camera.backend_name} | unit {speed.unit}"
            if aid.enabled:
                hud += f" | aid {len(aid_boxes)} (x{aid.inference_count})"

            if not show:
                # 1초에 한 번만 찍는다. 매 프레임 출력하면 그 자체가 루프를 느리게 만들어
                # 측정하려는 FPS를 왜곡한다.
                if fps_updated:
                    summary = f"{hud} | 사람 {len(boxes)}명"
                    if occupancy is not None:
                        summary += f" (확정 {len(confirmed)}, ID없음 {occupancy.untracked_count})"
                    print(summary, flush=True)
                    for line in person_lines:
                        print(f"    {line}", flush=True)
                    if aid_boxes:
                        print("    PRIORITY (mobility aid detected)", flush=True)
                continue

            cv2.putText(frame, hud, (10, 22), FONT, 0.6, WHITE, 2)
            if aid_boxes:
                cv2.putText(frame, "PRIORITY (mobility aid detected)", (10, 44),
                            FONT, 0.6, MAGENTA, 2)

            cv2.imshow("Detection + Zone + Speed (q: quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        # --no-display 모드의 정상 종료 경로. 카메라를 반드시 놓아주고 나간다
        # (Picamera2는 해제하지 않으면 다음 실행에서 열리지 않는 경우가 있다).
        print("\n[안내] 중단됨 (Ctrl+C).")
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
