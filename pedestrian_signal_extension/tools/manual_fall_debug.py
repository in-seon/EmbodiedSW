import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import config
from src.capture import CameraCapture, grab_one_frame
from src.detection import PersonDetector
from src.fall_detection import (
    FallDetectionPipeline, foot_in_roi, roi_from_ratio, roi_from_zones,
    roi_overlap_ratio, torso_angle_deg,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _kp_conf(kp, idx):
    if kp is None:
        return None
    return float(min(kp[i][2] for i in idx))


def _describe(person, cfg, roi_px):
    x1, y1, x2, y2 = person.bbox
    w, h = x2 - x1, y2 - y1
    ratio = (w / h) if h > 0 else 0.0
    ang = torso_angle_deg(person.keypoints)
    overlap = roi_overlap_ratio(person, roi_px)
    on_cw = overlap >= cfg["fall_roi_overlap"]

    lines = [f"  id={person.track_id} conf={person.conf:.2f} "
             f"box=({x1},{y1})-({x2},{y2}) w/h={ratio:.2f}"]

    if ang is not None:
        ok = ang > cfg["fall_angle_deg"]
        lines.append(f"     각도={ang:5.1f}° {'>' if ok else '<='} "
                     f"임계 {cfg['fall_angle_deg']}° -> 후보={ok}"
                     f"   (어깨conf={_kp_conf(person.keypoints, (5, 6)):.2f} "
                     f"엉덩이conf={_kp_conf(person.keypoints, (11, 12)):.2f})")
    else:
        ok = ratio > cfg["fall_aspect_ratio"]
        why = "키포인트 없음" if person.keypoints is None else (
            f"관절 신뢰도 낮음 (어깨={_kp_conf(person.keypoints, (5, 6)):.2f} "
            f"엉덩이={_kp_conf(person.keypoints, (11, 12)):.2f} < 0.3)")
        lines.append(f"     각도=None ({why}) -> 종횡비 폴백: "
                     f"{ratio:.2f} {'>' if ok else '<='} {cfg['fall_aspect_ratio']} -> 후보={ok}")

    lines.append(f"     ROI겹침={overlap:.2f} {'>=' if on_cw else '<'} "
                 f"{cfg['fall_roi_overlap']} -> ROI안={on_cw}   (발위치 ROI안={foot_in_roi(person, roi_px)})")
    lines.append(f"     => 확정 대상={'예' if (ok and on_cw) else '아니오'}")
    return lines


def _draw(frame, result, roi_px, cfg):
    import cv2

    cv2.rectangle(frame, roi_px[:2], roi_px[2:], (120, 120, 120), 1)
    for person, fallen in zip(result["people"], result["fallen_flags"]):
        x1, y1, x2, y2 = person.bbox
        confirmed = person.track_id in result["confirmed_ids"]
        color = (0, 0, 255) if confirmed else ((0, 165, 255) if fallen else (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        ang = torso_angle_deg(person.keypoints)
        tag = f"id={person.track_id} " + (f"{ang:.0f}deg" if ang is not None
                                          else f"wh={(x2-x1)/max(1,y2-y1):.2f}")
        cv2.putText(frame, tag, (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        kp = person.keypoints
        if kp is not None and ang is not None:
            sh = tuple(int(v) for v in kp[[5, 6]][:, :2].mean(axis=0))
            hip = tuple(int(v) for v in kp[[11, 12]][:, :2].mean(axis=0))
            cv2.line(frame, sh, hip, color, 2)

    head = ("!! FALL !!" if result["fall_confirmed"] else "OK") + \
           f"  people={len(result['people'])}"
    cv2.putText(frame, head, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255) if result["fall_confirmed"] else (0, 255, 0), 2)
    cv2.imshow("fall debug (q: quit)", frame)
    return (cv2.waitKey(1) & 0xFF) != ord("q")


def _report(result, roi_px, cfg, frame_no, now):
    print(f"\n[frame {frame_no}] t={now:.2f}s  검출 {len(result['people'])}명  "
          f"확정={result['confirmed_ids'] or '없음'}")
    if not result["people"]:
        print(" 사람 검출 0 "
              f"(conf 임계={config.DETECTION_CONFIDENCE_THRESHOLD}, imgsz={config.DETECTION_IMGSZ})")
        return
    for person in result["people"]:
        for line in _describe(person, cfg, roi_px):
            print(line)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=None,
                        help="카메라/영상/이미지. 생략하면 config.CAMERA_SOURCE")
    parser.add_argument("--display", action="store_true", help="창을 띄워 박스·몸통축을 표시")
    parser.add_argument("--every", type=float, default=1.0,
                        help="콘솔 출력 최소 간격(초). 0이면 매 프레임 (기본 1.0)")
    parser.add_argument("--zones", action="store_true",
                        help="zone 설정이 있으면 그 꼭짓점을 ROI로")
    args = parser.parse_args()

    source = args.source if args.source is not None else config.CAMERA_SOURCE
    cfg = config.FALL_CONFIG

    roi_from = "화면 비율 FALL_CONFIG['crosswalk_roi']"
    zones = None
    if args.zones:
        from src.zone import CrosswalkZones
        try:
            zones = CrosswalkZones.load()
            roi_from = "zone 캘리브레이션"
        except Exception as exc:
            print(f"zone을 못 읽어 화면 비율 사용: {exc}")

    print(f"[설정] source={source!r}  모델={config.DETECTION_MODEL_PATH}")
    print(f"[설정] conf임계={config.DETECTION_CONFIDENCE_THRESHOLD} imgsz={config.DETECTION_IMGSZ}")
    print(f"[설정] fall_angle_deg={cfg['fall_angle_deg']} "
          f"fall_aspect_ratio={cfg['fall_aspect_ratio']} "
          f"fall_roi_overlap={cfg['fall_roi_overlap']} "
          f"fall_confirm_sec={cfg['fall_confirm_sec']}")

    detector = PersonDetector()

    def make_pipeline(frame):
        roi = roi_from_zones(zones) if zones is not None else roi_from_ratio(frame.shape)
        print(f"[설정] ROI={roi} (출처: {roi_from}, 프레임={frame.shape[1]}x{frame.shape[0]})\n")
        return FallDetectionPipeline(roi), roi

    is_image = (isinstance(source, str)
                and Path(source).suffix.lower() in IMAGE_SUFFIXES
                and Path(source).is_file())

    if is_image:
        frame = grab_one_frame(source)
        pipeline, roi = make_pipeline(frame)
        result = pipeline.update(detector.detect(frame), 0.0)
        _report(result, roi, cfg, 0, 0.0)
        if args.display:
            import cv2
            _draw(frame, result, roi, cfg)
            cv2.waitKey(0)
        return

    pipeline = roi = None
    last_print = 0.0
    frame_no = 0
    t0 = time.monotonic()
    try:
        with CameraCapture(source=source) as camera:
            print(f"[안내] 백엔드={camera.backend_name}  종료: Ctrl+C\n")
            for frame in camera.frames():
                if pipeline is None:
                    pipeline, roi = make_pipeline(frame)
                now = time.monotonic()
                result = pipeline.update(detector.detect(frame), now)
                if args.every <= 0 or now - last_print >= args.every:
                    _report(result, roi, cfg, frame_no, now - t0)
                    last_print = now
                frame_no += 1
                if args.display and not _draw(frame, result, roi, cfg):
                    break
    except KeyboardInterrupt:
        print("\n[안내] 중단됨.")


if __name__ == "__main__":
    main()
