"""쓰러짐 감지가 왜 안 울리는지 **어디서 끊기는지** 보는 진단 도구.

`main.py --mode fall`은 결론("정상" / "!! FALL !!")만 찍는다. 그래서 안 울릴 때
원인이 검출인지, ROI인지, 각도 임계값인지 구분할 수가 없다 — 세 곳 중 어디가
끊겨도 화면에는 똑같이 "정상"으로만 보인다.

이 도구는 판정 체인을 그대로 다시 밟으면서 **각 단계의 중간값**을 사람별로 찍는다:

    검출  ->  ROI 겹침  ->  쓰러짐 후보(각도 or 종횡비)  ->  시간 누적 확정
     ①          ②                  ③                          ④

읽는 법 (한 줄이 사람 한 명):

    #0 id=1 conf=0.71 box=(210,300)-(402,378) w/h=2.54
       각도=87.3° (어깨conf=0.62 엉덩이conf=0.55)   <- ③의 근거. None이면 종횡비 폴백
       ROI겹침=0.94 >= 0.30 OK                      <- ②
       => 후보=True  ROI안=True  ==> 이 사람은 확정 대상

  - 사람 줄이 **아예 안 나온다** -> ① 검출 실패. DETECTION_CONFIDENCE_THRESHOLD /
    DETECTION_IMGSZ / 조명·거리 문제지 쓰러짐 로직 문제가 아니다.
  - ROI겹침이 임계값 미만 -> ② 카메라 화각 대비 crosswalk_roi가 안 맞는다.
    화면에 표시되는 회색 사각형(--display)이 실제 횡단보도를 덮는지 볼 것.
  - 각도가 None -> 키포인트 신뢰도가 낮아 torso_angle_deg가 포기한 상태.
    이때는 종횡비(w/h > fall_aspect_ratio)로만 판정한다. 모형이 작으면 흔하다.
  - 각도가 나오는데 임계값 미만 -> ③ fall_angle_deg를 낮출 것.

사용:

    python tools/manual_fall_debug.py                      # config 기본 카메라
    python tools/manual_fall_debug.py --source test.mp4 --display

"""

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
        lines.append(f"     ③ 각도={ang:5.1f}° {'>' if ok else '<='} "
                     f"임계 {cfg['fall_angle_deg']}° -> 후보={ok}"
                     f"   (어깨conf={_kp_conf(person.keypoints, (5, 6)):.2f} "
                     f"엉덩이conf={_kp_conf(person.keypoints, (11, 12)):.2f})")
    else:
        ok = ratio > cfg["fall_aspect_ratio"]
        why = "키포인트 없음" if person.keypoints is None else (
            f"관절 신뢰도 낮음 (어깨={_kp_conf(person.keypoints, (5, 6)):.2f} "
            f"엉덩이={_kp_conf(person.keypoints, (11, 12)):.2f} < 0.3)")
        lines.append(f"     ③ 각도=None ({why}) -> 종횡비 폴백: "
                     f"{ratio:.2f} {'>' if ok else '<='} {cfg['fall_aspect_ratio']} -> 후보={ok}")

    lines.append(f"     ② ROI겹침={overlap:.2f} {'>=' if on_cw else '<'} "
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
        print("  ① 사람 검출 0 — 쓰러짐 로직까지 가지도 못한다. "
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
                        help="zone 설정이 있으면 그 꼭짓점으로 ROI를 잡는다 (main과 동일)")
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
            print(f"[경고] zone을 못 읽어 화면 비율로 갑니다: {exc}")

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
