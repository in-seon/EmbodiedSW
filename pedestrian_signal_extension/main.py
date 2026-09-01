import argparse
import sys
import time

from config import config



class _NullSerial:
   
    ready = True
    port = "(없음)"

    def update_state(self, state, zone=None, now=None):
        return None

    def send_state(self, state, zone=None, now=None):
        return None

    def poll(self):
        return []

    def ping(self):
        return True

    def open(self):
        return self

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


def _make_serial(args):
    if args.no_serial:
        print("[안내] --no-serial: 아두이노로 보내지 않고 판정만")
        return _NullSerial()
    from src.serial_comm import SerialComm

    print("[안내] 아두이노가 리셋되어 부팅할 때까지 잠시 대기")
    return SerialComm(port=args.port)


def _load_zones(required: bool):

    from src.zone import CrosswalkZones

    if required:
        try:
            return CrosswalkZones.load()
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"횡단보도 구역 설정이 없습니다: {exc.filename}\n"
                "  캘리브레이션을 먼저 하세요 "
            ) from exc
    try:
        zones = CrosswalkZones.load()
        print(" zone 설정을 찾았습니다 — 캘리브레이션된 꼭짓점으로 ROI를 잡습니다.")
        return zones
    except FileNotFoundError:
        print(" zone 설정이 없어 화면 비율(FALL_CONFIG['crosswalk_roi'])로 ROI를 잡습니다.")
    except (ValueError, KeyError, OSError) as exc:
        print(" zone 설정을 읽었지만 쓸 수 없습니다. 화면 비율로 ROI를 잡고 계속합니다.")
        print(f" 사유: {exc}")
    return None


class _Reporter:
    def __init__(self, display):
        self.display = display
        self.t0 = time.monotonic()
        self.frames = 0
        self.fps = 0.0
        self.last_key = None

    def tick(self, key, text, force=False):
        self.frames += 1
        now = time.monotonic()
        second_passed = now - self.t0 >= 1.0
        if second_passed:
            self.fps = self.frames / (now - self.t0)
            self.t0, self.frames = now, 0
        if force or second_passed or key != self.last_key:
            print(f"FPS {self.fps:4.1f} | {text}", flush=True)
            self.last_key = key


def _describe(state, zone, result_line):

    from src.serial_comm import STATE_FALL, STATE_NORMAL, STATE_ZONE
    if state == STATE_FALL:
        mark = "!! FALL !!"
    elif state == STATE_NORMAL:
        mark = "정상"
    elif state == STATE_ZONE:
        mark = f"zone{zone} (진척도)"
    else:
        mark = state
    return f"{mark}{result_line}"


def run_fall_mode(args):
    from src.capture import CameraCapture
    from src.detection import PersonDetector
    from src.pipeline import FallAlarmPipeline
    from src.serial_comm import STATE_FALL, STATE_NORMAL

    zones = _load_zones(required=False)
    pipeline = FallAlarmPipeline(
        camera=CameraCapture(source=args.source),
        detector=PersonDetector(),
        serial_comm=_make_serial(args),
        zones=zones,
    )
    print(f" 카메라 백엔드: {pipeline.camera.backend_name} (source={args.source!r})")
    print(" 종료: " + ("창에서 q" if args.display else "Ctrl+C") + "\n")

    reporter = _Reporter(args.display)

    def on_result(result, frame):
        extra = f" (id={sorted(result.confirmed_ids)})" if result.confirmed_ids else ""
        if result.command_sent:
            extra += f"  -> {result.command_sent}"
        text = f"사람 {result.people_count}명 | " + _describe(
            STATE_FALL if result.fall_confirmed else STATE_NORMAL, None, extra)
        reporter.tick(result.fall_confirmed, text)
        if args.display:
            return _draw(frame, text, result.fall_confirmed, pipeline)
        return True

    _drive(pipeline, on_result)


def run_full_mode(args):
    from src.capture import CameraCapture
    from src.detection import PersonDetector
    from src.pipeline import CombinedPipeline, SignalExtensionPipeline, _NoSend

    from src.speed import SpeedEstimator
    from src.zone import CrosswalkOccupancy

    if args.confirm_frames is None:
        raise NotImplementedError(
            "ZONE_RESIDENCY_FRAMES가 아직 미정입니다." )

    zones = _load_zones(required=True)
    camera = CameraCapture(source=args.source)
    detector = PersonDetector()
    extension = SignalExtensionPipeline(
        camera=camera, detector=detector, zones=zones,
        occupancy=CrosswalkOccupancy(zones, confirm_frames=args.confirm_frames),
        serial_comm=_NoSend(),
        speed_estimator=SpeedEstimator(ground_plane=zones.ground_plane),
    )
    pipeline = CombinedPipeline(
        camera=camera,
        detector=detector,
        serial_comm=_make_serial(args),
        extension=extension,
        zones=zones,
    )
    print(f" 카메라 백엔드: {pipeline.camera.backend_name} (source={args.source!r})")
    print(f" 속도 단위: {pipeline.extension.speed.unit}  ")
    print(" 종료: " + ("창에서 q" if args.display else "Ctrl+C") + "\n")

    reporter = _Reporter(args.display)

    def on_result(result, frame):
        ext = result.extension
        extra = ""
        if ext.occupied_zones:
            extra += f" zone={ext.occupied_zones}"
        if ext.eta_sec is not None:
            extra += f" ETA={ext.eta_sec:.1f}s"  
        if result.line_sent:
            extra += f"  -> {result.line_sent}"
        text = (f"사람 {len(ext.pedestrians)}명 | "
                + _describe(result.state, result.zone, extra))
        reporter.tick((result.state, result.zone), text)
        if args.display:
            return _draw(frame, text, result.fall.fall_confirmed, pipeline)
        return True

    _drive(pipeline, on_result)


def _drive(pipeline, on_result):
    try:
        pipeline.run(on_result=on_result)
    except KeyboardInterrupt:
        print("\n[안내] 중단됨 (Ctrl+C).")
  


def _draw(frame, text, alarm, pipeline):
    import cv2

    roi = getattr(pipeline, "roi_px", None) or getattr(
        getattr(pipeline, "fall", None), "roi_px", None)
    if roi:
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)

    color = (0, 0, 255) if alarm else (0, 255, 0)
    ascii_text = text.encode("ascii", "replace").decode("ascii")
    cv2.putText(frame, ascii_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.imshow("Smart Crosswalk (q: quit, r: reset alarm)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        return False
    if key == ord("r"):
        pipeline.reset_alarm()
        print(" 알람을 수동으로 해제했습니다.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("full", "fall"), default="full",
                        help="full = 신호 연장 + 쓰러짐 감지(기본, zone 설정 필요) / "
                             "fall = 쓰러짐 감지만(zone 설정 없이도 동작)")
    parser.add_argument("--source", default=None,
                        help="카메라 소스. 생략하면 config.CAMERA_SOURCE.")
    parser.add_argument("--port", default=None,
                        help="아두이노 시리얼 포트. 생략하면 config.SERIAL_PORT, 그것도 없으면 자동 탐색.")
    parser.add_argument("--display", action="store_true",
                        help="창을 띄운다. 모니터가 없으면 쓰지 말 것(기본은 stdout 출력).")
    parser.add_argument("--no-serial", action="store_true",
                        help="아두이노 없이 판정만 확인한다.")
    parser.add_argument("--confirm-frames", type=int, default=config.ZONE_RESIDENCY_FRAMES,
                        help="'확정 보행자'로 보기까지 필요한 연속 검출 프레임 수. ")
    args = parser.parse_args()

    if args.source is None:
        args.source = config.CAMERA_SOURCE

    from src.capture import normalize_source
    args.source = normalize_source(args.source)

    if args.mode == "fall":
        run_fall_mode(args)
    else:
        run_full_mode(args)


if __name__ == "__main__":
    try:
        main()
    except (NotImplementedError, FileNotFoundError, RuntimeError) as exc:
        print(f"\n실행할 수 없습니다:\n  {exc}\n", file=sys.stderr)
        sys.exit(1)
