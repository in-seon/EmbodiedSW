"""라즈베리파이가 내보내는 스트림에서 사람 검출이 되는지 확인하는 수동 스크립트(자동 테스트 아님).

배치 B(카메라는 파이, 추론은 PC — CLAUDE.md 7장)에서 "스트림이 PC까지 잘 오는가"만 빠르게
확인하는 용도다. 구역 판정이나 속도는 표시하지 않는다. 그건
tools/manual_camera_person_check.py 를 쓴다.

사용법:
    python tools/manual_stream_person_check.py --source tcp://192.168.0.15:8888
    python tools/manual_stream_person_check.py --source http://192.168.0.15:8000/

    'q' : 종료

파이 IP는 네트워크마다 다르므로 인자로 받는다(하드코딩하면 다른 환경에서 그대로 실패한다).
"""

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import config

PERSON_CLASS_ID = 0  # COCO 데이터셋 기준 "person" 클래스 ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True,
        help="파이 스트림 주소 (예: tcp://<파이IP>:8888 또는 http://<파이IP>:8000/)",
    )
    parser.add_argument("--model", default=config.DETECTION_MODEL_PATH,
                        help="YOLO 가중치 경로 (기본: config.DETECTION_MODEL_PATH)")
    args = parser.parse_args()

    model = YOLO(args.model)

    print(f"스트림 연결 중: {args.source}  ('q'로 종료)")
    for result in model.predict(source=args.source, stream=True,
                                classes=[PERSON_CLASS_ID], verbose=False):
        cv2.imshow("Stream Person Detection (q: quit)", result.plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
