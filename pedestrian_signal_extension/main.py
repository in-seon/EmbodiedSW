"""실행 진입점 — 검출부터 제어부 전송까지 전체 실시간 루프.

    python main.py

검출 모델은 확정됐지만(yolov8n), 아래가 config에 채워져 있어야 실제로 돈다.
미확정이면 파이프라인 구성 단계에서 NotImplementedError로 "무엇을 먼저 확정해야 하는지"
알려주며 멈춘다(의도된 동작 — 임의값으로 몰래 동작하지 않게 하기 위함).

  - ZONE_RESIDENCY_FRAMES, REMAINING_TIME_THRESHOLD_SEC, MAX_TOTAL_EXTENSION_SEC,
    ZONE_EXTENSION_SEC의 2·3·4번
  - SERIAL_PORT, SERIAL_BAUDRATE (+ 메시지 포맷 구현)
  - data/zone_config.json (tools/zone_calibrator.py 로 생성)

시리얼 없이 비전 파트만 확인하려면 tools/manual_camera_person_check.py 를 쓴다.
"""

from src.pipeline import SignalExtensionPipeline


def main():
    pipeline = SignalExtensionPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
