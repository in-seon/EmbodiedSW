# 스마트 횡단보도 보행자 안전 시스템 (MiruDa)

제24회 임베디드 소프트웨어 경진대회 출품작

## 개요
횡단보도 보행자를 실시간 검출해 잔류 시 보행 신호를 연장하고,
쓰러짐이 확정되면 사이렌 및 자동신고를 발동하는 시스템.

## 실행
pip install -r requirements.txt
python crosswalk_poc.py --source usb # USB 웹캠
python crosswalk_poc.py --source csi # Pi CSI 카메라
python crosswalk_poc.py --source 영상.mp4 # 영상 파일

## 라즈베리파이 CSI 카메라
sudo apt install -y python3-picamera2
python -m venv --system-site-packages venv

## 데이터셋
`bash download_urfall.sh` 로 UR Fall Detection Dataset 다운로드

## 모델 가중치
최초 실행 시 yolov8n-pose.pt 자동 다운로드 (저장소에 미포함)

## 팀 구성
(팀원 이름 / 역할)

## License
AGPL-3.0 — Ultralytics YOLO(AGPL-3.0) 의존성에 따름
