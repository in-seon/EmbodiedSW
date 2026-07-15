# 스마트 신호등 — 보행자 잔류 감지 기반 신호 연장

임베디드 SW 경진대회 "스마트 신호등" 프로젝트 중 담당 파트(목표 1)의 구현체.
전체 배경, 하드웨어 구성, 판단 로직 설계는 [CLAUDE.md](CLAUDE.md)를 참고.

## 폴더 구조

```
config/            팀과 미확정인 값 포함 전역 설정
src/
├── capture/       카메라 입력 처리
├── detection/     사람 / 교통약자(휠체어·지팡이) 검출 래퍼
├── zone/          zone 정의, 잔류 판단 로직
├── signal_extend/ 신호 연장 상태 머신
└── serial_comm/   라즈베리파이-아두이노 통신
arduino/           아두이노 펌웨어
tools/             zone 캘리브레이션 등 스크립트
tests/             단위 테스트
data/              테스트 영상, 캘리브레이션 데이터 (git 추적 제외)
docs/              설계 결정 로그, 팀 인터페이스 합의 사항
```

## 현재 상태

뼈대 코드 단계. 사람 검출 모델 선정, 신호 연장 상한, 시리얼 통신 프로토콜 등
핵심 파라미터가 아직 팀과 확정되지 않아 `config/config.py`에 `None`으로 남아 있다.
값이 없는 상태로 관련 모듈을 실행하면 `NotImplementedError`가 발생하도록 만들어
미확정 값을 임의로 추정해 쓰지 않게 했다. 미확정 목록은 [docs/team_interface.md](docs/team_interface.md) 참고.

## 설치

```bash
pip install -r requirements.txt
```

## 테스트

```bash
pytest
```

## Zone 캘리브레이션

```bash
python tools/zone_calibrator.py --source 0
```
