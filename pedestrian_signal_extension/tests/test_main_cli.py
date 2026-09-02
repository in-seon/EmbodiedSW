"""main.py 진입점 — 인자 파싱과 모드 분기.

## 왜 이 파일이 필요한가

실제로 `--mode fall-only`이 `NameError`로 죽은 채 커밋된 적이 있다. 함수 이름을 바꾸면서
호출부 한 곳이 따라오지 않았는데, 테스트가 전부 `src/` 안쪽만 보고 있어서 아무도 못 잡았다.
파이프라인이 아무리 옳아도 **진입점이 부러지면 아무것도 못 돌린다.**

카메라도 모델도 없이 확인하려고, 실행 함수를 가짜로 갈아끼우고 '어디로 갔는가'만 본다.
"""

import pytest

import main as main_module
from config import config


@pytest.fixture
def dispatch(monkeypatch):
    """세 실행 함수를 가로채고, 호출된 이름과 args를 기록한다."""
    calls = []

    for name in ("run_full_mode", "run_fall_mode"):
        assert hasattr(main_module, name), f"{name} 이 사라졌다 — 분기가 부러진다"
        monkeypatch.setattr(
            main_module, name,
            lambda args, _n=name: calls.append((_n, args)),
        )
    monkeypatch.setattr("src.capture.normalize_source", lambda s: s)
    return calls


def run(dispatch, monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["main.py", *argv])
    main_module.main()
    assert len(dispatch) == 1, f"실행 함수가 {len(dispatch)}번 불렸다"
    return dispatch[0]


def test_default_mode_is_full(dispatch, monkeypatch):
    name, args = run(dispatch, monkeypatch)
    assert name == "run_full_mode"
    assert args.mode == "full"


def test_fall_mode_is_reachable(dispatch, monkeypatch):
    """★ 한때 함수 개명이 호출부에 반영되지 않아 NameError로 죽어 있던 경로."""
    name, args = run(dispatch, monkeypatch, "--mode", "fall")
    assert name == "run_fall_mode"
    assert args.mode == "fall"


def test_unknown_mode_is_rejected(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "--mode", "bogus"])
    with pytest.raises(SystemExit):
        main_module.main()


def test_no_serial_is_a_flag(dispatch, monkeypatch):
    _, args = run(dispatch, monkeypatch, "--no-serial")
    assert args.no_serial is True


def test_confirm_frames_defaults_to_config(dispatch, monkeypatch):
    _, args = run(dispatch, monkeypatch)
    assert args.confirm_frames == config.ZONE_RESIDENCY_FRAMES
