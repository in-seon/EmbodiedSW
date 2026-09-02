"""main.py 진입점 — 인자 파싱과 모드 분기."""

import pytest

import main as main_module
from config import config


@pytest.fixture
def dispatch(monkeypatch):
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
