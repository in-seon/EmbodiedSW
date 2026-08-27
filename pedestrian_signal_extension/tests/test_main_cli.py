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

    for name in ("run_full_mode", "run_fall_only_mode"):
        assert hasattr(main_module, name), f"{name} 이 사라졌다 — 분기가 부러진다"
        monkeypatch.setattr(
            main_module, name,
            lambda args, _n=name: calls.append((_n, args)),
        )
    # 카메라 소스 정규화는 실제 백엔드를 건드리므로 통과시킨다.
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


def test_fall_mode_runs_the_full_pipeline(dispatch, monkeypatch):
    """--mode fall 은 '쓰러짐만'이 아니라 **전체 + 서보 연출**이다.

    모형에 달린 두 모터가 맞물려 도는 시연 모드라, 신호 연장도 같이 돌아야 한다.
    """
    name, args = run(dispatch, monkeypatch, "--mode", "fall")
    assert name == "run_full_mode"
    assert args.mode == "fall"


def test_fall_only_mode_is_reachable(dispatch, monkeypatch):
    """★ 실제로 NameError로 죽어 있던 경로."""
    name, args = run(dispatch, monkeypatch, "--mode", "fall-only")
    assert name == "run_fall_only_mode"


def test_unknown_mode_is_rejected(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "--mode", "bogus"])
    with pytest.raises(SystemExit):
        main_module.main()


def test_fall_timings_default_to_config(dispatch, monkeypatch):
    """생략하면 None으로 넘어가고, 실제 기본값은 스케줄러를 만들 때 config에서 읽는다."""
    _, args = run(dispatch, monkeypatch, "--mode", "fall")
    assert args.fall_after is None
    assert args.fall_hold is None


def test_fall_timings_are_parsed_as_numbers(dispatch, monkeypatch):
    _, args = run(dispatch, monkeypatch,
                  "--mode", "fall", "--fall-after", "3", "--fall-hold", "7.5")
    assert args.fall_after == 3.0
    assert args.fall_hold == 7.5


def test_fall_timing_outside_fall_mode_warns(dispatch, monkeypatch, capsys):
    """조용히 무시하면 "왜 서보가 안 도는지"를 찾느라 시간을 버린다."""
    run(dispatch, monkeypatch, "--fall-after", "3")
    assert "--mode fall" in capsys.readouterr().out


def test_no_motor_and_no_serial_are_flags(dispatch, monkeypatch):
    _, args = run(dispatch, monkeypatch, "--no-motor", "--no-serial")
    assert args.no_motor is True
    assert args.no_serial is True


def test_confirm_frames_defaults_to_config(dispatch, monkeypatch):
    _, args = run(dispatch, monkeypatch)
    assert args.confirm_frames == config.ZONE_RESIDENCY_FRAMES
