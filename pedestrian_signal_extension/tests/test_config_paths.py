r"""config의 파일 경로가 실행 디렉터리와 무관하게 프로젝트 안을 가리키는지 검증한다.

이 테스트가 있는 이유(둘 다 실제로 겪은 사고다):

  1. 상대 경로("data/zone_config.json")로 두면 **어느 폴더에서 실행하느냐에 따라**
     다른 파일을 가리킨다. tools/에서 캘리브레이터를 돌리면 tools/data/ 가 생기고,
     프로젝트 루트에서 main.py를 돌리면 그걸 못 찾는다.
  2. 절대 경로로 바꾸더라도 조각에 앞 슬래시가 들어가면(`/ "/data"`) pathlib이
     앞 경로를 통째로 버려서 드라이브 루트(C:\data, /data)를 가리킨다.
     파이에서는 권한 오류로 저장조차 안 된다.

둘 다 "에러 메시지가 경로를 탓하지 않아서" 원인을 찾기 어려웠다.
"""

from pathlib import Path

import pytest

from config import config

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 프로젝트 안에 있어야 하는 경로 설정들
PATH_SETTINGS = ["ZONE_CONFIG_PATH", "DETECTION_MODEL_PATH", "MOBILITY_AID_MODEL_PATH"]


@pytest.mark.parametrize("name", PATH_SETTINGS)
def test_path_is_absolute_or_bare_name(name):
    """절대 경로이거나, (가중치 미존재 시 자동 다운로드용) 순수 파일명이어야 한다.

    "data/zone_config.json" 같은 **상대 경로**는 실행 위치에 따라 달라지므로 금지.
    """
    value = getattr(config, name)
    if value is None:
        pytest.skip(f"{name}이 설정되지 않음")
    path = Path(value)
    assert path.is_absolute() or path.parent == Path("."), (
        f"{name}이 상대 경로입니다: {value!r}. "
        "실행 디렉터리에 따라 다른 파일을 가리키게 됩니다."
    )


@pytest.mark.parametrize("name", PATH_SETTINGS)
def test_absolute_path_stays_inside_project(name):
    """절대 경로라면 프로젝트 루트 안이어야 한다 — 드라이브 루트를 가리키면 안 된다."""
    value = getattr(config, name)
    if value is None:
        pytest.skip(f"{name}이 설정되지 않음")
    path = Path(value)
    if not path.is_absolute():
        return  # 위 테스트가 다루는 '순수 파일명' 경우
    assert path.is_relative_to(PROJECT_ROOT), (
        f"{name}이 프로젝트 밖을 가리킵니다: {value!r}\n"
        f"프로젝트 루트: {PROJECT_ROOT}\n"
        "경로 조각에 앞 슬래시가 붙어 있지 않은지 확인하세요"
        '(Path(root) / "/data" 는 root를 버립니다).'
    )


def test_zone_config_lives_in_data_dir():
    """zone 설정은 프로젝트의 data/ 안이어야 한다 (tools/data/ 같은 곳이 아니라)."""
    assert Path(config.ZONE_CONFIG_PATH).parent == PROJECT_ROOT / "data"
