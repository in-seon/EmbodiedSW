r"""config의 파일 경로가 실행 디렉터리와 무관하게 프로젝트 안을 가리키는지 검증한다."""

from pathlib import Path

import pytest

from config import config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

PATH_SETTINGS = ["ZONE_CONFIG_PATH", "DETECTION_MODEL_PATH"]

ALLOWED_ROOTS = {
    "ZONE_CONFIG_PATH": (PROJECT_ROOT,),
    "DETECTION_MODEL_PATH": (PROJECT_ROOT, REPO_ROOT),
}


@pytest.mark.parametrize("name", PATH_SETTINGS)
def test_path_is_absolute_or_bare_name(name):
    value = getattr(config, name)
    if value is None:
        pytest.skip(f"{name}이 설정되지 않음")
    path = Path(value)
    assert path.is_absolute() or path.parent == Path("."), (
        f"{name}이 상대 경로입니다: {value!r}. "
        "실행 디렉터리에 따라 다른 파일을 가리킵니다."
    )


@pytest.mark.parametrize("name", PATH_SETTINGS)
def test_absolute_path_stays_inside_project(name):
    value = getattr(config, name)
    if value is None:
        pytest.skip(f"{name}이 설정되지 않음")
    path = Path(value)
    if not path.is_absolute():
        return
    roots = ALLOWED_ROOTS[name]
    assert any(path.is_relative_to(root) for root in roots), (
        f"{name}이 허용된 범위 밖을 가리킵니다: {value!r}\n"
        f"허용된 뿌리: {[str(r) for r in roots]}\n")


def test_zone_config_lives_in_data_dir():

    assert Path(config.ZONE_CONFIG_PATH).parent == PROJECT_ROOT / "data"
