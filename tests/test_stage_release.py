"""tools/stage_release.py takes each sibling's newest v* tag on PyPI, and refuses a tag whose
tree holds another version or a repo with no tag at all."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "stage_release.py"


def _tool():
    spec = importlib.util.spec_from_file_location("stage_release", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repo(tmp_path: Path, versions_and_tags: list[tuple[str, str | None]]) -> Path:
    repo = tmp_path / "RoyaleThing"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    for version, tag in versions_and_tags:
        (repo / "pyproject.toml").write_text(
            f'[project]\nname = "royalething"\nversion = "{version}"\n', encoding="utf-8"
        )
        git("add", "pyproject.toml")
        git("commit", "-q", "-m", version)
        if tag:
            git("tag", tag)
    return repo


def test_the_newest_tag_wins_by_version_not_by_name(tmp_path):
    tool = _tool()
    repo = _repo(tmp_path, [("0.9.0", "v0.9.0"), ("0.10.0", "v0.10.0"), ("0.10.1", None)])
    assert tool.newest_tag(repo) == "v0.10.0"
    assert tool.check_tag(repo, "v0.10.0") == "0.10.0"


def test_a_tag_on_a_tree_with_another_version_is_refused(tmp_path):
    tool = _tool()
    repo = _repo(tmp_path, [("0.4.1", "v0.4.2")])
    with pytest.raises(SystemExit, match=r"tag v0\.4\.2 holds version 0\.4\.1"):
        tool.check_tag(repo, "v0.4.2")


def test_a_repo_with_no_tag_is_refused(tmp_path):
    tool = _tool()
    repo = _repo(tmp_path, [("0.1.0", None)])
    with pytest.raises(SystemExit, match="has no v\\* tag"):
        tool.newest_tag(repo)


def test_a_sibling_is_staged_at_its_newest_version_on_pypi(tmp_path, capsys):
    """The page holds what `pip install "royalegym[all]"` gets from PyPI. A newer tag PyPI does
    not have is not released yet (on 2026-10-04 RoyaleLearn had tagged 0.5.6 and 0.5.7 and PyPI
    had 0.5.5), so it is left off the page, and the stderr says which."""
    tool = _tool()
    repo = _repo(tmp_path, [("0.5.5", "v0.5.5"), ("0.5.6", "v0.5.6"), ("0.5.7", "v0.5.7")])
    asked = []

    def published(package: str, version: str) -> bool:
        asked.append((package, version))
        return version == "0.5.5"

    assert tool.newest_published_tag(repo, "royalething", published) == "v0.5.5"
    assert asked == [("royalething", "0.5.7"), ("royalething", "0.5.6"), ("royalething", "0.5.5")]
    err = capsys.readouterr().err
    assert "v0.5.7" in err
    assert "v0.5.6" in err
    with pytest.raises(SystemExit, match="no v\\* tag of royalething is on PyPI"):
        tool.newest_published_tag(repo, "royalething", lambda package, version: False)


STAGED = {"royalesim": "0.1.5", "royalelearn": "0.5.3", "royaleviser": "0.1.1",
          "royaleimitate": "0.2.5"}
ALL = ["royalesim>=0.1.5", "royalelearn[torch]>=0.5.3", "royaleviser[media]>=0.1.1",
       "royaleimitate[replays]>=0.2.5"]


def test_floors_that_match_the_page_pass():
    assert _tool().floor_problems(ALL, STAGED) == []


def test_a_floor_below_the_staged_version_is_refused():
    """`pip install --upgrade "royalegym[all]"` keeps a package its floor already allows, so a
    floor left at an older version keeps users on it (measured 2026-10-03: royalesim 0.1.3)."""
    low = [r.replace("royalesim>=0.1.5", "royalesim>=0.1.4") for r in ALL]
    problems = _tool().floor_problems(low, STAGED)
    assert len(problems) == 1
    assert "royalesim" in problems[0]
    assert "0.1.4" in problems[0]
    assert "0.1.5" in problems[0]


def test_a_floor_above_the_page_or_none_at_all_is_refused():
    tool = _tool()
    high = [r.replace(">=0.2.5", ">=0.2.6") for r in ALL]
    assert any("royaleimitate" in p for p in tool.floor_problems(high, STAGED))
    bare = [r.replace("royaleviser[media]>=0.1.1", "royaleviser[media]") for r in ALL]
    assert any("royaleviser" in p and "no minimum" in p for p in tool.floor_problems(bare, STAGED))
    missing = [r for r in ALL if not r.startswith("royalelearn")]
    assert any("royalelearn" in p for p in tool.floor_problems(missing, STAGED))
