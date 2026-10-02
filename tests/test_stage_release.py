"""tools/stage_release.py takes each sibling's newest v* tag, and refuses a tag whose tree holds
another version or a repo with no tag at all."""

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
