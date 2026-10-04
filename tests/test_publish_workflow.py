"""A release reaches PyPI only through its fresh-user gate.

Nobody approves an upload by hand: pypi.yml runs when a fresh-user run completes, and publishes
only a run that was dispatched for one release tag and succeeded on every leg. The two workflow
files meet at one string, the run's name ("fresh-user vX.Y.Z"), which fresh-user.yml writes and
pypi.yml reads, so these tests pin the two sides to each other as well as the gate's conditions.
"""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
PYPI = (WORKFLOWS / "pypi.yml").read_text(encoding="utf-8")
FRESH = (WORKFLOWS / "fresh-user.yml").read_text(encoding="utf-8")


def _block(text: str, key: str) -> str:
    """The lines of a top-level YAML key, up to the next top-level key."""
    m = re.search(rf"^{key}:.*?(?=^\S)", text, re.S | re.M)
    assert m, f"no top-level {key}:"
    return m.group(0)


def _job(text: str, name: str) -> str:
    m = re.search(rf"^  {name}:\n.*?(?=^  \S|\Z)", text, re.S | re.M)
    assert m, f"no job {name}"
    return m.group(0)


def test_a_tag_push_publishes_nothing():
    on = _block(PYPI, "on")
    assert "push" not in on
    assert "tags" not in on
    assert "workflow_dispatch" not in on, "a dispatch would publish around the gate"
    assert re.search(r"workflow_run:\s*\n\s*workflows: \[fresh-user\]\s*\n\s*types: \[completed\]", on)


def test_it_listens_to_the_workflow_fresh_user_yml_is():
    assert re.search(r"^name: fresh-user$", FRESH, re.M)


def test_only_a_passed_gate_for_a_release_tag_publishes():
    gate = _job(PYPI, "gate")
    assert "github.event.workflow_run.conclusion == 'success'" in gate
    assert "github.event.workflow_run.event == 'workflow_dispatch'" in gate
    assert "startsWith(github.event.workflow_run.display_title, 'fresh-user v')" in gate
    assert 'TAG="${TITLE#fresh-user }"' in gate
    assert r"^v[0-9]+\.[0-9]+\.[0-9]+$" in gate
    assert "publish=false" in gate, "a version PyPI already has is not uploaded again"
    assert "if: needs.gate.outputs.publish == 'true'" in _job(PYPI, "build")


def test_the_run_name_is_what_the_gate_reads():
    """fresh-user.yml names a release run "fresh-user <tag>" and a nightly one "fresh-user",
    which the gate's startsWith('fresh-user v') never matches."""
    m = re.search(r"^run-name: (.+)$", FRESH, re.M)
    assert m, "fresh-user.yml has no run-name, so pypi.yml cannot tell which tag passed"
    assert m.group(1) == (
        "${{ inputs.release_tag && format('fresh-user {0}', inputs.release_tag) || 'fresh-user' }}"
    )


def test_the_build_is_the_tag_and_the_tag_is_the_version():
    build = _job(PYPI, "build")
    assert "ref: refs/tags/${{ needs.gate.outputs.tag }}" in build
    assert 'if [ "$TAG" != "v$VERSION" ]; then' in build


def test_only_the_upload_job_can_publish():
    assert PYPI.count("id-token: write") == 1
    upload = _job(PYPI, "pypi")
    assert "id-token: write" in upload
    assert "environment: pypi" in upload
    assert "needs: [gate, build]" in upload
    assert "pypa/gh-action-pypi-publish" in upload
    assert "needs: [gate, pypi]" in _job(PYPI, "installs")


def test_the_pypi_leg_tests_the_release_it_names():
    """Before a release is published, PyPI gives the previous version: the leg says so instead of
    testing that, and once it is published the install must be that version."""
    step = FRESH[FRESH.index("- name: the PyPI install line") :]
    assert "NOT-PUBLISHED" in step
    assert '--expect-version "$VERSION"' in step


def test_nothing_says_an_upload_waits_for_the_owner():
    for text in (PYPI, FRESH):
        assert not re.search(r"owner.{0,60}approv|approv.{0,60}owner|waits for", text, re.I | re.S)
