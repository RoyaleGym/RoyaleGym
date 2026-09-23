"""CI must install the way this repo's own pages tell a reader to install.

WHY THIS EXISTS
    A workflow that installs some other way tests nothing a reader will experience. The
    install path here was broken for the entire public life of these repos and no gate saw
    it, because every gate ran in a workspace that was already set up. That is the whole
    reason CI is worth having, and it is worth exactly nothing if the workflow quietly
    drifts to a shorter install that happens to be greener.

    The obvious shape -- `pip install -e ".[dev]"` -- would be a fiction here. RoyaleGym has
    NO standalone install: its README documents one path, and that path clones RoyaleSim as
    a sibling, generates the derived data, builds the engine with maturin, and then installs
    this package. The short form would work, every Rust-backed test would SKIP, and the
    suite would be green while testing almost nothing.

WHAT THIS PINS, AND WHY IT IS A TEST RATHER THAN A ONE-OFF CHECK
    That the commands in the workflow appear VERBATIM on the page a reader follows. Verified
    by hand once is a fact about the afternoon it was verified; the pages belong to the docs
    session and change without this repo's code changing at all. So the agreement is checked
    by something that runs.

WHAT IT CANNOT PIN
    That the commands WORK. Only a fresh runner can say that, and the install page states in
    as many words that nobody has ever typed the macOS/Linux commands. A red first CI run is
    the job working, not the job broken.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "suite.yml"
INSTALL_PAGE = REPO / "docs" / "site" / "pages" / "install.md"

#: The macOS/Linux install, in order, as docs/site/pages/install.md publishes it. CI runs
#: ubuntu, so these are the lines a reader on that platform is told to run.
POSIX_INSTALL = (
    "python -m venv .venv",
    ".venv/bin/python -m pip install maturin pytest hypothesis ruff",
    "../.venv/bin/python tools/extract_arena.py",
    "../.venv/bin/python tools/extract_cards.py --vintage 2018",
    "../.venv/bin/python tools/extract_cards.py --vintage 2018 --out data/derived/cards.json",
    "../.venv/bin/python tools/extract_globals.py",
)


@pytest.mark.parametrize("command", POSIX_INSTALL)
def test_the_workflow_runs_the_command_the_install_page_publishes(command: str) -> None:
    """Both sides, on purpose: the command must be on the page AND in the workflow.

    Checking only the workflow would let this pass after docs rewrote the page, which is the
    drift it exists to catch. Checking only the page would let the workflow install some
    other way.
    """
    assert WORKFLOW.exists(), "the CI workflow is gone; this repo's install is then unchecked"
    page = INSTALL_PAGE.read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert command in page, (
        f"the install page no longer publishes {command!r}. If the documented install "
        "changed, change the workflow to match and re-take this list; do not delete the "
        "line, because then CI tests a path nobody is told to run."
    )
    assert command in workflow, (
        f"the workflow does not run {command!r}, which the install page tells a reader to "
        "run. A workflow that installs some other way tests nothing a reader will meet."
    )


def test_the_workflow_builds_the_engine_rather_than_skipping_its_way_to_green() -> None:
    """The failure this repo is most exposed to, because it looks like success.

    Without `royalesim` the package still imports and every engine-backed test skips. A
    suite that skips its way to green is the one outcome this project refuses to call a
    pass, and it is what a shorter install would produce.
    """
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "maturin develop --release" in workflow, (
        "the workflow no longer builds the engine, so every Rust-backed test would skip and "
        "the suite would be green while testing almost nothing"
    )
    assert "core_available" in workflow, (
        "the workflow does not REFUSE to run the suite when the engine is missing. Building "
        "it is not enough: if the build silently fails, the suite still skips to green."
    )


def test_the_workflow_records_the_build_digest_beside_the_count() -> None:
    """A count is a fact about a build, and this one moves without a commit.

    The digest covers calibration.json and arena.json rather than code, so it changed three
    times on 2026-09-22 and twice with no commit in this repo. A CI line recording only a
    count is the version nobody can debug, because the first green-then-red-with-no-commit
    looks impossible.

    THE FIRST VERSION OF THIS TEST DID NOT WORK, and a plant caught it within a minute. It
    asserted ``"build_digest" in workflow``, and that substring also appears in the step's
    IMPORT line, so deleting the recording entirely left the test green. It was checking a
    token adjacent to the thing it claimed. This reads the recording step's own commands.

    NO YAML PARSER HERE, and that is the second thing CI taught this test. The version after
    that imported ``yaml``, which is not in this repo's documented install -- the install
    page lists maturin, pytest, hypothesis and ruff -- so a test ABOUT the workflow was the
    one test that could not run ON the workflow. It failed with ModuleNotFoundError on the
    first real run. Adding PyYAML to CI would have fixed the symptom by making the runner
    diverge from the documented install, which is the one thing this job exists to prevent.
    """
    text = WORKFLOW.read_text(encoding="utf-8")
    # The recording step, taken as the text between its name and the next step or the end.
    marker = "- name: record what this run was"
    assert marker in text, (
        "the workflow no longer has a step named 'record what this run was', so either it "
        "records nothing or this test is looking for the wrong step"
    )
    after = text.split(marker, 1)[1]
    body = after.split("\n      - name:", 1)[0]
    assert "GITHUB_STEP_SUMMARY" in body, (
        "the recording step does not write to GITHUB_STEP_SUMMARY, so a CI run records no "
        "provenance at all"
    )
    for token in ("build_digest=", "engine_binary=", "cards_vintage="):
        assert token in body, (
            f"the recording step does not emit {token}. Without it a later failure cannot be "
            "told apart from a rebuild underneath the same commit, which happened three "
            "times on 2026-09-22 and twice with no commit in this repo."
        )


def test_the_workflow_reports_what_it_could_not_exercise() -> None:
    """A count alone reads as "the suite passed" and means "the subset here passed".

    The gap is not cosmetic in this repo. A clean runner has the 2018 card table and the
    project's own machine has 15.535, so the tests that skip on a runner are precisely the
    ones depending on the most valuable data in the project: a clean-runner count
    systematically under-exercises exactly what is hardest to exercise. An instrument with
    that property should state it rather than leave it to whoever reads the number.

    So the suite step runs with -rs. Pinned as a test because it is one flag, it looks like
    noise to anyone tidying the workflow, and removing it turns an honest measurement back
    into a bare count with no visible loss.
    """
    text = WORKFLOW.read_text(encoding="utf-8")
    marker = "- name: suite"
    assert marker in text, "the workflow no longer has a step named 'suite'"
    body = text.split(marker, 1)[1].split("\n      - name:", 1)[0]
    assert "-rs" in body, (
        "the suite step no longer passes -rs, so a run records how many tests skipped and "
        "never which or why. On a clean runner the skips are the card-table tests, which "
        "makes the count a statement about a smaller population than a reader will assume."
    )
