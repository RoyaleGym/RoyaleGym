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
    "../.venv/bin/python tools/extract_globals.py",
)

#: WHICH FILE ENDS UP AS cards.json is checked separately, below, because it is the one
#: part of the install that three places have to agree about and the only part whose
#: disagreement changes what the engine computes. It was dropped from the list above when
#: it stopped being a single literal shared by both pages.
CARDS_JSON_WRITERS = {
    "2018": "--vintage 2018 --out data",
    "15.535": "cards-15.535.json",
}


def installed_table(text: str) -> set[str]:
    """Which card table a page or workflow puts in cards.json, by the line that writes it."""
    return {name for name, marker in CARDS_JSON_WRITERS.items() if marker in text}


def test_the_readme_the_install_page_and_the_workflow_install_the_SAME_card_table() -> None:
    """Three places describe one cards.json, and the engine loads whatever is in it.

    This is not tidiness. The card table changes what the engine COMPUTES: the same
    scripted battle ends `winner 0 crowns [2, 1]` on the 15.535 table and
    `winner 1 crowns [0, 1]` on the 2018 one, same engine build. So a reader who follows
    the page that disagrees gets a different game and no way to know which page was wrong.

    It fires the moment one of the three moves, which is what happened on 2026-09-23:
    README.md changed to copy the committed 15.535 table after sim ruled that a 2026
    simulator should not run on eight-year-old card data, this workflow followed it, and
    docs/site/pages/install.md kept publishing the two `--vintage 2018` lines. Two
    published pages then told a reader to put different tables in the same file, and
    whichever install they ran last won.
    """
    sources = {
        "README.md": (REPO / "README.md").read_text(encoding="utf-8"),
        "docs/site/pages/install.md": INSTALL_PAGE.read_text(encoding="utf-8"),
        ".github/workflows/suite.yml": WORKFLOW.read_text(encoding="utf-8"),
    }
    tables = {name: installed_table(text) for name, text in sources.items()}
    assert all(tables.values()), (
        f"a source names no cards.json writer at all, so this check cannot compare them: "
        f"{ {k: sorted(v) for k, v in tables.items()} }"
    )
    distinct = {frozenset(v) for v in tables.values()}
    assert len(distinct) == 1, (
        "these disagree about which card table becomes cards.json, and the engine loads "
        f"whatever is there: { {k: sorted(v) for k, v in tables.items()} }. Make the two "
        "published pages agree first -- the workflow follows the README."
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


def test_the_workflow_emits_the_collected_count_rather_than_leaving_it_derived() -> None:
    """Under the certification standard the COLLECTED figure is the load-bearing one.

    Two pass counts invite subtraction and the subtraction is meaningless without it: it
    cannot tell a different population from a different selection from missing tests. The
    integrator derived this repo's 856 as 840 + 15 + 1, which works only while xfail is the
    sole extra bucket. Add a marker class, an error or a deselection and that arithmetic
    reads wrong with nothing saying so.

    So the instrument states it. Pinned here because it is one line in a summary block, it
    looks like duplication of the suite step to anyone tidying, and removing it puts the
    certifier back to reconstructing a number the run already knew.
    """
    text = WORKFLOW.read_text(encoding="utf-8")
    marker = "- name: record what this run was"
    body = text.split(marker, 1)[1].split("\n      - name:", 1)[0]
    assert "collected=" in body, (
        "the recording step no longer emits a collected count, so a certifier has to "
        "reconstruct it by adding up the summary buckets"
    )
    assert "--collect-only" in body, (
        "the collected count is not taken from pytest --collect-only, so it is a number "
        "about something other than what this run collected"
    )


def test_the_provenance_block_is_readable_from_the_run_log() -> None:
    """A provenance block written only to the step summary cannot be checked.

    `$GITHUB_STEP_SUMMARY` is not returned by `gh run view` or by the jobs API, so a value
    sent only there is invisible to the person who needs it -- and the integrator reads run
    LOGS, quoting log lines for this repo's counts. Writing the collected count somewhere
    unreadable made "the workflow emits it" an unverified claim about the very instrument
    added to stop a number being unverified.

    So the block tees: the summary keeps its formatted copy and stdout carries the same
    text into the log.
    """
    text = WORKFLOW.read_text(encoding="utf-8")
    marker = "- name: record what this run was"
    body = text.split(marker, 1)[1].split("\n      - name:", 1)[0]
    assert "tee -a" in body, (
        "the provenance block no longer tees to stdout, so its values go only to the step "
        "summary, which cannot be read from the API or from `gh run view`. A number nobody "
        "can read is not a number the run reported."
    )
