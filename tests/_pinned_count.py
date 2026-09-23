"""One rule for every test count this repo states in prose.

WHY THIS IS SHARED
    The README badge and ``docs/architecture.md`` both tell a reader how many tests pass.
    Two places stating one fact is a machine for producing a wrong one -- whoever updates
    it updates the copy they are looking at -- and this repo has already been round that
    loop: the badge was corrected three times in one evening and overtaken three times,
    because five sessions add tests to this suite and a literal cannot race a moving
    suite. Every value was true when it was taken.

    So the rule is not "keep the number fresh". It is that a count is only a fact about
    the tree it was measured on, and is therefore only checkable ON that tree. A claim
    names its commit; this compares it against collection when HEAD is that commit with
    no tracked changes, and otherwise says out loud that it cannot check further rather
    than passing quietly or failing loudly on somebody else's work.

    ``docs/architecture.md`` was pinned to a DAY, which is weaker still: five sessions
    commit many times a day, and its figure was 96 tests stale when this was written.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Matches "<passed> passed, <skipped> skipped" and the commit, in prose or in a badge's
#: URL-encoded label. The caller supplies the surrounding pattern; this is the shape both
#: claims share.
CLAIM = re.compile(
    r"(?P<passed>\d+)\s*(?:%20)?passed(?:%2C|,)\s*(?:%20)?(?P<skipped>\d+)\s*(?:%20)?skipped"
)


def head_sha(length: int) -> str | None:
    """The short HEAD sha, or None when this is not a git checkout."""
    done = subprocess.run(
        ["git", "rev-parse", f"--short={length}", "HEAD"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    return done.stdout.strip() if done.returncode == 0 else None


def tracked_changes() -> list[str]:
    """Modified tracked files. Untracked files are excluded on purpose: a scratch file
    beside the tree does not change what the suite collects, and counting it would make
    this unusable in a working directory."""
    done = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=REPO
    )
    return [ln for ln in done.stdout.splitlines() if not ln.startswith("??")]


def why_it_cannot_be_checked(pinned: str) -> str | None:
    """None when the count is checkable here, otherwise the reason, for a skip message."""
    head = head_sha(len(pinned))
    if head is None:
        return "this is not a git checkout, so the pinned commit cannot be compared with HEAD"
    if head != pinned:
        return f"this tree is at {head}"
    if tracked_changes():
        return "this tree is that commit with uncommitted changes"
    return None


def collected_excluding_xfail() -> int:
    """How many tests the suite collects, minus the expected failures.

    The xfails are deselected because a run reports them as xfailed, so they are in
    neither number a claim states; counting them would make every claim wrong by exactly
    the number of open defects the suite is marking.
    """
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", "not xfail"],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=REPO,
    )
    # Collection has to have SUCCEEDED. A test file that cannot be imported is a
    # collection ERROR, and pytest still prints a count of what it did manage to collect,
    # so reading that number without the exit code lets this certify a claim against a
    # suite that does not run. Planted: three unimportable test files with the claim
    # lowered to match, and it passed while the rest never imported. A count is only a
    # fact about the suite if the suite could be read.
    assert done.returncode == 0, (
        f"pytest could not collect the suite (exit {done.returncode}), so the count it "
        f"printed is of whatever survived. This is a broken suite, not a wrong claim:\n"
        f"{(done.stdout + done.stderr)[-1500:]}"
    )
    found = re.search(r"(\d+)(?:/\d+)? tests? (?:collected|deselected)", done.stdout)
    assert found, f"could not read a collected count:\n{done.stdout[-800:]}"
    return int(found.group(1))


def skip_reason(where: str, passed: str, skipped: str, pinned: str, reason: str) -> str:
    return (
        f"SKIPPED, NOT PASSED: {where} states {passed} passed and {skipped} skipped at "
        f"{pinned}, and {reason}. The count is only checkable on the tree it was taken "
        "on; anywhere else collection answers a different question. Re-measure and "
        "re-pin rather than making this compare numbers from two different trees."
    )
