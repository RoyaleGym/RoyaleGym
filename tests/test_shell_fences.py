"""Would a reader's shell actually run the commands on this repo's pages?

WHY THIS EXISTS
    On 2026-09-22 the published install block ran in NO shell a newcomer would use, and
    had not since the repos were split. Four readers hit it from four entry points. The
    shapes, each confirmed on a real machine rather than argued: ``&&`` is a parse error
    in Windows PowerShell 5.1 before anything executes; a trailing ``#`` comment is not a
    comment in ``cmd``, so ``python -m venv .venv  # Python 3.12`` silently creates four
    directories and no environment; and a Windows backslash path inside a ```bash fence
    loses its backslashes.

    The second is the dangerous one, because it does not fail. The reader gets no error
    and no working venv.

WHAT IT CHECKS, AND WHAT IT DELIBERATELY DOES NOT
    Only those shapes, on the pages a newcomer actually lands on. It does not read the
    commands and cannot tell you the instructions are correct: the real check is a person
    running the recipe verbatim in PowerShell from an empty folder, which is how this was
    found. This is the guard that stops it coming back through ordinary editing.

    The checker is ``tests/_shell_fences.py``, a vendored copy kept byte-identical to the
    original so the four public repos' copies can be diffed. Its own ``--selftest`` plants
    all four shapes plus three that must NOT fire, and that self-test is run here too: a
    guard whose self-test is never run is a guard nobody has seen work.

A GUARD WITH FALSE POSITIVES IS ONE SOMEBODY SWITCHES OFF
    The checker's first version demanded a named shell near EVERY block rather than once
    per page, so it flagged the two pages that had just been fixed. That was corrected
    before it shipped. It is worth keeping in mind here: these pages pass today, so if
    this test ever goes red on a page a person has just verified by hand, suspect the
    guard before the page.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from _shell_fences import check_text, selftest

REPO = Path(__file__).resolve().parents[1]
CHECKER = Path(__file__).resolve().parent / "_shell_fences.py"

#: The pages a reader lands on and pastes from. Not every .md in the repo: the guard is
#: about what a newcomer runs, and a contributor reference that shows a POSIX command is
#: not the same promise.
READER_PAGES = ("README.md", "docs/site/pages/install.md")


@pytest.mark.parametrize("page", READER_PAGES)
def test_a_reader_could_paste_this_page_into_their_shell(page: str) -> None:
    path = REPO / page
    assert path.exists(), f"{page} is named here as a reader-facing page and does not exist"
    problems = check_text(path.read_text(encoding="utf-8"), page)
    assert not problems, (
        f"{page} has {len(problems)} block(s) a reader's shell would not run:\n  "
        + "\n  ".join(problems)
        + "\n\nThese are shapes, not opinions: && is a parse error in Windows PowerShell, "
        "a trailing # is not a comment in cmd, and bash eats backslashes. If the page is "
        "right and this is wrong, fix the guard rather than silencing it."
    )


def test_the_guards_own_self_test_passes() -> None:
    """It plants all four shapes and three that must not fire. Run it, do not assume it.

    Called in-process for the failure message, and as a subprocess below because the
    file is also meant to be runnable on its own by someone with a fresh clone and
    nothing installed.
    """
    # selftest() returns a COUNT and prints a line per case, so read the printed lines
    # for which case misbehaved. Comparing against 0 rather than truthiness on purpose:
    # a count is not a list, and formatting it as one would raise here instead of
    # reporting, turning a real failure into a confusing one.
    assert selftest() == 0, (
        "the shell-fence guard's own self-test does not behave. Its printed lines say "
        "which planted shape went unrefused, or which safe page was refused; run "
        "`python tests/_shell_fences.py --selftest` to see them."
    )


def test_the_guard_runs_standalone_on_a_fresh_clone() -> None:
    """No pytest, no package, no dependencies: ``python tests/_shell_fences.py --selftest``.

    That is the whole point of vendoring it. A contributor who has cloned the repo and
    not yet built anything can still check a page they are editing.
    """
    done = subprocess.run(
        [sys.executable, str(CHECKER), "--selftest"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO,
    )
    assert done.returncode == 0, (
        f"the vendored checker does not run standalone (exit {done.returncode}):\n"
        f"{(done.stdout + done.stderr)[-1500:]}"
    )


def test_the_checker_is_the_vendored_copy_and_not_a_local_rewrite() -> None:
    """It is a copy on purpose, and a copy that has drifted is worth knowing about.

    This cannot compare against the original, which lives outside this repo and is not
    present in a clone. What it can do is hold the copy to the shape that makes it a
    copy: no imports beyond the standard library, so it runs anywhere, and the self-test
    and entry point still present so a reader can run it.
    """
    text = CHECKER.read_text(encoding="utf-8")
    assert "def selftest(" in text, "the vendored checker has lost its self-test"
    assert "__main__" in text, "the vendored checker is no longer runnable on its own"
    imports = {
        line.split()[1].split(".")[0]
        for line in text.splitlines()
        if line.startswith("import ") or line.startswith("from ")
    }
    assert imports <= {"re", "sys", "pathlib", "__future__"}, (
        f"the vendored checker imports {sorted(imports - {'re', 'sys', 'pathlib', '__future__'})}, "
        "so it no longer runs on a fresh clone with nothing installed"
    )
