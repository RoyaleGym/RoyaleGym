"""The README says "this is the complete program" and prints what it produces. Check both.

WHY THIS EXISTS
    The front page shows a program and, directly under it, the output it produces --
    winner, crowns and tick. Those are the most load-bearing lines in the repo: they
    are what a reader runs first, and if the program does not run, or runs and prints
    something else, that is the first thing they learn about the project. Nothing read
    either of them before this.

    It is also the only claim here that a reader will check by accident.

WHAT IT CATCHES
    The snippet not running at all (a renamed class, a moved argument, a changed
    default), and the snippet running but producing a different battle -- which is what
    a card-table change or an engine change does, silently, to a printed result.

WHAT IT CANNOT CATCH
    Whether the prose AROUND the program is right: the sentences about three minutes
    plus overtime and whose king was healthier are read from this output by a human and
    are not re-derived here.

    Whether the eight named cards behave identically across catalogues. They need not:
    the README's point is that NAMING them makes the battle reproducible, and it holds
    on both vintages available here, measured rather than assumed. If a future
    catalogue changes one of them the output moves and this fails, which is correct --
    the printed result would then be wrong for that reader.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from royalegym.protocol import derived_cards_vintage
from royalegym.rust_engine import CORE_IMPORT_ERROR, core_available

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
#: The vintage the printed output was recorded against. Named so the failure message
#: can say it: a different catalogue is the first thing to suspect when the battle
#: changes. It is NOT a skip -- a printed result that is wrong for the reader's
#: catalogue is wrong, and skipping would hide it. Both vintages on this machine
#: produce the same battle, which is the README's point about naming cards rather than
#: numbering them, and it is checked rather than assumed.
RECORDED_VINTAGE = "retroroyale-2018"

pytestmark = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))


def readme_blocks() -> list[tuple[str, str]]:
    """(language, body) for every fenced block in README.md, in order."""
    return [
        (m.group(1) or "", m.group(2))
        for m in re.finditer(r"```(\w*)\n(.*?)```", README.read_text(encoding="utf-8"), re.S)
    ]


def try_it_program() -> tuple[str, str]:
    """The Try-it program and the output printed under it.

    Found by CONTENT, not by position: "the third fenced block" would keep passing
    while pointing at a different block the moment somebody adds one above it, which
    is how a test ends up checking something other than what it says.
    """
    blocks = readme_blocks()
    for i, (lang, body) in enumerate(blocks):
        if lang == "python" and "while env.agents:" in body and "ClashParallelEnv" in body:
            assert i + 1 < len(blocks), "the Try-it program has no output block under it"
            return body, blocks[i + 1][1]
    raise AssertionError("no Try-it program found in README.md")


def test_the_try_it_program_runs_and_prints_what_the_readme_says() -> None:
    program, expected = try_it_program()
    done = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=300, cwd=REPO
    )
    assert done.returncode == 0, (
        f"the README's 'complete program' does not run:\n{done.stderr[-2000:]}"
    )
    assert done.stdout.strip() == expected.strip(), (
        f"the README prints:\n  {expected.strip()}\nthe program prints:\n  "
        f"{done.stdout.strip()}\n\nThe first thing to suspect is the card table: this "
        f"engine was built with {derived_cards_vintage()!r} and the printed result was "
        f"recorded against {RECORDED_VINTAGE!r}. It held on both when it was written, so "
        "a difference means one of the eight named cards has changed, and the README's "
        "printed battle is now wrong for whoever has this catalogue."
    )


def test_the_badge_count_matches_what_the_suite_collects() -> None:
    """The front page says how many tests pass. Make that a checkable claim.

    It was stated in two places -- the badge and a comment in the command block -- and
    they disagreed with each other by fifty and with the suite by more. Two copies of a
    number is a machine for producing a wrong one: whoever updates it updates the copy
    they are looking at.

    There is one copy now, and this compares it against collection rather than against
    a run: ``--collect-only`` is quick, does not execute anything, and passed + skipped
    is exactly what gets collected. So adding a test without touching the badge turns
    this red, which is the friction that keeps the number true.
    """
    badge = re.search(r"pytest-(\d+)%20passed%2C%20(\d+)%20skipped", README.read_text("utf-8"))
    assert badge, "no pytest badge in README.md"
    claimed = int(badge.group(1)) + int(badge.group(2))

    done = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=REPO,
    )
    found = re.search(r"(\d+) tests? collected", done.stdout)
    assert found, f"could not read a collected count:\n{done.stdout[-800:]}"
    collected = int(found.group(1))
    assert claimed == collected, (
        f"the README badge says {badge.group(1)} passed and {badge.group(2)} skipped, "
        f"which is {claimed} tests, and pytest collects {collected}."
    )


def test_the_program_really_is_complete() -> None:
    """"Nothing but royalegym and numpy" -- so nothing else may be imported.

    The claim is not that it works, it is that it works with nothing else installed.
    An import of anything a fresh install does not have makes the sentence false for
    the one reader who takes it literally.
    """
    program, _ = try_it_program()
    imported = set(re.findall(r"^\s*(?:import|from)\s+([\w.]+)", program, re.M))
    roots = {name.split(".")[0] for name in imported}
    assert roots <= {"royalegym", "numpy", "np"}, (
        f"the README calls this the complete program with nothing but royalegym and "
        f"numpy, and it imports {sorted(roots)}"
    )
