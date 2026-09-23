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

from _pinned_count import collected_excluding_xfail, skip_reason, why_it_cannot_be_checked
from royalegym.protocol import derived_cards_vintage
from royalegym.rust_engine import CORE_IMPORT_ERROR, core_available
from royalegym.rust_engine import build_digest as engine_build_digest

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
#: The vintage the printed output was recorded against. Named so the failure message
#: can say it: a different catalogue is the first thing to suspect when the battle
#: changes. It is NOT a skip -- a printed result that is wrong for the reader's
#: catalogue is wrong, and skipping would hide it. Both vintages on this machine
#: produce the same battle, which is the README's point about naming cards rather than
#: numbering them, and it is checked rather than assumed.
RECORDED_VINTAGE = "retroroyale-2018"
#: The engine build the README's printed battle was recorded against. It covers
#: data/calibration.json and arena.json, so it moves when a ledger VALUE moves, which is
#: what turned this battle from a draw into a win on 2026-09-22 with no card changed.
RECORDED_BUILD = "f7628dd51148e4ce"

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
        f"{done.stdout.strip()}\n\n"
        "The printed battle is the OUTCOME of a whole simulation, so it moves with "
        "anything the engine reads, not only with the cards. What it depends on, in the "
        "order worth checking:\n"
        f"  the engine build   recorded {RECORDED_BUILD}, now {engine_build_digest()}\n"
        f"  the card table     recorded {RECORDED_VINTAGE!r}, now "
        f"{derived_cards_vintage()!r}\n"
        "The build digest covers data/calibration.json and arena.json, so a ledger value "
        "moving is enough on its own: the starting elixir went from 5 to 6 on 2026-09-22 "
        "and turned this battle from a 0-0 draw at tick 4800 into 1-0 at tick 3600, with "
        "no card changed at all. Until that was written down, this message named only the "
        "card table and sent a reader to the wrong file.\n"
        "Re-record by running the program and pasting its output under the block, and "
        "update RECORDED_BUILD and RECORDED_VINTAGE beside it."
    )


def test_the_badge_count_matches_what_the_suite_collects() -> None:
    """The front page says how many tests pass. Make that a checkable claim.

    It was stated in two places -- the badge and a comment in the command block -- and
    they disagreed with each other by fifty and with the suite by more. Two copies of a
    number is a machine for producing a wrong one: whoever updates it updates the copy
    they are looking at.

    There is one copy now, and it is PINNED TO A COMMIT, which is what makes it
    checkable at all. The badge reads "778 passed, 7 skipped at c94bb53", and this
    compares those numbers against collection only when HEAD is that commit with a clean
    tree. Anywhere else it checks that the badge names a commit and says, out loud, that
    it cannot check further.

    IT USED TO CHECK ON EVERY COMMIT AND THAT WAS WRONG. Five sessions add tests to this
    suite, so the count moved under the badge continually: 776 set, 777 measured, 777
    set, 778 measured, and two runs a minute apart reporting 785 then 786. Every number
    was true when it was taken. A literal racing a moving suite turns red on work that
    has nothing to do with the README, and a test that cries wolf is one people learn to
    ignore -- which costs more than the stale number it was guarding against.

    A count pinned to a commit is a smaller claim and a true one. The friction that keeps
    it fresh is the date and commit in the label, not this test going red on somebody
    else's work.
    """
    # There are two test-count badges and they count DIFFERENT populations: one a fresh
    # clone, which nothing here can re-run, and one this machine. Only the second is a
    # claim collection can check, so this finds it by its LABEL rather than by its
    # position. Keying on position would silently start checking the fresh-clone figure
    # against this machine the day the two badges are reordered, which is the failure
    # this test is supposed to prevent rather than commit.
    readme = README.read_text("utf-8")
    ours = re.findall(r"our%20machine[^\"']*?-(\d+)%20passed%2C%20(\d+)%20skipped", readme)
    counts = re.findall(r"-(\d+)%20passed%2C%20(\d+)%20skipped", readme)
    assert counts, "README.md states no test count at all; it used to state two"
    assert len(ours) == 1, (
        f"expected exactly one badge labelled as this machine's, found {len(ours)} among "
        f"{len(counts)} test-count badges. If the label changed, change it here too: a "
        "count nothing checks is the thing this test exists to prevent."
    )
    claimed = int(ours[0][0]) + int(ours[0][1])

    at = re.search(r"our%20machine%20at%20([0-9a-f]{7,40})", readme)
    assert at, (
        "the machine badge names no commit, so nothing can ever check it: a count "
        "without the tree it was taken on is a claim about a moving target. Label it "
        "`our machine at <sha>`."
    )
    # The rule itself lives in tests/_pinned_count.py, because docs/architecture.md
    # states a count too and two copies of a rule drift exactly like two copies of a
    # number. That file is also where the reasoning is written down.
    reason = why_it_cannot_be_checked(at.group(1))
    if reason:
        pytest.skip(skip_reason("the badge", ours[0][0], ours[0][1], at.group(1), reason))

    collected = collected_excluding_xfail()
    assert claimed == collected, (
        f"the README badge for this machine says {ours[0][0]} passed and {ours[0][1]} "
        f"skipped, which is {claimed} tests, and pytest collects {collected}."
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


def test_the_clone_badge_is_present_and_says_where_it_came_from() -> None:
    """The reader's own number, and the one thing this machine cannot verify.

    The front page states two counts because they are two POPULATIONS: what this
    machine's suite does, and what a reader's fresh clone does. They differ because the
    README's recipe installs the 2018 card table, so a clone gets 66 loadable cards
    against this machine's 100, and two scripted scenarios skip there rather than run.

    This test deliberately does NOT check the clone badge's split. It cannot: there is no
    clone here, and collection on this machine answers a different question. The two
    totals happen to be equal, 781 + 3 and 777 + 7 both being 784 when this was written,
    because those two scenarios move from passed to SKIPPED rather than leaving the
    suite. That equality is a coincidence of one day's arithmetic and not a property, so
    checking the clone's pair against this machine's collection would look like
    verification while resting on it. A clone scenario that stopped skipping and started
    failing would move the split without moving the total, and the badge would be wrong
    and green.

    What it checks is what an unverifiable claim needs: that it is there, that it says
    which population it counts, and that it says WHERE IT CAME FROM. A commit is the
    better form and is what the page carries, because it names the tree that was measured
    rather than the day; a date is accepted too, since a dated claim can at least be aged.
    """
    readme = README.read_text("utf-8")
    clone = re.search(r"your%20clone[^\"']*?-(\d+)%20passed%2C%20(\d+)%20skipped", readme)
    assert clone, (
        "README.md no longer carries a clone badge. It is the only count a reader can "
        "reproduce, so if it was removed, say in the Status prose what replaced it; if it "
        "was renamed, rename it here. This test does not check its numbers and never "
        "could: there is no clone on this machine."
    )
    assert int(clone.group(1)) > 0, "the clone badge claims no passing tests"
    label = clone.group(0).split("-")[0]
    dated = re.search(r"\d{4}--\d{2}--\d{2}", label)
    committed = re.search(r"%20at%20[0-9a-f]{7,40}$", label)
    assert dated or committed, (
        f"the clone badge says {label!r}, which names no commit and no date. Nothing here "
        "can re-measure it, so the one thing it must carry is where it came from: a "
        "commit for preference, since that names the tree, or a date so it can be aged."
    )
