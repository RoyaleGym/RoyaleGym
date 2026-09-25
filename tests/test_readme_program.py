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

    Whether the eight named cards behave identically across catalogues. They do NOT, and
    this docstring said they did until 2026-09-24: "it holds on both vintages available
    here, measured rather than assumed". It was measured once, and a later measurement on
    the same program at the same engine build gave `winner 0 crowns [2, 1]` on the 15.535
    table and `winner 1 crowns [0, 1]` on the 2018 one -- a different winner. Naming cards
    makes a battle reproducible WITHIN a table; it does not make two tables agree. That is
    why the card table is a stamp beside the build and not a footnote.
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import NamedTuple

import pytest

from _pinned_count import collected_excluding_xfail, skip_reason, why_it_cannot_be_checked
from royalegym.protocol import derived_cards_vintage
from royalegym.rust_engine import CORE_IMPORT_ERROR, core_available
from royalegym.rust_engine import build_digest as engine_build_digest

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
#: BOTH STAMPS ARE READ FROM THE README, not copied into here, and the copies are why.
#: This file held `RECORDED_BUILD = "d872d792711934c2"` and
#: `RECORDED_VINTAGE = "retroroyale-2018"` while the page itself said `d6715210f21ca0c3`
#: and the 15.535 table. Docs re-pinned the page and nothing made this follow, so there
#: were two recorded builds and a reader got whichever they happened to open: train read
#: the constant, docs read the prose, and they reported different numbers for one figure.
#:
#: A stamp the page states and a test restates is two copies of a number, and they drift
#: exactly the way two copies of a rule do. The page is the publication, so the page is
#: the source; this parses what it says.
_PROVENANCE = re.compile(
    r"(?:re-run (\d{4}-\d{2}-\d{2}) )?on engine build `([0-9a-f]{8,64})` with the "
    r"\*\*([^*]+?)\s*card\s*\n?table\*\*",
    re.S,
)
#: The RoyaleSim commit the figure was measured against, which the page states because
#: the digest cannot: `build_digest` hashes the calibration and arena compiled in and
#: never sees the Rust. Read for the MESSAGES only. Pass/fail stays on the digest,
#: because nothing installed here can say which commit its engine was compiled from.
_SIM_COMMIT = re.compile(r"building the engine from\s+\*\*RoyaleSim `([0-9a-f]{7,40})`\*\*")

#: How old a pinned figure may get before its skip says so in capitals. A WORDING
#: threshold, never a pass/fail one: failing on age would put the red back on
#: RoyaleLearn's and RoyaleViser's cross-repo CI, which is what the skip exists to stop.
STALE_AFTER_DAYS = 7


class Provenance(NamedTuple):
    """What the README says its printed battle is a fact ABOUT."""

    build: str
    table: str
    measured_on: date | None
    sim_commit: str | None

    def age(self) -> str:
        """How long the figure has gone unchecked, in a form a skip line can carry.

        Days, not builds. Train asked for the number of builds a skip has spanned, and
        that is the better measure, but nothing here keeps a history of builds -- only
        the page's own date is available. A date is a floor on the gap, not the gap.
        """
        if self.measured_on is None:
            return "on an UNDATED recording, so nobody can say how long it has gone unchecked"
        days = (date.today() - self.measured_on).days
        prefix = f"STALE FOR {days} DAYS: " if days >= STALE_AFTER_DAYS else ""
        return f"{prefix}recorded {self.measured_on}, {days} day(s) ago"


def recorded_provenance() -> Provenance:
    """The stamps exactly as README.md states them."""
    text = README.read_text(encoding="utf-8")
    m = _PROVENANCE.search(text)
    assert m, (
        "README.md no longer states the build and card table its printed battle was "
        "recorded on, in the form 'on engine build `<digest>` with the **<table> card "
        "table**'. That sentence is the only record of what the figure is a fact ABOUT, "
        "so a figure without it cannot be checked or aged by anyone."
    )
    sim = _SIM_COMMIT.search(text)
    return Provenance(
        build=m.group(2),
        table=" ".join(m.group(3).split()),
        measured_on=date.fromisoformat(m.group(1)) if m.group(1) else None,
        sim_commit=sim.group(1) if sim else None,
    )


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
    """The program must RUN on any build; its printed battle is only checkable on the
    build it was recorded on.

    THE SPLIT IS THE POINT, and it was one assertion until 2026-09-23. A printed battle is
    the outcome of a whole simulation, so a different engine build produces a different
    one legitimately. Comparing them is comparing numbers from two different trees --
    exactly what the commit-pinned count rule in this same file already refuses, for the
    same reason.

    Treating a moved build as a FAILURE also put the red in the wrong place. This repo's
    suite runs in RoyaleLearn's and RoyaleViser's cross-repo CI, so a stale figure here
    turned every one of their commits red for something they cannot fix and did not cause.
    A check that cries wolf on other people's work is one they learn to ignore, which
    costs more than the stale figure it was guarding.

    What is NOT weakened: on the build the figure was recorded on, a changed battle is
    still a hard failure, and that is where a real regression shows. A build that moved is
    reported as an unchecked claim with both digests named, which is what it is.

    THE SKIP CAN GO QUIET, which is the cost of that repair and was pointed out the same
    night: a skip fires on every later build move, nothing re-records the pin, and after a
    few builds the figure ages out of anyone's view while the suite stays green. So the
    skip carries its own age, read from the date the page states, and says STALE in
    capitals past STALE_AFTER_DAYS. It still never fails on age -- that would move the
    red back onto other repos' CI -- but it can no longer be silent about how long nothing
    has checked the number.

    THE DIGEST IS BLIND TO THE RUST, and the page says so beside the figure: two engines
    with different code and identical calibration share a digest. So a failure here on a
    matching digest can mean the engine's CODE moved rather than that anything regressed,
    and the message names the recorded RoyaleSim commit so a reader checks that first.
    Behaviour changes in RoyaleSim normally arrive as a ledger arm, which moves the digest;
    one that did not would fail here and in RoyaleSim's own cross-repo run first, which is
    where that red belongs.
    """
    p = recorded_provenance()
    program, expected = try_it_program()
    done = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=300, cwd=REPO
    )
    assert done.returncode == 0, (
        f"the README's 'complete program' does not run:\n{done.stderr[-2000:]}"
    )
    measured_against = f"RoyaleSim {p.sim_commit}" if p.sim_commit else "an unnamed RoyaleSim"
    if engine_build_digest() != p.build:
        pytest.skip(
            f"SKIPPED, NOT PASSED: the README's battle was {p.age()}, on engine build "
            f"{p.build} ({measured_against}), and this engine is {engine_build_digest()}. "
            "The printed battle is the OUTCOME of a whole simulation, so a different build "
            "ends it differently and comparing the two would be comparing numbers from two "
            f"different trees. It prints {done.stdout.strip()!r} here. To re-pin: run the "
            "program on a COMMITTED build, paste its output under the block, and update "
            "the 're-run <date> on engine build ... with the ... card table' sentence and "
            "the RoyaleSim commit beside it -- every stamp is read from the page, so there "
            "is nothing else to change."
        )
    assert done.stdout.strip() == expected.strip(), (
        f"the README prints:\n  {expected.strip()}\nthe program prints:\n  "
        f"{done.stdout.strip()}\n\n"
        "The printed battle is the OUTCOME of a whole simulation, so it moves with "
        "anything the engine reads, not only with the cards. What it depends on, in the "
        "order worth checking:\n"
        f"  the engine code    recorded {measured_against}; the digest below CANNOT see "
        "it, so if you built from a different RoyaleSim, check whether any Rust moved "
        "before reading this as a regression\n"
        f"  the engine build   recorded {p.build}, now {engine_build_digest()} (data only)\n"
        f"  the card table     recorded {p.table!r}, now {derived_cards_vintage()!r}\n"
        "The build digest covers data/calibration.json and arena.json, so a ledger value "
        "moving is enough on its own: the starting elixir went from 5 to 6 on 2026-09-22 "
        "and turned this battle from a 0-0 draw at tick 4800 into 1-0 at tick 3600, with "
        "no card changed at all. Until that was written down, this message named only the "
        "card table and sent a reader to the wrong file.\n"
        "Re-record by running the program on a committed build, pasting its output under "
        "the block, and updating the 're-run <date> on engine build ...' sentence and the "
        "RoyaleSim commit beside it. Every stamp is read from the page."
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
    # THERE USED TO BE TWO COUNT BADGES and this looked for the one labelled as this
    # machine's. They existed because a clean clone and a developer machine ran DIFFERENT
    # populations: the clone built the 2018 card table and the laptop the 15.535 one, so
    # each badge was about a population the other side could not reproduce.
    #
    # On 2026-09-23 the documented install moved to the committed 15.535 table and both
    # sides converged -- same collection, same nine skips. The second badge then had
    # nothing left to say, docs collapsed them into one, and this assertion started
    # failing on a premise that had been retired rather than on anything being wrong.
    # It is a check outliving the fact it encoded, which is the thing this file is full
    # of warnings about, arriving in the file itself.
    #
    # So: find THE count badge, whatever it is labelled, and require exactly one. Two
    # again would mean two populations again, and this test would have to be told which
    # of them collection can check.
    readme = README.read_text("utf-8")
    counts = re.findall(r"-(\d+)%20passed%2C%20(\d+)%20skipped", readme)
    assert counts, "README.md states no test count at all; it used to state one"
    assert len(counts) == 1, (
        f"found {len(counts)} test-count badges. One is the documented shape. If a second "
        "population is being reported again, say which of them this check should compare "
        "against collection, because they cannot both be it."
    )
    claimed = int(counts[0][0]) + int(counts[0][1])

    at = re.search(r"%20at%20([0-9a-f]{7,40})-", readme)
    assert at, (
        "the count badge names no commit, so nothing can ever check it: a count without "
        "the tree it was taken on is a claim about a moving target. Label it "
        "`<where> at <sha>`."
    )
    # The rule itself lives in tests/_pinned_count.py, because docs/architecture.md
    # states a count too and two copies of a rule drift exactly like two copies of a
    # number. That file is also where the reasoning is written down.
    reason = why_it_cannot_be_checked(at.group(1))
    if reason:
        pytest.skip(skip_reason("the badge", counts[0][0], counts[0][1], at.group(1), reason))

    collected = collected_excluding_xfail()
    assert claimed == collected, (
        f"the README badge says {counts[0][0]} passed and {counts[0][1]} skipped, which "
        f"is {claimed} tests, and pytest collects {collected}. The badge is measured on a "
        "clean runner; since 2026-09-23 that runner and this machine collect the same "
        "population, so a difference here is a real one rather than two card tables."
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
    # THE LABEL IS NOT PINNED HERE, on purpose. This looked for `your clone` until
    # 2026-09-23, when the two badges became one and it was renamed to name the clean
    # runner. A test that hard-codes a label fails on a rename rather than on a defect,
    # and the rename was the correct response to the populations converging. What must
    # not change is the PROPERTY: a count nobody here can re-measure has to say where it
    # came from. So this matches any count badge and checks that, not its wording.
    readme = README.read_text("utf-8")
    clone = re.search(r"-(\d+)%20passed%2C%20(\d+)%20skipped", readme)
    assert clone, (
        "README.md no longer carries a test-count badge. It is the count a reader can "
        "reproduce, so if it was removed, say in the Status prose what replaced it. This "
        "test does not check its numbers against this machine and should not: they are "
        "measured on a clean runner."
    )
    assert int(clone.group(1)) > 0, "the count badge claims no passing tests"
    label = readme[: clone.start()].rsplit("badge/", 1)[-1]
    dated = re.search(r"\d{4}--\d{2}--\d{2}", label)
    committed = re.search(r"%20at%20[0-9a-f]{7,40}$", label)
    assert dated or committed, (
        f"the clone badge says {label!r}, which names no commit and no date. Nothing here "
        "can re-measure it, so the one thing it must carry is where it came from: a "
        "commit for preference, since that names the tree, or a date so it can be aged."
    )
