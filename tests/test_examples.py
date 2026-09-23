"""Every example runs, and prints what it says it prints.

WHY THIS EXISTS
    An example that does not run is worse than no example: it is the first thing a
    reader tries, and when it fails they cannot tell whether they installed the library
    wrong or the library is wrong. Examples rot faster than anything else in a repo
    because nothing imports them, so nothing notices a renamed argument.

WHAT IT CATCHES
    An example that raises, and an example that runs but no longer demonstrates its
    point -- each one is checked for a line that only appears if the thing it exists to
    show actually happened. "It exited zero" is not enough: 06 would exit zero while
    printing that resuming does NOT work.

WHAT IT CANNOT CATCH
    Whether the example teaches well, or whether its prose is right. And it runs each
    one once, on this machine's card table, so an example that is right here and wrong
    on another catalogue passes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from royalegym.rust_engine import CORE_IMPORT_ERROR, core_available

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"

pytestmark = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))

#: The line that proves each example did the thing it is there to demonstrate, not
#: merely that python reached the end of the file.
#:
#: The first-step count was 1605 until 2026-09-22, when the engine started RELOCATING a
#: building whose tile box does not fit rather than refusing the tap, and the mask stopped
#: treating the bodies already on the board as a rule for buildings. Measured on this
#: example's own deck and seed, counting each hand slot under both rules: the whole
#: difference is the Cannon, 222 legal tiles before and 240 after, and Zap, Fireball and
#: Archer do not move at all. So this number changing again means a placement rule
#: changed, and the per-slot breakdown says which card.
PROOF = {
    # The first step now offers only the no-op -- a match refuses every deploy for its
    # opening ticks -- so the informative figure moved to the step where play opens. 1623
    # is the same count as before; what changed is which step it describes.
    "01_one_battle.py": "legal moves once play opens, on step 9: 1623 of 2305",
    "02_one_seat_against_a_bot.py": "against rush-left",
    "03_self_play_batch.py": "agent-episodes finished in 400 vector steps",
    "04_your_own_reward.py": "% of the signal",
    "05_record_and_replay.py": "re-run on a fresh engine: 0 divergences",
    "06_resume_a_run.py": "resumed run matches the original row for row: True",
    "07_is_this_bot_better.py": "too close to call",
    # NO NUMBER IN THIS ONE, unlike its neighbours, and for a reason worth keeping: the
    # example measures rates that depend on the card table, so pinning "10 cards" or
    # "51.7%" here would fail on a machine with a different vintage for a reason that has
    # nothing to do with placement. The sentence below is printed only when every card of
    # a footprint agreed AND a second footprint was present as a control, so it is the
    # finding rather than a reading of it.
    "measure_building_relocation.py": (
        "relocation is a property of the footprint, not the card"
    ),
}


def example_files() -> list[Path]:
    return sorted(p for p in EXAMPLES.glob("*.py") if not p.name.startswith("_"))


def test_every_example_is_covered_here() -> None:
    """A new example with no entry in PROOF would otherwise never be run.

    The vacuity guard for the whole file: without it, adding an example and forgetting
    to list it leaves this suite green and the example unchecked.
    """
    names = {p.name for p in example_files()}
    assert names == set(PROOF), (
        f"examples/ and PROOF disagree. Only on disk: {sorted(names - set(PROOF))}. "
        f"Only in PROOF: {sorted(set(PROOF) - names)}"
    )


@pytest.mark.parametrize("name", sorted(PROOF), ids=lambda n: n[:-3])
def test_the_example_runs_and_shows_what_it_claims(name: str) -> None:
    done = subprocess.run(
        [sys.executable, str(EXAMPLES / name)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=REPO,
    )
    assert done.returncode == 0, f"examples/{name} failed:\n{done.stderr[-2000:]}"
    assert PROOF[name] in done.stdout, (
        f"examples/{name} ran but did not show what it exists to show.\n"
        f"expected to find: {PROOF[name]!r}\ngot:\n{done.stdout[-1500:]}"
    )
