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
PROOF = {
    "01_one_battle.py": "legal moves on the first step: 1605 of 2305",
    "02_one_seat_against_a_bot.py": "against rush-left",
    "03_self_play_batch.py": "agent-episodes finished in 400 vector steps",
    "04_your_own_reward.py": "% of the signal",
    "05_record_and_replay.py": "re-run on a fresh engine: 0 divergences",
    "06_resume_a_run.py": "resumed run matches the original row for row: True",
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
