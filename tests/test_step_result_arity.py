"""The core's per-command tuple grew, and both arities have to keep working.

WHY THIS EXISTS
    On 2026-09-22 the compiled core began returning FIVE elements per command instead of
    three, the two new trailing ones being the resolved deploy position. This package
    unpacked exactly three with ``strict=True``, so every deploy raised ``too many values
    to unpack`` in every repo that uses it, the moment the extension was rebuilt. The
    trailing shape was chosen precisely so a tolerant reader would keep working. This
    reader was not one.

    Worse than the break: the working tree here was fixed within minutes while the
    PUBLISHED code still raised, so nothing anyone ran locally could see it. The
    integrator found it by running against origin.

WHY THE OLD ARM NEEDS A FAKE
    Every build on this machine has the new shape, so no ordinary engine test can reach
    the old arm. Compatibility that nothing exercises is a claim rather than a property,
    and it is the arm most likely to rot, because nothing will ever fail if it breaks. So
    it is driven through a stub core, and a second test pins what the REAL core returns,
    so the day that changes the stub is replaced rather than quietly left describing a
    world that has gone.

    Same treatment as the DUPLICATE_TEAM branch in test_landing.py, and for the same
    reason: a guard whose test can never fire is not a guard.

WHAT THE STUB'S SHAPE IS KNOWN FROM, and it is measured rather than reconstructed
    I first wrote that the old arm was UNEXERCISABLE. That was wrong, and the correction
    is worth more than the claim was. The integrator still had a clone whose venv carried
    a royalesim built before the rebuild -- build_digest f7628dd51148e4ce against this
    workspace's abac02d398ec90cb -- and ran real deploys through this wrapper on it:

        refused  (not enough elixir)   raw [(4, 4, 16)]   3 elements
        ACCEPTED (Knight, own half)    raw [(0, 0, 10)]   3 elements, reason 0

    The second row is the one that mattered. This stub defaults to reason 0, an ACCEPTED
    three-element row, and for a while that was the one case no real core had been seen
    to produce -- so the stub was modelling precisely the thing nobody could check. It is
    now measured. A fake is most dangerous exactly where it is most needed, because what
    it stands in for is what cannot be reached.

    The same clone also demonstrated why the fallback needs the flag, on the class that
    relocates: an accepted CANNON at (171000, 153000) came back through this wrapper as
    status OK at (171000, 153000). The tap, presented as a position, with nothing for a
    caller to interrogate. That is the false sentence the field's name writes, shown on a
    real pre-change core rather than argued from the measurement.

    A pre-change binary is a test fixture, and this project throws them away by default:
    every clone venv carries a dated engine and nobody keeps them on purpose. That clone
    is being kept deliberately now.
    """

from __future__ import annotations

import warnings

import pytest

from royalegym.protocol import DeployCommand, DeployStatus, MatchSetup
from royalegym.rust_engine import core_available

pytestmark = pytest.mark.skipif(
    not core_available(), reason="the compiled engine is not built; a skip here is not a pass"
)

TILE = 18000
TAP = (9 * TILE + TILE // 2, 8 * TILE + TILE // 2)


class ShortRowCore:
    """A stub of the core's ``Battle`` that returns the OLD three-element rows.

    Both this shape and this default are MEASURED against a real pre-change core
    (f7628dd51148e4ce): an accepted deploy there returns [(0, 0, 10)], three elements
    with reason 0. See the module docstring for why the accepted row specifically is
    the one that had to be checked.
    """

    def __init__(self, card_id: int = 7, reason: int = 0, tick: int = 0) -> None:
        self.row = (card_id, reason, tick)

    def step(self, wire, ticks):
        return [self.row for _ in wire]


def an_engine():
    from royalegym.rust_engine import RustEngine

    engine = RustEngine()
    engine.reset(seed=0, setup=MatchSetup(decks=[[0] * 8] * 2, elixir_milli=[10000] * 2))
    # Past the opening deploy lockout, not a flat 10 ticks: the probe tap below has to be
    # ACCEPTED for this to measure anything, and the test says so when it is not.
    engine.step([], max(10, engine.rules().deploy_lockout_ticks + 10))
    return engine


def test_the_real_core_returns_the_resolved_position() -> None:
    """Pins the arity the stub below exists to complement. If this goes red, the stub is
    describing a core that no longer exists and should be re-taken, not kept."""
    engine = an_engine()
    results = engine.step([DeployCommand(team=0, hand_slot=0, x=TAP[0], y=TAP[1])], 2)
    assert results[0].status == DeployStatus.OK, (
        f"the probe tap was refused as {DeployStatus(results[0].status).name}, so this "
        "measured nothing"
    )
    assert engine.reports_resolved_position is True, (
        "the compiled core no longer reports a resolved deploy position. If that is "
        "deliberate, the fallback in RustEngine.step becomes the only path and its "
        "warning becomes permanent noise; decide which rather than letting it drift."
    )


def test_an_old_core_still_works_and_says_it_is_falling_back() -> None:
    """The compatibility arm, which no real engine can reach any more."""
    engine = an_engine()
    engine._battle = ShortRowCore(card_id=7)
    command = DeployCommand(team=0, hand_slot=0, x=TAP[0], y=TAP[1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = engine.step([command], 2)
    assert len(results) == 1, "a three-element core must still produce one result per command"
    assert results[0].card_id == 7
    # The command's own point, which is what this field held before the core grew.
    assert (results[0].x, results[0].y) == TAP
    assert engine.reports_resolved_position is False
    messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert messages, (
        "falling back to the command was SILENT. That is the failure mode this exists for: "
        "the tap is a full tile or more from the landing on half of accepted building "
        "taps, so a silent fallback hands back a wrong number that looks right."
    )
    assert "resolved" in messages[0]


def test_the_warning_does_not_repeat_for_every_command() -> None:
    """A per-command warning on a training run is a log nobody reads.

    Checked because the obvious implementation warns inside the loop, and at one warning
    per deploy this would fire millions of times in a run.
    """
    engine = an_engine()
    engine._battle = ShortRowCore()
    command = DeployCommand(team=0, hand_slot=0, x=TAP[0], y=TAP[1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            engine.step([command], 1)
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime) == 1, (
        f"the fallback warned {len(runtime)} times over five steps. It should say so once "
        "per engine: a warning per deploy is noise a training run drowns in."
    )


def test_a_caller_can_tell_the_two_apart_before_reading_a_position() -> None:
    """The point of the flag. Without it the two answers are indistinguishable.

    Deliberately asserts the flag CHANGES with the core rather than that it is merely
    set: a property hardcoded to True would pass every other test in this file.
    """
    engine = an_engine()
    engine.step([DeployCommand(team=0, hand_slot=0, x=TAP[0], y=TAP[1])], 2)
    on_new = engine.reports_resolved_position
    engine._battle = ShortRowCore()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine.step([DeployCommand(team=0, hand_slot=0, x=TAP[0], y=TAP[1])], 1)
    on_old = engine.reports_resolved_position
    assert (on_new, on_old) == (True, False), (
        f"the flag read {on_new} against the real core and {on_old} against a "
        "three-element one. It has to distinguish them or it tells a caller nothing."
    )
