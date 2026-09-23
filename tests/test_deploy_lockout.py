"""The opening deploy lockout, and that the mask and the engine agree about it.

A match refuses every deploy for its first ``DeployRules.deploy_lockout_ticks`` ticks
(``DeployStatus.TOO_EARLY``). Until 2026-09-23 the mask knew nothing about it, so on the
opening steps of every battle it offered all four cards while the engine refused all four.

WHY THAT IS NOT COSMETIC, AND WHAT IT DID *NOT* DO. ``IllegalActionPenalty`` fires when a
seat commands and nothing of its appears, so under this mask a policy would be punished for
obeying it on the opening steps of every battle, and would have to learn from that penalty a
rule the mask could simply have told it.

It did not do that to any run so far, and the distinction is worth keeping because the first
version of this note got it wrong. The lockout arrives WITH the 2026-09-23 rebuild; before
that the engine accepted a tick-0 deploy, so there was no disagreement to pay for. Learn
checked the completed run: ``illegal_commands`` 0.000 across 9,048 episodes, and
``IllegalActionPenalty`` is not in that objective at all. The defect is real and
forward-looking, which is a different claim from the one a reader would make from "was
punished", and only the first is supported.

The disagreement itself is the one thing ``DeployRules`` exists to prevent -- its own
docstring calls itself "placement rules the action mask and the engine MUST share".

THE AGREEMENT TEST IS THE POINT, not the two bookend tests. Checking "the mask is empty
before tick N" against a constant would pin the mask to a number I typed; checking that
nothing the mask offers is refused by the engine grades it against the engine, and would
catch the mask being wrong in either direction at any boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

from royalegym.action import NOOP, TileActionParser
from royalegym.mock_engine import MockEngine
from royalegym.protocol import DeployCommand, DeployStatus, MatchSetup
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available

pytestmark = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))

DECK = list(range(8))


def bound(engine):
    parser = TileActionParser()
    parser.bind(engine)
    return parser


def fresh(engine, ticks: int):
    engine.reset(seed=0, setup=MatchSetup(decks=[DECK, DECK], elixir_milli=[10000, 10000]))
    if ticks:
        engine.step([], ticks)
    return engine.state()


def test_the_rules_carry_the_lockout_the_engine_was_built_with():
    """Read from calibration, not hardcoded: 0 is a real arm and must stay expressible."""
    engine = RustEngine()
    lockout = engine.rules().deploy_lockout_ticks
    assert isinstance(lockout, int)
    assert lockout >= 0
    if lockout == 0:
        pytest.skip(
            "SKIPPED, NOT PASSED: this engine was built with DEPLOY_LOCKOUT_TICKS=0, the "
            "arm that has no lockout, so the tests below would pass without exercising it."
        )


def test_before_the_lockout_the_mask_offers_nothing_but_the_no_op():
    engine = RustEngine()
    lockout = engine.rules().deploy_lockout_ticks
    if lockout == 0:
        pytest.skip("no lockout on this build")
    state = fresh(engine, 0)
    parser = bound(engine)
    for team in (0, 1):
        mask = parser.action_mask(state, team)
        assert mask[NOOP] == 1, "the no-op must stay available or the seat cannot act at all"
        assert mask.sum() == 1, (
            f"seat {team} was offered {int(mask.sum()) - 1} deploys at tick 0, and the "
            "engine refuses every one of them"
        )


def test_after_the_lockout_the_mask_offers_deploys_again():
    """The mirror, so a mask that simply never offered anything would not pass."""
    engine = RustEngine()
    lockout = engine.rules().deploy_lockout_ticks
    state = fresh(engine, lockout + 10)
    parser = bound(engine)
    mask = parser.action_mask(state, 0)
    assert mask.sum() > 1, (
        "no deploy was offered after the lockout, so the bookend above proves nothing"
    )


def test_nothing_the_mask_offers_is_refused_as_TOO_EARLY():
    """Graded against the engine across the boundary, which is the real check.

    Sweeps ticks either side of the lockout and, on each, plays the FIRST action the mask
    offers. A mask that lifted one tick late would leave a legal action unoffered; one that
    lifted a tick early would have the engine refuse an offered action. Both show up here.
    """
    engine = RustEngine()
    lockout = engine.rules().deploy_lockout_ticks
    if lockout == 0:
        pytest.skip("no lockout on this build")
    parser = bound(engine)
    offered_somewhere = refused = checked = 0
    for tick in range(max(0, lockout - 3), lockout + 4):
        state = fresh(engine, tick)
        assert state.tick == tick, f"asked for tick {tick}, engine is at {state.tick}"
        mask = parser.action_mask(state, 0)
        actions = np.flatnonzero(mask)
        actions = actions[actions != NOOP]
        if actions.size == 0:
            continue
        offered_somewhere += 1
        command = parser.parse(int(actions[0]), state, 0)
        assert command is not None
        results = engine.step([command], 1)
        checked += 1
        if results and results[0].status == DeployStatus.TOO_EARLY:
            refused += 1
    assert checked, "no offered action was ever played, so nothing was compared"
    assert offered_somewhere, "the mask offered nothing at any tick in the sweep"
    assert refused == 0, (
        f"{refused} of {checked} actions the mask OFFERED were refused by the engine as "
        f"TOO_EARLY; the mask lifts its lockout at a different tick than the engine does "
        f"(rules say {lockout})"
    )


def test_an_engine_that_states_no_lockout_is_unchanged():
    """MockEngine has none, and 0 must mean 'no rule' rather than 'lock everything'."""
    engine = MockEngine()
    assert engine.rules().deploy_lockout_ticks == 0
    state = fresh(engine, 0)
    mask = bound(engine).action_mask(state, 0)
    assert mask.sum() > 1, (
        "MockEngine offers nothing at tick 0; a default of 0 has been read as a lockout "
        "and every Mock-backed battle would start unable to act"
    )


def test_the_engine_refuses_a_deploy_inside_the_lockout():
    """The premise everything above rests on, asserted rather than assumed."""
    engine = RustEngine()
    lockout = engine.rules().deploy_lockout_ticks
    if lockout == 0:
        pytest.skip("no lockout on this build")
    fresh(engine, 0)
    results = engine.step([DeployCommand(team=0, hand_slot=0, x=9 * 18000, y=5 * 18000)], 1)
    assert results[0].status == DeployStatus.TOO_EARLY, (
        f"a tick-0 deploy returned {DeployStatus(results[0].status).name}, so the lockout "
        "this module is about does not exist on this build"
    )
