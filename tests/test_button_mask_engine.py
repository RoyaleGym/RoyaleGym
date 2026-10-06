"""A button the row calls available is offered only when the engine would take the press.

An ability row says ``available`` for a living hero off cooldown, but some buttons wait on more
than the row carries: the Hero Mega Minion's warp is taken only a while after the hero appears,
and only while it has a target (state.rs ``check_ability_button``, AbilityNotReady). A mask that
read the row alone offered the press from the hero's first tick, and the engine refused it: a
training run checking that every command its mask allowed was accepted stopped on it, about one
action in 52 000 (found 2026-10-05). The mask now asks the engine about every button the row
calls usable, as it already asks it where a building lands.
"""

from __future__ import annotations

import pytest

from royalegym.action import TileActionParser, mask_disagreements
from royalegym.protocol import BLUE, HAND_SIZE, DeployCommand, DeployStatus, MatchSetup, ability_row
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available
from royalegym.state_mutator import deck_ids

needs_engine = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
NAMES = ["MegaMinion", "Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Musketeer"]


@needs_engine
def test_the_mask_agrees_with_the_engine_on_the_warp_button() -> None:
    eng = RustEngine()
    deck = deck_ids(NAMES, eng.cards())
    try:
        eng.reset(1, MatchSetup(decks=[deck, deck], shuffle=0,
                                forms=[[2, 0, 0, 0, 0, 0, 0, 0], [0] * 8],
                                elixir_milli=[10000, 10000],
                                start_tick=eng.rules().deploy_lockout_ticks))
    except ValueError as exc:
        pytest.skip(f"SKIPPED, NOT PASSED: this card table deals no Hero Mega Minion ({exc})")
    parser = TileActionParser(ability_buttons=True)
    parser.bind(eng)
    t = eng.arena().subtile
    assert eng.step([DeployCommand(BLUE, 0, 9 * t, 10 * t)], 1)[0].status == DeployStatus.OK
    waited = taken = 0
    for _ in range(300):
        s = eng.state()
        assert mask_disagreements(eng, parser, s, BLUE) == [], s.tick
        rows = s.players[BLUE].abilities
        if rows and ability_row(rows[0]).available:
            verdict = eng.check_deploy(DeployCommand(BLUE, HAND_SIZE, 0, 0))
            waited += verdict == DeployStatus.ABILITY_NOT_READY
            taken += verdict == DeployStatus.OK
        if taken:
            break
        eng.step([], 1)
    assert waited, "vacuous: the row never called the button available while the engine waited"
