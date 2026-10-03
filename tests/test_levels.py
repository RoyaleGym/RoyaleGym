"""Per-side card and tower levels: ``MatchSetup(levels=..., tower_levels=...)``.

A real deck's cards each have their own level, and a ladder opponent's are not yours. RoyaleSim
0.1.7 takes them per side at reset (``Battle.reset(levels, tower_levels)``):
- ``levels``: [blue, red], each empty (the engine's ``card_level`` for every card) or one
  unified level per deck card, parallel to ``decks``;
- ``tower_levels``: [blue, red], each side's crown towers' level.
None plays both sides at the engine's levels, as before.

The SHAPE is ``setup_violation``'s rule, with the core's own messages, so both engines refuse a
malformed setup with one text and before touching any state. Which levels exist is the
engine's (a card's ladder), refused by the core. MockEngine plays every card and tower at CSV
level 1 and has no level model, so it refuses any level it is given.

SKIPS
    The engine test skips only without the engine. With one, it checks levels played or
    refused by name, whichever that engine must do: it never skips where CI builds it.
"""

from __future__ import annotations

import numpy as np
import pytest

from royalegym.mock_engine import MockEngine
from royalegym.protocol import (
    BLUE,
    DECK_SIZE,
    RED,
    DeployCommand,
    DeployStatus,
    MatchSetup,
    ShuffleMode,
    setup_violation,
    to_engine,
)
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, _core, core_available
from royalegym.state_mutator import DefaultStateMutator

HAS_LEVELS = core_available() and "tower_levels" in (
    getattr(_core.Battle.reset, "__text_signature__", "") or ""
)
DECK = list(range(DECK_SIZE))


def _why(**kw):
    eng = MockEngine()
    return setup_violation(eng.arena(), eng.cards(), MatchSetup(decks=[DECK, DECK], **kw))


def test_none_and_well_formed_levels_pass_the_shape_rule():
    assert _why() is None
    assert _why(levels=[[], []], tower_levels=[11, 11]) is None
    assert _why(levels=[[11] * DECK_SIZE, []]) is None
    assert _why(levels=[[9] * DECK_SIZE, [14] * DECK_SIZE], tower_levels=[9, 14]) is None


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"levels": [[11] * DECK_SIZE]}, "levels must be [blue, red]"),
        ({"levels": [[11] * 7, []]}, "levels[0] has 7 entries for a deck of 8"),
        ({"levels": [[], [11] * 9]}, "levels[1] has 9 entries for a deck of 8"),
        ({"tower_levels": [11]}, "tower_levels must be [blue, red]"),
        ({"tower_levels": [11, 11, 11]}, "tower_levels must be [blue, red]"),
        ({"levels": [[2**31] + [11] * 7, []]}, "level 2147483648 does not fit i32"),
        ({"tower_levels": [11, -(2**31) - 1]}, "level -2147483649 does not fit i32"),
    ],
)
def test_a_malformed_level_is_refused_with_the_cores_text(kw, message):
    why = _why(**kw)
    assert why is not None
    assert message in why


def test_mock_engine_refuses_levels_and_keeps_its_battle():
    eng = MockEngine()
    eng.reset(1, MatchSetup(decks=[DECK, DECK]))
    before = eng.state()
    for kw in ({"levels": [[11] * DECK_SIZE, []]}, {"tower_levels": [11, 11]}):
        with pytest.raises(NotImplementedError, match="MockEngine models no levels"):
            eng.reset(2, MatchSetup(decks=[DECK, DECK], **kw))
        assert eng.state() == before
    eng.reset(3, MatchSetup(decks=[DECK, DECK], levels=[[], []]))  # empty asks for nothing


def test_the_state_mutator_carries_levels_into_the_setup():
    m = DefaultStateMutator(decks=[DECK, DECK], levels=[[12] * DECK_SIZE, []], tower_levels=[12, 9])
    setup = m.build(np.random.default_rng(0), MockEngine().cards())
    assert setup.levels == [[12] * DECK_SIZE, []]
    assert setup.tower_levels == [12, 9]
    assert m.config()["levels"] == [[12] * DECK_SIZE, []]
    assert m.config()["tower_levels"] == [12, 9]
    plain = DefaultStateMutator(decks=[DECK, DECK])
    assert plain.build(np.random.default_rng(0), MockEngine().cards()).levels is None
    assert "levels" not in plain.config()


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_the_engine_plays_per_side_levels_or_refuses_them_by_name():
    """ONE test for both kinds of engine, so it never skips where the engine is built: an
    engine with levels must play them, and one without must refuse them by name rather
    than start a battle at levels nobody asked for."""
    eng = RustEngine()
    ids = {c.name: c.card_id for c in eng.cards()}
    deck = [ids[n] for n in ("Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Cannon",
                             "Musketeer")]
    if not HAS_LEVELS:
        with pytest.raises(NotImplementedError, match="per-side levels"):
            eng.reset(1, MatchSetup(decks=[deck, deck], tower_levels=[11, 11]))
        eng.reset(1, MatchSetup(decks=[deck, deck], levels=[[], []]))  # asks for nothing
        return
    card, tower = eng.card_level, eng._battle.tower_level()

    def battle(levels, tower_levels):
        eng.reset(1, MatchSetup(decks=[deck, deck], shuffle=ShuffleMode.NONE,
                                levels=levels, tower_levels=tower_levels,
                                start_tick=eng.rules().deploy_lockout_ticks))
        towers = [list(p.tower_max_hp) for p in eng.state().players]
        t = eng.arena().subtile
        plays = []
        for team in (BLUE, RED):
            x, y = to_engine(eng.arena(), team, 9 * t + t // 2, 10 * t + t // 2)
            plays.append(DeployCommand(team, eng.state().players[team].hand.index(ids["Knight"]),
                                       x, y))
        assert all(r.status == DeployStatus.OK for r in eng.step(plays, 1))
        eng.step([], 30)
        knights = {e.team: e.max_hp for e in eng.state().entities if e.card_id == ids["Knight"]}
        return towers, knights

    same, same_knights = battle(None, None)
    assert same[0] == same[1]
    assert same_knights[BLUE] == same_knights[RED]
    tilted, knights = battle([[card + 1] * DECK_SIZE, []], [tower + 1, tower])
    assert all(b > r for b, r in zip(tilted[0], tilted[1], strict=True)), (
        f"Blue's towers one level up are not stronger: {tilted}"
    )
    assert tilted[1] == same[1], "Red kept the engine's tower level"
    assert knights[BLUE] > knights[RED], f"Blue's Knight one level up is not stronger: {knights}"
    assert knights[RED] == same_knights[RED], "Red kept the engine's card level"
