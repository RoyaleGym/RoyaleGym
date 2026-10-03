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
    The engine-backed tests skip on an engine without per-side levels. Not a pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from royalegym.mock_engine import MockEngine
from royalegym.protocol import DECK_SIZE, MatchSetup, ShuffleMode, setup_violation
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, _core, core_available
from royalegym.state_mutator import DefaultStateMutator

HAS_LEVELS = core_available() and "tower_levels" in (
    getattr(_core.Battle.reset, "__text_signature__", "") or ""
)
needs_levels = pytest.mark.skipif(
    not HAS_LEVELS,
    reason=str(CORE_IMPORT_ERROR) if not core_available() else "this engine has no per-side levels",
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


@pytest.mark.skipif(not core_available() or HAS_LEVELS, reason="needs an engine WITHOUT levels")
def test_an_engine_without_levels_refuses_them_by_name():
    eng = RustEngine()
    with pytest.raises(NotImplementedError, match="per-side levels"):
        eng.reset(1, MatchSetup(decks=[DECK, DECK], tower_levels=[11, 11]))


@needs_levels
def test_a_higher_side_gets_stronger_units_and_towers():
    eng = RustEngine()
    ids = {c.name: c.card_id for c in eng.cards()}
    deck = [ids[n] for n in ("Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Cannon",
                             "Musketeer")]
    card, tower = eng.card_level, eng._battle.tower_level()

    def tower_max_hp(levels, tower_levels):
        eng.reset(1, MatchSetup(decks=[deck, deck], shuffle=ShuffleMode.NONE,
                                levels=levels, tower_levels=tower_levels,
                                start_tick=eng.rules().deploy_lockout_ticks))
        return [list(p.tower_max_hp) for p in eng.state().players]

    same = tower_max_hp(None, None)
    assert same[0] == same[1]
    tilted = tower_max_hp([[card + 1] * DECK_SIZE, []], [tower + 1, tower])
    assert all(b > r for b, r in zip(tilted[0], tilted[1], strict=True)), (
        f"Blue's towers one level up are not stronger: {tilted}"
    )
    assert tilted[1] == same[1], "Red kept the engine's level"
