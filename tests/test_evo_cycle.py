"""``CardInfo.evo_cycle``: the basic plays before each evolved play of a card's evolution.

royalesim 0.1.8 gives the catalogue a trailing "evo_cycle" column, the same number as the 4th
value of a side's own evo rows (Evo Skeletons 2, Evo Barbarians 1), 0 for a card with no
loadable evolution. It lets the enemy's evolution charge be counted from its plays, which until
now could only be read for the own side's cards. None means "the engine did not say": an engine
before the column, and MockEngine, which models no evolutions and so states no cycle. (It once
said 0, "no evolution", which is a claim about the card, and the 2018 cross-engine row caught it
disagreeing with an engine that does state the card's cycle.)

SKIPS
    The engine test skips only without the engine. With one, it checks the column's values or
    its absence, whichever that engine has: it never skips where CI builds it.
"""

from __future__ import annotations

import pytest

from royalegym.mock_engine import MockEngine
from royalegym.protocol import CardInfo
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, _core, core_available

HAS_COLUMN = core_available() and "evo_cycle" in getattr(_core, "CATALOGUE_FIELDS", ())


def test_the_field_trails_card_info_and_defaults_to_not_said():
    assert CardInfo.__struct_fields__[-1] == "evo_cycle"
    assert CardInfo(0, "x", 1, 0, 1, 0, False, 1).evo_cycle is None


def test_mock_engine_states_no_cycle():
    assert {c.evo_cycle for c in MockEngine().cards()} == {None}


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_the_engine_states_each_cards_cycle_or_says_nothing():
    """ONE test for both kinds of engine, so it never skips where the engine is built: with the
    column, each card's cycle is the one its own evo rows play by; without it, every card says
    None rather than a 0 that would read as "no evolution"."""
    from royalegym import make_env

    eng = RustEngine()
    if not HAS_COLUMN:
        assert {c.evo_cycle for c in eng.cards()} == {None}
        return
    by_name = {c.name: c for c in eng.cards()}
    assert by_name["Skeletons"].evo_cycle == 2
    assert by_name["Barbarians"].evo_cycle == 1
    assert all(c.evo_cycle >= 0 for c in eng.cards())
    assert any(c.evo_cycle == 0 for c in eng.cards()), "some card has no evolution"
    env = make_env(deck=["Skeletons", "Barbarians", "Knight", "Archer", "Giant", "Minions",
                         "Fireball", "Zap"], evolved=["Skeletons", "Barbarians"])
    env.reset(seed=0)
    rows = env.battle_state.players[0].evo
    assert len(rows) == 2, rows
    for card, _, _, cycle in (r[:4] for r in rows):
        assert eng.cards()[card].evo_cycle == cycle, eng.cards()[card].name
