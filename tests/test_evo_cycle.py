"""``CardInfo.evo_cycle``: the basic plays before each evolved play of a card's evolution.

RoyaleSim's ship35 gives the catalogue a trailing "evo_cycle" column, the same number as the 4th
value of a side's own evo rows (Evo Skeletons 2, Evo Barbarians 1), 0 for a card with no
loadable evolution. It lets the enemy's evolution charge be counted from its plays, which until
now could only be read for the own side's cards. None means "the engine did not say": an engine
before the column. MockEngine's 2018 game had no evolutions, so it says 0 for every card.

SKIPS
    The value checks skip on an engine without the column. Not a pass.
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


def test_mock_engine_says_no_card_evolves():
    assert {c.evo_cycle for c in MockEngine().cards()} == {0}


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_an_engine_without_the_column_says_nothing():
    if HAS_COLUMN:
        pytest.skip("this engine has the column")
    assert {c.evo_cycle for c in RustEngine().cards()} == {None}


@pytest.mark.skipif(not HAS_COLUMN, reason="this engine's catalogue has no evo_cycle column")
def test_the_column_matches_the_own_evo_rows():
    from royalegym import make_env

    eng = RustEngine()
    by_name = {c.name: c for c in eng.cards()}
    assert by_name["Skeletons"].evo_cycle == 2
    assert by_name["Barbarians"].evo_cycle == 1
    assert all(c.evo_cycle >= 0 for c in eng.cards())
    assert any(c.evo_cycle == 0 for c in eng.cards()), "some card has no evolution"
    env = make_env(deck=["Skeletons", "Barbarians", "Knight", "Archer", "Giant", "Minions",
                         "Fireball", "Zap"], evolved=["Skeletons", "Barbarians"])
    env.reset(seed=0)
    for card, _, _, cycle in (r[:4] for r in env.battle_state.players[0].evo):
        assert eng.cards()[card].evo_cycle == cycle, eng.cards()[card].name
