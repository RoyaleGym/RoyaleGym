"""RustEngine.unit_hitpoints: the engine's (role, unit name, hitpoints) rows for a card at a
level, passed through as they are. A consumer (RoyaleLearn's elixir potential) reads a unit's
hitpoints at the level it was played at, which the catalogue row gives only at its own."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, _core, core_available

pytestmark = pytest.mark.skipif(
    not core_available() or not hasattr(getattr(_core, "Battle", None), "unit_hitpoints"),
    reason=str(CORE_IMPORT_ERROR) if not core_available() else "this engine lists no unit rows",
)


def test_the_rows_are_the_engines_with_the_cards_own_row_first():
    eng = RustEngine()
    ids = {c.name: c.card_id for c in eng.cards()}
    knight = eng.cards()[ids["Knight"]]
    at = eng.card_level
    here, up = eng.unit_hitpoints(knight.card_id, at), eng.unit_hitpoints(knight.card_id, at + 1)
    # The own row at the catalogue's level is the catalogue's hitpoints; a level up is more.
    assert (here[0][0], here[0][2]) == ("own", knight.hitpoints), here
    assert up[0][0] == "own", up
    assert up[0][2] > here[0][2], (here, up)
    # A card that puts a second kind down lists it after its own row.
    gang = eng.unit_hitpoints(ids["GoblinGang"], at)
    assert [r[0] for r in gang][:2] == ["own", "second_summon"], gang
    # As the engine gives them, typed.
    for cid in (knight.card_id, ids["GoblinGang"], ids["Fireball"]):
        mine = eng.unit_hitpoints(cid, at)
        assert mine == [tuple(r) for r in eng._battle.unit_hitpoints(cid, at)]
        assert all(isinstance(r, str) and isinstance(n, str) and isinstance(hp, int)
                   for r, n, hp in mine)


def test_an_unknown_card_or_level_is_refused_and_an_older_build_says_so(monkeypatch):
    eng = RustEngine()
    with pytest.raises(ValueError, match="unknown card id"):
        eng.unit_hitpoints(10**6, eng.card_level)
    with pytest.raises(ValueError, match="level 99"):
        eng.unit_hitpoints(eng.cards()[0].card_id, 99)
    monkeypatch.setattr(eng, "_battle", SimpleNamespace())
    with pytest.raises(NotImplementedError, match="6909b6f"):
        eng.unit_hitpoints(0, eng.card_level)
