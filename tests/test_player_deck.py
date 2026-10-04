"""``PlayerState.deck`` and ``PlayerState.forms``: each side's whole deck and each card's form.

A player knows its own eight cards, and which are evolutions or heroes, from the first frame
(owner 2026-10-03: the bot knows every evolution and hero status of its own deck and hand). The
memory's ``own_deck`` only deduces the deck from the cycle, so at tick 0 it holds five cards.

- ``deck``: the side's 8 deck card ids; ``forms``: parallel to it, 0 basic, 1 evolution, 2 hero.
  Empty lists mean "the engine did not say".
- royalesim 0.1.9 sends both from the engine's own config. Until then RustEngine fills them from
  the MatchSetup it was reset with, and after ``load_state`` (a battle whose setup it never saw)
  it says nothing, rather than a stale deck.
- MockEngine reports the setup's order after a reset, as RustEngine does (the cross-engine state
  comparison reads every player field). After a load its hand and queue still hold the side's 8
  cards, so it reports them in card-id order. Forms all 0: it plays no forms.

SKIPS
    The engine test skips only without the engine.
"""

from __future__ import annotations

import pytest

from royalegym.mock_engine import MockEngine
from royalegym.protocol import BLUE, RED, MatchSetup, PlayerState, ShuffleMode
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available


def test_the_fields_default_to_not_said():
    p = PlayerState(0, 0, [0, 1, 2, 3], 4, 0, [1, 1, 1], [1, 1, 1], False)
    assert p.deck == []
    assert p.forms == []


def test_mock_engine_reports_each_sides_deck_at_any_time():
    eng = MockEngine()
    blue, red = [7, 3, 11, 0, 5, 9, 2, 14], [1, 4, 6, 8, 10, 12, 13, 15]
    eng.reset(1, MatchSetup(decks=[blue, red]))
    for team, deck in ((BLUE, blue), (RED, red)):
        p = eng.state().players[team]
        assert p.deck == deck
        assert p.forms == [0] * 8
    blob = eng.save_state()
    eng.reset(2, MatchSetup(decks=[red, blue]))
    eng.load_state(blob)
    assert eng.state().players[BLUE].deck == sorted(blue), "the deck survives a load, by id"


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_rust_engine_reports_the_setups_deck_and_forms_or_nothing_after_a_load():
    eng = RustEngine()
    ids = {c.name: c.card_id for c in eng.cards()}
    names = ("Skeletons", "Musketeer", "Knight", "Archer", "Giant", "Minions", "Fireball", "Zap")
    deck = [ids[n] for n in names]
    forms = [1, 2, 0, 0, 0, 0, 0, 0]
    eng.reset(1, MatchSetup(decks=[deck, list(reversed(deck))], forms=[forms, [0] * 8],
                            shuffle=ShuffleMode.INDEPENDENT))
    blue, red = eng.state().players
    assert blue.deck == deck
    assert blue.forms == forms
    assert red.deck == list(reversed(deck))
    assert red.forms == [0] * 8
    eng.reset(2, MatchSetup(decks=[deck, deck]))
    assert eng.state().players[BLUE].forms == [0] * 8, "no forms asked is all basic"
    blob = eng.save_state()
    eng.reset(3, MatchSetup(decks=[deck, deck], forms=[forms, forms]))
    eng.load_state(blob)
    after = eng.state().players[BLUE]
    if after.deck:  # an engine that reports them itself (royalesim 0.1.9 on)
        assert (after.deck, after.forms) == (deck, [0] * 8)
    else:
        assert after.forms == [], "a loaded battle's setup is unknown here, so nothing is said"
