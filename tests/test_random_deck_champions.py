"""A random deck holds at most one champion, as the ladder allows.

The random dealer (``random_deck``: DefaultStateMutator with no decks, a deck curriculum with no
pool) drew eight cards from the whole catalogue. On 2026-10-02 one draw held four champions,
and the engine refused the battle (at most 3 ability buttons a side). ``CardInfo.champion``
says which cards are champions: from the engine's ``champion`` column where it states one, else
from the RoyaleSim 0.1.3 list. The engine test checks the flag against what the engine DOES
(a champion's deck entry gets an ability button with no forms), card by card.

SKIPS
    The engine test skips without the engine. Not a pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from royalegym import make_env
from royalegym.protocol import BLUE, DECK_SIZE, CardInfo, ability_row
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available
from royalegym.state_mutator import MAX_CHAMPIONS, random_deck

needs_engine = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))


def _catalogue(n: int, champions: set[int]) -> list[CardInfo]:
    return [
        CardInfo(c, f"c{c}", 3, 0, 1, 9000, False, 100, champion=c in champions)
        for c in range(n)
    ]


def test_a_random_deck_never_holds_two_champions():
    cards = _catalogue(16, set(range(8)))  # half the catalogue is champions
    for seed in range(500):
        deck = random_deck(np.random.default_rng(seed), cards)
        assert len(set(deck)) == DECK_SIZE
        assert sum(cards[c].champion for c in deck) <= MAX_CHAMPIONS == 1, (seed, deck)


def test_a_draw_the_rule_allows_is_the_same_draw_as_before():
    """Only a draw with two or more champions is drawn again: every other deck, for the same
    seed, is the one the dealer dealt before the rule."""
    cards = _catalogue(136, {65, 68, 69, 70, 72, 133, 134, 135})
    kept = 0
    for seed in range(300):
        before = [int(c) for c in np.random.default_rng(seed).choice(136, DECK_SIZE, False)]
        if sum(cards[c].champion for c in before) <= 1:
            assert random_deck(np.random.default_rng(seed), cards) == before, seed
            kept += 1
    assert kept > 200  # most draws hold at most one champion


def test_a_catalogue_too_short_of_other_cards_is_refused():
    with pytest.raises(ValueError, match="champion"):
        random_deck(np.random.default_rng(0), _catalogue(9, set(range(3))))


@needs_engine
def test_the_champion_flag_is_what_the_engine_does():
    """Card by card: a deck of the card and seven plain ones, no forms. The engine gives the
    card an ability button exactly when CardInfo.champion says it is a champion."""
    cards = RustEngine().cards()
    plain = ["Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Cannon", "Musketeer"]
    by_name = {c.name: c for c in cards}
    assert not any(by_name[n].champion for n in plain)
    flagged, buttoned, refused = set(), set(), []
    for card in cards:
        assert card.champion is not None, f"{card.name}: the engine adapter left it unknown"
        if card.champion:
            flagged.add(card.name)
        fillers = [n for n in plain if n != card.name][: DECK_SIZE - 1]
        deck = [card.name, *fillers]
        try:
            env = make_env(deck=deck)
            env.reset(seed=0)
        except ValueError as ex:  # a card the engine will not deal at all
            refused.append(f"{card.name}: {ex}")
            continue
        rows = [ability_row(r) for r in env.battle_state.players[BLUE].abilities]
        if any(r.card_id == card.card_id for r in rows):
            buttoned.add(card.name)
    assert flagged == buttoned, (sorted(flagged - buttoned), sorted(buttoned - flagged))
    assert len(flagged) >= 8
    # Name what the engine refused to deal, so a refusal never hides a champion.
    assert len(refused) <= 12, refused


@needs_engine
def test_random_decks_reset_on_the_engine():
    env = make_env(deck="random")
    for seed in range(60):
        env.reset(seed=seed)
        for p in env.battle_state.players:
            rows = [ability_row(r) for r in p.abilities]
            assert len([r for r in rows if r.card_id >= 0]) <= MAX_CHAMPIONS
