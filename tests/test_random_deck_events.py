"""A random deck holds no event card, and deals what it dealt before the event cards loaded.

royalesim 0.1.24 loads the nine event cards of the 15.535 table (``EVENT_CARDS``) at the end of
the default catalogue. No ladder deck holds one, so the random dealer (``random_deck``:
DefaultStateMutator with no decks, a deck curriculum with no pool) leaves them out unless asked
(``events=True``), and draws over the rest in catalogue order: a seed deals the deck it dealt
from the 136-card catalogue of 0.1.23. A deck that names an event card still plays it.

SKIPS
    The engine test skips without the engine. Not a pass.
"""

from __future__ import annotations

import msgspec
import numpy as np
import pytest

from royalegym.protocol import DECK_SIZE, CardInfo
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available
from royalegym.state_mutator import (
    EVENT_CARDS,
    DeckCurriculumStateMutator,
    DefaultStateMutator,
    random_deck,
)

needs_engine = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))


def _catalogue(n: int, champions: set[int]) -> list[CardInfo]:
    return [
        CardInfo(c, f"c{c}", 3, 0, 1, 9000, False, 100, champion=c in champions)
        for c in range(n)
    ]


def _with_events(cards: list[CardInfo]) -> list[CardInfo]:
    """``cards`` with the nine event cards appended, as the engine lists them."""
    n = len(cards)
    return [*cards, *(msgspec.structs.replace(cards[0], card_id=n + k, name=name, champion=False)
                      for k, name in enumerate(sorted(EVENT_CARDS)))]


def test_a_seed_deals_the_deck_it_dealt_before_the_event_cards():
    before = _catalogue(136, {65, 68, 69, 70, 72, 133, 134, 135})
    after = _with_events(before)
    for seed in range(500):
        old = random_deck(np.random.default_rng(seed), before)
        new = random_deck(np.random.default_rng(seed), after)
        assert new == old, seed
        assert not any(after[c].name in EVENT_CARDS for c in new), seed


def test_events_true_draws_them_and_a_named_deck_keeps_them():
    cards = _with_events(_catalogue(20, set()))
    drawn = set()
    for seed in range(200):
        deck = random_deck(np.random.default_rng(seed), cards, events=True)
        assert len(set(deck)) == DECK_SIZE
        drawn |= {cards[c].name for c in deck} & EVENT_CARDS
    assert drawn == EVENT_CARDS  # vacuity: the filter is what keeps them out
    named = ["SuperWitch", "c1", "c2", "c3", "c4", "c5", "c6", "GlobalClone"]
    setup = DefaultStateMutator(decks=[named, named]).build(np.random.default_rng(0), cards)
    assert {cards[c].name for c in setup.decks[0]} >= {"SuperWitch", "GlobalClone"}


def test_a_catalogue_of_only_event_cards_is_refused():
    cards = _with_events(_catalogue(7, set()))
    with pytest.raises(ValueError, match="a random deck may hold"):
        random_deck(np.random.default_rng(0), cards)
    assert len(random_deck(np.random.default_rng(0), cards, events=True)) == DECK_SIZE


@needs_engine
def test_the_engine_s_event_cards_never_reach_a_random_deck():
    cards = RustEngine().cards()
    names = {c.name for c in cards}
    present = EVENT_CARDS & names
    # Every name or none: a misspelled name would leave a strict subset on royalesim 0.1.24 on.
    assert present in (frozenset(), EVENT_CARDS), sorted(EVENT_CARDS - names)
    plain = [c for c in cards if c.name not in EVENT_CARDS]
    assert [c.card_id for c in plain] == list(range(len(plain)))  # the events come last
    main = ["Knight", "Archer", "Goblins", "Giant", "Pekka", "Minions", "Balloon", "Witch"]
    curriculum = DeckCurriculumStateMutator(main, p=0.0)  # never the main deck: random decks
    for seed in range(300):
        rng = np.random.default_rng(seed)
        deck = random_deck(rng, cards)
        assert not {cards[c].name for c in deck} & EVENT_CARDS, seed
        assert deck == random_deck(np.random.default_rng(seed), plain), seed
        for setup in (DefaultStateMutator().build(rng, cards), curriculum.build(rng, cards)):
            for d in setup.decks:
                assert not {cards[c].name for c in d} & EVENT_CARDS, seed
