"""DefaultStateMutator takes decks by card NAME, looked up in the engine's catalogue.

Catalogue ids are positions and move when a card is added, so a name is the stable way to
write a deck. Ids still work, and the two can be mixed.
"""

from __future__ import annotations

import numpy as np
import pytest

from royalegym import DefaultStateMutator
from royalegym.mock_engine import MockEngine

CARDS = list(MockEngine().cards())
BY_NAME = {c.name: c.card_id for c in CARDS}
NAMES = [c.name for c in CARDS[:8]]


def test_names_resolve_to_the_catalogues_ids():
    setup = DefaultStateMutator(decks=[NAMES, NAMES[::-1]]).build(np.random.default_rng(0), CARDS)
    assert setup.decks == [[BY_NAME[n] for n in NAMES], [BY_NAME[n] for n in NAMES[::-1]]]


def test_ids_and_names_mix():
    mixed = [BY_NAME[NAMES[0]], *NAMES[1:]]
    setup = DefaultStateMutator(decks=[mixed, mixed]).build(np.random.default_rng(0), CARDS)
    assert setup.decks[0] == [BY_NAME[n] for n in NAMES]


def test_an_unknown_name_is_refused_by_name():
    deck = [*NAMES[:7], "NoSuchCard"]
    with pytest.raises(ValueError, match="names 'NoSuchCard'"):
        DefaultStateMutator(decks=[deck, deck]).build(np.random.default_rng(0), CARDS)


def test_config_keeps_the_names_as_given():
    assert DefaultStateMutator(decks=[NAMES, NAMES]).config()["decks"] == [NAMES, NAMES]
