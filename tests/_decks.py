"""Decks for the tests, written as card NAMES and resolved per engine.

A card id is a POSITION in one engine's catalogue. Two engines with different
catalogues give the same card different ids, and making one more card loadable
renumbers every card after it. So a deck spelled as numbers does not name the
same cards for long.

That is not a tidiness point. The decks below were picked for their SPREAD -- a
tank, a building, a rolling spell, two spells, a flying troop, a splash troop --
because the mask and the observation have to tell those apart. On a catalogue
where the numbers moved, the same list was eight ordinary troops, and every test
that meant to exercise a building or a spell exercised neither and passed.

``card_ids`` turns names into ids for whichever engine is under test, so a deck
means the same cards on all of them, and a card an engine does not have fails by
name on the spot. ``test_env_action_mask.py`` holds the guard that keeps the
spread honest.
"""

from __future__ import annotations

from collections.abc import Sequence

from royalegym.state_mutator import deck_ids

#: The spread: a cheap troop, a tank, a building, a rolling spell, two spells, a
#: flying troop, a splash troop. Four of the five placement classes, and both air
#: and ground. The fifth class is in MIXED_BARREL_DECK below.
MIXED_DECK = ("Knight", "Giant", "Cannon", "Log", "Fireball", "Zap", "Minions", "Valkyrie")

#: The same spread with Zap traded for the Goblin Barrel, whose placement class
#: (a spell refused only over water) is the one MIXED_DECK cannot also carry.
MIXED_BARREL_DECK = (
    "Knight",
    "Giant",
    "Cannon",
    "Log",
    "Fireball",
    "GoblinBarrel",
    "Minions",
    "Valkyrie",
)

#: A deck whose FIRST card is a spell, for tests that deploy hand slot 0 under
#: ShuffleMode.NONE and want the spell rules rather than the troop rules.
SPELL_FIRST_DECK = (
    "Fireball",
    "Knight",
    "Archer",
    "Goblins",
    "Giant",
    "MiniPekka",
    "Musketeer",
    "Skeletons",
)


def card_ids(names: Sequence[str], engine, where: str = "deck") -> list[int]:
    """Catalogue ids for ``names``, read off ``engine``'s own card table.

    ``engine`` is an engine or the ``cards()`` sequence from one. A name the
    catalogue does not have raises ValueError naming the card and the size of the
    table that was searched.
    """
    cards = engine.cards() if hasattr(engine, "cards") else engine
    return deck_ids(list(names), cards, where)


def card_id(name: str, engine) -> int:
    """The catalogue id of one named card."""
    return card_ids([name], engine, "card")[0]
