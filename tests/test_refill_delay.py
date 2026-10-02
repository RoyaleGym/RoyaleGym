"""A hand slot can stay EMPTY for a while after a play: the client refills one slot per period,
so a played card's slot holds -1 until the next card arrives. The memory must keep the cycle true
through that window, not shift it when the card leaves and again when the next one arrives.
"""

from __future__ import annotations

from royalegym.mock_engine import MockEngine
from royalegym.obs import MatchMemory
from royalegym.protocol import EMPTY_CARD, default_elixir_law

E = EMPTY_CARD


def _memory() -> MatchMemory:
    cards = list(MockEngine().cards())
    memory = MatchMemory(len(cards), default_elixir_law())
    memory.bind(cards)
    memory.start(0, 5000, 5000, [0, 1, 2, 3], 4)
    return memory


def _play(memory: MatchMemory, tick: int, card: int, hand: list[int], next_card: int) -> None:
    memory.advance(tick, 3600, False, [(tick - 1, card)], [])
    memory.show_own_hand(hand, next_card)


def _arrive(memory: MatchMemory, tick: int, hand: list[int], next_card: int) -> None:
    memory.advance(tick, 3600, False, [], [])
    memory.show_own_hand(hand, next_card)


def _learn_the_cycle(memory: MatchMemory) -> None:
    """Four plays with the slot refilled at once, as the engine did before the timer."""
    _play(memory, 10, 0, [4, 1, 2, 3], 5)
    _play(memory, 20, 1, [4, 5, 2, 3], 6)
    _play(memory, 30, 2, [4, 5, 6, 3], 7)
    _play(memory, 40, 3, [4, 5, 6, 7], 0)
    assert memory.own_cycle[:4] == [0, 1, 2, 3]


def test_instant_refills_keep_the_cycle_as_before():
    memory = _memory()
    _learn_the_cycle(memory)
    _play(memory, 50, 4, [0, 5, 6, 7], 1)
    assert memory.own_cycle[:4] == [1, 2, 3, 4]


def test_the_cycle_stays_true_while_the_slot_is_empty():
    memory = _memory()
    _learn_the_cycle(memory)
    # Card 4 is played; its slot is empty and card 0 is still next.
    _play(memory, 50, 4, [E, 5, 6, 7], 0)
    assert memory.own_cycle[0] == 0
    assert memory.own_cycle[1:4] == [1, 2, 3], memory.own_cycle
    # A step later the refill brings card 0 into the empty slot, and 1 is next.
    _arrive(memory, 60, [0, 5, 6, 7], 1)
    assert memory.own_cycle[:4] == [1, 2, 3, 4], memory.own_cycle
    assert memory.own_last_card == 4


def test_two_plays_before_one_refill():
    memory = _memory()
    _learn_the_cycle(memory)
    _play(memory, 50, 4, [E, 5, 6, 7], 0)
    _play(memory, 60, 5, [E, E, 6, 7], 0)
    assert memory.own_cycle[:4] == [0, 1, 2, 3], memory.own_cycle
    _arrive(memory, 70, [0, E, 6, 7], 1)  # one refill per period, lowest empty slot first
    _arrive(memory, 80, [0, 1, 6, 7], 2)
    assert memory.own_cycle[:4] == [2, 3, 4, 5], memory.own_cycle
