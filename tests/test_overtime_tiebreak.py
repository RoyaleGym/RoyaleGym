"""The post-overtime tiebreak in MockEngine (calibration match.OVERTIME_TIEBREAK).

The Rust engine's rule is pinned in ../RoyaleSim/crates/royalesim/tests/tiebreak.rs;
this file pins the mock to the same table so a parity run cannot diverge on the
last tick of overtime. Every scenario starts on the final overtime tick with
hand-set tower hp, so the verdict is the rule's alone.
"""

from __future__ import annotations

import copy

import pytest

from royalegym.mock_engine import MockEngine
from royalegym.protocol import BLUE, Calibration, MatchSetup, Winner, default_calibration

DECK = list(range(8))
RULES = ("lowest_tower_hp_absolute", "lowest_tower_hp_fraction", "none_draw")


def engine(rule: str) -> MockEngine:
    raw = copy.deepcopy(default_calibration().raw)
    raw["match"]["OVERTIME_TIEBREAK"]["value"] = rule
    return MockEngine(Calibration(raw))


def verdict(rule: str, blue: list[int], red: list[int]) -> int:
    """tower_hp is [team][TowerSlot] in each owner's OWN frame: [king, left, right]."""
    eng = engine(rule)
    last = eng.regular_ticks + eng.overtime_ticks - 1
    setup = MatchSetup(decks=[DECK, DECK], start_tick=last, tower_hp=[blue, red])
    eng.reset(5, setup)
    assert eng.state().overtime
    assert not eng.state().game_over
    eng.step([], 1)
    s = eng.state()
    assert s.game_over, "the final overtime tick must end the match"
    assert [p.crowns for p in s.players] == [0, 0]
    return s.winner


def tower_max() -> tuple[int, int]:
    eng = engine("none_draw")
    eng.reset(1, MatchSetup(decks=[DECK, DECK]))
    m = eng.state().players[BLUE].tower_max_hp
    return m[0], m[1]


def test_none_draw_keeps_the_old_verdict():
    assert verdict("none_draw", [1000, 100, 1000], [1000, 1000, 1000]) == Winner.DRAW


def test_absolute_the_weaker_weakest_tower_loses():
    assert verdict("lowest_tower_hp_absolute", [4000, 100, 1500], [4000, 900, 1200]) == Winner.RED
    assert verdict("lowest_tower_hp_absolute", [4000, 900, 1200], [4000, 100, 1500]) == Winner.BLUE


def test_fraction_is_exact_not_a_rounded_percentage():
    k, p = tower_max()
    assert verdict("lowest_tower_hp_fraction", [k, p - 1, p], [k, p, p]) == Winner.RED


def test_absolute_and_fraction_can_disagree_on_a_king_versus_a_princess():
    k, p = tower_max()
    assert verdict("lowest_tower_hp_absolute", [500, p, p], [k, 600, p]) == Winner.RED
    if 500 * p > 600 * k:
        expect = Winner.BLUE
    elif 500 * p < 600 * k:
        expect = Winner.RED
    else:
        expect = Winner.DRAW
    assert verdict("lowest_tower_hp_fraction", [500, p, p], [k, 600, p]) == expect


@pytest.mark.parametrize("rule", RULES)
def test_an_exact_tie_is_a_draw(rule):
    assert verdict(rule, [3000, 700, 1500], [3000, 1500, 700]) == Winner.DRAW


@pytest.mark.parametrize("rule", RULES)
def test_swapping_the_sides_swaps_the_winner(rule):
    boards = [
        ([4000, 100, 1500], [4000, 900, 1200]),
        ([500, 2000, 2000], [4000, 600, 2000]),
        ([2500, 2500, 2500], [2500, 2499, 2500]),
    ]
    swap = {Winner.BLUE: Winner.RED, Winner.RED: Winner.BLUE, Winner.DRAW: Winner.DRAW}
    for b, r in boards:
        assert verdict(rule, r, b) == swap[verdict(rule, b, r)], (rule, b, r)


def test_a_destroyed_princess_is_not_a_zero_hp_tower():
    # One princess gone per side (crowns level at 1-1); only standing towers rank.
    eng = engine("lowest_tower_hp_absolute")
    last = eng.regular_ticks + eng.overtime_ticks - 1
    hp = [[4000, 0, 800], [4000, 900, 0]]
    eng.reset(3, MatchSetup(decks=[DECK, DECK], start_tick=last, tower_hp=hp))
    assert [p.crowns for p in eng.state().players] == [1, 1]
    eng.step([], 1)
    s = eng.state()
    assert s.game_over
    assert s.winner == Winner.RED


def test_the_default_calibration_selects_the_absolute_rule():
    assert MockEngine().overtime_tiebreak == "lowest_tower_hp_absolute"


def test_an_unknown_rule_is_refused_at_construction():
    with pytest.raises(ValueError, match="OVERTIME_TIEBREAK"):
        engine("coin_flip")
