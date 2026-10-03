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
    """The two rules return DIFFERENT winners on one board. Both verdicts written out.

    This is the only test in the file that distinguishes the rules, so it is the only
    thing standing between ``lowest_tower_hp_fraction`` and being silently replaced by
    the absolute rule.

    It did not distinguish them before. It computed what to expect from ``500 * p >
    600 * k`` -- the same cross-multiplication the fraction rule itself performs -- so
    the assertion restated the implementation and held whatever the rule returned. On
    this arena the BLUE branch was unreachable (it needs princess_max > 1.2 *
    king_max), both rules in fact returned RED, and aliasing fraction to absolute left
    all thirteen tests in this file green.

    The board that separates them: each side's weakest tower is a different KIND.
    Blue's is a princess on 500 of 1400, which is low in hp and 36% of its pool. Red's
    is the king on 700 of 2400, which is higher in hp and 29% of its pool. So absolute
    calls Blue's the weaker and fraction calls Red's, and the rules pick opposite
    losers. No arithmetic here repeats theirs: the two expected winners are constants.
    """
    k, p = tower_max()
    blue = [k, 500, p]  # full king, one badly damaged princess
    red = [700, p, p]  # damaged king, both princesses full
    assert verdict("lowest_tower_hp_absolute", blue, red) == Winner.RED
    assert verdict("lowest_tower_hp_fraction", blue, red) == Winner.BLUE


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


# -- client_hp_drain (RoyaleSim ship31's rule, measured on both clients) -------------
#
# A level overtime does not end at t6000. From 3350 ms past it (t6067), at the head of each
# tick, every standing crown tower, kings too, loses one step chosen by the lowest of all six
# (50 from 1000, 40 from 500, 20 from 200, 10 from 21, else 1). A tower at 0 falls and the
# crowns decide. Weakest towers exactly level: one drain at t6067, then a Draw at t6147. Every
# number below is worked by hand from that rule, not computed by the code under test.

DRAIN = "client_hp_drain"


def drained(blue: list[int], red: list[int], until: int | None = None):
    """Start at overtime's end (t6000) with hand-set tower hp, and step one tick at a time
    until the match ends or ``until`` ticks have run. Returns the final state."""
    eng = engine(DRAIN)
    end = eng.regular_ticks + eng.overtime_ticks
    assert end == 6000
    eng.reset(5, MatchSetup(decks=[DECK, DECK], start_tick=end, tower_hp=[blue, red]))
    assert eng.state().overtime
    for _ in range(until if until is not None else 400):
        eng.step([], 1)
        if eng.state().game_over:
            break
    return eng.state()


def hp(s) -> list[list[int]]:
    return [list(p.tower_hp) for p in s.players]


def test_the_drain_rule_is_accepted():
    assert engine(DRAIN).overtime_tiebreak == DRAIN


def test_the_drain_step_table():
    from royalegym.mock_engine import tiebreak_drain_step

    cases = {5000: 50, 1000: 50, 999: 40, 500: 40, 499: 20, 200: 20, 199: 10, 21: 10, 20: 1, 1: 1}
    assert {lowest: tiebreak_drain_step(lowest) for lowest in cases} == cases


def test_nothing_moves_until_t6067_then_every_tower_loses_the_lowest_ones_step():
    s = drained([4000, 900, 1200], [4000, 1000, 1200], until=67)  # ticks 6000..6066
    assert s.tick == 6067
    assert not s.game_over
    assert hp(s) == [[4000, 900, 1200], [4000, 1000, 1200]]
    s = drained([4000, 900, 1200], [4000, 1000, 1200], until=68)  # and 6067: lowest 900
    assert hp(s) == [[3960, 860, 1160], [3960, 960, 1160]]


def test_the_weaker_weakest_tower_drains_out_first_and_the_crowns_decide():
    # Blue's 100: eight steps of 10 (100 -> 20), then twenty of 1, so it falls on the
    # 28th drain tick, t6094. Red's 150 stands at 50.
    s = drained([4000, 100, 4000], [4000, 150, 4000])
    assert s.game_over
    assert s.winner == Winner.RED
    assert [p.crowns for p in s.players] == [0, 1]
    assert s.tick == 6095


def test_exactly_level_weakest_towers_drain_once_then_draw_at_t6147():
    s = drained([4000, 700, 1500], [4000, 1500, 700])
    assert s.game_over
    assert s.winner == Winner.DRAW
    assert s.tick == 6148
    assert hp(s) == [[3960, 660, 1460], [3960, 1460, 660]]


def test_one_one_becomes_two_one():
    # One princess down a side; Blue's 300 drains 20 a tick to 200, then 10 a tick, and
    # falls before Red's 400.
    s = drained([4000, 0, 300], [4000, 400, 0])
    assert s.game_over
    assert s.winner == Winner.RED
    assert [p.crowns for p in s.players] == [1, 2]


def test_swapping_the_sides_swaps_the_drains_winner():
    a = drained([4000, 100, 4000], [4000, 150, 4000])
    b = drained([4000, 150, 4000], [4000, 100, 4000])
    assert (a.winner, b.winner) == (Winner.RED, Winner.BLUE)
    assert a.tick == b.tick


def test_the_mock_runs_the_rule_the_ledger_names():
    """Whatever the engine's ledger selects (lowest_tower_hp_absolute until RoyaleSim ship31,
    client_hp_drain from it), the mock runs that rule, and it is one this file pins."""
    named = str(default_calibration().value("match.OVERTIME_TIEBREAK"))
    assert named in (*RULES, DRAIN), named
    assert MockEngine().overtime_tiebreak == named


def test_an_unknown_rule_is_refused_at_construction():
    with pytest.raises(ValueError, match="OVERTIME_TIEBREAK"):
        engine("coin_flip")
