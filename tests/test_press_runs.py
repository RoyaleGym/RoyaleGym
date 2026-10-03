"""Under a command delay, the count charges the presses that RAN, from the engine's report.

A press accepted on one step runs ``delay`` ticks later, and the engine can refuse it then: the
hero died while it waited (NO_HERO), or a level overtime ended with it waiting (GAME_OVER). A
refused press is never paid. So where the engine reports each delayed command that ran or was
dropped during a step (``RustEngine.step_commands_run``, RoyaleSim 0.1.7 on), MatchMemory charges
exactly the presses that ran, at the tick they ran, and nothing it only saw accepted.

Without the report (an older engine), the memory charges every accepted press when it is due,
which over-charges a press refused at run: the strict xfail in test_command_delay.py says so.

SKIPS
    None here: the memory is driven with forged states. The engine's side is
    test_command_delay.py's exactness test.
"""

from __future__ import annotations

import msgspec

from royalegym.mock_engine import MockEngine
from royalegym.obs import MatchMemory
from royalegym.protocol import HAND_SIZE, ElixirLaw, MatchSetup, default_calibration

DELAY = 22
BUTTON = 0
SLOT = HAND_SIZE + BUTTON
PRICE = 2  # elixir


def _states():
    eng = MockEngine()
    eng.reset(1, MatchSetup(decks=[list(range(8)), list(range(8))], elixir_milli=[3000, 3000]))
    base = eng.state()
    hero = eng.cards()[3].card_id
    row = [1, 0, PRICE, hero, 0]

    def at(tick, rows=(row,)):
        players = [msgspec.structs.replace(p, abilities=[list(r) for r in rows])
                   for p in base.players]
        return msgspec.structs.replace(base, tick=tick, players=players)

    return eng, at


def _memory(eng, at, team=0):
    m = MatchMemory(len(eng.cards()), ElixirLaw.load(default_calibration()))
    m.bind(eng.cards())
    m.seed(at(100), team)
    m.delay = [DELAY, DELAY]
    return m


def _count_after(runs, presses=()):
    eng, at = _states()
    m = _memory(eng, at)
    # Red's press accepted at t100 (the step's list), run (or refused) at t122.
    m.observe(at(110), 0, presses=[(1, BUTTON)] if presses else [], runs=[])
    m.observe(at(130), 0, presses=[], runs=runs)
    return m.enemy_elixir_milli()


def test_a_press_refused_when_it_runs_is_never_charged():
    nothing = _count_after([], presses=())
    refused = _count_after([(122, 1, "ability", SLOT, 15)], presses=(1,))
    assert refused == nothing


def test_a_press_that_ran_is_charged_its_price():
    nothing = _count_after([], presses=())
    ran = _count_after([(122, 1, "ability", SLOT, 0)], presses=(1,))
    assert nothing - ran == PRICE * 1000


def test_the_own_side_and_plays_in_the_report_charge_no_enemy_press():
    nothing = _count_after([], presses=())
    other = _count_after([(122, 0, "ability", SLOT, 0), (122, 1, "deploy", 5, 0)], presses=(1,))
    assert other == nothing


def test_without_a_report_the_accepted_press_is_charged_when_due():
    """The older path, kept for engines without the report: charged at accept + delay."""
    eng, at = _states()
    m = _memory(eng, at)
    m.observe(at(110), 0, presses=[(1, BUTTON)])
    m.observe(at(130), 0, presses=[])
    assert _count_after([], presses=()) - m.enemy_elixir_milli() == PRICE * 1000
