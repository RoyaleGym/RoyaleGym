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
    # A play's ``what`` is a card id, here one that reads as this button's slot if misread.
    other = _count_after([(122, 0, "ability", SLOT, 0), (122, 1, "deploy", SLOT, 0)],
                         presses=(1,))
    assert other == nothing


def test_without_a_report_the_accepted_press_is_charged_when_due():
    """The older path, kept for engines without the report: charged at accept + delay."""
    eng, at = _states()
    m = _memory(eng, at)
    m.observe(at(110), 0, presses=[(1, BUTTON)])
    m.observe(at(130), 0, presses=[])
    assert _count_after([], presses=()) - m.enemy_elixir_milli() == PRICE * 1000


def _builder_count(runs_second):
    from royalegym.action import TileActionParser
    from royalegym.obs import SpatialObsBuilder

    eng, at = _states()
    parser = TileActionParser()
    parser.bind(eng)
    b = SpatialObsBuilder()
    b.bind(eng, parser)
    b.command_delay = (DELAY, DELAY)
    b.reset(at(100))
    mask = parser.action_mask(at(100), 0)
    b.see_presses([(1, BUTTON)])
    b.see_runs([])
    b.build(at(110), 0, mask)
    b.see_presses([])
    b.see_runs(runs_second)
    b.build(at(130), 0, mask)
    return b.memory[0].enemy_elixir_milli()


def test_the_builder_hands_its_memories_the_report():
    refused = _builder_count([(122, 1, "ability", SLOT, 15)])
    ran = _builder_count([(122, 1, "ability", SLOT, 0)])
    assert refused - ran == PRICE * 1000


def test_the_env_hands_its_builder_the_engines_report():
    from royalegym import ClashParallelEnv
    from royalegym.obs import SpatialObsBuilder

    report = [(7, 1, "ability", SLOT, 15)]

    class Reporting(MockEngine):
        def step_commands_run(self):
            return list(report)

    class Recording(SpatialObsBuilder):
        def see_runs(self, runs):
            seen.append(runs)
            super().see_runs(runs)

    seen = []
    env = ClashParallelEnv(engine=Reporting(), obs_builder=Recording())
    env.reset(seed=0)
    env.step({a: 0 for a in env.agents})
    assert seen == [report]
    seen.clear()
    plain = ClashParallelEnv(engine=MockEngine(), obs_builder=Recording())
    plain.reset(seed=0)
    plain.step({a: 0 for a in plain.agents})
    assert seen == [None], "an engine without the report hands None, not an empty list"
