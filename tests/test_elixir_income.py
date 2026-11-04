"""The counted bars take in the elixir the board pays, when asked (Train and Live, 2026-10-25).

WHY IT EXISTS
    An Elixir Collector pays its side one elixir every 13 s on the 15.535 table and one more
    when it dies, and each piece of an Elixir Golem pays the OTHER side when it dies. The fair
    count paid plays and presses and regenerated, and took in none of it: Live measured its
    own count 1.9 elixir low on average while an opponent's Collector stood (liveplay
    20261025-003433, up to ~3), and RoyaleGym's had the same blind spot, so a policy trained on
    it underrated a Collector player. ``MatchMemory.count_income`` follows each producer from
    the states as RoyaleSim's ``mana_pass`` runs it and pays each death the state shows;
    ``SpatialObsBuilder(elixir_income=True)`` turns it on. Off, nothing changes.

WHAT IT CHECKS
    a. A Collector with its side kept below the cap: both seats' counts equal the engine's two
       bars at every observation, at the env's 10-tick cadence and at every tick, through its
       payouts and its death; without the count the enemy count goes inexact (the control).
    b. Payouts that fall due at the cap are held and paid on release: still exact.
    c. A Zap cast on the Collector at an observation holds its timer: still exact, and with
       economy.STUN_PAUSES_PRODUCTION off (engine and count both) too.
    d. An Elixir Golem's pieces pay the side that kills them: exact.
    e. The builder: ``elixir_income`` is exact through a Collector, says so in ``config()``,
       survives a memory save and load, and is refused by an engine that does not say what
       each unit pays; MockEngine pays nothing and counts nothing. Off, the memory keeps the
       fields it always kept, so a memory saved by an earlier royalegym still loads.
    f. A calibration on an economy arm the count does not follow is refused.
    g. The engine says what each unit pays on the default table (``RustEngine.unit_mana``).

SKIPS
    Without royalesim built. Not a pass.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import pytest

from royalegym.action import TileActionParser
from royalegym.mock_engine import MockEngine
from royalegym.obs import IncomeLaw, MatchMemory, SpatialObsBuilder
from royalegym.protocol import (
    BLUE,
    RED,
    TEAMS,
    BattleState,
    DeployCommand,
    ElixirLaw,
    MatchSetup,
    ShuffleMode,
    card_is_spell,
    default_calibration,
)
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available

needs_core = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))

#: What the default (15.535) table pays, as RoyaleSim reports it (Sim, 2026-10-25).
DEFAULT_TABLE_MANA = {
    "ElixirCollector": (1000, 13000, 1000, 0),
    "ElixirGolem1": (0, 0, 0, 1000),
    "ElixirGolem2": (0, 0, 0, 500),
    "ElixirGolem4": (0, 0, 0, 500),
}
DECK = (
    "Elixir Collector",
    "ElixirGolem",
    "Knight",
    "Skeletons",
    "Minions",
    "Zap",
    "Arrows",
    "Fireball",
)

Script = Callable[[BattleState, dict[str, int], int], list[DeployCommand]]


class Run(NamedTuple):
    memories: list[MatchMemory]
    wrong: list[tuple[int, int, str]]  # every (tick, seat, "enemy"/"own") a count was off
    jumps: list[int]  # payouts each side's bar took in: jumps the regen cannot explain
    held: int  # observations at which Red's memory held a payout of Blue's at the cap
    stunned: int  # observations at which a producer of Blue's showed a stun


def counting(engine: RustEngine, on: bool = True) -> list[MatchMemory]:
    law = ElixirLaw.load()
    mana = engine.unit_mana()
    assert mana is not None, "this engine does not say what each unit pays"
    names = engine.unit_types()
    assert names is not None
    by_type = {names.index(n): row for n, row in mana.items()}
    income = IncomeLaw.load(default_calibration(), law, engine.calibration_overrides)
    out = []
    for _ in TEAMS:
        m = MatchMemory(len(engine.cards()), law)
        m.bind(engine.cards())
        if on:
            m.count_income(by_type, income)
        out.append(m)
    return out


def play_out(engine: RustEngine, script: Script, ticks: int, every: int = 10, on: bool = True):
    """Run ``script`` (commands only at observations, as the env sends them) for ``ticks``,
    observing every ``every`` ticks with both seats' memories (``Run``)."""
    ids = {c.name: c.card_id for c in engine.cards()}
    engine.reset(
        5,
        MatchSetup(
            decks=[[ids[n] for n in DECK]] * 2,
            shuffle=ShuffleMode.NONE,
            start_tick=engine.rules().deploy_lockout_ticks,
        ),
    )
    mem = counting(engine, on)
    s = engine.state()
    for team in TEAMS:
        mem[team].seed(s, team)
    wrong: list[tuple[int, int, str]] = []
    jumps = [0, 0]
    held = stunned = 0
    paying = set(engine.unit_mana() or ())
    names = engine.unit_types() or []
    for _ in range(ticks // every):
        if s.game_over:
            break
        cmds = script(s, ids, engine.arena().subtile)
        spent = [0, 0]
        for c in cmds:
            spent[c.team] += 1000 * engine.cards()[s.players[c.team].hand[c.hand_slot]].elixir
        before = [s.players[t].elixir_milli for t in TEAMS]
        engine.step(cmds, every)
        s = engine.state()
        for t in TEAMS:
            # A bar never gains more than the regen of ``every`` ticks at 3x without a payout.
            if s.players[t].elixir_milli - before[t] + spent[t] > every * 60:
                jumps[t] += 1
        for team in TEAMS:
            m = mem[team]
            m.observe(s, team)
            if m.enemy_elixir_milli() != s.players[1 - team].elixir_milli:
                wrong.append((s.tick, team, "enemy"))
            if m.own_elixir_milli() != s.players[team].elixir_milli:
                wrong.append((s.tick, team, "own"))
        if on:
            held += any(r[1] == BLUE and r[4] == 0 for r in mem[RED].producers)
        stunned += any(
            e.team == BLUE and e.stun_ticks > 0 and names[e.unit_type] in paying
            for e in s.entities
            if e.unit_type >= 0
        )
    return Run(mem, wrong, jumps, held, stunned)


#: Where each side spends without touching the scene: a spell on empty ground in the other
#: half, clear of the towers and of the Collector, or a troop in its own back corner (tiles).
EMPTY = {BLUE: (1, 22), RED: (16, 9)}
BACK = {BLUE: (2, 2), RED: (15, 29)}


def spend(team: int, player, cards, t: int, keep=()) -> list[DeployCommand]:
    """One play of the cheapest card in ``team``'s hand but ``keep``, a spell first."""
    live = [c for c in player.hand if c >= 0 and c not in keep]
    if not live:
        return []
    spells = [c for c in live if card_is_spell(cards[c])]
    pick = min(spells or live, key=lambda c: cards[c].elixir)
    if player.elixir_milli < 1000 * cards[pick].elixir:
        return []
    x, y = EMPTY[team] if pick in spells else BACK[team]
    return [DeployCommand(team, player.hand.index(pick), x * t + t // 2, y * t + t // 2)]


def collector_script(spend_after: int | None = 0, zap_after: int | None = None) -> Script:
    """Blue cycles to the Collector and plays it; then, from ``spend_after`` ticks after the
    play (None: never), spends whenever it nears the cap. With ``zap_after``, Red cycles to its
    Zap and casts it on the Collector that many ticks after the Collector's play. Red spends
    whenever it nears the cap. Neither side plays a Collector or an Elixir Golem but Blue's
    one Collector."""
    state = {"at": None, "zapped": False}

    def script(s: BattleState, ids: dict[str, int], t: int) -> list[DeployCommand]:
        cards = ENGINE_CARDS[0]
        blue, red = s.players[BLUE], s.players[RED]
        coll, golem, zap = ids["Elixir Collector"], ids["ElixirGolem"], ids["Zap"]
        spot = (9 * t + t // 2, 8 * t + t // 2)
        cmds: list[DeployCommand] = []
        if state["at"] is None:
            if coll not in blue.hand:
                cmds += spend(BLUE, blue, cards, t, keep=(coll, golem))
            elif blue.elixir_milli >= 1000 * cards[coll].elixir:
                cmds.append(DeployCommand(BLUE, blue.hand.index(coll), *spot))
                state["at"] = s.tick
        elif (
            spend_after is not None
            and s.tick >= state["at"] + spend_after
            and blue.elixir_milli >= 8500
        ):
            cmds += spend(BLUE, blue, cards, t, keep=(coll, golem))
        wants_zap = zap_after is not None and not state["zapped"]
        if wants_zap and zap in red.hand:
            ready = state["at"] is not None and s.tick >= state["at"] + zap_after
            if ready and red.elixir_milli >= 1000 * cards[zap].elixir:
                cmds.append(DeployCommand(RED, red.hand.index(zap), *spot))
                state["zapped"] = True
        elif wants_zap or red.elixir_milli >= 8500:
            cmds += spend(
                RED, red, cards, t, keep=(coll, golem, zap) if wants_zap else (coll, golem)
            )
        return cmds

    return script


#: The catalogue the scripts read, set by ``engine()`` (scripts get the state and ids only).
ENGINE_CARDS: list = [None]


def parser_for(e) -> TileActionParser:
    parser = TileActionParser()
    parser.bind(e)
    return parser


def engine(**kw) -> RustEngine:
    e = RustEngine(**kw)
    ENGINE_CARDS[0] = e.cards()
    return e


@needs_core
@pytest.mark.parametrize("every", [10, 1])
def test_a_collector_kept_below_the_cap_is_counted_exactly(every: int) -> None:
    e = engine()
    run = play_out(e, collector_script(), 2700, every)
    assert run.jumps[BLUE] >= 6, (
        f"the Collector paid {run.jumps[BLUE]} times: the scenario lost its point"
    )
    assert not run.wrong, run.wrong[:5]
    assert all(m.exact for m in run.memories)


@needs_core
@pytest.mark.parametrize("table", ["client160402017_20261019", "client160635007_20261103"])
def test_a_collector_on_a_15_second_table_is_counted_exactly(table: str) -> None:
    """The client's newer tables pay every 15 s, not 13 (royalesim 0.1.43): the period is the
    engine's, read through ``unit_mana``, not a constant here."""
    e = engine(calibration_overrides={"cards.CARD_TABLE": table})
    assert e.unit_mana()["ElixirCollector"][1] == 15000
    run = play_out(e, collector_script(), 2700, 1)
    assert run.jumps[BLUE] >= 5, f"the Collector paid {run.jumps[BLUE]} times"
    assert not run.wrong, run.wrong[:5]
    assert all(m.exact for m in run.memories)


@needs_core
def test_without_the_count_a_collector_leaves_the_enemy_count_low() -> None:
    run = play_out(engine(), collector_script(), 1500, on=False)
    assert run.jumps[BLUE] >= 3
    # Only the enemy count shows it: the own bar is visible and resynced every observation.
    assert any(team == RED and what == "enemy" for _, team, what in run.wrong), run.wrong[:5]


@needs_core
def test_payouts_due_at_the_cap_are_held_and_counted() -> None:
    e = engine()
    # Blue sits at the cap for 900 ticks after its Collector lands, then spends.
    run = play_out(e, collector_script(spend_after=900), 2000)
    assert run.held >= 3, f"a payout was held at {run.held} observations"
    assert run.jumps[BLUE] >= 2, "no payout after the release"
    assert not run.wrong, run.wrong[:5]
    assert all(m.exact for m in run.memories)


@needs_core
@pytest.mark.parametrize("pauses", [True, False])
def test_a_zap_on_the_collector_holds_its_timer_and_the_count_follows(pauses: bool) -> None:
    kw = {} if pauses else {"calibration_overrides": {"economy.STUN_PAUSES_PRODUCTION": False}}
    e = engine(**kw)
    run = play_out(e, collector_script(zap_after=100), 1500)
    assert run.stunned >= 1, "the Zap never stunned the Collector"
    assert run.jumps[BLUE] >= 2
    assert not run.wrong, run.wrong[:5]
    assert all(m.exact for m in run.memories)


def golem_script() -> Script:
    """Blue plays its Elixir Golem at the left bridge, and Red's towers take it apart; both
    sides spend whenever they near the cap, so no payout meets it."""
    state = {"at": None}

    def script(s: BattleState, ids: dict[str, int], t: int) -> list[DeployCommand]:
        cards = ENGINE_CARDS[0]
        blue, red = s.players[BLUE], s.players[RED]
        coll, golem = ids["Elixir Collector"], ids["ElixirGolem"]
        cmds: list[DeployCommand] = []
        if state["at"] is None:
            if golem not in blue.hand:
                cmds += spend(BLUE, blue, cards, t, keep=(coll, golem))
            elif blue.elixir_milli >= 1000 * cards[golem].elixir:
                cmds.append(
                    DeployCommand(BLUE, blue.hand.index(golem), 3 * t + t // 2, 13 * t + t // 2)
                )
                state["at"] = s.tick
        elif blue.elixir_milli >= 8500:
            cmds += spend(BLUE, blue, cards, t, keep=(coll, golem))
        if red.elixir_milli >= 7000:
            cmds += spend(RED, red, cards, t, keep=(coll, golem))
        return cmds

    return script


@needs_core
def test_an_elixir_golems_pieces_pay_the_side_that_kills_them() -> None:
    run = play_out(engine(), golem_script(), 1500)
    assert run.jumps[RED] >= 3, f"Red's bar took in {run.jumps[RED]} death payouts"
    assert not run.wrong, run.wrong[:5]
    assert all(m.exact for m in run.memories)


# -- the builder ------------------------------------------------------------------------


@needs_core
def test_the_builder_counts_income_when_asked_and_says_so() -> None:
    from royalegym import ClashParallelEnv

    on = SpatialObsBuilder(elixir_income=True)
    assert on.config()["elixir_income"] is True
    assert "elixir_income" not in SpatialObsBuilder().config()
    ClashParallelEnv(engine=engine(), obs_builder=on).reset(seed=5)
    assert all(m.income is not None for m in on.memory.values())
    off = SpatialObsBuilder()
    ClashParallelEnv(engine=engine(), obs_builder=off).reset(seed=5)
    assert all(m.income is None for m in off.memory.values())


@needs_core
def test_a_saved_memory_with_income_loads_and_counts_on() -> None:
    e = engine()
    builder = SpatialObsBuilder(elixir_income=True)
    builder.bind(e, parser_for(e))
    ids = {c.name: c.card_id for c in e.cards()}
    e.reset(
        5,
        MatchSetup(
            decks=[[ids[n] for n in DECK]] * 2,
            shuffle=ShuffleMode.NONE,
            start_tick=e.rules().deploy_lockout_ticks,
        ),
    )
    s = e.state()
    builder.reset(s)
    script = collector_script()
    saved = None
    for i in range(160):
        e.step(script(s, ids, e.arena().subtile), 10)
        s = e.state()
        for team in TEAMS:
            builder.memory[team].observe(s, team)
        if i == 60:
            assert builder.memory[RED].producers, "no Collector stood when the memory was saved"
            saved = builder.save_memory()
            builder.load_memory(saved, s)
    assert saved is not None
    assert all(builder.counts_are_exact(t) for t in TEAMS)


def test_an_engine_that_does_not_say_what_units_pay_is_refused() -> None:
    class Silent(MockEngine):
        def unit_mana(self):  # type: ignore[override]
            return None

    e = Silent()
    with pytest.raises(ValueError, match="does not say what each unit pays"):
        SpatialObsBuilder(elixir_income=True).bind(e, parser_for(e))


def test_mock_engine_pays_nothing_and_counts_nothing() -> None:
    e = MockEngine()
    b = SpatialObsBuilder(elixir_income=True)
    b.bind(e, parser_for(e))
    assert all(m.income == {} for m in b.memory.values())


def test_off_the_memory_keeps_the_fields_it_always_kept() -> None:
    m = MatchMemory(3, ElixirLaw.load())
    assert not hasattr(m, "producers")
    assert not hasattr(m, "payers")
    m.count_income({}, IncomeLaw.load(default_calibration(), ElixirLaw.load()))
    assert m.producers == []
    assert m.payers == []


def test_an_economy_arm_the_count_does_not_follow_is_refused() -> None:
    cal = default_calibration().with_override("economy.PRODUCTION_AT_CAP", "wasted_timer_runs_on")
    with pytest.raises(ValueError, match="PRODUCTION_AT_CAP"):
        IncomeLaw.load(cal, ElixirLaw.load())
    with pytest.raises(ValueError, match="PRODUCTION_OVERFLOW"):
        IncomeLaw.load(
            default_calibration(),
            ElixirLaw.load(),
            {"economy.PRODUCTION_OVERFLOW": "remainder_held"},
        )


@needs_core
def test_the_engine_says_what_each_unit_pays_on_the_default_table() -> None:
    assert engine().unit_mana() == DEFAULT_TABLE_MANA


@needs_core
@pytest.mark.parametrize("decision_ms", [50, 500])
def test_a_random_battle_with_collectors_and_golems_stays_exact(decision_ms: int) -> None:
    """Both seats hold an Elixir Collector and an Elixir Golem and play at random, as
    ``test_enemy_queue``'s strict xfail does with the count off: with it on, both seats'
    counts stay exact to the end, through payouts and deaths, one tick or ten per decision."""
    from royalegym import ClashParallelEnv
    from royalegym.state_mutator import DefaultStateMutator

    deck = [
        "Elixir Collector",
        "ElixirGolem",
        "Knight",
        "Archer",
        "Goblins",
        "Musketeer",
        "Arrows",
        "Zap",
    ]
    e = engine()
    names = list(e.unit_types() or [])
    paying = {names.index(n) for n in e.unit_mana() or ()}
    env = ClashParallelEnv(
        e,
        action_parser=TileActionParser(),
        obs_builder=SpatialObsBuilder(elixir_income=True),
        decision_ms=decision_ms,
        state_mutator=DefaultStateMutator(decks=[deck, deck]),
    )
    env.reset(seed=4)
    import numpy as np

    rng = np.random.default_rng(4)
    standing: set[int] = set()
    seen = died = 0
    for _ in range(150_000 // decision_ms):
        if not env.agents:
            break
        acts = {}
        for agent in env.agents:
            legal = np.flatnonzero(env.action_masks(agent))
            legal = legal[legal != 0]
            acts[agent] = int(rng.choice(legal)) if len(legal) and rng.random() < 0.3 else 0
        env.step(acts)
        now = {x.uid for x in env.battle_state.entities if x.unit_type in paying}
        seen += len(now - standing)
        died += len(standing - now)
        standing = now
        for team in TEAMS:
            assert env.obs_builder.counts_are_exact(team), f"seat {team} at {env.battle_state.tick}"
    assert seen >= 4, f"only {seen} paying units stood: the battle lost its point"
    assert died >= 2, f"only {died} paying units died"
