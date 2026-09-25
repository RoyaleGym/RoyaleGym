"""The published primitive against the env: dated plays in, the env's fair fields out.

``MatchMemory.start`` and ``MatchMemory.advance`` keep a player's memory from dated plays,
``MatchClock.at`` gives the clock from a tick, and ``fair_fields`` reads every fair vector
field but the board's four. Anything outside this repo that builds observations without
an engine rests on those four names, so this file holds them to the env itself: play a
battle, date every accepted play, and at every step compare each field with the slice of
the vector the env's builder wrote, for both seats, with and without ``enemy_last_card``.
It also holds ``MatchClock.at`` to the clock the engine reports, tick by tick.

The caller of the primitive supplies what a player sees anyway: here the own hand and
next card are read from the state, and the own bar is the memory's own count, so that
count is checked too.

The battles reach the places a field could quietly read the board or the rate: spells in
flight (RustEngine; MockEngine reports none), the switch to 2x, the end of regulation,
and ticks past 4800. Each is asserted, not assumed.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pytest

from royalegym.action import TileActionParser
from royalegym.mock_engine import MockEngine
from royalegym.obs import (
    BOARD_FIELDS,
    FAIR_FIELDS,
    MatchClock,
    MatchMemory,
    SpatialObsBuilder,
    fair_fields,
    vector_layout,
)
from royalegym.protocol import (
    DECK_SIZE,
    TEAMS,
    DeployStatus,
    ElixirLaw,
    MatchSetup,
    ShuffleMode,
    default_calibration,
)
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available

DECK = ["Fireball", "Zap", "Arrows", "Knight", "Archer", "Giant", "Musketeer", "Minions"]
CADENCES = (10, 7, 1, 13, 4)
#: Chance a seat plays on a step, once its bar holds MIN_ELIXIR. Blue waits for a full bar, so
#: it leaks. Red waits for 5, the dearest card in DECK: spending the moment anything is
#: affordable leaves the dear cards stuck in hand, so the enemy never shows all eight.
PLAY = {0: 0.25, 1: 0.8}
MIN_ELIXIR = {0: 10, 1: 5}

ENGINES = {
    "mock": MockEngine,
    "rust": pytest.param(
        RustEngine, marks=pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
    ),
}


class Played(NamedTuple):
    mismatches: list[str]
    plays: dict[int, int]
    seen: dict[str, Any]


@pytest.fixture(scope="module", params=list(ENGINES.values()), ids=list(ENGINES))
def engine(request):
    return request.param()


def play_out(engine, seed: int, start_tick: int, ticks: int, late: int = 0) -> Played:
    """A battle, with the env's fields and the primitive's side by side at every step.

    ``late`` dates Blue's own plays that many ticks late in Blue's memory: the plant.
    """
    card = {c.name: i for i, c in enumerate(engine.cards())}
    deck = [card[n] for n in DECK]
    cards = engine.cards()
    cal = default_calibration()
    law = ElixirLaw.load(cal)
    max_mana = cal.int("match.MAX_MANA")
    rng = np.random.default_rng(seed)
    parser = TileActionParser()
    parser.bind(engine)
    builders = {False: SpatialObsBuilder(), True: SpatialObsBuilder(card_identity=True)}
    for b in builders.values():
        b.bind(engine, parser)
    setup = MatchSetup(decks=[deck, deck], shuffle=ShuffleMode.NONE, start_tick=start_tick)
    engine.reset(seed, setup)
    state = engine.state()
    for b in builders.values():
        b.reset(state)
    memories = {}
    for team in TEAMS:
        m = MatchMemory(len(cards), law)
        m.bind(cards)
        me, foe = state.players[team], state.players[1 - team]
        m.start(state.tick, me.elixir_milli, foe.elixir_milli, me.hand, me.next_card)
        memories[team] = m
    dated: dict[int, tuple[list, list]] = {t: ([], []) for t in TEAMS}  # own, enemy
    seen: dict[str, Any] = {
        "spells": 0, "rates": set(), "overtime": set(), "max_tick": 0,
        "leaked": False, "narrowed": False, "clock": [],
    }
    mismatches: list[str] = []
    plays = {t: 0 for t in TEAMS}
    step, end = 0, state.tick + ticks
    while not state.game_over and state.tick < end:
        seen["spells"] += bool(state.spells)
        seen["rates"].add(state.elixir_rate)
        seen["overtime"].add(state.overtime)
        seen["max_tick"] = max(seen["max_tick"], state.tick)
        clock = MatchClock.at(state.tick)
        if clock != MatchClock.of(state):
            seen["clock"].append((clock, MatchClock.of(state)))
        for team in TEAMS:
            m = memories[team]
            if state.tick > m.tick:
                # Each play goes in at the first observation after its date, as any
                # caller holding a log would do; one dated late can land a step later.
                own, enemy = ([p for p in d if p[0] < state.tick] for d in dated[team])
                dated[team] = tuple([p for p in d if p[0] >= state.tick] for d in dated[team])
                m.advance(state.tick, clock.regular_ticks, clock.overtime, own, enemy)
                me = state.players[team]
                m.show_own_hand(me.hand, me.next_card)
            me = state.players[team]
            for flag, builder in builders.items():
                vec = builder.build(state, team, parser.action_mask(state, team))["vector"]
                offsets = builder.vector_offsets()
                got = fair_fields(
                    m, clock, me.hand, me.next_card, law.to_milli(m.own_fine), cards, max_mana,
                    enemy_last_card=flag,
                )
                for field, value in got.items():
                    want = vec[offsets[field]]
                    if not np.array_equal(value, want):
                        mismatches.append(
                            f"tick {state.tick} seat {team} enemy_last_card={flag} {field}: "
                            f"primitive {value.tolist()} env {want.tolist()}"
                        )
                seen["leaked"] |= bool(got["own_elixir_leaked"][0] > 0)
                seen["narrowed"] |= bool(0 < got["enemy_possible_hand"].sum() < DECK_SIZE)
        commands = []
        for team in TEAMS:
            ready = state.players[team].elixir_milli >= 1000 * MIN_ELIXIR[team]
            if not ready or rng.random() >= PLAY[team]:
                continue
            # A card first, then a tile for it. Uniform over (card, tile) pairs would pick
            # spells about twice as often as troops, which are legal on half the board, and
            # a troop could sit in hand unplayed for the whole battle.
            by_slot = parser.action_mask(state, team)[1:].reshape(4, -1)
            slots = np.flatnonzero(by_slot.any(axis=1))
            if len(slots):
                slot = int(rng.choice(slots))
                tile = int(rng.choice(np.flatnonzero(by_slot[slot])))
                cmd = parser.parse(1 + slot * by_slot.shape[1] + tile, state, team)
                if cmd is not None:
                    commands.append(cmd)
        for r in engine.step(commands, CADENCES[step % len(CADENCES)]):
            if r.status != DeployStatus.OK:
                continue
            plays[r.team] += 1
            for team in TEAMS:
                if r.team == team:
                    dated[team][0].append((r.tick + (late if team == 0 else 0), r.card_id))
                else:
                    dated[team][1].append((r.tick, r.card_id))
        step += 1
        state = engine.state()
    return Played(mismatches, plays, seen)


def clean(run: Played) -> None:
    assert not run.mismatches, "\n".join(run.mismatches[:10])
    assert not run.seen["clock"], f"MatchClock.at differs from the engine: {run.seen['clock'][:3]}"


def test_the_primitive_gives_the_env_fields_through_an_opening(engine):
    run = play_out(engine, seed=11, start_tick=engine.rules().deploy_lockout_ticks, ticks=2400)
    clean(run)
    assert run.plays[1] > DECK_SIZE, f"Red played {run.plays[1]}: its queue never came round"
    assert run.seen["leaked"], "no seat ever leaked, so the leak field compared only zeros"
    assert run.seen["narrowed"], "the enemy's possible hand never narrowed below the catalogue"
    if isinstance(engine, RustEngine):
        assert run.seen["spells"] > 0, "no state had a spell in flight"


def test_the_primitive_gives_the_env_fields_across_the_switch_to_2x(engine):
    start = MatchClock.at(0).regular_ticks - ElixirLaw.load().speedup_ticks - 150
    run = play_out(engine, seed=5, start_tick=start, ticks=600)
    clean(run)
    assert run.seen["rates"] == {1, 2}, f"rates seen {run.seen['rates']}"


def test_the_primitive_gives_the_env_fields_across_regulation_and_past_4800(engine):
    """From 35 ticks before the end of regulation on an empty board, so nobody can take a
    crown and it goes on into overtime; then a second battle started past tick 4800.

    35 because the cadences 10, 7, 1, 13, 4 then land exactly on the tick regulation ends,
    the one tick where "still running means overtime" could be off by one."""
    run = play_out(engine, seed=9, start_tick=MatchClock.at(0).regular_ticks - 35, ticks=400)
    clean(run)
    assert run.seen["overtime"] == {False, True}, f"overtime seen {run.seen['overtime']}"
    assert run.plays[1] > 0
    # A crown ends overtime, so reaching 4800 by playing through is luck; start past it.
    run = play_out(engine, seed=9, start_tick=4700, ticks=300)
    clean(run)
    assert run.seen["max_tick"] > 4800, f"got only to tick {run.seen['max_tick']}"
    assert run.plays[1] > 0


def test_plant_plays_dated_one_tick_late_are_caught(engine):
    run = play_out(
        engine, seed=11, start_tick=engine.rules().deploy_lockout_ticks, ticks=2400, late=1
    )
    assert run.mismatches, "PLANT DID NOT LAND: plays dated a tick late matched the env"
    assert all(" seat 0 " in m for m in run.mismatches), run.mismatches[:5]


# -- the primitive alone -------------------------------------------------------------


@pytest.fixture(scope="module")
def mock_cards():
    return MockEngine().cards()


def memory_of(cards, deck, tick=0, own=None, enemy=None) -> MatchMemory:
    m = MatchMemory(len(cards), ElixirLaw.load())
    m.bind(cards)
    start = 1000 * default_calibration().int("match.START_MANA")
    own = start if own is None else own
    enemy = start if enemy is None else enemy
    m.start(tick, own, enemy, deck[:4], deck[4])
    return m


def test_the_whole_published_surface_is_still_here():
    """Every name code outside this repo builds on, so a rename fails HERE, not only there.

    The list is the one docs/observation-spec.md publishes ("The whole surface such a
    caller may rely on"). An engine-free caller elsewhere (RoyaleImitate's log memory)
    uses all of it; with only that repo's CI exercising these names, a rename in this
    one would be found by somebody else's red.
    """
    import inspect

    from royalegym import obs, protocol

    memory = MatchMemory(3, ElixirLaw.load())
    for name in ("bind", "start", "advance", "show_own_hand"):
        assert callable(getattr(memory, name, None)), f"MatchMemory.{name} is gone"
    for name in ("tick", "own_fine", "foe_fine", "unaffordable"):
        assert hasattr(memory, name), f"MatchMemory.{name} is gone"
    for name in ("MatchClock", "fair_fields", "FAIR_FIELDS", "BOARD_FIELDS", "MatchMemory"):
        assert hasattr(obs, name), f"royalegym.obs.{name} is gone"
    for name in ("ElixirLaw", "default_calibration", "CardInfo", "DECK_SIZE", "HAND_SIZE"):
        assert hasattr(protocol, name), f"royalegym.protocol.{name} is gone"
    assert callable(ElixirLaw.load)
    assert callable(memory.law.to_milli)
    assert list(inspect.signature(MatchMemory.show_own_hand).parameters) == [
        "self", "hand", "next_card"
    ]
    assert list(inspect.signature(MatchMemory.start).parameters)[:6] == [
        "self", "tick", "own_elixir_milli", "enemy_elixir_milli", "own_hand", "next_card"
    ]
    assert list(inspect.signature(MatchMemory.advance).parameters) == [
        "self", "tick", "regular_ticks", "overtime", "own_plays", "foe_plays"
    ]


def test_the_published_names_cover_the_fair_block_exactly():
    fair = [f.key for f in vector_layout(5) if f.fair]
    assert [k for k in fair if k not in BOARD_FIELDS] == list(FAIR_FIELDS)
    assert set(BOARD_FIELDS) <= set(fair)


def test_splitting_the_clock_changes_nothing_but_the_seen_tick(mock_cards):
    """Advanced every tick or every 10 ticks, with plays in between: the same fields.

    A human plays the moment a card is affordable, which is rarely on an observation
    tick. Both bars start empty, so at 1x (one elixir per 56 ticks) a Knight becomes
    affordable at tick 168 exactly, and the Knight is played then: the last observation
    before it, at 160, sees 2.86 elixir. Paying at 160 would floor the bar at zero and
    hand back 0.14 elixir that was never there, for the rest of the match. The Archer at
    745 comes 17 ticks after the bar filled, so the leak is checked as well.

    Only ``own_ticks_since_play`` may differ: it counts from the observation that first
    showed the play, which is what the env's memory has always done.
    """
    names = {c.name: i for i, c in enumerate(mock_cards)}
    deck = [names[n] for n in ("Skeletons", "Knight", "Archer", "Goblins", "Minions", "Fireball",
                               "Musketeer", "Valkyrie")]
    plays = [(168, deck[1]), (745, deck[2])]
    fine, coarse = (memory_of(mock_cards, deck, own=0, enemy=0) for _ in range(2))
    max_mana = default_calibration().int("match.MAX_MANA")
    law = ElixirLaw.load()

    def run(m: MatchMemory, every: int) -> dict[int, dict]:
        out, pending = {}, list(plays)
        for tick in range(every, 901, every):
            due = [p for p in pending if p[0] < tick]
            pending = [p for p in pending if p[0] >= tick]
            clock = MatchClock.at(tick)
            m.advance(tick, clock.regular_ticks, clock.overtime, due, due)
            out[tick] = fair_fields(m, clock, deck[:4], deck[4], law.to_milli(m.own_fine),
                                    mock_cards, max_mana)
        return out

    a, b = run(fine, 1), run(coarse, 10)
    for tick, got in b.items():
        for field, value in got.items():
            if field != "own_ticks_since_play":
                assert np.array_equal(a[tick][field], value), (tick, field, a[tick][field], value)
    assert fine.unaffordable == coarse.unaffordable == [0, 0]
    assert b[900]["own_elixir_leaked"][0] > 0, "the bar never filled"


def test_a_play_the_bar_cannot_pay_is_counted(mock_cards):
    names = {c.name: i for i, c in enumerate(mock_cards)}
    deck = [names[n] for n in ("Skeletons", "Knight", "Archer", "Goblins", "Minions", "Fireball",
                               "Musketeer", "Valkyrie")]
    m = memory_of(mock_cards, deck, own=1000)
    clock = MatchClock.at(1)
    m.advance(1, clock.regular_ticks, clock.overtime, [(0, deck[1])], [])  # a Knight on 1 elixir
    assert m.unaffordable == [1, 0]


def test_a_play_outside_the_interval_is_refused(mock_cards):
    deck = list(range(8))
    m = memory_of(mock_cards, deck, tick=100)
    clock = MatchClock.at(110)
    with pytest.raises(ValueError, match="outside"):
        m.advance(110, clock.regular_ticks, clock.overtime, [(110, deck[0])], [])


def test_plant_an_overtime_rule_one_tick_late_is_caught(engine, monkeypatch):
    """``MatchClock.at`` calling tick == regulation's end "not yet overtime" must fail."""
    real = MatchClock.at.__func__

    def late(cls, tick, calibration=None):
        c = real(cls, tick, calibration)
        return c._replace(overtime=tick > c.regular_ticks) if tick == c.regular_ticks else c

    monkeypatch.setattr(MatchClock, "at", classmethod(late))
    run = play_out(engine, seed=9, start_tick=MatchClock.at(0).regular_ticks - 35, ticks=100)
    assert run.seen["clock"], "PLANT DID NOT LAND: the regulation boundary was never compared"
