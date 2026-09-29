"""MatchMemory charges an ability press what it cost (found by the docs session, 2026-09-29).

WHY IT EXISTS
    A press of an ability button (a hero's, a champion's) is paid from the bar and changes no
    hand slot, and MatchMemory read plays only from hand slots. So the counted enemy bar read
    high by the press's cost from then on, and ``exact`` went False for the match: the docs
    session's finder measured one Golden Knight press by Red leaving Blue's count at 3571
    against a real 2571. The press is public (the hero turns spent, the champion's button goes
    dark while he stands; in the game you see the ability), so the count can charge it.

WHAT IT CHECKS
    a. Against the engine, a champion: both seats play a Golden Knight and press him when the
       engine takes it; each seat's memory, observing every tick, keeps its enemy count equal
       to the engine's enemy bar and stays ``exact``, through the press, the chain and the
       cooldown.
    b. The same for a hero (the Hero Musketeer), whose press spends its one charge.
    c. A champion's button going dark because he DIED is not a press: nothing is charged.
    d. The dated path RoyaleImitate uses (``advance``) charges the presses it is given.

SKIPS
    Without royalesim built, or on an engine with no champion or hero button. Not a pass.
"""

from __future__ import annotations

import msgspec
import pytest

from royalegym.obs import MatchMemory
from royalegym.protocol import (
    HAND_SIZE,
    TEAMS,
    DeployCommand,
    DeployStatus,
    ElixirLaw,
    EntityKind,
    EntityState,
    MatchSetup,
    ShuffleMode,
    ability_row,
    to_engine,
)
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available

needs_core = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
FILLERS = ("Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Cannon")


def memories(engine: RustEngine) -> list[MatchMemory]:
    out = []
    for _ in TEAMS:
        m = MatchMemory(len(engine.cards()), ElixirLaw.load())
        m.bind(engine.cards())
        out.append(m)
    return out


def press_through(engine: RustEngine, placements, ticks: int) -> tuple[list[str], dict[int, int]]:
    """Put ``placements`` (card name, own-frame tile) down for both seats, then press button 0
    whenever the engine takes it, observing every tick with both seats' memories. Returns the
    steps where a count left the engine's bar or a memory stopped being exact, and the tick of
    each seat's first press."""
    ids = {c.name: c.card_id for c in engine.cards()}
    a, t = engine.arena(), engine.arena().subtile
    mems = memories(engine)
    problems: list[str] = []
    pressed: dict[int, int] = {}

    def look(what: str) -> None:
        s = engine.state()
        for team, m in zip(TEAMS, mems, strict=True):
            m.observe(s, team)
            foe = s.players[1 - team]
            if m.enemy_elixir_milli() != foe.elixir_milli or not m.exact:
                problems.append(
                    f"{what}, tick {s.tick}, seat {team}: counted enemy bar "
                    f"{m.enemy_elixir_milli()}, engine {foe.elixir_milli}, exact {m.exact}"
                )

    look("start")
    for name, (tx, ty) in placements:
        s = engine.state()
        cmds = [
            DeployCommand(team, s.players[team].hand.index(ids[name]),
                          *to_engine(a, team, tx * t + t // 2, ty * t + t // 2))
            for team in TEAMS
        ]
        assert [r.status for r in engine.step(cmds, 1)] == [DeployStatus.OK] * 2
        look(f"after {name}")
    for _ in range(ticks):
        s = engine.state()
        cmds = [
            DeployCommand(team, HAND_SIZE, 0, 0) for team in TEAMS
            if team not in pressed
            and engine.check_deploy(DeployCommand(team, HAND_SIZE, 0, 0)) == DeployStatus.OK
        ]
        for c in cmds:
            pressed[c.team] = s.tick
        results = engine.step(cmds, 1)
        assert all(r.status == DeployStatus.OK for r in results), results
        look("pressed" if cmds else "waiting")
    return problems, pressed


def battle(engine: RustEngine, first: str, forms=None) -> None:
    ids = {c.name: c.card_id for c in engine.cards()}
    deck = [ids[first], *(ids[n] for n in FILLERS)]
    engine.reset(1, MatchSetup(
        decks=[deck, deck], shuffle=ShuffleMode.NONE, forms=forms,
        elixir_milli=[10000, 10000], start_tick=engine.rules().deploy_lockout_ticks,
    ))


@needs_core
def test_a_champion_press_is_charged_to_the_count():
    engine = RustEngine()
    if "GoldenKnight" not in {c.name for c in engine.cards()}:
        pytest.skip("SKIPPED, NOT PASSED: this catalogue holds no Golden Knight")
    battle(engine, "GoldenKnight")
    if not engine.state().players[0].abilities:
        pytest.skip("SKIPPED, NOT PASSED: this engine gives the Golden Knight no button")
    # Each Golden Knight meets the other seat's Giant: something in reach to dash at.
    problems, pressed = press_through(
        engine, (("GoldenKnight", (3, 13)), ("Giant", (14, 14))), 300
    )
    assert set(pressed) == set(TEAMS), f"a seat never pressed: {pressed}"
    assert problems == [], problems[:6]


@needs_core
def test_a_hero_press_is_charged_to_the_count():
    engine = RustEngine()
    battle(engine, "Musketeer", forms=[[2] + [0] * 7, [2] + [0] * 7])
    if not engine.state().players[0].abilities:
        pytest.skip("SKIPPED, NOT PASSED: this engine has no hero button")
    problems, pressed = press_through(engine, (("Musketeer", (9, 10)),), 200)
    assert set(pressed) == set(TEAMS), f"a seat never pressed: {pressed}"
    assert problems == [], problems[:6]


@needs_core
def test_a_champion_going_dark_because_he_died_is_not_charged():
    """The same step twice from one start, differing only in whether the Golden Knight still
    stands when his button goes dark: standing, it is a press and costs its elixir; gone, it
    is his death and costs nothing."""
    engine = RustEngine()
    if "GoldenKnight" not in {c.name for c in engine.cards()}:
        pytest.skip("SKIPPED, NOT PASSED: this catalogue holds no Golden Knight")
    gk = next(c.card_id for c in engine.cards() if c.name == "GoldenKnight")
    battle(engine, "GoldenKnight")
    start = engine.state()
    if not start.players[1].abilities:
        pytest.skip("SKIPPED, NOT PASSED: this engine gives the Golden Knight no button")
    row = [*start.players[1].abilities[0]]
    lit, dark = [1, *row[1:]], [0, *row[1:]]
    knight = EntityState(
        uid=10**6, team=1, kind=EntityKind.TROOP, card_id=gk, tower_slot=-1, x=0, y=0,
        hp=1000, max_hp=1000, radius=1, flying=False, deploy_ticks=0,
    )

    def state(tick: int, rows, standing: bool, elixir: int):
        red = msgspec.structs.replace(start.players[1], abilities=[rows], elixir_milli=elixir)
        ents = [*start.entities, knight] if standing else list(start.entities)
        return msgspec.structs.replace(start, tick=tick, players=[start.players[0], red],
                                       entities=ents)

    counted = {}
    for standing in (True, False):
        m = memories(engine)[0]
        m.observe(state(start.tick, lit, True, 3000), 0)
        m.observe(state(start.tick + 10, dark, standing, 3000), 0)
        counted[standing] = m.enemy_elixir_milli()
    cost = ability_row(row).cost
    assert counted[False] - counted[True] == cost * 1000, (counted, cost)


def test_the_dated_path_charges_the_presses_it_is_given():
    law = ElixirLaw.load()
    m = MatchMemory(4, law)
    m.bind([])
    m.start(100, 5000, 5000, [0, 1, 2, 3], 0, [0, 1, 2, 3])
    m.advance(110, 3600, False, [], [], foe_presses=[(100, 2)])
    n = MatchMemory(4, law)
    n.bind([])
    n.start(100, 5000, 5000, [0, 1, 2, 3], 0, [0, 1, 2, 3])
    n.advance(110, 3600, False, [], [])
    assert n.enemy_elixir_milli() - m.enemy_elixir_milli() == 2000
