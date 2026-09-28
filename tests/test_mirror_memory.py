"""MatchMemory charges a Mirror play what it cost (found by the learn session, 2026-09-28).

WHY IT EXISTS
    A Mirror costs the card it copies plus its own one (match.MIRROR_COST_RULE), and it copies
    its side's last accepted play that was not a Mirror (match.MIRROR_RECORD). MatchMemory
    charged each play its listed elixir, so after an enemy Mirror the fair ``enemy_elixir``
    read high by the copied card's cost. RoyaleImitate's public log keeps the same memory, so
    the env and the imitation data were wrong the same way, and a test holding the two
    together could not see it. The Mirror joined RoyaleSim's default catalogue in 20fdc49.

WHAT IT CHECKS
    a. Against the engine: both seats play a Knight and then a Mirror, and each seat's
       memory, observing every step, keeps its enemy count equal to the engine's enemy bar
       and stays ``exact``.
    b. The dated-play path RoyaleImitate uses (``advance``): a Mirror after a Knight costs
       the Knight plus one; a Mirror with nothing seen to copy clears ``exact``.

PLANTS
    the listed-price charge (every card priced at its own elixir) fails a.

SKIPS
    Without royalesim built, or on a catalogue without a Mirror. A skip is not a pass.
"""

from __future__ import annotations

import pytest

from royalegym.obs import MatchClock, MatchMemory
from royalegym.protocol import (
    DECK_SIZE,
    EMPTY_CARD,
    TEAMS,
    DeployCommand,
    DeployStatus,
    ElixirLaw,
    MatchSetup,
    Placement,
    ShuffleMode,
    to_engine,
)
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available

needs_core = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
FILLERS = ("Musketeer", "Valkyrie", "MiniPekka", "Giant", "Prince", "Wizard")


@pytest.fixture(scope="module")
def engine() -> RustEngine:
    e = RustEngine()
    if "Mirror" not in {c.name for c in e.cards()}:
        pytest.skip("SKIPPED, NOT PASSED: this catalogue holds no Mirror")
    return e


def memories(engine: RustEngine) -> list[MatchMemory]:
    out = []
    for _ in TEAMS:
        m = MatchMemory(len(engine.cards()), ElixirLaw.load())
        m.bind(engine.cards())
        out.append(m)
    return out


def play_knight_then_mirror(engine: RustEngine, mems: list[MatchMemory]) -> list[str]:
    """Both seats play a Knight, then a Mirror as soon as it is in hand and paid for; every
    state is observed by both memories. Returns every step where a memory's enemy count left
    the engine's enemy bar, or a memory stopped being exact."""
    ids = {c.name: c.card_id for c in engine.cards()}
    deck = [ids["Mirror"], ids["Knight"], *[ids[n] for n in FILLERS]][:DECK_SIZE]
    engine.reset(
        1,
        MatchSetup(
            decks=[deck, deck], shuffle=ShuffleMode.NONE, elixir_milli=[7000, 6500],
            start_tick=engine.rules().deploy_lockout_ticks,
        ),
    )
    a, t = engine.arena(), engine.arena().subtile
    problems: list[str] = []

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
    for card, tile in (("Knight", (9, 6)), ("Mirror", (5, 6))):
        for _ in range(200):
            s = engine.state()
            ready = [
                ids[card] in p.hand
                and p.elixir_milli >= 1000 * p.hand_costs[p.hand.index(ids[card])]
                for p in s.players
            ]
            if all(ready):
                break
            engine.step([], 5)
            look(f"waiting for {card}")
        s = engine.state()
        plays = [
            DeployCommand(team, s.players[team].hand.index(ids[card]),
                          *to_engine(a, team, tile[0] * t + t // 2, tile[1] * t + t // 2))
            for team in TEAMS
        ]
        statuses = [DeployStatus(r.status).name for r in engine.step(plays, 10)]
        assert statuses == ["OK", "OK"], f"{card}: {statuses}"
        look(f"after {card}")
    for _ in range(6):
        engine.step([], 10)
        look("after")
    return problems


@needs_core
def test_the_counted_enemy_bar_follows_a_mirror_play(engine):
    mems = memories(engine)
    problems = play_knight_then_mirror(engine, mems)
    assert not problems, f"{len(problems)} steps off; first {problems[:4]}"
    knight = next(c for c in engine.cards() if c.name == "Knight").card_id
    assert [m.last_play[1] for m in mems] == [knight, knight], "each saw the enemy's Knight"


@needs_core
def test_plant_the_listed_price_is_caught(engine, monkeypatch):
    mems = memories(engine)
    for m in mems:
        monkeypatch.setattr(m, "is_mirror", [False] * len(engine.cards()))
    problems = play_knight_then_mirror(engine, mems)
    assert any("after Mirror" in p for p in problems), f"PLANT DID NOT LAND: {problems[:4]}"


@needs_core
def test_the_dated_play_path_prices_a_mirror_by_what_it_copies(engine):
    """``advance``, the path a caller with a timed log of plays takes (RoyaleImitate)."""
    ids = {c.name: c.card_id for c in engine.cards()}
    knight, mirror = ids["Knight"], ids["Mirror"]
    assert engine.cards()[mirror].placement == Placement.MIRROR
    law = ElixirLaw.load()
    k, mi = engine.cards()[knight].elixir, engine.cards()[mirror].elixir
    clock = MatchClock.at(1)

    def after(plays: list[tuple[int, int]]) -> tuple[int, bool]:
        """The enemy count one tick on from 6 elixir, with ``plays`` at tick 0."""
        m = memories(engine)[0]
        m.start(0, 6000, 6000, [EMPTY_CARD] * 4, EMPTY_CARD)
        m.advance(1, clock.regular_ticks, clock.overtime, [], plays)
        return m.foe_fine, m.exact

    def paid(elixir: int) -> int:
        """6 elixir, less ``elixir``, one tick on: the law's own arithmetic."""
        start = law.seed_fine(6000)
        fine, _ = law.advance(start, elixir, 0, 1, clock.regular_ticks, clock.overtime)
        return fine

    assert after([(0, knight)]) == (paid(k), True)
    assert after([(0, knight), (0, mirror)]) == (paid(2 * k + mi), True), "a Knight, then its copy"
    cost_of_listed = paid(k + mi)
    assert paid(2 * k + mi) != cost_of_listed, "the listed price must read differently"
    assert after([(0, mirror)]) == (paid(mi), False), "nothing seen to copy"
