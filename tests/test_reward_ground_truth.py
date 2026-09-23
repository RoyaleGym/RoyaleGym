"""Every reward term graded against the quantity its name claims, read from engine.state().

A term can be internally consistent and still measure the wrong thing. A crown term that
scores the other seat's crowns, a tower term that divides by the wrong maximum, a leak
penalty that watches the wrong bar: each of these is antisymmetric, finite and plausible,
and each passes a test that compares the reward with itself or with the other seat. So
nothing here compares one seat to the other. Each check takes a term's name as a claim
and computes that quantity from the engine's own states before and after the step, using
only ``engine.state()``, the card catalogue, the arena and ``protocol.to_own``:

    WinLossReward         +1 / -1 on the step the engine declares this seat's win / loss
    CrownReward           change in own crowns minus change in enemy crowns
    TowerHPReward         change in own towers' hp / max_hp minus the enemy's, read from
                          the tower ENTITIES, weighted per tower
    ElixirTradeReward     elixir value of enemy units that left the board minus own
                          (this deck casts no spell and summons nothing, which is what
                          keeps that the whole truth -- see ``true_trade``)
    ElixirLeakPenalty     -1 when this seat's bar sat at the cap across the step, the cap
                          being what the engine clamps a huge starting bar to
    PlacementDepthReward  2 * y_own / height - 1 of the unit this seat's play put down
    IllegalActionPenalty  -1 when this seat sent a command and nothing of its appeared

THE BATTLE. It starts 30 s before the end of regulation. The leader has a Knight and two
Skeletons at the trailer's own-left princess tower, which starts at half hp, so the
leader takes a crown mid-battle and goes on to win, at the end of regulation or in
overtime. The trailer has an Archer and a Valkyrie at the leader's towers, which start
damaged by different amounts. The leader starts with a full bar and plays rarely; the
trailer starts with two elixir and plays often. Each seat sends commands the engine
refuses, on different steps. Played cards are single-unit cards, which put one unit on
the tile they were given on both engines, so a placement can be read back from the
board. The battle runs once with Blue leading and once with Red leading, so every claim
is graded from both seats in both roles.

Before any term is graded, ``test_every_term_can_see_a_seat_swap`` requires that the
term's true value is non-zero on some step and differs between the seats on some step.
Without that, a term that never fires, or a swap of the two seats, would pass.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cache

import numpy as np
import pytest
from _pytest.outcomes import Skipped

from royalegym import ClashParallelEnv
from royalegym.mock_engine import MockEngine
from royalegym.protocol import (
    BLUE,
    HAND_SIZE,
    RED,
    BattleState,
    EntityKind,
    MatchSetup,
    Placement,
    ShuffleMode,
    SpawnSpec,
    TowerSlot,
    Winner,
    derived_cards_vintage,
    to_engine,
    to_own,
)
from royalegym.reward import (
    CombinedReward,
    CrownReward,
    ElixirLeakPenalty,
    ElixirTradeReward,
    IllegalActionPenalty,
    PlacementDepthReward,
    TowerHPReward,
    WinLossReward,
)
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available

needs_rust = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))

DECK = ("Knight", "Giant", "MiniPekka", "Musketeer", "HogRider", "Valkyrie", "Cannon", "Archer")
# The cards the scripted seats play. Each puts one unit on its tap on both engines (a
# formation spreads several around it), so the board says where a play landed. Archer
# only cycles.
PLAYED = frozenset(DECK[:7])
TOWERS = (EntityKind.KING_TOWER, EntityKind.PRINCESS_TOWER)
# A card that puts nothing of its own on the board, so its elixir is spent at the tap.
SPELL_PLACEMENTS = (Placement.SPELL, Placement.ROLLING, Placement.SPELL_NOT_ON_WATER)

# Distinct weights, so a term reported under another term's name, or with another term's
# weight, cannot come out right. The draw value is distinct from 0 and from +-1 for the
# same reason.
KING_WEIGHT, PRINCESS_WEIGHT = 2.0, 0.5
TRADE_SCALE = 4.0
DRAW_VALUE = 0.25
WEIGHTS = {
    "WinLossReward": 1.0,
    "CrownReward": 0.7,
    "TowerHPReward": 0.3,
    "ElixirTradeReward": 0.2,
    "ElixirLeakPenalty": 0.05,
    "PlacementDepthReward": 0.11,
    "IllegalActionPenalty": 0.13,
}
TOL = 1e-9

# The two seats' scripts: play a legal card every `every` steps, send a refused command
# on the steps listed. Different rates and steps, so the seats' values differ.
LEADER_SCRIPT = (9, (12,))
TRAILER_SCRIPT = (3, (5, 16, 27))


def make_engine(kind: str):
    return RustEngine() if kind == "rust" else MockEngine()


def scenario_setup(engine, leader: int) -> MatchSetup:
    """The battle in the module doc, laid out in the leader's own frame."""
    trailer = 1 - leader
    cards = {c.name: c for c in engine.cards()}
    deck = [cards[n].card_id for n in DECK]
    arena = engine.arena()
    tile = arena.subtile
    engine.reset(1, MatchSetup(decks=[deck, deck]))
    probe = engine.state()
    full = {(e.team, e.tower_slot): e.max_hp for e in probe.entities if e.kind in TOWERS}
    tower_hp = [[0, 0, 0], [0, 0, 0]]
    tower_hp[leader] = [
        full[(leader, TowerSlot.KING)] - 300,
        full[(leader, TowerSlot.LEFT)] - 700,
        full[(leader, TowerSlot.RIGHT)],
    ]
    tower_hp[trailer] = [
        full[(trailer, TowerSlot.KING)],
        full[(trailer, TowerSlot.LEFT)] // 2,
        full[(trailer, TowerSlot.RIGHT)] - 100,
    ]

    def unit(team, name, tx, ty):
        x, y = to_engine(arena, team, int(tx * tile), int(ty * tile))
        return SpawnSpec(team=team, card_id=cards[name].card_id, x=x, y=y)

    elixir = [0, 0]
    elixir[leader] = 10**7  # clamped to the cap by the engine
    elixir[trailer] = 2000
    return MatchSetup(
        decks=[deck, deck],
        shuffle=ShuffleMode.NONE,
        # 3 ticks off a whole number of steps, so the last step is a short one.
        start_tick=probe.regular_ticks - 603,
        elixir_milli=elixir,
        tower_hp=tower_hp,
        spawns=[
            unit(leader, "Knight", 14.5, 20.0),
            unit(leader, "Skeletons", 14.0, 23.0),
            unit(leader, "Skeletons", 15.0, 23.0),
            unit(trailer, "Archer", 14.5, 20.5),
            unit(trailer, "Valkyrie", 3.5, 21.0),
        ],
    )


def scripted_action(state: BattleState, team: int, mask, t: int, script, names, rng) -> int:
    """A legal single-unit play, a refused command, or the no-op, by the seat's script."""
    every, refused_at = script
    hand = state.players[team].hand
    per_slot = (len(mask) - 1) // HAND_SIZE
    playable = np.zeros(len(mask) - 1, dtype=bool)
    for slot, card in enumerate(hand):
        if names.get(card) in PLAYED:
            playable[slot * per_slot : (slot + 1) * per_slot] = True
    legal = np.asarray(mask[1:], dtype=bool)
    if t in refused_at:
        pool = np.flatnonzero(playable & ~legal)
    elif t % every == every - 1:
        pool = np.flatnonzero(playable & legal)
    else:
        return 0
    return int(rng.choice(pool)) + 1 if pool.size else 0


@dataclass
class Step:
    prev: BattleState
    cur: BattleState
    sent: dict[int, bool]
    reward: dict[int, float]
    terms: dict[int, dict[str, float]]
    # What each seat's observation says the OTHER seat's bar holds, as a fraction of the
    # cap, and the info dict the step returned (the episode summary on the last step).
    enemy_elixir: dict[int, float] = field(default_factory=dict)
    info: dict[int, dict] = field(default_factory=dict)


@dataclass
class Battle:
    engine: object
    start: BattleState | None = None
    steps: list[Step] = field(default_factory=list)
    cap: int = 0
    names: dict[int, str] = field(default_factory=dict)
    value: dict[int, Fraction] = field(default_factory=dict)
    # What one unit of each card looks like, and which cards are spells: both only so
    # ``true_trade`` can check that this battle stays inside the case it grades.
    rows: dict[int, tuple[int, int, bool]] = field(default_factory=dict)
    spell_ids: set[int] = field(default_factory=set)


def all_terms():
    return CombinedReward(
        [
            (WinLossReward(draw=DRAW_VALUE), WEIGHTS["WinLossReward"]),
            (CrownReward(), WEIGHTS["CrownReward"]),
            (
                TowerHPReward(king_weight=KING_WEIGHT, princess_weight=PRINCESS_WEIGHT),
                WEIGHTS["TowerHPReward"],
            ),
            (ElixirTradeReward(scale=TRADE_SCALE), WEIGHTS["ElixirTradeReward"]),
            (ElixirLeakPenalty(), WEIGHTS["ElixirLeakPenalty"]),
            (PlacementDepthReward(), WEIGHTS["PlacementDepthReward"]),
            (IllegalActionPenalty(), WEIGHTS["IllegalActionPenalty"]),
        ]
    )


@cache
def battle(kind: str, leader: int) -> Battle:
    """Play the scripted battle once per (engine, leader) and keep every transition."""
    engine = make_engine(kind)
    setup = scenario_setup(engine, leader)
    env = ClashParallelEnv(engine, reward_fn=all_terms())
    obs, _ = env.reset(seed=3, options={"setup": setup})
    out = Battle(engine=engine, start=engine.state())
    enemy_elixir_at = env.obs_builder.vector_offsets()["enemy_elixir"]
    out.names = {c.card_id: c.name for c in engine.cards()}
    out.value = {c.card_id: Fraction(c.elixir, max(1, c.count)) for c in engine.cards()}
    out.rows = {c.card_id: (c.hitpoints, c.radius, c.flying) for c in engine.cards()}
    held = {c.card_id for c in engine.cards() if c.name in DECK}
    out.spell_ids = {
        c.card_id
        for c in engine.cards()
        if c.card_id in held and c.placement in SPELL_PLACEMENTS
    }
    # The cap as the ENGINE applies it: the leader asked for far more than any bar holds.
    out.cap = engine.state().players[leader].elixir_milli
    scripts = {leader: LEADER_SCRIPT, 1 - leader: TRAILER_SCRIPT}
    rng = np.random.default_rng(3)
    for t in range(200):
        prev = engine.state()
        actions = {}
        for agent, team in (("blue", BLUE), ("red", RED)):
            actions[agent] = scripted_action(
                prev, team, obs[agent]["action_mask"], t, scripts[team], out.names, rng
            )
        obs, rewards, term, trunc, infos = env.step(actions)
        seats = (("blue", BLUE), ("red", RED))
        counted = {team: float(obs[agent]["vector"][enemy_elixir_at][0]) for agent, team in seats}
        out.steps.append(
            Step(
                prev=prev,
                cur=engine.state(),
                sent={BLUE: actions["blue"] != 0, RED: actions["red"] != 0},
                reward={BLUE: rewards["blue"], RED: rewards["red"]},
                terms={team: dict(env.reward_fn.terms_for(team)) for team in (BLUE, RED)},
                enemy_elixir=counted,
                info={team: dict(infos[agent]) for agent, team in seats},
            )
        )
        if term["blue"] or trunc["blue"]:
            break
    return out


# ---------------------------------------------------------------- the quantities


def new_units(prev: BattleState, cur: BattleState, team: int):
    before = {e.uid for e in prev.entities}
    return [e for e in cur.entities if e.team == team and e.uid not in before]


def true_winloss(b: Battle, s: Step, team: int) -> float:
    if not s.cur.game_over or s.prev.game_over:
        return 0.0
    if s.cur.winner == Winner.DRAW:
        return DRAW_VALUE
    return 1.0 if s.cur.winner == team else -1.0


def true_crowns(b: Battle, s: Step, team: int) -> float:
    def gained(t):
        return s.cur.players[t].crowns - s.prev.players[t].crowns

    return float(gained(team) - gained(1 - team))


def tower_potential(state: BattleState, team: int) -> Fraction:
    """Weighted hp / max_hp over this team's standing tower entities. A fallen tower is 0."""
    total = Fraction(0)
    for e in state.entities:
        if e.team == team and e.kind in TOWERS:
            w = KING_WEIGHT if e.kind == EntityKind.KING_TOWER else PRINCESS_WEIGHT
            total += Fraction(w) * Fraction(max(0, e.hp), e.max_hp)
    return total


def true_towers(b: Battle, s: Step, team: int) -> float:
    own = tower_potential(s.cur, team) - tower_potential(s.prev, team)
    foe = tower_potential(s.cur, 1 - team) - tower_potential(s.prev, 1 - team)
    return float(own - foe)


def true_trade(b: Battle, s: Step, team: int) -> float:
    """Elixir of the enemy units that left the board, minus this seat's.

    THE TERM HAS TWO MORE CLAUSES AND THIS BATTLE EXERCISES NEITHER. It also charges a
    seat the elixir of a spell it cast, and scores zero for a unit a card produced
    rather than summoned, because the catalogue prices no such unit and reports it
    under whichever card could have made it. This deck holds no spell, and every unit
    in it is the one its card's row describes, so the plain sum below is still the
    whole truth here. The two asserts say that rather than a comment: widen DECK and
    they fire instead of grading a term whose definition has moved. The clauses
    themselves are graded in tests/test_rewards.py.
    """
    assert not b.spell_ids, "this deck now holds a spell; the trade truth must charge the cast"
    alive = {e.uid for e in s.cur.entities}
    total = Fraction(0)
    for e in s.prev.entities:
        if e.uid in alive or e.kind in TOWERS:
            continue
        # ``rows.get``, not ``rows[...]``: a unit a card produced can be reported under no
        # card at all, and that is the very case this has to name rather than crash on.
        assert b.rows.get(e.card_id) == (e.max_hp, e.radius, e.flying), (
            f"a unit left the board that is not the one its card describes "
            f"({b.names.get(e.card_id, e.card_id)}); the trade truth must score a unit a "
            "card produced at zero"
        )
        total += b.value[e.card_id] if e.team != team else -b.value[e.card_id]
    return float(total / Fraction(TRADE_SCALE))


def true_leak(b: Battle, s: Step, team: int) -> float:
    before, after = s.prev.players[team].elixir_milli, s.cur.players[team].elixir_milli
    return -1.0 if before >= b.cap and after >= b.cap else 0.0


def true_depth(b: Battle, s: Step, team: int) -> float:
    """Read at tile resolution, the resolution the action space places at.

    On the compiled engine a unit that is still deploying can be nudged by a neighbour
    (a Hog Rider stood 3.7k units from its tap after one step, of an 18k tile), so the
    tile it stands on is the record of where it was put, not its exact point.
    """
    arena = b.engine.arena()
    tile = arena.subtile
    total = 0.0
    for e in new_units(s.prev, s.cur, team):
        _, y_own = to_own(arena, team, e.x, e.y)
        y_tile = (y_own // tile) * tile + tile // 2
        total += 2.0 * y_tile / arena.height - 1.0
    return total


def true_refused(b: Battle, s: Step, team: int) -> float:
    return -1.0 if s.sent[team] and not new_units(s.prev, s.cur, team) else 0.0


TRUTH = {
    "WinLossReward": true_winloss,
    "CrownReward": true_crowns,
    "TowerHPReward": true_towers,
    "ElixirTradeReward": true_trade,
    "ElixirLeakPenalty": true_leak,
    "PlacementDepthReward": true_depth,
    "IllegalActionPenalty": true_refused,
}

CASES = [
    pytest.param("mock", BLUE, id="mock-blue-leads"),
    pytest.param("mock", RED, id="mock-red-leads"),
    pytest.param("rust", BLUE, id="rust-blue-leads", marks=needs_rust),
    pytest.param("rust", RED, id="rust-red-leads", marks=needs_rust),
]

# Until 2026-09-22 the compiled engine reported PlayerState.tower_max_hp on the CARD
# ladder rather than the tower ladder, so a full king read 4824 of a 6144 maximum while
# the tower terms divide by tower_max_hp: a king and a princess were scaled by different
# factors and a full tower was never 1. These cases were strict expected failures naming
# it, and they flipped to XPASS the day the engine started reporting the towers' own
# maximum. They now run with no allowance on either engine.


# Two allowances, both naming an open defect rather than hiding one, both strict so the
# day the defect is fixed the test says so.
#
# RELOCATED, a DeployResult that reports the tap and not the unit. Since 2026-09-22 the
# engine moves a building whose tile box does not fit to the nearest place it fits, and
# DeployResult still carries the COMMANDED x and y. PlacementDepthReward reads those, so
# it scores a point the building is not on (measured: -0.0721875 against a true
# -0.0859375). Sim is adding the resolved position to the step result; when it lands,
# PlacementDepthReward reads it and these pass.
RELOCATED_REASON = (
    "DeployResult carries the commanded position, and the engine now relocates a building "
    "whose box does not fit, so a depth read from the command is not where the unit stands"
)
RELOCATED_XFAIL = pytest.mark.xfail(strict=True, raises=AssertionError, reason=RELOCATED_REASON)
#
# RED_LEADS_OUTCOME. The scripted scenario is not a mirror of itself (on MockEngine the
# Blue-leader battle ends at tick 3600 and the Red-leader one at 4182), so "the leader
# wins" is a claim about this scenario and not a symmetry property. On the compiled
# engine the Red-leader battle now ends at regulation with Blue ahead, and it does so for
# every reset seed 0..9, so it is structural rather than luck. The likely cause is the
# tower-maximum fix of the same rebuild moving a tower-hp tiebreak. Reported to sim.
# The scenario needs a leader advantage that does not rest on a tiebreak.
RED_LEADS_REASON = (
    "the scripted Red-leader battle no longer ends with Red ahead on the compiled engine, "
    "for any reset seed; the leader's advantage rests on a tower-hp tiebreak that moved"
)
RED_LEADS_XFAIL = pytest.mark.xfail(strict=True, raises=AssertionError, reason=RED_LEADS_REASON)


def only(cases, kind, leader, mark):
    """``cases`` with ``mark`` added to the one (kind, leader) case."""
    return [
        pytest.param(*p.values, id=p.id, marks=[*p.marks, mark])
        if p.values == (kind, leader)
        else p
        for p in cases
    ]


#: The card table these scripted scenarios were tuned against. They set tower HP a few
#: hundred points from a crown and give the leader unlimited elixir for the last 603
#: ticks, so whether a tower actually falls depends on the CARDS' damage, which is a
#: property of the installed table and not of this repo.
SCENARIO_CARD_TABLE = "15.535.29"


def require_a_decided_battle(b: Battle, kind: str) -> None:
    """Refuse to grade seat differences on a battle that ended level, and say which.

    A clone following the README runs ``extract_cards.py --vintage 2018`` and gets a
    different table: 66 loadable cards where this machine has 100, with different damage.
    Measured on two independent clones, 2026-09-22. On that table this scenario ends with
    the seats level, and the tests below then compare two identical numbers and fail with
    ``assert 1 != 1``, which tells a contributor nothing about why.

    So: level on a DIFFERENT table is a loud skip naming the table, and level on the
    table these were tuned for is a real failure. A skip is not a pass, and this one says
    what would have to be true to run it.
    """
    end = b.steps[-1].cur
    if end.players[BLUE].crowns != end.players[RED].crowns:
        return
    vintage = derived_cards_vintage()
    if SCENARIO_CARD_TABLE not in vintage:
        pytest.skip(
            f"SKIPPED, NOT PASSED: the scripted {kind} battle ends level on this card "
            f"table ({vintage!r}), so the two seats cannot be told apart and nothing "
            f"below would be graded. It was tuned against {SCENARIO_CARD_TABLE!r}, where "
            "the leader takes a tower inside the scripted window. The reward terms "
            "themselves are still graded by the MockEngine cases, which do not depend on "
            "this table."
        )
    raise AssertionError(
        f"the scripted {kind} battle ended level on {vintage!r}, the very table it was "
        "tuned against. That is a regression in the scenario or in the engine, not a "
        "difference in installed data."
    )


def mismatches(b: Battle, name: str):
    out = []
    for i, s in enumerate(b.steps):
        for team in (BLUE, RED):
            want = WEIGHTS[name] * TRUTH[name](b, s, team)
            got = s.terms[team][name]
            if abs(got - want) > TOL:
                out.append((i, team, got, want))
    return out


# ---------------------------------------------------------------- the checks


@pytest.mark.parametrize(("kind", "leader"), only(CASES, "rust", RED, RED_LEADS_XFAIL))
def test_the_battle_ends_with_the_leader_winning_inside_the_window(kind, leader):
    b = battle(kind, leader)
    last = b.steps[-1].cur
    assert last.game_over
    assert last.winner == leader
    assert last.players[leader].crowns > last.players[1 - leader].crowns
    # Every play landed one unit or none, so the board can say where each one went.
    for s in b.steps:
        for team in (BLUE, RED):
            assert len(new_units(s.prev, s.cur, team)) <= 1


@pytest.mark.parametrize(("kind", "leader"), CASES)
@pytest.mark.parametrize("name", sorted(TRUTH))
def test_every_term_can_see_a_seat_swap(kind, leader, name):
    """Non-zero on some step, and different between the seats on some step."""
    b = battle(kind, leader)
    values = [(TRUTH[name](b, s, BLUE), TRUTH[name](b, s, RED)) for s in b.steps]
    assert any(v != 0.0 for pair in values for v in pair), f"{name} never fires"
    assert any(abs(bv - rv) > TOL for bv, rv in values), f"{name} is the same for both seats"


@pytest.mark.parametrize(("kind", "leader"), CASES)
def test_winloss_is_the_engine_verdict_on_the_step_it_arrives(kind, leader):
    assert mismatches(battle(kind, leader), "WinLossReward") == []


@pytest.mark.parametrize(("kind", "leader"), CASES)
def test_crown_reward_is_the_crown_difference_the_engine_records(kind, leader):
    assert mismatches(battle(kind, leader), "CrownReward") == []


@pytest.mark.parametrize(("kind", "leader"), CASES)
def test_tower_reward_follows_the_tower_entities_hp_over_their_max(kind, leader):
    assert mismatches(battle(kind, leader), "TowerHPReward") == []


@pytest.mark.parametrize(("kind", "leader"), CASES)
def test_elixir_trade_values_the_units_that_left_the_board(kind, leader):
    assert mismatches(battle(kind, leader), "ElixirTradeReward") == []


@pytest.mark.parametrize(("kind", "leader"), CASES)
def test_leak_penalty_fires_only_while_the_bar_sits_at_the_engine_cap(kind, leader):
    assert mismatches(battle(kind, leader), "ElixirLeakPenalty") == []


@pytest.mark.parametrize(("kind", "leader"), only(CASES, "rust", RED, RELOCATED_XFAIL))
def test_placement_depth_is_where_the_unit_stands_in_its_owners_frame(kind, leader):
    assert mismatches(battle(kind, leader), "PlacementDepthReward") == []


@pytest.mark.parametrize(("kind", "leader"), CASES)
def test_illegal_action_penalty_counts_the_commands_that_placed_nothing(kind, leader):
    assert mismatches(battle(kind, leader), "IllegalActionPenalty") == []


@pytest.mark.parametrize(("kind", "leader"), only(CASES, "rust", RED, RELOCATED_XFAIL))
def test_the_scalar_reward_is_the_weighted_sum_of_the_true_terms(kind, leader):
    """The number the learner actually receives, per seat, against the same quantities."""
    b = battle(kind, leader)
    bad = []
    for i, s in enumerate(b.steps):
        for team in (BLUE, RED):
            want = sum(WEIGHTS[n] * f(b, s, team) for n, f in TRUTH.items())
            if abs(s.reward[team] - want) > TOL:
                bad.append((i, team, s.reward[team], want))
    assert bad == []


@pytest.mark.parametrize("kind", ["mock", pytest.param("rust", marks=needs_rust)])
def test_winloss_pays_the_draw_value_to_both_seats_on_a_drawn_battle(kind):
    """Overtime runs out with both sides' towers at equal hp, which every tiebreak calls a draw."""
    engine = make_engine(kind)
    cards = {c.name: c for c in engine.cards()}
    deck = [cards[n].card_id for n in DECK]
    engine.reset(1, MatchSetup(decks=[deck, deck]))
    probe = engine.state()
    hp = [2000, 1000, 1000]
    last = probe.regular_ticks + probe.overtime_ticks - 25
    env = ClashParallelEnv(engine, reward_fn=WinLossReward(draw=DRAW_VALUE))
    env.reset(
        seed=2,
        options={"setup": MatchSetup(decks=[deck, deck], start_tick=last, tower_hp=[hp, hp])},
    )
    seen = []
    while True:
        prev = engine.state()
        _, rewards, term, trunc, _ = env.step({"blue": 0, "red": 0})
        cur = engine.state()
        seen.append((prev.game_over, cur.game_over, cur.winner, rewards["blue"], rewards["red"]))
        if term["blue"] or trunc["blue"]:
            break
    assert len(seen) > 1
    assert seen[-1][:3] == (False, True, Winner.DRAW)
    assert seen[-1][3:] == (DRAW_VALUE, DRAW_VALUE)
    assert all(step[3:] == (0.0, 0.0) for step in seen[:-1])


def _level_battle(crowns: tuple[int, int]):
    """The smallest thing shaped like what ``require_a_decided_battle`` reads."""
    player = lambda c: types.SimpleNamespace(crowns=c)  # noqa: E731
    cur = types.SimpleNamespace(players=[player(crowns[0]), player(crowns[1])])
    return types.SimpleNamespace(steps=[types.SimpleNamespace(cur=cur)])


def test_a_level_battle_skips_on_another_card_table_and_fails_on_this_one(monkeypatch) -> None:
    """The guard has two branches and they must not be the same branch.

    A clone's table ending the scenario level is a fact about the reader's data and is a
    loud skip. The SAME outcome on the table these were tuned for is a regression and has
    to fail. A guard that skipped in both cases would hide the second, which is the only
    one that means something is broken.
    """
    decided = _level_battle((1, 0))
    require_a_decided_battle(decided, "decided")  # returns, no skip, no failure

    monkeypatch.setattr(
        "test_reward_ground_truth.derived_cards_vintage",
        lambda *a, **k: "~2018 client data (PRE-2025)",
    )
    with pytest.raises(Skipped) as skipped:
        require_a_decided_battle(_level_battle((1, 1)), "clone")
    assert "SKIPPED, NOT PASSED" in str(skipped.value)
    assert "2018" in str(skipped.value), "the skip does not name the table that caused it"

    monkeypatch.setattr(
        "test_reward_ground_truth.derived_cards_vintage",
        lambda *a, **k: f"{SCENARIO_CARD_TABLE} client (2026, LIVE build family)",
    )
    with pytest.raises(AssertionError, match="the very table it was tuned against"):
        require_a_decided_battle(_level_battle((1, 1)), "ours")
