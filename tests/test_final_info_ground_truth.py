"""The episode summary a trainer reads, graded against the engine's own final state.

The last info of an episode says how the battle went: who won, the crowns, the towers
left, how long it ran, how long each bar sat full, and the reward term by term. A learner
plots these numbers and a ladder scores them. A summary can be internally consistent and
still describe the wrong seat, the wrong clock or the next battle, so each field here is
compared with a quantity computed from ``engine.state()``, never with another field:

    winner, outcome            the engine's winner, and +1 / -1 / 0 from this seat's side
    own_crowns, enemy_crowns   the engine's crowns for this seat and for the other one
    own_tower_hp_frac,         the mean over the three towers of the tower entity's
    enemy_tower_hp_frac        hp / max_hp, a fallen tower counting 0
    episode_ticks              the engine tick at game over minus the tick after reset
    episode_steps              the number of steps taken
    elixir_leak_steps          steps across which this seat's bar sat at the engine's cap
    elixir_count_exact         True when this seat's observed count of the other bar
                               matched the engine at every step (only that direction)
    reward_sum/<Term>          the term's true per-step values, summed over the episode
    tick                       the engine tick at game over

The battle is the one tests/test_reward_ground_truth.py plays, on both engines, with each
seat leading once. The seats come out different in crowns, tower fractions, leak steps,
outcome and every reward sum, and the tick count is not the step count times the ticks
per step. The first test checks that before anything is graded, because a summary that
swaps the seats cannot be caught by a battle in which the seats come out equal. Only
elixir_count_exact comes out the same, True for both seats.

The vector env delivers the summary in ``infos["final_info"]``, after it has already
reset the game for the next episode. The last test reads it there and grades it against
the state the engine held just before that reset.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from royalegym import ClashParallelEnv, ClashSelfPlayVecEnv
from royalegym.mock_engine import MockEngine
from royalegym.protocol import BLUE, RED, BattleState, Winner
from royalegym.rust_engine import RustEngine
from royalegym.state_mutator import StateMutator
from test_reward_ground_truth import (
    CASES,
    DRAW_VALUE,
    LEADER_SCRIPT,
    TOL,
    TOWER_XFAIL,
    TOWERS,
    TRAILER_SCRIPT,
    TRUTH,
    WEIGHTS,
    Battle,
    all_terms,
    battle,
    make_engine,
    scenario_setup,
    scripted_action,
)

SEATS = (BLUE, RED)


def with_tower_xfail(cases):
    """The compiled engine's cases, expected to fail strictly on the tower maximum."""
    return [
        p
        if p.values[0] == "mock"
        else pytest.param(*p.values, id=p.id, marks=[*p.marks, TOWER_XFAIL])
        for p in cases
    ]


def summary(b: Battle, team: int) -> dict:
    return b.steps[-1].info[team]


def final(b: Battle) -> BattleState:
    return b.steps[-1].cur


def outcome(state: BattleState, team: int) -> int:
    if state.winner == Winner.DRAW:
        return 0
    return 1 if state.winner == team else -1


def tower_fraction(state: BattleState, team: int) -> Fraction:
    """Mean hp / max_hp over the three towers, from the tower entities. Fallen is 0."""
    standing = sum(
        (
            Fraction(max(0, e.hp), e.max_hp)
            for e in state.entities
            if e.team == team and e.kind in TOWERS
        ),
        Fraction(0),
    )
    return standing / 3


def leak_steps(b: Battle, team: int) -> int:
    return sum(
        1
        for s in b.steps
        if s.prev.players[team].elixir_milli >= b.cap and s.cur.players[team].elixir_milli >= b.cap
    )


# ---------------------------------------------------------------- the checks


@pytest.mark.parametrize(("kind", "leader"), CASES)
def test_the_battle_can_tell_the_seats_and_the_clock_apart(kind, leader):
    b = battle(kind, leader)
    end = final(b)
    assert end.game_over
    assert end.players[BLUE].crowns != end.players[RED].crowns
    assert tower_fraction(end, BLUE) != tower_fraction(end, RED)
    assert leak_steps(b, BLUE) != leak_steps(b, RED)
    assert outcome(end, BLUE) != outcome(end, RED)
    for name, truth in TRUTH.items():
        sums = [sum(truth(b, s, team) for s in b.steps) for team in SEATS]
        assert abs(sums[BLUE] - sums[RED]) > TOL, f"{name} sums to the same for both seats"
    ticks = end.tick - b.start.tick
    assert ticks != len(b.steps) * (b.steps[0].cur.tick - b.steps[0].prev.tick)
    for s in b.steps[:-1]:
        assert "own_crowns" not in s.info[BLUE], "a summary arrived before the episode ended"


@pytest.mark.parametrize(("kind", "leader"), CASES)
@pytest.mark.parametrize("team", SEATS, ids=["blue", "red"])
def test_the_summary_names_the_engines_result(kind, leader, team):
    b = battle(kind, leader)
    end = final(b)
    info = summary(b, team)
    assert info["winner"] == end.winner
    assert info["outcome"] == outcome(end, team)
    assert info["own_crowns"] == end.players[team].crowns
    assert info["enemy_crowns"] == end.players[1 - team].crowns
    assert info["tick"] == end.tick


@pytest.mark.parametrize(("kind", "leader"), CASES)
@pytest.mark.parametrize("team", SEATS, ids=["blue", "red"])
def test_the_summary_counts_the_episode_the_engine_ran(kind, leader, team):
    b = battle(kind, leader)
    info = summary(b, team)
    assert info["episode_steps"] == len(b.steps)
    assert info["episode_ticks"] == final(b).tick - b.start.tick
    assert info["elixir_leak_steps"] == leak_steps(b, team)


@pytest.mark.parametrize(("kind", "leader"), with_tower_xfail(CASES))
def test_tower_fractions_are_the_tower_entities_hp_over_their_max(kind, leader):
    b = battle(kind, leader)
    end = final(b)
    got = [
        (summary(b, t)["own_tower_hp_frac"], summary(b, t)["enemy_tower_hp_frac"]) for t in SEATS
    ]
    want = [(float(tower_fraction(end, t)), float(tower_fraction(end, 1 - t))) for t in SEATS]
    assert np.allclose(got, want, rtol=0, atol=TOL), (got, want)


def reward_sum_cases():
    out = []
    for name in sorted(TRUTH):
        cases = with_tower_xfail(CASES) if name == "TowerHPReward" else CASES
        out += [pytest.param(*p.values, name, id=f"{p.id}-{name}", marks=p.marks) for p in cases]
    return out


@pytest.mark.parametrize(("kind", "leader", "name"), reward_sum_cases())
def test_reward_sums_are_the_true_terms_summed_over_the_episode(kind, leader, name):
    b = battle(kind, leader)
    for team in SEATS:
        want = sum(WEIGHTS[name] * TRUTH[name](b, s, team) for s in b.steps)
        got = summary(b, team)[f"reward_sum/{name}"]
        assert abs(got - want) <= 1e-7, (team, got, want)


@pytest.mark.parametrize(("kind", "leader"), CASES)
def test_elixir_count_is_exact_when_the_count_matched_the_engine(kind, leader):
    """The flag's one checkable direction: a count that never strayed must say exact."""
    b = battle(kind, leader)
    worst = max(
        abs(s.enemy_elixir[team] * b.cap - s.cur.players[1 - team].elixir_milli)
        for s in b.steps
        for team in SEATS
    )
    assert worst < 1.0, f"the observed count strayed by {worst} milli-elixir"
    assert [summary(b, t)["elixir_count_exact"] for t in SEATS] == [True, True]


# ---------------------------------------------------------------- through the vector env


class RecordsFinalStates:
    """Keeps the state each battle ended in, taken just before the next reset replaces it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.finals: list[BattleState] = []
        self._running = False

    def reset(self, seed, setup):
        if self._running:
            self.finals.append(self.state())
        super().reset(seed, setup)
        self._running = True


class RecordedMock(RecordsFinalStates, MockEngine):
    pass


class RecordedRust(RecordsFinalStates, RustEngine):
    pass


class FixedSetup(StateMutator):
    def __init__(self, setup):
        self.setup = setup

    def build(self, rng, cards):
        return self.setup


@pytest.mark.parametrize(("kind", "leader"), CASES)
def test_the_vector_envs_final_info_describes_the_battle_that_ended(kind, leader):
    setup = scenario_setup(make_engine(kind), leader)
    engine = RecordedRust() if kind == "rust" else RecordedMock()
    vec = ClashSelfPlayVecEnv(
        1,
        lambda: ClashParallelEnv(engine, reward_fn=all_terms(), state_mutator=FixedSetup(setup)),
        viser=None,
    )
    vec.reset(seed=3)
    start = engine.state()
    names = {c.card_id: c.name for c in engine.cards()}
    cap = start.players[leader].elixir_milli
    scripts = {leader: LEADER_SCRIPT, 1 - leader: TRAILER_SCRIPT}
    rng = np.random.default_rng(3)
    elixir = []  # both bars before each step; after a step is before the next one
    for t in range(200):
        prev = engine.state()
        masks = vec.action_masks()
        actions = [
            scripted_action(prev, team, masks[team], t, scripts[team], names, rng) for team in SEATS
        ]
        _, _, terms, truncs, infos = vec.step(np.array(actions))
        elixir.append([p.elixir_milli for p in prev.players])
        if terms[0] or truncs[0]:
            break
    assert len(engine.finals) == 1, "the episode did not end, or ended more than once"
    end = engine.finals[0]
    assert end.game_over
    assert engine.state().tick == start.tick, "the next battle should have just begun"
    after = [*elixir[1:], [p.elixir_milli for p in end.players]]
    leaked = [
        sum(1 for a, b in zip(elixir, after, strict=True) if min(a[team], b[team]) >= cap)
        for team in SEATS
    ]
    # The seats must come out different, or a summary for the wrong seat would pass.
    assert end.players[BLUE].crowns != end.players[RED].crowns
    assert leaked[BLUE] != leaked[RED]
    fi = infos["final_info"]
    for team in SEATS:
        assert fi["own_crowns"][team] == end.players[team].crowns
        assert fi["enemy_crowns"][team] == end.players[1 - team].crowns
        assert fi["winner"][team] == end.winner
        assert fi["outcome"][team] == outcome(end, team)
        assert fi["tick"][team] == end.tick
        assert fi["episode_ticks"][team] == end.tick - start.tick
        assert fi["episode_steps"][team] == len(elixir)
        assert fi["elixir_leak_steps"][team] == leaked[team]
        want_winloss = WEIGHTS["WinLossReward"] * (
            DRAW_VALUE if end.winner == Winner.DRAW else outcome(end, team)
        )
        assert fi["reward_sum/WinLossReward"][team] == want_winloss
