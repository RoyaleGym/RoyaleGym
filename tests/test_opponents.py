"""The scripted opponents: that they play legally, differ from each other, and that the
ladder claims only what was measured.

WHY THIS EXISTS
    The package shipped two opponents, one that never plays and one that picks
    uniformly, and the tutorial's own twelve line bot three-crowns the second one. So
    there was no next rung to train against.

WHAT IT CATCHES
    An opponent returning an illegal action, which reaches the engine as a refusal and
    turns a scripted bot into a slightly worse random one without anything saying so.
    An opponent that ignores its own rule -- a defender placing over the river, a
    pusher hugging its own towers. Two opponents that are different names for the same
    behaviour. And the ladder claiming an ordering the games do not support.

WHAT IT CANNOT CATCH
    Whether these are good opponents to train against. They are cheap, legible and
    beatable, which is what a floor is for.

    Anything at MockEngine-versus-real-engine resolution: these run on the mock,
    because what is under test is the rules the bots follow, not the battles.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from royalegym import ClashParallelEnv, DefaultStateMutator, MockEngine, evaluate
from royalegym.action import NOOP
from royalegym.done_condition import GameOverCondition, StepLimitCondition
from royalegym.opponents import (
    DefendOpponent,
    FirstAffordableOpponent,
    PatientOpponent,
    PushOpponent,
    ladder,
)

#: The only two orderings the round robin in opponents.py established, at 40 games a
#: pairing. Everything else was too close to call and is deliberately NOT here.
#:
#: The third entry is how many games it takes to SHOW each one, which is not the same
#: for both and is the point of the exercise: noop losing is obvious in twenty, and
#: patient beating random is 13-27 at forty and "too close to call" at twenty. Checking
#: the second one with twenty games fails -- I wrote that version first and it failed,
#: correctly, for the reason this module is about.
MEASURED_ORDERINGS = (("noop", "random", 20), ("random", "patient", 40))


def env_fn(max_steps: int = 700):
    def build() -> ClashParallelEnv:
        return ClashParallelEnv(
            engine=MockEngine(),
            state_mutator=DefaultStateMutator(),
            termination_cond=GameOverCondition(),
            truncation_cond=StepLimitCondition(max_steps),
        )

    return build


def play(opponent, steps: int = 60, seed: int = 0) -> list[tuple[int, dict]]:
    """Run one bot as Blue and return (action, info) for each step it took."""
    env = env_fn()()
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(steps):
        if not env.agents:
            break
        action = int(opponent.act(obs["blue"], obs["blue"]["action_mask"], rng))
        out.append((action, obs["blue"]))
        obs, _, _, _, info = env.step({"blue": action, "red": NOOP})
        out[-1] = (action, info["blue"])
    env.close()
    return out


# ---------------------------------------------------------------------------
# legality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "opponent",
    [FirstAffordableOpponent(), DefendOpponent(), PushOpponent(), PatientOpponent(2)],
    ids=lambda o: type(o).__name__,
)
def test_an_opponent_never_returns_an_action_the_engine_refuses(opponent) -> None:
    """An illegal action is turned into a no-op and logged, so a bot that emits them
    is quietly a worse bot than it looks -- it does nothing on some fraction of steps
    and nothing raises."""
    statuses = [info["deploy_status"] for _, info in play(opponent)]
    refused = [s for s in statuses if s not in (-1, 0)]  # -1 no-op, 0 accepted
    assert not refused, f"{type(opponent).__name__} emitted refused actions: {refused[:5]}"


@pytest.mark.parametrize(
    "opponent",
    [FirstAffordableOpponent(), DefendOpponent(), PushOpponent()],
    ids=lambda o: type(o).__name__,
)
def test_an_opponent_that_is_meant_to_play_does_play(opponent) -> None:
    """Vacuity guard. A bot that returns no-op forever passes every legality test in
    this file, and three of these are defined by WHERE they place."""
    accepted = [1 for _, info in play(opponent) if info["deploy_status"] == 0]
    assert sum(accepted) >= 3, f"{type(opponent).__name__} never played a card"


# ---------------------------------------------------------------------------
# each one follows its own rule
# ---------------------------------------------------------------------------


def own_frame_rows(opponent, steps: int = 80) -> list[int]:
    """The own-frame row each accepted placement landed on."""
    env = env_fn()()
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(steps):
        if not env.agents:
            break
        action = int(opponent.act(obs["blue"], obs["blue"]["action_mask"], rng))
        planes = np.asarray(obs["blue"]["mask_planes"])
        obs, _, _, _, info = env.step({"blue": action, "red": NOOP})
        if action != NOOP and info["blue"]["deploy_status"] == 0:
            _, ny, nx = planes.shape
            rows.append(((action - 1) % (ny * nx)) // nx)
    env.close()
    return rows


def test_the_defender_stays_on_its_own_half() -> None:
    rows = own_frame_rows(DefendOpponent())
    assert rows, "the defender never placed anything"
    assert max(rows) < 16, f"placed at own-frame row {max(rows)}, past the halfway line"


def test_the_pusher_places_further_forward_than_the_defender() -> None:
    """The two are each other's opposite, so if their placements overlap, one of them
    is not doing what its name says."""
    push = own_frame_rows(PushOpponent())
    defend = own_frame_rows(DefendOpponent())
    assert push, "the pusher never placed anything"
    assert defend, "the defender never placed anything"
    assert min(push) > max(defend), f"push {min(push)}..{max(push)} vs defend {max(defend)}"


def decisions(steps: int = 400, seeds: int = 3) -> list[dict]:
    """Observations from battles a random bot plays on both seats, for comparing rules."""
    out = []
    for seed in range(seeds):
        env = env_fn(steps)()
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        rnd = dict(ladder())["random"]
        while env.agents:
            out.extend(obs[a] for a in env.agents)
            obs, *_ = env.step(
                {a: int(rnd.act(obs[a], obs[a]["action_mask"], rng)) for a in env.agents}
            )
        env.close()
    return out


def test_no_two_rungs_are_the_same_bot() -> None:
    """Two rungs that choose the same action are one rung listed twice.

    It happened: the defender used to take the lowest, then leftmost, legal cell of its
    own half, and the first-affordable bot's first legal cell is that same back corner.
    They agreed on every decision, and the round robin reported two identical rows as
    two results. Each pair must disagree on at least a tenth of the decisions where
    either one plays.
    """
    obs_list = decisions()
    rungs = [(n, o) for n, o in ladder() if n not in ("noop", "random")]
    rng = np.random.default_rng(0)
    acts = {n: [int(o.act(ob, ob["action_mask"], rng)) for ob in obs_list] for n, o in rungs}
    for (a, _), (b, _) in itertools.combinations(rungs, 2):
        played = [(x, y) for x, y in zip(acts[a], acts[b], strict=True) if x or y]
        assert len(played) >= 50, f"{a} and {b} played only {len(played)} times"
        differ = sum(x != y for x, y in played) / len(played)
        assert differ >= 0.1, f"{a} and {b} chose the same action on {1 - differ:.0%} of plays"


def test_the_patient_one_waits() -> None:
    """It holds until several cards are affordable, so it plays strictly less often."""
    eager = [1 for _, i in play(FirstAffordableOpponent()) if i["deploy_status"] == 0]
    patient = [1 for _, i in play(PatientOpponent(ready=4)) if i["deploy_status"] == 0]
    assert sum(patient) < sum(eager), f"patient played {sum(patient)}, eager {sum(eager)}"


def test_wanting_fewer_than_one_card_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1 card"):
        PatientOpponent(ready=0)


def test_an_observation_with_no_planes_gets_a_noop_rather_than_a_crash() -> None:
    """A custom ObsBuilder need not provide mask_planes, and an opponent meeting one
    should decline rather than raise inside somebody's rollout."""
    for opponent in (FirstAffordableOpponent(), DefendOpponent(), PushOpponent()):
        assert opponent.act({}, np.ones(5, dtype=np.int8), np.random.default_rng(0)) == NOOP


# ---------------------------------------------------------------------------
# the ladder claims only what was measured
# ---------------------------------------------------------------------------


def test_the_ladder_holds_the_orderings_it_claims() -> None:
    """Re-measure the two results opponents.py says are established.

    Each ordering is checked with the number of games that established it, because
    those numbers differ: twenty is plenty to show noop losing and not enough to show
    patient beating random, which is 13-27 at forty and inside the interval at twenty.
    """
    rungs = dict(ladder())
    for weaker, stronger, games in MEASURED_ORDERINGS:
        result = evaluate(
            rungs[weaker], rungs[stronger], env_fn(), games=games, seed=0, names=(weaker, stronger)
        )
        assert result.better == stronger, (
            f"opponents.py claims {stronger} beats {weaker}, and 20 games say "
            f"{result.summary()}"
        )


def test_the_middle_rungs_are_not_claimed_to_be_ordered() -> None:
    """The inverse test, and the one that keeps the docs honest.

    first-affordable, defend and push are described in increasing order of
    sophistication and cannot be told apart by forty games. If a future edit claims an
    ordering between them, this fails -- which is the right way round, because the
    claim is the thing that would be wrong, not the code.
    """
    names = [name for name, _ in ladder()]
    assert names.index("first-affordable") < names.index("defend") < names.index("push")
    middle = {("first-affordable", "defend"), ("defend", "push"), ("first-affordable", "push")}
    claimed = {(weaker, stronger) for weaker, stronger, _ in MEASURED_ORDERINGS}
    assert not middle & claimed, (
        "an ordering between the middle rungs has been claimed; the round robin in "
        "opponents.py found all three too close to call at 40 games"
    )
