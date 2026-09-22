"""Write a reward function, and see which term the bot is actually chasing.

The reward is the one piece you are expected to write. Everything else ships with a
default; what your bot should WANT is the question the library cannot answer for you.

This shows both halves: writing a term, and reading the per-term breakdown, which is
the number that says whether a policy won the match or just farmed a shaping term.

    python examples/04_your_own_reward.py
"""

import numpy as np

from royalegym import (
    ClashParallelEnv,
    CombinedReward,
    CrownReward,
    DefaultStateMutator,
    RandomLegalOpponent,
    RewardFunction,
    RustEngine,
    WinLossReward,
)
from royalegym.done_condition import GameOverCondition, StepLimitCondition
from royalegym.protocol import to_own


class PlayItForward(RewardFunction):
    """Reward placing troops in the enemy half, punish hiding at the back.

    The archetypal shaping term, and the one that shows the trap. ``DeployResult``
    gives the position in the ENGINE frame, which is the same frame for both seats --
    so a term that scores ``r.y`` directly rewards Blue for pushing and Red for
    retreating, and a self-play run learns which colour it is instead of how to play.

    ``to_own`` turns the engine frame into the acting player's own frame, where
    forward is forward for both of them. That one call is the whole difference.
    """

    def __init__(self, weight: float = 0.01) -> None:
        self.weight = weight

    def bind(self, engine) -> None:
        self.arena = engine.arena()

    def reset(self, state) -> None:
        pass

    def get_reward(self, team, prev, state, results) -> float:
        total = 0.0
        for r in results:
            if r.team != team or r.status != 0:  # not mine, or the engine refused it
                continue
            _, y_own = to_own(self.arena, team, r.x, r.y)
            total += (2.0 * y_own / max(1, self.arena.height)) - 1.0
        return self.weight * total


def main() -> None:
    reward = CombinedReward(
        [
            (WinLossReward(), 1.0),
            (CrownReward(), 0.3),
            (PlayItForward(), 1.0),
        ]
    )
    env = ClashParallelEnv(
        engine=RustEngine(),
        reward_fn=reward,
        state_mutator=DefaultStateMutator(),
        termination_cond=GameOverCondition(),
        truncation_cond=StepLimitCondition(600),
    )
    obs, _ = env.reset(seed=3)
    rng = np.random.default_rng(3)
    policy = RandomLegalOpponent(0.5)

    while env.agents:
        actions = {a: policy.act(obs[a], obs[a]["action_mask"], rng) for a in env.agents}
        obs, rewards, terminated, truncated, info = env.step(actions)

    # env.reward_terms(team) is this episode's reward, term by term, for one seat. The
    # same numbers reach a training run as reward_sum/<TermName> in the terminal info,
    # so nothing has to reach into the env to get them.
    for name, team in (("blue", 0), ("red", 1)):
        terms = env.reward_terms(team)
        total = sum(terms.values())
        # Share of the MAGNITUDE, not of the signed total: with a negative total a
        # signed share makes a term that lost you the match look like a positive
        # contributor. What you want to know is how much of the signal each term was.
        weight = sum(abs(v) for v in terms.values()) or 1.0
        print(f"{name}: total {total:+.3f}")
        for term, value in sorted(terms.items()):
            print(f"    {term:20} {value:+8.3f}   {abs(value) / weight:5.1%} of the signal")

    print()
    print("If a shaping term is most of the total, that is what your bot is learning.")


if __name__ == "__main__":
    main()
