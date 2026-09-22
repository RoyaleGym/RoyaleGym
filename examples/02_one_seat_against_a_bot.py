"""One seat as the learner, the other a scripted opponent, through Gymnasium.

This is the shape a single-agent trainer expects: one observation, one action, one
reward. The other seat is played by an ``Opponent``, which is any object with an
``act(obs, mask, rng)`` method -- so a scripted bot, a saved policy, or a previous
version of the one you are training.

    python examples/02_one_seat_against_a_bot.py
"""

import gymnasium as gym
import numpy as np

import royalegym  # noqa: F401  -- importing the package is what registers the ids
from royalegym import RandomLegalOpponent


class RushLeft:
    """A scripted opponent: always play the cheapest legal move on the left.

    Deliberately simple, and a fair first thing to beat. An Opponent sees the same
    observation a learner would, so anything you can write as a policy you can drop
    in here -- which is how you build a ladder to train against.
    """

    def act(self, obs, mask, rng):
        del obs, rng
        legal = np.flatnonzero(mask)
        return int(legal[0]) if legal.size else 0


def play(env, policy, seed: int) -> float:
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total = 0.0
    while True:
        action = policy.act(obs, obs["action_mask"], rng)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total += reward
        if terminated or truncated:
            return total, info


def main() -> None:
    # The id names the engine. royalegym/ClashRoyale-v0 also exists and takes whatever
    # engine= you pass it, but it warns when nobody chose, because which engine
    # produced a number is not something a library should decide quietly.
    for name, opponent in (("random", RandomLegalOpponent(0.7)), ("rush-left", RushLeft())):
        env = gym.make("royalegym/ClashRoyaleRust-v0", opponent=opponent).unwrapped
        total, info = play(env, RandomLegalOpponent(0.7), seed=1)
        # The LAST info of an episode carries how the episode went, so a training run
        # reads it out of the info rather than keeping its own copy of the state.
        print(
            f"against {name:9}  return {total:+.3f}  "
            f"crowns {info['own_crowns']}-{info['enemy_crowns']}  "
            f"tower hp {info['own_tower_hp_frac']:.2f} vs {info['enemy_tower_hp_frac']:.2f}  "
            f"{info['episode_steps']} decisions"
        )
        env.close()


if __name__ == "__main__":
    main()
