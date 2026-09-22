"""Stop a run and start it again, and get the same battles you would have had.

The problem this solves is not obvious until it bites. An autoreset with no seed leaves
the generator running, so episode 400 of a game is only reachable by playing the 399
before it. A trainer that restores its weights perfectly still diverges from the run it
is continuing, and every difference is in the environment rather than the learner.

``autoreset_seed_fn`` names episodes instead of counting them, and ``episode_ordinals``
is the counter a checkpoint carries.

    python examples/06_resume_a_run.py
"""

import numpy as np

from royalegym import ClashParallelEnv, DefaultStateMutator, RustEngine
from royalegym.done_condition import GameOverCondition, StepLimitCondition
from royalegym.env import ClashSelfPlayVecEnv

GAMES, EPISODE_STEPS, RUN_SEED = 2, 20, 1234


def episode_seed(game: int, ordinal: int) -> int:
    """The seed of episode ``ordinal`` of game ``game``. A pure function of both.

    That is the whole trick: an episode has a NAME, so it can be reached directly
    rather than by replaying everything before it.
    """
    return (RUN_SEED * 1_000_003 + game * 9973 + ordinal) % 2**31


def build_env() -> ClashParallelEnv:
    return ClashParallelEnv(
        engine=RustEngine(),
        state_mutator=DefaultStateMutator(),
        termination_cond=GameOverCondition(),
        truncation_cond=StepLimitCondition(EPISODE_STEPS),
    )


def fingerprint(obs) -> str:
    return f"{np.asarray(obs['vector']).sum():.6f}"


def roll(vec, steps: int) -> list[str]:
    zero = np.zeros(vec.num_envs, dtype=np.int64)
    return [fingerprint(vec.step(zero)[0]) for _ in range(steps)]


def main() -> None:
    original = ClashSelfPlayVecEnv(GAMES, build_env, viser=None, autoreset_seed_fn=episode_seed)
    original.reset()
    roll(original, 3 * EPISODE_STEPS)  # three episodes played and gone

    # What a checkpoint carries. Two integers per game, and that is the whole of it.
    saved = original.episode_ordinals
    print(f"checkpoint at episode_ordinals = {saved}")

    # The original finishes the episode it was in, then plays on.
    kept_going = roll(original, EPISODE_STEPS)[-1:] + roll(original, EPISODE_STEPS)

    # A brand-new object, told nothing but those ordinals, continues the same row.
    resumed_env = ClashSelfPlayVecEnv(GAMES, build_env, viser=None, autoreset_seed_fn=episode_seed)
    resumed_env.set_episode_ordinals(saved)  # BEFORE reset: reset starts an episode
    obs, _ = resumed_env.reset()
    resumed = [fingerprint(obs), *roll(resumed_env, EPISODE_STEPS)]

    match = resumed == kept_going
    print(f"resumed run matches the original row for row: {match}")
    print(f"  original  {kept_going[:3]} ...")
    print(f"  resumed   {resumed[:3]} ...")

    # And the same thing WITHOUT the hook, which is what a run does by default.
    plain = ClashSelfPlayVecEnv(GAMES, build_env, viser=None)
    plain.reset(seed=RUN_SEED)
    roll(plain, 3 * EPISODE_STEPS)
    original_plain = roll(plain, EPISODE_STEPS)[-1:] + roll(plain, EPISODE_STEPS)
    fresh = ClashSelfPlayVecEnv(GAMES, build_env, viser=None)
    restarted = [fingerprint(fresh.reset(seed=RUN_SEED)[0]), *roll(fresh, EPISODE_STEPS)]
    print(f"without the hook, the same restart matches: {restarted == original_plain}")
    print("  -- it replays episode 0, because that is the only one it can reach")

    print()
    print("The counter names the NEXT episode, so the one that was half played when the")
    print("checkpoint was written is dropped rather than replayed: its transitions were")
    print("already in the original run's buffer.")
    for env in (original, resumed_env, plain, fresh):
        env.close()


if __name__ == "__main__":
    main()
