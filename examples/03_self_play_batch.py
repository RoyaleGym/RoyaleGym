"""Four battles as eight agent slots, so one policy learns from both sides of each.

This is the shape you train in. N games become 2N slots, and every slot sees the board
from its OWN side -- its king at the bottom -- so a single set of weights can play
either seat without learning which colour it is.

    python examples/03_self_play_batch.py
"""

import numpy as np

from royalegym import ClashParallelEnv, DefaultStateMutator, RustEngine
from royalegym.done_condition import GameOverCondition, StepLimitCondition
from royalegym.env import ClashSelfPlayVecEnv

GAMES = 4


def build_env() -> ClashParallelEnv:
    """One game. A FACTORY, not an instance.

    Every environment needs its own engine, observation builder, parser and reward:
    each of them carries per-battle state, and sharing one instance between games
    silently crosses the wires -- three windows onto one battle, each stepping it
    again on its own turn.
    """
    return ClashParallelEnv(
        engine=RustEngine(),
        state_mutator=DefaultStateMutator(),
        termination_cond=GameOverCondition(),
        truncation_cond=StepLimitCondition(200),
    )


def main() -> None:
    vec = ClashSelfPlayVecEnv(GAMES, build_env, viser=None)
    obs, _ = vec.reset(seed=0)

    print(f"{GAMES} games, {vec.num_envs} agent slots")
    for key, value in sorted(obs.items()):
        print(f"  {key:14} {np.asarray(value).shape}  {np.asarray(value).dtype}")

    rng = np.random.default_rng(0)
    finished = 0
    for _ in range(400):
        # Sample uniformly from the LEGAL moves of each slot. A real policy puts its
        # logits through the same mask; action_masks() is the bool array sb3-contrib's
        # MaskablePPO asks for.
        masks = vec.action_masks()
        actions = np.array([rng.choice(np.flatnonzero(m)) for m in masks])
        obs, rewards, terms, truncs, infos = vec.step(actions)

        # Autoreset is SAME_STEP: when a game ends, the observation returned is already
        # the NEXT game's first one, and the episode that just ended is in final_info.
        #
        # final_info is BATCHED the way gymnasium batches info: a dict of arrays over
        # the slots, each with a boolean "_<key>" beside it saying which entries are
        # real. It is not a list of dicts. infos["_final_info"] is the mask of slots
        # that ended on this step.
        ended = infos.get("_final_info")
        if ended is None or not ended.any():
            continue
        final = infos["final_info"]
        for slot in np.flatnonzero(ended):
            finished += 1
            if finished <= 2:
                print(
                    f"  slot {slot} episode ended: "
                    f"{final['episode_steps'][slot]} decisions, "
                    f"crowns {final['own_crowns'][slot]}-{final['enemy_crowns'][slot]}, "
                    f"elixir wasted on {final['elixir_leak_steps'][slot]} steps"
                )
    print(f"{finished} agent-episodes finished in 400 vector steps")
    print(f"rewards are {rewards.dtype}, which is what a policy's buffers hold")
    vec.close()


if __name__ == "__main__":
    main()
