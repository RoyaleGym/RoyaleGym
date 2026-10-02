"""Train a Clash Royale bot.

    pip install "royalegym[all]"
    python quickstart.py

It prints a line after every update. To watch it play, run `royaleviser` in a second
terminal. Stop it with Ctrl+C whenever you like: it saves, and when you run it again it
carries on where it stopped.
"""


def build_env():
    from royalegym import ClashParallelEnv, RustEngine
    from royalegym import DefaultStateMutator, SpatialObsBuilder, TileActionParser
    from royalegym import CombinedReward, CrownReward, TowerHPReward, WinLossReward
    from royalegym import GameOverCondition

    deck = ["Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Cannon", "Musketeer"]
    decision_ms = 500  # the bot decides what to do every half second of game time

    # How each battle starts: both sides get the deck above.
    state_mutator = DefaultStateMutator(decks=[deck, deck])

    # What the bot sees: the arena as a grid of tiles, plus its hand, elixir and the clock.
    obs_builder = SpatialObsBuilder()

    # What the bot can do: wait, or play one of the 4 cards in its hand on one of 18 x 32 tiles.
    action_parser = TileActionParser()

    # What the bot is paid for. Winning is what counts; the other two help it learn faster.
    reward_fn = CombinedReward([
        (WinLossReward(), 1.0),  # +1 for a win, -1 for a loss
        (CrownReward(), 0.2),    # for each crown taken, minus each crown lost
        (TowerHPReward(), 0.1),  # tower damage done, minus tower damage taken
    ])

    # When a battle ends. A battle always ends by itself (3 minutes, plus overtime),
    # so nothing needs cutting short.
    termination_cond = GameOverCondition()
    truncation_cond = None

    return ClashParallelEnv(
        RustEngine(),
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_cond,
        truncation_cond=truncation_cond,
        decision_ms=decision_ms,
    )


if __name__ == "__main__":
    from royalelearn import Learner

    learner = Learner(
        build_env,
        n_envs=32,                    # battles played at once; more keeps the GPU busier
        opponent="random",            # who it plays: "random" (a bot making random moves),
                                      # "noop" (never plays a card) or "self" (copies of itself)
        device="auto",                # your graphics card if torch can see it, else the CPU
        trunk_channels=64,            # the network's width...
        trunk_blocks=4,               # ...and depth. Bigger can learn more; each update is slower
        steps_per_update=16_384,      # decisions collected before each update of the network
        ppo_batch_size=16_384,        # decisions per update; set this equal to steps_per_update
        ppo_minibatch_size=2_048,     # decisions per gradient step: as many as your GPU's memory
                                      # holds (the trainer says so before it starts if it can't)
        ppo_epochs=2,                 # passes over each batch
        policy_lr=2e-4,               # how big a step the policy takes each update
        critic_lr=2e-4,               # the same, for the part that predicts the reward
        ppo_ent_coef=0.01,            # how much it is pushed to keep trying new moves
        checkpoint_every=200_000,     # save every 200,000 decisions (Ctrl+C saves too)
        timestep_limit=1_000_000_000, # stop after a billion decisions
        save_dir="runs/quickstart",   # checkpoints and logs go here
        viser=True,                   # watch a battle live: run `royaleviser` in a second terminal
        log_to_wandb=False,           # True draws charts on wandb.ai (pip install wandb)
    )
    learner.learn()
