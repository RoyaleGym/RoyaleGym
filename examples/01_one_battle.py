"""One whole battle, both seats, random legal moves.

The shortest thing that is a real battle. Run it and you have watched two players
play a match to the end in under a second.

    python examples/01_one_battle.py
"""

import numpy as np

from royalegym import ClashParallelEnv, DefaultStateMutator, RandomLegalOpponent, RustEngine

# Cards are looked up BY NAME. A card id is a position in the catalogue, and positions
# move between card tables, so the same number is not the same card on every machine.
DECK = ("Knight", "Archer", "Giant", "Minions", "Fireball", "Cannon", "Zap", "Musketeer")


def main() -> None:
    engine = RustEngine()
    by_name = {c.name: c.card_id for c in engine.cards()}
    deck = [by_name[n] for n in DECK]

    env = ClashParallelEnv(
        engine=engine,
        state_mutator=DefaultStateMutator(decks=[deck, deck]),
    )
    obs, _ = env.reset(seed=0)

    # The mask is the useful part: both players only ever choose from moves the engine
    # would accept, so neither wastes a turn on a card it cannot afford or a tile it is
    # not allowed to deploy on.
    first_legal = int(obs["blue"]["action_mask"].sum())

    # A match refuses every deploy for its opening ticks, so on the very first decision
    # the only legal move is the no-op and that number is 1. Worth reporting rather than
    # hiding: it is the engine's rule, the mask tells the truth about it, and a reader who
    # saw only "1" would think something was broken. So the interesting figure is the
    # first step on which play actually opens, tracked below.
    opens_at, opens_legal = None, 0

    rng = np.random.default_rng(0)
    policy = RandomLegalOpponent(noop_prob=0.7)

    # env.agents empties when the battle ends, so this loop is the whole match. One
    # step is one decision for each player: half a second of game time, ten ticks.
    steps = 0
    while env.agents:
        actions = {a: policy.act(obs[a], obs[a]["action_mask"], rng) for a in env.agents}
        obs, reward, terminated, truncated, info = env.step(actions)
        steps += 1
        if opens_at is None and "blue" in obs:
            legal = int(obs["blue"]["action_mask"].sum())
            if legal > 1:
                opens_at, opens_legal = steps, legal

    s = env.battle_state
    print(f"winner {s.winner}  crowns {[p.crowns for p in s.players]}  tick {s.tick}")
    print(f"{steps} decisions each, {s.tick} ticks of game time")

    total = env.action_space("blue").n
    lockout = env.engine.rules().deploy_lockout_ticks
    print(f"legal moves on the first step: {first_legal} of {total}"
          f"  (the match refuses every deploy for its first {lockout} ticks)")
    print(f"legal moves once play opens, on step {opens_at}: {opens_legal} of {total}")


if __name__ == "__main__":
    main()
