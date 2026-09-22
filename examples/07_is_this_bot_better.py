"""Compare two bots and get an answer with an error bar, or an honest "cannot tell".

The question you ask after every change. It is also the question a hand-written loop
answers wrongly: it plays your bot in one seat, prints a bare percentage, and counts a
battle the step limit cut short as a draw.

    python examples/07_is_this_bot_better.py
"""

import numpy as np

from royalegym import (
    ClashParallelEnv,
    DefaultStateMutator,
    NoopOpponent,
    RandomLegalOpponent,
    RustEngine,
    evaluate,
)
from royalegym.done_condition import GameOverCondition, StepLimitCondition


class PlayEverything:
    """Dump whatever you can afford, wherever it is legal, as fast as possible."""

    def act(self, obs, mask, rng):
        del obs
        legal = np.flatnonzero(mask)
        legal = legal[legal != 0]  # index 0 is the no-op
        return int(rng.choice(legal)) if legal.size else 0


def build_env() -> ClashParallelEnv:
    return ClashParallelEnv(
        engine=RustEngine(),
        state_mutator=DefaultStateMutator(),
        termination_cond=GameOverCondition(),
        truncation_cond=StepLimitCondition(700),
    )


def main() -> None:
    matchups = (
        ("greedy", PlayEverything(), "noop", NoopOpponent()),
        ("greedy", PlayEverything(), "random", RandomLegalOpponent(0.7)),
    )
    for name_a, a, name_b, b in matchups:
        result = evaluate(a, b, build_env, games=20, seed=0, names=(name_a, name_b))
        print(result.summary())
        for seat in result.by_seat:
            print(
                f"    as {seat.seat:4} {seat.wins}-{seat.losses}-{seat.draws}"
                + (f", {seat.undecided} unfinished" if seat.undecided else "")
            )

    print()
    print("Two things to read off the summary.")
    print("  'too close to call' means the interval covers 50%. It is not a tie; it is")
    print("  not enough games. Raise games= and ask again.")
    print("  A large seat gap means part of what you measured was the colour, not skill.")


if __name__ == "__main__":
    main()
