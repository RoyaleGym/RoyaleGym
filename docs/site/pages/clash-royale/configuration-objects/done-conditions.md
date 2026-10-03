# Done Conditions

A done condition decides when a battle is over. The environment takes two of them, and the
difference between the two matters to the bot:

- **`termination_cond`**: the battle really ended. Someone won, or it was a draw. Nothing that
  happens afterwards counts.
- **`truncation_cond`**: we stopped the battle early, for example to save time. It wasn't
  decided, so the trainer still counts on what the bot would have earned had it gone on.

Put a condition in the wrong slot and the bot misjudges how good the last moments of a battle
were. So each condition that comes with RoyaleGym says which slot it belongs in, and the
environment refuses one in the wrong slot.

## The Ones That Come With RoyaleGym

| Condition | Slot | Ends the battle when |
|---|---|---|
| `GameOverCondition()` | termination | the game says it's over: a king tower fell, or time ran out. |
| `FirstCrownCondition()` | termination | either side takes a crown. Good for short practice battles. |
| `StepLimitCondition(max_steps)` | truncation | the bot has made `max_steps` decisions. |
| `TickLimitCondition(max_tick)` | truncation | the battle clock reaches `max_tick` ticks (20 ticks a second). |
| `AnyCondition([...])` | either (with RoyaleLearn: termination only) | any of the conditions in the list is met. |
| `AllCondition([...])` | either (with RoyaleLearn: termination only) | all of them are met. |

The defaults are `GameOverCondition()` and no truncation. A Clash Royale battle always ends by
itself, after three minutes plus at most two of overtime, so most bots never need a truncation.

## How They Work

Every done condition has one method you must write, and one you can:

```python
# Required. Called once per step: is the battle over as of `state`?
def is_done(self, state): ...

# Optional. Called at the start of every battle.
def reset(self, state): ...
```

Subclass `TerminationCondition` or `TruncationCondition`, so the environment knows which slot
it belongs in.

## Creating Your Own

This one ends a battle as soon as either side loses a tower, so the bot practises the first
push and the first defence over and over:

```python
from royalegym import TerminationCondition


class TowerDownCondition(TerminationCondition):
    """Ends the battle as soon as either side loses a tower, or when the game is over."""

    def is_done(self, state):
        if state.game_over:
            return True
        return any(hp == 0 for player in state.players for hp in player.tower_hp)
```

Always end the battle when the game is over too, as the first two lines of `is_done` do.
Otherwise a battle in which no tower falls never ends.

A battle you end early has no winner, so `WinLossReward` pays nothing in it. Keep `CrownReward`
and `TowerHPReward` in your reward when you use a condition like this.

```python
from royalegym import make_env

env = make_env(termination_cond=TowerDownCondition())
print(type(env.termination).__name__)
```

```
TowerDownCondition
```

!!! tip "In your quickstart.py"
    Paste the class above the line `def build_env():`, and change the termination line in
    `build_env` to `termination_cond = TowerDownCondition()`.

!!! note "A truncation of your own, with RoyaleLearn"
    RoyaleLearn builds a fresh copy of your truncation condition for every battle it runs.
    Give yours a `config()` method that returns its constructor's arguments as a dict, with one
    entry per argument, so it can. For a condition made with `__init__(self, max_seconds)`:

    ```python
    def config(self):
        return {"max_seconds": self.max_seconds}
    ```

    `StepLimitCondition` and `TickLimitCondition` need nothing extra. `AnyCondition` and
    `AllCondition` can't be rebuilt this way, so with RoyaleLearn use them for termination only.
