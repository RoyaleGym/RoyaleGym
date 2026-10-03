# Reward Functions

A reward function decides what the bot is paid for. After every step it gives each seat a
number: more is better. The bot learns to do whatever makes that number big, so the reward is
the main way you tell it what you want.

## The Ones That Come With RoyaleGym

| Reward | Pays for |
|---|---|
| `WinLossReward()` | +1 when the bot wins, -1 when it loses, 0 for a draw (change it with `draw=`). Paid once, at the end. |
| `CrownReward()` | +1 for each crown taken, -1 for each crown lost. |
| `TowerHPReward()` | Damage done to enemy towers minus damage taken, as a share of each tower's health. Taking a whole tower is worth 1. |
| `ElixirTradeReward()` | Elixir the opponent lost (units killed, spells cast) minus elixir the bot lost, divided by 10. |
| `ElixirLeakPenalty()` | -1 for each step the bot sits at 10 elixir, wasting what it would have gained. |
| `IllegalActionPenalty()` | -1 for each move the game refused. Always 0 while the bot follows the mask. |
| `PlacementDepthReward()` | How far forward the bot plays its cards: +1 at the enemy's back edge, -1 at its own. |

`make_env()` uses `TowerHPReward()` on its own. It pays out often during a battle, which helps
a new bot get started.

## Combining Rewards

`CombinedReward` adds several rewards together, each multiplied by a weight:

```python
from royalegym import CombinedReward, CrownReward, TowerHPReward, WinLossReward

reward_fn = CombinedReward([
    (WinLossReward(), 1.0),
    (CrownReward(), 0.2),
    (TowerHPReward(), 0.1),
])
```

Keep winning the biggest part. The smaller rewards are hints that help the bot find its way to
a win. If a hint is worth more than winning, the bot learns to chase the hint instead.

While it trains, each line of `metrics.jsonl` in your `save_dir` has `env/reward_terms/<name>`:
how much each part paid per battle, after its weight. That tells you which part the bot is really
earning from. (In your own code, the last step's `info` of a battle has the same as
`reward_sum/<name>`.)

## How They Work

Every reward function has one method you must write, and two you can:

```python
# Required. Called once per seat after every step. Return that seat's reward.
#   team:    the seat being scored: 0 is Blue, 1 is Red
#   prev:    the battle before the step
#   state:   the battle after the step
#   results: one entry per card played during the step, by either side
def get_reward(self, team, prev, state, results): ...

# Optional. Called once when the environment is made, to read what you need from the engine.
def bind(self, engine): ...

# Optional. Called at the start of every battle.
def reset(self, state): ...
```

## Creating Your Own

Here is a reward that pays only for damage done to enemy towers, and ignores damage taken:

```python
from royalegym import RewardFunction


class TowerDamageDealtReward(RewardFunction):
    """Pays for damage done to the enemy's towers, as a share of their health."""

    def get_reward(self, team, prev, state, results):
        foe = 1 - team
        before = sum(prev.players[foe].tower_hp)
        after = sum(state.players[foe].tower_hp)
        return (before - after) / sum(state.players[foe].tower_max_hp)
```

!!! tip "In your quickstart.py"
    Paste the class above the line `def build_env():`. Then use it in the `reward_fn` lines, for
    example by changing `(TowerHPReward(), 0.1),` to `(TowerDamageDealtReward(), 0.1),`. In
    `ClashParallelEnv` the reward's keyword is `reward_fn=`; in `make_env`, below, it's `reward=`.

Use it like any other reward, on its own or in a `CombinedReward`:

```python
import numpy as np
from royalegym import make_env

env = make_env(reward=TowerDamageDealtReward())
obs, info = env.reset(seed=0)
rng = np.random.default_rng(0)
total = 0.0
while env.agents:
    actions = {a: int(rng.choice(np.flatnonzero(obs[a]["action_mask"]))) for a in env.agents}
    obs, rewards, terminated, truncated, info = env.step(actions)
    total += rewards["blue"]
print(f"Blue earned {total:.2f}")
```

```text
Blue earned 0.36
```

Your number may differ: a battle between two random players goes differently on each version
of the engine.

## What a Reward Can Read

| Field | What it is |
|---|---|
| `state.players[team].crowns` | Crowns taken so far. |
| `state.players[team].elixir_milli` | Elixir, in thousandths: 10 elixir is `10_000`. |
| `state.players[team].tower_hp`, `.tower_max_hp` | Health of the king tower, then the left and right princess towers. 0 means destroyed. |
| `state.entities` | Every unit, building and tower: `team`, `kind`, `card_id`, `x`, `y`, `hp`, `max_hp`. |
| `state.game_over`, `state.winner` | Whether the battle is over, and who won (0 Blue, 1 Red, 2 a draw). |
| `results` | Each card played this step: `team`, `card_id`, `x`, `y`, and `status`, which is 0 if the game accepted it. |

`prev` has the same fields, from one step earlier. Comparing the two is how most rewards work.

More examples are on [Training an Agent](../training-an-agent.md#your-own-rewards).
