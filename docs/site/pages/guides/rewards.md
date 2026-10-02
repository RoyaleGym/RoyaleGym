# Rewards

The reward is how you tell the bot what you want. After every step, each side gets a number.
Bigger is better. The quickstart uses tower damage. This page shows how to write your own, how to
mix several, and how to see which part paid.

## Write one

Subclass `RewardFunction` and fill in `get_reward`. It gets the seat it scores (0 is Blue, 1 is
Red), the board before the step and the board after it. This one pays for damage to the enemy
towers and charges for damage to your own.

```python
import numpy as np
from royalegym import RewardFunction, make_env


class TowerDamage(RewardFunction):
    """Pay for damage dealt to the enemy towers, minus damage taken."""

    def get_reward(self, team, prev, state, results):
        def standing(s, t):
            p = s.players[t]
            return sum(hp / mx for hp, mx in zip(p.tower_hp, p.tower_max_hp))

        dealt = standing(prev, 1 - team) - standing(state, 1 - team)
        taken = standing(prev, team) - standing(state, team)
        return dealt - taken


def play(reward):
    """One battle between two random players. Prints Blue's total reward."""
    env = make_env(reward=reward)
    obs, info = env.reset(seed=0)
    rng, total = np.random.default_rng(0), 0.0
    while env.agents:
        moves = {a: int(rng.choice(np.flatnonzero(obs[a]["action_mask"]))) for a in env.agents}
        obs, rewards, terminated, truncated, info = env.step(moves)
        total += rewards["blue"]
    print(f"Blue's total: {total:+.3f}")
    return env


play(TowerDamage())
```

```
Blue's total: -1.164
```

Blue lost more tower health than it took, so its total is below zero. Each tower counts as a
fraction of its full health, so a king tower is not worth more just because it has more hitpoints.

To train with it, pass it to `make_env(reward=...)` in the quickstart. That is the one line to
change.

## Combine them

`CombinedReward` adds several rewards, each with a weight. Winning is the real goal. The small
terms give the bot something to go on before the battle ends.

```python
from royalegym import CombinedReward, CrownReward, WinLossReward

shaped = CombinedReward([
    (WinLossReward(), 1.0),
    (CrownReward(), 0.2),
    (TowerDamage(), 0.1),
])
play(shaped)
print("Blue's last step, term by term:", shaped.terms_for(0))
```

```
Blue's total: -1.316
Blue's last step, term by term: {'WinLossReward': -1.0, 'CrownReward': 0.0, 'TowerDamage': 0.0}
```

## See which part paid

`terms_for(seat)` breaks the last reward into its terms. Here the last step paid -1 for the loss
and nothing else. Log it while you train. When a bot does something odd, the breakdown usually
shows which term is paying for it.

## What comes in the box

| reward | pays for |
|---|---|
| `WinLossReward` | +1 for a win, -1 for a loss |
| `CrownReward` | crowns taken minus crowns lost |
| `TowerHPReward` | tower health taken minus lost, as fractions of full |
| `ElixirTradeReward` | elixir the enemy lost in units and spells, minus your own |
| `ElixirLeakPenalty` | sitting at full elixir |
| `IllegalActionPenalty` | a move the engine refused; with the mask it stays zero |

`default_reward()` mixes the first four.

These outputs were run on 2026-10-01 on engine build `a581356680589e2c`. Battles change with the
engine version, so your totals may differ.

Next: [Observations and Actions](observations-and-actions.md), what the bot sees and the moves it
can make.
