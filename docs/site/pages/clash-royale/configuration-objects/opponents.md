# Opponents

An opponent is a bot that sits in the other seat. Some are simple scripts that come with
RoyaleGym. You can write your own in a few lines, or use a bot you trained.

## In the Trainer

RoyaleLearn picks the opponent by name, with `Learner(..., opponent=...)`:

| `opponent` | Who your bot plays |
|---|---|
| `"random"` | A bot that plays a random legal card now and then. |
| `"noop"` | A bot that never plays a card. |
| `"self"` | About half its battles against itself, a third against older versions of itself, and the rest against some of the scripted bots below. |

RoyaleLearn trains only against these three. The scripted bots below, and opponents you write,
are for testing a bot: with `evaluate`, with `play_battle`, or in `ClashGymEnv` with another
training library.

## The Scripted Bots

Each of these plays one simple idea. They are useful to test a bot against.

| Opponent | What it does |
|---|---|
| `NoopOpponent()` | Never plays a card. |
| `RandomLegalOpponent(noop_prob=0.9)` | Waits most of the time; otherwise plays a random legal card. |
| `FirstAffordableOpponent()` | Plays the first card it can afford, as soon as it can. |
| `DefendOpponent()` | Only plays on its own half, in front of its king tower. |
| `PushOpponent()` | Plays every card as far forward as it can. |
| `PatientOpponent(ready=3)` | Waits until it can afford three cards, then plays forward. |

`ladder()` returns one of each, roughly from simplest to cleverest:

```python
from royalegym import ladder

print([name for name, opponent in ladder()])
```

```
['noop', 'random', 'first-affordable', 'defend', 'push', 'patient']
```

## How They Work

An opponent needs just one method:

```python
# Pick a move for one seat. `obs` is what that seat sees, `mask` its legal moves,
# and `rng` a numpy random generator. Return the move's number (0 waits).
def act(self, obs, mask, rng): ...
```

## Creating Your Own

This one saves up until it has 9 elixir, then plays a random legal card:

```python
import numpy as np


class SaveUpOpponent:
    """Waits until it has 9 elixir, then plays a random legal card."""

    def act(self, obs, mask, rng):
        elixir = obs["vector"][0] * 10  # the first number is its own elixir, out of 10
        legal = np.flatnonzero(mask)
        legal = legal[legal != 0]  # every legal move except waiting
        if elixir < 9 or len(legal) == 0:
            return 0
        return int(rng.choice(legal))
```

## Who Is Better?

`evaluate` plays two bots against each other, half the battles on each side, and says which is
better, or that it is too close to call:

```python
from royalegym import RandomLegalOpponent, evaluate, make_env

result = evaluate(SaveUpOpponent(), RandomLegalOpponent(), make_env, games=20,
                  names=("save-up", "random"))
print(result.summary())
```

```text
save-up vs random: 10-10-0 over 20 games. win rate 50.0% (29.9% to 70.1% at 95%) -- too close to call. seat gap -20.0%, mean 3621 ticks.
```

The range in brackets is where the real win rate probably lies. With only 20 battles it is
wide, so it can't separate the two bots. Play more battles to narrow it.

A bot you trained can play here too. `Learner.load_policy("runs/my_bot")` loads it, and
`CallableOpponent(lambda obs, mask: bot(obs))` turns it into an opponent. Test it with your own
`build_env` (`from quickstart import build_env`) in place of `make_env`, so it plays its own
deck: `make_env()` always deals the starter deck. The [FAQ](../../faq.md#how-do-i-know-how-good-my-bot-is)
has a whole file that tests your bot against every bot above.

## A Single-Seat Environment

`ClashGymEnv` puts an opponent in one seat, so the environment looks like an ordinary
one-player Gymnasium environment. That's handy for other training libraries:

```python
from royalegym import ClashGymEnv, PatientOpponent, RustEngine

env = ClashGymEnv(agent="blue", opponent=PatientOpponent(), engine=RustEngine())
obs, info = env.reset(seed=0)
print(env.action_space)
```

```
Discrete(2305)
```

`env.action_masks()` gives the legal moves in the form Stable-Baselines3's `MaskablePPO` asks
for.
