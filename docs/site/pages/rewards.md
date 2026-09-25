# Writing a reward function

<p align="center">
  <img alt="Interface" src="https://img.shields.io/badge/interface-one%20method-0b7285?style=flat-square">
  <img alt="Terms" src="https://img.shields.io/badge/shipped%20terms-7-555?style=flat-square">
  <img alt="Recompile" src="https://img.shields.io/badge/recompile%20needed-no-2ea043?style=flat-square">
  <img alt="Blocks" src="https://img.shields.io/badge/code%20blocks%20on%20this%20page-all%20run-2ea043?style=flat-square">
  <img alt="Training loop: runs, no finished bot yet" src="https://img.shields.io/badge/training%20loop-runs%3B%20no%20finished%20bot%20yet-orange?style=flat-square">
</p>

The reward function is the one piece you are expected to write. It says what your bot
should want. Everything else in the box already has a working version.

It is one Python class with one method. No Rust, no rebuild, no restart. You edit the
file and run again.

This page climbs a ladder, from a reward function that does nothing to one you would
actually train with. Every program on it was run on a real battle and every output
below it is the output that came back.

<div class="grid cards" markdown>

- __1. The empty one__

    ---

    Returns 0.0 on every step. Proves the interface is three lines long.

- __2. Crowns__

    ---

    What you actually want. It paid out on 2 steps out of 480, and they cancelled.

- __3. Tower damage__

    ---

    The same goal, measured more often. It paid out on 56 steps out of 480.

- __4. Both together__

    ---

    A weighted sum, with a breakdown you can print.

</div>

!!! warning "The numbers on this page describe an older battle"
    Every program here plays the same battle, between two players choosing at random. The
    engine has changed since these outputs were recorded. They come from a battle that went to
    overtime, 480 steps, with one tower falling each way, and the prose below describes that
    battle. The same program now ends differently: in RoyaleGym's README, re-run on 2026-09-24,
    Red takes one of Blue's princess towers and wins at tick 3600, which is 360 steps with no
    overtime. So the step counts and totals below are out of date and due a re-run.

## The interface

This is the real base class, from `royalegym/reward.py`:

```python
class RewardFunction(ABC):
    def bind(self, engine) -> None: ...          # optional: you get the card catalogue
    def reset(self, state) -> None: ...          # optional: called at the start of a battle
    def config(self) -> dict: ...                # optional: what goes in the checkpoint

    @abstractmethod
    def get_reward(self, team, prev, state, results) -> float: ...
```

Only `get_reward` is required. Its four arguments:

| argument | what it is |
|---|---|
| `team` | which seat you are scoring. 0 is Blue, 1 is Red. The env calls this once per seat on the same step |
| `prev` | the board before this step |
| `state` | the board after it |
| `results` | what the engine did with the two cards that were played this step, if any |

`prev` and `state` are both a `BattleState`. The fields you will use most:

| field | what it holds |
|---|---|
| `state.players[team].crowns` | crowns taken, 0 to 3 |
| `state.players[team].tower_hp` | hitpoints left, as `[king, left, right]` |
| `state.players[team].tower_max_hp` | the same three towers at full health |
| `state.players[team].elixir_milli` | elixir in thousandths, so 5500 is five and a half |
| `state.entities` | everything alive on the board right now |
| `state.game_over`, `state.winner` | whether the battle has ended, and who won |

You return one number per step. Bigger is better for that seat.

## Rung 1: a reward function that does nothing

Start here, because it proves the plumbing works before you have any opinion about the
game. This is the whole program. The helper `play` at the top is reused by every later
rung on this page, so keep the file.

```python
import numpy as np
from royalegym import (ClashParallelEnv, DefaultStateMutator, RandomLegalOpponent,
                       RewardFunction, RustEngine)

engine = RustEngine()
by_name = {c.name: c.card_id for c in engine.cards()}
deck = [by_name[n] for n in ("Knight", "Archer", "Giant", "Minions",
                             "Fireball", "Cannon", "Zap", "Musketeer")]


def play(reward_fn):
    """Play one whole battle with this reward function and report Blue's side of it."""
    env = ClashParallelEnv(engine=engine,
                           state_mutator=DefaultStateMutator(decks=[deck, deck]),
                           reward_fn=reward_fn)
    obs, info = env.reset(seed=0)
    rng, policy = np.random.default_rng(0), RandomLegalOpponent(noop_prob=0.7)
    total, steps, paid = 0.0, 0, 0
    while env.agents:
        actions = {a: policy.act(obs[a], obs[a]["action_mask"], rng) for a in env.agents}
        obs, reward, terminated, truncated, info = env.step(actions)
        total += reward["blue"]
        steps += 1
        paid += reward["blue"] != 0.0
    print(f"{type(reward_fn).__name__}: blue total {total:+.3f}"
          f"  over {steps} steps, {paid} of them non-zero")


class ZeroReward(RewardFunction):
    def get_reward(self, team, prev, state, results):
        return 0.0


play(ZeroReward())
```

```
ZeroReward: blue total +0.000  over 480 steps, 0 of them non-zero
```

Three lines of your own code and it ran a whole match. 480 steps is one decision every half
second, for three minutes of game time plus the sixty seconds of overtime this battle went to.

Name the eight cards, as the program does. A card id is only a position in the
catalogue and positions move between card tables, so the same number is not the same
card on every machine.

## Rung 2: reward crowns

Crowns are what the game scores, so this is close to what you actually want. Add this
to the end of the file from rung 1.

```python
class CrownReward(RewardFunction):
    def get_reward(self, team, prev, state, results):
        mine = state.players[team].crowns - prev.players[team].crowns
        theirs = state.players[1 - team].crowns - prev.players[1 - team].crowns
        return float(mine - theirs)


play(CrownReward())
```

```
CrownReward: blue total +0.000  over 480 steps, 2 of them non-zero
```

That is the shipped `CrownReward`, near enough line for line, and this battle makes its
weakness unusually visible. Two towers fell, one each way, so exactly two of 480 decisions
got a number that was not zero. The other 478 told the bot nothing at all. And because the
two payments were +1 and -1, the total for the whole match is zero. A bot reading only this
signal cannot tell this battle apart from one in which nothing happened.

That is the problem with scoring only the thing you care about. It is correct and it is
almost silent.

## Rung 3: reward damage to the enemy towers

A tower does not fall in one hit. It loses hitpoints all the way down, and that is
something you can pay out on every step. Add this next.

```python
def tower_fraction(state, team):
    """How much of this player's tower hitpoints is still standing, 0 to 3."""
    p = state.players[team]
    return sum(hp / max(1, mx) for hp, mx in zip(p.tower_hp, p.tower_max_hp))


class TowerDamageReward(RewardFunction):
    def get_reward(self, team, prev, state, results):
        dealt = tower_fraction(prev, 1 - team) - tower_fraction(state, 1 - team)
        taken = tower_fraction(prev, team) - tower_fraction(state, team)
        return dealt - taken


play(TowerDamageReward())
```

```
TowerDamageReward: blue total -0.360  over 480 steps, 56 of them non-zero
```

Same battle, same bots, and now 56 steps carry a number instead of 2. That is what
people mean when they call a reward *dense*. The bot gets told it is getting warmer
long before anything falls over.

Dividing by `tower_max_hp` matters. Without it a king tower, which has far more
hitpoints than a princess tower, would quietly be worth several times as much as it
should be, and the number would also change whenever your card table changes.

## Rung 4: combine them

`CombinedReward` takes a list of `(term, weight)` pairs and adds them up. Add this.

```python
from royalegym import CombinedReward, WinLossReward

shaped = CombinedReward([
    (WinLossReward(), 1.0),
    (CrownReward(), 0.2),
    (TowerDamageReward(), 0.1),
])
play(shaped)
print("blue's last step, term by term:", shaped.terms_for(0))
```

```
CombinedReward: blue total -1.036  over 480 steps, 57 of them non-zero
blue's last step, term by term: {'WinLossReward': -1.0, 'CrownReward': 0.0, 'TowerDamageReward': 0.0}
```

`WinLossReward` pays +1 the moment the battle is won, -1 when it is lost, and nothing
before that. Blue loses this one, which is why the last step reads -1.0 and why every total
on this page from here down is negative. That is the actual objective. The other two terms exist to give the bot something to go on in
the meantime, which is why their weights are small.

`terms_for(seat)` gives you the breakdown of that seat's last reward. Log it. When a
bot starts doing something strange, the breakdown usually says which term is paying for
it. Ask for the seat you mean: 0 for Blue, 1 for Red.

## What ships in the box

You do not have to write any of this. `default_reward()` is what you get if you pass no
reward function at all.

| term | what it pays for | weight in `default_reward()` |
|---|---|---|
| `WinLossReward` | +1 for winning, -1 for losing. The real objective | 1.0 |
| `CrownReward` | crowns taken minus crowns conceded | 0.2 |
| `TowerHPReward` | tower hitpoints taken minus lost, as fractions of full | 0.1 |
| `ElixirTradeReward` | elixir value of enemy units killed minus your own lost | 0.02 |
| `ElixirLeakPenalty` | sitting at full elixir, which wastes the regeneration | not in the default |
| `PlacementDepthReward` | how far up the board your cards land, from -1 at your back line to +1 at the far end | not in the default, on purpose |
| `IllegalActionPenalty` | a move the engine refused. Should always be zero | not in the default |

Running the shipped default on the same battle:

```python
import json
from royalegym import default_reward

play(default_reward())
print(json.dumps(default_reward().config(), indent=2)[:400])
```

```
CombinedReward: blue total -1.048  over 480 steps, 132 of them non-zero
{
  "terms": [
    {
      "class": "WinLossReward",
      "weight": 1.0,
      "params": {
        "draw": 0.0
      }
    },
    {
      "class": "CrownReward",
      "weight": 0.2,
      "params": {}
    },
    {
      "class": "TowerHPReward",
      "weight": 0.1,
      "params": {
        "king_weight": 1.0,
        "princess_weight": 1.0
      }
    },
    {
      "class": "ElixirTradeReward
```

132 non-zero steps out of 480, because `ElixirTradeReward` pays out every time anything
dies. The JSON is cut off at 400 characters by the `[:400]` in the program, which is why
the last line stops mid word. It is the whole recipe, and it is how the reward ends up
written into a checkpoint. `ClashParallelEnv.config()` carries the same thing under
`reward_fn`, so a saved run always says what it was being paid for.

`IllegalActionPenalty` is worth knowing about even though it is not switched on. With a
correct legality mask it is always exactly zero, so if you ever see it go negative in a
log, something is wrong with your mask or your policy is ignoring it.

## Keep it zero sum

One bot plays both seats in self play. If a reward can pay both seats at once, the two
of them can learn to cooperate rather than compete, and you get a very peaceful bot
that loses to everything else.

The check is one line: play a battle and add the seats up.

```python
def both_seats(reward_fn):
    env = ClashParallelEnv(engine=engine,
                           state_mutator=DefaultStateMutator(decks=[deck, deck]),
                           reward_fn=reward_fn)
    obs, info = env.reset(seed=0)
    rng, policy = np.random.default_rng(0), RandomLegalOpponent(noop_prob=0.7)
    totals = {"blue": 0.0, "red": 0.0}
    while env.agents:
        actions = {a: policy.act(obs[a], obs[a]["action_mask"], rng) for a in env.agents}
        obs, reward, terminated, truncated, info = env.step(actions)
        for seat in totals:
            totals[seat] += reward[seat]
    print(f"{type(reward_fn).__name__}: blue {totals['blue']:+.3f}  red {totals['red']:+.3f}"
          f"  sum {totals['blue'] + totals['red']:+.3f}")


both_seats(CrownReward())
both_seats(TowerDamageReward())
```

```
CrownReward: blue +0.000  red +0.000  sum +0.000
TowerDamageReward: blue -0.360  red +0.360  sum +0.000
```

Every shipped term is meant to add to zero like this, except three that score only one
player's own play. `ElixirLeakPenalty` is one on purpose: both players really can waste elixir
at the same time. `PlacementDepthReward` scores each seat's own placements. `IllegalActionPenalty`
should be zero for both seats anyway.

The suite does not check all of them. One term, `TowerHPReward`, has its breakdown asserted
antisymmetric across the seats in `tests/test_train_harness.py`. For the rest it is a design
intent, and the program above is how you confirm it for a term of your own.

## What not to do

This is the part that saves you weeks.

`CrownReward` and `TowerHPReward` each measure one quantity before the step and again
after, and pay the difference. Add those differences up across a whole battle and everything in
the middle cancels out. What is left is the score at the end minus the score at the
start. So the shaping makes the signal louder without changing which way of playing is
best. It hurries the bot along the same road.

A term you invent that is not a before-and-after difference of one quantity does not
have that property. It can pay your bot for something that is not winning, and your bot
will happily take the payment. A bot that farms elixir trades and never pushes is the
classic result. The shipped `PlacementDepthReward` is a term like that. It pays for where you
play, which decides in advance whether pushing or defending is better. That is for your bot to
learn, so the term ships as an example to copy and stays out of the default.

`ElixirTradeReward`, the third shaping term in the default, does not fit the rule either. It
pays when units die, not for the change in one quantity. A player who never plays a card still
collects when enemy units die at its towers, so it can pay for sitting back. Keep its weight
small, as the default does (0.02). RoyaleLearn's own default reward swaps it for a term that
does fit.

Then the rule, and it is from the person who designed the training harness:

!!! tip "Add a term. Do not nudge a weight."
    If your bot hoards elixir and you turn the elixir penalty up, then down, then up
    again as other behaviours shift, that weight is standing in for a reward term you
    have not written yet. Find the thing you actually object to and give it its own
    term. Weights should settle and stay put.

Two more habits worth having:

- Keep `WinLossReward` in the mix and keep it the biggest weight. It is the only term
  that is the real objective. Everything else is a hint.
- Normalise. Divide hitpoints by full hitpoints, as rung 3 does. A raw hitpoint number
  is in the thousands, it swamps everything next to it, and it changes when your card
  table changes.

## What this page cannot show you yet

Nobody can show you yet that a reward function *trains* anything. The loop is not what is
missing. RoyaleLearn closed its training loop on 2026-09-22, and `python -m royalelearn train`
runs end to end with the torch extra installed. Real training runs started that day. None has
run long enough to produce a finished bot.

So what you have is a reward function that runs on a real battle, on both seats, and gives
back numbers you can look at, and a trainer that consumes it. If you write a reward function
and train on it until a bot comes out, you will be among the first to learn whether any of
this works, and the project would very much like to hear what happened.

A custom reward is named in the run's config rather than pasted into the trainer, so the
checkpoint records it:

!!! warning "UNVERIFIED"
    This particular snippet has not been run here. The command line and the training loop both
    exist and work; `examples/custom_reward.py` in RoyaleLearn is the maintained version of
    this, and is the one to copy.

```
python -m royalelearn train --config examples/configs/laptop.json
```

Before you spend hours training, judge a reward function the way this page does. Play a
battle with it. Count how many steps it actually said something on. Check it adds to zero
across the seats. Print the breakdown.

## Read next

- [What the bot sees and does](observations-and-actions.md), for the other half of the
  contract.
- [The environments](pieces/environments.md), for the five swappable pieces this one
  belongs to.
- [`royalegym/reward.py`](https://github.com/RoyaleGym/RoyaleGym/blob/main/royalegym/reward.py)
  for all seven shipped terms. They are short, and they are the best examples there are.
- The [Discord](https://discord.gg/4D2BS5JBHP) if your reward function does something
  odd. Bring the term breakdown.
