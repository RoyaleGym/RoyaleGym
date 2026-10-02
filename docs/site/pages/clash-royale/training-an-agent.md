# Training an Agent

This guide builds on the [Quick Start Guide](../quickstart.md). We'll write our own reward
functions, give the trainer more to work with, let the bot play against itself, and watch it
play. Along the way it explains what the trainer is doing and what its numbers mean.

## A Better Agent

If you followed [Install](../install.md), you have everything this page uses, the viewer
included.

### Your own rewards

A reward function looks at the battle after each step and pays the bot for what it did. Here
are three small ones to start from. It's tidiest to put them in a file of their own, say
`rewards.py`, and import them into your training script, but they can live anywhere.

```python
from royalegym import EntityKind, RewardFunction, to_own


class TroopsInEnemyHalfReward(RewardFunction):
    """Pays a little for each of the bot's troops on the enemy's half of the arena."""

    def bind(self, engine):
        self.arena = engine.arena()  # the arena's size, for the halfway line

    def get_reward(self, team, prev, state, results):
        count = 0
        for unit in state.entities:
            if unit.team == team and unit.kind == EntityKind.TROOP:
                x, y = to_own(self.arena, team, unit.x, unit.y)  # as this seat sees it
                if y > self.arena.height // 2:
                    count += 1
        return count / 10


class FullElixirPenalty(RewardFunction):
    """Charges the bot for every step it sits at 10 elixir, wasting what it would gain."""

    def get_reward(self, team, prev, state, results):
        return -1.0 if state.players[team].elixir_milli >= 10_000 else 0.0


class CardsPlayedReward(RewardFunction):
    """Pays for every card the bot played this step."""

    def get_reward(self, team, prev, state, results):
        return sum(1.0 for r in results if r.team == team and r.status == 0)
```

Each one gets the seat it is scoring (`team`: 0 is Blue, 1 is Red), the battle before the step
(`prev`), the battle after it (`state`), and the result of every card played during the step
(`results`). [Reward Functions](configuration-objects/reward-functions.md) explains every
field you can read.

### The environment

Now set up the battle with these rewards mixed in. `CombinedReward` adds several rewards
together, each with a weight. Keep the win itself the biggest part: the others are hints that
help the bot find its way to a win, and if a hint pays more than winning, the bot will chase
the hint instead.

```py
def build_env():
    from royalegym import ClashParallelEnv, RustEngine
    from royalegym import DefaultStateMutator, SpatialObsBuilder, TileActionParser
    from royalegym import CombinedReward, CrownReward, TowerHPReward, WinLossReward
    from royalegym import GameOverCondition
    from rewards import CardsPlayedReward, FullElixirPenalty, TroopsInEnemyHalfReward

    deck = ["Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Cannon", "Musketeer"]

    reward_fn = CombinedReward([
        (WinLossReward(), 1.0),
        (CrownReward(), 0.2),
        (TowerHPReward(), 0.1),
        (TroopsInEnemyHalfReward(), 0.002),
        (FullElixirPenalty(), 0.002),
        (CardsPlayedReward(), 0.002),
    ])

    return ClashParallelEnv(
        RustEngine(),
        state_mutator=DefaultStateMutator(decks=[deck, deck]),
        obs_builder=SpatialObsBuilder(),
        action_parser=TileActionParser(),
        reward_fn=reward_fn,
        termination_cond=GameOverCondition(),
        truncation_cond=None,
        decision_ms=500,
    )
```

### The trainer

Then the trainer. It's the quickstart's, except that the bot plays against copies of itself, and
it streams a battle to the viewer:

```py
if __name__ == "__main__":
    from royalelearn import Learner

    learner = Learner(
        build_env,
        n_envs=32,                    # battles played at once
        opponent="self",              # play against copies of itself, old and new
        device="auto",                # your graphics card
        trunk_channels=64,            # the network's width
        trunk_blocks=4,               # the network's depth
        steps_per_update=16_384,      # decisions collected before each update
        ppo_batch_size=16_384,        # decisions per update; set this equal to steps_per_update
        ppo_minibatch_size=2_048,     # decisions per gradient step; as many as your GPU's memory holds
        ppo_epochs=2,                 # passes over each batch
        policy_lr=2e-4,               # policy learning rate
        critic_lr=2e-4,               # learning rate of the part that predicts rewards
        ppo_ent_coef=0.01,            # how much it is pushed to keep exploring
        checkpoint_every=200_000,     # save every 200,000 decisions (Ctrl+C saves too)
        timestep_limit=1_000_000_000, # stop after a billion decisions
        save_dir="runs/my_bot",
        viser=True,                   # stream one battle to the viewer
        log_to_wandb=False,           # True to chart the run on Weights & Biases
    )
    learner.learn()
    learner.save("runs/my_bot/bot")   # the finished bot, for watching it later
```

Every setting has a default, so you only need the ones you want to change. The
[RoyaleLearn](../resources/royalelearn.md) page lists them all.

## Understanding the Training Process

Training goes in cycles, and each cycle has two halves.

- **Collecting experience.** The bot plays `n_envs` battles at once. On every step it sees the
  arena, picks a move, and gets a reward. The trainer keeps going until it has
  `steps_per_update` decisions saved.
- **Learning.** The trainer goes over those decisions and nudges the bot's neural network:
  moves that led to more reward than it expected become more likely, and moves that led to less
  become less likely. That's PPO.

Then the cycle starts again with the improved bot. After each cycle the trainer prints a line:

```text
update    12  steps     393,216  battles    N  crowns  +N.NN a battle     N s
```

| Column | What it means |
|---|---|
| `update` | How many cycles have finished. |
| `steps` | Decisions made so far, across every battle. |
| `battles` | Battles that finished during this cycle. |
| `crowns` | Crowns taken minus crowns lost, per battle. Above zero, the bot is winning more than losing. |
| `s` | Seconds the cycle took. |

Against itself (`opponent="self"`), `crowns` stays near zero even while the bot improves,
because both sides improve together. To see real progress, watch it play, or measure it
against a fixed opponent (see [Opponents](configuration-objects/opponents.md)). After a few
million steps, `metrics.jsonl` in your `save_dir` also gets a rating,
`ladder/rating_above_v0`: how much stronger the bot is than its first saved version.

If training is slow, look at `n_envs` first. The battles run on your CPU and the learning on
your graphics card. More battles at once keeps the graphics card busier, until your CPU is full.

## Self-play

`opponent` decides who the bot plays:

| `opponent` | Who sits in the other seat |
|---|---|
| `"random"` | A bot that plays a random legal card now and then. Easy to beat, which is good for a first run. |
| `"noop"` | A bot that never plays a card. Useful to check that your bot learns to attack at all. |
| `"self"` | About half its battles against itself, a third against older versions of itself, and the rest against simple scripted bots. |

Against `"random"`, a bot can get good at beating a bad player and stop there. Against
itself, it always has an opponent at its own level, which is how game-playing bots usually
get strong.

## Custom Decks

Change the deck by changing the eight names. Give each side its own deck by passing two lists,
Blue's first:

```python
from royalegym import DefaultStateMutator

blue = ["Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Cannon", "Musketeer"]
red = ["HogRider", "Valkyrie", "Skeletons", "IceSpirits", "Fireball", "Log", "Cannon", "Musketeer"]
state_mutator = DefaultStateMutator(decks=[blue, red])
```

`DefaultStateMutator()` with no decks deals both sides eight random cards every battle. Card
names are the engine's, without spaces: `HogRider`, `MiniPekka`, `IceSpirits`. To list them
all:

```python
from royalegym import RustEngine

names = [card.name for card in RustEngine().cards()]
print(names[:5])
```

```
['Knight', 'Archer', 'Goblins', 'Giant', 'Pekka']
```

Evolutions and heroes are covered in [State Mutators](configuration-objects/state-mutators.md).

## Watching Your Bot

With `viser=True`, the trainer streams one of its battles while it trains. Open the viewer in a
second terminal:

```
royaleviser
```

It attaches to the running training and shows the battle as it happens. Space pauses, `h`
lists every key, `q` quits.

To watch a bot after training, load it from its `save_dir`, play a battle with it, and open the file:

```py
from royalegym import play_battle
from royalelearn import Learner

bot = Learner.load_policy("runs/my_bot")
battle = play_battle(build_env(), blue=bot, red="random", save_to="my_bot_battle.msgpack")
print(f"Crowns {battle.crowns[0]}-{battle.crowns[1]}")
```

```
royaleviser my_bot_battle.msgpack
```

`Learner.load_policy` loads the newest save in the folder, even from a run you stopped with
Ctrl+C. [Viewer](configuration-objects/viewer.md) has more.

## Monitoring Progress

RoyaleLearn can chart a run on [Weights & Biases](https://wandb.ai) (wandb). Make an account,
run `pip install wandb` and `wandb login`, then set `log_to_wandb=True` in the `Learner`. Its
web page shows the reward, the crowns, how fast the bot is learning, and much more, as graphs
that update while it trains.

Every run also writes its numbers to `metrics.jsonl` in its `save_dir`, one line per update,
whether or not you use wandb.
