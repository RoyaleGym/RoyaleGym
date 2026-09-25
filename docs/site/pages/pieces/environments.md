# The environments

[![repo](https://img.shields.io/badge/repo-RoyaleGym-3776AB?style=flat-square&logo=python&logoColor=white)](https://github.com/RoyaleGym/RoyaleGym)
![APIs](https://img.shields.io/badge/APIs-Gymnasium%20%2B%20PettingZoo-0b7285?style=flat-square)
![action space](https://img.shields.io/badge/action%20space-2305%20moves-555?style=flat-square)
![pieces](https://img.shields.io/badge/swappable%20pieces-5-8957e5?style=flat-square)
![speed](https://img.shields.io/badge/env%20steps%2Fs-957-2ea043?style=flat-square)

**This is where you live.** Everything you will actually write when you make a bot, you write
against RoyaleGym. It takes [the engine](engine.md) underneath and turns a battle into the shape
a learning library expects: you get an observation, you pick a move, you get a reward.

It speaks the two standard Python interfaces for this, Gymnasium and PettingZoo, so the code
looks like every tutorial you have already read.

![A whole battle between two players picking at random from their legal moves, played back in the RoyaleViser window](../media/whole-battle.gif)

That is a whole battle between two players picking at random from the moves that are legal. It
is the program further down this page, watched in [the viewer](viewer.md).

## The five pieces

A match is made of five decisions. Each one is a small Python class, and each one already has a
version that works, so you can ignore four of them and still have a bot.

| The decision | The class you subclass | What ships in the box |
|---|---|---|
| What your bot sees | `ObsBuilder` | `SpatialObsBuilder`: a picture of the board plus a list of numbers. Also `EntityListObsBuilder` |
| What its moves mean | `ActionParser` | `TileActionParser`: 2305 moves, one per card-and-tile pair, plus waiting. Also `HalfTileActionParser` at finer resolution |
| What it is rewarded for | `RewardFunction` | `default_reward()`: winning 1.0, crowns 0.2, tower damage 0.1, elixir trades 0.02, added up |
| How a match starts | `StateMutator` | `DefaultStateMutator`: a fresh battle. Also mid-game, a board you set up by hand, a saved snapshot, or a weighted mix of those |
| When the match ends | `DoneCondition` | `GameOverCondition` for the real ending, `StepLimitCondition` and `TickLimitCondition` for cutting an episode short |

Most people change exactly one of these: the reward. That is
[Writing a reward function](../rewards.md), and it is a few lines.

The last row has a wrinkle worth knowing before it bites you. There are two ways an episode can
stop, and they are not the same thing. **Terminated** means the match is genuinely over and the
next state is worth nothing. **Truncated** means you cut it short and the next state still
mattered. Mixing them up quietly ruins every value estimate near the cut, so the shipped
conditions declare which one they are, and the environment refuses one in the wrong slot.

!!! tip "These names come from RLGym"
    If you have used RLGym v2 for Rocket League bots, the vocabulary here is deliberately the
    same one. `StateMutator`, `TerminationCondition`, `TruncationCondition`. The older names
    `StateSetter` and `TerminalCondition` still import, so old code keeps working.

## Two APIs, and which one you want

Both drive the same battle. The difference is how many seats you are filling.

=== "Both seats are bots"

    Use **`ClashParallelEnv`**, the PettingZoo parallel API. You hand it a move for each
    player and it hands you back an observation, a reward and a done flag for each player.

    This is what you want for self-play, where one bot learns from both sides of every match.
    `ClashSelfPlayVecEnv` runs N battles at once as 2N player slots, so a single batch carries
    both sides of all of them.

=== "One seat, and the env plays the other"

    Use **`ClashGymEnv`**, the plain Gymnasium API. You are one player. The other seat is filled
    by an `Opponent` you pass in, and you never see its moves.

    This is what you want when you are starting out, when you want a fixed opponent to measure
    against, or when you want to hand the environment straight to a library that only speaks
    single-agent Gymnasium.

There is a ready-made opponent in the box either way. `RandomLegalOpponent` picks at random from
the moves that are legal right now, and `NoopOpponent` never plays anything. So you have
something to train against from the first minute.

## A whole battle, both seats

This is the complete program. Nothing but `royalegym` and numpy.

```python
import numpy as np
from royalegym import (ClashParallelEnv, DefaultStateMutator, RandomLegalOpponent, RustEngine)

engine = RustEngine()
by_name = {c.name: c.card_id for c in engine.cards()}
deck = [by_name[n] for n in ("Knight", "Archer", "Giant", "Minions",
                             "Fireball", "Cannon", "Zap", "Musketeer")]

env = ClashParallelEnv(engine=engine,
                       state_mutator=DefaultStateMutator(decks=[deck, deck]))
obs, info = env.reset(seed=0)
rng, policy = np.random.default_rng(0), RandomLegalOpponent(noop_prob=0.7)

while env.agents:
    actions = {a: policy.act(obs[a], obs[a]["action_mask"], rng) for a in env.agents}
    obs, reward, terminated, truncated, info = env.step(actions)

s = env.battle_state
print(f"winner {s.winner}  crowns {[p.crowns for p in s.players]}  tick {s.tick}")
```

```
winner 1  crowns [0, 1]  tick 3600
```

Blue is player 0 and Red is player 1, and Red took one of Blue's princess towers. Tick 3600 is the full three minutes,
so this one finished in regulation on crowns and never reached overtime (re-run 2026-09-24 on engine build `565def31a74817fe` with the 15.535 card table). Had
it finished level, it would have gone to overtime and then to a tiebreak, where the side whose
weakest standing tower has less health left loses.

One env step is half a second of game time, which is 10 ticks. That battle was 360 steps, so
each player made 360 decisions. It takes well under a second of real time.

!!! warning "Name the deck, and name it card by card"
    Notice the deck is looked up by name and not by number. A card id is only a position in the
    catalogue, and the positions move between card tables, so the same number is not the same
    card on two machines.

    If you leave the deck out entirely, each side is dealt eight random cards from whatever
    catalogue your machine built. The same seed then gives you a different battle from the one
    above. Naming eight cards makes the battle a function of the program.

## One seat, and the env plays the other

Same battle, Gymnasium shape. You are Blue. Red is the random-legal opponent.

```python
import numpy as np
from royalegym import (ClashGymEnv, DefaultStateMutator, RandomLegalOpponent, RustEngine)

engine = RustEngine()
by_name = {c.name: c.card_id for c in engine.cards()}
deck = [by_name[n] for n in ("Knight", "Archer", "Giant", "Minions",
                             "Fireball", "Cannon", "Zap", "Musketeer")]

env = ClashGymEnv(agent="blue",                                # you are Blue
                  opponent=RandomLegalOpponent(noop_prob=0.7),  # Red is scripted
                  engine=engine,
                  state_mutator=DefaultStateMutator(decks=[deck, deck]))
obs, info = env.reset(seed=0)
print("legal moves on the first step:", int(obs["action_mask"].sum()), "of 2305")

rng, me = np.random.default_rng(0), RandomLegalOpponent(noop_prob=0.7)
total, steps, done = 0.0, 0, False
while not done:
    obs, reward, terminated, truncated, info = env.step(me.act(obs, obs["action_mask"], rng))
    total += reward
    steps += 1
    done = terminated or truncated

print(f"steps {steps}  reward {total:.3f}  terminated {terminated}")
```

```
legal moves on the first step: 1267 of 2305
steps 333  reward -1.417  terminated True
```

Blue lost. 333 steps is about 3,330 ticks, so the match was over before the three minutes were
up, and `terminated True` says it ended for real rather than being cut short. The reward is negative
because `default_reward()` pays 1.0 for a win and charges 1.0 for a loss, with the crown and
tower terms on top.

Swap `me.act(...)` for your own policy and that loop is a training loop with the learning taken
out. Run it twice and you get the same numbers, because the seed fixes everything.

The exact numbers depend on the deck and on the card table your machine built, so treat them as
"this ran", not as constants.

## The legality mask, which is the part people like

Your bot is told exactly which moves are legal before it picks one. Every observation carries an
`action_mask` with one entry per move. It never wastes a decision on a card it cannot afford or a
tile it is not allowed to deploy on.

Legality is a property of the **pair**, the card and the tile together. Elixir depends on the
card. Where you may place depends on the card too: spells go anywhere, buildings never go into
the enemy pocket, troops need your own territory. A mask that could only say "this card" or "that
tile" separately would happily suggest putting a Knight on the enemy king. This one cannot.

What it covers: elixir, territory, water, the river band, building footprints, and the no-deploy
rectangle around each living enemy crown tower.

On the first step of a battle only the wait is legal, because a match refuses every deploy for
its first 90 ticks, so the program above now prints 1 there rather than the 1267 shown, and that
block is due a re-run. After that the number moves with your hand, your elixir and your deck.

Three practical notes.

- The mask arrives as `int8`, which is what PettingZoo's `parallel_api_test` and Gymnasium's
  `Discrete.sample(mask=...)` expect. `action_masks()` returns it as booleans, which is what
  sb3-contrib's MaskablePPO calls for.
- `mask_planes` is the same mask reshaped into a grid, for a network with convolution layers.
- If you send a move the engine refuses anyway, it becomes a wait and is reported in
  `info["deploy_status"]`. Nothing explodes.

The mask is worked out from the board on the Python side, completely separately from the engine,
and then compared against the engine's own ruling for every position in the test suite. So a
wrong mask fails a test instead of quietly poisoning a training run.

## What your bot can and cannot see

By default your bot sees what a person watching the match could write down. It does **not** see
the opponent's hand. It does get the opponent's elixir, as a count kept from the plays it
watched, the way a player counts it in their head.

You can turn a hidden thing on with a `Reveal`, for a curriculum or for debugging. Doing so makes
the observation **wider** rather than filling in blanks, and `ClashParallelEnv.config()` records
that you did it. So a checkpoint always says whether the bot was allowed to cheat. Every channel
and every slot is in
[`docs/observation-spec.md`](https://github.com/RoyaleGym/RoyaleGym/blob/main/docs/observation-spec.md).

!!! warning "Never write down an observation width"
    The size of the observation depends on how many cards your machine's catalogue holds, and
    that changes. Read the shapes off a running environment instead of typing them into your
    code. Everything in the project does it that way for the same reason.

## When you would touch it

- You want a different reward. This is the common one. See [Writing a reward function](../rewards.md).
- You want your bot to see something extra, or see it differently.
- You want moves at a different resolution, or a different action space entirely.
- You want matches to start somewhere other than 0-0, for instance from a damaged mid-game board
  or from a saved snapshot, to build a curriculum.
- You want to record a battle and re-run it later, or check that it replays identically.

## When you would not

| You want to | Go here instead |
|---|---|
| fix a card that behaves wrong | [The engine](engine.md) |
| add a mechanic the battle does not model | [The engine](engine.md) |
| run PPO, keep checkpoints, rate policies against each other | [The learner](learner.md) |
| look at what your bot did | [The viewer](viewer.md) |

## Speed, and the honest caveat

The durable figure is a ratio: the Rust engine runs **1.16 to 1.38 times** the pure-Python
stand-in, measured by alternating the two inside one process so that whatever the machine is
doing it does to both. An env step is one decision for each player.

Absolute rates on the maintainer's laptop have ranged from 859 to 1812 env steps per second,
depending on the hour and what else was running. At 10 ticks a step, that is roughly 8,600 to
18,100 three-minute battles an hour in one process. Take any single figure as an illustration
rather than a target.

Stepped directly, the same engine does tens of thousands of ticks a second. The gap is Python:
on every single step it builds both players' observations and both players' legal-move lists.

That gap is the project's first open item, and the fix is to move the default observation into
the engine. Until that lands, most of the time it takes to play a battle through an environment
goes into building observations and legal-move lists rather than into the battle itself.
Nobody is pretending otherwise.

That is also why the Rust engine is only 1.16 to 1.38 times as fast as `MockEngine`, the
pure-Python stand-in, and not the landslide you might expect. The Python around both engines is
most of the cost.

## Two more things that are easy to miss

**You do not need the Rust engine to start.** `royalegym` imports without it. `RustEngine()` then
raises an error that names the build command, and `MockEngine` runs the whole API in the
meantime. It is a stand-in and not a second simulator: spells resolve instantly, there are no
stuns or knockbacks, and cards run at base level. Fine for writing code, wrong for judging a bot.

**[The learner](learner.md) trains on these environments, as of 2026-09-22, and no bot has
been trained with it yet.** The environments also expose `action_masks()` in the form
MaskablePPO expects, so you can point an existing library at them instead if you prefer.

## Where the detail is

- [RoyaleGym's README](https://github.com/RoyaleGym/RoyaleGym) for the whole picture.
- [`docs/architecture.md`](https://github.com/RoyaleGym/RoyaleGym/blob/main/docs/architecture.md)
  for the layers, the engine contract, the action space and the module map.
- [`docs/observation-spec.md`](https://github.com/RoyaleGym/RoyaleGym/blob/main/docs/observation-spec.md)
  for every channel and every vector slot.
- [`docs/background.md`](https://github.com/RoyaleGym/RoyaleGym/blob/main/docs/background.md)
  for what is publicly known about the game's rules.

Ready to build one? [Your first bot](../first-bot.md){ .md-button .md-button--primary }
