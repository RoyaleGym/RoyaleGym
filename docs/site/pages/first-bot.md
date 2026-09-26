# Your first bot

<p align="center">
  <img alt="Works today" src="https://img.shields.io/badge/part_1-works_today-2ea043?style=flat-square">
  <img alt="New, barely tested" src="https://img.shields.io/badge/part_2-new,_barely_tested-d29922?style=flat-square">
  <img alt="No GPU" src="https://img.shields.io/badge/GPU-not_needed_to_start-2ea043?style=flat-square">
  <img alt="No cloud" src="https://img.shields.io/badge/cloud-not_needed-2ea043?style=flat-square">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white">
  <a href="https://discord.gg/4D2BS5JBHP"><img alt="Discord" src="https://img.shields.io/badge/stuck%3F-ask_in_discord-5865F2?style=flat-square&logo=discord&logoColor=white"></a>
</p>

!!! warning "Read this before you scroll"
    **Training started working on 2026-09-22, and nobody has trained a bot with it yet.** Keep
    both halves of that in mind. `python -m royalelearn train` runs end to end now, but every run
    so far has been a short test. An iteration is one round of playing battles and then learning
    from them. As of 2026-09-22 the longest run on the real engine had ten iterations, each a
    quarter of the usual size, and the smoke test that proves the loop closes has three. So
    nothing on this page can tell you how long a useful run takes or whether the bot comes out
    any good. You would be the first to find out.

    **Training needs torch, and it is an extra rather than part of the plain install.** Run
    `pip install -e "RoyaleLearn[torch]"` before you try to train. Skip it and `train`, `doctor`
    and `bench` all stop with `ModuleNotFoundError: No module named 'torch'`. That is the one
    first-run failure worth recognising on sight. Being able to install the rest without torch
    is deliberate: the settings, the run identity and the workers are all tested to work without
    it.

    This page has two halves. Part 1 is code that runs on your machine today, and every block in
    it was run to write this page, with the real output underneath. Part 2 is the training run
    itself. Its commands are real now, but they are new. This page ran them only briefly, and
    it says which results it checked and which it took from the people who wrote them.

What you get out of Part 1 is a program that plays a whole Clash Royale battle, a bot of your own
that is better than random, and a window you can watch it in. That is the whole scaffolding the
learner plugs into. The one piece you are expected to write for training is a reward function.

## What you need first

The [Install](install.md) page. You need the folder called `Royale` with `RoyaleSim` and
`RoyaleGym` cloned inside it, one virtual environment at the root, and the engine built. The
viewer section below also wants `RoyaleViser` installed, which is one more `pip install -e` line.
Part 2 wants `RoyaleLearn` with its torch extra, as the box at the top says.

Every command on this page is run from inside a repo folder, and the Python is
`..\.venv\Scripts\python`. On macOS and Linux it is `../.venv/bin/python`.

---

## Part 1: what works today

### Play a whole battle

Here is a complete program. It builds a battle, plays it to the end with both players choosing at
random among their legal moves, and prints who won.

!!! note "About the deck"
    These eight cards are here so the battle comes out the same every time you run it. Your
    numbers can still differ from ours. The output on this page came from the 15.535 card table,
    the same one the Install page gives you, but the engine keeps changing and a newer engine can
    end the same battle differently. The eight cards are an
    example, not a recommendation. All eight are from the 18 whose behaviour is checked against
    recordings, which `cards.json` lists under `thin_slice`, so the example leans on the
    best-measured part of the engine. Any eight will do, and if you leave the deck
    out each team is dealt a random eight, which is the default.

```python
import numpy as np
from royalegym import (ClashParallelEnv, DefaultStateMutator, RandomLegalOpponent, RustEngine)

engine = RustEngine()
by_name = {c.name: c.card_id for c in engine.cards()}          # look cards up BY NAME
deck = [by_name[n] for n in ("Knight", "Archer", "Giant", "Minions",
                             "Fireball", "Cannon", "Zap", "Musketeer")]

env = ClashParallelEnv(engine=engine,
                       state_mutator=DefaultStateMutator(decks=[deck, deck]))
obs, info = env.reset(seed=0)
rng, policy = np.random.default_rng(0), RandomLegalOpponent(noop_prob=0.7)

while env.agents:                                              # one whole battle
    actions = {a: policy.act(obs[a], obs[a]["action_mask"], rng) for a in env.agents}
    obs, reward, terminated, truncated, info = env.step(actions)

s = env.battle_state
print(f"winner {s.winner}  crowns {[p.crowns for p in s.players]}  tick {s.tick}")
```

```
winner 0  crowns [1, 0]  tick 3728
```

Blue is player 0 and Red is player 1. A *tick* is the game's own 50 ms step and there are 20 in a
second, so tick 3600 is exactly three minutes, the end of normal time. Neither side had a crown
then, so this match went to overtime, where the first crown wins. Blue took one of Red's princess
towers at tick 3728, 6.4 seconds in (re-run 2026-09-24 on engine build `cb784bb583586789` with the
15.535 card table). Had overtime run out level too, a tiebreak would have decided it: the side
whose weakest standing tower has less health left loses.

One env step is half a second of game time, which is 10 ticks. So each player made 373 decisions
in that battle, the last one cut short when the tower fell. It took about half a second of real
time.

!!! tip "Always name the deck, card by card"
    If you leave `state_mutator` out, each side is dealt eight random cards from whatever card
    catalogue your machine built, and the same seed gives you a different battle from the one
    above. Look cards up by **name** and never by number. A card id is only a position in the
    catalogue, and positions move between card tables.

### Make it play something of your own

`RandomLegalOpponent` is not special. A policy in RoyaleGym is anything with an `act` method that
takes an observation and a legality mask and returns one integer. That integer is the move.

Two things to know about the numbering:

- Action `0` is the no-op. It means "play nothing this step". Waiting is a legal move and a good
  one, so it has its own number.
- Every other action is one card in your hand on one tile. There are 2304 of those, so 2305
  actions in all. The action parser turns a hand slot and a tile into that number for you with
  `parser.encode(slot, x, y)`, so you never have to do the arithmetic.

The `action_mask` in the observation is an array of 0s and 1s, one per action, and a 1 means the
engine will accept that move right now. Your bot is never allowed to waste a turn on a card it
cannot afford or a tile it may not deploy on.

Here is the smallest bot that is not random. It plays whatever card is in hand slot 0 on one
fixed tile, every single time that is legal, and waits otherwise.

```python
import numpy as np
from royalegym import (ClashParallelEnv, DefaultStateMutator, RandomLegalOpponent,
                       RustEngine, TileActionParser)

NOOP = 0                                   # action 0 means "play nothing this step"


class OneTilePolicy:
    """Play hand slot 0 on one fixed tile whenever that is legal. Otherwise nothing."""

    def __init__(self, parser, slot=0, tile=(9, 5)):
        self.action = parser.encode(slot, tile[0], tile[1])
        self.plays = 0

    def act(self, obs, mask, rng):
        if mask[self.action]:
            self.plays += 1
            return self.action
        return NOOP


engine = RustEngine()
by_name = {c.name: c.card_id for c in engine.cards()}
deck = [by_name[n] for n in ("Knight", "Archer", "Giant", "Minions",
                             "Fireball", "Cannon", "Zap", "Musketeer")]

parser = TileActionParser()
env = ClashParallelEnv(engine=engine, action_parser=parser,
                       state_mutator=DefaultStateMutator(decks=[deck, deck]))
obs, info = env.reset(seed=0)

rng = np.random.default_rng(0)
blue = OneTilePolicy(parser)               # your bot
red = RandomLegalOpponent(noop_prob=0.7)   # the opponent that ships

while env.agents:
    obs, reward, terminated, truncated, info = env.step({
        "blue": blue.act(obs["blue"], obs["blue"]["action_mask"], rng),
        "red": red.act(obs["red"], obs["red"]["action_mask"], rng),
    })

s = env.battle_state
print(f"winner {s.winner}  crowns {[p.crowns for p in s.players]}  tick {s.tick}")
print(f"blue deployed {blue.plays} times")
```

```
winner 0  crowns [3, 1]  tick 4466
blue deployed 36 times
```

Read that result. Blue won with three crowns, but it did not take three towers. Blue took one of
Red's princess towers first, Red took one of Blue's back, and it was level at one crown each when
the three minutes ran out. In overtime Blue brought down Red's king tower at tick 4466, 43
seconds in, and a fallen king tower counts as all three crowns. Blue deployed 36 times in the
whole match, because it plays the instant it can afford whatever is in slot 0 and does nothing
else at all (run 2026-09-26 on engine build `e2ab40401fe9395f` with the 15.535 card table).

That is a stupid bot and it beat the random one here. That is the point. A fixed rule with no
learning in it can already beat picking legal moves out of a hat, which tells you the random
opponent is a floor and not a wall.

Run it twice and you get the same two lines. Same seed, same battle, every time.

!!! note "Your exact result can differ, and here are the two reasons why"
    The engine keeps changing, and a newer engine can end this same battle differently, even
    with the same seed. And card levels come from the card table your engine reads: if it reads
    the 2018 table rather than the 15.535 one the Install page sets up, you may see a different
    crown count here. Either way the program is still correct.

Things worth trying from here, all of them a few lines:

| Change | What you do | What you learn |
|---|---|---|
| A different tile | change `tile=(9, 5)` | where on the board a dumb bot does best |
| A different slot | change `slot=0` | which card in the cycle is worth spamming |
| Save elixir | return `NOOP` unless the elixir field says you are near 10 | whether waiting beats spamming |
| Two of your own bots | pass `OneTilePolicy` for both seats | what a mirror match looks like |

### Watch what it did

Numbers tell you who won. They do not tell you why. The viewer,
[RoyaleViser](pieces/viewer.md), draws the battle in a window so you can look at it.

There are two ways in. Record a battle to a file and open it later, or attach to a run that is
happening now.

#### Record a file, open it later

This saves every tick of a 200 step battle, which is 2001 frames.

```python
import numpy as np
from royalegym import (ClashParallelEnv, DefaultStateMutator, RandomLegalOpponent,
                       ReplayRecorder, RustEngine, StepLimitCondition, save_trace)

engine = RustEngine()
by_name = {c.name: c.card_id for c in engine.cards()}
deck = [by_name[n] for n in ("Knight", "Archer", "Giant", "Minions",
                             "Fireball", "Cannon", "Zap", "Musketeer")]

rec = ReplayRecorder(frame_every_tick=True)          # keep every 50 ms tick, not every step
env = ClashParallelEnv(engine=engine, recorder=rec,
                       state_mutator=DefaultStateMutator(decks=[deck, deck]),
                       truncation_cond=StepLimitCondition(200))
obs, _ = env.reset(seed=2026)
rng, opp = np.random.default_rng(0), RandomLegalOpponent(noop_prob=0.3)
while env.agents:
    obs, *_ = env.step({a: opp.act(obs[a], obs[a]["action_mask"], rng) for a in env.agents})

save_trace(rec.trace, "battle.msgpack")
print(f"{len(rec.trace.frames)} frames saved to battle.msgpack")
```

```
2001 frames saved to battle.msgpack
```

Then open it. A window appears and plays the battle back. Space pauses, the arrow keys step one
frame, clicking a unit pins it and lists every field the file holds about it, and `h` lists every
other key.

=== "Windows"

    ```
    ..\.venv\Scripts\python -m royaleviser battle.msgpack
    ```

=== "macOS and Linux"

    ```
    ../.venv/bin/python -m royaleviser battle.msgpack
    ```

Add `--start-tick 900` to open partway in. Add `--seconds 8` and the viewer closes itself after
eight seconds and prints how long each picture took to draw. That is what was run to check this
page, on a machine with the display switched off:

```
royaleviser: 80 draws, mean 6.10 ms, max 11.16 ms
```

Six milliseconds a picture. A replay needs twenty pictures a second, so there is a lot of room.

#### Attach to a battle that is running now

The viewer is always a separate program. Your run never waits for it, and nothing is sent at all
until a viewer says hello. So you can leave this switched on in a run nobody is watching.

Open two terminals. In the first, start the viewer and leave it there:

```
..\.venv\Scripts\python -m royaleviser --stream 127.0.0.1:9870
```

In the second, run a battle that publishes. A single `ClashParallelEnv` is handed a publisher
directly:

```python
import time
import numpy as np
from royalegym import (ClashParallelEnv, DefaultStateMutator, RandomLegalOpponent, RustEngine)
from royalegym.viser import ViserPublisher

engine = RustEngine()
by_name = {c.name: c.card_id for c in engine.cards()}
deck = [by_name[n] for n in ("Knight", "Archer", "Giant", "Minions",
                             "Fireball", "Cannon", "Zap", "Musketeer")]

pub = ViserPublisher()                                  # 127.0.0.1:9870
env = ClashParallelEnv(engine=engine, viser=pub,
                       state_mutator=DefaultStateMutator(decks=[deck, deck]))
obs, _ = env.reset(seed=0)
rng, opp = np.random.default_rng(0), RandomLegalOpponent(noop_prob=0.7)
while env.agents:
    obs, *_ = env.step({a: opp.act(obs[a], obs[a]["action_mask"], rng) for a in env.agents})
    time.sleep(0.05)                                    # slow it down so a human can watch
print(f"frames sent {pub.sent}  dropped {pub.dropped}")
```

```
frames sent 3528  dropped 0
```

The `time.sleep` is there because a battle otherwise finishes in half a second and you would see
nothing. A stream sends one frame per engine tick, and this battle ran 3,728 ticks. It sent 3,528
because nothing goes out until the run has heard a hello from the viewer, and the run only
listens for one once a second. So the first second or so went unsent, even though the viewer was
already open when the run started (run 2026-09-24 on engine build `cb784bb583586789`). Attach
the viewer later and you get fewer frames. Nothing is sent while nobody is listening.

!!! warning "One env does not read the `ROYALEVISER` variable. This trips people up."
    You may have seen `set ROYALEVISER=127.0.0.1:9870` written as the way to switch the viewer on.
    That works, but only for the batched self-play environment, not for a single
    `ClashParallelEnv`. A single env is handed a publisher explicitly, as in the program above.

    The reason is that a viewer shows one battle at a time and holds one UDP port. Eight
    environments all reaching for that port is an error, not eight streams. So
    `ClashSelfPlayVecEnv` is the thing that reads the variable. It binds the publisher once and
    hands it to game 0 only.

    ```
    set ROYALEVISER=127.0.0.1:9870      # before your program starts
    ```

    ```python
    from royalegym import ClashSelfPlayVecEnv
    env = ClashSelfPlayVecEnv(8)        # binds one publisher, watches game 0
    ```

    Both of these were run to check this page and both fed the viewer.

    `9870` is a default, not the address. One run holds that port while it streams, so a second
    run on the same machine needs its own: set `ROYALEVISER=127.0.0.1:9872` and point that run's
    viewer at the same number. Starting a second run on a port that is taken fails when it binds,
    and the message names the port rather than anything about training, so it reads as a broken
    example when it usually means something else on the machine is already streaming.

---

### Is it actually better?

This is the question you ask after every change, and it is the one a hand-written loop gets
wrong. A loop plays your bot in one seat, prints a bare percentage, and counts a battle the
step limit cut short as a draw. `evaluate` plays both seats and gives you an interval.

```
python examples/07_is_this_bot_better.py
```

```
greedy vs noop: 20-0-0 over 20 games. win rate 100.0% (83.9% to 100.0% at 95%) -- greedy is better. seat gap +0.0%, mean 2405 ticks.
    as blue 10-0-0
    as red  10-0-0
greedy vs random: 8-12-0 over 20 games. win rate 40.0% (21.9% to 61.3% at 95%) -- too close to call. seat gap -20.0%, mean 3310 ticks.
    as blue 3-7-0
    as red  5-5-0
```

Two things to read off that, and the second is the one people get wrong. Both numbers below come
from the run printed above, which is 20 games on your machine and nobody else's.

**"Too close to call" is not a tie.** It means the interval still covers 50%, so the games you
played cannot separate the two bots. Play more and ask again. A bare "40%" would have told you
the opposite of the truth here.

**A large seat gap means you measured the colour, not the skill.** Blue and Red are not the same
job: whoever is behind has to attack. `evaluate` plays every pairing both ways and reports the
gap, so you can see when that is what your number is made of.

!!! tip "How many games is enough? More than you think"
    RoyaleGym's own maintainer ran a 40-game round robin over the six shipped opponents, on the
    pure-Python stand-in engine. It established exactly two orderings: the do-nothing opponent
    loses to everything, and the patient one beats random 27 to 13. The three in the middle,
    which are described in increasing order of sophistication, came out 19-19, 20-19 and 20-19.
    Forty games could not tell them apart. If you change your reward function and see a five
    point move over twenty games, you have seen noise.

## How long a run will take

Two numbers, and they are different on purpose. Both were measured on a four core Windows laptop
with 7.8 GB of RAM and several other jobs running, so both are pessimistic.

| What is being measured | Ticks per second | What that means for you |
|---|---|---|
| The engine on its own, driven straight from Python | 51,582 | about **51,000 whole battles an hour** on one core |
| The engine through a RoyaleGym environment | 9,570 | about **9,600 battles an hour** on one core |

The second row is RoyaleGym's own throughput test on 2026-09-22: 957 environment steps a
second, and each step is 10 ticks. The same test has printed as much as 1,812 steps a second on
that laptop, depending on the hour and what else was running.

The arithmetic is exact rather than a trick. A three minute battle is 3,600 ticks and an hour
is 3,600 seconds, so a ticks-per-second figure is also a battles-per-hour figure for one
process.

The gap between the two rows is Python. On every step the environment builds both players'
observations and works out the full list of legal moves. That is real work and it happens outside
the engine. Closing that gap is the first open item in
[RoyaleGym's status](https://github.com/RoyaleGym/RoyaleGym#status), and until it is
closed, a training run spends more time describing the battle than playing it.

!!! danger "Nobody knows how many battles a good bot needs"
    The loop runs and a few short runs have gone through it. Nothing has been trained to
    the point of being good, or of being measured against anything that would tell you. So
    there is no honest answer to "how long until my bot is good", and anyone who gives you one
    is guessing. What you can take from the table above is the cost of a battle, not the number
    of battles required.

---

## Part 2: training

!!! note "What has and has not been run here"
    `train` was run for this page and it completed:

    ```
    royalelearn train --config examples\configs\smoke.json
    ...
    run 85b4ce0a1d6e8f1d stopped at iteration 3
    ```

    It left a checkpoint with the network, the advantage scaler and the ladder's pool under
    `runs/smoke-85b4ce0a1d6e8f1d/checkpoints/`.

    Read what that proves narrowly. `smoke.json` runs on `MockEngine`, the pure-Python
    stand-in, with a `timestep_limit` of 96. It is a self-test that the loop closes, not a
    training run and not the real engine.

    **The real profile now runs, and nobody has trained a bot with it.** `laptop.json` uses
    `RustEngine` and a limit of 100,000,000 timesteps. An earlier bug stopped it in its first
    collection round. That bug is fixed, and a test now guards against it. Checked here rather
    than taken on report: it collects full rounds.

    **A long run stops itself today (2026-09-22).** After a few iterations `laptop.json` raises
    `MixtureDrifted`, a self-check comparing how many collected rows were thrown away against
    how many the opponent mixture says should be. It is a real disagreement in the harness, not
    your setup, and it is being fixed. Short runs and the smoke profile are unaffected. If you
    hit it, that is the known one.

    **The first round of a run is much slower than the ones after it.** That, and not your
    hardware, is the thing to know before you time anything. The workers are spawning and
    nothing is warm yet. The same 228-cycle round, across six measurements on one laptop:

    | | how long | spread |
    |---|---|---|
    | first round of a run | 46 s, 79 s, 179 s | 3.9x |
    | every round after | 12.6 s, 13 s, 19 s | 1.5x |

    Read the right-hand column. Once a run is warm it is fairly steady even on a busy machine,
    and the first round is between 2.4 and 14 times the steady one depending on what else is
    happening. So wait for the second round before you believe any number, including the ones
    on this page.

    Its author reports two complete iterations with metrics rows, at 518 s and 544 s. Read that
    with its settings attached: 2026-09-22, the laptop profile at **minibatch 512**, which is
    not what ships any more. The default is 256 now, because 512 does not fit a 4 GB card and
    spills into system memory. Nobody has timed a full laptop-profile iteration at 256, so
    treat the nine minutes as the last measured figure rather than as what you will see. It
    also degrades badly under contention: one iteration sharing eight processors with a second
    training run had still not finished after 46 minutes.

    What does hold across every run so far, at a smaller geometry (2 workers, 24 battles each,
    8,192 timesteps an iteration, minibatch 256, about a hundred iterations over three runs on
    an RTX 3050): the learning step is 71 to 83 percent of each iteration. Training time on a
    machine like that is graphics-card time, not battle-simulation time. Making the engine
    faster would buy you nothing there.

    Two iterations is a loop that works. It is not a result about learning, and a useful run
    is many hours. So the honest state is that the machinery runs end to end and nobody yet
    knows whether the bot it produces is any good. That is the thing left to find out, and it
    is available to whoever does it first.

Source for this section: RoyaleLearn's own README and its owner, on 2026-09-22.

### The four commands

All four of these work. `config` runs without torch; the other three need the torch extra.

`config` writes a config file you can edit. `doctor` runs the first-run checks, before you commit
hours. `bench` measures YOUR machine rather than someone else's. `train` is the run itself.

```
python -m royalelearn config --profile laptop -o run.json
python -m royalelearn doctor --config run.json
python -m royalelearn bench
python -m royalelearn train --config run.json
```

Start with the middle two, not the last one.

- `config` writes out a config file with sensible settings so you have something to edit rather
  than a blank page. The profiles are `laptop`, `workstation` and `many_core`. Every training run
  so far, as of 2026-09-22, used `laptop`. No run has used the other two yet.
- `doctor` builds one environment and runs the start-up gates on it. These are the same gates
  `train` runs before it begins, so this is what they look like, copied from a real run:

        mask gate     4608 actions checked, 0 disagreements
        no-op gate    1000 sampled states and a finished battle, all legal
        action layout 2304 actions exhaustive, mask planes checked on 1000 states
        engine build  calibration dbd052b6cdce build c3f431117e93 catalogue d6170aa68d21
        memory        projected total                  3292 MB
                      free right now                    429 MB
                      WARNING: the projection is 2862 MB over what is free.
        geometry      3 workers x 32 battles = 96 battles, 192 slots, 144 of them learner rows
        iteration     228 cycles for 32768 timesteps; credit horizon 38.6 s

  It checks every legal move against the engine exhaustively rather than sampling, prints the
  three build digests so you can tell whether your engine matches your data, and projects the
  memory a run will need. It refuses a run that is over the memory budget in your config, and
  it warns, as above, when a run needs more than is free right now. Run it first; it is quick,
  and it is where a mismatched build shows up.
- `bench` measures how fast your own machine is, so you can plan a run against your number
  instead of the table above. Budget time for it. It starts a set of worker processes, runs at
  least one whole training iteration whatever `--seconds` says, and prints nothing while it
  works. On a 4-core laptop with other jobs running it had produced no output after fifteen
  minutes, at which point it was stopped rather than left to finish, so what it finally prints
  is not recorded here. Run it when you can leave the machine alone.
- `train` is the run.

There is also a script, `examples/train_1v1.py`, for people who would rather edit Python than a
command line. It loads `examples/configs/laptop.json`, gives the run a name, changes one
learning setting, and hands the result to the coordinator. This is the shape of it, not a
transcript:

```python
# examples/train_1v1.py, about fifteen lines: load a config, change a few fields, then
with LearningCoordinator(config) as run:
    run.learn(until_timesteps=100_000_000)
```

Weights & Biases, an online dashboard for training numbers, is off in the script. To use it, set
`USE_WANDB = True` at the top, install the `wandb` extra and sign in to a W&B account. Either way,
the run's numbers go to the console and to a `metrics.jsonl` file in the run's folder.

The command line and the script go through the same object. Neither is a wrapper around the other.

### The reward function, which is the part you actually write

This is the one piece a bot creator is expected to change, and it is the reason this project
exists. A reward function says what your bot should want.

It lives in `royalelearn/rewards.py`, composed in a function called
`default_potential_reward()`. To change it you write a subclass of `RewardFunction`, which is
RoyaleGym's base class, and you name your class in the config's `env` block. Naming it in the
config rather than editing the default means the checkpoint records which reward the bot was
trained on, so you can never lose track of what a saved bot was trying to do.
`examples/custom_reward.py` shows the whole thing. It adds one new term to the shipped four,
names it in the config, and adds the one line that lets the worker processes import your file.

The shipped composition is four terms. The objective is winning. The other three are there to make
the first hour of a run readable: crowns, tower hitpoints, and elixir you have committed to the
board.

!!! danger "Do not re-tune the shipped weights. This is the newcomer mistake."
    The three shaping terms are written as differences of potentials. That form is what makes them
    unable to change which strategy is actually best. They speed the bot up without lying to it.

    A weight you nudge every time you measure a new behaviour is standing in for a term that is
    missing. Write the missing term instead. There is an alarm called `shaping_dominates` that
    watches the shaping terms against the objective for exactly this reason.

### Numbers you should never hard-code

Every shape, every vector width and every field's place in the observation is read off the running
environment when the harness starts. None of them is typed into the code. The card catalogue
changed size recently and nothing in the code needed editing. If you find yourself writing a
number like "the observation is N wide" into your own script, that script will break on somebody
else's machine.

---

## How to tell whether it is working

A training run reports a lot of numbers. Here are the ones to watch and what they mean for you.
The metric names come from RoyaleLearn's metric schema and alarm table, which are written and
tested. What no one can tell you yet is what healthy numbers look like on a real run, because no
run has gone long enough to show it. Treat the thresholds below as the code's own defaults, not
as measured facts.

### Good signs

- **`ladder/score_vs_random_legal` climbing.** This is your bot's score against the random
  opponent you met in Part 1. The schema's healthy band for it is 0.7 to 1.0. A bot that cannot
  beat random is not learning.
- **`ladder/score_vs_noop` near 1.0.** The no-op opponent never plays a card. Losing to it means
  something is badly wrong, not that training is slow.
- **About 22 cards played per match.** That is the figure the harness treats as healthy. It means
  your bot is actually playing the game rather than sitting on its elixir.

### Three things that mean it is not working

**1. The score against the frozen opponents stops moving.**

What you would see: `ladder/score_vs_random_legal` flat for a long stretch, and
`ladder/consecutive_gate_failures` climbing. There is an alarm for this called `gate_starved`,
described in the code as the plateau signal stated as an event. The usual company it keeps is
`ev_negative`, which fires on `ppo/explained_variance` and means the critic is not predicting
the outcome at all.

What to do: this is the ordinary shape of a run that has found a local habit and stuck in it. The
first thing to look at is the reward function, not the learning rate. Watch a few battles in the
viewer and ask whether the thing your bot is doing is actually what your reward pays for.

**2. The bot collapses onto one move and plays it forever.**

What you would see: `ppo/noop_entropy` falling through the floor, then the alarm
`noop_collapse`, then `noop_collapse_severe`, which is a halt rather than a warning and means the
policy has stopped playing cards at all. A related one is `tile_spam`, which fires when a quarter
of every card played goes on one tile.

What to do: the `noop_entropy` number is the leading indicator, so it moves before the card count
does. If it starts falling, stop and look rather than waiting for the halt.

**3. Illegal moves are reaching the engine.**

What you would see: `health/illegal_action_rate` above zero, at all. The schema says this is
exactly zero by construction and calls it an alert rather than a plot. There is a companion,
`health/mask_disagreements`, and a run refuses to start if that is above zero.

What to do: this is not a training problem. It means the list of legal moves is not reaching your
policy, or the policy is ignoring it. Check that you are reading `obs["action_mask"]` and applying
it before you sample. A bot that can pick illegal moves spends its whole run learning to want
things it cannot have.

!!! note "Which of this is measured and which is judgement"
    The metric names, the alarms and the thresholds are real. They are in RoyaleLearn's code and
    they are tested. **The advice under "what to do" is judgement, not measurement.** No run has
    gone long enough yet for anyone to try these fixes and see which one works.

---

## Where to go next

| If you want to | Read |
|---|---|
| Understand what your bot actually sees | [What the bot sees and does](observations-and-actions.md) |
| Write the reward function | [Writing a reward function](rewards.md) |
| Know how close the engine is to the real game | [How accurate is the engine](accuracy.md) |
| Get more out of the viewer | [The viewer](pieces/viewer.md) |
| Fix something that broke | [Troubleshooting](troubleshooting.md) |

Ask in the Discord. This is the kind of project where a question about one reward function is a
perfectly good first message.

<p align="center">
  <a href="https://discord.gg/4D2BS5JBHP"><img alt="Join the Discord" src="https://img.shields.io/badge/Discord-join%20the%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white"></a>
</p>
