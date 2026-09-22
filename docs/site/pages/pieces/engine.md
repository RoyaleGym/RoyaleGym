# The engine

[![repo](https://img.shields.io/badge/repo-RoyaleSim-DEA584?style=flat-square&logo=rust&logoColor=white)](https://github.com/RoyaleGym/RoyaleSim)
![cards](https://img.shields.io/badge/cards-78%20public%2C%20144%20full-555?style=flat-square)
![tick](https://img.shields.io/badge/tick-50%20ms%2C%2020%20per%20second-555?style=flat-square)
![speed](https://img.shields.io/badge/one%20worker-16%2C100%20battles%2Fhour-2ea043?style=flat-square)
![same seed](https://img.shields.io/badge/same%20seed-same%20battle-2ea043?style=flat-square)

**This is the thing the battle actually happens in.** You play a card, a Giant walks out, a tower
shoots it, the crowns get counted. All of that is RoyaleSim. It is written in Rust and you drive
it from Python by importing `royalesim`.

There is no game to run and nothing to connect to. No phone, no copy of the game, no account and
nothing to wait for over a network. You install a Python module and call it.

If you are here to make a bot, the honest summary is this. You will almost never open this repo.
Your bot talks to [the environments](environments.md), and the environments talk to the engine.

<div class="grid cards" markdown>

-   **It plays the whole match**

    ---

    Elixir, hands and the cycle, deploys, walking, targeting, fighting, spells, towers, double
    elixir, 60 seconds of overtime, the three-crown win and the tiebreak.

-   **The same seed gives the same battle**

    ---

    Run it twice with the same seed and you get the same match, unit for unit, on any machine.
    It does whole-number arithmetic only, so nothing drifts. A battle that went wrong once can
    be made to go wrong again.

-   **It is fast enough to stop being the problem**

    ---

    About 16,100 three-minute battles an hour from one worker process, and about 65,000 from
    six of them. Add cores and you add battles. The engine stepped on its own, with nothing
    built on top, goes about three times faster again, which is the point: the slow part of
    training is the Python around the engine, not the engine.

-   **The movement rules were measured**

    ---

    How units pick routes, walk and shove each other apart was measured against recordings of
    real matches. Every constant records which game version it came from and whether it is a
    measurement or a guess.

</div>

## What it is not

It is not the real game and it does not claim to be. It knows nothing about rewards, observations
or training. It does not know what a bot is. It hands you a board and moves it forward.

It is also not perfect. [How accurate is the engine](../accuracy.md) gives the numbers and says
where it is still wrong.

## Try it

This is the smallest interesting thing the engine does. A Giant is played for Blue, which is
team 0 and the bottom half of the arena, and then left alone for 24 seconds of game time. Nobody
tells it where to walk.

You need the install from [Install](../install.md) first, up to and including the engine build.

```python
import json, royalesim

deck = ["Giant", "Knight", "Archers", "Musketeer", "Fireball", "Arrows", "Minions", "Zap"]
b = royalesim.Battle(card_names=deck, slot_of_k=[[0, 1, 2], [0, 1, 2]])
b.reset(seed=1, decks=[list(range(8))] * 2, shuffle=0, start_tick=0,
        elixir_milli=[10_000, 10_000], tower_hp=None, spawns=[])

T = royalesim.SUBTILE                   # positions are in subtiles: 18000 to one arena tile
b.step([(0, 0, 5 * T, 10 * T)], 0)      # Blue plays hand slot 0 (the Giant) on tile (5, 10)
for _ in range(6):
    b.step([], 80)                      # 80 ticks = four seconds of game time, one call
    s = json.loads(bytes(b.state_json()))
    giant = next(e for e in s["entities"] if e[3] == 0)
    print(f"t={s['tick']}  giant at ({giant[5] / T:.2f}, {giant[6] / T:.2f})"
          f"  hp={giant[7]}  red left tower hp={s['players'][1]['tower_hp'][1]}")
```

```
t=80  giant at (4.34, 12.56)  hp=3968  red left tower hp=3052
t=160  giant at (3.86, 16.15)  hp=3968  red left tower hp=3052
t=240  giant at (3.77, 19.82)  hp=3532  red left tower hp=3052
t=320  giant at (3.77, 22.57)  hp=2987  red left tower hp=2799
t=400  giant at (3.77, 22.57)  hp=2442  red left tower hp=2293
t=480  giant at (3.77, 22.57)  hp=1897  red left tower hp=1534
```

Read that as a story. The Giant slid left onto the bridge column, crossed the river around t=160,
walked into princess-tower fire, stopped within its own reach of the tower at t=320 and started
hitting it. Nobody steered it. It picked its own route.

!!! warning "Your hitpoints may not match, and that is fine"
    The positions and the timing above are the same on any checkout. The hitpoints are not.
    Card levels come from the card table your machine built, and the run above used the 15.535
    table. On the `--vintage 2018` table, which is the one a fresh clone builds, the two
    right-hand columns move. If your route matches and your hitpoints do not, nothing is wrong.

To watch a battle instead of reading numbers, run `python tools\watch_battle.py --open` from the
RoyaleSim folder. It plays a random three-minute match, runs five checks on it, and opens a
self-contained HTML page you can scrub tick by tick. The whole thing took 2.07 seconds here, and
all five checks came back green:

```
winner: RED
crowns: [0, 2]
final_tick: 3600
troops: 61
[OK ] determinism: re-simulated hash-for-hash on a fresh engine
[OK ] vacuity: 3601 frames (floor 50); blue accepted 24 deploys; red accepted 23 deploys; ...
[OK ] arena: trace grid == data/derived/arena.json (64x36 half-cells)
[OK ] dry: 138 of 37819 ground entity positions on water (the live game allows it; bound 5 %)
[OK ] render: battle.html, 2440450 bytes, 3601 frames, self-contained
EVERY GATE GREEN.
```

You get that battle, not a different one: `--seed` defaults to 1. Only the timings move, so a
difference in the result is a signal rather than noise. Pass `--seed N` for another battle and
`--open` to open the page in a browser.

## When you would touch it

Four reasons, and they are all "the battle itself is wrong or missing something".

**A card behaves wrong.** A unit walks somewhere it should not, a spell does the wrong damage, a
building sits in the wrong place. That is engine behaviour, so it is fixed here.

**A mechanic is not modelled.** These are the ones the engine does not do yet, in plain words:
dash and morph, air units doing anything cleverer than flying straight at their target,
evolutions, champion abilities and tower troops. Only the 18 cards in `thin_slice` are checked
against recordings. The other 126 carry data that no test covers.

**You want a constant changed.** Every number the engine uses lives in `data/calibration.json`
with a status attached, from guess to measured, and the name of the recording that pinned it. You
can change one and rebuild.

**You want to help close the accuracy gap.** The two biggest sources of error are where
multi-unit cards put their units and how units push each other apart on contact. Together those
are 63% of what is left. Both are open work.

!!! warning "Rebuild after you change data"
    The engine compiles `data/calibration.json` and `data/derived/arena.json` into itself, and
    the card table is fixed when the engine is **built**, not when it is run. So change data
    first, build second. RoyaleGym refuses an engine build that is older than the data on disk,
    which is the error you will see if you forget.

## When you would not

| You want to | Go here instead |
|---|---|
| change what your bot is rewarded for | [Writing a reward function](../rewards.md) |
| change what your bot sees, or what its moves mean | [What the bot sees and does](../observations-and-actions.md) |
| start a match from a mid-game position | [The environments](environments.md) |
| run a training loop | [The learner](learner.md) |
| watch a battle, or save a picture of one | [The viewer](viewer.md) |

None of those need the engine rebuilt. That is the point of the split. The engine knows nothing
about rewards, so you can change a reward without recompiling anything.

## The numbers, and where they come from

**Speed.** On a 4-core laptop with 8 GB of RAM, with other programs running, the engine did
18,000 ticks in 0.35 to 0.42 seconds on one core. That is 43,000 to 51,000 ticks a second.

There is a piece of arithmetic here worth keeping. A three-minute battle is 3,600 ticks, and an
hour is 3,600 seconds. So a ticks-per-second figure is also a battles-per-hour figure for one
process. 51,582 ticks a second is 51,582 whole battles an hour on one core.

RoyaleSim's README measures how that scales across worker processes. The scaling is the part
that should hold on your machine:

| workers | speed-up over one worker |
|---|---|
| 1 | 1.00x |
| 2 | 1.93x |
| 4 | 3.06x |
| 6 | 4.05x |

The fall-off past four workers is four cores running out. The absolute rates behind those
ratios, on that laptop with other programs running, were about 16,100 battles an hour on one
worker and 65,200 on six. Treat those as an illustration: the same measurement on this hardware
has moved by a factor of two inside one evening.

What that means for you: overnight rather than a fortnight, on a laptop, for a run of the size
people usually reach for. Nobody has trained a bot yet, so that is arithmetic on the battle rate
rather than experience.

**Accuracy.** Real matches are replayed in the engine and compared tick by tick. Leaving the six
towers out, because towers do not move and counting them flatters the result: a unit is within a
quarter of a tile of where it really was 49.4% of the time, and its hitpoints are exactly right
81.4% of the time. Single units are much better than swarms. A Knight is within a quarter tile
82.5% of the time. Goblins manage 42.5%.

Check [How accurate is the engine](../accuracy.md) before you rely on a specific interaction, and
check it again in a month, because these numbers are moving.

**Tests.** The engine's Python suite here was 108 passed in 134 seconds.

## Where the detail is

The engine has its own docs, and they go far deeper than this page.

- [RoyaleSim's README](https://github.com/RoyaleGym/RoyaleSim) for the whole picture.
- [`docs/architecture.md`](https://github.com/RoyaleGym/RoyaleSim/blob/main/docs/architecture.md)
  for how the engine is built.
- [`docs/pathfinding.md`](https://github.com/RoyaleGym/RoyaleSim/blob/main/docs/pathfinding.md)
  for the measured routes and the contact law, with the evidence.
- [`docs/mechanics.md`](https://github.com/RoyaleGym/RoyaleSim/blob/main/docs/mechanics.md)
  for what is modelled, what is not, and the two known collision defects.
- [`docs/calibration.md`](https://github.com/RoyaleGym/RoyaleSim/blob/main/docs/calibration.md)
  for the constants file and what each status word means.
- [`docs/replay-parity.md`](https://github.com/RoyaleGym/RoyaleSim/blob/main/docs/replay-parity.md)
  for the full accuracy table and how it is produced.

Engine questions and calibration work happen in [the Discord](https://discord.gg/4D2BS5JBHP).
