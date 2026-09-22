# Install

<p align="center">
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="Rust 1.80+" src="https://img.shields.io/badge/rust-1.80+-DEA584?style=flat-square&logo=rust&logoColor=white">
  <img alt="Python half: run from clean clones" src="https://img.shields.io/badge/python%20half-run%20from%20clean%20clones-2ea043?style=flat-square">
  <img alt="Release build: unmeasured" src="https://img.shields.io/badge/release%20build-unmeasured-orange?style=flat-square">
  <img alt="Tested on Windows" src="https://img.shields.io/badge/commands%20tested%20on-Windows-0078D4?style=flat-square">
</p>

At the end of this page you have four repositories side by side in one folder, one Python virtual
environment shared by all of them, and a compiled engine you can run battles in. Then you have a
short list of commands to check that it worked, with the output you should see.

## What you need first

- **Python 3.12.** Older versions are not supported.
- **Rust 1.80 or newer, with cargo.** Get it from [rustup.rs](https://rustup.rs). This is only
  needed to build the engine. If you cannot get Rust working, read
  [If the build will not work](#if-the-build-will-not-work) at the bottom. You can still start.
- **git.**

## How much of this page has actually been run

This matters more than it sounds, so it is near the top rather than in a footnote.

!!! success "The Python half was run from clean clones, and it works"

    Somebody cloned all four repositories fresh into an empty directory, with no build leftovers
    and no files that git ignores, and ran the recipe below: the clones, the virtual environment,
    the four extract commands and the `pip install -e` lines. RoyaleGym's test suite then passed
    there, 342 passed and 6 skipped in 56 seconds.

!!! warning "The release build has never been run from a clean clone"

    `maturin develop --release` is the one step nobody has done end to end from a fresh clone.
    A **debug** build was done in that clean clone, and it took **36 seconds** on a machine with
    about **1.5 GB of memory free**. That is the only build measurement that exists.

    How long the **release** build takes, how much memory it needs, and whether
    `import royalesim` works immediately afterwards with no further step, are all unmeasured.
    If you run it, the [Discord](https://discord.gg/4D2BS5JBHP) would like the numbers.

Everything below that has not been run carries a box saying so.

## Step 1: the folder, the clones and the virtual environment

Make one folder called `Royale`, put all four repositories inside it, and make one virtual
environment at the top. The repositories find each other by sitting next to each other, so the
layout is not optional.

!!! warning "UNVERIFIED: every macOS and Linux tab on this page"

    Nobody has run this from a clean install yet. Every command that was verified was run on
    Windows. The macOS and Linux tabs are the same commands with `.venv/bin` in place of
    `.venv\Scripts` and forward slashes in the paths, but nobody has typed them.

=== "Windows"

    ```
    mkdir Royale && cd Royale
    git clone https://github.com/RoyaleGym/RoyaleSim.git
    git clone https://github.com/RoyaleGym/RoyaleGym.git
    git clone https://github.com/RoyaleGym/RoyaleViser.git
    git clone https://github.com/RoyaleGym/RoyaleLearn.git
    python -m venv .venv
    .venv\Scripts\python -m pip install maturin pytest hypothesis ruff
    ```

=== "macOS and Linux"

    ```
    mkdir Royale && cd Royale
    git clone https://github.com/RoyaleGym/RoyaleSim.git
    git clone https://github.com/RoyaleGym/RoyaleGym.git
    git clone https://github.com/RoyaleGym/RoyaleViser.git
    git clone https://github.com/RoyaleGym/RoyaleLearn.git
    python -m venv .venv
    .venv/bin/python -m pip install maturin pytest hypothesis ruff
    ```

You should end up with this:

```
Royale/
    .venv/
    RoyaleSim/
    RoyaleGym/
    RoyaleViser/
    RoyaleLearn/
```

The rest of this page writes the Python as `.venv\Scripts\python`, from inside one of those four
folders, so it reads `..\.venv\Scripts\python`.

## Step 2: generate the card and arena data

A fresh clone has no card table and no arena. Neither `data/derived/` nor the folders under
`data/raw/cr-*/` are in the repository. You generate them, and the engine will not build without
them.

=== "Windows"

    ```
    cd RoyaleSim
    ..\.venv\Scripts\python tools\extract_arena.py
    ..\.venv\Scripts\python tools\extract_cards.py --vintage 2018
    ..\.venv\Scripts\python tools\extract_cards.py --vintage 2018 --out data\derived\cards.json
    ..\.venv\Scripts\python tools\extract_globals.py
    cd ..
    ```

=== "macOS and Linux"

    ```
    cd RoyaleSim
    ../.venv/bin/python tools/extract_arena.py
    ../.venv/bin/python tools/extract_cards.py --vintage 2018
    ../.venv/bin/python tools/extract_cards.py --vintage 2018 --out data/derived/cards.json
    ../.venv/bin/python tools/extract_globals.py
    cd ..
    ```

That writes `RoyaleSim/data/derived/`, which is the arena, the card table and the game's global
constants.

### Why `--vintage 2018`, and why it is not optional

`extract_cards.py` defaults to a newer card table that is built from a client asset pack the
project is not allowed to redistribute. A public clone does not have that pack, so the command
with no flag fails. This is the real message you get:

```
missing .../data/raw/cr-15.535.29/csv_logic: decode the 15.535.29 assets first
```

That is not a broken install. It is the tool telling you it needs files that are not there and
will not be there. `--vintage 2018` builds the card table from the 2018 files instead, and those
**are** in the repository, 24 tracked files, along with the calibration data. So the 2018 path is
self contained and it is the one that works for everybody.

### Why `extract_cards.py` runs twice

The two runs build the same card table and write it to two different names, because two different
things read it by name.

| The run | Writes | Read by |
|---|---|---|
| `--vintage 2018` | `data/derived/cards-2018.json` | one of the engine's own Rust tests, which loads it by that exact filename |
| `--vintage 2018 --out data\derived\cards.json` | `data/derived/cards.json` | the engine itself, every time it starts |

Leave either one out and something later goes looking for a file that is not there.

### The order matters: extract first, build second

!!! danger "The card table is fixed when the engine is built, not when it runs"

    Pointing the `ROYALESIM_DATA_DIR` environment variable at a different data folder afterwards
    does **not** change the card table a compiled engine holds. This was tried. An engine built
    elsewhere, pointed at a 2018 only checkout, still reported the other card table.

    So: extract, then build. And if you ever regenerate the data, build again. Otherwise you get
    an engine quietly disagreeing with the files next to it, and card counts that change for no
    visible reason.

RoyaleGym helps you here. `RustEngine()` refuses an engine build that is older than the data files
on disk, rather than running with a mismatch.

## Step 3: build the engine

!!! warning "UNVERIFIED"

    Nobody has run this from a clean install yet. Only a debug build has been done from a clean
    clone, and that took 36 seconds on a machine with about 1.5 GB of memory free. The release
    build's time and memory are unmeasured, and so is whether `import royalesim` works
    immediately after it with no further step.

=== "Windows"

    ```
    cd RoyaleSim
    ..\.venv\Scripts\maturin develop --release
    cd ..
    ```

=== "macOS and Linux"

    ```
    cd RoyaleSim
    ../.venv/bin/maturin develop --release
    cd ..
    ```

This compiles the Rust engine and installs it into the shared virtual environment as a Python
module called `royalesim`. It is a first build of a Rust project, so expect it to take a while
and to use a good chunk of memory. If your machine is short on memory, drop `--release` and take
a slower engine for now. The debug build is the one that has actually been done from a clean
clone.

!!! warning "One hazard if you keep more than one checkout"

    `maturin develop` installs the engine **into the virtual environment you run it from**. If two
    checkouts share one virtual environment, building from the second one swaps the engine out
    from under the first. The symptom is nasty: card counts change mid session and nothing tells
    you why.

    For a second checkout, build a wheel instead and install it somewhere of its own:

    === "Windows"

        ```
        cd RoyaleSim
        ..\.venv\Scripts\maturin build --release
        cd ..
        ```

    === "macOS and Linux"

        ```
        cd RoyaleSim
        ../.venv/bin/maturin build --release
        cd ..
        ```

    Then `pip install` the wheel it produces into a throwaway virtual environment. One engine per
    environment, always.

## Step 4: install the Python packages

=== "Windows"

    ```
    .venv\Scripts\python -m pip install -e RoyaleGym
    .venv\Scripts\python -m pip install -e RoyaleViser
    .venv\Scripts\python -m pip install -e RoyaleLearn
    .venv\Scripts\python -m pip install -e "RoyaleLearn[torch]"
    ```

=== "macOS and Linux"

    ```
    .venv/bin/python -m pip install -e RoyaleGym
    .venv/bin/python -m pip install -e RoyaleViser
    .venv/bin/python -m pip install -e RoyaleLearn
    .venv/bin/python -m pip install -e "RoyaleLearn[torch]"
    ```

You can stop after the RoyaleGym line. RoyaleViser is the viewer and it is optional, though it is
small and it needs no engine build of its own. RoyaleLearn is the training harness. It trains,
and it needs the torch extra, which is a large download. Leave it until you want it.

## If the build will not work

You can still start. `royalegym` imports fine without the compiled engine.

`RustEngine()` then raises an `ImportError` whose message names the build command you need to run,
so you are not left guessing. And `MockEngine`, a plain Python stand in, runs the whole API in the
meantime. This program needs Step 2 and Step 4, and does not need Step 3:

```python
import numpy as np
from royalegym import ClashParallelEnv, MockEngine, RandomLegalOpponent

engine = MockEngine()                       # no compiled engine needed
print(len(engine.cards()), "cards in the stand-in")

env = ClashParallelEnv(engine=engine)
obs, info = env.reset(seed=0)
rng, policy = np.random.default_rng(0), RandomLegalOpponent(noop_prob=0.7)

while env.agents:
    actions = {a: policy.act(obs[a], obs[a]["action_mask"], rng) for a in env.agents}
    obs, reward, terminated, truncated, info = env.step(actions)

s = env.battle_state
print(f"winner {s.winner}  crowns {[p.crowns for p in s.players]}  tick {s.tick}")
```

```
16 cards in the stand-in
winner 0  crowns [3, 1]  tick 4255
```

Player 0 took all three towers, which ends a match instantly, and tick 4255 is past the three
minute mark, so that one went into overtime.

Be clear about what you are using, though. `MockEngine` is a stand in, not a second simulator. It
carries 16 cards rather than the full catalogue, spells resolve instantly, there are no stuns or
knockbacks, and cards run at their base level. It is there so you can write and debug your code.
It is not there to train against.

## Did it work?

Four checks, from easiest to slowest. Each output below is what these commands actually printed.

### The engine imports

=== "Windows"

    ```
    cd RoyaleSim
    ..\.venv\Scripts\python -c "import royalesim; print('engine imported, one arena tile is', royalesim.SUBTILE, 'subtiles')"
    ```

=== "macOS and Linux"

    ```
    cd RoyaleSim
    ../.venv/bin/python -c "import royalesim; print('engine imported, one arena tile is', royalesim.SUBTILE, 'subtiles')"
    ```

```
engine imported, one arena tile is 18000 subtiles
```

If that prints, Step 3 worked. A subtile is the engine's unit of position, and there are 18000 of
them to one arena tile, which is why positions come back as large whole numbers.

### Watch a battle and check five things about it

```
cd RoyaleSim
..\.venv\Scripts\python tools\watch_battle.py
```

```
winner: RED
crowns: [2, 3]
final_tick: 2968
troops: 46
[OK ] determinism: re-simulated hash-for-hash on a fresh engine
[OK ] vacuity: 2969 frames (floor 50); blue accepted 18 deploys; red accepted 17 deploys; ...
[OK ] arena: trace grid == data/derived/arena.json (64x36 half-cells)
[OK ] dry: 74 of 27489 ground entity positions on water (the live game allows it; bound 5 %)
[OK ] render: battle.html, 1841257 bytes, 2969 frames, self-contained
EVERY GATE GREEN.
```

That took about two seconds: a whole match and five checks on it.

**Do not expect the winner and the counts to match.** The tool deals each side a random deck out
of the card table your clone built, so a `--vintage 2018` install plays a different battle from
the one above, which used the 15.535 table. `--seed` defaults to 1, so the same checkout does
repeat the same battle, and `--seed 7` gives you another.

**The five `[OK ]` lines are the part that should be green on any install.** That is what to
check. In order: the battle replayed on a fresh engine and every board
hash came back identical, both sides actually deployed and fought rather than standing still, the
arena the battle was played on matches the arena file, ground units stayed out of the water, and
the HTML page it wrote holds every frame and needs nothing else to open.

Add `--seed 7 --open` to fix the battle and open that page in your browser. You can scrub through
it tick by tick.

### The engine's own Python tests

```
cd RoyaleSim
..\.venv\Scripts\python -m pytest -q
```

```
108 passed in 134.27s (0:02:14)
```

Give this one a couple of minutes.

### How fast is it on your machine

```
cd RoyaleGym
..\.venv\Scripts\python -m pytest -q tests/test_rust_engine.py -k throughput -s
```

```
[throughput] env.step/s  rust 957 (9 live)  mock 608 (10 live)  at 10 ticks/step | engine ticks/s incl. state+mask per 20 ticks: rust 20219  mock 11126
1 passed, 77 deselected in 1.66s
```

Read that as: 957 env steps per second, which is 957 decisions per player per second, with the
Rust engine. That run had five other jobs going on the machine.

Your number will be different, and the spread is much wider than you would expect. The same
command on this one laptop has printed anywhere from about 350 to about 960 env steps per
second on the same day, depending on what else was running. Do not read anything into the
figure itself. The thing to check is that the line prints at all and that `rust` is not
dramatically below `mock`, which would mean you are running a debug build.

The last figure is worth knowing about. 20219 engine ticks per second is also roughly 20,219
battles per hour for one process, and that is exact arithmetic rather than a coincidence: a three
minute battle is 3600 ticks and an hour is 3600 seconds.

The gap between the two numbers is Python, not the engine. Python builds both players'
observations and their lists of legal moves on every single step. That is a known open item in
RoyaleGym, not a mystery.

### And the whole environment suite, if you want it

```
cd RoyaleGym
..\.venv\Scripts\python -m pytest -q
```

On the clean clones this printed **342 passed, 6 skipped, in 56 s**. Six skips is normal and a
skip is not a failure. If you built the engine, the two Rust backed test files on their own were
**96 passed in 148 s** on that clean clone, on the debug build.

Without the engine built, the Rust backed tests skip instead of failing. An engine build that is
older than the data files fails them instead of skipping, which is the reminder to go back and
redo Step 3.

## Where to go next

[Your first bot](first-bot.md){ .md-button .md-button--primary }
[Back to Start here](index.md){ .md-button }
[Ask in the Discord](https://discord.gg/4D2BS5JBHP){ .md-button }
