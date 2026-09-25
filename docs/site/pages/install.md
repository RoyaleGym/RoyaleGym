# Install

<p align="center">
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="Rust 1.80+" src="https://img.shields.io/badge/rust-1.80+-DEA584?style=flat-square&logo=rust&logoColor=white">
  <img alt="Recipe with a debug build: run from fresh clones, 2026-09-22" src="https://img.shields.io/badge/recipe%2C%20debug%20build-run%20from%20fresh%20clones-2ea043?style=flat-square">
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

!!! success "The recipe works from fresh clones, with a debug build"

    On 2026-09-22 all four repositories were cloned fresh into an empty folder, with no build
    leftovers and no files that git ignores. The recipe below was then run as written, except
    that the engine was built in debug mode (`maturin develop` without `--release`). That build
    took **2 minutes 38 seconds**. RoyaleGym's test suite in that clone gave **385 passed,
    0 skipped** (RoyaleGym at commit `afb6d1e`).

    Nothing skipped, so every test ran. The suite has grown since (492 tests at commit
    `be58cac`), and has not been re-run from a fresh clone.

!!! warning "The release build has not been run from a fresh clone yet"

    `maturin develop --release` is the one step nobody has done end to end from a fresh clone.
    How long it takes there, how much memory it needs, and whether `import royalesim` works
    straight afterwards with no further step, are all unmeasured.
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
    mkdir Royale
    cd Royale
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
    Copy-Item data\derived\cards-15.535.json data\derived\cards.json
    ..\.venv\Scripts\python tools\extract_globals.py
    cd ..
    ```

=== "macOS and Linux"

    ```
    cd RoyaleSim
    ../.venv/bin/python tools/extract_arena.py
    ../.venv/bin/python tools/extract_cards.py --vintage 2018
    cp data/derived/cards-15.535.json data/derived/cards.json
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

That is not a broken install. It is the tool telling you it needs files that are not there,
because they are not distributed. `--vintage 2018` builds the card table from the 2018 files
instead, and those **are** in the repository: 23 tracked files (counted on 2026-09-22), along
with the calibration data. So the 2018 path is self contained and works for everybody. It is not
the table the engine plays, though: the copy line puts the committed 15.535 table in place for
that.

### Why there are two card table lines

The two lines put two different card tables under two different names, because two different
things read them by name.

| The run | Writes | Read by |
|---|---|---|
| `--vintage 2018` | `data/derived/cards-2018.json` | one of the engine's own Rust tests, which loads it by that exact filename |
| `Copy-Item ...cards-15.535.json ...cards.json` | `data/derived/cards.json` | the engine itself, every time you create one |

Leave either one out and something later goes looking for a file that is not there.

### The order matters: extract first, build second

!!! danger "The engine reads its card table from the folder it was built in"

    The build copies the arena and the calibration constants into the engine, so it needs
    `data/derived/arena.json` to exist first. The card table works differently. Every time you
    create an engine, it reads `data/derived/cards.json` from the RoyaleSim folder it was
    **built in**.

    Two things follow. Re-running `extract_cards.py` in that folder changes the cards the
    engine uses, with no rebuild. And pointing the `ROYALESIM_DATA_DIR` environment variable at
    a different data folder does **not** change the card table the engine reads. This was tried.
    An engine built elsewhere, pointed at a 2018 only checkout, still reported the other card
    table.

    So: extract, then build, and build in the checkout whose data you want. If you change the
    calibration file or regenerate the arena, build again.

RoyaleGym helps you here. `RustEngine()` refuses to start if the calibration or arena file on
disk differs from the copy built into the engine, rather than running with a mismatch. The card
table is not part of that check.

## Step 3: build the engine

!!! warning "UNVERIFIED"

    The release build has not been run from a fresh clone yet. A debug build has: on
    2026-09-22 it took 2 minutes 38 seconds, and the tests passed against it. The release
    build's time and memory on a fresh clone are unmeasured, and so is whether
    `import royalesim` works straight after it with no further step.

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
a slower engine for now. On 2026-09-22 a debug engine played battle ticks about 17 times slower
than a release one on the same laptop. The debug build is the one that has actually been done from a fresh
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
of the card table your clone built, and the engine and its card list keep changing, so your
battle can differ from the one above. `--seed` defaults to 1, so the same checkout does
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
179 passed, 10 skipped
```

Give this one a couple of minutes. That is RoyaleSim's PYTHON suite on a fresh clone, measured on
2026-09-22 at commit `2b85ce1`. It is not the Rust suite, which is a separate command with a
separate result, and which is failing today on one test. Read the ten skips rather than ignoring
them: each names the thing it could not find and says a skip is not a pass.

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
command on one laptop has printed anywhere from about 350 to about 960 env steps per
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

On fresh clones with a debug engine build, on 2026-09-22, this printed **385 passed,
0 skipped**. The suite has grown since (492 tests at commit `be58cac`), so expect a bigger
count. Expect some skips. Six tests that compare the two engines skip on purpose, because the compiled
engine and `MockEngine` read different card tables. Others skip if Node.js is not on your PATH or
RoyaleViser is not installed. A clean runner on 2026-09-23 showed nine skips in all. Add `-rs` to read the reason for any skip,
and see [tests that skip](troubleshooting.md#6-tests-that-skip-instead-of-failing).

Without the engine built, the Rust backed tests skip instead of failing. An engine built from a
different calibration or arena file than the one on disk fails them instead of skipping, which
is the reminder to go back and redo Step 3.

## Where to go next

[Your first bot](first-bot.md){ .md-button .md-button--primary }
[Back to Start here](index.md){ .md-button }
[Ask in the Discord](https://discord.gg/4D2BS5JBHP){ .md-button }
