# Build from source

Most people never need this page: `pip install "royalegym[all]"` ([Install](install.md)) gives you
everything, prebuilt. This page is for working on the engine itself, or on more than one of the
repositories at once.

<p align="center">
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="Rust 1.80+" src="https://img.shields.io/badge/rust-1.80+-DEA584?style=flat-square&logo=rust&logoColor=white">
  <img alt="Recipe with a debug build: run from fresh clones, 2026-09-22" src="https://img.shields.io/badge/recipe%2C%20debug%20build-run%20from%20fresh%20clones-2ea043?style=flat-square">
  <img alt="Release build: run from fresh clones on a 4-CPU Linux machine on 2026-09-27, 163 seconds and about 1 GB of memory" src="https://img.shields.io/badge/release%20build-163%20s%20on%204--CPU%20Linux%2C%202026--09--27-2ea043?style=flat-square">
  <img alt="Tested on Windows and on Linux" src="https://img.shields.io/badge/commands%20tested%20on-Windows%20and%20Linux-0078D4?style=flat-square">
</p>

At the end of this page you have four repositories side by side in one folder, one Python virtual
environment shared by all of them, and a compiled engine you can run battles in. Then you have a
short list of commands to check that it worked, with the output you should see.

## What you need first

- **Python 3.12 or newer.** Older versions are not supported. An older Python compiles the
  engine for several minutes and is only refused at the install step, so Step 1 checks the
  version first.
- **Rust 1.80 or newer, with cargo.** Get it from [rustup.rs](https://rustup.rs). This is only
  needed to build the engine. If you cannot get Rust working, read
  [If the build will not work](#if-the-build-will-not-work) at the bottom. You can still start.
- **git.**

## How much of this page has actually been run

This matters more than it sounds, so it is near the top rather than in a footnote.

!!! success "The recipe works from fresh clones, with a release build"

    On 2026-09-27 the Linux commands on this page were run from fresh clones on a 4-CPU Linux
    machine, with a Python 3.12 venv. The release build, `maturin develop --release`, took
    **163 seconds** and about **1 GB** of memory, and `import royalesim` worked afterwards with
    no extra step.

    RoyaleGym's CI repeats Steps 1 to 4 on Linux from fresh clones on every push, for RoyaleSim,
    RoyaleGym and RoyaleViser, with Python 3.12. Then it runs the pytest suite. Run 36347317029,
    at RoyaleGym commit `ecbb80b` on 2026-09-27, gave **1028 passed, 9 skipped, 1 xfailed**,
    nothing failing.

    Earlier, on 2026-09-22, the recipe was run on Windows with a debug build (`maturin develop`
    without `--release`). That build took 2 minutes 38 seconds.

Everything below that has not been run carries a box saying so.

## Step 1: the folder, the clones and the virtual environment

Make one folder called `Royale`, put all four repositories inside it, and make one virtual
environment at the top. The repositories find each other by sitting next to each other, so the
layout is not optional.

!!! warning "UNVERIFIED: the macOS tabs, and two newer lines"

    The Linux tabs were run from fresh clones on a 4-CPU Linux machine on 2026-09-27. Nobody has
    typed them on a Mac yet. They are the Windows commands with `.venv/bin/python` in place of
    `.venv\Scripts\python`, forward slashes in the paths, `cp` in place of `Copy-Item`, and
    `export X=Y` where Windows sets a variable.

    Two lines are newer than any run from a fresh clone: `py -3.12 -m venv .venv` on Windows,
    and the CPU torch line in Step 4.

=== "Windows"

    ```
    mkdir Royale
    cd Royale
    git clone https://github.com/RoyaleGym/RoyaleSim.git
    git clone https://github.com/RoyaleGym/RoyaleGym.git
    git clone https://github.com/RoyaleGym/RoyaleViser.git
    git clone https://github.com/RoyaleGym/RoyaleLearn.git
    py -3.12 -m venv .venv
    .venv\Scripts\python --version
    .venv\Scripts\python -m pip install maturin pytest hypothesis ruff numpy msgspec
    ```

=== "macOS and Linux"

    ```
    mkdir Royale && cd Royale
    git clone https://github.com/RoyaleGym/RoyaleSim.git
    git clone https://github.com/RoyaleGym/RoyaleGym.git
    git clone https://github.com/RoyaleGym/RoyaleViser.git
    git clone https://github.com/RoyaleGym/RoyaleLearn.git
    python3.12 -m venv .venv
    .venv/bin/python --version
    .venv/bin/python -m pip install maturin pytest hypothesis ruff numpy msgspec
    ```

The `--version` line must print 3.12 or newer before you build. An older Python compiles the
engine for several minutes and is only refused at the install step. If `python --version`
already prints 3.12 or newer, `python -m venv .venv` works too.

Optional: [RoyaleImitate](https://github.com/RoyaleGym/RoyaleImitate) adds imitation learning to
RoyaleLearn. Its README has its own two install lines.

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

A clone carries only `cards-15.535.json` in `data/derived/`; these lines generate the rest,
including the arena. The folders under `data/raw/cr-*/` are not in the repository either. The
engine will not build without the generated files.

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

!!! success "Run from fresh clones"

    On a 4-CPU Linux machine on 2026-09-27 the release build took 163 seconds and about 1 GB of
    memory. It can go quiet for a minute or more on the engine itself; let it finish. A debug
    build on Windows, on 2026-09-22, took 2 minutes 38 seconds.

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
than a release one on the same laptop.

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
    ```

=== "macOS and Linux"

    ```
    .venv/bin/python -m pip install -e RoyaleGym
    .venv/bin/python -m pip install -e RoyaleViser
    .venv/bin/python -m pip install -e RoyaleLearn
    ```

You can stop after the RoyaleGym line. RoyaleViser is the viewer and it is optional, though it is
small and it needs no engine build of its own. RoyaleLearn is the training harness. It trains,
and it needs the torch extra, which is a large download. Leave it until you want it.

### Optional: torch, for training

No NVIDIA GPU? Install CPU torch first. On a 4-CPU Linux machine with no GPU, on 2026-09-27, the
default torch download pulled about 5 GB of CUDA wheels.

=== "Windows"

    ```
    .venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    ```

=== "macOS and Linux"

    ```
    .venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    ```

Then install the torch extra:

=== "Windows"

    ```
    .venv\Scripts\python -m pip install -e "RoyaleLearn[torch]"
    ```

=== "macOS and Linux"

    ```
    .venv/bin/python -m pip install -e "RoyaleLearn[torch]"
    ```

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
winner 0  crowns [3, 1]  tick 5585
```

Each player took one tower in the three minutes, so the match went to overtime, and player 0 won
it by taking the king tower at tick 5585, a minute and a half into overtime. A king tower is worth
all three crowns. `MockEngine` reads RoyaleSim's calibration file each time it starts, so this
result moves when that file does. It was run on 2026-09-29 against the calibration in RoyaleSim
`1e843ea`, the one with triple elixir late in overtime.

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
    cd ..
    ```

=== "macOS and Linux"

    ```
    cd RoyaleSim
    ../.venv/bin/python -c "import royalesim; print('engine imported, one arena tile is', royalesim.SUBTILE, 'subtiles')"
    cd ..
    ```

```
engine imported, one arena tile is 18000 subtiles
```

If that prints, Step 3 worked. A subtile is the engine's unit of position, and there are 18000 of
them to one arena tile, which is why positions come back as large whole numbers.

### Watch a battle and check five things about it

=== "Windows"

    ```
    cd RoyaleSim
    ..\.venv\Scripts\python tools\watch_battle.py
    cd ..
    ```

=== "macOS and Linux"

    ```
    cd RoyaleSim
    ../.venv/bin/python tools/watch_battle.py
    cd ..
    ```

This is what it printed on 2026-09-27, at RoyaleSim `1d661b0` and RoyaleGym `ecbb80b`, on
Windows. The path on the last line is your own RoyaleSim folder; it is shortened here:

```
BASELINE (rust engine)
    engine: rust
    seed: 1
    steps: 331
    decision_ms: 500
    noop_prob: 0.55
    accepted: {'blue': 24, 'red': 30}
    rejected: 0
    winner: BLUE
    crowns: [3, 1]
    final_tick: 3306
    troops: 89
    hp_drops: 1733
    max_displacement_subtiles: 549000
    ground_entity_positions_checked: 30356
    ground_entity_positions_wet: 46
  [OK ] determinism: re-simulated hash-for-hash on a fresh engine
  [OK ] vacuity: 3307 frames (floor 50); blue accepted 24 deploys; red accepted 30 deploys; 89 troops existed (floor 4); largest displacement 549000 subtiles; 1733 hp drops
  [OK ] arena: trace grid == data/derived/arena.json (64x36 half-cells)
  [OK ] dry: 46 of 30356 ground entity positions on water (the live game allows it; bound 5 %)
  [OK ] render: battle.html, 3491051 bytes, 3307 frames, self-contained

EVERY GATE GREEN. Open C:\...\Royale\RoyaleSim\battle.html in a browser to watch it.
```

That took about three seconds on one desktop: a whole match and five checks on it.

**Do not expect the winner and the counts to match.** The tool deals each side a random deck out
of the card table your clone built, and the engine and its card list keep changing, so your
battle can differ from the one above. `--seed` defaults to 1, so the same checkout does
repeat the same battle, and `--seed 7` gives you another.

`--seed 7` printed `winner: NONE` at that commit, and that is not a fault. The tool stops after
360 decisions unless you pass `--steps`. That is tick 3600, the end of the three regular minutes.
The seed 7 battle was still level then, so it was cut off before overtime and has no winner.
With `--steps 480` it plays on into overtime, and Blue won it at tick 3626.

**The five `[OK ]` lines are the part that should be green on any install.** That is what to
check. In order: the battle replayed on a fresh engine and every board
hash came back identical, both sides actually deployed and fought rather than standing still, the
arena the battle was played on matches the arena file, ground units stayed out of the water, and
the HTML page it wrote holds every frame and needs nothing else to open.

Add `--open` to open that page in your browser when every gate is green. You can scrub through it
tick by tick.

### The engine's own Python tests

=== "Windows"

    ```
    cd RoyaleSim
    ..\.venv\Scripts\python -m pytest -q
    cd ..
    ```

=== "macOS and Linux"

    ```
    cd RoyaleSim
    ../.venv/bin/python -m pytest -q
    cd ..
    ```

```
846 passed, 35 skipped, 7 xfailed in 140.64s (0:02:20)
```

Give this one a couple of minutes. That is RoyaleSim's PYTHON suite at RoyaleSim commit
`1d661b0`, from RoyaleSim CI run 36334969584 on 2026-09-27. It printed those counts on both
GitHub's Linux and Windows runners; the line above is the Linux one. Your counts can differ a
little. What matters is that nothing says `failed`. Read the skips rather than ignoring them:
most of them (27 of the 35 in that run) need the game's asset pack, which is not distributed,
and each one names what it could not find.

The Rust suite is a separate command with its own result, and it takes tens of minutes. In the
same CI run every Rust test binary reported 0 failed. On a 4-CPU Linux machine on 2026-09-27 the
Rust suite gave 737 passed, 0 failed, 3 ignored.

### How fast is it on your machine

=== "Windows"

    ```
    cd RoyaleGym
    ..\.venv\Scripts\python -m pytest -q tests/test_rust_engine.py -k throughput -s
    cd ..
    ```

=== "macOS and Linux"

    ```
    cd RoyaleGym
    ../.venv/bin/python -m pytest -q tests/test_rust_engine.py -k throughput -s
    cd ..
    ```

```
[throughput] env.step/s  rust 1151 (12 live)  mock 835 (7 live)  at 10 ticks/step | engine ticks/s incl. state+mask per 20 ticks: rust 26551  mock 17794
1 passed, 100 deselected in 1.78s
```

Read that as: 1151 env steps per second, which is 1151 decisions per player per second, with the
Rust engine. The number in brackets is how many towers, units and buildings were on the board
when the timing stopped. That run was on a Windows desktop with 12 logical processors and other
jobs going, on 2026-09-27.

Your number will be different. Three runs in a row on that desktop printed 987, 1151 and 1412 for
`rust`. On a 4-CPU Linux machine on 2026-09-27, six runs printed 1477 to 1612. The report times
each engine once, one after the other, so the gap between `rust` and `mock` moves from run to run
too. Do not read anything into the figure itself. The thing to check is that the line prints at
all and that `rust` is not dramatically below `mock`, which would mean you are running a debug
build.

The last figure is worth knowing about. Engine ticks per second is also roughly battles per hour
for one process, and that is exact arithmetic rather than a coincidence: a three minute battle is
3600 ticks and an hour is 3600 seconds.

The gap between the two numbers is Python, not the engine. Python builds both players'
observations and their lists of legal moves on every single step. That is a known open item in
RoyaleGym, not a mystery.

### And the whole environment suite, if you want it

=== "Windows"

    ```
    cd RoyaleGym
    ..\.venv\Scripts\python -m pytest -q
    cd ..
    ```

=== "macOS and Linux"

    ```
    cd RoyaleGym
    ../.venv/bin/python -m pytest -q
    cd ..
    ```

RoyaleGym's CI run 36347317029, at commit `ecbb80b` on 2026-09-27, printed **1028 passed,
9 skipped, 1 xfailed** for this pytest suite, in 8 minutes 38 seconds on GitHub's Linux runner.
On one desktop the suite takes about 10 to 12 minutes with nothing else running, and 20 minutes
or more beside other jobs.

Expect those nine skips. The summary at the end of the run gives each one's reason:

- Six tests that compare the compiled engine with `MockEngine` skip on purpose, because the two
  read different card tables.
- One test states that the two engines model a building's footprint differently.
- Two tests check a test count written in the docs, and each can only check it at the commit it
  names.

More skip if Node.js is not on your PATH, if RoyaleViser is not installed, if the other repos
are not next to this one, or if something already holds port 9870. See
[tests that skip](troubleshooting.md#6-tests-that-skip-instead-of-failing).

Without the engine built, the Rust backed tests skip instead of failing. An engine built from a
different calibration or arena file than the one on disk fails them instead of skipping, which
is the reminder to go back and redo Step 3.

## Where to go next

[Your first bot](quickstart.md){ .md-button .md-button--primary }
[Back to Start here](index.md){ .md-button }
[Ask in the Discord](https://discord.gg/4D2BS5JBHP){ .md-button }
