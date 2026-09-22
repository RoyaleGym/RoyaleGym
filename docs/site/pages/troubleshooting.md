# Troubleshooting

<p align="center">
  <img alt="Errors covered" src="https://img.shields.io/badge/errors%20covered-7-0b7285?style=flat-square">
  <img alt="Messages" src="https://img.shields.io/badge/messages-quoted%20from%20real%20runs-2ea043?style=flat-square">
  <a href="https://discord.gg/4D2BS5JBHP"><img alt="Discord" src="https://img.shields.io/badge/stuck%3F-ask%20in%20the%20Discord-5865F2?style=flat-square&logo=discord&logoColor=white"></a>
</p>

Seven things go wrong, and they go wrong in roughly this order. Each one below gives the message
you actually see, what it means, and what to type next.

Every message on this page was copied from a real run, except where a line says otherwise.

!!! tip "The two commands that answer most questions"
    Run these from your `RoyaleGym` folder. The first says whether the Rust engine is built and
    importable. The second prints a short hash of the data the engine was built with.

    ```
    ..\.venv\Scripts\python -c "import royalegym; print(royalegym.core_available())"
    ..\.venv\Scripts\python -c "from royalegym import RustEngine; print(RustEngine.build_digest())"
    ```

    `True` from the first one means the engine is there. `False` means skip to problem 5. If the
    second one raises instead of printing a hash, that is problem 1.

## 1. The engine is older than the data files

**What you see.** A `RuntimeError`, the moment you construct `RustEngine()`:

```
RuntimeError: royalesim was built against different data; rebuild with `maturin develop --release`:
  calibration time.TICK_MS: built 50, now 51
```

The last line names what differs. There is one line per difference. You may instead see a line
like `arena.json differs from the one compiled in (...)`.

**What it means.** The engine compiles the constants file and the arena into itself when you
build it. So the build carries a copy. If you edit `data/calibration.json`, or regenerate
`data/derived/arena.json`, the copy inside the engine and the file on disk stop agreeing.
RoyaleGym refuses to start rather than run a battle whose rules are half old and half new.

There is a second error that looks like this one, about the tiles a tower forbids:

```
RuntimeError: the Rust engine and the action mask disagree on troop territory; rebuild with
`maturin develop --release` after regenerating cards.json:
```

That one has a different cause. The engine and RoyaleGym are reading two different
`cards.json` files. The engine reads the one in the RoyaleSim folder it was built in. RoyaleGym
reads the one under `ROYALESIM_DATA_DIR`, or under `../RoyaleSim/data` when that is not set. Make
them the same file: point `ROYALESIM_DATA_DIR` at the data folder of the checkout the engine was
built in, or build the engine in the checkout whose data you want.

**What to do.** For the first error, build again. From your `RoyaleSim` folder:

!!! warning "UNVERIFIED"
    The release build has not been timed from a fresh clone yet, so there is no measured figure
    for how long it takes or how much memory it needs there.

```
..\.venv\Scripts\maturin develop --release
```

The READMEs say to give it a few minutes and some free memory. A debug build from fresh clones
took 2 minutes 38 seconds on 2026-09-22.

Order matters here and it catches people out. Generate the data first, build second, because
the build copies the arena file into the engine. The card table is different. Every time you
create an engine, it reads `data/derived/cards.json` from the RoyaleSim folder it was **built
in**. Re-running `extract_cards.py` there changes the cards with no rebuild. Pointing
`ROYALESIM_DATA_DIR` somewhere else does not change which card table the engine reads.

## 2. The data files were never generated

**What you see.** A `FileNotFoundError` naming `cards.json`. The message spells out the fix:

```
FileNotFoundError: ...\data\derived\cards.json is absent. In the sibling RoyaleSim checkout run
the commands from its README's Install section, which are:
    python tools/extract_cards.py --vintage 2018
    python tools/extract_cards.py --vintage 2018 --out data/derived/cards.json
--vintage 2018 is not optional on a public clone: without it the extractor wants a client asset
pack that is not redistributed, and fails. (data dir: ROYALESIM_DATA_DIR or ...\RoyaleSim\data)
```

The two `...` are the full folder path on your machine. Everything else is the real message.

**What it means.** `data/derived/` is not in the repo. A fresh clone has no `cards.json`, no
`arena.json` and no `globals.json`. Generating them is a mandatory install step, not an optional
one. This was checked on clean clones of all four repos.

**What to do.** Run all four generator commands, from your `RoyaleSim` folder:

```
..\.venv\Scripts\python tools\extract_arena.py
..\.venv\Scripts\python tools\extract_cards.py --vintage 2018
..\.venv\Scripts\python tools\extract_cards.py --vintage 2018 --out data\derived\cards.json
..\.venv\Scripts\python tools\extract_globals.py
```

Then build the engine, as in problem 1. These commands were run on fresh clones of the four
repos on 2026-09-22, followed by a debug build, and RoyaleGym's suite passed against the result:
385 passed, 0 skipped.

!!! warning "Keep `--vintage 2018` on BOTH `extract_cards.py` lines"
    They are not a typo of each other. One writes `data/derived/cards-2018.json`, which a Rust
    test loads by that name. The other writes the same table over `data/derived/cards.json`,
    which is the file the engine loads. You want both.

## 3. `extract_cards.py` fails with no `--vintage`

**What you see.** Running the extractor with no flag, on a clone:

```
missing .../data/raw/cr-15.535.29/csv_logic: decode the 15.535.29 assets first
```

The `...` is your folder path.

**What it means.** With no flag the extractor builds the newer of the two card tables, and that
one needs a client asset pack that is not redistributed. A public clone does not have it. The
2018 tables ARE tracked in the repo, 23 files of them (counted on 2026-09-22), so
`--vintage 2018` is self-contained and works everywhere.

**What to do.** Add `--vintage 2018`, as in problem 2.

**What you give up.** A 2018 table runs the engine, the examples and the Python suite. Three
checks want the newer table specifically and cannot run without it: `tests/levels.rs`,
`tests/jump16402.rs`, and the live-level rows of `tools/check_data.py`. Those are engine checks.
Nothing you need for writing a bot depends on them.

## 4. The viewer shows nothing

**What you see.** RoyaleViser opens, and stays empty. Your training loop or your env script is
clearly running.

**What it means.** Almost always, nothing is publishing. The correction that catches most people:

!!! warning "A plain `ClashParallelEnv` does not read `ROYALEVISER`"
    Setting the `ROYALEVISER` environment variable and constructing `ClashParallelEnv` on its own
    does nothing at all. Only `ClashSelfPlayVecEnv` reads that variable. A single env publishes
    only when you hand it a publisher.

    This was checked. With `ROYALEVISER=127.0.0.1:9870` set,
    `ClashParallelEnv(engine=RustEngine()).viser` is `None`, while
    `ViserPublisher.from_env()` returns a publisher.

**What to do.** For one env, hand it a publisher yourself:

```python
from royalegym import ClashParallelEnv, RustEngine
from royalegym.viser import ViserPublisher

env = ClashParallelEnv(engine=RustEngine(), viser=ViserPublisher())   # 127.0.0.1:9870
print("publisher:", env.viser is not None, "attached:", env.viser.attached)
```

```
publisher: True attached: False
```

For self-play, set `ROYALEVISER=host:port` and build `ClashSelfPlayVecEnv(8)`. It binds one
publisher and gives it to game 0. That is deliberate. A viewer watches one battle and holds one
port, so eight envs each grabbing that port is an address-already-in-use error.

Then start the viewer in another terminal:

!!! warning "UNVERIFIED"
    Nobody has run this from a clean install yet. The `--stream` flag is real and appears in the
    viewer's own `--help`, but no window was opened while writing this page.

```
..\.venv\Scripts\python -m royaleviser --stream 127.0.0.1:9870
```

Two more reasons a viewer stays empty, in the order worth checking:

- `attached: False` above is normal before the viewer starts. The publisher sends nothing until
  a viewer says hello, and stops again about three seconds after it goes quiet. So start the
  viewer, then look again.
- With self-play, only game 0 is published. Games 1 to 7 are invisible on purpose.

## 5. `ImportError` when you construct `RustEngine()`

**What you see.**

```
ImportError: royalesim is not built (No module named 'royalesim'); run `maturin develop --release` in the sibling RoyaleSim checkout (../RoyaleSim) with the workspace venv active. The card data has to be extracted BEFORE that build; the "Install" section of the RoyaleGym README.md has both steps in order.
```

**What it means.** Exactly what it says, with one wrong turning in it. The Rust engine was never
built into the venv you are running, or you are running a different Python from the one you built
into. Note that `import royalegym` itself still works. The package is designed to import without
the engine.

!!! tip "Read the middle clause, not just the command"
    The part people skip is that the data has to be generated **before** the build. The build
    copies `arena.json` into the engine, so on a clone with no generated data it stops. With
    the arena generated but not the cards, the build works and the engine fails later, when it
    looks for `cards.json`. Either way, run the four commands in problem 2 first, then build.

**What to do.** Either build it, which is the `maturin develop --release` line in problem 1, or
carry on without it for now:

```python
from royalegym import ClashParallelEnv, MockEngine

env = ClashParallelEnv(engine=MockEngine())
```

`MockEngine` is a plain Python stand-in. It runs the whole API, so you can write and debug a
reward function against it. It is not a second simulator: spells resolve instantly, there are no
stuns or knockbacks, and cards run at their base level. Anything you conclude about game
behaviour from `MockEngine` is about `MockEngine`.

If you are unsure which of the two you have, run the `core_available()` line at the top of this
page.

## 6. Tests that skip instead of failing

**What you see.** Green output with an `s` in it, and no failures:

```
..\.venv\Scripts\python -m pytest -q tests/test_rust_engine.py -k test_mock_and_rust_agree_on_setup_state -rs
```

```
sssss                                                                    [100%]
=========================== short test summary info ===========================
SKIPPED [5] tests\test_rust_engine.py:562: the two engines are reading different card tables, so this comparison would measure the DATA and not the engines. A SKIP IS NOT A PASS -- to run it, use a checkout with no private client pack AND BUILD royalesim IN IT: the compiled engine carries the table it was built with, so changing the data on disk alone moves only MockEngine's half and these will still skip. cards.json vintage '15.535.29 client (2026, LIVE build family)' vs MockEngine's 'retroroyale-2018'. Differences (rust/mock) -- count: Goblins 4/3
5 skipped, 73 deselected in 2.13s
```

**What it means.** A skip is not a pass. The check above compares the two engines against each
other. It can only say something about the ENGINES when both are reading the same card table. On
a machine where they are not, it would be measuring the difference between two card tables
instead, so it steps aside and says why, naming both tables.

Add `-rs` to any pytest run to see the reason for every skip. Without it you get a letter.

!!! info "What the two engines actually disagree about"
    `MockEngine` is an independent Python reading of the same card data, not a copy of the Rust
    engine, which is what makes comparing them worth anything. On the tracked 2018 table they
    agree across the measured set.

    The one difference this check reports on a machine holding a private client pack is the
    **unit count of Goblins: 4 in the newer table, 3 in the tracked one.** That is a difference
    between the two card tables rather than between the two engines, which is exactly why the
    comparison steps aside instead of failing.

    It is worth knowing before you meet it. If you compare a Goblins battle across the two
    engines on a machine like that, the unit counts will not line up, and nothing is broken.
    The example deck used elsewhere in these docs is drawn entirely from the 18 cards whose
    behaviour is checked against recordings, and Goblins is not one of them.

**What to do.** On a normal clone, nothing. Both engines come from the tracked 2018 table there,
the check runs, and it passes. That was confirmed on fresh clones of all four repos on
2026-09-22: RoyaleGym's whole suite gave 385 passed, 0 skipped, on a debug engine build.

The project's own machine does hold a newer card table, so six tests skip there in exactly this
way. If you see a result with 6 skipped quoted somewhere in the docs, that is where it came from.

If you see skips you did not expect, read the reason before you trust the green. Skips also
happen when the engine is not built at all, which is problem 5. A few tests skip when Node.js is
not on your PATH or RoyaleViser is not installed. And some skip when a test needs recordings of
real matches, which are private and not distributed.

## 7. One venv, two checkouts, and the card count changes under you

**What you see.** Numbers that move for no reason between runs. The clearest symptom is the size
of the card catalogue changing:

```
..\.venv\Scripts\python -c "from royalegym import RustEngine; print(len(RustEngine().cards()))"
```

Run that, work in a second checkout for an hour, run it again, and get a different number.

**What it means.** `maturin develop` installs the engine INTO the venv it is run from. If two
checkouts share one venv, building in the second one swaps the engine out from under the first.
Everything still imports. Everything still runs. It is just a different engine now, built from
different data, and nothing announces it.

This is the worst failure on the page, because it does not look like a failure.

**What to do.** Keep one venv per checkout, or, for the second checkout, build a wheel and put it
in a throwaway venv instead of installing into the shared one:

!!! warning "UNVERIFIED"
    Nobody has run this from a clean install yet. The wheel build is the recommended workaround,
    not something this project has timed.

```
..\.venv\Scripts\maturin build --release
```

The build hash at the top of this page is the cheap check, but it only covers the calibration
and arena built into the engine. It does not cover the card table. So print the hash and the
card count above at the start of a run, and again at the end. If either changed, something
changed underneath you.

## Still stuck

Ask in the project's Discord. Paste the command you ran, the whole error, and the output of the
two lines at the top of this page. Those two say more about a broken install than a paragraph of
description.

<p align="center">
  <a href="https://discord.gg/4D2BS5JBHP"><img alt="Join the Discord" src="https://img.shields.io/badge/Discord-join%20the%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white"></a>
</p>

- [Install](install.md) for the whole setup from an empty folder.
- [How accurate is the engine](accuracy.md) if the engine runs but a battle does not look right.
