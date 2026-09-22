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

There is a second version of this with the same cause, about the tiles a tower forbids:

```
RuntimeError: the Rust engine and the action mask disagree on troop territory; rebuild with
`maturin develop --release` after regenerating cards.json:
```

**What to do.** Build again. From your `RoyaleSim` folder:

!!! warning "UNVERIFIED"
    Nobody has run this from a clean install yet. The release build has not been timed or run
    end to end by anyone on this project, so treat the minute and the memory figure below as the
    rough shape rather than a measurement.

```
..\.venv\Scripts\maturin develop --release
```

The install notes in the READMEs put that at about a minute and about 1.5 GB of RAM.

Order matters here and it catches people out. Generate the data first, build second. The card
table is fixed when the engine is **built**, not when it is run. Pointing `ROYALESIM_DATA_DIR`
somewhere else afterwards does not change what the compiled engine holds.

## 2. The data files were never generated

**What you see.** A `FileNotFoundError` naming `cards.json`. The message spells out the fix:

```
FileNotFoundError: ...\data\derived\cards.json is absent. In the sibling RoyaleSim checkout run
its README's Setup block, which is:
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

Then build the engine, as in problem 1. Those four commands were run on a clean clone of the
four repos by another session, and RoyaleGym's suite passed against the result: 342 passed, 6
skipped, in 56 s.

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
one needs a client asset pack that is not redistributed. A public clone does not have it and
never will. The 2018 tables ARE tracked in the repo, 24 files of them, so `--vintage 2018` is
self-contained and works everywhere.

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
ImportError: royalesim is not built (No module named 'royalesim'); run `maturin develop --release` in the sibling RoyaleSim checkout (../RoyaleSim) with the workspace venv active (README.md, Setup)
```

**What it means.** Exactly what it says, with one wrong turning in it. The Rust engine was never
built into the venv you are running, or you are running a different Python from the one you built
into. Note that `import royalegym` itself still works. The package is designed to import without
the engine.

!!! note "The message points at the wrong README"
    It ends `(README.md, Setup)`. The error comes from `royalegym`, so you will look in
    RoyaleGym's README, and that section is called **Install**. RoyaleSim's is called Install
    too, and RoyaleSim's is the one you actually want, because the build happens there. The
    only **Setup** heading in the project is in RoyaleViser's README, which is the repo this
    error has least to do with. The message is being corrected.

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

**What to do.** On a normal clone, nothing. Both engines come from the tracked 2018 table there,
the check runs, and it passes. That has been confirmed on a clean clone of all four repos: 96
passed, 0 skipped over the two Rust-backed test files.

If you see skips you did not expect, read the reason before you trust the green. Skips also
happen when the engine is not built at all, which is problem 5, and when a test needs recordings
of real matches, which are private and not distributed.

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

The build hash at the top of this page is the cheap check. Print it at the start of a run and
again at the end. If it changed, something rebuilt the engine underneath you.

## Still stuck

Ask in the project's Discord. Paste the command you ran, the whole error, and the output of the
two lines at the top of this page. Those two say more about a broken install than a paragraph of
description.

<p align="center">
  <a href="https://discord.gg/4D2BS5JBHP"><img alt="Join the Discord" src="https://img.shields.io/badge/Discord-join%20the%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white"></a>
</p>

- [Install](install.md) for the whole setup from an empty folder.
- [How accurate is the engine](accuracy.md) if the engine runs but a battle does not look right.
