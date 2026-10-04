# AGENTS.md: RoyaleGym for AI coding agents

This file is for AI agents working in this repository. People should start at the [README](README.md).

## What this is

RoyaleGym turns Clash Royale battles into reinforcement-learning environments: a PettingZoo parallel
environment (`ClashParallelEnv`) and a Gymnasium wrapper (`ClashGymEnv`). Battles run in RoyaleSim, a
Rust engine; this package adds observations, actions and masks, rewards, battle setups, opponents and
replays. RoyaleLearn trains on it, and RoyaleViser draws its battles.

## Layout

| Path | What it holds |
|---|---|
| `royalegym/protocol.py` | The engine interface: `BattleState`, `PlayerState`, `EntityState` and friends (msgspec structs), `MatchSetup` and its rules, the status bits |
| `royalegym/rust_engine.py` | `RustEngine`, the adapter over RoyaleSim's compiled `Battle`, and the column-order checks |
| `royalegym/mock_engine.py` | `MockEngine`, a pure-Python stand-in that plays the 2018 card tables |
| `royalegym/env.py` | `ClashParallelEnv`, `ClashGymEnv` and `make_env` |
| `royalegym/obs.py` | Observation builders (`SpatialObsBuilder`, `EntityListObsBuilder`), `MatchMemory` (what a player can count), `Reveal` |
| `royalegym/action.py` | Action parsers and action masks |
| `royalegym/reward.py`, `done_condition.py`, `state_mutator.py` | Rewards, end conditions, and how each battle begins |
| `royalegym/opponents.py`, `selfplay.py`, `evaluate.py` | Scripted opponents, self-play and an opponent pool, and "is bot A better than bot B" |
| `royalegym/replay.py`, `render.py`, `viser.py` | Battle traces (record, load, verify), the HTML replay page, and the feed to RoyaleViser |
| `royalegym/landing.py` | Where a deploy actually put something, as opposed to where it was tapped |
| `tests/` | The suite |
| `examples/` | Runnable examples, `quickstart.py` among them |
| `tools/` | `stage_release.py` (the release page's wheels) and `fresh_user_test.py` (an install from nothing) |
| `docs/` | `architecture.md`, `observation-spec.md`, `guide.md`, and `docs/site/`, the website |

## Build and test

These are CI's own commands (`.github/workflows/suite.yml`). The repositories sit side by side in one
folder with one shared venv: RoyaleGym, RoyaleSim, RoyaleLearn and RoyaleViser.

```bash
python -m venv .venv
.venv/bin/python -m pip install maturin pytest hypothesis ruff
cd RoyaleSim
../.venv/bin/python tools/extract_arena.py
../.venv/bin/python tools/extract_cards.py --vintage 2018
cp data/derived/cards-15.535.json data/derived/cards.json
../.venv/bin/python tools/extract_globals.py
../.venv/bin/maturin develop --release
cd ..
.venv/bin/python -m pip install -e RoyaleGym -e RoyaleViser
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e "RoyaleLearn[torch]"
cd RoyaleGym
../.venv/bin/python -m pytest -q -p no:randomly -rs
../.venv/bin/python -m ruff check royalegym tests examples
```

A test that cannot run says `SKIPPED, NOT PASSED` and why. A skip is not a pass.

## Rules that the tests enforce

- **Rows decoded by position stay in order.** `EntityState`, `SpellState` and `ProjectileState` arrive
  as arrays. A new field goes at the end with a "not said" default (-1, None or empty), and the engine's
  column list must be a prefix of this package's (`check_field_order`).
- **"Not said" is never zero.** A default means the engine did not report the value. Read status bits
  through `status_of`. A builder that needs a value an engine does not send refuses that engine by
  name, rather than drawing zeros that look like an answer.
- **The observation only grows at the end.** Every option is off by default and appends its fields or
  planes after the existing ones, so no existing offset moves. Spatial planes hold whole thousandths
  (`tests/test_spatial_exact_storage.py`).
- **Card ids are positions** in the engine's catalogue. Tests and callers name cards, never bare ids.
- **The two engines agree.** `tests/test_rust_engine.py` compares MockEngine and RustEngine field by
  field (CI's 2018 row runs it). A field MockEngine does not model is None and listed as unreported.
- **Test plants.** A test that claims to catch a defect is shown to fail with the defect put back.
- **Docs gates.** Site examples run and must print what the page shows (`tests/test_site_examples.py`),
  shell blocks on reader pages run in the shells they name, and install lines name the newest release.

## Public text

This repository is public. Its text covers the environments, the game's mechanics and how to use them.
Do not add training results, training recipes, bot settings, run names, or private tooling and paths.

## Deeper docs

- [docs/architecture.md](docs/architecture.md): how the package is built.
- [docs/observation-spec.md](docs/observation-spec.md): every observation field and plane.
- [docs/guide.md](docs/guide.md): the user guide.
- [CONTRIBUTING.md](CONTRIBUTING.md) and the [website](https://royalegym.github.io/RoyaleGym/).
