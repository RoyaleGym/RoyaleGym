# RoyaleGym

A Clash Royale reinforcement-learning stack built the way the Rocket League community built
theirs: **RLGym** (environment API) over **RocketSim** (engine) trained by **RLGym-PPO**
(learner), with **rlviser** on the side. Same split, same names with `Royale` in front, same
conventions. Hot loops (simulation, rollouts) live in Rust; Python exists only so bot creators
can define *what* to train (rewards, observations, actions, starting states, termination).

This repo is the environment layer, package `royalegym`, and the family's front door: the
project is named after it and this page carries the overview. The repos live under the
GitHub organization [RoyaleGym](https://github.com/RoyaleGym): RoyaleSim, RoyaleGym,
RoyaleLearn and RoyaleViser are public; RoyaleLive is private.

## The family

Three pillars and two tools, five sibling repos, one workspace, one venv.

| Repo | Analog | Responsibility | Language | Package |
|---|---|---|---|---|
| [RoyaleSim](https://github.com/RoyaleGym/RoyaleSim) | RocketSim | Deterministic, integer-only tick engine: pathfinding, targeting, collision, combat, spells, elixir, win conditions. Bit-exact against the real client where measured; every constant in `data/calibration.json` carries the client version it was measured on. One PyO3 extension module, `royalesim`, that this repo's `RustEngine` wraps. | Rust + PyO3 | `royalesim` |
| **RoyaleGym** (this repo) | RLGym | Environment API. Bot creators compose `ObsBuilder`, `ActionParser`, `RewardFunction`, `StateSetter`, `TerminalCondition` and get Gymnasium / PettingZoo / vectorised self-play envs. Defaults are shipped for each, and the per-tick ones move into the engine so the common path never enters Python (see Status). | Python | `royalegym` |
| [RoyaleLearn](https://github.com/RoyaleGym/RoyaleLearn) | RLGym-PPO | High-throughput training harness: vectorised self-play rollout workers, PPO learner, frozen-pool ladder with confidence intervals, checkpoints, metrics sink. | Python now; rollout workers move to Rust when they become the bottleneck | `royalelearn` |
| [RoyaleViser](https://github.com/RoyaleGym/RoyaleViser) | rlviser | The viewer, a separate process never in the tick loop: replays live captures, engine traces and a running env (`python -m royaleviser`). The engine side is `royalegym.viser.ViserPublisher` (UDP, one frame per env step, sent only while a viewer's heartbeat is fresh). | Python (pygame) | `royaleviser` |
| RoyaleLive (private) | - | The client instrument that records ground-truth traces from the real game, which calibrate RoyaleSim. | - | private |

Dependency direction is strictly `RoyaleLearn -> RoyaleGym -> RoyaleSim`. RoyaleSim knows
nothing about rewards or observations; RoyaleGym knows nothing about PPO. RoyaleViser depends
on RoyaleGym (traces, the publisher's wire form) and on nothing else; RoyaleLive depends on
RoyaleSim's data (the card tables) and on RoyaleViser, which it drives like any other caller.
A measurement is cited here the way every other calibrated number is: as a measured fact with
its client version ("measured on client 16.402, RoyaleLive traces").

Two design documents sit behind this layout: `docs/architecture.md` (the layers, the engine
contract, the conventions and the reasons for each) and `docs/background.md` (what is
publicly established about the game's rules, what is not, and why the engine is calibrated
against recordings rather than reasoned out).

## Setup: the shared workspace

Clone the siblings into one folder and build one venv at that folder's root. Sibling
checkouts are the documented layout: `royalegym` finds the engine's data at
`../RoyaleSim/data` (override with `ROYALESIM_DATA_DIR`), and RoyaleViser reads recorded
battles from wherever `ROYALELIVE_REPORTS` points, defaulting to its own `tests/captures`
(its capture tests skip when that folder is empty).
Python 3.12; Rust 1.80+ with cargo for RoyaleSim.

```
mkdir Royale && cd Royale
git clone https://github.com/RoyaleGym/RoyaleSim.git
git clone https://github.com/RoyaleGym/RoyaleGym.git
git clone https://github.com/RoyaleGym/RoyaleViser.git
git clone https://github.com/RoyaleGym/RoyaleLearn.git
python -m venv .venv
.venv\Scripts\python -m pip install maturin pytest hypothesis ruff
cd RoyaleSim && ..\.venv\Scripts\python tools\extract_arena.py && ..\.venv\Scripts\python tools\extract_cards.py && ..\.venv\Scripts\python tools\extract_globals.py && cd ..   # 0. RoyaleSim/data/derived/ (gitignored; the crate compiles arena.json in)
cd RoyaleSim && ..\.venv\Scripts\maturin develop --release && cd ..   # 1. the engine: builds royalesim into the venv (~1 min, fat LTO, ~1.5 GB RAM)
.venv\Scripts\python -m pip install -e RoyaleGym                       # 2. this repo: pulls numpy, gymnasium, pettingzoo, msgspec
.venv\Scripts\python -m pip install -e RoyaleViser                     # 3. the viewer: pulls pygame
.venv\Scripts\python -m pip install -e RoyaleLearn                     # 4. the learner
                                                                       # RoyaleLive (private): scripts run from its folder, no package
```

Install order matters only in that RoyaleGym's Rust-backed tests need `royalesim` built first;
`royalegym` imports without it (`MockEngine` is the pure-Python reference) and `RustEngine()`
raises an `ImportError` naming the build command. Rebuild `royalesim` after touching
`RoyaleSim/data/calibration.json` or `data/derived/arena.json`: `RustEngine` refuses a build
whose embedded copies differ from the files on disk.

## This repo

```
royalegym/
  protocol.py      the Engine contract (BattleState, MatchSetup, Calibration, Arena, DeployRules), data_dir()
  env.py           ClashParallelEnv (PettingZoo), ClashGymEnv (Gymnasium, id royalegym/ClashRoyale-v0), ClashSelfPlayVecEnv
  mock_engine.py   MockEngine, the pure-Python reference engine the RL layer is tested against
  rust_engine.py   RustEngine over the compiled royalesim core; SymmetricRustEngine for the rotation-mirror gates
  obs.py           ObsBuilder: SpatialObsBuilder, EntityListObsBuilder
  action.py        ActionParser: TileActionParser (Discrete 2305), HalfTileActionParser; PlacementOracle (the legality mask)
  reward.py        RewardFunction: WinLoss, Crown, TowerHP, ElixirTrade, ElixirLeak, IllegalAction, Combined
  terminal.py      TerminalCondition: GameOver, StepLimit, TickLimit, FirstCrown, Any
  state_setter.py  StateSetter: Default, MidGame, ScriptedBoard, Snapshot, Weighted
  selfplay.py      OpponentPool, RandomLegalOpponent, NoopOpponent
  replay.py        ReplayRecorder: records a trace; verify_trace re-runs it bit-for-bit
  render.py        the offline HTML replay page (python -m royalegym.render trace.msgpack -o out.html)
  viser.py         ViserPublisher, the engine side of RoyaleViser
tests/             pytest: the env layer against MockEngine, and the Rust engine through RustEngine
docs/              architecture.md (the design), background.md (what is known about the game's rules)
```

```python
from royalegym import ClashParallelEnv, RustEngine, SpatialObsBuilder, TileActionParser, WinLossReward

env = ClashParallelEnv(engine=RustEngine(), obs_builder=SpatialObsBuilder(),
                       action_parser=TileActionParser(), reward_fn=WinLossReward())
obs, info = env.reset(seed=0)
```

Every user-facing behaviour is an ABC with swappable implementations; the defaults are
provided and none is special. The engine is deterministic and seedable, and a trace recorded
from it re-verifies bit-for-bit. Python is kept off the per-tick path in the common case,
because that is where the throughput goes: a Python obs builder caps throughput at about 520
env-steps/s against the engine's ~25 000 ticks/s.

## Tests

```
cd RoyaleGym && ..\.venv\Scripts\python -m pytest -q      # 235 passed (~90 s; the Rust tests skip, not pass, without royalesim)
..\.venv\Scripts\python -m ruff check royalegym tests     # clean
```

`tests/test_viser.py` needs `royaleviser` installed (it round-trips a published frame through
the viewer's decoder) and skips otherwise. A stale `royalesim` build is a failure, not a skip.

## Status

Working: the full API shape (`ObsBuilder`, `ActionParser`, `RewardFunction`,
`TerminalCondition`, `StateSetter`, the `Engine` protocol; the Gymnasium, PettingZoo and
self-play vectorised envs), on both `MockEngine` and `RustEngine`, with the legality mask
held to both engines at every half-cell.

Open:

- **RLGym v2 naming.** `StateSetter` becomes `StateMutator` and `TerminalCondition` splits
  into `TerminationCondition` / `TruncationCondition`, to match the API this layer is modelled
  on. It is one rename pass with the tests, not a piecemeal change.
- **Rust-backed default observations and rewards.** Today's defaults are Python, which caps
  throughput at roughly 520 env-steps/s against the engine's ~25 000 ticks/s. Until they move
  down into the engine, training spends its time in the observation builder.
- **A `RustEngine` source for the viewer.** Traces and streams have been exercised on
  `MockEngine` only (RoyaleViser's `docs/internals.md`).

## Community

The project's Discord is the front door for the whole family — bot creators, engine work and training runs:
[**https://discord.gg/4D2BS5JBHP**](https://discord.gg/4D2BS5JBHP)

Issues and pull requests on this repo are welcome too.
