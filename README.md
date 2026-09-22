# RoyaleGym

Reinforcement-learning environments for **Clash Royale**. You write the parts that decide
*what* to train — observations, actions, rewards, starting states, terminal conditions — in
Python. The battle itself runs in a deterministic, integer-only Rust engine
([RoyaleSim](https://github.com/RoyaleGym/RoyaleSim)) whose constants are measured against
recordings of the real game rather than guessed.

Gymnasium and PettingZoo APIs, an exact legality mask over the card-and-tile action space,
batched self-play, and traces that re-verify bit for bit.

## A whole battle, end to end

Nothing but `royalegym` and numpy. This is the complete program:

```python
import numpy as np
from royalegym import ClashParallelEnv, RustEngine, RandomLegalOpponent

env = ClashParallelEnv(engine=RustEngine(), render_mode="ansi")   # PettingZoo parallel API
obs, info = env.reset(seed=0)
rng, policy = np.random.default_rng(0), RandomLegalOpponent(noop_prob=0.7)

while env.agents:                      # a whole battle, both seats played
    actions = {a: policy.act(obs[a], obs[a]["action_mask"], rng) for a in env.agents}
    obs, reward, terminated, truncated, info = env.step(actions)

s = env.battle_state
print(env.render())
print(f"winner {s.winner}  crowns {[p.crowns for p in s.players]}  tick {s.tick}")
```

```
tick 4129  blue 4.6e 1c  red 4.6e 0c
#####........#####
..............b...
........#t........
........##........
..................
..................
....u.........t...
...u.u............
..................
..................
..................
...u.u............
..uuu.............
..................
#................#
~~...~~~~~~~~...~~
~~...~~~~~~~~...~~
#................#
..................
..................
..................
..................
..............r...
..................
............uu....
...T..........T...
.................u
..................
........#T........
........##........
..................
#####........#####
winner 0  crowns [1, 0]  tick 4129
```

Blue is at the bottom. `T`/`t` are crown towers, `B`/`b` buildings, `u`/`r` units, `~` the
river, `#` tiles nothing may be deployed on. Eight Blue units are pushing through Red's half
against one Red unit in Blue's, and one of Red's princess towers is gone from the row that
started with two — that is the crown in `crowns [1, 0]`. Two players of uniformly random
*legal* moves produced it in about 1.2 s of wall clock, which is the point: a masked random
policy is already a valid opponent, so there is no bootstrap problem when you plug in a real
one.

## What you get

**Two APIs over the same battle.** PettingZoo parallel (`ClashParallelEnv`, both seats),
Gymnasium single-agent (`ClashGymEnv`, registered as `royalegym/ClashRoyale-v0`, the other
seat driven by an `Opponent`), and `ClashSelfPlayVecEnv`, which exposes *N* simultaneous
games as 2*N* agent slots so one shared policy collects both players' experience:

```python
>>> vec = ClashSelfPlayVecEnv(4, env_fn=lambda: ClashParallelEnv(engine=RustEngine()))
>>> obs, info = vec.reset(seed=0)
>>> vec.num_envs, obs["spatial"].shape, vec.action_masks().shape
(8, (8, 21, 32, 18), (8, 2305))
```

**An exact legality mask, not an approximation.** The action space is a joint
`Discrete(2305)` = NO-OP + 4 hand slots x 18 x 32 tiles, in the acting player's own frame so
one policy head plays both seats. Every observation carries a mask of what is actually
playable right now — elixir, territory, water, the river band, building footprints, the
no-deploy rect around each living enemy crown tower:

```python
>>> obs, _ = env.reset(seed=0)
>>> mask = obs["blue"]["action_mask"]          # int8[2305], and env.action_masks() as bool
>>> int(mask.sum()), mask.size
(691, 2305)
```

Legality is a property of the *pair* (card, tile) — Giant legal where Fireball is not — which
is why the space is joint rather than `MultiDiscrete`: a per-dimension mask, as
sb3-contrib's MaskablePPO applies one, can only mask the marginals and would happily sample
"Knight on the enemy king". `HalfTileActionParser` ships alongside at `Discrete(9217)` so the
resolution trade can be measured instead of argued.

The mask is computed by `PlacementOracle` from the state snapshot, arena and `DeployRules`,
**independently of the engine**, which enforces the same rules on its own code path. The two
are then compared exhaustively in the test suite — every half-cell centre and corner, both
teams, seven tower states — so a silently wrong mask fails a test instead of a training run.

**Five swappable pieces, none of them privileged.** Each is an ABC with shipped defaults:

| Piece | Shipped implementations |
|---|---|
| `ObsBuilder` | `SpatialObsBuilder` (21-channel board + vector + mask), `EntityListObsBuilder` |
| `ActionParser` | `TileActionParser` (`Discrete(2305)`), `HalfTileActionParser` |
| `RewardFunction` | `WinLoss`, `Crown`, `TowerHP`, `ElixirTrade`, `ElixirLeak`, `IllegalAction`, `Combined` |
| `StateSetter` | `Default`, `MidGame`, `ScriptedBoard`, `Snapshot`, `Weighted` (curriculum lives here) |
| `TerminalCondition` | `GameOver`, `StepLimit`, `TickLimit`, `FirstCrown`, `Any` |

Observations are always in the acting player's **own frame**, so a mirrored battle yields a
bit-identical observation for the other seat and one policy can play both. The seat symmetry
is the 180-degree rotation, not the `y`-reflection, and the difference is not cosmetic: the
engine breaks ties in each team's own frame, so reflected twins diverge. Measured, multi-unit
deploys desynced the two seats of a shared policy in **80 of 144** reflected placements while
single-unit deploys desynced in **0 of 240** (`docs/architecture.md`); the suite asserts that
the rotation holds *and* that the reflection does not.

The opponent's elixir and hand are hidden by default, as in the live game.

**Two engines behind one protocol.** `royalegym.protocol.Engine` is the whole contract.
`RustEngine` drives the compiled core; `MockEngine` is a pure-Python reference engine,
independently written, that the RL layer is developed and tested against — so agreement
between the two means something. `royalegym` imports and runs on `MockEngine` with no Rust
toolchain present:

```python
>>> from royalegym import MockEngine
>>> int(ClashParallelEnv(engine=MockEngine()).reset(seed=0)[0]["blue"]["action_mask"].sum())
1235
```

**Determinism you can hand to someone else.** Record a battle, re-simulate it, and every
state hash matches tick for tick:

```python
from royalegym import ClashParallelEnv, RustEngine, ReplayRecorder, verify_trace, save_trace

rec = ReplayRecorder()
env = ClashParallelEnv(engine=RustEngine(), recorder=rec)
...                                          # play as above
trace = rec.trace
print(trace.result.winner, trace.result.crowns, trace.result.final_tick, len(trace.frames))
print("divergences:", verify_trace(trace, RustEngine()))
save_trace(trace, "battle.msgpack")
```

```
0 [1, 0] 4129 4130
divergences: []
```

`save_state` / `load_state` round-trips exactly, RNG included, so a position can be resumed
for curriculum starts or search — which is what `SnapshotStateSetter` does.

**Watch it, off the hot path.** Turn that trace into a self-contained page with no engine and
no server:

```
python -m royalegym.render battle.msgpack -o battle.html
wrote battle.html (3321553 bytes, 4130 frames)
```

For a live view, `royalegym.viser.ViserPublisher` streams one frame per env step over UDP to
[RoyaleViser](https://github.com/RoyaleGym/RoyaleViser), and only while a viewer's heartbeat
is fresh — an env nobody is watching pays one `if` per step.

## Measured

The suite ends with a throughput report you can run on your own machine:

```
cd RoyaleGym && ..\.venv\Scripts\python -m pytest -q tests/test_rust_engine.py -k throughput
[throughput] env.step/s  rust 620 (11 live)  mock 423 (10 live)  at 10 ticks/step | engine ticks/s incl. state+mask per 20 ticks: rust 19941  mock 11373
```

Runs on one machine have printed anywhere from 10 000 to 30 000 engine ticks/s and a few
hundred env.step/s through the full Python stack; the spread is machine load, so the figure that matters is the one
your own run prints. For scale, the engine on its own measures roughly **25 000 ticks/s**
against the 20 ticks per battle-second the game runs at — about 1 250x real time — and the
pure-Python simulator this project started from managed about **20 ticks/s** on a full board
of 156 entities, which is real time exactly and no faster (`docs/architecture.md`).

The gap between those numbers and `env.step/s` is Python: a Python observation builder capped
throughput at about **520 env-steps/s**, which is why moving the default observations and
rewards down into the engine is the headline open item below, and why the rule here is that
Python is never on the per-tick path in the common case.

## Install

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

## The family

This repo is the environment layer, package `royalegym`, and the family's front door: the
project is named after it and this page carries the overview. Three pillars and two tools,
five sibling repos, one workspace, one venv. The repos live under the GitHub organization
[RoyaleGym](https://github.com/RoyaleGym): RoyaleSim, RoyaleGym, RoyaleLearn and RoyaleViser
are public; RoyaleLive is private.

| Repo | What it does | Language | Package |
|---|---|---|---|
| [RoyaleSim](https://github.com/RoyaleGym/RoyaleSim) | The battle: pathfinding, targeting, collision, combat, spells, elixir, win conditions, on a deterministic integer-only tick loop. Its rules were measured against recordings of the real client and are scored case by case: 751 of 752 published path node lists reproduced node for node, 0.9924 of unit-tick positions exact. Every constant in `data/calibration.json` carries the client version it was measured on. One PyO3 extension module, `royalesim`, that this repo's `RustEngine` wraps. | Rust + PyO3 | `royalesim` |
| **RoyaleGym** (this repo) | The environment API. Compose `ObsBuilder`, `ActionParser`, `RewardFunction`, `StateSetter`, `TerminalCondition` and get Gymnasium / PettingZoo / vectorised self-play envs. Defaults are shipped for each, and the per-tick ones move into the engine so the common path never enters Python (see Status). | Python | `royalegym` |
| [RoyaleLearn](https://github.com/RoyaleGym/RoyaleLearn) | The training harness: vectorised self-play rollout workers, PPO learner, frozen-pool ladder with confidence intervals, checkpoints, metrics sink. | Python now; rollout workers move to Rust when they become the bottleneck | `royalelearn` |
| [RoyaleViser](https://github.com/RoyaleGym/RoyaleViser) | The viewer, a separate process never in the tick loop: replays recorded battles, engine traces and a running env (`python -m royaleviser`). The engine side is `royalegym.viser.ViserPublisher`. | Python (pygame) | `royaleviser` |
| RoyaleLive (private) | The client instrument that records ground-truth traces from the real game, which calibrate RoyaleSim. | - | private |

The layering will look familiar if you know RLGym and RocketSim: an environment API over a
fast engine, a learner on top, a viewer out of process. That is prior art worth crediting.

Dependency direction is strictly `RoyaleLearn -> RoyaleGym -> RoyaleSim`. RoyaleSim knows
nothing about rewards or observations; RoyaleGym knows nothing about PPO. RoyaleViser depends
on RoyaleGym (traces, the publisher's wire form) and on nothing else; RoyaleLive depends on
RoyaleSim's data (the card tables) and on RoyaleViser, which it drives like any other caller.
A measurement is cited here the way every other calibrated number is: as a measured fact with
its client version ("measured on client 16.402, RoyaleLive traces").

Two design documents sit behind this layout: [`docs/architecture.md`](docs/architecture.md)
(the layers, the engine contract, the conventions and the reasons for each) and
[`docs/background.md`](docs/background.md) (what is publicly established about the game's
rules, what is not, and why the engine is calibrated against recordings rather than reasoned
out).

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

## Tests

```
cd RoyaleGym && ..\.venv\Scripts\python -m pytest -q      # 235 passed (~80 s; the Rust tests skip, not pass, without royalesim)
..\.venv\Scripts\python -m ruff check royalegym tests     # All checks passed!
```

`tests/test_viser.py` needs `royaleviser` installed (it round-trips a published frame through
the viewer's decoder) and skips otherwise. A stale `royalesim` build is a failure, not a skip.

## Status

Working: the full API shape (`ObsBuilder`, `ActionParser`, `RewardFunction`,
`TerminalCondition`, `StateSetter`, the `Engine` protocol; the Gymnasium, PettingZoo and
self-play vectorised envs), on both `MockEngine` and `RustEngine`, with the legality mask
held to both engines at every half-cell.

Open:

- **Rust-backed default observations and rewards.** Today's defaults are Python, which caps
  throughput at roughly 520 env-steps/s against the engine's ~25 000 ticks/s. Until they move
  down into the engine, training spends its time in the observation builder.
- **Naming.** `StateSetter` becomes `StateMutator`, and `TerminalCondition` splits into
  `TerminationCondition` / `TruncationCondition` so that a natural end and a time-out stop
  being the same thing. It is one rename pass with the tests, not a piecemeal change.

## Community

The project's Discord is the front door for the whole family — bot creators, engine work and training runs:
[**https://discord.gg/4D2BS5JBHP**](https://discord.gg/4D2BS5JBHP)

Issues and pull requests on this repo are welcome too.
