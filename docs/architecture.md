# Architecture

How the Royale packages fit together, and why the seams are where they are. This is the
design document behind `royalegym`; the module docstrings carry the detail for each piece.

## Three layers, one direction

The battle rules, the environment API and the training harness are three separate packages,
with the viewer off to the side. (The shape is prior art: RLGym and RocketSim settled on
the same seams for Rocket League.)

| Layer | Owns | Language |
|---|---|---|
| RoyaleSim | the battle: pathfinding, targeting, collision, combat, spells, elixir, win conditions | Rust, exposed as the PyO3 module `royalesim` |
| RoyaleGym | the environment API: observations, actions, rewards, state mutators, done conditions, the Gymnasium / PettingZoo / vectorised self-play envs | Python |
| RoyaleLearn | the training harness: rollout workers, PPO, the frozen-pool ladder, checkpoints, metrics | Python |
| RoyaleViser | drawing a battle, in a separate process | Python (pygame) |

Dependencies run strictly `RoyaleLearn -> RoyaleGym -> RoyaleSim`, with
`RoyaleViser -> RoyaleGym` off to the side. RoyaleSim knows nothing about rewards or
observations; RoyaleGym knows nothing about PPO; the viewer imports the env layer and
nothing imports the viewer.

The point of the direction is that each layer can be replaced without the others noticing.
A reward experiment needs no recompile. A faster engine needs no change to a bot's
observation builder. A different learner drives the same envs.

## Why the battle rules are in Rust

The project started from a pure-Python simulator. It was correct enough to play a match and
far too slow to train against: measured on a full board of 156 entities it managed about 20
ticks per second, against the 20 ticks per *battle* second the game itself runs at, with an
O(n squared) collision pass that tuning does not fix. The Rust core runs the same battle at
roughly 25 000 ticks/s.

The consequence for this repo is the rule that **Python is never on the per-tick path in the
common case**. Measured: a Python observation builder capped throughput at about 520
env-steps/s against the engine's ~25 000 ticks/s, so the observation builder, not the
simulation, was the whole cost of training. Defaults that run every tick belong in Rust,
behind a Python override for people who want to experiment; never the other way round.

## The engine contract

`royalegym.protocol` is the whole contract between the RL layer and a simulator: the
`Engine` protocol plus the value types (`BattleState`, `MatchSetup`, `Calibration`, `Arena`,
`DeployRules`). Two implementations satisfy it:

- `MockEngine` (`mock_engine.py`), pure Python. It is the reference the RL layer is
  developed and tested against, and it deliberately does not reproduce the Rust engine's
  mechanics (spells resolve instantly, cards run at CSV level 1). Its job is to be simple
  and independently written, so that agreement between the two engines means something.
- `RustEngine` (`rust_engine.py`), the adapter over the compiled `royalesim` core. A step is
  one Rust call (validate, apply, tick N times with the GIL released); the state snapshot
  comes back as one JSON byte string decoded straight into protocol structs.

Everything else in `royalegym` is engine-agnostic: no module other than those two knows
which engine it is driving.

### Frames and units

- Positions are integer **subtiles** in the **engine frame**: team 0 (Blue) defends low `y`,
  team 1 (Red) defends high `y`, origin at Blue's back-left corner. One tile is
  `Arena.subtile` subtiles.
- A team's **own frame** is the engine frame for Blue and the engine frame rotated 180
  degrees for Red (`x_own = W - x`, `y_own = H - y`).
- Elixir is integer thousandths (`elixir_milli`). Affordability is
  `elixir_milli >= cost * 1000`, which is exact.
- **Nothing in the contract is a float.** `arena.json` carries convenience float fields; this
  layer reads only the integer half-cell indices.

### Why a rotation and not a mirror

The seat symmetry is the 180-degree rotation, not the `y`-reflection. A rotation is what the
opponent literally sees across the table: "my left princess tower" means the same thing to
both players, so one policy head can play both seats. The shipped tilemap is exactly
invariant under that rotation with the left/right lane bits swapped.

The distinction is not cosmetic. The engine decides ties in each team's own frame, so a
`y`-reflected battle diverges as soon as play reaches one: reflected twins break ties toward
different engine sides. Multi-unit cards are where it shows first — in one measurement,
multi-unit deploys desynced the two seats of a shared policy in 80 of 144 placements, while
single-unit deploys desynced in 0 of 240. The gates in `tests/test_rust_engine.py` therefore
assert that the rotation holds *and* that the reflection does not.

## Composition, not configuration

Every user-facing behaviour is an abstract base class with swappable implementations:
`ObsBuilder`, `ActionParser`, `RewardFunction`, `DoneCondition`, `StateMutator`, and the
opponent policies used for self-play. Defaults are provided and none of them is special — a
default is just the implementation that ships. A new default lands with a test that holds it
to both engines.

The names are RLGym v2's, so a reader coming from there meets the vocabulary they know. A
`DoneCondition` is used in one of two roles: the env takes a `termination_cond`, which sets
Gymnasium's `terminated` (the outcome is settled and the next state is worth nothing), and a
`truncation_cond`, which sets `truncated` (the episode was cut and the next state is still
worth bootstrapping from). Confusing the two biases every value estimate near the cut, so
the shipped conditions declare their role as `TerminationCondition` or `TruncationCondition`
subclasses and the env refuses one in the wrong slot. A `StateMutator` describes the whole
starting state (`MatchSetup`, or a `Snapshot` to resume) and hands it to the engine; because
it describes rather than edits, mutators compose by choice (`WeightedStateMutator` picks one
per episode, and a training loop anneals the weights) rather than by chaining, which is why
there is no `MutatorSequence` here. Before the rename (2026-09-21) the same objects were
`TerminalCondition` and `StateSetter`; those names, and the `terminal_conditions=` /
`state_setter=` constructor arguments, still work as aliases.

| Piece | Shipped implementations |
|---|---|
| `ObsBuilder` | `SpatialObsBuilder` (20-channel board + `mask_planes` + a `12n + 37`-float vector + the mask), `EntityListObsBuilder` |
| `ActionParser` | `TileActionParser` (`Discrete(2305)`), `HalfTileActionParser` (`Discrete(9217)`) |
| `RewardFunction` | `WinLoss`, `Crown`, `TowerHP`, `ElixirTrade`, `ElixirLeak`, `IllegalAction`, `Combined`; `default_reward()` is WinLoss 1.0 + Crown 0.2 + TowerHP 0.1 + ElixirTrade 0.02 |
| `StateMutator` | `Default`, `MidGame`, `ScriptedBoard`, `Snapshot`, `Weighted` (curriculum lives here) |
| `DoneCondition` | `GameOver`, `FirstCrown` (terminations); `StepLimit`, `TickLimit` (truncations); `Any`, `All` (either role) |

Imperfect information is the default, and it is the default in the form a player actually
plays in: the opponent's hand is absent, and their elixir is a COUNT the builder keeps from
the plays it saw and the regeneration rate everyone knows — exact, and checked against the
engine's own bar every step of a played-out battle. `Reveal(enemy_hand=True, ...)` opens one
half of the hidden state at a time for curricula, distillation and debugging; an enabled
field ADDS channels and slots rather than filling zeroed ones, so a fair observation and a
cheating one are not even the same width, and `ClashParallelEnv.config()` records the
`Reveal` so a checkpoint says which one produced it. Every channel and every slot, with its
range and whether it is fair, is in [observation-spec.md](observation-spec.md).

The design rationale for each family lives with the code:

- `action.py` — why the action space is a joint `Discrete(2305)` over (hand slot, tile)
  rather than a `MultiDiscrete`, why tile and not half-tile resolution, and how the legality
  mask is computed independently of the engine so a wrong mask fails a test instead of a
  training run.
- `obs.py` — the observation layouts, and why every accumulation is integer and every row
  sort key names every field the row is built from (float addition is not associative, and
  the engine's entity order is private, so either one breaks seat symmetry by 1 ulp).
- `protocol.py` — what is refused, what is clamped, and which rules are measured against the
  real game versus carried as explicit open questions.

## The action space and its mask

`TileActionParser` is a joint `Discrete(2305)`: index 0 is no-op, index
`1 + slot * 576 + ty * 18 + tx` plays hand slot `slot` (one of 4) at the centre of tile
`(tx, ty)` (one of 18 x 32), in the acting player's own frame, so one policy head plays both seats. Legality is a property of
the *pair* (card, tile): elixir is per card, territory depends on the card's placement type
(spells anywhere, a Goblin Barrel anywhere but water, a Log where a troop may go but over
buildings, buildings never into the enemy pocket), footprints depend on the card's radius. A
joint space is the only shape in which a mask can say "Giant here, Fireball not": a
per-dimension mask, as sb3-contrib's MaskablePPO applies one to a `MultiDiscrete`, can only
mask the marginals and would happily sample "Knight on the enemy king". `HalfTileActionParser`
ships alongside at `Discrete(9217)` so the resolution trade can be measured instead of argued.

The mask is computed by `PlacementOracle` from the state snapshot, the arena and
`DeployRules`, independently of the engine, which enforces the same rules on its own code
path (`Engine.check_deploy`). The two are compared exhaustively in the test suite — every
half-cell centre and corner, both teams, seven tower states — so a silently wrong mask fails a
test instead of a training run. What it covers: elixir, territory, water, the river band,
building footprints and the no-deploy rectangle around each living enemy crown tower.

Measured at `reset(seed=0)` with the default random decks: 691 of 2305 actions are legal on
the first step on `RustEngine`, 1235 on `MockEngine` (whose cards run at CSV level 1). Every
observation dict carries the mask as `int8` (what PettingZoo's `parallel_api_test` and
Gymnasium's `Discrete.sample(mask=...)` expect), and `mask_planes` — the same mask minus the
no-op, reshaped to `[4, 32, 18]` for a convolutional trunk — beside it; `action_masks()`
returns it as `bool`, the form MaskablePPO calls for. An action the engine rejects becomes a
no-op and is reported in `info["deploy_status"]`. Because the mask already states per-slot,
per-tile legality exactly, the observation has no placement-zone channels for the acting
player: they restated it more coarsely and cost two of the three
`PlacementOracle.point_grid` calls the observation made per seat per step. Measured: 6.66
`point_grid` calls per `env.step` before and 2.66 after. Measured by alternating the two trees three times in one session, because the absolute numbers move by a third with machine load while the comparison does not: the suite's own throughput report went 766 -> 953 `env.step`/s on the Rust engine and 475 -> 580 on `MockEngine` (medians of three rounds).

## Constants are read, never copied

Every calibrated number lives once, in RoyaleSim's `data/calibration.json`, and carries the
client version it was measured on. This layer reads it:

- `Calibration` raises on a missing key rather than defaulting to something plausible.
- `Arena.load` and `RustEngine()` refuse to start when the numbers on disk differ from the
  numbers compiled into the engine, so a rebuild cannot be silently skipped.
- `data/derived/` is generated by RoyaleSim's extractors and is never hand-edited.

Where a rule is *not* measured, it says so in the data rather than hiding behind a plausible
default: `DeployRules` carries a `*_status` field per rule, and the unmeasured ones are
listed as open questions in the calibration ledger.

## Determinism

The engine is deterministic and seedable. A trace recorded from it re-verifies bit for bit
(`royalegym.replay.verify_trace` re-simulates it on a fresh engine and lists every divergence),
which is what makes a recorded battle usable as a regression test and
a bug report. `save_state`/`load_state` round-trips exactly, RNG included, so a battle can be
resumed for curriculum starts or search.

## Measured throughput

The suite ends with a throughput report that anyone can run:

```
cd RoyaleGym && ..\.venv\Scripts\python -m pytest -q tests/test_rust_engine.py -k throughput
```

Two runs on the same machine, one day apart, show the spread machine load alone produces:

```
2026-09-20  [throughput] env.step/s  rust 620 (11 live)  mock 423 (10 live)  at 10 ticks/step | engine ticks/s incl. state+mask per 20 ticks: rust 19941  mock 11373
2026-09-21  [throughput] env.step/s  rust 826 (11 live)  mock 763 (10 live)  at 10 ticks/step | engine ticks/s incl. state+mask per 20 ticks: rust 31967  mock 16035
```

`env.step/s` is the full Python stack (both players' observations and masks per step, 10
ticks per step at the default `decision_ms=500`); the engine figure steps 20 ticks per call
with a state snapshot and a mask in between. Runs on one machine have printed anywhere from
10 000 to 30 000 engine ticks/s, so the figure that matters is the one your own run prints.
For scale: the engine alone measures roughly 25 000 ticks/s against the 20 ticks per battle
second the game runs at — about 1 250x real time — and the pure-Python simulator this project
started from managed about 20 ticks/s on a full board of 156 entities, real time exactly.

## Tests

```
cd RoyaleGym && ..\.venv\Scripts\python -m pytest -q      # 315 passed, 6 skipped, 5 failed (2026-09-21)
..\.venv\Scripts\python -m ruff check royalegym tests     # All checks passed!
```

Without `royalesim` built the Rust-backed tests skip, not pass. A `royalesim` build older
than `calibration.json` or `derived/arena.json` on disk is a failure, not a skip: `RustEngine()`
refuses it and names every differing key. `tests/test_viser.py` needs `royaleviser` installed
(it round-trips a published frame through the viewer's decoder) and skips otherwise.

## Module map

| Module | What it holds |
|---|---|
| `protocol.py` | the `Engine` contract and its value types (`BattleState`, `MatchSetup`, `Calibration`, `Arena`, `DeployRules`); `data_dir()` |
| `env.py` | `ClashParallelEnv` (PettingZoo), `ClashGymEnv` (Gymnasium, id `royalegym/ClashRoyale-v0`), `ClashSelfPlayVecEnv` (N games as 2N agent slots: four games give a `spatial` batch of shape (8, 20, 32, 18) and eight masks), `make_gym_vec_env` |
| `mock_engine.py` | `MockEngine`, the pure-Python reference engine the RL layer is tested against |
| `rust_engine.py` | `RustEngine` over the compiled `royalesim` core; `SymmetricRustEngine` for the rotation-mirror gates |
| `obs.py` | `ObsBuilder`: `SpatialObsBuilder`, `EntityListObsBuilder` |
| `action.py` | `ActionParser`: `TileActionParser`, `HalfTileActionParser`; `PlacementOracle` (the legality mask) |
| `reward.py` | `RewardFunction` and the shipped terms; `default_reward()` |
| `done_condition.py` | `DoneCondition`, `TerminationCondition`, `TruncationCondition` and the shipped conditions |
| `state_mutator.py` | `StateMutator` and the shipped mutators; `Snapshot` |
| `selfplay.py` | `Opponent`, `NoopOpponent`, `RandomLegalOpponent`, `CallableOpponent`, `OpponentPool` (uniform / latest / PFSP sampling, Elo bookkeeping) |
| `replay.py` | `ReplayRecorder`, `Trace`, `save_trace` / `load_trace`, `verify_trace` |
| `render.py` | the offline HTML replay page (`python -m royalegym.render trace.msgpack -o out.html`) |
| `viser.py` | `ViserPublisher`, the engine side of RoyaleViser |

`tests/` holds the pytest suite: the env layer against `MockEngine`, and the Rust engine
through `RustEngine`. `docs/` holds this file and `background.md`.

## Everything else stays off the hot path

Rendering, logging and monitoring are out of process or off the tick loop:

- `royalegym.render` writes a self-contained offline HTML page from a trace, with a timeline
  scrubber and a hover tooltip per unit; it opens by double-click, no server needed.
- `render_mode="ansi"` gives `env.render()` as a tile-resolution text board, Blue at the
  bottom: `T`/`t` crown towers, `B`/`b` buildings, `u`/`r` units (Blue upper case or `u`, Red
  lower case or `r`), `~` the river, `#` tiles nothing may be deployed on, with tick, elixir and
  crowns in the header line. It is a debugging aid; the viewer and the replay page are the
  pictures.
- `royalegym.viser.ViserPublisher` streams one frame per env step over UDP, and only while a
  viewer's heartbeat is fresh — an env nobody is watching pays one `if` per step.
- Metrics belong to the learner, which is a separate process from the envs.
