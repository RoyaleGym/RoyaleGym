# Architecture

This page is for contributors working on `royalegym`, or on the layers either side of it.
It is the design document behind the package: how the Royale packages fit together, and
why the seams are where they are. The module docstrings carry the detail for each piece.

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
far too slow to train against. Measured on a full board of 156 entities, it managed about 20
ticks per second, against the 20 ticks per *battle* second the game itself runs at. Its
collision pass is O(n squared), and tuning does not fix that. The Rust core runs the same
battle at roughly 25 000 ticks/s.

The consequence for this repo is the rule that **Python is never on the per-tick path in the
common case**. Measured: a Python observation builder capped throughput at about 520
env-steps/s against the engine's ~25 000 ticks/s. The observation builder, not the
simulation, was the whole cost of training. Defaults that run every tick belong in Rust,
behind a Python override for people who want to experiment. Never the other way round.

## The engine contract

`royalegym.protocol` is the whole contract between the RL layer and a simulator: the
`Engine` protocol plus the value types (`BattleState`, `MatchSetup`, `Calibration`, `Arena`,
`DeployRules`). Two implementations satisfy it:

- `MockEngine` (`mock_engine.py`), pure Python. It is the reference the RL layer is
  developed and tested against. It deliberately does not reproduce the Rust engine's
  mechanics (spells resolve instantly, cards run at CSV level 1). Its job is to be simple
  and independently written, so that agreement between the two engines means something.
- `RustEngine` (`rust_engine.py`), the adapter over the compiled `royalesim` core. A step is
  one Rust call: validate, apply, then tick N times with the GIL released. The state
  snapshot comes back as one JSON byte string, decoded straight into protocol structs.

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
different engine sides. Multi-unit cards are where it shows first. In one measurement,
multi-unit deploys desynced the two seats of a shared policy in 80 of 144 placements, while
single-unit deploys desynced in 0 of 240. The gates in `tests/test_rust_engine.py` therefore
assert that the rotation holds *and* that the reflection does not.

## Composition, not configuration

Every user-facing behaviour is an abstract base class with swappable implementations:
`ObsBuilder`, `ActionParser`, `RewardFunction`, `DoneCondition`, `StateMutator`, and the
opponent policies used for self-play. Defaults are provided and none of them is special. A
default is just the implementation that ships. A new default lands with a test that holds it
to both engines.

The names are RLGym v2's, so a reader coming from there meets the vocabulary they know. A
`DoneCondition` fills one of two roles. The env takes a `termination_cond`, which sets
Gymnasium's `terminated` (the outcome is settled and the next state is worth nothing). It
also takes a `truncation_cond`, which sets `truncated` (the episode was cut and the next
state is still worth bootstrapping from). Confusing the two biases every value estimate near
the cut. So the shipped conditions declare their role as `TerminationCondition` or
`TruncationCondition` subclasses, and the env refuses one in the wrong slot.

A `StateMutator` describes the whole starting state (`MatchSetup`, or a `Snapshot` to
resume) and hands it to the engine. Because it describes rather than edits, mutators compose
by choice rather than by chaining: `WeightedStateMutator` picks one per episode, and a
training loop anneals the weights. That is why there is no `MutatorSequence` here. Before
the rename (2026-09-21) the same objects were `TerminalCondition` and `StateSetter`. Those
names, and the `terminal_conditions=` / `state_setter=` constructor arguments, still work as
aliases.

| Piece | Shipped implementations |
|---|---|
| `ObsBuilder` | `SpatialObsBuilder` (20-channel board + `mask_planes` + a `12n + 37`-float vector + the mask), `EntityListObsBuilder` |
| `ActionParser` | `TileActionParser` (`Discrete(2305)`), `HalfTileActionParser` (`Discrete(9217)`) |
| `RewardFunction` | `WinLoss`, `Crown`, `TowerHP`, `ElixirTrade`, `ElixirLeak`, `PlacementDepth`, `IllegalAction`, `Combined`; `default_reward()` is WinLoss 1.0 + Crown 0.2 + TowerHP 0.1 + ElixirTrade 0.02. `PlacementDepth` and `IllegalAction` ship unused, as templates |
| `StateMutator` | `Default`, `MidGame`, `ScriptedBoard`, `Snapshot`, `Weighted`, `DeckCurriculum` (curriculum lives here) |
| `DoneCondition` | `GameOver`, `FirstCrown` (terminations); `StepLimit`, `TickLimit` (truncations); `Any`, `All` (either role) |

Imperfect information is the default, and it is the default in the form a player actually
plays in. The opponent's hand is absent. Their elixir is a COUNT the builder keeps from the
plays it saw and the regeneration rate everyone knows. That count is exact, and it is
checked against the engine's own bar every step of a played-out battle.
`Reveal(enemy_hand=True, ...)` opens one half of the hidden state at a time for curricula,
distillation and debugging. An enabled field ADDS channels and slots rather than filling
zeroed ones, so a fair observation and a cheating one are not even the same width. And
`ClashParallelEnv.config()` records the `Reveal`, so a checkpoint says which one produced
it. Every channel and every slot, with its range and whether it is fair, is in
[observation-spec.md](observation-spec.md).

The design rationale for each family lives with the code:

- `action.py`: why the action space is a joint `Discrete(2305)` over (hand slot, tile)
  rather than a `MultiDiscrete`, why tile and not half-tile resolution, and how the legality
  mask is computed independently of the engine so a wrong mask fails a test instead of a
  training run.
- `obs.py`: the observation layouts, and why every accumulation is integer and every row
  sort key names every field the row is built from (float addition is not associative, and
  the engine's entity order is private, so either one breaks seat symmetry by 1 ulp, the
  smallest step a float can take).
- `protocol.py`: what is refused, what is clamped, and which rules are measured against the
  real game versus carried as explicit open questions.

## The action space and its mask

`TileActionParser` is a joint `Discrete(2305)`. Index 0 is the no-op. Index
`1 + slot * 576 + ty * 18 + tx` plays hand slot `slot` (one of 4) at the centre of tile
`(tx, ty)` (one of 18 x 32), in the acting player's own frame, so one policy head plays both
seats. Legality is a property of the *pair* (card, tile). Elixir is per card. Territory
depends on the card's placement type (spells anywhere, a Goblin Barrel anywhere but water, a
Log where a troop may go but over buildings, buildings never into the enemy pocket).
Footprints depend on the card's radius. A joint space is the only shape in which a mask can
say "Giant here, Fireball not". A per-dimension mask, as sb3-contrib's MaskablePPO applies
one to a `MultiDiscrete`, can only mask the marginals, and would happily sample "Knight on
the enemy king". `HalfTileActionParser` ships alongside at `Discrete(9217)`, so the
resolution trade can be measured instead of argued.

`PlacementOracle` computes the mask from the state snapshot, the arena and `DeployRules`,
independently of the engine. The engine enforces the same rules on its own code path
(`Engine.check_deploy`). The test suite compares the two exhaustively: every half-cell
centre and corner, both teams, seven tower states. So a silently wrong mask fails a test
instead of a training run. The mask covers elixir, territory, water, the river band,
building footprints and the no-deploy rectangle around each living enemy crown tower.

Measured at `reset(seed=0)` with the default random decks, 2026-09-24. On `RustEngine` the
first step offers only the no-op, because no card can be played during the opening lockout
(`DeployRules.deploy_lockout_ticks`, 90 ticks). After nine no-op steps, at tick 90, 921 of 2305
actions are legal. `MockEngine` has no lockout and offers 1235 on the first step (its cards run
at CSV level 1). Read that pair carefully rather than as a property of either engine. A random deck
is drawn from the catalogue, so which cards land in the first hand moves with the catalogue,
and a hand of four troops offers far fewer tiles than one holding a spell, which is legal
almost everywhere. The `RustEngine` figure here is a hand of four troops. Every
observation dict carries the mask as `int8`, which is what PettingZoo's `parallel_api_test`
and Gymnasium's `Discrete.sample(mask=...)` expect. Beside it sits `mask_planes`, the same
mask minus the no-op, reshaped to `[4, 32, 18]` for a convolutional trunk. `action_masks()`
returns the mask as `bool`, the form MaskablePPO calls for. An action the engine rejects
becomes a no-op, and `info["deploy_status"]` reports it. Because the mask already states
per-slot, per-tile legality exactly, the observation has no placement-zone channels for the
acting player. Those channels restated the mask more coarsely, and they cost two of the
three `PlacementOracle.point_grid` calls the observation made per seat per step. Measured:
6.66 `point_grid` calls per `env.step` before and 2.66 after. The measurement alternated the
two trees three times in one session, because the absolute numbers move by a third with
machine load while the comparison does not. The suite's own throughput report went 766 ->
953 `env.step`/s on the Rust engine and 475 -> 580 on `MockEngine` (medians of three
rounds).

## Constants are read, never copied

Every calibrated number lives once, in RoyaleSim's `data/calibration.json`, and carries the
client version it was measured on. This layer reads it:

- `Calibration` raises on a missing key rather than defaulting to something plausible.
- `Arena.load` and `RustEngine()` refuse to start when the numbers on disk differ from the
  numbers compiled into the engine, so a rebuild cannot be silently skipped.
- `data/derived/` is generated by RoyaleSim's extractors and is never hand-edited.
- The card table is not compiled in. The engine reads `data/derived/cards.json` from the
  RoyaleSim checkout it was built in, each time one is constructed. Regenerating that file
  changes the next engine's cards with no rebuild. `ROYALESIM_DATA_DIR` does not move it, so
  build in the checkout whose data you want. `RustEngine.config()` records which table it
  read: `cards_json_fnv1a64` and `cards_vintage`.

Where a rule is *not* measured, it says so in the data rather than hiding behind a plausible
default. `DeployRules` carries a `*_status` field per rule, and the calibration ledger lists
the unmeasured ones as open questions.

## Determinism

The engine is deterministic and seedable. A trace recorded from it re-verifies bit for bit:
`royalegym.replay.verify_trace` re-simulates it on a fresh engine and lists every
divergence. That is what makes a recorded battle usable as a regression test and a bug
report. `save_state`/`load_state` round-trips exactly, RNG included, so you can resume a
battle for curriculum starts or search.

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

`env.step/s` is the full Python stack: both players' observations and masks per step, 10
ticks per step at the default `decision_ms=500`. The engine figure steps 20 ticks per call,
with a state snapshot and a mask in between. Runs on one machine have printed anywhere from
10 000 to 30 000 engine ticks/s, so the figure that matters is the one your own run prints.
For scale, the engine alone measures roughly 25 000 ticks/s against the 20 ticks per battle
second the game runs at, which is about 1 250x real time. The pure-Python simulator this
project started from managed about 20 ticks/s on a full board of 156 entities, real time
exactly.

## Tests

```
cd RoyaleGym && ..\.venv\Scripts\python -m pytest -q   # 825 passed, 9 skipped, 1 xfailed at afd259d on build d872d792711934c2
..\.venv\Scripts\python -m ruff check royalegym tests examples   # All checks passed!
```

That count names the commit it was measured at AND the engine build, because it is only a
fact about one tree compiled against one ledger. Both move: the build digest changed three
times on 2026-09-22 alone, twice without any commit in this repo, because it covers
calibration.json and arena.json rather than code. A test checks it, and only on that commit: anywhere else
it says so rather than comparing two different trees.

The split moves between a workspace and a fresh clone even when the total does not. At 5ceeb2c, on
the earlier build f7628dd51148e4ce, this machine gave 793 passed and 8 skipped and a clone
gave 797 and 4, both 805. Those two are kept as a matched pair taken on one build; do not
compare either against the figure above, which is a different build.
More tests RUN on the clone, which is the opposite of what you would expect. The reason is
the card table: six of the skips here are two-engine comparisons that refuse to run while
this machine's compiled engine carries a newer table than MockEngine reads, and on a clone
both sides read the 2018 table, so the comparison measures the engines rather than the
data. Four of the eight is the measured flip; which four has not been pinned down.

Without `royalesim` built the Rust-backed tests skip, not pass. A `royalesim` build older
than `calibration.json` or `derived/arena.json` on disk is a failure, not a skip: `RustEngine()`
refuses it and names every differing key. `tests/test_viser.py` needs `royaleviser`
installed, because it round-trips a published frame through the viewer's decoder. It skips
otherwise.

## Module map

| Module | What it holds |
|---|---|
| `protocol.py` | the `Engine` contract and its value types (`BattleState`, `MatchSetup`, `Calibration`, `Arena`, `DeployRules`); `data_dir()` |
| `env.py` | `ClashParallelEnv` (PettingZoo), `ClashGymEnv` (Gymnasium, ids `royalegym/ClashRoyale-v0`, `royalegym/ClashRoyaleMock-v0`, `royalegym/ClashRoyaleRust-v0`), `ClashSelfPlayVecEnv` (N games as 2N agent slots: four games give a `spatial` batch of shape (8, 20, 32, 18) and eight masks), `make_gym_vec_env` |
| `mock_engine.py` | `MockEngine`, the pure-Python reference engine the RL layer is tested against |
| `rust_engine.py` | `RustEngine` over the compiled `royalesim` core; `SymmetricRustEngine` for the rotation-mirror gates |
| `obs.py` | `ObsBuilder`: `SpatialObsBuilder`, `EntityListObsBuilder` |
| `public_log.py` | `PublicLogMemory`: the vector's fair fields, except the board's four, rebuilt from a timed log of card plays with no engine running. It drives the same `MatchMemory` and `build_vector` the env uses |
| `action.py` | `ActionParser`: `TileActionParser`, `HalfTileActionParser`; `PlacementOracle` (the legality mask) |
| `reward.py` | `RewardFunction` and the shipped terms; `default_reward()` |
| `landing.py` | `landings()`: where a deploy actually put its units, read from `engine.state()`. A `DeployResult` carries the TAP, and the engine relocates a building whose footprint does not fit, so for buildings the two differ on half of accepted taps and by a full tile or more |
| `done_condition.py` | `DoneCondition`, `TerminationCondition`, `TruncationCondition` and the shipped conditions |
| `state_mutator.py` | `StateMutator` and the shipped mutators; `Snapshot` |
| `selfplay.py` | `Opponent`, `NoopOpponent`, `RandomLegalOpponent`, `CallableOpponent`, `OpponentPool` (uniform / latest / PFSP sampling, Elo bookkeeping) |
| `replay.py` | `ReplayRecorder`, `Trace`, `save_trace` / `load_trace`, `verify_trace` |
| `render.py` | the offline HTML replay page (`python -m royalegym.render trace.msgpack -o out.html`) |
| `viser.py` | `ViserPublisher`, the engine side of RoyaleViser |
| `evaluate.py` | `evaluate()`, bot against bot on both seats with a Wilson interval; `MatchResult` |
| `opponents.py` | four scripted opponents and `ladder()`, whose ordering is measured rather than assumed |

`tests/` holds the pytest suite: the env layer against `MockEngine`, and the Rust engine
through `RustEngine`. `docs/` holds this file, `background.md` and `observation-spec.md`.

## Everything else stays off the hot path

Rendering, logging and monitoring are out of process or off the tick loop:

- `royalegym.render` writes a self-contained offline HTML page from a trace, with a timeline
  scrubber and a hover tooltip per unit. It opens by double-click, with no server needed.
- `render_mode="ansi"` gives `env.render()` as a tile-resolution text board, Blue at the
  bottom: `T`/`t` crown towers, `B`/`b` buildings, `u`/`r` units (Blue upper case or `u`, Red
  lower case or `r`), `~` the river, `#` tiles nothing may be deployed on, with tick, elixir and
  crowns in the header line. It is a debugging aid. The viewer and the replay page are the
  pictures.
- `royalegym.viser.ViserPublisher` streams one frame per env step over UDP, and only while a
  viewer's heartbeat is fresh. An env nobody is watching pays one `if` per step.
- Metrics belong to the learner, which is a separate process from the envs.
