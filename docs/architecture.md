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
there is no `MutatorSequence` here.

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
(`royalegym.replay`), which is what makes a recorded battle usable as a regression test and
a bug report. `save_state`/`load_state` round-trips exactly, RNG included, so a battle can be
resumed for curriculum starts or search.

## Everything else stays off the hot path

Rendering, logging and monitoring are out of process or off the tick loop:

- `royalegym.render` writes a self-contained offline HTML page from a trace.
- `royalegym.viser.ViserPublisher` streams one frame per env step over UDP, and only while a
  viewer's heartbeat is fresh — an env nobody is watching pays one `if` per step.
- Metrics belong to the learner, which is a separate process from the envs.
