# Observation spec

This page is for contributors writing or reading observation code. It lists every
channel and every vector slot a `royalegym` observation builder writes, the range each
one takes, and whether it is **fair** (something a person watching the match could
write down) or a **reveal**, read out of the half of the state a player cannot see.

The code is `royalegym/obs.py`. This file is the reference. `vector_layout()`,
`vector_offsets()`, `spatial_channels()` and `ObsBuilder.channel_names()` are the same
thing in a form a program can read, and the test suite asserts the two agree.

Throughout, **n** is the number of cards in the engine's catalogue
(`len(engine.cards())`). Everything is in the **acting seat's own frame**: Red's board
is rotated 180°, Red's towers are "own". A mirrored battle gives the other seat a
bit-identical observation, which is why one policy can play both seats.

---

## 1. The fair / reveal split

A builder takes a `Reveal`, a frozen dataclass whose five fields all default to
`False`:

```python
SpatialObsBuilder()                                   # fair
SpatialObsBuilder(reveal=Reveal(enemy_elixir=True))   # cheating, and says so
```

**An enabled field adds channels or slots; it is never present-but-zero.** A fair
observation and a cheating one do not have the same width, so a checkpoint cannot
quietly be trained with one and evaluated with the other. Fair fields come first and
keep their indices, so turning a reveal on never moves a fair feature.
`ClashParallelEnv.config()` records the `Reveal` for the checkpoint.

| `Reveal` field | what it opens | cost |
|---|---|---|
| `enemy_elixir` | the opponent's bar, read from the state | **no new slot**: it swaps the source of the slot that already holds the count (§4) |
| `enemy_hand` | the opponent's four hand slots | +4(n+1) vector slots |
| `enemy_next_card` | the opponent's cycle position 5 | +(n+1) vector slots |
| `enemy_deck` | as much of the opponent's deck as the state holds | +n vector slots |
| `enemy_spell_aim` | where the opponent's live spells will land | +1 spatial channel; +2 appended columns on every entity-list `spells` row |

`own_spell_aim` is **not** gated. The player chose where to throw their own spell.

---

## 2. `SpatialObsBuilder`: `spatial`, float32 `[C, 32, 18]`

C = 20 fair, +1 with `Reveal.enemy_spell_aim`. Entities are rasterised into the tile
containing their centre in the own frame. Every plane is clipped to `[0, 64]`
(`SPATIAL_CLIP`).

| # | channel | range | meaning | fair? |
|---|---|---|---|---|
| 0 | `own_ground_troops` | 0..64 | count of own non-flying troops in the tile | fair |
| 1 | `own_air_troops` | 0..64 | count of own flying troops | fair |
| 2 | `own_buildings` | 0..64 | count of own **buildings** (crown towers excluded) | fair |
| 3 | `own_towers` | 0..64 | count of own crown towers by centre | fair |
| 4 | `own_hp` | 0..64 | sum of own entity hp / 1000 | fair |
| 5 | `enemy_ground_troops` | 0..64 | as 0, for the opponent | fair |
| 6 | `enemy_air_troops` | 0..64 | as 1 | fair |
| 7 | `enemy_buildings` | 0..64 | as 2 | fair |
| 8 | `enemy_towers` | 0..64 | as 3 | fair |
| 9 | `enemy_hp` | 0..64 | as 4 | fair |
| 10 | `own_deploying` | 0..64 | own entities still inside their deploy timer | fair |
| 11 | `enemy_deploying` | 0..64 | as 10 | fair |
| 12 | `water` | 0..1 | fraction of the tile's four half-cells that are water (static) | fair |
| 13 | `no_deploy` | 0..1 | fraction flagged no-deploy (static) | fair |
| 14 | `enemy_troop_zone` | 0 or 1 | where the OPPONENT could put a troop right now | fair |
| 15 | `own_spells` | 0..64 | own live spell objects by current centre | fair |
| 16 | `enemy_spells` | 0..64 | the opponent's live spell objects by current centre | fair |
| 17 | `own_spell_aim` | 0..64 | own live spells by aim point (landing point; a Log's roll end) | fair |
| 18 | `own_stunned` | 0..64 | own entities with `stun_ticks > 0` | fair |
| 19 | `enemy_stunned` | 0..64 | as 18 | fair |
| 20 | `enemy_spell_aim` | 0..64 | the opponent's live spells by aim point | **reveal** |

Channels 0–11 are integer sums converted to float32 exactly once, so no plane can
depend on the engine's entity list order (see `obs.py`, NO FLOAT MAY DEPEND ON ENTITY
LIST ORDER).

`ObsBuilder.spatial_layout()` returns `(name, is_static)` per plane. **Static** means a
function of the arena alone (`water` and `no_deploy`, and nothing else), so a consumer
that stores observations can hold those two once instead of once per transition. Read
the declaration rather than deciding by sampling. `own_towers` and `enemy_towers` are
constant on any sample in which no tower falls, because a crown tower never moves. A
consumer that concluded "static" from that would never see a tower destroyed. A test
asserts exactly that.

### What is deliberately not here

* `own_troop_zone` and `own_building_zone` were deleted. The action space is
  `Discrete(2305)` = no-op + 4 hand slots × 18 × 32 tiles, so the **mask already
  states per-slot, per-tile legality exactly**. The zones were a coarser restatement
  of it. They cost two of the three `PlacementOracle.point_grid` calls the
  observation made per seat per step. Measured: 6.66 → 2.66 `point_grid` calls per
  `env.step` (both seats). The measurement alternated the two trees three times in one session, because the absolute numbers move by a third with machine load while the comparison does not. The suite's own throughput report went 766 -> 953 `env.step`/s on the Rust engine and 475 -> 580 on `MockEngine` (medians of three rounds).
* `enemy_troop_zone` stays. It is about the opponent's options and is in no mask.
* A knockback channel. Under `knockback.DURATION_MS = 0` the push is instant and the
  timer reads 0 between ticks, so the plane would be a constant zero no coverage
  guard could check. `knockback_ticks` survives as a per-entity feature.

### An open question: stun is nearly invisible at 500 ms decisions

A Zap's stun is 10 ticks and a decision is 10 ticks. Measured on the Rust engine: a
Zap cast at tick k of a decision leaves `stun_ticks` = k at the **one** observation
that follows (1, 4, 6, 10 for k = 0, 3, 5, 9) and 0 at every observation after. So
`own_stunned` / `enemy_stunned` fire for at most one step per Zap, and the entity
row's `stun_ticks / 100` reads 0.01 to 0.10 for that one step.

The features stay as "is stunned **now**". That is what the engine reports, and it is
what the seat-flip and cell-by-cell tests can check exactly. "Was stunned since the
last observation" is the feature a policy could actually use, but it is a different
thing: it depends on the decision rate and not only on the state. Recorded here rather
than changed quietly.

## 3. `mask_planes`, int8 `[4, 32, 18]`, both builders

The flat `action_mask` with index 0 (the no-op) removed, reshaped. The action space
is laid out as `1 + slot * ny * nx + y * nx + x`, so this is a **view**, not a
recomputation: `mask_planes[slot, y, x] == action_mask[encode(slot, x, y)]`.

`action_mask`, int8 `[2305]`, is still there for the policy head. The info dict no
longer repeats it. `ClashParallelEnv.state()` leaves out both. Legality is not
state, and a centralised critic does not need 2 304 duplicated numbers per seat.

A parser whose action space is not a grid returns `None` from `mask_plane_shape()` and
the key is simply absent.

---

## 3b. `card_ids`, uint8 `[2, 32, 18]`, `SpatialObsBuilder` only (behind a flag, default off)

AGREED WITH learn BEFORE IT IS WIRED, because their stem does an embedding lookup on these
and most of the ways to get it wrong fail silently rather than loudly.

`spatial` cannot tell a Giant from a Knight standing on the same tile: it carries no
per-unit card identity. That is a ceiling on play quality and it is invisible in any
throughput comparison, which is why D2 was nearly decided on the wrong axis. These planes
close it without a new trunk.

| | |
|---|---|
| key | `card_ids`, its OWN key, not a slice of `spatial` |
| shape | `[2, 32, 18]` — plane 0 own, plane 1 enemy, in the seat's own frame |
| dtype | `uint8`, declared `Box(0, vocab - 1, dtype=np.uint8)` |
| vocab | `num_cards + 2`, from the LOADED card table (102 here, 68 on a 2018 clone) |

**Why its own key and not a channel of `spatial`.** `spatial` is declared
`Box(0.0, SPATIAL_CLIP, float32)`, and a learner's codec picks exact-versus-scaled-half
storage from the declared bounds. A card id inside a float box is stored as a scaled half,
comes back as 6.997 instead of 7, and `.long()` reads card 6. Nothing fails; the policy
learns from a catalogue that is subtly wrong. `mask_planes` is already a separate
integer-valued key, so this follows a path the codec has met.

**The vocabulary.**

| index | meaning |
|---|---|
| 0 | empty tile |
| 1 | a crown tower, which is not a card |
| 2 + `card_id` | a unit of that card |

Index 0 is reserved so that "nothing here" and "the first card in the catalogue" are not
the same number. Crown towers get index 1 for exactly the same reason one step along: they
are the only entity class carrying no card (`card_id == EMPTY_CARD == -1`), and leaving
their tiles at 0 would make 0 mean both "empty ground" and "a tower stands here". It also
carries real information rather than a constant, because a destroyed tower's tile becomes
genuinely empty. Units a spell releases are NOT in this class; the engine reports them
under the releasing spell's catalogue id, so they are ordinary cards.

**The vocabulary size is not a constant.** It is a fact about the table the engine loaded:
100 cards on the 15.535 table, 66 on a 2018 clone. It is published in the EnvSpec so a
network sizes `nn.Embedding(vocab, C)` at construction. A literal would be wrong on a
clone, and an embedding sized by guessing truncates silently.

**Catalogue ids are POSITIONAL, and that is the trap this ships with a guard against.**
Making one more card loadable renumbers every later id, so an embedding table indexed by
those ids would have the same checkpoint reading a different game afterwards with nothing
failing. So the planes ship with an explicit `card_names` list recorded in the run
identity: a renumber is then a REFUSAL rather than a quiet reinterpretation. If the
vocabulary ever exceeds 255, construction refuses and names the storage decision instead of
widening the dtype silently.

**Ties are broken by lowest `uid`, and this is not cosmetic.** `BattleState.entities` is
NOT in uid order — a live battle gives `[0, 2, 4, 1, 3, 5]` — so "whichever comes first"
would make the plane a function of iteration order, and two runs of one seed could differ.
`uid` is unique for a whole battle and never reused.

**Spells in flight do not appear here**, and that is structural rather than a choice:
`BattleState.spells` is a separate list from `entities`, so a live spell has no entity and
no tile. This matches the same object being invisible to a learner's committed-elixir
accounting, which is the one outcome that creates no new discrepancy between the two.

**Default OFF until train's policy-head arm has run.** Flipping an observation shape under
a paired comparison invalidates both arms and looks like a result.

**FLIPPING THE FLAG IS A TWO-REPO CHANGE, NOT A CONFIG EDIT** (learn). A new observation
KEY is the right storage decision and it is not free on the learner's side, where a new
float channel inside `spatial` would have cost nothing. It touches four things there: the
codec learns a third key, since it reads `spatial` and `vector` and derives `mask_planes`
from the mask and has no general any-key path; the buffer's row layout and `row_bytes`
change, and that is the number the shared-memory rectangle is sized from; `ObsBatch` gains
a field, and so does every site that constructs one; and the stem does the lookup and
concatenates. So the flag is off by default on BOTH sides, and turning it on is a
coordinated change rather than a switch. The cost is worth paying: the alternative was a
card id stored as a scaled half, which fails silently, and silent is worse than work.


## 4. The flat `vector`, float32 `[12n + 37]` (fair)

All slots are clipped to `[0, 1]`. The offsets in the table are for n = 16, which is
`MockEngine`'s default catalogue and what the test suite runs on. It is **not** the
full card list, which is larger and gives a wider vector. Read offsets from
`vector_offsets(n, reveal)`. Never copy a number out of this table into code.

| slots (n=16) | field | size | range | meaning | fair? |
|---|---|---|---|---|---|
| 0 | `own_elixir` | 1 | 0..1 | own elixir / MAX_MANA | fair |
| 1 | `enemy_elixir` | 1 | 0..1 | opponent's elixir / MAX_MANA (**counted**, see below) | fair (source switches under `Reveal.enemy_elixir`) |
| 2–69 | `own_hand_cards` | 4(n+1) | 0/1 | hand slot card one-hot; index n = empty slot | fair |
| 70–73 | `own_hand_cost` | 4 | 0..1 | hand slot elixir cost / MAX_MANA | fair |
| 74–77 | `own_hand_affordable` | 4 | 0/1 | affordable right now | fair |
| 78–94 | `own_next_card` | n+1 | 0/1 | cycle position 5 | fair |
| 95–145 | `own_cycle_6_8` | 3(n+1) | 0/1 | cycle positions 6, 7, 8; index n = not deduced yet | fair |
| 146–161 | `own_deck` | n | 0/1 | own deck multi-hot, as deduced so far | fair |
| 162–178 | `own_last_card` | n+1 | 0/1 | last card own played; index n = none yet | fair |
| 179 | `own_ticks_since_play` | 1 | 0..1 | ticks since own last play / 600, clipped | fair |
| 180 | `own_elixir_leaked` | 1 | 0..1 | own elixir lost to the cap so far / 20, clipped | fair |
| 181–196 | `enemy_cards_seen` | n | 0/1 | cards the opponent has played at least once | fair |
| 197–212 | `enemy_possible_hand` | n | 0/1 | cards that could be in the opponent's hand now | fair |
| 213 | `enemy_plays` | 1 | 0..1 | opponent's plays this match / 40, clipped | fair |
| 214–216 | `own_tower_hp` | 3 | 0..1 | tower hp / max, `[king, left, right]` | fair |
| 217–219 | `enemy_tower_hp` | 3 | 0..1 | the same, in the opponent's own-frame slots | fair |
| 220–221 | `crowns` | 2 | 0..1 | own crowns / 3, enemy crowns / 3 | fair |
| 222–223 | `king_active` | 2 | 0/1 | own king active, enemy king active | fair |
| 224–226 | `clock` | 3 | 0..1 | regulation left / regulation, in overtime, overtime left / overtime | fair |
| 227–228 | `elixir_rate` | 2 | 0/1 | one-hot over 1x, 2x | fair |
| appended | `enemy_hand_cards` | 4(n+1) | 0/1 | the opponent's hand | **reveal** (`enemy_hand`) |
| appended | `enemy_next_card` | n+1 | 0/1 | the opponent's cycle position 5 | **reveal** (`enemy_next_card`) |
| appended | `enemy_deck` | n | 0/1 | the opponent's deck | **reveal** (`enemy_deck`) |

600, 20 and 40 are presentation constants (`PLAY_GAP_TICKS`, `LEAK_SCALE`,
`PLAYS_SCALE`) that decide where a feature saturates. They are not game numbers and
are not read from calibration.

### 12n + 37, not 12n + 36

The specification this rewrite was built to called the width 12n + 36. It is 12n + 37,
and the extra slot is real rather than an accident. The arithmetic, term by term:

| block | width |
|---|---|
| the previous layout | 5n + 30 |
| `own_deck` | + n |
| `own_cycle_6_8` | + 3n + 3 |
| `own_last_card` | + n + 1 |
| `enemy_cards_seen` | + n |
| `enemy_possible_hand` | + n |
| `own_ticks_since_play`, `own_elixir_leaked`, `enemy_plays` | + 3 |
| **total** | **12n + 37** |

The previous layout's 5n + 30 includes the one enemy-elixir slot, which is kept and
now holds the count, so it is not double-counted. A test asserts 12n + 37 directly.

### The counted enemy elixir

`enemy_elixir` is maintained by the builder, not read from the state, and this is the
one fair feature that needs saying carefully.

* It is **seeded once**, at the start of the match, from the opponent's bar. The
  starting amount is public, and a curriculum start that hands one side extra elixir
  is equally public. After that the state's copy is never read again.
* From there it is the engine's own arithmetic (`protocol.ElixirLaw`, derived from
  `calibration.json` and `globals.csv`): pay for each play seen, regenerate over the
  ticks that passed at the rate the clock says, clamp at the cap. All in integer
  "fine" units (one elixir is `lcm(regen 1x, regen 2x)` of them), because
  milli-elixir cannot carry the law. A tick is worth 17.857… milli at the shipped
  numbers, and a milli-space sum drifts inside one match.
* A **play** is a public event: a unit appears, a spell is cast. The builder reads it
  from the opponent's hand changing between two observed states, which names the same
  event and names the card exactly.

The result is bit-exact against the bar the engine keeps, which is why it belongs in
the fair set rather than being an estimate. `Reveal.enemy_elixir` swaps in the value
read from the state. A test plays a battle out and asserts the two agree at every
step, from both seats, on a busy game and on a quiet one (the quiet game is what
exercises the cap). Verified exact, per step, on: the opening, a `start_tick` that
crosses the 2x threshold, a start already in overtime, asymmetric starting elixir,
`decision_ms` of 50 (one tick per step) and 3000 (sixty), a randomised mid-game
curriculum, and a game quiet enough to sit at the cap.

**`MatchMemory.exact`** says when it cannot be. The same law runs on the player's own
bar, which is visible, so the count is checked every step against a number the memory
is not allowed to guess at. The moment the two disagree, `exact` goes False and stays
False for the match. Two things make that happen: a deck that repeats a card (a play
that swaps a card for itself changes no hand slot, so it is unseen; a real deck is
eight distinct cards), and an engine whose elixir law is not the one in
`calibration.json`. The builder then resyncs the own bar from the observed value so it
stops drifting. It leaves the enemy count alone, because the only way to repair it
would be to read it. A training run that wants the guarantee can assert `exact`.

### The cycle features

A card played goes to the back of an 8-card cycle. Hand is positions 1–4,
`next_card` is 5, and 6–8 are behind it.

* `own_cycle_6_8` starts unknown (all three one-hots point at index n) and learns one
  position per play, so the whole cycle is visible after three plays. `own_deck` fills
  in the same way.
* `enemy_possible_hand` is the same rule applied to the opponent: the last four cards
  they played are exactly the four behind their hand, so those are 0 and everything
  else is 1. "Everything else" is the whole catalogue until eight distinct cards have
  been seen, at which point their deck is known and the answer narrows to it.

### The builder is stateful

These features make the builder carry a `MatchMemory` per seat. Two guards keep one
episode out of the next. `ObsBuilder.reset(state)`, which the env calls, re-seeds both
seats. And `MatchMemory.observe` re-seeds whenever the clock moves **backwards**,
which can only be a new battle. `observe` is also idempotent by tick, so building the
same state twice cannot drift. That is what lets the seat-flip and list-order gates
build the same state dozens of times.

Tests replay a seeded episode twice in the same env and require the two observation
sequences to be identical, with a plant that removes both guards and shows the
difference.

---

## 5. `EntityListObsBuilder`

For attention / transformer policies. `entities [N, 18 + n + 1]`, `spells [M, 14 + n]`,
plus the same `vector`, `action_mask` and `mask_planes`. Column names are in
`ENTITY_FEATURE_NAMES` and `SPELL_FEATURE_NAMES`; the full per-column meaning is in
the class docstring.

Rows are sorted by a key made of **every field the row is built from**, so two rows
that tie are identical and the order is seat-invariant whatever order the engine
listed them in.

The only reveal here is the aim point. Columns 9 and 10 are the aim of the **viewer's
own** spells and are 0 on an enemy row; `Reveal.enemy_spell_aim` **appends two more
columns** for the opponent's, so the space really does change width.

The sort key puts the aim **last**, and that is not cosmetic. Row order is observable
(it decides whose delay and hit count appear first), and the key must still name every
field a row is built from, so the aim cannot leave it. With the aim ranked early, two
enemy spells alike in everything visible came out in an order set by where they were
going. Two states differing *only* in two hidden aim points gave delay columns
`[0.03, 0.07]` and `[0.07, 0.03]`. With the aim last, hidden data can only order rows
whose every visible field ties, and those rows write the same numbers. A test and a
plant hold this.

---

## 6. Ground truth

The observation is built from `BattleState`, which is whatever the engine reports
(`royalegym/protocol.py`). Where a channel is described as matching the live game, the
comparison is against traces recorded by RoyaleLive, the client instrument that
records ground-truth traces from the real game.

MockEngine resolves spells inside a tick and models no status effects, so on it the
spell and stun channels and the `spells` array are always zero. The Rust engine's
tests are where those channels are exercised on real battles.
