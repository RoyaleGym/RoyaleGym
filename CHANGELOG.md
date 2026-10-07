# Changelog

All notable changes to RoyaleGym. The format follows [Keep a Changelog](https://keepachangelog.com/),
and the version follows [Semantic Versioning](https://semver.org/). Until 1.0, a minor version may
change the observation's shape; each such change is listed here.


## 0.1.19 (2026-10-07)

### Fixed
- The hero planes (`heroes=True`) and `card_status`'s `enemy_seen_hero` no longer count a
  champion as a hero. royalesim up to 0.1.19 marks a champion's unit as a hero, and a
  champion's button is never spent, so an enemy Golden Knight or Archer Queen showed as an
  enemy hero with an unused charge for its whole life. Units of cards the catalogue calls
  champions are skipped.
- A flier a Vines catch holds on the ground counts as a ground troop for the hold: in the
  spatial troop channels and in the entity list's `flying` feature. A player sees it on the
  ground, and ground attacks reach it. Up to royalesim 0.1.19 it stayed in the air channel. It
  needs royalesim 0.1.20, which sets `STATUS_GROUNDED` (512) and lists its bits
  (`Engine.status_bits()`); with an older engine nothing changes.

### Added
- `SpatialObsBuilder(enemy_queue=True)` appends `enemy_queue_5_8`: the enemy's next card and the
  three after it, in order, as a player who watched its plays works them out (its last plays,
  reaching further back while a played slot waits for its refill). A position no play has
  reached yet reads "not deduced". Off by default.
- `CombinedReward.from_config` rebuilds a combined reward from its `config()`. A term from
  outside `royalegym.reward` is now named in that record by its full dotted path, so a term of
  your own that shares a name with one of this package's comes back as yours. This package's
  terms keep their bare names, so existing records are unchanged.
- `SpatialObsBuilder(effect_identity=True)`, with `spell_identity`: `spell_ids` grows from 4
  planes to 8. Spell cards' objects stay in the first four; the objects troops' and buildings'
  units leave (an Evo Cannon's bombs, a death bomb, an Evo Firecracker's fireworks) move to
  four more (`obs.EFFECT_ID_PLANES`). A player tells a cast Fireball from those. Off by default.

### Changed
- `obs.SAME_UNIT_ALIASES` merges three more pairs a player sees as one unit: the Goblins
  card's and the Goblin Gang's goblins with the Goblin Barrel's (as Goblin), the Goblin Hut's
  waves with Spear Goblins, and the second of the Three Musketeers with the first. Each differs
  from its base only in a timing no player reads off the unit, listed in the new
  `obs.SAME_UNIT_ALIAS_DIFFERENCES`. With `unit_aliases=SAME_UNIT_ALIASES` the vocabulary has
  three fewer names, and `unit_ids_digest` changes.
- The `spell_ids` and `SpellState.card_id` descriptions say what the engine reports: every live
  spell object, under the card that made it. That includes effects troops and buildings leave,
  such as the Evo Firecracker's fireworks or a Balloon's death bomb. docs/observation-spec.md
  gains a `spell_ids` section.
- royalesim 0.1.20: a flier held on the ground reports `STATUS_GROUNDED`, the evolved and
  hero bits follow the unit through its forms and leave champions, and the evolved Goblin
  Cage's brawler plays at the plain brawler's hp. royalelearn 0.5.12 and royaleimitate
  0.2.10 save `environment.json` beside every bot, so `Learner.load_env` rebuilds its env.


## 0.1.18 (2026-10-06)

### Fixed
- The action mask offers an ability button only when the engine would take the press. Some
  buttons wait on more than their row shows: the Hero Mega Minion's warp is ready only a while
  after the hero appears, and only while it has a target. The mask offered it from the first
  tick, and the engine refused it.

### Added
- `SpatialObsBuilder(unit_identity=True)` adds `unit_ids`: the type of each unit on the board,
  beside `card_ids`, which names the card that produced it. A skeleton reads "Skeleton" whether
  the Witch or a Tombstone made it. Off by default, so existing observations and configs do not
  change. Needs royalesim 0.1.17, which says each unit's type (`EntityState.unit_type`,
  `RustEngine.unit_types()`); saved battles record the type list.
- `unit_aliases`: write a unit the game's data keeps on a row of its own as its base type, when
  the two are the same unit to a player. `obs.SAME_UNIT_ALIASES` lists seven such rows (the
  Graveyard's skeleton is a Skeleton, a Tri-Wizard a Wizard, ...); a test checks each pair
  still has identical stats. `builder.unit_ids_digest` names the vocabulary the ids count in.
- `SpatialObsBuilder(button_index=True)` adds each own ability button's card and status by its
  INDEX (`own_button_cards` and three `*_by_index` fields), the index the action space presses.
  Which card a button is follows the deck order, and the card-keyed fields could not say it.

### Changed
- royalesim 0.1.17 says each unit's own type (`EntityState.unit_type`), which `unit_ids` reads,
  and an ability button's ready flag is now the engine's own answer.
- royalelearn 0.5.10 and royaleimitate 0.2.9 store and read `unit_ids`.

## 0.1.17 (2026-10-05)

### Fixed
- The action mask offers no play once a tied level overtime has run out. The engine accepts no
  play from that moment while the tiebreak decides the match, but the mask waited for the
  match to be over and kept offering every card, so each play in the tiebreak was refused.
  A training run that checks the engine accepted every play the mask allowed stopped there.

### Changed
- royalesim 0.1.14: units a death spawns (a Golem's Golemites) plan their route from where they
  appear.

## 0.1.16 (2026-10-05)

### Fixed
- A battle with an evolved card no longer crashes the viewer. The engine sends four values per
  evolution row and RoyaleGym's viewer feed read three, so watching such a battle live stopped
  with "too many values to unpack". royaleviser 0.1.2 fixes the same in playing back a saved
  battle.

### Changed
- royalesim 0.1.13: a unit already doomed by the shots in flight is judged by homing shots only,
  and a ranged unit does not fire at a target that has moved well beyond its reach. Battles can
  end differently.
- royalelearn 0.5.9: `doctor.vram_fraction` sets the GPU memory cap from a config.

## 0.1.15 (2026-10-04)

### Fixed
- royalelearn 0.5.8: training on an NVIDIA card on Windows starts again. From 0.5.5 to 0.5.7 every
  such run stopped at start-up. `pip install --upgrade "royalegym[all]"` brings the fix.
- An install from PyPI now records which card table its engine uses. The engine there carries its
  table inside, and `RustEngine.card_table_stamp()` (in `config()` and in every saved battle)
  said "unavailable". It now gives that table's hash and version, marked "embedded".

### Changed
- royalesim 0.1.12: a Hunter's pellets reach a building's whole square, and the engine draws the
  Hunter's random delays the way the game does. Battles with a Hunter can end differently.

## 0.1.14 (2026-10-04)

### Added
- Start episodes from saved moments of a battle. `ClashParallelEnv.snapshot()` saves the engine's
  state and what each seat's observation remembers of the match (the cycle, the counted elixir,
  the cards seen, the enemy's forms). A reset from it shows each seat the observation it had,
  and the battle goes on exactly as the original would under the same actions.
- `Snapshot` has a `seat`, a `max_ticks` and a `tag`. `max_ticks` (or the reset option
  `max_ticks`) ends an episode as a truncation once that many ticks have passed. The reset's info
  gives `start_seat` and the episode's last info gives `start_tag`.
- `SnapshotStateMutator` draws from a bank by weight (one per snapshot, or one per tag) and by
  seat (`seat="blue"` draws only the starts made for Blue or for either seat). It reads a bank
  from a file: `save_snapshots` and `load_snapshots`.

### Changed
- A release reaches PyPI once it passes its install test on Windows, macOS and Linux with
  Python 3.12 to 3.14, and not before. That test also checks that pip installs the version
  under test.

### Fixed
- royalesim 0.1.11: with `ShuffleMode.MIRRORED` and per-card `levels`, each card now plays at its
  own level. Before, the levels stayed in setup order while the cards were shuffled, so a card
  could play at another card's level. A new test plays all eight cards on both sides and checks
  every unit's level.

## 0.1.13 (2026-10-04)

### Changed
- Every package is on PyPI: install with plain `pip install "royalegym[all]"`.
- royalelearn 0.5.5: on Windows a CUDA run uses at most 80% of the card by default
  (`Learner(vram_fraction=...)`), so training no longer slows down by spilling into system RAM.

## 0.1.12 (2026-10-03)

### Changed
- royalesim 0.1.10: the Golden Knight's dash can be used once per deploy, as in the game. After it,
  his button stays off for the rest of his life. Battles with him can end differently.

## 0.1.11 (2026-10-03)

### Changed
- royalesim 0.1.9: the engine reports each side's deck and each card's form, so
  `PlayerState.deck` / `PlayerState.forms`, and `SpatialObsBuilder(card_status=True)`, also work
  for a battle restored with `load_state`.

## 0.1.10 (2026-10-03)

### Changed
- royalesim 0.1.8: units report their build-up, a tunneller's landing point and the ticks left in
  an ability, four more status bits (a Clone's copy, an ability winding up or running, fully
  charged), and the catalogue states each card's evolution cycle.
- `own_hp_frac` / `enemy_hp_frac` (`unit_status=True`) are now in thousandths, so every spatial
  plane stores exactly as a whole number of thousandths.
- MockEngine states no evolution cycle (`CardInfo.evo_cycle` None): it plays no evolutions.

### Added
- `reveal` can be given as a dict of `Reveal`'s field names, so a JSON config can set it:
  `"reveal": {"enemy_elixir": true}`. Unknown names are refused.
- `SpatialObsBuilder(card_status=True)`: every card's evolution and hero status, own and opponent.
  Your whole deck from the first frame, each card's form (per card and per hand slot), every
  evolved card's charge wherever it sits, and each ability button by its card. For the opponent,
  counted from the plays you have seen: which cards were played evolved or as heroes, each
  card's evolution charge, a hero's used charge and a champion's cooldown. Needs royalesim 0.1.8.
  Off by default.
- `PlayerState.deck` and `PlayerState.forms`: each side's deck and each card's form.
- `SpatialObsBuilder(unit_actions=True)`: fourteen planes, own and opponent, of what units are
  doing: build-up (a Prince's charge, a Sparky's load, an Inferno's ramp), fully charged, winding up
  or running an ability and the ticks it has left, a Clone's copies, and where a tunneller will come
  up. Needs royalesim 0.1.8. Off by default.
- `EntityState` gets values for `charge`, `dest_x`, `dest_y` and `ability_ticks`, and
  `CardInfo.evo_cycle` gets its values, from royalesim 0.1.8.

## 0.1.9 (2026-10-03)

### Changed
- royalesim 0.1.7: each side can play its own card and tower levels, and the engine reports which
  waiting commands ran in each step. royalelearn 0.5.4 and royaleimitate 0.2.6 store the new
  `spell_ids` observation key.
- `SpatialObsBuilder(spell_aim_after_ticks=0)` now shows every opponent spell's target from the
  first tick the spell exists, a thrown spell still waiting to move included.

### Fixed
- Under a command delay, the count of the opponent's elixir charged an ability press twice: once
  when the press was accepted, and again when its button flipped as it ran on a later step. With
  royalesim 0.1.7 the count charges exactly the presses that ran, so a press the engine refuses
  when it comes to run (its hero died while it waited) is never charged.

### Added
- `MatchSetup(levels=[blue, red], tower_levels=[blue, red])` and the same two arguments on
  `DefaultStateMutator`: each side's card levels (one per deck card, or empty for the engine's) and
  crown tower level. Needs royalesim 0.1.7; MockEngine, which has no levels, refuses them.
- `SpatialObsBuilder(heroes=True)`: three planes, `own_hero`, `enemy_hero` and
  `enemy_hero_unspent`, where each side's hero units stand and which of the opponent's still have
  their one ability charge. Off by default.
- `SpatialObsBuilder(spell_identity=True)`: a `spell_ids` key, uint8 [4, 32, 18], naming which card
  made each live spell object, a cast spell or an effect a troop or building left, in the `card_ids`
  vocabulary: own and opponent spell objects at their centre tile, own ones at their aim tile, and
  the opponent's at their aim tile once `spell_aim_after_ticks` allows. Needs `card_identity=True`
  and `spell_aim_after_ticks`. Off by default.
- `SpatialObsBuilder(unit_status=True)`: eighteen planes, own and opponent: shield hp left, units
  under a Rage, units slowed by cold, units whose target is a crown tower or another building, the
  hp fraction of each tile's strongest unit, and invisible, tunnelling and hidden units. Off by
  default.
- `EntityState` gains `charge`, `dest_x`, `dest_y` and `ability_ticks`, and `CardInfo` gains
  `evo_cycle`, all "not reported" until an engine sends them; four more status bits
  (`STATUS_CLONE`, `STATUS_WINDUP`, `STATUS_ABILITY_ACTIVE`, `STATUS_CHARGED`).

## 0.1.8 (2026-10-03)

### Changed
- royalesim 0.1.6: battle logic as measured in the game: crown-tower target ties, jump landings,
  Evo Musketeer snipes, a hooked unit's release, chase limits, death bombs and headings while
  casting. No change to installing or calling the engine; battles can end differently.

## 0.1.7 (2026-10-03)

### Changed
- royalesim 0.1.5: every unit on the board reports the card whose play put it down. A
  Tombstone's skeletons report the Tombstone, a hut's goblins report the hut, the Tri Wizards'
  three wizards report the Tri Wizards. The tower-damage and elixir-trade rewards are unchanged:
  a unit is still priced only when it is the unit its card's row describes, and a play of the
  Tri Wizards now totals its own 7 elixir.
- Every piece in `[all]` (and in its own extra) names a minimum version, so
  `pip install --upgrade "royalegym[all]" --find-links ...` moves each one. Before, an upgrade
  from an older release page could keep the old engine.

## 0.1.6 (2026-10-03)

### Added
- Clone human players: `[imitate]` and `[all]` need `royaleimitate[replays]` 0.2.5, whose
  `from_replays` turns human games from the public IL_Replay dataset into the rows `clone`
  reads. It brings pyarrow and huggingface_hub with it.

## 0.1.5 (2026-10-02)

### Changed
- A level overtime ends as the game ends it (royalesim 0.1.4): no card can be played from
  overtime's end, two ticks later every troop, building and spell leaves the board, and from
  3.35 s past the end every crown tower loses the same hp each tick until one falls; the crowns
  then decide. Towers exactly level drain once and the match is a draw 4 s later. A match can
  now run past tick 6000 (to about tick 6200). `MockEngine` runs the same rule when its
  calibration selects it.
- `[learn]` and `[all]` need royalelearn 0.5.3, whose start-up check allows a match past tick
  6000.

### Added
- `SpellState.ticks_flown`: how many ticks a thrown spell has flown (-1 from an engine before
  royalesim 0.1.4). `SpatialObsBuilder(spell_aim_after_ticks=k)` reads it when the engine reports
  it, so each spell object is timed exactly.

## 0.1.4 (2026-10-02)

### Fixed
- A random deck (`DefaultStateMutator` with no decks, `make_env(deck="random")`, a deck
  curriculum with no pool) holds at most one champion, as the ladder allows. It could hold four,
  and the engine refused the battle. A draw with two or more champions is drawn again, so about
  8% of seeds now deal a different random deck than 0.1.3 did; every other seed deals the same.
- Starting a bot from a saved one (RoyaleImitate's `warm_start`) works from the install line:
  `[learn]` and `[all]` need royalelearn 0.5.2, which no longer refuses a package installed from a
  wheel.
- `[imitate]` and `[all]` need royaleimitate 0.2.3, whose `public_log` waits for the engine's
  hand refill timer as royalesim 0.1.3 does.

### Added
- `CardInfo.champion`: whether a card is a champion, from the engine's catalogue (or, for
  royalesim 0.1.3 and before, its list of champions).
- `play_battle` takes the scripted bots by name (`"first-affordable"`, `"defend"`, `"push"`,
  `"patient"`), the names RoyaleImitate's `record` takes for a teacher.

## 0.1.3 (2026-10-02)

### Fixed
- The observation's next cards (positions 6 to 8) stay right while a played card's hand slot waits
  for its refill: the game refills one empty slot per period, so a slot can show no card for a few
  ticks after a quick second play.

### Added
- `SpatialObsBuilder(spell_aim_after_ticks=k)`: the plane `enemy_spell_aim_seen`, an enemy spell's
  landing point once a player could read it off the screen. A thrown spell shows it after k ticks of
  flight (counted from when it starts moving); a rolling or area spell from the first sight. Off by
  default.

### Changed
- `[imitate]` and `[all]` need royaleimitate 0.2.1, which records battles, clones a bot from them,
  and saves the clone so `Learner.load_policy` can play it.
- The release page carries royalesim 0.1.3, royalelearn 0.5.1 and royaleimitate 0.2.1.

## 0.1.2 (2026-10-02)

### Fixed
- Every "the engine is not installed" message points at the install page, whose line works before
  PyPI, instead of a `pip install` line that does not; `make_env` uses the same words.
- `MockEngine()` in an installed engine (which carries no raw 2018 card tables) says so and names the
  real engine, instead of failing on a missing file.

### Added
- `.github/workflows/pypi.yml`: publishing to PyPI from a version tag, switched off until the PyPI
  side is ready.
- The fresh-user test runs on Python 3.12, 3.13 and 3.14.

### Changed
- The release page carries royaleviser 0.1.1, whose viewer uses pygame-ce (it installs on Python
  3.14). If you installed an earlier release, run `pip uninstall pygame` before upgrading.

## 0.1.1 (2026-10-02)

### Fixed
- `Learner(viser=True)` streams to the viewer on every OS: `ROYALEVISER=1` (or `true`, `on`, `yes`)
  now means the default address, 127.0.0.1:9870. It was read as port 1, which Linux and macOS refuse
  and where no viewer looks on Windows.
- A viewer address that cannot be used says why; only a port already taken is reported as taken.
- The quickstart trains on macOS: the learner's shared memory names fit macOS's limit
  (royalelearn 0.4.2 is now required).

### Added
- `tools/stage_release.py` stages a release page, each package from its newest version tag.

## 0.1.0 (2026-10-02)

### Added
- `pip install "royalegym[all]"` installs every piece: the engine, the learner, the viewer and
  imitation. The extras `[sim]`, `[learn]`, `[viser]` and `[imitate]` install one each.
- `make_env()`: a two-seat battle in one call, with a starter deck and tower damage as the reward.
- `play_battle()`: play one battle with a trained bot and save it for the viewer.
- `examples/quickstart.py`: one file that names every piece of the env (engine, state mutator,
  obs builder, action parser, rewards, end conditions) and trains on the GPU. Watch it live
  with `royaleviser`.
- `SpatialObsBuilder(evolutions=True)`: which hand cards play evolved now, and the evolved units
  on the board. `evolution_progress=True` adds how far each counter has got.
- `DefaultStateMutator(decks=...)` takes card names.
- `EntityKind`, `to_own`, `MatchSetup`, `deck_ids`, `Winner` and `TowerSlot` import from
  `royalegym`, so every page can say everything does.
- `royalegym.__version__`, and type hints that type checkers read (`py.typed`).
- The command delay (`command_delay_ticks`): a play runs some ticks after the tap, as in the game.
- Ability buttons for heroes and champions (`TileActionParser(ability_buttons=True)`).
- Recorded battles carry each side's evolution and ability-button state.

### Changed
- The observation vector is 12n + 43 wide (was 12n + 37): a third elixir-rate slot for triple
  elixir late in overtime, and the seat's own waiting commands. Retrain or re-convert anything
  built on the old width.
- The engine's data is found through the installed engine, so no RoyaleSim clone is needed.
