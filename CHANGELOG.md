# Changelog

All notable changes to RoyaleGym. The format follows [Keep a Changelog](https://keepachangelog.com/),
and the version follows [Semantic Versioning](https://semver.org/). Until 1.0, a minor version may
change the observation's shape; each such change is listed here.

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
