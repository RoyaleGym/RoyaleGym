# Changelog

All notable changes to RoyaleGym. The format follows [Keep a Changelog](https://keepachangelog.com/),
and the version follows [Semantic Versioning](https://semver.org/). Until 1.0, a minor version may
change the observation's shape; each such change is listed here.

## 0.1.3.1 (2026-10-03)

A backport onto 0.1.3 of one fix from 0.1.9. Nothing else changes.

### Fixed
- Under a command delay, the count of the enemy's elixir charged an ability press twice: once when
  the press was accepted, and again when its button flipped as it ran on a later step. The count now
  charges exactly the presses the env accepted.

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
- The release page carries royalesim 0.1.2, royalelearn 0.5.1 and royaleimitate 0.2.1.

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
