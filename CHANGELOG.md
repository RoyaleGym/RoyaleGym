# Changelog

All notable changes to RoyaleGym. The format follows [Keep a Changelog](https://keepachangelog.com/),
and the version follows [Semantic Versioning](https://semver.org/). Until 1.0, a minor version may
change the observation's shape; each such change is listed here.

## Unreleased

### Added
- `pip install "royalegym[all]"` installs every piece: the engine, the learner, the viewer and
  imitation. The extras `[sim]`, `[learn]`, `[viser]` and `[imitate]` install one each.
- `make_env()`: a two-seat battle in one call, with a starter deck and tower damage as the reward.
- `play_battle()`: play one battle with a trained bot and save it for the viewer.
- `examples/quickstart.py`: train a first bot in one file, then watch it.
- `royalegym.__version__`, and type hints that type checkers read (`py.typed`).
- The command delay (`command_delay_ticks`): a play runs some ticks after the tap, as in the game.
- Ability buttons for heroes and champions (`TileActionParser(ability_buttons=True)`).
- Recorded battles carry each side's evolution and ability-button state.

### Changed
- The observation vector is 12n + 43 wide (was 12n + 37): a third elixir-rate slot for triple
  elixir late in overtime, and the seat's own waiting commands. Retrain or re-convert anything
  built on the old width.
- The engine's data is found through the installed engine, so no RoyaleSim clone is needed.
