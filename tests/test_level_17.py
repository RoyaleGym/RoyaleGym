"""Level 17 through the env (royalesim 0.1.27 on): a deck at 17 plays, the observation is the same
in shape, 18 is the engine's to refuse, and a Mirror of a level-16 card puts down a level-17 copy.

The game's max-level players field cards at 17 (a Common's 450 %). RoyaleGym checks only that a
level fits i32; which levels a card has is the engine's, so admitting 17 needed no change here.
These tests hold that a battle at 17 goes through reset, build and step unchanged, that the
engine's refusal of 18 reaches the caller by name, and that a Mirror's one-level-up copy of a
16 stands at 17 rather than being refused. ``tests/test_obs_reads_no_level.py`` holds the other
half: no observation or mask reads the level, so levels 1-16 observe exactly as before.

SKIPS
    Everything here needs the engine. Not a pass without it. On an engine before royalesim 0.1.27,
    which refuses level 17, these fail: they ship with the release that admits it.
"""

from __future__ import annotations

import numpy as np
import pytest

from royalegym.action import TileActionParser
from royalegym.env import ClashParallelEnv
from royalegym.obs import SpatialObsBuilder
from royalegym.protocol import BLUE, EntityKind, ShuffleMode
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available
from royalegym.state_mutator import DefaultStateMutator

pytestmark = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))

DECK = ["Knight", "Mirror", "Archer", "Goblins", "Giant", "Musketeer", "Arrows", "Zap"]


def _env(levels, shuffle=ShuffleMode.INDEPENDENT):
    return ClashParallelEnv(
        RustEngine(),
        action_parser=TileActionParser(),
        obs_builder=SpatialObsBuilder(card_identity=True),
        state_mutator=DefaultStateMutator(decks=[DECK, DECK], levels=levels, shuffle=shuffle),
    )


def _troop_levels(env) -> set[tuple[int, int]]:
    return {(e.team, e.level) for e in env.battle_state.entities if e.kind == EntityKind.TROOP}


def test_a_level_17_deck_plays_through_reset_build_and_step():
    env = _env([[17] * 8, [16] * 8])
    plain = _env([[16] * 8, [16] * 8])
    obs, _ = env.reset(seed=3)
    want, _ = plain.reset(seed=3)
    for agent in ("blue", "red"):
        assert obs[agent].keys() == want[agent].keys()
        for k in want[agent]:
            assert (obs[agent][k].dtype, obs[agent][k].shape) == (
                want[agent][k].dtype, want[agent][k].shape), k
        assert env.observation_space(agent).contains(obs[agent])
    rng = np.random.default_rng(3)
    seen: set[tuple[int, int]] = set()
    for _ in range(300):
        if not env.agents:
            break
        acts = {}
        for agent in env.agents:
            legal = np.flatnonzero(obs[agent]["action_mask"])
            legal = legal[legal != 0]
            acts[agent] = int(rng.choice(legal)) if len(legal) and rng.random() < 0.3 else 0
        obs, *_ = env.step(acts)
        seen |= _troop_levels(env)
    assert (0, 17) in seen, seen  # Blue's troops stand at 17
    assert (1, 16) in seen, seen


def test_level_18_is_the_engine_s_to_refuse_and_reaches_the_caller_by_name():
    with pytest.raises(ValueError, match="level 18"):
        _env([[18] * 8, [16] * 8]).reset(seed=1)


def _play(env, name: str, x: int, y: int) -> None:
    """Blue plays ``name`` at own-frame tile (x, y): asserts the move is legal first."""
    hand = env.battle_state.players[BLUE].hand
    ids = {c.name: c.card_id for c in env.engine.cards()}
    slot = hand.index(ids[name])
    p = env.action_parser
    action = 1 + slot * p.nx * p.ny + y * p.nx + x
    assert env.action_masks("blue")[action], f"{name} cannot go at ({x}, {y})"
    env.step({"blue": action, "red": 0})


def _wait_for(env, milli: int) -> None:
    for _ in range(200):
        state = env.battle_state
        if state.tick >= env.engine.rules().deploy_lockout_ticks and (
            state.players[BLUE].elixir_milli >= milli
        ):
            return
        env.step({"blue": 0, "red": 0})
    raise AssertionError(f"Blue never reached {milli} milli-elixir")


def test_a_mirror_of_a_level_16_card_puts_down_a_level_17_copy():
    """Unshuffled, Blue's opening hand is Knight, Mirror, Archer, Goblins: a Knight at 16, then
    the Mirror, whose copy is one level up."""
    env = _env([[16] * 8, [16] * 8], shuffle=ShuffleMode.NONE)
    env.reset(seed=5)
    _wait_for(env, 3000)
    _play(env, "Knight", 4, 10)
    _wait_for(env, 4000)  # the copy costs the Knight's 3 plus the Mirror's 1
    _play(env, "Mirror", 13, 10)
    knights = set()
    knight = next(c.card_id for c in env.engine.cards() if c.name == "Knight")
    for _ in range(6):
        knights |= {e.level for e in env.battle_state.entities
                    if e.team == BLUE and e.card_id == knight and e.kind == EntityKind.TROOP}
        env.step({"blue": 0, "red": 0})
    assert knights == {16, 17}, knights
