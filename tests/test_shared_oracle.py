"""One PlacementOracle for the builder and the parser, and blockers worked out once per state.

Since 2026-09-26 an observation builder bound to an env uses the action parser's oracle when
both were built from the same arena, rules and cards (``ObsBuilder.bind``), and the oracle
works out a state's blocker tuple once instead of once per ``grid_key`` call
(``PlacementOracle._blockers``). Both are speed changes and nothing else (train's ask). So
this file plays the same battles through an env built that way and through a reference env
whose builder and parser each hold their OWN oracle, recomputing blockers on every call:
the old path. It compares every observation key, masks included, byte for byte at every step.
"""

from __future__ import annotations

import numpy as np
import pytest

from royalegym import ClashParallelEnv, DefaultStateMutator
from royalegym.action import HalfTileActionParser, PlacementOracle, TileActionParser
from royalegym.mock_engine import MockEngine
from royalegym.obs import EntityListObsBuilder, SpatialObsBuilder
from royalegym.protocol import EntityKind
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available

DECK = ["Cannon", "Knight", "Archer", "Giant", "Minions", "Fireball", "Log", "Zap"]
ENGINES = {
    "mock": MockEngine,
    "rust": pytest.param(
        RustEngine, marks=pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
    ),
}


class OldOracle(PlacementOracle):
    """The oracle as it was: blockers sorted afresh on every ``grid_key`` call."""

    def _blockers(self, state):
        return tuple(
            sorted((e.x, e.y, e.radius) for e in state.entities if e.kind != EntityKind.TROOP)
        )


def make_env(engine_cls, parser_cls, builder_cls):
    engine = engine_cls()
    ids = {c.name: c.card_id for c in engine.cards()}
    deck = [ids[n] for n in DECK]
    return ClashParallelEnv(
        engine=engine,
        action_parser=parser_cls(),
        obs_builder=builder_cls(),
        state_mutator=DefaultStateMutator(decks=[deck, deck]),
    )


def as_old(env) -> None:
    """Give the env's parser and builder an oracle each, of the old kind."""
    old = [OldOracle(env.engine.arena(), env.engine.rules(), list(env.engine.cards()))
           for _ in range(2)]
    env.action_parser.oracle, env.obs_builder.oracle = old


def differences(engine_cls, parser_cls, builder_cls, seed: int, steps: int):
    new, old = make_env(engine_cls, parser_cls, builder_cls), make_env(
        engine_cls, parser_cls, builder_cls
    )
    assert new.obs_builder.oracle is new.action_parser.oracle, "the new env does not share"
    as_old(old)
    obs_n, _ = new.reset(seed=seed)
    obs_o, _ = old.reset(seed=seed)
    rng = np.random.default_rng(seed)
    bad: list[str] = []
    seen = {"steps": 0, "buildings_down": 0, "masked_plays": 0}
    for step in range(steps):
        for agent in obs_o:
            for k in obs_o[agent]:
                a, b = np.asarray(obs_n[agent][k]), np.asarray(obs_o[agent][k])
                if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                    bad.append(f"step {step} {agent} {k}")
        if bad or not new.agents:
            break
        s = new.battle_state
        seen["buildings_down"] += any(e.kind == EntityKind.BUILDING for e in s.entities)
        actions = {}
        for agent in new.agents:
            legal = np.flatnonzero(obs_n[agent]["action_mask"][1:]) + 1
            if len(legal) and rng.random() < 0.4:
                actions[agent] = int(rng.choice(legal))
                seen["masked_plays"] += 1
            else:
                actions[agent] = 0
        obs_n, *_ = new.step(actions)
        obs_o, *_ = old.step(actions)
        seen["steps"] += 1
    return bad, seen


@pytest.fixture(params=list(ENGINES.values()), ids=list(ENGINES))
def engine_cls(request):
    return request.param


@pytest.mark.parametrize("parser_cls", [TileActionParser, HalfTileActionParser])
@pytest.mark.parametrize("builder_cls", [SpatialObsBuilder, EntityListObsBuilder])
def test_one_shared_oracle_gives_the_observations_two_oracles_gave(
    engine_cls, parser_cls, builder_cls
):
    bad, seen = differences(engine_cls, parser_cls, builder_cls, seed=4, steps=240)
    assert not bad, f"{len(bad)} differences; first {bad[:4]}"
    assert seen["steps"] >= 200, seen
    assert seen["masked_plays"] >= 30, seen
    assert seen["buildings_down"] >= 20, seen


def test_a_builder_bound_to_another_rule_set_keeps_its_own_oracle():
    engine = MockEngine()
    parser = TileActionParser()
    parser.bind(engine)
    other = MockEngine()
    builder = SpatialObsBuilder()
    parser.oracle.rules = None  # the parser's oracle no longer matches the builder's engine
    builder.bind(other, parser)
    assert builder.oracle is not parser.oracle


def test_plant_a_grid_key_that_forgets_the_pitch_is_caught(monkeypatch):
    """What sharing adds, broken: the builder asks for tile grids and the half-tile parser for
    half-cell ones. With the pitch missing from the key, only a SHARED oracle can serve one
    as the other. The reference env's two oracles each see one pitch, so they are unaffected
    (a plant that broke both envs alike would pass unseen: this comparison is between them)."""
    real = PlacementOracle.grid_key

    def no_pitch(self, state, team, card, pitch_div):
        key = real(self, state, team, card, pitch_div)
        return None if key is None else (*key[:2], 0, *key[3:])

    monkeypatch.setattr(PlacementOracle, "grid_key", no_pitch)
    try:
        bad, _seen = differences(
            MockEngine, HalfTileActionParser, SpatialObsBuilder, seed=4, steps=240
        )
    except ValueError as e:  # a tile grid handed to the half-cell mask does not even fit
        bad = [f"raised {e}"]
    assert bad, "a grid shared across pitches passed the byte comparison"


def test_plant_a_blocker_memo_that_never_refreshes_is_caught(monkeypatch):
    """The per-state blocker memo, stuck on the first state: buildings placed later never
    reach the key, so a stale grid is served. The reference oracle recomputes every call."""

    def stuck(self, state):
        if self._blocker_memo[0] is None:
            self._blocker_memo = (state, OldOracle._blockers(self, state))
        return self._blocker_memo[1]

    monkeypatch.setattr(PlacementOracle, "_blockers", stuck)
    bad, _seen = differences(MockEngine, TileActionParser, SpatialObsBuilder, seed=4, steps=240)
    assert bad, "a stale blocker memo passed the byte comparison"
