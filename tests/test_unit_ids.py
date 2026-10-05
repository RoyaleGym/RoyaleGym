"""``unit_ids``: which unit TYPE stands on each tile, beside ``card_ids``, which says which card
produced it (observation requirement row 22, owner 2026-10-05).

A summon carries its producer's card id: the Witch's skeletons read "Witch", Tombstone's read
"Tombstone", and a network that sees only ``card_ids`` learns one unit six times. The engine
says each entity's own type (``EntityState.unit_type``, an index into
``Engine.unit_types()``, royalesim 0.1.17 on). These tests use a MockEngine that says it the way
the engine does, so the planes are checked before and apart from any engine build: Goblins
dealt by the Goblins card and by the Goblin Barrel are one type under two producers.
"""

from __future__ import annotations

import msgspec
import numpy as np
import pytest

from royalegym.env import ClashParallelEnv
from royalegym.mock_engine import MockEngine
from royalegym.obs import UNIT_ID_EMPTY, UNIT_ID_OFFSET, SpatialObsBuilder, unit_id_planes
from royalegym.protocol import BLUE, RED, EntityKind, ShuffleMode
from royalegym.replay import ReplayRecorder
from royalegym.state_mutator import DefaultStateMutator

DECK = ["Goblins", "GoblinBarrel", "Knight", "Archer", "Giant", "Minions", "Fireball", "Zap"]
#: The unit each card puts down; two producers of one type, as the engine's data has them.
UNIT_OF = {"Goblins": "Goblin", "GoblinBarrel": "Goblin", "Knight": "Knight",
           "Archer": "Archer", "Giant": "Giant", "Minions": "Minion"}
UNITS = sorted({*UNIT_OF.values(), "KingTower", "PrincessTower", "Musketeer", "Valkyrie"})


class UnitTypedMock(MockEngine):
    """MockEngine that says each entity's unit type, in the engine's layout."""

    def unit_types(self) -> list[str]:
        return list(UNITS)

    def state(self):
        s = super().state()
        names = {c.card_id: c.name for c in self.cards()}

        def typed(e):
            if e.kind == EntityKind.KING_TOWER:
                unit = "KingTower"
            elif e.kind == EntityKind.PRINCESS_TOWER:
                unit = "PrincessTower"
            else:
                unit = UNIT_OF[names[e.card_id]]
            return msgspec.structs.replace(e, unit_type=UNITS.index(unit))

        return msgspec.structs.replace(s, entities=[typed(e) for e in s.entities])


def _env(engine=None, **builder) -> ClashParallelEnv:
    eng = engine or UnitTypedMock()
    ids = {c.name: c.card_id for c in eng.cards()}
    deck = [ids[n] for n in DECK]
    return ClashParallelEnv(
        eng, obs_builder=SpatialObsBuilder(card_identity=True, unit_identity=True, **builder),
        # Unshuffled: the opening hand is the deck's first four, Goblins and the Barrel among them.
        state_mutator=DefaultStateMutator(decks=[deck, deck], shuffle=ShuffleMode.NONE),
    )


def _play(env, name: str, x_tile: int, y_tile: int) -> None:
    """Blue plays ``name`` on its own half, at a tile in its own frame."""
    hand = env.battle_state.players[BLUE].hand
    ids = {c.name: c.card_id for c in env.engine.cards()}
    slot = hand.index(ids[name])
    parser = env.action_parser
    action = 1 + slot * parser.nx * parser.ny + y_tile * parser.nx + x_tile
    assert env.action_masks("blue")[action], f"{name} cannot go at ({x_tile}, {y_tile})"
    env.step({"blue": action, "red": 0})


def test_one_unit_type_under_two_producers() -> None:
    """Goblins from the Goblins card and from a Goblin Barrel: card_ids names two producers,
    unit_ids names one type."""
    env = _env()
    env.reset(seed=3)
    while env.battle_state.tick < env.engine.rules().deploy_lockout_ticks:
        env.step({"blue": 0, "red": 0})
    ids = {c.name: c.card_id for c in env.engine.cards()}
    seen: dict[int, set[int]] = {}
    for _ in range(80):
        hand = env.battle_state.players[BLUE].hand
        for name, x in (("Goblins", 3), ("GoblinBarrel", 14)):
            if ids[name] in hand and env.battle_state.players[BLUE].elixir_milli >= 3000:
                _play(env, name, x, 10)
        env.step({"blue": 0, "red": 0})
        obs = env._obs["blue"]
        for card in ("Goblins", "GoblinBarrel"):
            where = obs["card_ids"][0] == 2 + ids[card]
            if where.any():
                seen.setdefault(ids[card], set()).update(obs["unit_ids"][0][where].tolist())
        if len(seen) == 2:
            break
    goblin = UNIT_ID_OFFSET + UNITS.index("Goblin")
    assert seen == {ids["Goblins"]: {goblin}, ids["GoblinBarrel"]: {goblin}}, seen


def test_each_tile_shows_the_entity_card_ids_shows() -> None:
    """Two entities on one tile: both planes keep the lowest uid, so unit_ids and card_ids
    describe the same unit cell for cell, on both seats' frames."""
    env = _env()
    env.reset(seed=1)
    state = env.battle_state
    arena = env.engine.arena()
    towers = [e for e in state.entities if e.kind == EntityKind.PRINCESS_TOWER]
    a, b = towers[0], towers[1]
    knight = msgspec.structs.replace(a, uid=10_000, kind=EntityKind.TROOP, card_id=0,
                                     tower_slot=-1, unit_type=UNITS.index("Knight"))
    first = msgspec.structs.replace(b, uid=1, x=a.x, y=a.y, team=a.team)
    entities = [knight, first, *state.entities]
    for team in (BLUE, RED):
        units = unit_id_planes(entities, team, arena, len(UNITS))
        from royalegym.obs import card_id_planes

        cards = card_id_planes(entities, team, arena, len(env.engine.cards()))
        same = (units == UNIT_ID_EMPTY) == (cards == 0)
        assert same.all(), "the two planes cover the same tiles"
    own = unit_id_planes(entities, a.team, arena, len(UNITS))
    assert UNIT_ID_OFFSET + UNITS.index("Knight") not in own, "the higher uid lost the tile"


def test_the_key_its_space_and_the_config() -> None:
    env = _env()
    obs, _ = env.reset(seed=2)
    space = env.observation_space("blue")["unit_ids"]
    assert space.dtype == np.uint8
    assert space.shape == (2, env.engine.arena().tiles_y, env.engine.arena().tiles_x)
    assert int(space.high.max()) + 1 == len(UNITS) + UNIT_ID_OFFSET
    assert space.contains(obs["blue"]["unit_ids"])
    tower = UNIT_ID_OFFSET + UNITS.index("PrincessTower")
    assert (obs["blue"]["unit_ids"][0] == tower).any()
    assert (obs["blue"]["unit_ids"][1] == tower).any()
    config = env.obs_builder.config()
    assert config["unit_identity"] is True
    assert config["unit_names"] == UNITS


def test_off_by_default_and_nothing_moves() -> None:
    """Without the switch: no key, and config() has none of its fields, so an existing config
    and its hash are what they were."""
    plain = SpatialObsBuilder(card_identity=True)
    env = ClashParallelEnv(UnitTypedMock(), obs_builder=plain)
    obs, _ = env.reset(seed=1)
    assert "unit_ids" not in obs["blue"]
    assert "unit_ids" not in env.observation_space("blue").spaces
    assert not {"unit_identity", "unit_names"} & set(plain.config())


def test_a_pinned_vocabulary_refuses_another() -> None:
    other = [*UNITS[:3], "Zzz", *UNITS[4:]]
    with pytest.raises(ValueError, match="unit_ids"):
        _env(unit_names=other)
    # The same names bind.
    _env(unit_names=list(UNITS))
    with pytest.raises(ValueError, match="unit_identity"):
        SpatialObsBuilder(unit_names=UNITS)


def test_an_engine_that_does_not_say_the_type_is_refused() -> None:
    """No vocabulary (MockEngine as it ships, an engine before royalesim 0.1.17): refused at
    bind. A vocabulary but an entity whose type is not said: refused at the first build, rather
    than drawn as a type."""
    with pytest.raises(ValueError, match="unit type"):
        _env(engine=MockEngine())

    class Unsaid(UnitTypedMock):
        def state(self):
            s = MockEngine.state(self)
            return s  # every unit_type left at -1

    env = _env(engine=Unsaid())
    with pytest.raises(ValueError, match="unit_type"):
        env.reset(seed=1)


def test_the_trace_header_carries_the_vocabulary() -> None:
    rec = ReplayRecorder()
    env = _env()
    env.recorder = rec
    env.reset(seed=1)
    env.step({"blue": 0, "red": 0})
    assert rec.trace.header.unit_types == UNITS
    plain = ReplayRecorder()
    env = ClashParallelEnv(MockEngine(), recorder=plain)
    env.reset(seed=1)
    assert plain.trace.header.unit_types == []


def test_unit_type_is_the_last_entity_column_and_not_said_by_default() -> None:
    """``unit_type`` trails ``ability_ticks``; a row from an engine before it decodes as -1."""
    from royalegym.protocol import EntityState

    fields = EntityState.__struct_fields__
    assert fields[-2:] == ("ability_ticks", "unit_type")
    eng = MockEngine()
    eng.reset(1, DefaultStateMutator().build(np.random.default_rng(0), eng.cards()))
    row = msgspec.to_builtins(eng.state().entities[0])
    assert len(row) == len(fields)
    old = msgspec.convert(row[:-1], EntityState)
    assert old.unit_type == -1
    new = msgspec.convert([*row[:-1], 17], EntityState)
    assert new.unit_type == 17
