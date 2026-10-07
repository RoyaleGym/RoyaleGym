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

import json

import msgspec
import numpy as np
import pytest

from royalegym.env import ClashParallelEnv
from royalegym.mock_engine import MockEngine
from royalegym.obs import UNIT_ID_EMPTY, UNIT_ID_OFFSET, SpatialObsBuilder, unit_id_planes
from royalegym.protocol import BLUE, RED, EntityKind, ShuffleMode
from royalegym.replay import ReplayRecorder
from royalegym.rust_engine import CORE_IMPORT_ERROR, core_available
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


class SplitBarrelMock(UnitTypedMock):
    """The Barrel's goblins on a row of their own, as the game's data gives some summons."""

    def unit_types(self) -> list[str]:
        return sorted([*UNITS, "BarrelGoblin"])

    def state(self):
        s = MockEngine.state(self)
        names = {c.card_id: c.name for c in self.cards()}
        vocab = self.unit_types()

        def typed(e):
            if e.kind == EntityKind.KING_TOWER:
                unit = "KingTower"
            elif e.kind == EntityKind.PRINCESS_TOWER:
                unit = "PrincessTower"
            elif names[e.card_id] == "GoblinBarrel":
                unit = "BarrelGoblin"
            else:
                unit = UNIT_OF[names[e.card_id]]
            return msgspec.structs.replace(e, unit_type=vocab.index(unit))

        return msgspec.structs.replace(s, entities=[typed(e) for e in s.entities])


def _goblin_values(env) -> dict[str, set[int]]:
    """unit_ids values under Blue's Goblins and Goblin Barrel goblins, after both are played."""
    env.reset(seed=3)
    while env.battle_state.tick < env.engine.rules().deploy_lockout_ticks:
        env.step({"blue": 0, "red": 0})
    ids = {c.name: c.card_id for c in env.engine.cards()}
    seen: dict[str, set[int]] = {}
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
                seen.setdefault(card, set()).update(obs["unit_ids"][0][where].tolist())
        if len(seen) == 2:
            return seen
    return seen


def test_an_alias_writes_a_row_as_its_base_type() -> None:
    """Without aliases the Barrel's own row is its own type; with the alias it is a Goblin, and
    the vocabulary drops the aliased name."""
    apart = _goblin_values(_env(engine=SplitBarrelMock()))
    assert apart["Goblins"] != apart["GoblinBarrel"], apart
    env = _env(engine=SplitBarrelMock(), unit_aliases={"BarrelGoblin": "Goblin"})
    merged = _goblin_values(env)
    assert merged["Goblins"] == merged["GoblinBarrel"], merged
    builder = env.obs_builder
    assert builder.unit_vocabulary == UNITS
    assert int(env.observation_space("blue")["unit_ids"].high.max()) + 1 == (
        len(UNITS) + UNIT_ID_OFFSET
    )
    assert builder.config()["unit_aliases"] == {"BarrelGoblin": "Goblin"}
    assert builder.config()["unit_names"] == sorted([*UNITS, "BarrelGoblin"])


def test_the_digest_names_the_effective_vocabulary() -> None:
    plain = _env(engine=SplitBarrelMock()).obs_builder.unit_ids_digest
    merged = _env(engine=SplitBarrelMock(), unit_aliases={"BarrelGoblin": "Goblin"})
    again = _env(engine=SplitBarrelMock(), unit_aliases={"BarrelGoblin": "Goblin"})
    assert merged.obs_builder.unit_ids_digest == again.obs_builder.unit_ids_digest
    assert merged.obs_builder.unit_ids_digest != plain
    assert len(plain) == 16


def test_a_bad_alias_map_is_refused() -> None:
    for aliases, word in (
        ({"Nobody": "Goblin"}, "Nobody"),
        ({"BarrelGoblin": "Nobody"}, "Nobody"),
        ({"BarrelGoblin": "Goblin", "Goblin": "Knight"}, "chain"),
        ({"Goblin": "Goblin"}, "itself"),
    ):
        with pytest.raises(ValueError, match=word):
            _env(engine=SplitBarrelMock(), unit_aliases=aliases)
    with pytest.raises(ValueError, match="unit_identity"):
        SpatialObsBuilder(unit_aliases={"BarrelGoblin": "Goblin"})


def _engine_card_table() -> dict:
    from royalegym.rust_engine import RustEngine, embedded_card_table

    text = embedded_card_table()
    if text is None:
        text = RustEngine().cards_json_path.read_text(encoding="utf-8")
    return json.loads(text)


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_every_shipped_alias_pair_differs_only_on_its_declared_fields() -> None:
    """SAME_UNIT_ALIASES merges rows that are one unit to a player. In the card table the engine
    reads, each pair has identical stats except the fields SAME_UNIT_ALIAS_DIFFERENCES declares
    for it (a timing no player can read off the unit), and differs on every one of those. A
    data build that splits a pair on anything else fails here, rather than the planes quietly
    mixing two units under one id. So does a declared difference that is gone, which would leave
    the declaration out of date. Bookkeeping fields (the row's own name, where its numbers came
    from) are not stats."""
    from royalegym.obs import SAME_UNIT_ALIAS_DIFFERENCES, SAME_UNIT_ALIASES

    table = _engine_card_table()
    units = table["units"]
    missing = sorted({n for pair in SAME_UNIT_ALIASES.items() for n in pair} - set(units))
    if missing:
        pytest.skip(
            f"SKIPPED, NOT PASSED: this card table ({table['provenance']['vintage']}) "
            f"has no rows {missing[:4]}: the aliases name the 15.535 table's rows"
        )
    assert set(SAME_UNIT_ALIAS_DIFFERENCES) <= set(SAME_UNIT_ALIASES), "a difference no alias has"
    bookkeeping = {"name", "damage_source", "overlays", "raw", "source_table"}
    for alias, base in SAME_UNIT_ALIASES.items():
        a = {k: v for k, v in units[alias].items() if k not in bookkeeping}
        b = {k: v for k, v in units[base].items() if k not in bookkeeping}
        differ = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        declared = sorted(SAME_UNIT_ALIAS_DIFFERENCES.get(alias, ()))
        assert differ == declared, f"{alias} -> {base}: the rows differ on {differ}"


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_no_shipped_alias_pair_is_split_by_a_calibration_value() -> None:
    """The card table is not all the engine plays: a calibration value table (a section's entry
    whose value maps row names to field values, such as cards.CLIENT16402_VALUES) replaces
    fields of the rows it NAMES when the battle loads. Two rows with identical table entries
    then play differently when only one is named (the evolved Goblin Cage's brawler had 1080 hp
    at level 11, the plain one 1121). Each alias and its base must be named the same way in
    every such table the engine was built with."""
    from royalegym.obs import SAME_UNIT_ALIASES
    from royalegym.rust_engine import RustEngine

    raw = RustEngine().calibration.raw
    tables = {}
    for section, entries in raw.items():
        if not isinstance(entries, dict):
            continue
        for key, entry in entries.items():
            value = entry.get("value") if isinstance(entry, dict) else None
            values = value.get("values") if isinstance(value, dict) else None
            if isinstance(values, dict):
                tables[f"{section}.{key}"] = values
    assert "cards.CLIENT16402_VALUES" in tables, sorted(tables)
    for name, values in tables.items():
        for alias, base in SAME_UNIT_ALIASES.items():
            assert values.get(alias) == values.get(base), (
                f"{name} sets {alias} to {values.get(alias)} and {base} to "
                f"{values.get(base)}: the engine plays the pair differently"
            )


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_goblins_from_both_goblin_cards_and_the_three_musketeers_read_one_type_aliased() -> None:
    """On RustEngine: the Goblins card's goblins (row Goblin_Stab) and the Goblin Barrel's
    (Goblin) are two rows in raw ``unit_ids`` and one, Goblin, under SAME_UNIT_ALIASES; the
    Three Musketeers' three rows become one. Train's ruling, 2026-10-06: what a player sees,
    and card_ids still names each goblin's card."""
    from royalegym.obs import SAME_UNIT_ALIASES, unit_vocabulary
    from royalegym.protocol import DeployCommand, DeployStatus, MatchSetup
    from royalegym.rust_engine import RustEngine
    from royalegym.state_mutator import deck_ids

    def types_by_card(names: list[str], plays: list[tuple[int, int]]) -> dict[str, set[str]]:
        eng = RustEngine()
        deck = deck_ids(names, eng.cards())
        eng.reset(1, MatchSetup(decks=[deck, deck], shuffle=0, elixir_milli=[10000, 10000],
                                start_tick=eng.rules().deploy_lockout_ticks))
        t = eng.arena().subtile
        for slot, x_tile in plays:
            cmd = DeployCommand(BLUE, slot, x_tile * t + t // 2, 8 * t + t // 2)
            assert eng.step([cmd], 1)[0].status == DeployStatus.OK, names[slot]
        unit_names = eng.unit_types()
        card_names = {c.card_id: c.name for c in eng.cards()}
        seen: dict[str, set[str]] = {}
        for _ in range(100):  # the barrel lands and opens within two seconds or so
            eng.step([], 1)
            for e in eng.state().entities:
                if e.team == BLUE and e.kind == EntityKind.TROOP:
                    seen.setdefault(card_names[e.card_id], set()).add(unit_names[e.unit_type])
        return seen

    def aliased(rows: set[str]) -> set[str]:
        names = RustEngine().unit_types()
        vocab, lookup = unit_vocabulary(names, SAME_UNIT_ALIASES)
        return {vocab[lookup[names.index(r)]] for r in rows}

    rest = ["Knight", "Archer", "Giant", "Minions", "Fireball", "Zap"]
    goblins = types_by_card(["Goblins", "GoblinBarrel", *rest], [(0, 4), (1, 13)])
    assert goblins == {"Goblins": {"Goblin_Stab"}, "GoblinBarrel": {"Goblin"}}, goblins
    assert aliased(goblins["Goblins"]) == aliased(goblins["GoblinBarrel"]) == {"Goblin"}
    musketeers = types_by_card(["ThreeMusketeers", *rest, "Musketeer"], [(0, 9)])
    rework = {f"ThreeMusketeer_Rework_Character_{i}" for i in (1, 2, 3)}
    assert musketeers == {"ThreeMusketeers": rework}, musketeers
    assert aliased(musketeers["ThreeMusketeers"]) == {"ThreeMusketeer_Rework_Character_1"}


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_the_engine_names_each_unit_by_its_own_type() -> None:
    """On RustEngine (royalesim 0.1.17 on): the Witch's skeletons and the Skeletons card's give
    one unit type with two producers, and the towers give their own rows. The vocabulary is
    sorted, and every entity's type lies inside it."""
    from royalegym.protocol import DeployCommand, DeployStatus, MatchSetup
    from royalegym.rust_engine import RustEngine
    from royalegym.state_mutator import deck_ids

    eng = RustEngine()
    names = eng.unit_types()
    assert names == sorted(names)
    assert {"Skeleton", "KingTower", "PrincessTower"} <= set(names)
    deck = deck_ids(["Witch", "Skeletons", "Knight", "Archer", "Giant", "Minions", "Fireball",
                     "Zap"], eng.cards())
    eng.reset(1, MatchSetup(decks=[deck, deck], shuffle=0, elixir_milli=[10000, 10000],
                            start_tick=eng.rules().deploy_lockout_ticks))
    arena = eng.arena()
    t = arena.subtile
    for slot, x_tile in ((0, 4), (1, 13)):  # the Witch, then the Skeletons
        cmd = DeployCommand(BLUE, slot, x_tile * t + t // 2, 8 * t + t // 2)
        assert eng.step([cmd], 1)[0].status == DeployStatus.OK
    ids = {c.name: c.card_id for c in eng.cards()}
    producers: dict[str, set[int]] = {}
    for _ in range(400):  # the Witch summons her first skeletons a few seconds in
        eng.step([], 5)
        for e in eng.state().entities:
            assert 0 <= e.unit_type < len(names), e
            producers.setdefault(names[e.unit_type], set()).add(e.card_id)
        if {ids["Witch"], ids["Skeletons"]} <= producers.get("Skeleton", set()):
            break
    assert {ids["Witch"], ids["Skeletons"]} <= producers["Skeleton"], producers
    kinds = {names[e.unit_type] for e in eng.state().entities if e.kind in (
        EntityKind.KING_TOWER, EntityKind.PRINCESS_TOWER)}
    assert kinds == {"KingTower", "PrincessTower"}
    builder = SpatialObsBuilder(card_identity=True, unit_identity=True)
    env = ClashParallelEnv(eng, obs_builder=builder,
                           state_mutator=DefaultStateMutator(decks=[deck, deck]))
    obs, _ = env.reset(seed=1)
    assert env.observation_space("blue")["unit_ids"].contains(obs["blue"]["unit_ids"])
    assert builder.config()["unit_names"] == names


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_the_graveyards_skeletons_are_skeletons_only_through_the_alias_map() -> None:
    """On RustEngine: a cast Graveyard's skeletons read their own data row in raw ``unit_ids``
    and Skeleton with ``unit_aliases=SAME_UNIT_ALIASES`` (Train's ruling: the same unit to a
    player)."""
    from royalegym.obs import SAME_UNIT_ALIASES
    from royalegym.protocol import ShuffleMode
    from royalegym.rust_engine import RustEngine
    from royalegym.state_mutator import deck_ids

    if "Graveyard_rework_Skeleton" not in (RustEngine().unit_types() or []):
        pytest.skip("SKIPPED, NOT PASSED: this card table has no Graveyard_rework_Skeleton row")
    names = ["Graveyard", "Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Musketeer"]
    read = {}
    for label, aliases in (("raw", None), ("aliased", SAME_UNIT_ALIASES)):
        eng = RustEngine()
        deck = deck_ids(names, eng.cards())
        builder = SpatialObsBuilder(card_identity=True, unit_identity=True, unit_aliases=aliases)
        env = ClashParallelEnv(eng, obs_builder=builder, state_mutator=DefaultStateMutator(
            decks=[deck, deck], shuffle=ShuffleMode.NONE))
        env.reset(seed=1)
        while env.battle_state.tick < eng.rules().deploy_lockout_ticks + 200:
            env.step({"blue": 0, "red": 0})
        parser = env.action_parser
        cast = 1 + 26 * parser.nx + 9  # hand slot 0 (the Graveyard), enemy half, own frame
        assert env.action_masks("blue")[cast]
        env.step({"blue": cast, "red": 0})
        graveyard = {c.name: c.card_id for c in eng.cards()}["Graveyard"]
        seen: set[int] = set()
        for _ in range(40):
            env.step({"blue": 0, "red": 0})
            obs = env._obs["blue"]
            seen |= set(obs["unit_ids"][0][obs["card_ids"][0] == 2 + graveyard].tolist())
        read[label] = sorted(builder.unit_vocabulary[v - UNIT_ID_OFFSET] for v in seen)
    assert read == {"raw": ["Graveyard_rework_Skeleton"], "aliased": ["Skeleton"]}, read
