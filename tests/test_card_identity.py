"""The card-identity planes and ``enemy_last_card`` (decision D2; docs/observation-spec.md 3b).

WHAT IS GRADED AGAINST WHAT
    The planes are checked against units SPAWNED at known tiles with known, DISTINCT cards,
    not against a second implementation of the rasteriser -- a restated rule agrees with
    the original about anything the two share. Ties are checked against the uids the
    ENGINE assigned. Seat symmetry is checked on a mirrored board, because the whole point
    of the own frame is that one policy plays both seats.

WHAT DEFAULT OFF MEANS, checked first
    Nothing about the shipped observation may move: no new key, the same vector width, the
    same offset for every field, and the same ``config()``. A run or a checkpoint from
    before this change must not be able to tell it happened.
"""

from __future__ import annotations

import numpy as np
import pytest

from royalegym.action import TileActionParser
from royalegym.obs import (
    CARD_ID_OFFSET,
    CARD_ID_TOWER,
    SpatialObsBuilder,
    card_id_planes,
    vector_layout,
    vector_offsets,
)
from royalegym.protocol import (
    BLUE,
    RED,
    DeployCommand,
    DeployStatus,
    EntityKind,
    MatchSetup,
    SpawnSpec,
    to_engine,
)
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available

pytestmark = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))

SPAWNED = ("Knight", "Giant", "Musketeer", "Valkyrie")


@pytest.fixture(scope="module")
def engine() -> RustEngine:
    return RustEngine()


def ids(engine) -> dict[str, int]:
    return {c.name: c.card_id for c in engine.cards()}


def bound(engine, **kw) -> tuple[SpatialObsBuilder, TileActionParser]:
    parser = TileActionParser()
    parser.bind(engine)
    builder = SpatialObsBuilder(**kw)
    builder.bind(engine, parser)
    return builder, parser


def observe(engine, builder, parser, team):
    state = engine.state()
    builder.reset(state)
    return builder.build(state, team, parser.action_mask(state, team))


def spawn_at(engine, team, card, tx, ty) -> SpawnSpec:
    """A unit at the centre of OWN-frame tile (tx, ty) for ``team``."""
    a = engine.arena()
    x, y = to_engine(a, team, tx * a.subtile + a.subtile // 2, ty * a.subtile + a.subtile // 2)
    return SpawnSpec(team=team, card_id=ids(engine)[card], x=x, y=y)


def board(engine, spawns, **kw) -> None:
    deck = [ids(engine)[n] for n in ("Knight", "Archer", "Giant", "Minions",
                                      "Fireball", "Cannon", "Zap", "Musketeer")]
    engine.reset(seed=0, setup=MatchSetup(decks=[deck, deck], spawns=list(spawns), **kw))


# ---------------------------------------------------------------------------- default off


def test_off_by_default_changes_nothing_a_run_could_see(engine):
    builder, _ = bound(engine)
    space = builder.observation_space()
    assert "card_ids" not in space.spaces
    assert builder.vec_size == sum(f.size for f in vector_layout(builder.num_cards))
    assert "enemy_last_card" not in builder.vector_offsets()
    assert builder.config() == {"reveal": builder.reveal.as_dict()}, (
        "config() changed with the flag off, so an existing run's recorded identity would "
        "no longer match a fresh one"
    )


def test_turning_it_on_moves_no_existing_fair_offset(engine):
    """The new field goes at the END of the fair block; nothing before it may shift."""
    n = len(engine.cards())
    before, after = vector_offsets(n), vector_offsets(n, enemy_last_card=True)
    fair_before = [f.key for f in vector_layout(n) if f.fair]
    for key in fair_before:
        assert after[key] == before[key], f"{key} moved when enemy_last_card was enabled"
    last_fair = max(before[k].stop for k in fair_before)
    assert after["enemy_last_card"] == slice(last_fair, last_fair + n + 1)


# ---------------------------------------------------------------------------- the planes


def test_the_space_publishes_the_vocabulary(engine):
    builder, _ = bound(engine, card_identity=True)
    box = builder.observation_space()["card_ids"]
    a = engine.arena()
    assert box.dtype == np.uint8
    assert box.shape == (2, a.tiles_y, a.tiles_x)
    assert int(box.low.min()) == 0
    assert int(box.high.max()) + 1 == len(engine.cards()) + CARD_ID_OFFSET, (
        "a network sizes nn.Embedding from the Box's high + 1; it must equal num_cards + 2"
    )


def test_each_spawned_unit_is_named_on_its_own_tile(engine):
    """Four DISTINCT cards at four known tiles, per seat, graded by construction."""
    card = ids(engine)
    tiles = [(3, 10), (8, 12), (13, 9), (5, 14)]
    spawns = [spawn_at(engine, BLUE, name, *t) for name, t in zip(SPAWNED, tiles, strict=True)]
    spawns += [spawn_at(engine, RED, name, *t) for name, t in
               zip(reversed(SPAWNED), tiles, strict=True)]
    board(engine, spawns)
    builder, parser = bound(engine, card_identity=True)
    blue = observe(engine, builder, parser, BLUE)["card_ids"]
    for name, (tx, ty) in zip(SPAWNED, tiles, strict=True):
        assert blue[0, ty, tx] == CARD_ID_OFFSET + card[name], (
            f"Blue's own {name} at tile {(tx, ty)} reads {blue[0, ty, tx]}"
        )
    red = observe(engine, builder, parser, RED)["card_ids"]
    for name, (tx, ty) in zip(reversed(SPAWNED), tiles, strict=True):
        assert red[0, ty, tx] == CARD_ID_OFFSET + card[name], (
            f"Red's own {name} at tile {(tx, ty)} reads {red[0, ty, tx]} in Red's frame"
        )


def test_the_board_is_the_same_from_both_seats_on_a_mirrored_battle(engine):
    """One policy plays both seats, so the own frame must make a mirrored board identical."""
    spawns = []
    for team in (BLUE, RED):
        spawns += [spawn_at(engine, team, "Knight", 4, 11), spawn_at(engine, team, "Giant", 9, 13)]
    board(engine, spawns)
    builder, parser = bound(engine, card_identity=True)
    blue = observe(engine, builder, parser, BLUE)["card_ids"]
    red = observe(engine, builder, parser, RED)["card_ids"]
    assert np.array_equal(blue, red), "a mirrored board reads differently from the two seats"
    assert (blue >= CARD_ID_OFFSET).sum() >= 4, "too few units on the board to prove anything"


def test_crown_towers_read_as_towers_and_a_fallen_one_as_empty(engine):
    board(engine, [], tower_hp=[[4000, 0, 2000], [4000, 2000, 2000]])
    builder, parser = bound(engine, card_identity=True)
    obs = observe(engine, builder, parser, BLUE)["card_ids"]
    towers = [e for e in engine.state().entities
              if e.kind in (EntityKind.KING_TOWER, EntityKind.PRINCESS_TOWER)]
    assert towers, "no towers on the board"
    assert (obs == CARD_ID_TOWER).sum() == len(towers), (
        f"{len(towers)} towers stand and {(obs == CARD_ID_TOWER).sum()} tiles read as tower; "
        "a fallen tower's tile must read EMPTY, not tower"
    )
    assert int(obs.max()) == CARD_ID_TOWER, "something other than a tower is on a fresh board"


def test_two_units_on_one_tile_resolve_to_the_lower_uid(engine):
    """Graded against the uids the ENGINE assigned, not against spawn order in the list."""
    card = ids(engine)
    board(engine, [spawn_at(engine, BLUE, "Giant", 7, 12), spawn_at(engine, BLUE, "Knight", 7, 12)])
    state = engine.state()
    here = sorted((e for e in state.entities if e.card_id in (card["Giant"], card["Knight"])),
                  key=lambda e: e.uid)
    assert len(here) == 2, f"expected the two spawned units, found {len(here)}"
    builder, parser = bound(engine, card_identity=True)
    obs = observe(engine, builder, parser, BLUE)["card_ids"]
    assert obs[0, 12, 7] == CARD_ID_OFFSET + here[0].card_id, (
        f"the tile reads {obs[0, 12, 7]}; the lower uid is {here[0].uid} "
        f"(card {here[0].card_id}), so a later list position must not win"
    )


def test_the_planes_do_not_depend_on_entity_list_order(engine):
    """BattleState.entities is not in uid order, so reversing it must change nothing."""
    board(engine, [spawn_at(engine, BLUE, "Giant", 7, 12), spawn_at(engine, BLUE, "Knight", 7, 12),
                   spawn_at(engine, RED, "Valkyrie", 2, 9)])
    s = engine.state()
    a = engine.arena()
    n = len(engine.cards())
    forward = card_id_planes(s.entities, BLUE, a, n)
    backward = card_id_planes(list(reversed(s.entities)), BLUE, a, n)
    assert np.array_equal(forward, backward)


def test_a_spell_in_flight_is_not_on_the_board(engine):
    """A live spell has no entity and no tile; it must not appear as a card."""
    card = ids(engine)
    deck = [card["Fireball"]] * 8
    engine.reset(seed=0, setup=MatchSetup(decks=[deck, deck], elixir_milli=[10000, 10000],
                                          start_tick=engine.rules().deploy_lockout_ticks))
    a = engine.arena()
    x, y = to_engine(a, BLUE, 9 * a.subtile + a.subtile // 2, 20 * a.subtile + a.subtile // 2)
    res = engine.step([DeployCommand(BLUE, 0, x, y)], 2)
    assert res[0].status == DeployStatus.OK
    assert engine.state().spells, "the Fireball resolved already, so this compared nothing"
    builder, parser = bound(engine, card_identity=True)
    obs = observe(engine, builder, parser, BLUE)["card_ids"]
    assert not (obs == CARD_ID_OFFSET + card["Fireball"]).any()


# ---------------------------------------------------------------------------- positional ids


def test_config_rebuilds_a_builder_that_binds(engine):
    builder, _ = bound(engine, card_identity=True)
    cfg = builder.config()
    assert cfg["card_identity"] is True
    assert cfg["card_names"] == [c.name for c in engine.cards()]
    parser = TileActionParser()
    parser.bind(engine)
    again = SpatialObsBuilder(card_identity=True, card_names=cfg["card_names"])
    again.bind(engine, parser)
    assert again.card_names == cfg["card_names"]


def test_a_renumbered_catalogue_is_refused(engine):
    """The trap D2 ships with: one more loadable card renumbers every later id."""
    names = [c.name for c in engine.cards()]
    shifted = [names[0], "SomeNewCard", *names[1:-1]]
    parser = TileActionParser()
    parser.bind(engine)
    with pytest.raises(ValueError, match=r"has 'SomeNewCard' at id 1"):
        SpatialObsBuilder(card_identity=True, card_names=shifted).bind(engine, parser)


def test_card_names_without_the_planes_is_refused():
    with pytest.raises(ValueError, match="only exists with card_identity=True"):
        SpatialObsBuilder(card_names=["Knight"])


# ---------------------------------------------------------------------------- the vector field


def test_enemy_last_card_is_the_enemy_play_and_none_before_it(engine):
    card = ids(engine)
    deck = [card[n] for n in ("Knight", "Archer", "Giant", "Minions",
                               "Fireball", "Cannon", "Zap", "Musketeer")]
    engine.reset(seed=0, setup=MatchSetup(decks=[deck, deck], elixir_milli=[10000, 10000],
                                          start_tick=engine.rules().deploy_lockout_ticks))
    builder, parser = bound(engine, card_identity=True)
    n = len(engine.cards())
    sl = builder.vector_offsets()["enemy_last_card"]
    state = engine.state()
    builder.reset(state)
    vec = builder.build(state, BLUE, parser.action_mask(state, BLUE))["vector"]
    assert int(np.argmax(vec[sl])) == n, "before any enemy play the field must read 'none'"

    # TWO plays of DIFFERENT cards. With one play, "first" and "last" are the same card, and
    # a field reading the FIRST play would pass -- which the first version of this test did.
    a = engine.arena()
    x, y = to_engine(a, RED, 9 * a.subtile + a.subtile // 2, 8 * a.subtile + a.subtile // 2)
    first = state.players[RED].hand[0]
    res = engine.step([DeployCommand(RED, 0, x, y)], 10)
    assert res[0].status == DeployStatus.OK, "the first enemy play was refused"
    # OBSERVE BETWEEN THE PLAYS, as the env does once per decision. The memory learns a
    # play by diffing two hand snapshots, and a seat plays at most one card per decision.
    # Stepping twice without looking let the SECOND card -- the one that replaced the
    # first in the hand -- enter and leave inside one gap, where no diff can see it; the
    # first version of this test did exactly that and blamed the builder.
    state = engine.state()
    builder.build(state, BLUE, parser.action_mask(state, BLUE))
    hand = state.players[RED].hand
    slot = next(i for i, c in enumerate(hand) if c != first)
    second = hand[slot]
    res = engine.step([DeployCommand(RED, slot, x, y)], 10)
    assert res[0].status == DeployStatus.OK, "the second enemy play was refused"
    assert second != first, "the two plays were the same card, so first and last coincide"
    state = engine.state()
    vec = builder.build(state, BLUE, parser.action_mask(state, BLUE))["vector"]
    assert vec[sl].sum() == 1.0
    assert int(np.argmax(vec[sl])) == second, (
        f"the enemy played {first} then {second}; the field says {int(np.argmax(vec[sl]))}"
    )
