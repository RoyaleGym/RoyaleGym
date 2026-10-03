"""What is on a unit: ``SpatialObsBuilder(unit_status=True)``.

Eighteen planes, own then enemy, per tile like the troop planes:
- ``own_shield`` / ``enemy_shield``: shield hp left, summed and scaled like the hp planes (a
  Dark Prince's shield is drawn over it, and a shot breaks it before any hp goes);
- ``own_raged`` / ``enemy_raged``: units under a Rage (drawn purple; they move and hit faster);
- ``own_slowed`` / ``enemy_slowed``: units slowed by cold (an Ice Wizard's, a hero Ice Golem's);
- ``own_on_tower`` / ``enemy_on_tower``: units whose current target is a crown tower. A unit
  locked on a tower ignores a building put down to pull it; one not locked yet can be pulled.
- ``own_on_building`` / ``enemy_on_building``: units whose current target is a building that
  is not a crown tower (a Hog pulled by a Cannon).
- ``own_hp_frac`` / ``enemy_hp_frac``: hp / max hp of the tile's strongest unit (the largest
  max hp; ties to the lowest uid), so a damaged P.E.K.K.A differs from a fresh one. Not towers.
- ``own_invisible`` / ``enemy_invisible``, ``own_underground`` / ``enemy_underground``,
  ``own_hidden`` / ``enemy_hidden``: units with that status bit, still in the troop planes
  where they stand (the client shows an invisible unit's shimmer, a tunneller's trail, a hidden
  building's mound).
A frozen unit is not here: a Freeze sets ``stun_ticks``, which the stunned planes already count.

A unit's ``buffs`` entry names an EFFECT FAMILY, its members joined by "|" (the engine's own
grouping: "IceWizardSlowDown|IceWizardCold|..."), so a family is matched by one member's name
and never by the whole string, which grows when the engine adds a member.

Off by default, and appended after every other optional fair plane.

SKIPS
    The engine-backed test skips without the engine. Not a pass.
"""

from __future__ import annotations

import msgspec
import numpy as np
import pytest

from royalegym.action import TileActionParser
from royalegym.mock_engine import MockEngine
from royalegym.obs import HP_SCALE, Reveal, SpatialObsBuilder, spatial_channels, status_channels
from royalegym.protocol import (
    BLUE,
    RED,
    STATUS_HIDDEN,
    STATUS_INVISIBLE,
    STATUS_UNDERGROUND,
    DeployCommand,
    DeployStatus,
    EntityKind,
    EntityState,
    MatchSetup,
    ShuffleMode,
    to_engine,
    to_own,
)
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available

needs_engine = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))

PLANES = [
    "own_shield", "enemy_shield", "own_raged", "enemy_raged", "own_slowed", "enemy_slowed",
    "own_on_tower", "enemy_on_tower", "own_on_building", "enemy_on_building",
    "own_hp_frac", "enemy_hp_frac", "own_invisible", "enemy_invisible",
    "own_underground", "enemy_underground", "own_hidden", "enemy_hidden",
]
SLOW = "IceWizardSlowDown|IceWizardCold|IceGolemiteHero_Slow_Buff_Tower"


# --- layout -------------------------------------------------------------------------------


def test_off_by_default_and_after_every_other_optional_fair_plane():
    assert not set(PLANES) & {n for n, _ in spatial_channels()}
    rev = Reveal(enemy_spell_aim=True)
    before = [n for n, _ in spatial_channels(rev, True, True, heroes=True)]
    after = [n for n, _ in spatial_channels(rev, True, True, heroes=True, unit_status=True)]
    assert after[: len(before) - 1] == before[:-1], "an existing plane moved"
    assert after[len(before) - 1 : len(before) - 1 + len(PLANES)] == PLANES
    assert after[-1] == "enemy_spell_aim", "the revealed planes stay last"


def test_config_records_the_flag():
    assert SpatialObsBuilder(unit_status=True).config()["unit_status"] is True
    assert "unit_status" not in SpatialObsBuilder().config()


# --- the planes, on a forged board ---------------------------------------------------------


def _board():
    eng = MockEngine()
    eng.reset(1, MatchSetup(decks=[list(range(8)), list(range(8))]))
    return eng.state(), eng.arena()


def _unit(uid, team, point, shield=0, buffs=(), kind=EntityKind.TROOP, target=-1, hp=100,
          max_hp=100, status=0):
    return EntityState(
        uid, team, int(kind), 3, -1, point[0], point[1], hp, max_hp, 500, False, 0,
        shield=shield, buffs=tuple(buffs), status_flags=status, target_uid=target,
    )


def _with(state, entities):
    towers = [msgspec.structs.replace(e, status_flags=0) for e in state.entities
              if e.kind in (EntityKind.KING_TOWER, EntityKind.PRINCESS_TOWER)]
    return msgspec.structs.replace(state, entities=towers + entities)


def _tile(arena, viewer, point):
    ox, oy = to_own(arena, viewer, *point)
    return oy // arena.subtile, ox // arena.subtile


def _at(a, x, y):
    return (a.subtile * x + 100, a.subtile * y + 100)


def test_a_shield_is_summed_and_scaled_like_hp_on_both_sides():
    state, a = _board()
    p = _at(a, 9, 12)
    s = _with(state, [_unit(100, BLUE, p, shield=300), _unit(101, BLUE, p, shield=150)])
    blue = status_channels(s.entities, BLUE, a)
    red = status_channels(s.entities, RED, a)
    assert blue.shape == (len(PLANES), a.tiles_y, a.tiles_x)
    assert blue[0][_tile(a, BLUE, p)] == pytest.approx(450 / HP_SCALE)
    assert blue[0].sum() == pytest.approx(450 / HP_SCALE)
    assert blue[1].sum() == 0
    assert red[1][_tile(a, RED, p)] == pytest.approx(450 / HP_SCALE)
    assert red[0].sum() == 0


def test_rage_and_cold_are_matched_by_a_member_of_the_family():
    state, a = _board()
    raged, slowed, both = _at(a, 4, 20), _at(a, 14, 22), _at(a, 12, 6)
    s = _with(state, [
        _unit(100, RED, raged, buffs=[("Rage", 3000)]),
        _unit(101, RED, slowed, buffs=[(SLOW, 1200), ("Poison", 500)]),
        _unit(102, BLUE, both, buffs=[("Rage|SomeLaterRage", 800), (SLOW + "|Later", 100)]),
    ])
    blue = status_channels(s.entities, BLUE, a)
    assert blue[3][_tile(a, BLUE, raged)] == 1, "enemy raged"
    assert blue[3].sum() == 1, "the slowed unit is not raged"
    assert blue[5][_tile(a, BLUE, slowed)] == 1, "enemy slowed"
    assert blue[5].sum() == 1, "the raged unit is not slowed"
    assert blue[2][_tile(a, BLUE, both)] == 1, "own raged, the family grown by a member"
    assert blue[4][_tile(a, BLUE, both)] == 1, "own slowed, the family grown by a member"
    assert blue[2].sum() == 1
    assert blue[4].sum() == 1


def _buff_planes(planes):
    """The raged and slowed planes, own and enemy: what a buff entry can light."""
    return planes[[PLANES.index(n) for n in PLANES if n.endswith(("_raged", "_slowed"))]]


def test_a_name_that_only_contains_a_member_does_not_match():
    state, a = _board()
    p = _at(a, 4, 20)
    lookalikes = [("BarbarianRage", 3000), ("IceWizardSlowDownX", 9)]
    s = _with(state, [_unit(100, RED, p, buffs=lookalikes)])
    assert _buff_planes(status_channels(s.entities, BLUE, a)).sum() == 0


def test_poison_tornado_and_freeze_are_not_status_planes():
    state, a = _board()
    p = _at(a, 4, 20)
    s = _with(state, [_unit(100, RED, p, buffs=[
        ("Poison", 900), ("Tornado", 400), ("Freeze|ZapFreeze", 300),
    ])])
    assert _buff_planes(status_channels(s.entities, BLUE, a)).sum() == 0


def test_a_unit_locked_on_a_crown_tower_is_counted_and_one_on_anything_else_is_not():
    state, a = _board()
    towers = {e.team: e.uid for e in state.entities if e.kind == EntityKind.PRINCESS_TOWER}
    hog, other, idle, cannon = _at(a, 3, 22), _at(a, 14, 22), _at(a, 9, 18), _at(a, 9, 9)
    s = _with(state, [
        _unit(100, BLUE, hog, target=towers[RED]),
        _unit(101, BLUE, other, target=102),
        _unit(102, RED, cannon, kind=EntityKind.BUILDING, target=101),
        _unit(103, BLUE, idle),
        _unit(104, RED, cannon, target=towers[BLUE]),
    ])
    blue = status_channels(s.entities, BLUE, a)
    red = status_channels(s.entities, RED, a)
    assert blue[6][_tile(a, BLUE, hog)] == 1
    assert blue[6].sum() == 1, "a unit on a unit, or on nothing, is not on a tower"
    assert blue[7][_tile(a, BLUE, cannon)] == 1
    assert blue[7].sum() == 1
    assert red[7][_tile(a, RED, hog)] == 1
    assert red[6][_tile(a, RED, cannon)] == 1


def test_a_unit_on_a_building_is_counted_apart_from_one_on_a_tower():
    state, a = _board()
    towers = {e.team: e.uid for e in state.entities if e.kind == EntityKind.PRINCESS_TOWER}
    hog, other, cannon = _at(a, 3, 22), _at(a, 14, 22), _at(a, 9, 9)
    s = _with(state, [
        _unit(100, BLUE, hog, target=102),
        _unit(101, BLUE, other, target=towers[RED]),
        _unit(102, RED, cannon, kind=EntityKind.BUILDING),
    ])
    blue = status_channels(s.entities, BLUE, a)
    on_building, on_tower = PLANES.index("own_on_building"), PLANES.index("own_on_tower")
    assert blue[on_building][_tile(a, BLUE, hog)] == 1
    assert blue[on_building].sum() == 1, "a unit on a tower is not on a building"
    assert blue[on_tower].sum() == 1
    red = status_channels(s.entities, RED, a)
    assert red[PLANES.index("enemy_on_building")][_tile(a, RED, hog)] == 1


def test_hp_frac_is_the_strongest_units_on_its_tile():
    state, a = _board()
    p, q = _at(a, 9, 12), _at(a, 4, 20)
    s = _with(state, [
        _unit(100, BLUE, p, hp=1500, max_hp=3000),  # a damaged Giant
        _unit(101, BLUE, p, hp=90, max_hp=100),  # a fresh small unit beside it
        _unit(102, RED, q, hp=250, max_hp=1000),
        _unit(103, RED, q, hp=1000, max_hp=1000),  # a tie on max hp: the lower uid holds
    ])
    blue = status_channels(s.entities, BLUE, a)
    own, foe = PLANES.index("own_hp_frac"), PLANES.index("enemy_hp_frac")
    assert blue[own][_tile(a, BLUE, p)] == pytest.approx(0.5)
    assert blue[foe][_tile(a, BLUE, q)] == pytest.approx(0.25)
    assert blue[own].sum() == pytest.approx(0.5), "towers and empty tiles hold 0"
    reordered = _with(state, list(reversed(s.entities[len(s.entities) - 4 :])))
    assert np.array_equal(status_channels(reordered.entities, BLUE, a), blue)


@pytest.mark.parametrize(
    ("bit", "plane"),
    [(STATUS_INVISIBLE, "invisible"), (STATUS_UNDERGROUND, "underground"),
     (STATUS_HIDDEN, "hidden")],
)
def test_a_status_bit_marks_its_unit_where_it_stands(bit, plane):
    state, a = _board()
    p = _at(a, 4, 20)
    s = _with(state, [
        _unit(100, RED, p, status=bit),
        _unit(101, RED, p, status=bit ^ (STATUS_INVISIBLE | STATUS_UNDERGROUND | STATUS_HIDDEN)),
        _unit(102, BLUE, p, status=bit),
    ])
    blue = status_channels(s.entities, BLUE, a)
    assert blue[PLANES.index(f"enemy_{plane}")][_tile(a, BLUE, p)] == 1
    assert blue[PLANES.index(f"enemy_{plane}")].sum() == 1
    assert blue[PLANES.index(f"own_{plane}")].sum() == 1


def test_an_engine_that_does_not_report_unit_status_is_refused():
    state, a = _board()  # MockEngine reports no status_flags
    with pytest.raises(ValueError, match="does not report unit status"):
        status_channels(state.entities, BLUE, a)


def test_the_flag_refuses_an_engine_that_does_not_report_unit_status():
    from royalegym import ClashParallelEnv

    env = ClashParallelEnv(engine=MockEngine(), obs_builder=SpatialObsBuilder(unit_status=True))
    with pytest.raises(ValueError, match="does not report unit status"):
        env.reset(seed=0)


# --- on the engine ------------------------------------------------------------------------


@needs_engine
def test_a_dark_princes_shield_then_a_rage_on_it():
    eng = RustEngine()
    ids = {c.name: c.card_id for c in eng.cards()}
    names = ("DarkPrince", "Rage", "Knight", "Archer", "Giant", "Minions", "Fireball", "Zap")
    if any(n not in ids for n in names):
        pytest.skip(f"this engine's catalogue lacks {[n for n in names if n not in ids]}")
    deck = [ids[n] for n in names]
    eng.reset(1, MatchSetup(
        decks=[deck, deck], shuffle=ShuffleMode.NONE, elixir_milli=[10000, 10000],
        start_tick=eng.rules().deploy_lockout_ticks,
    ))
    parser = TileActionParser()
    parser.bind(eng)
    builder = SpatialObsBuilder(unit_status=True)
    builder.bind(eng, parser)
    at = builder.channel_names()
    plane = {n: at.index(n) for n in PLANES}
    builder.reset(eng.state())

    def sums(state, team):
        sp = builder.build(state, team, parser.action_mask(state, team))["spatial"]
        return {n: float(sp[i].sum()) for n, i in plane.items()}

    t = eng.arena().subtile
    hand = eng.state().players[BLUE].hand
    x, y = to_engine(eng.arena(), BLUE, 9 * t + t // 2, 8 * t + t // 2)
    assert eng.step([DeployCommand(BLUE, hand.index(ids["DarkPrince"]), x, y)], 1)[0].status \
        == DeployStatus.OK
    eng.step([], 25)
    state = eng.state()
    prince = [e for e in state.entities if e.team == BLUE and e.card_id == ids["DarkPrince"]]
    assert prince, "the Dark Prince is not on the board"
    assert prince[0].shield > 0, "this engine gives the Dark Prince no shield"
    blue, red = sums(state, BLUE), sums(state, RED)
    assert blue["own_shield"] == pytest.approx(prince[0].shield / HP_SCALE)
    assert red["enemy_shield"] == blue["own_shield"]
    assert blue["own_raged"] == 0
    hand = state.players[BLUE].hand
    assert eng.step([DeployCommand(BLUE, hand.index(ids["Rage"]), prince[0].x, prince[0].y)], 1)[
        0
    ].status == DeployStatus.OK
    for _ in range(40):
        eng.step([], 1)
        state = eng.state()
        if sums(state, BLUE)["own_raged"]:
            break
    blue, red = sums(state, BLUE), sums(state, RED)
    assert blue["own_raged"] >= 1, "the Rage never showed on the Dark Prince"
    assert red["enemy_raged"] == blue["own_raged"]
    assert blue["enemy_raged"] == 0


@needs_engine
def test_a_hog_rider_locks_on_a_tower():
    eng = RustEngine()
    ids = {c.name: c.card_id for c in eng.cards()}
    names = ("HogRider", "Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Cannon")
    if any(n not in ids for n in names):
        pytest.skip(f"this engine's catalogue lacks {[n for n in names if n not in ids]}")
    deck = [ids[n] for n in names]
    eng.reset(1, MatchSetup(
        decks=[deck, deck], shuffle=ShuffleMode.NONE, elixir_milli=[10000, 10000],
        start_tick=eng.rules().deploy_lockout_ticks,
    ))
    parser = TileActionParser()
    parser.bind(eng)
    builder = SpatialObsBuilder(unit_status=True)
    builder.bind(eng, parser)
    at = builder.channel_names()
    own, foe = at.index("own_on_tower"), at.index("enemy_on_tower")
    builder.reset(eng.state())
    t = eng.arena().subtile
    hand = eng.state().players[BLUE].hand
    x, y = to_engine(eng.arena(), BLUE, 3 * t + t // 2, 14 * t + t // 2)
    hog_play = DeployCommand(BLUE, hand.index(ids["HogRider"]), x, y)
    assert eng.step([hog_play], 1)[0].status == DeployStatus.OK
    towers = {e.uid for e in eng.state().entities if e.kind == EntityKind.PRINCESS_TOWER}
    seen = []
    for _ in range(400):
        eng.step([], 1)
        state = eng.state()
        hog = [e for e in state.entities if e.team == BLUE and e.card_id == ids["HogRider"]]
        if not hog:
            break
        blue = builder.build(state, BLUE, parser.action_mask(state, BLUE))["spatial"]
        red = builder.build(state, RED, parser.action_mask(state, RED))["spatial"]
        seen.append((hog[0].target_uid in towers, blue[own].sum(), red[foe].sum()))
    assert any(locked for locked, _, _ in seen), "the Hog never locked on a tower"
    assert all(b == r == float(locked) for locked, b, r in seen), seen[:20]
    assert not seen[0][0], "a Hog still deploying is not locked on anything"
