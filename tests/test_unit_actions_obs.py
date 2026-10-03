"""What a unit is doing: ``SpatialObsBuilder(unit_actions=True)``, from RoyaleSim ship35's columns.

Fourteen planes, own then enemy, per tile like the troop planes:
- ``own_charge`` / ``enemy_charge``: the most build-up on the tile, permille / 1000 (a Prince's
  run-up, a Sparky's load, an Inferno's ramp);
- ``own_charged`` / ``enemy_charged``: units fully charged (STATUS_CHARGED);
- ``own_windup`` / ``enemy_windup``: units winding up an ability (STATUS_WINDUP);
- ``own_ability_active`` / ``enemy_ability_active``: units with an ability running (a cloak, a
  dash chain, a hero's effect: STATUS_ABILITY_ACTIVE);
- ``own_ability_ticks`` / ``enemy_ability_ticks``: the most ability ticks left on the tile /
  ``ABILITY_TICKS_SCALE``;
- ``own_clone`` / ``enemy_clone``: a Clone's copies (STATUS_CLONE);
- ``own_tunnel_dest`` / ``enemy_tunnel_dest``: tunnellers counted at the tile they will come up
  on, from the first tick they are under (owner 2026-10-03: everything as soon as the engine
  has it, so no readability delay).

An engine before ship35 reports none of this: its units say -1 ("not said") for charge, and the
builder refuses it by name rather than draw zeros that read as "nothing charging".

Off by default, appended after every other optional plane.

SKIPS
    The engine test skips only without the engine.
"""

from __future__ import annotations

import msgspec
import pytest

from royalegym.action import TileActionParser
from royalegym.mock_engine import MockEngine
from royalegym.obs import (
    ABILITY_TICKS_SCALE,
    CARD_STATUS_FIELDS,
    FAIR_FIELDS,
    Reveal,
    SpatialObsBuilder,
    action_channels,
    spatial_channels,
)
from royalegym.protocol import (
    BLUE,
    RED,
    STATUS_ABILITY_ACTIVE,
    STATUS_CHARGED,
    STATUS_CLONE,
    STATUS_UNDERGROUND,
    STATUS_WINDUP,
    EntityKind,
    EntityState,
    MatchSetup,
    to_own,
)
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, _core, core_available

PLANES = [
    "own_charge", "enemy_charge", "own_charged", "enemy_charged", "own_windup", "enemy_windup",
    "own_ability_active", "enemy_ability_active", "own_ability_ticks", "enemy_ability_ticks",
    "own_clone", "enemy_clone", "own_tunnel_dest", "enemy_tunnel_dest",
]


def test_off_by_default_and_after_every_other_optional_plane():
    assert not set(PLANES) & {n for n, _ in spatial_channels()}
    rev = Reveal(enemy_spell_aim=True)
    before = [n for n, _ in spatial_channels(rev, True, True, heroes=True, unit_status=True)]
    after = [n for n, _ in spatial_channels(rev, True, True, heroes=True, unit_status=True,
                                            unit_actions=True)]
    assert after[: len(before) - 1] == before[:-1], "an existing plane moved"
    assert after[len(before) - 1 : len(before) - 1 + len(PLANES)] == PLANES
    assert after[-1] == "enemy_spell_aim"


def test_config_records_the_flag():
    assert SpatialObsBuilder(unit_actions=True).config()["unit_actions"] is True
    assert "unit_actions" not in SpatialObsBuilder().config()


def test_card_status_stays_out_of_the_fair_fields_imitate_rebuilds():
    """RoyaleImitate's public_log rebuilds FAIR_FIELDS from a play log and its parity tests
    compare them with the env's vector. The card_status fields need the board and the
    engine's rows, which a play log does not have, so moving one into FAIR_FIELDS would turn
    those tests red in another repo. Tell Learn before changing this."""
    assert not set(CARD_STATUS_FIELDS) & set(FAIR_FIELDS)


# --- forged board ---------------------------------------------------------------------------


def _board():
    eng = MockEngine()
    eng.reset(1, MatchSetup(decks=[list(range(8)), list(range(8))]))
    return eng.state(), eng.arena()


def _unit(uid, team, point, status=0, charge=0, dest=(-1, -1), ticks=0):
    return EntityState(
        uid, team, int(EntityKind.TROOP), 3, -1, point[0], point[1], 100, 100, 500, False, 0,
        status_flags=status, charge=charge, dest_x=dest[0], dest_y=dest[1], ability_ticks=ticks,
    )


def _with(state, entities):
    towers = [msgspec.structs.replace(e, status_flags=0, charge=0, dest_x=-1, dest_y=-1,
                                      ability_ticks=0)
              for e in state.entities
              if e.kind in (EntityKind.KING_TOWER, EntityKind.PRINCESS_TOWER)]
    return msgspec.structs.replace(state, entities=towers + entities)


def _tile(arena, viewer, point):
    ox, oy = to_own(arena, viewer, *point)
    return oy // arena.subtile, ox // arena.subtile


def _at(a, x, y):
    return (a.subtile * x + 100, a.subtile * y + 100)


def _plane(planes, name):
    return planes[PLANES.index(name)]


def test_charge_is_the_most_on_the_tile_and_bits_are_counted():
    state, a = _board()
    p, q = _at(a, 9, 12), _at(a, 4, 20)
    s = _with(state, [
        _unit(100, BLUE, p, charge=400),
        _unit(101, BLUE, p, charge=900, status=STATUS_CHARGED),
        _unit(102, RED, q, status=STATUS_WINDUP),
        _unit(103, RED, q, status=STATUS_ABILITY_ACTIVE, ticks=50),
        _unit(104, RED, q, status=STATUS_CLONE | STATUS_ABILITY_ACTIVE, ticks=80),
    ])
    blue = action_channels(s, BLUE, a)
    assert blue.shape == (len(PLANES), a.tiles_y, a.tiles_x)
    assert _plane(blue, "own_charge")[_tile(a, BLUE, p)] == pytest.approx(0.9)
    assert _plane(blue, "own_charge").sum() == pytest.approx(0.9)
    assert _plane(blue, "own_charged").sum() == 1
    assert _plane(blue, "enemy_windup")[_tile(a, BLUE, q)] == 1
    assert _plane(blue, "enemy_ability_active")[_tile(a, BLUE, q)] == 2
    assert _plane(blue, "enemy_ability_ticks")[_tile(a, BLUE, q)] == pytest.approx(
        80 / ABILITY_TICKS_SCALE
    )
    assert _plane(blue, "enemy_clone").sum() == 1
    assert _plane(blue, "own_windup").sum() == 0
    red = action_channels(s, RED, a)
    assert _plane(red, "enemy_charge")[_tile(a, RED, p)] == pytest.approx(0.9)
    assert _plane(red, "own_ability_active").sum() == 2


def test_a_tunneller_is_counted_where_it_will_come_up_from_its_first_tick_under():
    state, a = _board()
    p, dest = _at(a, 9, 4), _at(a, 3, 24)
    s = _with(state, [_unit(100, RED, p, status=STATUS_UNDERGROUND, dest=dest),
                      _unit(101, RED, p)])
    blue = action_channels(s, BLUE, a)
    assert _plane(blue, "enemy_tunnel_dest")[_tile(a, BLUE, dest)] == 1
    assert _plane(blue, "enemy_tunnel_dest").sum() == 1, "a unit with no dest is not landing"
    red = action_channels(s, RED, a)
    assert _plane(red, "own_tunnel_dest")[_tile(a, RED, dest)] == 1


def test_an_engine_before_the_columns_is_refused():
    state, a = _board()
    with pytest.raises(ValueError, match="does not report what units are doing"):
        action_channels(state, BLUE, a)  # MockEngine reports no status
    old = msgspec.structs.replace(state, entities=[
        msgspec.structs.replace(e, status_flags=0) for e in state.entities
    ])  # status reported, charge not: an engine before ship35
    with pytest.raises(ValueError, match="does not report what units are doing"):
        action_channels(old, BLUE, a)


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_the_engine_reports_the_columns_or_the_flag_refuses_it():
    """ONE test for both kinds of engine, so it never skips where CI builds the engine: an
    engine before ship35 is refused at the first build; one with it puts a Miner's landing
    tile on the board from its first tick under."""
    from royalegym import ClashParallelEnv

    has = "charge" in getattr(_core, "ENTITY_FIELDS", ())
    eng = RustEngine()
    names = {c.name for c in eng.cards()}
    deck = ["Miner", "Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Cannon"]
    if not has:
        env = ClashParallelEnv(eng, obs_builder=SpatialObsBuilder(unit_actions=True))
        with pytest.raises(ValueError, match="does not report what units are doing"):
            env.reset(seed=0)
        return
    assert "Miner" in names
    ids = {c.name: c.card_id for c in eng.cards()}
    eng.reset(1, MatchSetup(decks=[[ids[n] for n in deck]] * 2, shuffle=0,
                            elixir_milli=[10000, 10000],
                            start_tick=eng.rules().deploy_lockout_ticks))
    parser = TileActionParser()
    parser.bind(eng)
    b = SpatialObsBuilder(unit_actions=True)
    b.bind(eng, parser)
    b.reset(eng.state())
    plane = b.channel_names().index("enemy_tunnel_dest")
    from royalegym.protocol import DeployCommand, DeployStatus, to_engine

    t = eng.arena().subtile
    x, y = to_engine(eng.arena(), BLUE, 3 * t + t // 2, 20 * t + t // 2)  # off any tower
    hand = eng.state().players[BLUE].hand
    played = eng.step([DeployCommand(BLUE, hand.index(ids["Miner"]), x, y)], 1)
    assert played[0].status == DeployStatus.OK, played
    lit = []
    for _ in range(60):
        state = eng.state()
        under = [e for e in state.entities if e.team == BLUE and e.card_id == ids["Miner"]
                 and e.dest_x >= 0]
        red = b.build(state, RED, parser.action_mask(state, RED))["spatial"][plane]
        lit.append((bool(under), float(red.sum())))
        eng.step([], 1)
    assert any(u for u, _ in lit), "the Miner never reported a dest"
    assert all((s == 1.0) == u for u, s in lit), lit[:10]
