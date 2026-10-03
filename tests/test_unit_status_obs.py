"""What is on a unit: ``SpatialObsBuilder(unit_status=True)``.

Six planes, own then enemy, counted per tile like the troop planes:
- ``own_shield`` / ``enemy_shield``: shield hp left, summed and scaled like the hp planes (a
  Dark Prince's shield is drawn over it, and a shot breaks it before any hp goes);
- ``own_raged`` / ``enemy_raged``: units under a Rage (drawn purple; they move and hit faster);
- ``own_slowed`` / ``enemy_slowed``: units slowed by cold (an Ice Wizard's, a hero Ice Golem's).
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
import pytest

from royalegym.action import TileActionParser
from royalegym.mock_engine import MockEngine
from royalegym.obs import HP_SCALE, Reveal, SpatialObsBuilder, spatial_channels, status_channels
from royalegym.protocol import (
    BLUE,
    RED,
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

PLANES = ["own_shield", "enemy_shield", "own_raged", "enemy_raged", "own_slowed", "enemy_slowed"]
SLOW = "IceWizardSlowDown|IceWizardCold|IceGolemiteHero_Slow_Buff_Tower"


# --- layout -------------------------------------------------------------------------------


def test_off_by_default_and_after_every_other_optional_fair_plane():
    assert not set(PLANES) & {n for n, _ in spatial_channels()}
    rev = Reveal(enemy_spell_aim=True)
    before = [n for n, _ in spatial_channels(rev, True, True, heroes=True)]
    after = [n for n, _ in spatial_channels(rev, True, True, heroes=True, unit_status=True)]
    assert after[: len(before) - 1] == before[:-1], "an existing plane moved"
    assert after[len(before) - 1 : len(before) + 5] == PLANES
    assert after[-1] == "enemy_spell_aim", "the revealed planes stay last"


def test_config_records_the_flag():
    assert SpatialObsBuilder(unit_status=True).config()["unit_status"] is True
    assert "unit_status" not in SpatialObsBuilder().config()


# --- the planes, on a forged board ---------------------------------------------------------


def _board():
    eng = MockEngine()
    eng.reset(1, MatchSetup(decks=[list(range(8)), list(range(8))]))
    return eng.state(), eng.arena()


def _unit(uid, team, point, shield=0, buffs=(), kind=EntityKind.TROOP):
    return EntityState(
        uid, team, int(kind), 3, -1, point[0], point[1], 100, 100, 500, False, 0,
        shield=shield, buffs=tuple(buffs), status_flags=0,
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
    assert blue.shape == (6, a.tiles_y, a.tiles_x)
    assert blue[0][_tile(a, BLUE, p)] == pytest.approx(450 / HP_SCALE)
    assert blue[0].sum() == pytest.approx(450 / HP_SCALE)
    assert blue[1].sum() == 0
    assert red[1][_tile(a, RED, p)] == pytest.approx(450 / HP_SCALE)
    assert red[0].sum() == 0


def test_rage_and_cold_are_matched_by_a_member_of_the_family():
    state, a = _board()
    p, q = _at(a, 4, 20), _at(a, 12, 6)
    s = _with(state, [
        _unit(100, RED, p, buffs=[("Rage", 3000)]),
        _unit(101, RED, p, buffs=[(SLOW, 1200), ("Poison", 500)]),
        _unit(102, BLUE, q, buffs=[("Rage|SomeLaterRage", 800), (SLOW + "|Later", 100)]),
    ])
    blue = status_channels(s.entities, BLUE, a)
    assert blue[3][_tile(a, BLUE, p)] == 1, "enemy raged"
    assert blue[5][_tile(a, BLUE, p)] == 1, "enemy slowed"
    assert blue[2][_tile(a, BLUE, q)] == 1, "own raged, the family grown by a member"
    assert blue[4][_tile(a, BLUE, q)] == 1, "own slowed, the family grown by a member"
    assert blue[2].sum() == 1
    assert blue[4].sum() == 1


def test_a_name_that_only_contains_a_member_does_not_match():
    state, a = _board()
    p = _at(a, 4, 20)
    lookalikes = [("BarbarianRage", 3000), ("IceWizardSlowDownX", 9)]
    s = _with(state, [_unit(100, RED, p, buffs=lookalikes)])
    assert status_channels(s.entities, BLUE, a).sum() == 0


def test_poison_tornado_and_freeze_are_not_status_planes():
    state, a = _board()
    p = _at(a, 4, 20)
    s = _with(state, [_unit(100, RED, p, buffs=[
        ("Poison", 900), ("Tornado", 400), ("Freeze|ZapFreeze", 300),
    ])])
    assert status_channels(s.entities, BLUE, a).sum() == 0


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
