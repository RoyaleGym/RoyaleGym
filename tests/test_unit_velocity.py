"""How far each unit moved: ``SpatialObsBuilder(unit_velocity=True)`` (2026-11-11).

Eight planes, own then enemy, per tile like the troop planes: ``own_dx_plus`` / ``enemy_dx_plus``,
``own_dx_minus`` / ``enemy_dx_minus``, then the same four for dy. Each unit's displacement since
the seat's last observation, in its own frame, scaled from the real tick gap to 5 ticks (so 250
and 500 ms decisions read one speed), in whole thousandths of a tile, each axis clipped to 3 tiles,
added to the plane of its sign at the unit's tile now. +y is toward the enemy king for both seats.
Crown towers are in none of them, and a unit the seat did not see last time adds nothing.

The builder keeps one track per seat: the env builds both seats from one builder on every state,
and a shared track would measure Blue's step against Red's view. The seat mirror cannot catch that
(Red's view of the mirrored state equals Blue's of the original), so a test builds both seats.

SKIPS
    The engine tests skip only without the engine.
"""

from __future__ import annotations

import msgspec
import numpy as np
import pytest

from royalegym.env import ClashParallelEnv
from royalegym.mock_engine import MockEngine
from royalegym.obs import (
    VELOCITY_SPATIAL_CHANNELS,
    Reveal,
    SpatialObsBuilder,
    UnitTrack,
    spatial_channels,
    velocity_channels,
)
from royalegym.protocol import BLUE, RED, EntityKind, EntityState, MatchSetup, mirror_state, to_own
from royalegym.rust_engine import CORE_IMPORT_ERROR, core_available

PLANES = [
    "own_dx_plus", "enemy_dx_plus", "own_dx_minus", "enemy_dx_minus",
    "own_dy_plus", "enemy_dy_plus", "own_dy_minus", "enemy_dy_minus",
]
needs_engine = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))


def test_off_by_default_and_after_every_other_optional_plane():
    assert [n for n, _ in VELOCITY_SPATIAL_CHANNELS] == PLANES
    assert not set(PLANES) & {n for n, _ in spatial_channels()}
    rev = Reveal(enemy_spell_aim=True)
    kw = dict(heroes=True, unit_status=True, unit_actions=True)
    before = [n for n, _ in spatial_channels(rev, True, True, **kw)]
    after = [n for n, _ in spatial_channels(rev, True, True, **kw, unit_velocity=True)]
    assert after[: len(before) - 1] == before[:-1], "an existing plane moved"
    assert after[len(before) - 1 : len(before) - 1 + len(PLANES)] == PLANES
    assert after[-1] == "enemy_spell_aim"


def test_config_records_the_flag():
    assert SpatialObsBuilder(unit_velocity=True).config()["unit_velocity"] is True
    assert "unit_velocity" not in SpatialObsBuilder().config()


# --- forged board ---------------------------------------------------------------------------


def _board():
    eng = MockEngine()
    eng.reset(1, MatchSetup(decks=[list(range(8)), list(range(8))]))
    return eng.state(), eng.arena()


def _unit(uid, team, point):
    return EntityState(uid, team, int(EntityKind.TROOP), 3, -1, point[0], point[1], 100, 100,
                       500, False, 0)


def _with(state, tick, units):
    towers = [e for e in state.entities
              if e.kind in (EntityKind.KING_TOWER, EntityKind.PRINCESS_TOWER)]
    return msgspec.structs.replace(state, tick=tick, entities=towers + units)


def _at(a, x, y):
    """A point a little inside tile (x, y), so a move of a fraction of a tile stays on it."""
    return (a.subtile * x + a.subtile // 3, a.subtile * y + a.subtile // 3)


def _planes(steps, team=BLUE):
    """velocity_channels after a track saw each (tick, units) in turn, for ``team``."""
    state, a = _board()
    track = UnitTrack()
    for tick, units in steps:
        s = _with(state, tick, units)
        track.see(s, team, a)
    return velocity_channels(s.entities, team, a, track), a, s


def _value(planes, name, a, team, point):
    ox, oy = to_own(a, team, *point)
    return float(planes[PLANES.index(name)][oy // a.subtile, ox // a.subtile])


def test_a_unit_moving_up_reads_its_tiles_per_five_ticks_on_its_tile_and_nowhere_else():
    _, a = _board()
    p0, p1 = _at(a, 4, 10), (_at(a, 4, 10)[0], _at(a, 4, 10)[1] + a.subtile // 10)
    planes, a, _ = _planes([(100, [_unit(7, BLUE, p0)]), (105, [_unit(7, BLUE, p1)])])
    assert _value(planes, "own_dy_plus", a, BLUE, p1) == pytest.approx(0.1, abs=1e-6)
    assert np.count_nonzero(planes) == 1, "one unit, one axis, one tile"


def test_each_sign_and_axis_has_its_plane():
    _, a = _board()
    p0 = _at(a, 9, 15)
    s = a.subtile // 5
    moves = {
        "own_dx_plus": (p0[0] + s, p0[1]), "own_dx_minus": (p0[0] - s, p0[1]),
        "own_dy_plus": (p0[0], p0[1] + s), "own_dy_minus": (p0[0], p0[1] - s),
    }
    for name, p1 in moves.items():
        planes, a, _ = _planes([(100, [_unit(7, BLUE, p0)]), (105, [_unit(7, BLUE, p1)])])
        assert _value(planes, name, a, BLUE, p1) == pytest.approx(0.2, abs=1e-6), name
        assert np.count_nonzero(planes) == 1, name
        planes, a, _ = _planes([(100, [_unit(8, RED, p0)]), (105, [_unit(8, RED, p1)])])
        # Red's unit seen by Blue: the enemy plane, Blue's frame (no flip).
        assert _value(planes, name.replace("own", "enemy"), a, BLUE, p1) == pytest.approx(0.2)


def test_opposite_movers_on_one_tile_both_show_and_alike_ones_sum():
    _, a = _board()
    p = _at(a, 6, 12)
    q = a.subtile // 10
    planes, a, _ = _planes([
        (100, [_unit(1, BLUE, p), _unit(3, BLUE, p), _unit(5, BLUE, p)]),
        (105, [_unit(1, BLUE, (p[0] + q, p[1])), _unit(3, BLUE, (p[0] - 2 * q, p[1])),
               _unit(5, BLUE, (p[0] + q, p[1]))]),
    ])
    assert _value(planes, "own_dx_plus", a, BLUE, p) == pytest.approx(0.2, abs=1e-6)
    assert _value(planes, "own_dx_minus", a, BLUE, p) == pytest.approx(0.2, abs=1e-6)


def test_a_jump_is_clipped_to_three_tiles():
    _, a = _board()
    p0, p1 = _at(a, 3, 5), _at(a, 3, 9)  # 4 tiles in 5 ticks
    planes, a, _ = _planes([(100, [_unit(7, BLUE, p0)]), (105, [_unit(7, BLUE, p1)])])
    assert _value(planes, "own_dy_plus", a, BLUE, p1) == pytest.approx(3.0)


def test_a_new_unit_and_the_towers_add_nothing():
    state, a = _board()
    p = _at(a, 5, 8)
    planes, a, s = _planes([(100, []), (105, [_unit(9, BLUE, p)])])
    assert not planes.any(), "a unit not seen last time has no motion yet"
    moved = [msgspec.structs.replace(e, x=e.x + a.subtile) for e in s.entities
             if e.kind in (EntityKind.KING_TOWER, EntityKind.PRINCESS_TOWER)]
    track = UnitTrack()
    track.see(_with(state, 100, []), BLUE, a)
    later = msgspec.structs.replace(s, tick=105, entities=moved)
    track.see(later, BLUE, a)
    assert not velocity_channels(later.entities, BLUE, a, track).any()


def test_the_speed_is_per_five_ticks_whatever_the_gap():
    """The real gap is scaled to 5 ticks: the same speed at gaps 5, 7 and 10 reads alike."""
    _, a = _board()
    p0 = _at(a, 9, 10)
    per_tick = a.subtile // 50  # 0.02 tiles a tick, 0.1 per 5
    got = []
    for gap in (5, 7, 10):
        p1 = (p0[0], p0[1] + per_tick * gap)
        planes, a, _ = _planes([(100, [_unit(7, BLUE, p0)]), (100 + gap, [_unit(7, BLUE, p1)])])
        got.append(_value(planes, "own_dy_plus", a, BLUE, p1))
    assert got == [pytest.approx(0.1, abs=1e-6)] * 3, got


def test_red_reads_a_mirrored_motion_as_blue_reads_the_original():
    """Own frame for both: Red's view of the mirrored state is Blue's of the original."""
    state, a = _board()
    p0 = _at(a, 4, 10)
    p1 = (p0[0] + a.subtile // 10, p0[1] + a.subtile // 4)  # 0.1 and 0.25 tiles in 5 ticks
    e0, e1 = _with(state, 100, [_unit(6, BLUE, p0)]), _with(state, 105, [_unit(6, BLUE, p1)])
    blue, red = UnitTrack(), UnitTrack()
    for s in (e0, e1):
        blue.see(s, BLUE, a)
        red.see(mirror_state(a, s), RED, a)
    b = velocity_channels(e1.entities, BLUE, a, blue)
    m = mirror_state(a, e1)
    r = velocity_channels(m.entities, RED, a, red)
    assert b.any(), "vacuous: nothing moved"
    assert np.array_equal(b, r)
    # And the same unit, seen by Red in the original battle, is an enemy coming down.
    seen = UnitTrack()
    for s in (e0, e1):
        seen.see(s, RED, a)
    enemy = velocity_channels(e1.entities, RED, a, seen)
    assert _value(enemy, "enemy_dy_minus", a, RED, p1) == pytest.approx(0.25, abs=1e-6)
    assert _value(enemy, "enemy_dx_minus", a, RED, p1) == pytest.approx(0.1, abs=1e-6)


def test_the_track_forgets_on_reset_and_when_the_tick_goes_back():
    state, a = _board()
    p0, p1 = _at(a, 4, 10), _at(a, 4, 11)
    track = UnitTrack()
    track.see(_with(state, 200, [_unit(7, BLUE, p0)]), BLUE, a)
    later = _with(state, 105, [_unit(7, BLUE, p1)])  # a new battle reusing uid 7, unreset
    track.see(later, BLUE, a)
    assert not velocity_channels(later.entities, BLUE, a, track).any()
    track.see(_with(state, 110, [_unit(7, BLUE, p0)]), BLUE, a)
    track.reset()
    again = _with(state, 115, [_unit(7, BLUE, p1)])
    track.see(again, BLUE, a)
    assert not velocity_channels(again.entities, BLUE, a, track).any()


def test_a_rebuild_of_the_same_tick_reads_the_same_motion():
    _, a = _board()
    p0, p1 = _at(a, 4, 10), _at(a, 4, 11)
    first, a, s = _planes([(100, [_unit(7, BLUE, p0)]), (105, [_unit(7, BLUE, p1)])])
    track = UnitTrack()
    state, _ = _board()
    for tick, units in ((100, [_unit(7, BLUE, p0)]), (105, [_unit(7, BLUE, p1)]),
                        (105, [_unit(7, BLUE, p1)])):
        track.see(_with(state, tick, units), BLUE, a)
    assert np.array_equal(velocity_channels(s.entities, BLUE, a, track), first)


# --- the builder in an env ------------------------------------------------------------------


def _played(builder, steps=80, seed=4):
    env = ClashParallelEnv(MockEngine(), obs_builder=builder)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    out = [obs]
    for _ in range(steps):
        if not env.agents:
            break
        acts = {}
        for agent in env.agents:
            legal = np.flatnonzero(obs[agent]["action_mask"])[1:]
            acts[agent] = int(rng.choice(legal)) if legal.size and rng.random() < 0.4 else 0
        obs, *_ = env.step(acts)
        out.append(obs)
    return env, out


def test_each_seat_keeps_its_own_track():
    """Both seats are built from one builder on each state; Blue's planes must be those of a
    builder that built only Blue."""
    env, both = _played(SpatialObsBuilder(unit_velocity=True))
    names = env.obs_builder.channel_names()
    at = names.index("own_dx_plus")
    solo = SpatialObsBuilder(unit_velocity=True)
    replay = ClashParallelEnv(MockEngine(), obs_builder=solo)
    obs, _ = replay.reset(seed=4, options={"mask_only": ["red"]})
    rng = np.random.default_rng(4)
    moving = 0
    for i, expect in enumerate(both):
        got = obs["blue"]["spatial"][at : at + 8]
        assert np.array_equal(got, expect["blue"]["spatial"][at : at + 8]), f"step {i}"
        moving += int(np.count_nonzero(got))
        if i + 1 == len(both) or not replay.agents:
            break
        acts = {}
        for agent in replay.agents:
            legal = np.flatnonzero(both[i][agent]["action_mask"])[1:]
            acts[agent] = int(rng.choice(legal)) if legal.size and rng.random() < 0.4 else 0
        obs, *_ = replay.step(acts)
    assert moving > 20, "vacuous: nothing moved"


def test_planes_hold_whole_thousandths_and_motion_shows_in_a_played_battle():
    env, seen = _played(SpatialObsBuilder(unit_velocity=True), steps=120)
    names = env.obs_builder.channel_names()
    at = names.index("own_dx_plus")
    nonzero = 0
    for obs in seen:
        for agent in obs:
            v = obs[agent]["spatial"][at : at + 8].astype(np.float32)
            q = np.rint(v.astype(np.float64) * 1000)
            assert (q >= 0).all()
            assert np.array_equal(q.astype(np.float32) / np.float32(1000), v)
            nonzero += int(np.count_nonzero(v))
    assert nonzero > 50, f"only {nonzero} cells moved"


@needs_engine
def test_a_walker_reads_one_speed_at_250_and_500_ms():
    """A lone Knight walking straight: its own_dy_plus at its tile is the same speed at both
    cadences (Speed 60 is 60 millitiles a tick: 0.3 tiles per 5 ticks)."""
    from royalegym.rust_engine import RustEngine
    from royalegym.state_mutator import DefaultStateMutator

    medians = {}
    for ms in (250, 500):
        engine = RustEngine()
        ids = {c.name: c.card_id for c in engine.cards()}
        deck = [ids[n] for n in ("Knight", "Archer", "Giant", "Minions", "Fireball", "Zap",
                                 "Cannon", "Musketeer")]
        env = ClashParallelEnv(engine, decision_ms=ms,
                               obs_builder=SpatialObsBuilder(unit_velocity=True),
                               state_mutator=DefaultStateMutator(decks=[deck, deck]))
        obs, _ = env.reset(seed=2)
        at = env.obs_builder.channel_names().index("own_dy_plus")
        played, values = False, []
        while env.agents and env.battle_state.tick < 900:
            act = 0
            hand = env.battle_state.players[BLUE].hand
            if not played and ids["Knight"] in hand:
                parser = env.action_parser
                lo = 1 + hand.index(ids["Knight"]) * parser.nx * parser.ny
                legal = np.flatnonzero(obs["blue"]["action_mask"])
                knight = legal[(legal >= lo) & (legal < lo + parser.nx * parser.ny)]
                if knight.size:
                    act = int(knight[len(knight) // 2])
                    played = True
            obs, *_ = env.step({"blue": act, "red": 0})
            plane = obs["blue"]["spatial"][at]
            if played and plane.any():
                values.append(float(plane.max()))
        assert played, ms
        assert len(values) > 10, (ms, len(values))
        medians[ms] = float(np.median(values))
    assert medians[250] == pytest.approx(medians[500], abs=0.01), medians
    assert 0.2 < medians[250] < 0.4, medians
