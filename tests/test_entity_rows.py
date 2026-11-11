"""A fuller per-unit row on ``EntityListObsBuilder``: appended columns behind four flags (11-11).

``unit_motion`` (has_prev, dx, dy), ``unit_stats`` (log_max_hp, shield, has_target, target_dx,
target_dy), ``unit_status`` (one column per status bit) and ``unit_actions`` (charge,
ability_ticks, the attack phase one-hot, facing_x/_y, has_dest, dest_dx/_dy). Each is off by
default and appends after the card one-hot, so the default row is byte for byte what it was.
Signed columns are [-1, 1], and the entities Box carries each column's own bounds, so a signed
value is not clipped to 0. The three columns MockEngine cannot report refuse it by name.

SKIPS
    The engine tests skip only without the engine.
"""

from __future__ import annotations

import math

import msgspec
import numpy as np
import pytest

from royalegym.action import TileActionParser
from royalegym.env import ClashParallelEnv
from royalegym.mock_engine import MockEngine
from royalegym.obs import ENTITY_EXTRA_COLUMNS, EntityListObsBuilder
from royalegym.protocol import (
    BLUE,
    RED,
    STATUS_BIT_NAMES,
    EntityKind,
    EntityState,
    MatchSetup,
    mirror_state,
)
from royalegym.rust_engine import CORE_IMPORT_ERROR, core_available

needs_engine = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
ALL = dict(unit_motion=True, unit_stats=True, unit_status=True, unit_actions=True)
DECK = ["Miner", "Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Musketeer"]


def _bound(engine=None, **flags):
    eng = engine or MockEngine()
    parser = TileActionParser()
    parser.bind(eng)
    b = EntityListObsBuilder(**flags)
    b.bind(eng, parser)
    return b, eng, parser


def test_off_by_default_the_row_is_what_it_was():
    plain, _, _ = _bound()
    assert plain.features == plain.BASE_FEATURES + plain.num_cards + 1
    assert not any(k in plain.config() for k in ALL)
    box = plain.observation_space()["entities"]
    assert float(box.low.min()) == 0.0
    assert float(box.high.max()) == 1.0
    assert plain.channel_names()[plain.channel_names().index("entity.card_one_hot") + 1] == (
        "spell.present"
    )


def test_the_columns_follow_the_card_one_hot_in_flag_order_with_their_bounds():
    b, _, _ = _bound(**ALL)
    want = [c for flag in ALL for c in ENTITY_EXTRA_COLUMNS[flag]]
    names = b.channel_names()
    at = names.index("entity.card_one_hot") + 1
    assert names[at : at + len(want)] == [f"entity.{n}" for n, _, _ in want]
    assert b.features == b.BASE_FEATURES + b.num_cards + 1 + len(want)
    box = b.observation_space()["entities"]
    first = b.BASE_FEATURES + b.num_cards + 1
    assert box.low[0, first:].tolist() == [lo for _, lo, _ in want]
    assert box.high[0, first:].tolist() == [hi for _, _, hi in want]
    assert [n for n, _, _ in ENTITY_EXTRA_COLUMNS["unit_status"]] == [
        f"status_{n}" for n in STATUS_BIT_NAMES
    ]
    cfg = b.config()
    assert all(cfg[k] is True for k in ALL)


@pytest.mark.parametrize("flag", ["unit_stats", "unit_status", "unit_actions"])
def test_an_engine_that_says_nothing_of_its_units_is_refused_by_name(flag):
    env = ClashParallelEnv(MockEngine(), obs_builder=EntityListObsBuilder(**{flag: True}))
    with pytest.raises(ValueError, match=flag):
        env.reset(seed=1)
    ClashParallelEnv(MockEngine(), obs_builder=EntityListObsBuilder(unit_motion=True)).reset(seed=1)


# --- forged rows ------------------------------------------------------------------------------


def _state():
    eng = MockEngine()
    eng.reset(1, MatchSetup(decks=[list(range(8)), list(range(8))]))
    s = eng.state()
    towers = [msgspec.structs.replace(e, status_flags=0, charge=0, attack_phase=0)
              for e in s.entities if e.kind in (EntityKind.KING_TOWER, EntityKind.PRINCESS_TOWER)]
    return s, towers


def _unit(uid, team, x, y, **kw):
    kw.setdefault("status_flags", 0)
    kw.setdefault("charge", 0)
    kw.setdefault("attack_phase", 0)
    return EntityState(uid, team, int(EntityKind.TROOP), 3, -1, x, y, 300, 1200, 500, False, 0,
                       **kw)


def _troops(b, out):
    """The rows of troops (kind one-hot troop), as {name: value} dicts, in row order."""
    names = b.channel_names()
    at = names.index("entity.card_one_hot")
    extra = [n.removeprefix("entity.") for n in names[at + 1 :] if n.startswith("entity.")]
    first = b.BASE_FEATURES + b.num_cards + 1
    return [
        dict(zip(extra, row[first:].tolist(), strict=True))
        for row in out["entities"]
        if row[0] == 1 and row[3] == 1
    ]


def test_motion_reads_tiles_per_five_ticks_over_three_and_has_prev():
    b, eng, parser = _bound(unit_motion=True)
    s, towers = _state()
    t = eng.arena().subtile
    p = (5 * t + t // 3, 10 * t + t // 3)
    first = msgspec.structs.replace(s, tick=100, entities=[*towers, _unit(2, BLUE, *p)])
    later = msgspec.structs.replace(s, tick=105, entities=[
        *towers, _unit(2, BLUE, p[0], p[1] + t // 2), _unit(4, BLUE, p[0] + 3 * t, p[1])])
    b.build(first, BLUE, parser.action_mask(first, BLUE))
    rows = _troops(b, b.build(later, BLUE, parser.action_mask(later, BLUE)))
    moved = next(r for r in rows if r["has_prev"] == 1)
    new = next(r for r in rows if r["has_prev"] == 0)
    assert moved["dx"] == 0
    assert moved["dy"] == pytest.approx(0.5 / 3, abs=1e-6)
    assert (new["dx"], new["dy"]) == (0, 0)


def test_stats_read_the_target_offset_shield_and_log_max_hp():
    b, eng, parser = _bound(unit_stats=True)
    s, towers = _state()
    t = eng.arena().subtile
    hunter = _unit(2, BLUE, 5 * t, 10 * t, target_uid=5, shield=500)
    prey = _unit(5, RED, 7 * t, 7 * t)
    st = msgspec.structs.replace(s, entities=[*towers, hunter, prey])
    for seat, sign in ((BLUE, 1), (RED, -1)):
        rows = _troops(b, b.build(st, seat, parser.action_mask(st, seat)))
        h = next(r for r in rows if r["has_target"] == 1)
        assert h["target_dx"] == pytest.approx(sign * 0.2, abs=1e-6), seat
        assert h["target_dy"] == pytest.approx(sign * -0.3, abs=1e-6), seat
        assert h["shield"] == pytest.approx(0.5)
        assert h["log_max_hp"] == pytest.approx(math.log1p(1200) / 16, abs=1e-6)
        other = next(r for r in rows if r["has_target"] == 0)
        assert (other["target_dx"], other["target_dy"], other["shield"]) == (0, 0, 0)


def test_status_has_one_column_per_bit():
    b, eng, parser = _bound(unit_status=True)
    s, towers = _state()
    t = eng.arena().subtile
    st = msgspec.structs.replace(s, entities=[*towers, _unit(2, BLUE, 5 * t, 10 * t,
                                                             status_flags=1 | 8 | 512)])
    (row,) = _troops(b, b.build(st, BLUE, parser.action_mask(st, BLUE)))
    on = {n.removeprefix("status_") for n, v in row.items() if v == 1}
    assert on == {"underground", "evolved", "grounded"}


def test_actions_read_charge_ticks_phase_facing_and_the_landing_point():
    b, eng, parser = _bound(unit_actions=True)
    s, towers = _state()
    t = eng.arena().subtile
    miner = _unit(2, BLUE, 5 * t, 10 * t, charge=500, ability_ticks=100, attack_phase=1,
                  facing=(3, 4), dest_x=5 * t, dest_y=15 * t)
    st = msgspec.structs.replace(s, entities=[*towers, miner])
    for seat, sign in ((BLUE, 1), (RED, -1)):
        (row,) = _troops(b, b.build(st, seat, parser.action_mask(st, seat)))
        assert row["charge"] == pytest.approx(0.5)
        assert row["ability_ticks"] == pytest.approx(0.5)
        assert (row["attack_idle"], row["attack_windup"], row["attack_fire"]) == (0, 1, 0)
        assert row["facing_x"] == pytest.approx(sign * 0.6, abs=1e-6)
        assert row["facing_y"] == pytest.approx(sign * 0.8, abs=1e-6)
        assert row["has_dest"] == 1
        assert row["dest_dx"] == 0
        assert row["dest_dy"] == pytest.approx(sign * 0.25, abs=1e-6)


def test_the_drop_count_says_how_many_rows_did_not_fit():
    eng = MockEngine()
    parser = TileActionParser()
    parser.bind(eng)
    b = EntityListObsBuilder(max_entities=4)
    b.bind(eng, parser)
    eng.reset(1, MatchSetup(decks=[list(range(8)), list(range(8))]))
    s = eng.state()
    b.build(s, BLUE, parser.action_mask(s, BLUE))
    assert b.entities_dropped[BLUE] == len(s.entities) - 4 > 0


# --- played engine states ---------------------------------------------------------------------


def _engine_states(n=260, seed=9):
    from royalegym import make_env

    env = make_env(deck=DECK)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    states = [env.battle_state]
    for _ in range(n):
        if not env.agents:
            break
        acts = {}
        for agent in env.agents:
            legal = np.flatnonzero(obs[agent]["action_mask"])[1:]
            acts[agent] = int(rng.choice(legal)) if legal.size and rng.random() < 0.5 else 0
        obs, *_ = env.step(acts)
        states.append(env.battle_state)
    return env.engine, states


@needs_engine
def test_red_on_the_mirror_sees_the_rows_blue_sees_with_every_column():
    engine, states = _engine_states()
    b, _, parser = _bound(engine, **ALL)
    arena = engine.arena()
    bad, shown = [], {"has_prev": 0, "has_target": 0, "has_dest": 0}
    for s in states:
        m = mirror_state(arena, s)
        ob = b.build(s, BLUE, parser.action_mask(s, BLUE))["entities"]
        orr = b.build(m, RED, parser.action_mask(m, RED))["entities"]
        if not np.array_equal(ob, orr):
            bad.append(s.tick)
        for row in _troops(b, {"entities": ob}):
            for k in shown:
                shown[k] += int(row[k] == 1)
    assert bad == [], bad[:5]
    assert all(v > 0 for v in shown.values()), f"vacuous: {shown}"


@needs_engine
def test_the_rows_do_not_depend_on_the_engines_list_order():
    engine, states = _engine_states(n=120)
    rng = np.random.default_rng(3)
    for i in range(30, len(states), 15):
        prev, s = states[i - 1], states[i]
        outs = []
        for trial in range(3):
            b, _, parser = _bound(engine, **ALL)
            ents = list(s.entities)
            if trial:
                rng.shuffle(ents)
            shuffled = msgspec.structs.replace(s, entities=ents)
            for seat in (BLUE, RED):
                b.build(prev, seat, parser.action_mask(prev, seat))
                outs.append(b.build(shuffled, seat, parser.action_mask(shuffled, seat))["entities"])
        assert all(np.array_equal(outs[k], outs[k % 2]) for k in range(len(outs))), s.tick


@needs_engine
def test_every_value_sits_inside_its_columns_bounds_in_a_played_battle():
    engine, states = _engine_states(n=200)
    b, _, parser = _bound(engine, **ALL)
    box = b.observation_space()["entities"]
    lowest = np.zeros(b.features)
    for s in states:
        for seat in (BLUE, RED):
            out = b.build(s, seat, parser.action_mask(s, seat))["entities"]
            assert box.contains(out), s.tick
            lowest = np.minimum(lowest, out.min(axis=0))
    assert lowest.min() < 0, "vacuous: no signed column ever went below 0"


@needs_engine
def test_a_snapshot_carries_each_seats_track_and_refuses_one_without_it():
    from royalegym import make_env

    def env(**flags):
        return make_env(deck=DECK, obs_builder=EntityListObsBuilder(**flags))

    a = env(**ALL)
    obs, _ = a.reset(seed=4)
    rng = np.random.default_rng(4)
    for _ in range(40):
        acts = {}
        for g in a.agents:
            legal = np.flatnonzero(obs[g]["action_mask"])[1:]
            acts[g] = int(rng.choice(legal)) if legal.size and rng.random() < 0.4 else 0
        obs, *_ = a.step(acts)
    snap = a.snapshot()
    b = env(**ALL)
    resumed, _ = b.reset(options={"snapshot": snap})
    for g in ("blue", "red"):
        assert np.array_equal(resumed[g]["entities"], obs[g]["entities"]), g
    for _ in range(15):
        acts = {g: 0 for g in a.agents}
        oa, *_ = a.step(acts)
        ob, *_ = b.step(acts)
        for g in oa:
            assert np.array_equal(oa[g]["entities"], ob[g]["entities"]), g
    plain = env()
    plain.reset(seed=4)
    with pytest.raises(ValueError, match="unit_motion"):
        env(**ALL).reset(options={"snapshot": plain.snapshot()})


def test_twins_that_differ_only_in_an_appended_column_sort_by_it():
    """Two units alike in every base column, one with a target and one without: their order must
    come from the appended columns, not from the engine's list order."""
    b, eng, parser = _bound(unit_stats=True)
    s, towers = _state()
    t = eng.arena().subtile
    one = _unit(2, BLUE, 5 * t, 10 * t, target_uid=5)
    two = _unit(4, BLUE, 5 * t, 10 * t)
    prey = _unit(5, RED, 7 * t, 7 * t)
    outs = [
        b.build(st, BLUE, parser.action_mask(st, BLUE))["entities"]
        for st in (msgspec.structs.replace(s, entities=[*towers, one, two, prey]),
                   msgspec.structs.replace(s, entities=[*towers, two, one, prey]))
    ]
    assert np.array_equal(*outs)


def test_each_seat_keeps_its_own_motion_track():
    """Both seats are built from one builder on each state, as the env does; Blue's motion must
    be that of a builder that built only Blue. The mirror test cannot see a shared track."""
    def run(mask_red: bool):
        env = ClashParallelEnv(MockEngine(), obs_builder=EntityListObsBuilder(unit_motion=True))
        obs, _ = env.reset(seed=4, options={"mask_only": ["red"]} if mask_red else None)
        rng = np.random.default_rng(4)
        seen = []
        for _ in range(60):
            if not env.agents:
                break
            seen.append(obs["blue"]["entities"].copy())
            acts = {}
            for agent in env.agents:
                legal = np.flatnonzero(obs[agent]["action_mask"])[1:]
                acts[agent] = int(rng.choice(legal)) if legal.size and rng.random() < 0.4 else 0
            obs, *_ = env.step(acts)
        return env.obs_builder, seen

    builder, both = run(False)
    _, solo = run(True)
    assert len(both) == len(solo) > 30
    at = builder.BASE_FEATURES + builder.num_cards + 1
    assert any(np.abs(o[:, at + 1 : at + 3]).sum() > 0 for o in both), "vacuous: nothing moved"
    for i, (x, y) in enumerate(zip(both, solo, strict=True)):
        assert np.array_equal(x, y), f"step {i}"
