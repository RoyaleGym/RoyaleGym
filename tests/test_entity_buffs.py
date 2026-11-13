"""Two more per-unit column sets on ``EntityListObsBuilder``: buffs and the engine's extra unit
values (royalesim 0.1.53, 11-13).

``unit_buffs``: up to ``BUFF_SLOTS`` buffs per unit, the most time left first. Each slot's id is in
the ``entity_buffs`` key (0: no buff), and its time left / 10 s is an appended column. Ids come
from the buff vocabulary: the MEMBER names of the engine's ``buff_kinds()`` (a kind can be a '|'
group of rows the card table gives identical values), sorted, or a ``buff_names`` pin in any
order. A group writes the id of whichever member comes first in the vocabulary.

``unit_extras``: the engine's extra unit values (``protocol.ENTITY_EXTRA_FIELDS``): a spawner's
clock and the units its wave still has to put down, the dash state and its ticks, the speed and
hit speed the unit's buffs give it, and the river jump. The builder asks the engine for them at
bind (``RustEngine.set_entity_extras``).

Both are off by default and append after ``unit_actions``'s columns. MockEngine says neither,
and is refused by name.

SKIPS
    The engine tests skip without the engine, and without royalesim 0.1.53 (the first with
    ``Battle.buff_kinds_json`` and ``state_json(extra=True)``).
"""

from __future__ import annotations

import json
import re
import types

import msgspec
import numpy as np
import pytest

from royalegym.action import TileActionParser
from royalegym.env import ClashParallelEnv
from royalegym.mock_engine import MockEngine
from royalegym.obs import BUFF_SLOTS, ENTITY_EXTRA_COLUMNS, EntityListObsBuilder
from royalegym.protocol import (
    BLUE,
    ENTITY_EXTRA_FIELDS,
    RED,
    EntityKind,
    EntityState,
    MatchSetup,
    mirror_state,
)
from royalegym.rust_engine import CORE_IMPORT_ERROR, check_field_order, core_available

#: The kinds the forged engine says: one '|' group, whose members sort apart.
KINDS = ["IceWizardSlowDown", "Poison", "Rage", "ZapFreeze|Freeze"]
#: Their members, sorted: the vocabulary when nothing is pinned.
MEMBERS = ["Freeze", "IceWizardSlowDown", "Poison", "Rage", "ZapFreeze"]
ID = {name: 1 + i for i, name in enumerate(MEMBERS)}
NOT_SAID = dict.fromkeys(ENTITY_EXTRA_FIELDS, -1)
NONE_EXTRA = dict(spawn_ms=0, spawn_wave_left=0, dash_state=0, dash_ticks=0, speed_pct=100,
                  hit_speed_pct=100, jumping=0)


class BuffEngine(MockEngine):
    """MockEngine that says a buff vocabulary and takes the extras request, for forged rows."""

    def __init__(self, kinds=KINDS):
        super().__init__()
        self._kinds = list(kinds)
        self.extras_asked: list[bool] = []

    def buff_kinds(self):
        return list(self._kinds)

    def set_entity_extras(self, on=True):
        self.extras_asked.append(bool(on))


def _bound(engine=None, **flags):
    eng = engine or BuffEngine()
    parser = TileActionParser()
    parser.bind(eng)
    b = EntityListObsBuilder(**flags)
    b.bind(eng, parser)
    return b, eng, parser


def _state(eng):
    eng.reset(1, MatchSetup(decks=[list(range(8)), list(range(8))]))
    s = eng.state()
    towers = [msgspec.structs.replace(e, **NONE_EXTRA) for e in s.entities
              if e.kind in (EntityKind.KING_TOWER, EntityKind.PRINCESS_TOWER)]
    return s, towers


def _unit(uid, team, x, y, **kw):
    for k, v in NONE_EXTRA.items():
        kw.setdefault(k, v)
    return EntityState(uid, team, int(EntityKind.TROOP), 3, -1, x, y, 300, 1200, 500, False, 0,
                       **kw)


def _troops(b, out):
    """(appended columns as {name: value}, buff ids) of each troop row, in row order."""
    names = b.channel_names()
    at = names.index("entity.card_one_hot")
    extra = [n.removeprefix("entity.") for n in names[at + 1 :] if n.startswith("entity.")]
    first = b.BASE_FEATURES + b.num_cards + 1
    return [
        (dict(zip(extra, row[first:].tolist(), strict=True)),
         out["entity_buffs"][i].tolist() if "entity_buffs" in out else None)
        for i, row in enumerate(out["entities"])
        if row[0] == 1 and row[3] == 1
    ]


def _build(b, parser, s, team=BLUE):
    return b.build(s, team, parser.action_mask(s, team))


# --- layout -----------------------------------------------------------------------------------


def test_off_by_default_nothing_changes():
    plain, eng, _ = _bound()
    assert plain.features == plain.BASE_FEATURES + plain.num_cards + 1
    assert "entity_buffs" not in plain.observation_space().spaces
    assert not {"unit_buffs", "unit_extras", "buff_names"} & set(plain.config())
    assert eng.extras_asked == [], "a builder without unit_extras asked the engine"


def test_the_columns_follow_unit_actions_in_flag_order():
    flags = dict(unit_motion=True, unit_stats=True, unit_status=True, unit_actions=True,
                 unit_buffs=True, unit_extras=True)
    b, eng, _ = _bound(**flags)
    assert list(ENTITY_EXTRA_COLUMNS)[-2:] == ["unit_buffs", "unit_extras"]
    want = [c for flag in flags for c in ENTITY_EXTRA_COLUMNS[flag]]
    names = b.channel_names()
    at = names.index("entity.card_one_hot") + 1
    assert names[at : at + len(want)] == [f"entity.{n}" for n, _, _ in want]
    assert [n for n, _, _ in ENTITY_EXTRA_COLUMNS["unit_buffs"]] == [
        f"buff_left_{k}" for k in range(BUFF_SLOTS)
    ]
    assert ENTITY_EXTRA_COLUMNS["unit_extras"] == (
        ("spawn_clock", 0.0, 1.0),
        ("spawn_wave_left", 0.0, 1.0),
        ("dash_standing", 0.0, 1.0),
        ("dash_dashing", 0.0, 1.0),
        ("dash_ticks", 0.0, 1.0),
        ("speed", -1.0, 1.0),
        ("hit_speed", -1.0, 1.0),
        ("jumping", 0.0, 1.0),
    )
    box = b.observation_space()["entity_buffs"]
    assert (box.shape, box.dtype) == ((b.max_entities, BUFF_SLOTS), np.uint8)
    assert (int(box.low.min()), int(box.high.max())) == (0, len(MEMBERS))
    cfg = b.config()
    assert cfg["unit_buffs"] is True
    assert cfg["unit_extras"] is True
    assert cfg["buff_names"] == MEMBERS, "an unpinned builder's config pins what it bound to"
    assert eng.extras_asked == [True]


@pytest.mark.parametrize("flag", ["unit_buffs", "unit_extras"])
def test_an_engine_that_says_neither_is_refused_by_name(flag):
    with pytest.raises(ValueError, match=flag):
        ClashParallelEnv(MockEngine(), obs_builder=EntityListObsBuilder(**{flag: True}))


def test_a_buff_pin_needs_the_buff_flag():
    with pytest.raises(ValueError, match="unit_buffs"):
        EntityListObsBuilder(buff_names=MEMBERS)


def test_an_unbound_builders_config_pins_only_what_it_was_given():
    """Before bind there is no vocabulary to record: an unpinned builder's config must not say
    "pinned to nothing", which a rebuild would read as a pin and refuse every engine with."""
    cfg = EntityListObsBuilder(unit_buffs=True).config()
    assert "buff_names" not in cfg
    b, _, _ = _bound(unit_buffs=True, buff_names=cfg.get("buff_names"))
    assert b.config()["buff_names"] == MEMBERS
    pinned = EntityListObsBuilder(unit_buffs=True, buff_names=["Rage", *MEMBERS[:3], "ZapFreeze"])
    assert pinned.config()["buff_names"] == ["Rage", *MEMBERS[:3], "ZapFreeze"]


# --- the vocabulary ---------------------------------------------------------------------------


def test_slots_hold_the_most_time_left_first_and_a_group_writes_its_first_member():
    b, eng, parser = _bound(unit_buffs=True)
    s, towers = _state(eng)
    t = eng.arena().subtile
    u = _unit(2, BLUE, 5 * t, 10 * t, buffs=(("ZapFreeze|Freeze", 1000), ("Rage", 2500)))
    st = msgspec.structs.replace(s, entities=[*towers, u])
    ((cols, ids),) = _troops(b, _build(b, parser, st))
    assert ids == [ID["Rage"], ID["Freeze"], 0, 0, 0, 0]
    assert [cols[f"buff_left_{k}"] for k in range(BUFF_SLOTS)] == pytest.approx(
        [0.25, 0.1, 0, 0, 0, 0]
    )
    assert not _build(b, parser, msgspec.structs.replace(s, entities=towers))["entity_buffs"].any()


def test_a_pin_in_any_order_gives_ids_by_name():
    pin = ["Rage", "Unused", "ZapFreeze", "Poison", "IceWizardSlowDown", "Freeze"]
    b, eng, parser = _bound(unit_buffs=True, buff_names=pin)
    s, towers = _state(eng)
    t = eng.arena().subtile
    u = _unit(2, BLUE, 5 * t, 10 * t, buffs=(("ZapFreeze|Freeze", 1000), ("Rage", 2500)))
    ((_, ids),) = _troops(b, _build(b, parser, msgspec.structs.replace(s, entities=[*towers, u])))
    assert ids == [1, 3, 0, 0, 0, 0], "Rage is pin place 0, the group's first pinned member 2"
    assert b.config()["buff_names"] == pin
    assert int(b.observation_space()["entity_buffs"].high.max()) == len(pin)


def test_a_pin_without_one_of_the_engines_members_is_refused_by_name():
    with pytest.raises(ValueError, match=r"lacks: \['ZapFreeze'\]"):
        _bound(unit_buffs=True, buff_names=[m for m in MEMBERS if m != "ZapFreeze"])
    with pytest.raises(ValueError, match="twice"):
        _bound(unit_buffs=True, buff_names=[*MEMBERS, "Rage"])


def test_a_buff_the_engines_list_lacks_is_refused():
    b, eng, parser = _bound(unit_buffs=True)
    s, towers = _state(eng)
    t = eng.arena().subtile
    u = _unit(2, BLUE, 5 * t, 10 * t, buffs=(("Rage|Freeze", 900),))
    with pytest.raises(ValueError, match=re.escape("buff 'Rage|Freeze', which")):
        _build(b, parser, msgspec.structs.replace(s, entities=[*towers, u]))


def test_slots_past_the_last_are_dropped_and_counted_and_ties_go_by_id():
    b, eng, parser = _bound(unit_buffs=True, buff_names=[f"B{k}" for k in range(9)],
                            engine=BuffEngine([f"B{k}" for k in range(9)]))
    s, towers = _state(eng)
    t = eng.arena().subtile
    buffs = tuple((f"B{k}", 500 + 100 * (k // 2)) for k in range(8))  # pairs tie on time left
    u = _unit(2, BLUE, 5 * t, 10 * t, buffs=buffs)
    st = msgspec.structs.replace(s, entities=[*towers, u])
    ((cols, ids),) = _troops(b, _build(b, parser, st))
    assert ids == [7, 8, 5, 6, 3, 4], "most time left first, then the lower id"
    assert cols["buff_left_0"] == pytest.approx(0.08)
    assert b.buffs_dropped[BLUE] == 2


def test_time_left_reads_ten_seconds_as_one_and_saturates():
    b, eng, parser = _bound(unit_buffs=True)
    s, towers = _state(eng)
    t = eng.arena().subtile
    u = _unit(2, BLUE, 5 * t, 10 * t, buffs=(("Poison", 25000), ("Rage", 50)))
    ((cols, _),) = _troops(b, _build(b, parser, msgspec.structs.replace(s, entities=[*towers, u])))
    assert (cols["buff_left_0"], cols["buff_left_1"]) == pytest.approx((1.0, 0.005))


def test_buff_ids_stay_with_their_rows_through_the_sort():
    """Two units the sort puts in the other order from the engine's list: each row's ids must be
    its own unit's, and twins differing only in buffs sort by them, whatever the list order."""
    b, eng, parser = _bound(unit_buffs=True)
    s, towers = _state(eng)
    t = eng.arena().subtile
    near = _unit(2, BLUE, 5 * t, 6 * t, buffs=(("Poison", 700),))
    far = _unit(4, BLUE, 5 * t, 14 * t, buffs=(("Rage", 300),))
    out = _build(b, parser, msgspec.structs.replace(s, entities=[*towers, far, near]))
    rows = [(round(float(r[8]) * eng.arena().height), ids)
            for (r, ids) in zip(out["entities"], out["entity_buffs"].tolist(), strict=True)
            if r[0] == 1 and r[3] == 1]
    assert rows == [(6 * t, [ID["Poison"], 0, 0, 0, 0, 0]), (14 * t, [ID["Rage"], 0, 0, 0, 0, 0])]
    one = _unit(6, BLUE, 5 * t, 10 * t, buffs=(("Rage", 300),))
    two = _unit(8, BLUE, 5 * t, 10 * t, buffs=(("Poison", 300),))
    outs = [_build(b, parser, msgspec.structs.replace(s, entities=[*towers, *pair]))
            for pair in ((one, two), (two, one))]
    for key in ("entities", "entity_buffs"):
        assert np.array_equal(outs[0][key], outs[1][key]), key


# --- the engine's extra unit values -----------------------------------------------------------


def test_extras_read_the_spawner_dash_speed_and_jump():
    b, eng, parser = _bound(unit_extras=True)
    s, towers = _state(eng)
    t = eng.arena().subtile
    u = _unit(2, BLUE, 5 * t, 10 * t, spawn_ms=7500, spawn_wave_left=2, dash_state=1,
              dash_ticks=10, speed_pct=65, hit_speed_pct=135, jumping=1)
    v = _unit(4, BLUE, 9 * t, 10 * t, dash_state=2, speed_pct=0, hit_speed_pct=400,
              spawn_ms=60000, spawn_wave_left=20, dash_ticks=50)
    out = _build(b, parser, msgspec.structs.replace(s, entities=[*towers, u, v]))
    (a, _), (c, _) = _troops(b, out)
    assert a == pytest.approx(dict(spawn_clock=0.5, spawn_wave_left=0.25, dash_standing=1,
                                   dash_dashing=0, dash_ticks=0.5, speed=-0.35, hit_speed=0.35,
                                   jumping=1))
    assert c == pytest.approx(dict(spawn_clock=1, spawn_wave_left=1, dash_standing=0,
                                   dash_dashing=1, dash_ticks=1, speed=-1, hit_speed=1,
                                   jumping=0))


def test_extras_not_said_or_an_unknown_dash_state_are_refused():
    b, eng, parser = _bound(unit_extras=True)
    s, towers = _state(eng)
    t = eng.arena().subtile
    for bad, match in ((_unit(2, BLUE, 5 * t, 10 * t, **NOT_SAID), "uid 2 has none"),
                       (_unit(2, BLUE, 5 * t, 10 * t, dash_state=3), "dash_state 3")):
        with pytest.raises(ValueError, match=match):
            _build(b, parser, msgspec.structs.replace(s, entities=[*towers, bad]))


def test_the_extra_fields_trail_unit_type_and_read_as_not_said_when_absent():
    fields = list(EntityState.__struct_fields__)
    at = fields.index("unit_type") + 1
    assert tuple(fields[at:]) == ENTITY_EXTRA_FIELDS == (
        "spawn_ms", "spawn_wave_left", "dash_state", "dash_ticks", "speed_pct", "hit_speed_pct",
        "jumping",
    )
    eng = MockEngine()
    eng.reset(1, MatchSetup(decks=[list(range(8)), list(range(8))]))
    row = msgspec.to_builtins(eng.state().entities[0])
    old = msgspec.convert(row[:at], EntityState)
    assert all(getattr(old, f) == -1 for f in ENTITY_EXTRA_FIELDS)
    new = msgspec.convert([*row[:at], 1, 2, 3, 4, 5, 6, 7], EntityState)
    assert [getattr(new, f) for f in ENTITY_EXTRA_FIELDS] == [1, 2, 3, 4, 5, 6, 7]


def test_the_field_order_check_holds_the_extras_after_the_engines_row():
    fields = list(EntityState.__struct_fields__)
    base = fields[: fields.index("unit_type") + 1]
    good = types.SimpleNamespace(ENTITY_FIELDS=base, ENTITY_EXTRA_FIELDS=list(ENTITY_EXTRA_FIELDS))
    assert check_field_order(good)["ENTITY_EXTRA_FIELDS"] is True
    assert check_field_order(types.SimpleNamespace(ENTITY_FIELDS=base))["ENTITY_EXTRA_FIELDS"] is (
        False
    )
    swapped = list(ENTITY_EXTRA_FIELDS)
    swapped[4], swapped[5] = swapped[5], swapped[4]
    with pytest.raises(RuntimeError, match="hit_speed_pct"):
        check_field_order(types.SimpleNamespace(ENTITY_FIELDS=base, ENTITY_EXTRA_FIELDS=swapped))
    # An engine whose row grows a column: its extras would land one place late here.
    grown = types.SimpleNamespace(ENTITY_FIELDS=[*base, "something_later"],
                                  ENTITY_EXTRA_FIELDS=list(ENTITY_EXTRA_FIELDS))
    with pytest.raises(RuntimeError, match="something_later"):
        check_field_order(grown)


# --- the engine -------------------------------------------------------------------------------


def _has_053() -> bool:
    if not core_available():
        return False
    import royalesim

    return hasattr(royalesim, "ENTITY_EXTRA_FIELDS")


needs_053 = pytest.mark.skipif(
    not _has_053(),
    reason=str(CORE_IMPORT_ERROR) if not core_available() else
    "royalesim before 0.1.53 says no buff kinds and no extra unit values",
)
TABLE = "client160635007_20261103"
DECKS = [["IceWizard", "Freeze", "Rage", "Witch", "MegaKnight", "Zap", "HogRider", "Poison"],
         ["Tombstone", "GoblinHut", "Assassin", "Lightning", "IceGolemite", "Bats", "MegaKnight",
          "Graveyard"]]


def _engine(**kw):
    from royalegym.rust_engine import RustEngine

    return RustEngine(calibration_overrides={"cards.CARD_TABLE": TABLE}, **kw)


@needs_053
def test_the_engine_says_its_buff_kinds_and_its_extras_match_its_own_rows():
    import royalesim
    from royalegym import make_env

    eng = _engine()
    assert eng.field_order_checked["ENTITY_EXTRA_FIELDS"] is True
    assert list(royalesim.ENTITY_EXTRA_FIELDS) == list(ENTITY_EXTRA_FIELDS)
    kinds = eng.buff_kinds()
    assert kinds == json.loads(eng._battle.buff_kinds_json())
    assert any("|" in k for k in kinds), "vacuous: no group to split"
    b, _, _ = _bound(eng, unit_buffs=True)
    assert b.config()["buff_names"] == sorted({m for k in kinds for m in k.split("|")})
    env = make_env(deck=DECKS, engine=eng)
    env.reset(seed=2)
    plain = eng.state()
    assert all(getattr(e, f) == -1 for e in plain.entities for f in ENTITY_EXTRA_FIELDS)
    eng.set_entity_extras(True)
    raw = json.loads(eng._battle.state_json(extra=True))["entities"]
    got = [[getattr(e, f) for f in ENTITY_EXTRA_FIELDS] for e in eng.state().entities]
    assert got == [r[-len(ENTITY_EXTRA_FIELDS):] for r in raw]
    eng.set_entity_extras(False)
    assert eng.state() == plain


def _played(n=700, seed=7, **flags):
    from royalegym import make_env

    env = make_env(deck=DECKS, engine=_engine(),
                   obs_builder=EntityListObsBuilder(unit_buffs=True, unit_extras=True, **flags))
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    states, outs = [env.battle_state], [obs]
    for _ in range(n):
        if not env.agents:
            break
        acts = {}
        for agent in env.agents:
            legal = np.flatnonzero(obs[agent]["action_mask"])[1:]
            acts[agent] = int(rng.choice(legal)) if legal.size and rng.random() < 0.3 else 0
        obs, *_ = env.step(acts)
        states.append(env.battle_state)
        outs.append(obs)
    return env, states, outs


@needs_053
def test_a_played_battle_shows_buffs_and_extras_inside_their_bounds():
    env, _, outs = _played()
    b = env.obs_builder
    space = b.observation_space()
    seen = dict.fromkeys(
        ("buff", "spawn_clock", "dash_standing", "dash_dashing", "speed", "jumping"), 0
    )
    for obs in outs:
        for seat in obs.values():
            assert space["entities"].contains(seat["entities"])
            assert space["entity_buffs"].contains(seat["entity_buffs"])
            seen["buff"] += int(seat["entity_buffs"].any())
            for cols, _ in _troops(b, seat):
                for k in ("spawn_clock", "dash_standing", "dash_dashing", "jumping"):
                    seen[k] += int(cols[k] > 0)
                seen["speed"] += int(cols["speed"] != 0)
    assert all(v > 0 for v in seen.values()), f"vacuous: {seen}"


@needs_053
def test_red_on_the_mirror_sees_the_buffs_and_extras_blue_sees():
    env, states, _ = _played(n=400)
    b, parser = env.obs_builder, env.action_parser
    arena = env.engine.arena()
    bad, buffed = [], 0
    for s in states:
        m = mirror_state(arena, s)
        ob = b.build(s, BLUE, parser.action_mask(s, BLUE))
        orr = b.build(m, RED, parser.action_mask(m, RED))
        for key in ("entities", "entity_buffs"):
            if not np.array_equal(ob[key], orr[key]):
                bad.append((s.tick, key))
        buffed += int(ob["entity_buffs"].any())
    assert bad == [], bad[:5]
    assert buffed > 0, "vacuous: no buff in the battle"


@needs_053
def test_with_every_flag_on_the_buff_and_extra_columns_are_what_they_are_alone():
    """unit_actions used to be the last block, so nothing read the offsets after it: with every
    flag on, the buff and extra columns must hold what a builder with only them writes. Rows
    are compared as a multiset, since the sort key takes every appended value."""
    flags = dict(unit_motion=True, unit_stats=True, unit_status=True, unit_actions=True)
    env, states, _ = _played(n=300, **flags)
    full, parser = env.obs_builder, env.action_parser
    alone, _, _ = _bound(env.engine, unit_buffs=True, unit_extras=True)
    names = [n for flag in ("unit_buffs", "unit_extras") for n, _, _ in ENTITY_EXTRA_COLUMNS[flag]]

    def rows(b, s, seat):
        out = b.build(s, seat, parser.action_mask(s, seat))
        return sorted(
            (tuple(round(cols[n], 6) for n in names), tuple(ids))
            for cols, ids in _troops(b, out)
        )

    nonzero = 0
    for s in states:
        for seat in (BLUE, RED):
            got, want = rows(full, s, seat), rows(alone, s, seat)
            assert got == want, (s.tick, seat)
            nonzero += sum(any(v != 0 for v in c) or any(i) for c, i in want)
    assert nonzero > 100, f"vacuous: {nonzero} troop rows with a buff or an extra value"
