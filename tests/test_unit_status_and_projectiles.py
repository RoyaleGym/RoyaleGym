"""What a unit is doing, and what is in the air -- carried from the engine to the viewer.

Asked for by the viser session on 2026-09-24: the viewer must show status effects and
projectiles at a glance. The engine tracks all of it; until now ``unit_dict`` hard-coded
target, direction and state to None and nothing carried buffs or projectiles at all. This
file checks the carrying half: the new EntityState columns, ProjectileState, the trace
that replays are built from, and the dicts the viewer reads.

WHY THE VALUES BELOW ARE ALL DIFFERENT. EntityState and ProjectileState are decoded BY
POSITION. ``target_uid`` and ``attack_phase`` are adjacent ints, so a transposition raises
nothing -- and a test using 0 and 0, or 1 and 1, would pass straight through one. Every
field below gets a value no neighbour shares, so a value landing in the wrong column fails.

WHAT A DEFAULT MEANS. A default is what an ABSENT column decodes to, i.e. "the engine did
not say". So ``attack_phase`` defaults to -1, not to sim's 0: sim's 0 means idle, and an
engine exporting nothing must not look like a board of idle units. The flip side is also
tested -- a real 0 must survive as 0 and not be swallowed as "unknown".
"""

from __future__ import annotations

import types

import msgspec
import pytest

from royalegym.protocol import BattleState, EntityState, PlayerState, ProjectileState
from royalegym.replay import TraceFrame, TraceHeader
from royalegym.rust_engine import (
    CORE_IMPORT_ERROR,
    POSITIONAL_LAYOUTS,
    RustEngine,
    check_field_order,
    core_available,
)
from royalegym.viser import frame_dict, projectile_dict, unit_dict

NAMES = {7: "Knight", 12: "Archer"}


def name_of(card_id: int) -> str:
    return NAMES.get(card_id, f"#{card_id}")


#: The fifteen columns every engine has sent since footprints, with distinct values.
LEGACY = [101, 1, 2, 7, -1, 9000, 21000, 600, 1400, 500, False, 0, 3, 4, [1, 2, 3, 4]]
#: The five the engine gained on 2026-09-24, in sim's order, again all distinct.
NEW = [37, 2, [5, -9], 140, [["Rage|Slow", 1200], ["Poison", 800]]]


def decode_entity(row: list) -> EntityState:
    return msgspec.json.decode(msgspec.json.encode(row), type=EntityState)


def test_the_new_columns_land_in_the_fields_they_name():
    e = decode_entity(LEGACY + NEW)
    assert (e.target_uid, e.attack_phase, e.facing, e.shield) == (37, 2, (5, -9), 140), (
        f"a positional column landed in the wrong field: target_uid={e.target_uid} "
        f"attack_phase={e.attack_phase} facing={e.facing} shield={e.shield}"
    )
    assert e.buffs == (("Rage|Slow", 1200), ("Poison", 800))
    # The legacy columns must not move either: the new ones are TRAILING.
    assert (e.uid, e.card_id, e.stun_ticks, e.knockback_ticks) == (101, 7, 3, 4)


def test_an_engine_that_sends_only_the_old_columns_decodes_as_not_reported():
    """The engine installed today, and every trace recorded before 2026-09-24."""
    e = decode_entity(LEGACY)
    u = unit_dict(e, name_of)
    assert u["target"] is None
    assert u["direction"] is None, "a zero facing must not reach the viewer as [0, 0]"
    assert u["state"] is None, "an engine that said nothing must not read as every unit idle"
    assert u["status"] == []
    assert u["extra"]["shield"] == 0


def test_the_viewer_gets_each_value_where_it_reads_it():
    u = unit_dict(decode_entity(LEGACY + NEW), name_of)
    assert u["target"] == 37
    assert u["direction"] == [5, -9], "direction is passed as sim gives it; the viewer normalises"
    assert u["state"] == 2
    assert u["status"] == [["Rage|Slow", 1200], ["Poison", 800]], (
        "buff names must arrive WHOLE -- the viewer splits sim's '|' names itself"
    )
    assert u["extra"]["shield"] == 140
    assert u["extra"]["tower_slot"] == -1
    assert u["extra"]["knockback_ticks"] == 4


def test_a_real_idle_phase_and_a_uid_of_zero_survive_as_values():
    """The -1 sentinel must not swallow real zeroes: idle is a state, uid 0 is a unit."""
    e = decode_entity([*LEGACY, 0, 0, [0, 1], 0, []])
    u = unit_dict(e, name_of)
    assert u["state"] == 0, "attack_phase 0 is IDLE, a reported value, not 'unknown'"
    assert u["target"] == 0, "uid 0 is a real entity and must not read as 'no target'"
    assert u["direction"] == [0, 1]


PROJECTILE = [1, 4000, 5000, 4100, 5200, 55, 900, 12]


def test_a_projectile_decodes_by_position_and_names_its_firer():
    p = msgspec.json.decode(msgspec.json.encode(PROJECTILE), type=ProjectileState)
    assert (p.team, p.x, p.y, p.aim_x, p.aim_y, p.target_uid, p.splash, p.firer_card_id) == (
        1, 4000, 5000, 4100, 5200, 55, 900, 12
    )
    d = projectile_dict(p, name_of)
    assert d == {
        "team": 1, "x": 4000, "y": 5000, "aim_x": 4100, "aim_y": 5200,
        "target": 55, "splash": 900, "name": "Archer",
    }


def test_a_tower_shot_is_labelled_and_a_lost_target_is_none():
    p = ProjectileState(0, 1, 2, 3, 4, target_uid=-1, splash=0, firer_card_id=-1)
    d = projectile_dict(p, name_of)
    assert d["name"] == "tower", "a crown tower's shot must not print as an unknown card id"
    assert d["target"] is None



def test_an_unknown_firer_is_not_labelled_as_a_tower():
    """-2 is NOT RECORDED (a projectile from an older snapshot), and it must stay unknown.

    Sim keeps it apart from -1 so that "unknown" never reads as "a tower fired this". The
    first version of projectile_dict mapped every negative id to "tower", which undid that.
    """
    p = ProjectileState(0, 1, 2, 3, 4, target_uid=9, splash=0, firer_card_id=-2)
    assert projectile_dict(p, name_of)["name"] is None

def _state(projectiles: list[ProjectileState]) -> BattleState:
    towers = [4000, 2000, 2000]
    player = PlayerState(0, 5000, [7, 12, 7, 12], 7, 0, towers, towers, False)
    return BattleState(
        tick=10, tick_ms=50, regular_ticks=3600, overtime_ticks=2400, elixir_rate=1,
        overtime=False, players=[player, player], entities=[decode_entity(LEGACY + NEW)],
        game_over=False, winner=-1, projectiles=projectiles,
    )


def test_a_live_frame_carries_the_projectiles():
    p = ProjectileState(1, 4000, 5000, 4100, 5200, 55, 900, 12)
    frame = frame_dict(_state([p]), name_of, units_per_tile=18000)
    assert frame["projectiles"] == [projectile_dict(p, name_of)]
    assert frame["units"][0]["state"] == 2


def test_a_battle_state_without_projectiles_still_decodes():
    """BattleState is keyed, so an engine that sends no `projectiles` key gives []."""
    raw = msgspec.json.encode(_state([]))
    legacy = msgspec.json.decode(raw)
    del legacy["projectiles"]
    s = msgspec.json.decode(msgspec.json.encode(legacy), type=BattleState)
    assert s.projectiles == []


def test_a_trace_frame_keeps_its_projectiles_and_an_old_one_still_loads():
    """Replays are built from traces, so a projectile absent here never reaches the screen."""
    p = ProjectileState(1, 4000, 5000, 4100, 5200, 55, 900, 12)
    frame = TraceFrame(10, [decode_entity(LEGACY + NEW)], [5000, 5000], [0, 0],
                       [[7, 12], [7, 12]], "00", [], [7, 12], [p])
    back = msgspec.msgpack.decode(msgspec.msgpack.encode(frame), type=TraceFrame)
    assert back.projectiles == [p]
    assert back.entities[0].buffs == (("Rage|Slow", 1200), ("Poison", 800))

    before = [30, [LEGACY], [5000, 5000], [0, 0], [[7, 12], [7, 12]], "abc", [], [7, 12]]
    old = msgspec.msgpack.decode(msgspec.msgpack.encode(before), type=TraceFrame)
    assert old.projectiles == [], "a trace recorded before projectiles must load with none"
    assert old.entities[0].attack_phase == -1


def test_the_recorder_writes_the_projectiles_into_the_frame():
    """Through the RECORDER, which is the path a replay actually takes.

    The test above builds a TraceFrame by hand, so it would pass with the recorder dropping
    projectiles on the floor. This one records from an engine whose state has a projectile
    and reads it back out of the trace the recorder wrote.
    """
    from royalegym.replay import _frame

    p = ProjectileState(1, 4000, 5000, 4100, 5200, 55, 900, 12)
    engine = types.SimpleNamespace(state=lambda: _state([p]), state_hash=lambda: 0xABC)
    frame = _frame(engine)
    assert frame.projectiles == [p], "the recorder left the projectiles out of the trace"
    assert frame.entities[0].attack_phase == 2


def test_the_trace_header_names_the_projectile_columns():
    assert "projectile_fields" in TraceHeader.__struct_fields__
    assert list(ProjectileState.__struct_fields__) == [
        "team", "x", "y", "aim_x", "aim_y", "target_uid", "splash", "firer_card_id"
    ]


# ---------------------------------------------------------------------------
# The positional contract: refuse a mismatch instead of trusting two people typed alike.


def core_with(**exports) -> types.SimpleNamespace:
    return types.SimpleNamespace(**exports)


ENTITY = list(EntityState.__struct_fields__)


def test_an_engine_that_exports_no_field_list_passes_unchecked_and_says_so():
    assert check_field_order(core_with()) == {name: False for name, _ in POSITIONAL_LAYOUTS}


def test_the_same_order_is_checked_and_accepted():
    assert check_field_order(core_with(ENTITY_FIELDS=ENTITY))["ENTITY_FIELDS"] is True


def test_an_older_engine_with_fewer_columns_is_accepted():
    """A newer royalegym reading an older engine: the missing columns decode as defaults."""
    assert check_field_order(core_with(ENTITY_FIELDS=ENTITY[:15]))["ENTITY_FIELDS"] is True


def test_two_swapped_columns_are_refused_and_named():
    """The exact silent failure: target_uid and attack_phase are both ints."""
    swapped = list(ENTITY)
    i, j = swapped.index("target_uid"), swapped.index("attack_phase")
    swapped[i], swapped[j] = swapped[j], swapped[i]
    with pytest.raises(RuntimeError, match=rf"column {i} is 'attack_phase' in the engine"):
        check_field_order(core_with(ENTITY_FIELDS=swapped))


def test_an_engine_newer_than_this_package_is_refused():
    with pytest.raises(RuntimeError, match="engine is newer than this package"):
        check_field_order(core_with(ENTITY_FIELDS=[*ENTITY, "halo"]))


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_the_installed_engine_had_its_layout_checked():
    """Loud when the engine exports no field list: the layout is then trusted, not verified."""
    checked = RustEngine().field_order_checked
    if not checked["ENTITY_FIELDS"]:
        pytest.skip(
            "SKIPPED, NOT PASSED: the installed engine exports no ENTITY_FIELDS, so the "
            "positional order of EntityState is TRUSTED rather than checked. Requested from "
            "sim on 2026-09-24 beside the status and projectile export."
        )
    assert checked["ENTITY_FIELDS"] is True
