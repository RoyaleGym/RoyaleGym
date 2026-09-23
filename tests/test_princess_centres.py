"""Crown tower centres: the Arena the mask draws with against the engine's own towers.

WHY IT EXISTS
    The static map has no princess towers, so their centres are the one piece of tower
    geometry that arena.json does not carry yet. Until it does, ``Arena.load`` places
    them with a fallback constant. When the file carries ``princess_tower_centres``,
    ``Arena.load`` reads them instead. Either way the mask, MockEngine and the Rust
    engine must put every tower in the same place.

WHAT IT CHECKS
    a. The Arena's king and princess centres equal ``royalesim.Battle.tower_positions()``
       for both teams, with each team's own-left tower on the left of that team's own
       frame. Blue and Red towers sit at different y, so a swap between seats fails.
       The engine's state after a reset puts every crown tower on those centres too.
    b. ``Arena.load`` reads ``princess_tower_centres`` from arena.json when present
       (either order within a team), and refuses a malformed field.

PLANTS (each on a green baseline)
    the fallback constant at 12 half-cells instead of 13; Arena.load ignoring the
    field; Red's fallback towers not rotated; Red named by engine x; the engine state
    moving one Red tower by a subtile.

HOW THESE PLANTS DIFFER FROM THE ONES ELSEWHERE IN THIS SUITE, since both conventions are
in use here and the difference matters. Ten other files carry plants as TESTS -- fifty-odd
of them, `test_plant_*` -- which re-verify on every run that a check still bites. The list
above is a DATED RECORD: each plant was run once, on a green baseline, and each maps to a
test below that would catch it. So the tests are the standing enforcement and the list is
the evidence that they were once shown to bite, not a property anything re-checks. If one
of these tests loses its sensitivity, nothing here will say so.
"""

from __future__ import annotations

import json

import pytest

from royalegym import protocol
from royalegym.protocol import (
    BLUE,
    PRINCESS_CENTRES_KEY,
    RED,
    TEAMS,
    Arena,
    EntityKind,
    MatchSetup,
    data_dir,
    default_calibration,
    to_own,
)
from royalegym.rust_engine import CORE_IMPORT_ERROR, core_available

needs_core = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))

SHARED = ("Knight", "Minions", "Cannon", "Giant", "Archer", "Goblins", "MiniPekka", "HogRider")


@needs_core
def test_arena_tower_centres_are_the_engines_tower_positions():
    import royalesim

    arena = Arena.load(default_calibration())
    engine = royalesim.Battle.tower_positions()  # [team][king, low-x princess, high-x]
    assert engine[BLUE][0][1] != engine[RED][0][1], "the seats' towers are distinct"
    for team in TEAMS:
        assert tuple(arena.king_centers[team]) == tuple(engine[team][0]), f"team {team} king"
        princesses = [tuple(p) for p in arena.princess_centers[team]]
        assert sorted(princesses) == sorted(tuple(p) for p in engine[team][1:]), (
            f"team {team} princess centres: arena {princesses}, engine {engine[team][1:]}"
        )
        # Named in the OWNER's frame: own-left has the smaller own-frame x.
        own_x = [to_own(arena, team, x, y)[0] for x, y in princesses]
        assert own_x[0] < own_x[1], f"team {team} own-left is not on the left: {own_x}"


@needs_core
def test_arena_tower_centres_are_where_the_engine_puts_its_towers():
    """The same, graded against the battle's own state after a reset: every crown
    tower entity stands at an Arena centre of its own team and kind."""
    from royalegym.rust_engine import RustEngine

    arena = Arena.load(default_calibration())
    engine = RustEngine(card_names=SHARED)
    engine.reset(3, MatchSetup(decks=[list(range(8)), list(range(8))]))
    for team in TEAMS:
        placed = sorted(
            (e.kind, e.x, e.y)
            for e in engine.state().entities
            if e.team == team and e.kind in (EntityKind.KING_TOWER, EntityKind.PRINCESS_TOWER)
        )
        want = sorted(
            [(EntityKind.KING_TOWER, *arena.king_centers[team])]
            + [(EntityKind.PRINCESS_TOWER, x, y) for x, y in arena.princess_centers[team]]
        )
        assert placed == want, f"team {team}: engine state {placed}, arena {want}"


def arena_with(tmp_path, centres) -> Arena:
    doc = json.loads((data_dir() / "derived" / "arena.json").read_text(encoding="utf-8"))
    doc[PRINCESS_CENTRES_KEY] = centres
    path = tmp_path / "arena.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return Arena.load(default_calibration(), path)


def test_arena_reads_princess_centres_from_arena_json(tmp_path):
    shipped = Arena.load(default_calibration())
    # Values the fallback cannot produce: a different depth per seat, and x off the
    # bridge centre line, so every tower is told apart from every other.
    blue = [[60000, 100000], [250000, 100000]]  # engine order, low x first
    red = [[70000, 470000], [260000, 470000]]
    got = arena_with(tmp_path, [blue, red])
    assert got.princess_center_status == f"arena.json {PRINCESS_CENTRES_KEY}"
    assert got.princess_centers[BLUE] == [(60000, 100000), (250000, 100000)]
    # Red's own-left is the one with the higher engine x (the 180-degree rotation).
    assert got.princess_centers[RED] == [(260000, 470000), (70000, 470000)]
    # The order inside a team is free.
    swapped = arena_with(tmp_path, [blue[::-1], red[::-1]])
    assert swapped.princess_centers == got.princess_centers
    assert got.king_centers == shipped.king_centers


@pytest.mark.parametrize(
    "bad",
    [
        [[[60000, 100000], [250000, 100000]]],  # one team
        [[[60000, 100000]], [[70000, 470000], [260000, 470000]]],  # one tower
        [[[60000.0, 100000], [250000, 100000]], [[70000, 470000], [260000, 470000]]],
        [[[True, 100000], [250000, 100000]], [[70000, 470000], [260000, 470000]]],
        [[[60000, 470000], [250000, 470000]], [[70000, 470000], [260000, 470000]]],  # half
        [[[60000, 100000], [60000, 120000]], [[70000, 470000], [260000, 470000]]],  # same x
    ],
)
def test_a_malformed_princess_field_is_refused(tmp_path, bad):
    with pytest.raises(ValueError, match=PRINCESS_CENTRES_KEY):
        arena_with(tmp_path, bad)


def test_the_fallback_is_the_constant_while_arena_json_has_no_centres():
    doc = json.loads((data_dir() / "derived" / "arena.json").read_text(encoding="utf-8"))
    if PRINCESS_CENTRES_KEY in doc:
        pytest.skip(f"arena.json carries {PRINCESS_CENTRES_KEY}, so the fallback is unused")
    arena = Arena.load(default_calibration())
    assert arena.princess_center_status.startswith("guess")
    y = protocol.MOCK_PRINCESS_CENTER_Y_HALF_CELLS * arena.half_size
    assert {p[1] for p in arena.princess_centers[BLUE]} == {y}
    assert {p[1] for p in arena.princess_centers[RED]} == {arena.height - y}
