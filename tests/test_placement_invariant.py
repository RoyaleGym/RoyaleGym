"""The invariant the building action space rests on: a box that fits can be tapped for.

WHY THIS IS A TEST AND NOT A SENTENCE
    The ``taps_where_the_building_stays`` arm is only worth having because it is
    LOSSLESS: every tile a building can end up on is reachable by a tap that leaves it
    there. That is not luck, it follows from one relationship between two of the
    engine's rules:

        BOX legality implies TAP legality.

    The argument. A landing is a position where the box fits. The tile that snaps to
    that landing lies INSIDE the box: the centre tile for an odd building, and for an
    even one the tapped tile is one of the four the box covers, since a 2x2 centred on a
    tile's corner spans that tile and its neighbours. Box legality requires every
    half-cell under the box to be in the placer's own half and off water and no-deploy.
    The tapped tile's own cells are under the box, so they pass the tap rule too. So
    tapping that tile is offered AND the building stays there.

    The argument holds only while the engine's tap rule asks nothing the box rule does
    not already guarantee. If a tap-point rule is added that box legality does not imply
    -- "you may not tap a tile a troop is standing on", say -- the arm becomes lossy and
    nothing else in the suite would notice: the per-board losslessness checks would start
    failing on some boards and not others, which reads like a flaky test rather than a
    rule change.

    So the invariant is checked directly, and this file is the thing to read when it
    goes red. It is not a proof either. It is the guard that turns a silent change in
    someone else's rule into a named failure here.

WHERE THE TWO RULES COULD PART, from the engine side
    The tap rule's cell test is CLOSED, so a point on a cell boundary touches every cell
    it borders, while the box test is half-open and a box flush against a cell's edge
    does not cover it. That mismatch is what would break the implication. It does not
    today, because the box reaches at least a full tile beyond the tapped tile on every
    side, so every point of that tile, boundary included, touches only cells the box
    already covers. A tile CENTRE is itself such a boundary point: it touches all four
    of the tile's half-cells, which is why the mask's rule is "the whole 2x2 block".

    These tests tap tile centres, and only centres, because that is the whole action
    space. A tap on a tile CORNER touches four cells belonging to four different tiles,
    two of them outside the box, so it can fail without the theorem failing; checking it
    would raise alarms about points no policy can choose. The even-sided case carries
    the interesting geometry anyway: a 2x2 building LANDS on a corner, so Tesla exercises
    a landing whose position sits on a four-cell boundary.

    The named, live way this breaks: ``box_zone`` applies no enemy-rect test, and the tap
    rule applies one only under the ``enemy_tower_no_deploy_rects`` territory model.
    Buildings use own-half today, so no rect is consulted for a building at all. The
    ledger records the rect model as the RIVAL arm for ``placement.BUILDING_TERRITORY``
    and says the corpus cannot tell them apart. If it is ever flipped, the tap rule gains
    a condition box legality does not imply, and the arm becomes lossy. The engine's
    ledger entry says that flip must add the rect test to the box side in the same
    change; that is a note, and a note is not a mechanism, which is why this asserts the
    invariant independently.
"""

from __future__ import annotations

import pytest

from royalegym.protocol import (
    BLUE,
    RED,
    MatchSetup,
    Placement,
    SpawnSpec,
    to_engine,
    to_own,
)
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available

pytestmark = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))

BUILDINGS = ("Cannon", "Tesla")
SEAT = {BLUE: "Blue", RED: "Red"}


@pytest.fixture(scope="module")
def engine() -> RustEngine:
    return RustEngine()


def occupancy(engine: RustEngine, card: str, lattice: int) -> MatchSetup:
    """A board whose own halves carry a lattice of buildings, both teams alike."""
    ids = {c.name: c.card_id for c in engine.cards()}
    a = engine.arena()
    t = a.subtile
    spawns = []
    for team in (BLUE, RED) if lattice else ():  # lattice 0 is the empty board
        for tx in range(1, 17, lattice):
            for ty in range(1, 14, lattice):
                x, y = to_engine(a, team, tx * t + t // 2, ty * t + t // 2)
                spawns.append(SpawnSpec(team=team, card_id=ids["Cannon"], x=x, y=y))
    deck = [ids[card]] * 8
    return MatchSetup(decks=[deck, deck], elixir_milli=[10000, 10000], spawns=spawns)


@pytest.mark.parametrize("lattice", [0, 6, 5, 4, 3])
@pytest.mark.parametrize("card", BUILDINGS)
def test_a_tile_a_building_can_land_on_is_a_tile_that_can_be_tapped_for(
    engine, card, lattice
) -> None:
    """The invariant, asked of the engine on boards of rising occupancy.

    For every tap the engine accepts, take where the building LANDED and tap that tile:
    the engine must accept that too, and leave the building there. If it refuses, box
    legality no longer implies tap legality and the arm is lossy on this board.
    """
    engine.reset(11, occupancy(engine, card, lattice))
    a = engine.arena()
    t = a.subtile
    broken: list[str] = []
    for team in (BLUE, RED):
        for ty in range(a.tiles_y):
            for tx in range(a.tiles_x):
                x, y = to_engine(a, team, tx * t + t // 2, ty * t + t // 2)
                landed = engine.building_placement(team, card, x, y)
                if landed is None:
                    continue
                ox, oy = to_own(a, team, landed[0], landed[1])
                lt = (ox // t, oy // t)
                lx, ly = to_engine(a, team, lt[0] * t + t // 2, lt[1] * t + t // 2)
                again = engine.building_placement(team, card, lx, ly)
                if again is None:
                    broken.append(
                        f"{SEAT[team]} tap {(tx, ty)} lands on {lt}, and a tap on {lt} "
                        "is REFUSED: a box fits there but the tile cannot be tapped for"
                    )
                elif (again[0], again[1]) != (landed[0], landed[1]):
                    broken.append(
                        f"{SEAT[team]} tap {(tx, ty)} lands on {lt}, and a tap on {lt} "
                        f"lands somewhere else again: relocation is not idempotent"
                    )
    assert not broken, (
        f"{card}, lattice {lattice}: the invariant the building arm rests on no longer "
        "holds. Box legality has stopped implying tap legality, so a landing exists "
        f"that no honest tap reaches.\n  " + "\n  ".join(broken[:8])
    )


@pytest.mark.parametrize("card", BUILDINGS)
def test_the_boards_this_checks_are_not_all_the_same_board(engine, card) -> None:
    """Vacuity guard. If every lattice gave the same board, or gave a board with no
    legal tap at all, the invariant above would pass while checking almost nothing."""
    counts = []
    a = engine.arena()
    t = a.subtile
    for lattice in (0, 6, 5, 4, 3):
        engine.reset(11, occupancy(engine, card, lattice))
        n = 0
        for ty in range(a.tiles_y):
            for tx in range(a.tiles_x):
                x, y = to_engine(a, BLUE, tx * t + t // 2, ty * t + t // 2)
                n += engine.building_placement(BLUE, card, x, y) is not None
        counts.append(n)
    assert len(set(counts)) > 1, f"{card}: every lattice gave the same {counts[0]} taps"
    assert max(counts) > 0, f"{card}: no lattice offered a single legal tap"


def test_both_cards_are_buildings_and_differ_in_parity(engine) -> None:
    """The two parities snap differently, so a check that used one would miss the other:
    an odd building goes to the tapped tile's centre, an even one to a corner of it."""
    cards = {c.name: c for c in engine.cards()}
    sizes = set()
    for name in BUILDINGS:
        assert cards[name].placement == Placement.BUILDING, f"{name} is not a building"
        tiles = cards[name].footprint_tiles
        assert tiles is not None, f"the catalogue states no footprint for {name}"
        sizes.add(tiles % 2)
    assert sizes == {0, 1}, f"both cards have the same parity of side: {sizes}"
