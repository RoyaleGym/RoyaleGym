"""Where the units of one deploy stand relative to the tap.

WHY THIS EXISTS
    The README shows a program and the battle it prints. That printed result is a pinned
    outcome, so anything that moves a unit moves the page. When sim announced a change to
    spawner emission, the question was whether the README's eight-card deck was exposed.
    It has no spawner and no death spawn, so the reasoned answer was "probably safe".

    The reasoned answer was right by luck and its model was wrong. The suspicion fell on
    Musketeer, the only card in the deck carrying ``spawn_radius_milli``, on the assumption
    that the field offsets a unit from the tap. Measured, it does not: a Musketeer lands
    exactly at the tap, displacement 0, indistinguishable from a Knight, which has no such
    field. The exposure is the two MULTI-unit cards, whose members are arranged around the
    tap -- Minions on a ring, Archer as a pair -- which is the arrangement the change is
    actually about.

    Docs put the lesson better than I can: a correct decision taken on a wrong model is not
    a validated model, it just did not get tested that time. This is the test.

WHY PIN NUMBERS THAT ARE MEANT TO CHANGE
    So that when they change, the failure NAMES what moved. Without it the first symptom
    is the README printing a different winner, which says only that something moved
    somewhere in a whole battle. A red here says "Minions went from a ring of 0.58 tile
    to this", which is a diagnosis.

    When it goes red because the engine legitimately changed, RE-TAKE these numbers. Do not
    widen the comparison: a tolerance would hide exactly the half-tile moves this is for.
"""

from __future__ import annotations

import pytest

from royalegym.landing import landings
from royalegym.protocol import DeployCommand, DeployStatus, MatchSetup, derived_cards_vintage
from royalegym.rust_engine import core_available


def battle_setup(engine, idx: int) -> MatchSetup:
    """One card in every slot, starting PAST the opening deploy lockout.

    A match refuses every command for its first `deploy_lockout_ticks` ticks, so a battle
    beginning at 0 answers TOO_EARLY to every tap here and these tests would grade a timing
    rule rather than the one they are about. Read from the engine: 0 is a real calibration
    arm, so a literal 90 would be wrong on a build without a lockout.
    """
    return MatchSetup(
        decks=[[idx] * 8] * 2,
        elixir_milli=[10000] * 2,
        start_tick=engine.rules().deploy_lockout_ticks,
    )

pytestmark = pytest.mark.skipif(
    not core_available(), reason="the compiled engine is not built; a skip here is not a pass"
)

TILE = 18000

#: These offsets are a fact about ONE card table. A clone reads the 2018 build and would
#: measure different cards, so comparing there would grade the data rather than the engine.
CARD_TABLE = "15.535.29"

#: Offsets from the tap, in subtiles, sorted. Measured on build_digest f7628dd51148e4ce,
#: engine binary 36cefd6569fe232d, cards_json 5a1dac3d2fb1b4a9.
PINNED = {
    "Knight": [(0, 0)],
    "Giant": [(0, 0)],
    "Cannon": [(0, 0)],
    # Carries spawn_radius_milli 800 and still lands ON the tap. This row is the one that
    # corrected the wrong model, so it is worth a test of its own rather than a comment.
    "Musketeer": [(0, 0)],
    "Archer": [(-9018, 0), (9000, 0)],
    "Minions": [(-8982, -5184), (0, 10422), (9036, -5184)],
}


def require_this_card_table() -> None:
    vintage = derived_cards_vintage()
    if CARD_TABLE not in vintage:
        pytest.skip(
            f"SKIPPED, NOT PASSED: these offsets were measured against the {CARD_TABLE!r} "
            f"card table and this engine reads {vintage!r}. Comparing across tables would "
            "grade the DATA and not the placement rule. Re-measure on this table rather "
            "than relaxing the comparison."
        )


@pytest.mark.parametrize("card", sorted(PINNED))
def test_a_deploys_units_stand_where_they_did(card: str) -> None:
    from royalegym.rust_engine import RustEngine

    require_this_card_table()
    engine = RustEngine()
    names = [c.name for c in engine.cards()]
    if card not in names:
        pytest.skip(f"this catalogue has no {card!r}; a skip here is not a pass")
    idx = names.index(card)
    engine.reset(seed=0, setup=battle_setup(engine, idx))
    engine.step([], 10)
    x, y = 9 * TILE + TILE // 2, 8 * TILE + TILE // 2
    results, land = landings(engine, [DeployCommand(team=0, hand_slot=0, x=x, y=y)], 2)
    assert results[0].status == DeployStatus.OK, (
        f"the {card} tap was refused as {DeployStatus(results[0].status).name}, so this "
        "measured nothing. Open ground in own territory was chosen for this reason."
    )
    got = sorted((e.x - x, e.y - y) for e in land[0].entities)
    assert got == PINNED[card], (
        f"{card} now stands at {got} relative to the tap, and was pinned at {PINNED[card]}. "
        "If the engine changed on purpose, RE-TAKE these numbers and check the README's "
        "printed battle with docs, because that page shows a pinned outcome and this deck "
        "is what produces it. Do not add a tolerance: a tolerance hides the half-tile "
        "moves this exists to catch."
    )


def test_the_pinned_set_still_contains_a_card_that_places_a_group() -> None:
    """Four of these six place one unit at the tap, and a rule could break only the group.

    If the multi-unit rows were ever dropped, every remaining row would pass by placing a
    unit exactly where it was asked, and this file would look healthy while testing none
    of the arrangement it is named for.
    """
    groups = {card: offs for card, offs in PINNED.items() if len(offs) > 1}
    assert len(groups) >= 2, (
        f"only {sorted(groups)} place more than one unit. The arrangement of a GROUP around "
        "the tap is what this file is for; without one it checks only that a single unit "
        "lands where it was tapped."
    )
    for card, offs in groups.items():
        assert any(o != (0, 0) for o in offs), f"{card}'s units are all pinned at the tap"


def offsets_at(engine, idx: int, tx: int, ty: int = 8) -> tuple[tuple[int, int], ...] | None:
    """Sorted per-unit offsets from a tap at that tile centre, at the DEPLOY tick."""
    from royalegym.landing import landings

    engine.reset(seed=0, setup=battle_setup(engine, idx))
    engine.step([], 10)
    x, y = tx * TILE + TILE // 2, ty * TILE + TILE // 2
    results, land = landings(engine, [DeployCommand(team=0, hand_slot=0, x=x, y=y)], 1)
    if results[0].status != DeployStatus.OK or not land[0].entities:
        return None
    return tuple(sorted((e.x - x, e.y - y) for e in land[0].entities))


def mirrored(offs: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((-ox, oy) for ox, oy in offs))


def a_card_with_count(engine, count: int) -> int:
    idx = next((i for i, c in enumerate(engine.cards()) if c.count == count), None)
    if idx is None:
        pytest.skip(f"this catalogue has no count-{count} card; a skip here is not a pass")
    return idx


def test_a_three_unit_formation_mirrors_about_the_arena_centre() -> None:
    """The triple is internally asymmetric by 54 subtiles and the asymmetry MIRRORS.

    This replaces a symmetry check that would have been wrong. The offsets are not
    symmetric: they sum to 54 in x rather than 0. What IS exact is that a tap on the left
    half gets the mirror image of a tap on the right half, and the boundary is the arena
    centre -- columns 1..8 share one triple and 9..16 share its mirror.

    Worth protecting because 0.003 of a tile is exactly the size of residual somebody
    rounds away as noise, and this project has a standing rule (D12) that a measured
    client asymmetry is reproduced rather than tidied. Whether this 54 comes from the
    client or from integer rounding inside the formation is not decidable from out here;
    what is decidable is that it is deterministic and structured, so it is not noise.

    THE MIRROR CHECK ALONE CANNOT CATCH THE TIDYING, which is why each pair also asserts
    HANDEDNESS. If somebody made the triple exactly symmetric, then left == right and
    mirroring either leaves it unchanged, so `left == mirrored(right)` would hold and this
    would pass while the thing it exists to protect had been removed. Verified rather than
    reasoned: the symmetric triple (-9000, 0, +9000) satisfies the mirror relation exactly.
    That is the same shape as three other assertions in this repo tonight -- the sentence
    above the assertion asking a different question from the assertion.
    """
    from royalegym.rust_engine import RustEngine

    require_this_card_table()
    engine = RustEngine()
    idx = a_card_with_count(engine, 3)
    pairs = 0
    for tx in range(9):
        left, right = offsets_at(engine, idx, tx), offsets_at(engine, idx, 17 - tx)
        if left is None or right is None:
            continue
        pairs += 1
        # Handedness first. Without it a symmetric formation satisfies the mirror
        # relation trivially and this test reports health after the asymmetry is gone.
        assert left != right, (
            f"columns {tx} and {17 - tx} give the same triple {left}, so the formation has "
            "no handedness at all. If it was made symmetric on purpose, this file's whole "
            "subject has gone and these tests should be re-taken rather than left passing."
        )
        assert left == mirrored(right), (
            f"a count-3 tap at column {tx} gives {left} and its mirror column {17 - tx} "
            f"gives {right}, whose mirror image is {mirrored(right)}. These were exact "
            "mirrors when measured, including at the walls. If the formation was made "
            "symmetric on purpose, re-take this; do not relax it to a tolerance, because "
            "the residual being protected is 54 subtiles."
        )
    assert pairs >= 8, f"only {pairs} mirror pairs were testable; this used to be 9"


def test_the_two_halves_are_not_simply_identical() -> None:
    """Non-vacuity. If both halves gave the same triple, mirroring would be trivially true
    for any symmetric formation and the test above would prove nothing."""
    from royalegym.rust_engine import RustEngine

    require_this_card_table()
    engine = RustEngine()
    idx = a_card_with_count(engine, 3)
    left, right = offsets_at(engine, idx, 4), offsets_at(engine, idx, 13)
    if left is None or right is None:
        pytest.skip("a count-3 tap was refused on one side; a skip here is not a pass")
    assert left != right, (
        "the two halves give identical offsets, so the mirror check above passes without "
        "the formation having any handedness at all and is not testing what it says"
    )


def test_the_walls_are_a_different_formation_and_still_mirror() -> None:
    """Columns 0 and 17 are NOT the interior triple: the units are pushed inward.

    Recorded because a mirror test written from interior samples alone would look correct
    and would never have visited the case where the formation changes shape. The walls
    still mirror each other, which is why the test above can span them.
    """
    from royalegym.rust_engine import RustEngine

    require_this_card_table()
    engine = RustEngine()
    idx = a_card_with_count(engine, 3)
    wall, interior = offsets_at(engine, idx, 0), offsets_at(engine, idx, 4)
    if wall is None or interior is None:
        pytest.skip("a count-3 tap was refused; a skip here is not a pass")
    assert wall != interior, (
        "the wall column now gives the same formation as the interior. When measured it "
        "did not: the units are shifted inward to keep them in the arena, and that is the "
        "case an interior-only sweep would never have visited."
    )


def test_a_two_unit_formation_does_not_mirror_and_carries_a_fixed_bias() -> None:
    """Count 2 behaves differently from count 3, which is worth pinning rather than assuming.

    Its pair is the SAME on both halves rather than mirrored, and carries a fixed bias of
    -18 subtiles in x. So "the formation mirrors" is a claim about count 3 and not a
    property of formations, and anybody generalising it would be wrong.
    """
    from royalegym.rust_engine import RustEngine

    require_this_card_table()
    engine = RustEngine()
    idx = a_card_with_count(engine, 2)
    left, right = offsets_at(engine, idx, 4), offsets_at(engine, idx, 13)
    if left is None or right is None:
        pytest.skip("a count-2 tap was refused; a skip here is not a pass")
    assert left == right, (
        f"a count-2 formation now differs between halves: {left} against {right}. When "
        "measured it was identical on both sides, which is what makes it NOT mirrored."
    )
    assert sum(ox for ox, _ in left) == -18, (
        f"the count-2 pair sums to {sum(ox for ox, _ in left)} in x and was measured at "
        "-18. That bias is the thing this pins; it is a thousandth of a tile and exactly "
        "the size of residual that gets tidied away."
    )
