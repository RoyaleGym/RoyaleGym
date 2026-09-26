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
    # Re-taken 2026-09-25 on build_digest 734032113c44fbcc (RoyaleSim 7bfcfd2), where
    # formation.STAGGER_WAIT took its measured arm. It was (9036, -5184) for the third
    # member before: see test_the_three_unit_formation_is_the_measured_symmetric_ring.
    "Minions": [(-8982, -5184), (0, 10422), (8982, -5184)],
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


Offsets = tuple[tuple[int, int], ...]


def ring_problems(offs: Offsets, expected: Offsets) -> list[str]:
    """What is wrong with a count-3 triple, against the measured symmetric ring."""
    problems = []
    if offs != expected:
        problems.append(f"it is {offs}, not the measured {expected}")
    if offs != mirrored(offs):
        problems.append(f"it is not symmetric about the tap: its mirror is {mirrored(offs)}")
    return problems


def test_the_three_unit_formation_is_the_measured_symmetric_ring() -> None:
    """Minions land exactly symmetric about the tap, the same on every interior column.

    THIS TEST ASSERTED THE OPPOSITE UNTIL 2026-09-25, and why it changed is the point.
    The triple used to lean by 54 subtiles (+9036 on one side against -8982), and the lean
    mirrored between arena halves. It was protected under D12 -- a measured client
    asymmetry is reproduced, not tidied -- with the caveat that nobody could tell from out
    here whether it came from the client or from the engine.

    Sim decided it. The lean came from the ENGINE: its first-tick separation scan pushed a
    member that was still waiting out its stagger by 3 native units (54 subtiles), and the
    push order made the push mirror by half. Under formation.STAGGER_WAIT's measured arm a
    waiting member does not move (1083 of 1083 corpus frame pairs still), and the lean is
    gone. The client never showed it: in the 16.402 corpus, Minions deployed on the LEFT
    half with a recorded tap sit at exactly +-499 native (+-8982 subtiles) on their first
    frame in 15 of 16 groups, where the old engine put one at 502. There is no right-half
    group with a recorded tap, so the right half here is the measured law applied, not a
    measurement. D12 protects measured client asymmetries, and this one never was.

    What this guards now is the lean coming back: every interior column must give the
    pinned triple exactly, and the triple must be its own mirror.
    """
    from royalegym.rust_engine import RustEngine

    require_this_card_table()
    engine = RustEngine()
    names = [c.name for c in engine.cards()]
    if "Minions" not in names:
        pytest.skip("this catalogue has no Minions; a skip here is not a pass")
    idx = names.index("Minions")
    expected = tuple(sorted(PINNED["Minions"]))
    columns = 0
    for tx in range(1, 17):
        offs = offsets_at(engine, idx, tx)
        if offs is None:
            continue
        columns += 1
        problems = ring_problems(offs, expected)
        assert not problems, (
            f"a Minions tap at column {tx}: {'; '.join(problems)}. If a lean like the old "
            "54 subtiles is back, formation.STAGGER_WAIT or the first-tick separation scan "
            "moved; do not relax this to a tolerance."
        )
    assert columns >= 14, f"only {columns} interior columns were testable; this used to be 16"


def test_plant_the_old_leaning_triple_is_refused() -> None:
    """The check above, fed the triple the engine produced before 2026-09-25."""
    expected = tuple(sorted(PINNED["Minions"]))
    old = ((-8982, -5184), (0, 10422), (9036, -5184))
    problems = ring_problems(old, expected)
    assert len(problems) == 2, f"PLANT DID NOT LAND: the old lean gave {problems}"


def test_the_walls_are_a_different_formation_and_still_mirror() -> None:
    """Columns 0 and 17 are NOT the interior triple: the units are pushed inward.

    Recorded because a mirror test written from interior samples alone would look correct
    and would never have visited the case where the formation changes shape. The ring
    test above therefore covers columns 1 to 16 only.
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
