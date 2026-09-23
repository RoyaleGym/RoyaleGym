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
    engine.reset(seed=0, setup=MatchSetup(decks=[[idx] * 8] * 2, elixir_milli=[10000] * 2))
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
