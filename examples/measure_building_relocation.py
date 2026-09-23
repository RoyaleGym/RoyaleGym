"""How often a building tap does not give the agent the tile it asked for.

    python examples/measure_building_relocation.py

This is a MEASUREMENT, not a lesson, which is why it has no number: the numbered files
are a sequence to read in order and this one exists to re-derive a table that would
otherwise be a number somebody typed. It backs the figures in the ``BUILDING_TAP_ARMS``
comment in ``royalegym/action.py``.

WHY IT REPORTS THREE NUMBERS AND NOT ONE
    "The building moved" is three different questions and they give three different
    answers, which is how the original figure came to be quoted for the wrong one.

    * POINT -- the centre did not land on the point tapped. Worthless, and computed here
      only so nobody measures it again believing it means something: an EVEN footprint
      snaps to a tile CORNER, so its centre can NEVER sit on a tile centre and this
      criterion reports 100% for a 2x2 whatever the engine does. It is geometry.
    * TILE -- the centre's tile is not the tile tapped. This is what the arm
      ``taps_where_the_building_stays`` filters on, and what the original sweep counted.
    * COVERAGE -- the footprint does not stand on the tile that was tapped. This is the
      one that bears on credit assignment, because the action space chooses a TILE: a 3x3
      shifted by one tile is displaced, but it is still on the tile the agent asked for.

    COVERAGE is the lower bound on the harm and POINT is the upper bound. Quote a single
    number only with the criterion attached to it.

WHY IT VALIDATES ITS OWN INSTRUMENT FIRST
    The table comes from ``engine.building_placement``, a resolver that ANSWERS where a
    building would land rather than putting one there. That is ~100x faster than playing
    a battle per tap, and it is worth nothing unless it agrees with an actual deploy. So
    the script checks that first and REFUSES to print the table if it does not.

    The check exists because of how the first version of it failed: ``building_placement``
    returns ``(x, y, box)`` and it was compared against the deployed ``(x, y)``, so a
    3-tuple was tested against a 2-tuple and nothing ever matched. It reported that the
    resolver disagreed with deployment on 56 of 56 samples, which read as a serious engine
    finding and was a bug in the comparison. A validation step that can fail this way has
    to say what it compared, so it prints both counts rather than a verdict.

WHAT THIS SAMPLE IS, SO THE FIGURE IS NOT READ WIDER THAN IT WAS MEASURED
    Every tile of the board, tile centres, one seat, a near-empty board ten ticks in.
    Tile centres are the right population because ``GridActionParser`` has
    ``pitch_div = 1`` and taps tile centres.

    A COARSE SAMPLE OF THE SAME BOARD GIVES A DIFFERENT ANSWER, which is why this scans
    every tile. Sampling even x and odd y over the agent's own half gave 47.5% and 39.0%
    where the full board gives 51.7% and 30.4% -- and the narrowed gap was briefly read as
    evidence that board crowding drives relocation. It is not: the full scan of this
    near-empty board reproduces the busy-board sweep exactly. The lattice was the whole
    difference.
"""

from __future__ import annotations

from royalegym.landing import landings
from royalegym.protocol import DeployCommand, DeployStatus, MatchSetup, Placement
from royalegym.rust_engine import RustEngine, build_digest

TILE = 18000


def covers(cx: int, cy: int, footprint: int, tx: int, ty: int) -> bool:
    """Does a footprint that many tiles a side, centred at ``(cx, cy)``, stand on ``(tx, ty)``?"""
    half = footprint * TILE // 2
    return tx in range((cx - half) // TILE, (cx + half) // TILE) and ty in range(
        (cy - half) // TILE, (cy + half) // TILE
    )


def fresh(engine: RustEngine, card_id: int) -> None:
    """A battle holding one card in every slot, ten ticks in, before anything is placed."""
    engine.reset(
        seed=0, setup=MatchSetup(decks=[[card_id] * 8] * 2, elixir_milli=[10000, 10000])
    )
    engine.step([], 10)


def validate_resolver(engine: RustEngine, cards) -> bool:
    """The resolver must predict a real deploy, or the table below is about nothing."""
    agree = disagree = 0
    for name in ("Cannon", "Tesla"):
        card_id = next(i for i, c in enumerate(cards) if c.name == name)
        for tx in range(0, 18, 3):
            for ty in range(1, 15, 3):
                fresh(engine, card_id)
                x, y = tx * TILE + TILE // 2, ty * TILE + TILE // 2
                asked = engine.building_placement(0, name, x, y)
                results, landed = landings(
                    engine, [DeployCommand(team=0, hand_slot=0, x=x, y=y)], 1
                )
                if results[0].status != DeployStatus.OK or landed[0].x is None:
                    continue
                # asked is (x, y, box); compare the POSITION only, and say so.
                if asked[:2] == (landed[0].x, landed[0].y):
                    agree += 1
                else:
                    disagree += 1
    print(f"resolver vs real deploy, position only: {agree} agree, {disagree} disagree")
    if not agree:
        print("Nothing was compared, so this check passed vacuously and proves nothing.")
        return False
    return disagree == 0


def main() -> None:
    engine = RustEngine()
    cards = engine.cards()

    if not validate_resolver(engine, cards):
        raise SystemExit("the resolver does not predict deployment; no table is printed")

    print(f"\nbuild_digest={build_digest()}  full tile scan, near-empty board, one seat")
    print(f"{'card':18s} {'fp':>2s} {'fits':>5s} {'POINT':>7s} {'TILE':>7s} {'COVERAGE':>9s}")

    by_footprint: dict[int, set[tuple[float, float, float]]] = {}
    for card_id, card in enumerate(cards):
        if card.placement != Placement.BUILDING:
            continue
        fresh(engine, card_id)
        fits = point = tile = coverage = 0
        for tx in range(18):
            for ty in range(32):
                x, y = tx * TILE + TILE // 2, ty * TILE + TILE // 2
                landed = engine.building_placement(0, card.name, x, y)
                if landed is None:
                    continue  # nothing fits within the engine's search: refused
                fits += 1
                point += landed[:2] != (x, y)
                tile += (landed[0] // TILE, landed[1] // TILE) != (tx, ty)
                coverage += not covers(landed[0], landed[1], card.footprint_tiles, tx, ty)

        if not fits:
            print(f"{card.name:18s} {card.footprint_tiles:2d}     0   no tap accepted")
            continue
        rates = (
            round(100 * point / fits, 1),
            round(100 * tile / fits, 1),
            round(100 * coverage / fits, 1),
        )
        by_footprint.setdefault(card.footprint_tiles, set()).add(rates)
        print(
            f"{card.name:18s} {card.footprint_tiles:2d} {fits:5d} "
            f"{rates[0]:6.1f}% {rates[1]:6.1f}% {rates[2]:8.1f}%"
        )

    # THE ACTUAL FINDING, stated by the script rather than left for a reader to notice:
    # every card of a given footprint agrees to the decimal, so this is a property of the
    # footprint and not of the card. The 2x2 differing from the 3x3 is the control -- if
    # the card never reached the engine, every row would be identical and the agreement
    # below would mean nothing.
    print()
    for footprint, distinct in sorted(by_footprint.items()):
        seen = sum(
            1
            for c in cards
            if c.placement == Placement.BUILDING and c.footprint_tiles == footprint
        )
        verdict = "one rate" if len(distinct) == 1 else f"{len(distinct)} DIFFERENT rates"
        print(f"footprint {footprint}x{footprint}: {seen:2d} cards -> {verdict}")
    # THE VERDICT IS A SENTENCE THE SCRIPT EARNS, not a count a reader has to add up, and
    # it is deliberately not phrased with any number in it: the card table differs between
    # machines, so a line naming ten cards or 51.7% would be a claim about this laptop.
    # What survives a different table is the SHAPE -- every card of a footprint agreeing,
    # with a second footprint present to prove the card reaches the engine at all.
    agree = all(len(d) == 1 for d in by_footprint.values())
    if agree and len(by_footprint) >= 2:
        print("\nrelocation is a property of the footprint, not the card")
    elif len(by_footprint) < 2:
        print(
            "\nOnly one footprint was measured, so the agreement above has no control and "
            "does not show that the card reaches the engine at all."
        )
    else:
        print(
            "\nCards of one footprint disagree, so relocation is NOT determined by "
            "footprint alone on this build."
        )


if __name__ == "__main__":
    main()
