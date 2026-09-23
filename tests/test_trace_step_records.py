"""What a recorded STEP carries: the card each command played, and where it landed.

Both are things the ENGINE already resolved and the trace used to discard, keeping only the
status. A trace could say a command succeeded without saying what it played or where it
ended up.

WHY THESE ARE GRADED THE WAY THEY ARE
    The recorder copies these out of the `DeployResult` list. So comparing a recorded step
    against that same result list would be comparing a thing to itself: both sides read one
    source, they move together, and the test would agree with any transposition.

    Instead the deck is eight DISTINCT cards and the expectation comes from the ENGINE's own
    `PlayerState.hand` read BEFORE the play -- a different path to the same fact. Distinct
    cards are the load-bearing part: with a deck of one card repeated, a bug reporting the
    wrong slot's card, or always slot 0, produces the right number anyway.
"""

from __future__ import annotations

import msgspec
import numpy as np
import pytest

from royalegym.mock_engine import MockEngine
from royalegym.protocol import EMPTY_CARD, DeployCommand, DeployStatus, MatchSetup, Placement
from royalegym.replay import ReplayRecorder, TraceStep
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available

TILE = 18000


def distinct_deck(engine) -> list[int]:
    """Eight different cards, so a slot and its card are not the same number twice."""
    ids = [c.card_id for c in engine.cards()][:8]
    assert len(set(ids)) == 8, "the catalogue did not offer 8 distinct cards"
    return ids


@pytest.mark.parametrize("slot", [0, 1, 2, 3])
def test_a_step_records_the_card_the_slot_held(slot):
    """Graded against the hand the engine reported before the play, not against the result."""
    engine = MockEngine()
    deck = distinct_deck(engine)
    setup = MatchSetup(decks=[deck, deck], elixir_milli=[10000, 10000])
    engine.reset(seed=7, setup=setup)
    rec = ReplayRecorder()
    rec.begin(engine, 7, setup)

    hand_before = list(engine.state().players[0].hand)
    expected = hand_before[slot]
    assert expected != EMPTY_CARD, "the slot was empty, so this case proves nothing"
    assert len(set(hand_before)) == len(hand_before), (
        f"hand {hand_before} repeats a card, so the slot cannot be told from the card"
    )

    command = DeployCommand(team=0, hand_slot=slot, x=9 * TILE, y=5 * TILE)
    results = engine.step([command], 10)
    rec.record_frame(engine)
    rec.record_step(engine.state().tick, 10, [command], results)

    step = rec.trace.steps[-1]
    assert step.card_ids, "the step recorded no card ids at all"
    assert step.card_ids[0] == expected, (
        f"slot {slot} held card {expected} before the play and the trace recorded "
        f"{step.card_ids[0]}; hand was {hand_before}"
    )


def test_a_step_records_a_position_for_every_command():
    """Parallel to `commands`, or a reader cannot tell which landing belongs to which tap."""
    engine = MockEngine()
    deck = distinct_deck(engine)
    setup = MatchSetup(decks=[deck, deck], elixir_milli=[10000, 10000])
    engine.reset(seed=1, setup=setup)
    rec = ReplayRecorder()
    rec.begin(engine, 1, setup)

    commands = [
        DeployCommand(team=0, hand_slot=0, x=5 * TILE, y=5 * TILE),
        DeployCommand(team=1, hand_slot=0, x=7 * TILE, y=20 * TILE),
    ]
    results = engine.step(commands, 10)
    rec.record_frame(engine)
    rec.record_step(engine.state().tick, 10, commands, results)

    step = rec.trace.steps[-1]
    assert len(step.landed) == len(commands) == len(step.card_ids), (
        f"{len(commands)} commands but {len(step.landed)} landings and "
        f"{len(step.card_ids)} card ids; they must be parallel"
    )
    assert all(len(p) == 2 for p in step.landed), f"a landing is not a pair: {step.landed}"


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_the_recorded_landing_is_where_the_engine_PUT_it_not_where_the_tap_asked():
    """The whole reason the position is worth recording.

    A building whose footprint does not fit is MOVED, so the landing differs from the
    command. If this recorded the command instead, a trace would agree with itself and a
    reader measuring relocation from it would measure zero.

    The test refuses to pass vacuously: it requires a tap that actually relocated, so a
    build where nothing moves fails here rather than quietly proving nothing.
    """
    engine = RustEngine()
    cards = engine.cards()
    cannon = next(
        (i for i, c in enumerate(cards)
         if c.name == "Cannon" and c.placement == Placement.BUILDING),
        None,
    )
    if cannon is None:
        pytest.skip("this card table has no Cannon")

    setup = MatchSetup(decks=[[cannon] * 8] * 2, elixir_milli=[10000, 10000])
    moved = same = 0
    for tx in range(2, 16):
        for ty in range(3, 13, 2):
            engine.reset(seed=0, setup=setup)
            engine.step([], 10)
            rec = ReplayRecorder()
            rec.begin(engine, 0, setup)
            x, y = tx * TILE + TILE // 2, ty * TILE + TILE // 2
            command = DeployCommand(team=0, hand_slot=0, x=x, y=y)
            results = engine.step([command], 1)
            rec.record_frame(engine)
            rec.record_step(engine.state().tick, 1, [command], results)
            if results[0].status != DeployStatus.OK:
                continue
            got = rec.trace.steps[-1].landed[0]
            if got == [x, y]:
                same += 1
            else:
                moved += 1

    assert moved + same, "no tap was accepted, so this test compared nothing"
    assert moved, (
        f"{same} accepted taps and not one relocated, so this test cannot tell a recorded "
        "landing from a recorded command"
    )
    assert same, (
        f"all {moved} accepted taps relocated; with no honest tap in the sample a field "
        "that always reported 'somewhere else' would also pass"
    )


def test_a_step_recorded_before_these_fields_existed_still_decodes():
    """`array_like=True` means positional fields, so an older step is a SHORTER array."""
    before = [30, 10, [], [0]]
    step = msgspec.msgpack.decode(msgspec.msgpack.encode(before), type=TraceStep)
    assert step.tick == 30
    assert step.card_ids == [], (
        "an older step decoded with something other than an empty list, so a reader would "
        "see cards the recording never contained"
    )
    assert step.landed == []


# ---------------------------------------------------------------------------
# Reading those fields back as a relocation rate, which is what they were added for.


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
@pytest.mark.parametrize(("card", "footprint"), [("Cannon", 3), ("Tesla", 2)])
def test_the_three_criteria_disagree_and_the_struct_keeps_them_apart(card, footprint):
    """An EVEN footprint makes `moved_point` useless, and that is the whole warning.

    A 2x2 snaps to a tile CORNER, so its centre can never be the tapped tile CENTRE:
    `moved_point` is then 100% no matter how well the engine behaved. If a reader takes that
    for a relocation rate they conclude Tesla is the worst card in the game. This pins that
    the struct reports the three separately and that they really do differ.
    """
    from royalegym.landing import relocations

    engine = RustEngine()
    ids = {c.name: i for i, c in enumerate(engine.cards())}
    if card not in ids:
        pytest.skip(f"this card table has no {card}")
    assert engine.cards()[ids[card]].footprint_tiles == footprint, (
        f"{card} is not {footprint}x{footprint} on this table, so this case tests something else"
    )

    setup = MatchSetup(decks=[[ids[card]] * 8] * 2, elixir_milli=[10000, 10000])
    rec = ReplayRecorder()
    engine.reset(seed=0, setup=setup)
    engine.step([], 10)
    rec.begin(engine, 0, setup)
    for tx in range(2, 16):
        for ty in range(3, 13, 2):
            command = DeployCommand(
                team=0, hand_slot=0, x=tx * TILE + TILE // 2, y=ty * TILE + TILE // 2
            )
            results = engine.step([command], 1)
            rec.record_frame(engine)
            rec.record_step(engine.state().tick, 1, [command], results)

    rows = relocations(rec.trace)
    assert rows, "no accepted command was recorded, so nothing below is measured"
    assert all(r.card_name == card for r in rows), (
        f"the card name came back as {rows[0].card_name!r}; it is read from the trace header"
    )
    point = sum(r.moved_point for r in rows)
    lost = sum(r.lost_tile for r in rows)
    assert lost <= point, "a tap that kept its tile cannot have moved less than not at all"
    if footprint % 2 == 0:
        assert point == len(rows), (
            f"an even footprint cannot land on a tile centre, so moved_point should be all "
            f"{len(rows)} taps, got {point}"
        )
        assert lost < point, (
            "every tap counted as lost as well as moved, so this build gives the reader no "
            "reason to prefer the criterion that matters"
        )


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_a_trace_without_the_fields_refuses_instead_of_reporting_no_relocation():
    """Returning [] would read as 0% relocation, which is wrong AND plausible."""
    from royalegym.landing import relocations

    engine = RustEngine()
    setup = MatchSetup(decks=[list(range(8))] * 2, elixir_milli=[10000, 10000])
    engine.reset(seed=0, setup=setup)
    rec = ReplayRecorder()
    rec.begin(engine, 0, setup)
    command = DeployCommand(team=0, hand_slot=0, x=9 * TILE, y=5 * TILE)
    results = engine.step([command], 1)
    rec.record_step(engine.state().tick, 1, [command], results)
    # Age the step by hand: exactly what an older recording decodes to.
    rec.trace.steps[-1].card_ids = []
    rec.trace.steps[-1].landed = []
    with pytest.raises(ValueError, match="predates them"):
        relocations(rec.trace)


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_every_criterion_answers_for_a_card_with_no_footprint():
    """THE POPULATION THE OTHER TESTS HERE COULD NOT SEE.

    Every test above uses a deck of buildings, so none of them ever held a row whose
    `footprint` is None -- which is troops, spells and spawners, and in a real trace is most
    rows. `lost_tile` raised a TypeError on all of them while `moved_point` and `moved_tile`
    returned cleanly, so the criterion documented as the one to USE was the only one that
    crashed. Found by train on their first real trace: 58 troop commands against 6 building
    ones.

    A deck of one placement class is a sample that cannot fail in the way the code does.
    """
    from royalegym.landing import relocations

    engine = RustEngine()
    cards = engine.cards()
    by_placement: dict[int, int] = {}
    for i, c in enumerate(cards):
        by_placement.setdefault(int(c.placement), i)
    troop = by_placement.get(int(Placement.TROOP))
    building = by_placement.get(int(Placement.BUILDING))
    assert troop is not None, "this card table has no troop, so the defect cannot appear"
    assert building is not None, "this card table has no building to contrast against"
    # Everything else the table offers, so no placement class is left untested by omission.
    skip = (int(Placement.TROOP), int(Placement.BUILDING))
    others = [i for p, i in by_placement.items() if p not in skip]

    deck = [troop, building, *others][:8]
    deck = (deck * 8)[:8]
    setup = MatchSetup(decks=[deck, deck], elixir_milli=[10000, 10000])
    engine.reset(seed=0, setup=setup)
    engine.step([], 10)
    rec = ReplayRecorder()
    rec.begin(engine, 0, setup)
    for slot in range(4):
        command = DeployCommand(
            team=0, hand_slot=slot, x=9 * TILE + TILE // 2, y=5 * TILE + TILE // 2
        )
        results = engine.step([command], 1)
        rec.record_frame(engine)
        rec.record_step(engine.state().tick, 1, [command], results)

    rows = relocations(rec.trace)
    assert rows, "nothing was accepted, so this test measured nothing"
    footless = [r for r in rows if not r.has_footprint]
    assert footless, (
        "every accepted row had a footprint, so this test is the buildings-only sample "
        "again and cannot see the defect it exists for"
    )
    for r in rows:
        # The point of the test: none of the three may raise, for any card.
        assert isinstance(r.moved_point, bool)
        assert isinstance(r.moved_tile, bool)
        assert isinstance(r.lost_tile, bool), f"{r.card_name} did not answer lost_tile"

    # And the filter the module documents must work on exactly these rows.
    rate = sum(r.lost_tile for r in rows if r.has_footprint)
    assert rate >= 0
    with pytest.raises(TypeError):
        # Pinned so the docs are never quietly changed back: this is why has_footprint exists.
        [r for r in rows if r.footprint >= 2]


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_a_card_id_the_table_does_not_have_is_named_rather_than_guessed():
    """Found by a plant that changed nothing: the only difference between two versions of
    the footprint lookup was an out-of-range card id, and no test reached that path.

    A refused command can carry EMPTY_CARD, and `accepted_only=False` is a documented way
    to ask for those rows. Guessing a footprint of 1 for an unknown card would make
    `lost_tile` answer confidently about a card it cannot identify.
    """
    from royalegym.landing import relocations

    engine = RustEngine()
    setup = MatchSetup(decks=[list(range(8))] * 2, elixir_milli=[10000, 10000])
    engine.reset(seed=0, setup=setup)
    engine.step([], 10)
    rec = ReplayRecorder()
    rec.begin(engine, 0, setup)
    command = DeployCommand(team=0, hand_slot=0, x=9 * TILE, y=5 * TILE)
    results = engine.step([command], 1)
    rec.record_frame(engine)
    rec.record_step(engine.state().tick, 1, [command], results)
    rec.trace.steps[-1].card_ids = [10_000]  # no table has this

    rows = relocations(rec.trace, accepted_only=False)
    assert len(rows) == 1
    row = rows[0]
    assert row.card_name == "card:10000", f"an unknown card was named {row.card_name!r}"
    assert row.footprint is None, (
        f"an unknown card was given footprint {row.footprint!r}; a guessed footprint makes "
        "lost_tile answer confidently about a card it cannot identify"
    )
    assert isinstance(row.lost_tile, bool)


# ---------------------------------------------------------------------------
# The per-tile map: the same question asked of the BOARD instead of of a tap.


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_the_map_agrees_with_the_rate_that_was_measured_by_sweep():
    """The map is the sweep, per tile. If they disagree one of them is wrong.

    The published figures are 15.0% for a 3x3 and 10.0% for a 2x2 over every offered tile of
    a fresh board, and they are quoted in `action.py` and in two handoffs. A map that did not
    reproduce them would mean the number in the docs describes nothing anyone can recompute.
    """
    from royalegym.landing import tile_loss_map

    engine = RustEngine()
    names = {c.name for c in engine.cards()}
    for card, expected in (("Cannon", 0.150), ("Tesla", 0.100)):
        if card not in names:
            pytest.skip(f"this card table has no {card}")
        engine.reset(seed=0, setup=MatchSetup(decks=[list(range(8))] * 2))
        engine.step([], 10)
        m = tile_loss_map(engine, card)
        assert int(m.offered.sum()) == 240, (
            f"{card} was offered {int(m.offered.sum())} tiles, not the 240 the sweep saw; "
            "the map and the published rate are not over the same population"
        )
        assert abs(m.rate() - expected) < 0.005, (
            f"{card} map says {m.rate():.3f}, the documented sweep says {expected:.3f}"
        )


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_two_cards_of_one_footprint_give_the_SAME_map():
    """'Relocation depends only on the footprint' was a measurement; this makes it a test.

    It was reported to another session as a finding and used to redesign their experiment, so
    it should not rest on a table someone printed once. The 2x2 differing from the 3x3 is the
    control: without it, a map that ignored the card entirely would also pass.
    """
    from royalegym.landing import tile_loss_map

    engine = RustEngine()
    names = {c.name for c in engine.cards()}
    threes = [n for n in ("Cannon", "Mortar", "Tombstone", "BombTower") if n in names]
    if len(threes) < 2 or "Tesla" not in names:
        pytest.skip("need two 3x3 cards and a Tesla on this table")

    engine.reset(seed=0, setup=MatchSetup(decks=[list(range(8))] * 2))
    engine.step([], 10)
    maps = {n: tile_loss_map(engine, n) for n in [*threes, "Tesla"]}
    first = maps[threes[0]]
    for other in threes[1:]:
        assert np.array_equal(first.lost, maps[other].lost), (
            f"{threes[0]} and {other} are both 3x3 but lose different tiles"
        )
        assert np.array_equal(first.offered, maps[other].offered)
    assert not np.array_equal(first.lost, maps["Tesla"].lost), (
        "the 3x3 and the 2x2 maps are identical, so this function is not reading the card "
        "at all and the agreement above means nothing"
    )


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_an_unoffered_tile_says_None_rather_than_keeps_its_tile():
    """False would fold 'you cannot play here' into 'you can, and it stays put'."""
    from royalegym.landing import tile_loss_map

    engine = RustEngine()
    if "Cannon" not in {c.name for c in engine.cards()}:
        pytest.skip("this card table has no Cannon")
    engine.reset(seed=0, setup=MatchSetup(decks=[list(range(8))] * 2))
    engine.step([], 10)
    m = tile_loss_map(engine, "Cannon")
    unoffered = np.argwhere(~m.offered)
    assert len(unoffered), "every tile was offered, so this test checks nothing"
    ty, tx = unoffered[0]
    assert m.would_lose_tile(int(tx), int(ty)) is None
    offered = np.argwhere(m.offered)
    oy, ox = offered[0]
    assert isinstance(m.would_lose_tile(int(ox), int(oy)), bool)
    assert m.would_lose_tile(-1, 0) is None, "off-board should not be reported as a tile"


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_a_card_with_no_footprint_is_refused_rather_than_mapped_all_False():
    """An all-False map reads as 'this card never loses its tile', which is a measurement.

    For a troop it is a category error, and the difference matters because the caller is
    weighting a tap distribution by these maps.
    """
    from royalegym.landing import tile_loss_map

    engine = RustEngine()
    troop = next((c.name for c in engine.cards() if c.placement == Placement.TROOP), None)
    if troop is None:
        pytest.skip("this card table has no troop")
    engine.reset(seed=0, setup=MatchSetup(decks=[list(range(8))] * 2))
    with pytest.raises(ValueError, match="no footprint"):
        tile_loss_map(engine, troop)
