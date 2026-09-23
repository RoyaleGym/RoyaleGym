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
