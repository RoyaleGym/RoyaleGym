"""A trace carries each player's special-form rows, so a replay can draw them.

The viewer plays traces, and it draws a hero's crown, an evolution's frame and a button's
state from the engine's per-player rows: ``evo`` ([card_id, plays since the last evolved
play, 1 when the next play is evolved]) and ``abilities`` ([available, spent, cost, ...]).
A trace recorded only the deck's forms, once, in its header, so no replay could show a
button pressed or an evolution coming up. Each frame now copies both rows from the state.

WHAT IT CHECKS
    a. Against the engine: a battle with a hero Musketeer and an evolved Cannon, the hero
       pressed; every recorded frame's rows are the engine's rows at that tick, and the
       press shows (the hero's charge turns spent).
    b. A frame recorded before the fields existed still decodes, with empty rows.

SKIPS
    Without royalesim built, or on an engine without hero or evolved forms. Not a pass.
"""

from __future__ import annotations

import msgspec
import pytest

from royalegym.protocol import (
    HAND_SIZE,
    TEAMS,
    DeployCommand,
    DeployStatus,
    MatchSetup,
    ShuffleMode,
    to_engine,
)
from royalegym.replay import ReplayRecorder, TraceFrame
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available

DECK = ("Musketeer", "Cannon", "Knight", "Archer", "Giant", "Minions", "Fireball", "Zap")
FORMS = [2, 1, 0, 0, 0, 0, 0, 0]  # the Musketeer's hero form, the Cannon's evolution


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_every_frame_carries_the_engines_evo_and_ability_rows():
    engine = RustEngine()
    ids = {c.name: c.card_id for c in engine.cards()}
    deck = [ids[n] for n in DECK]
    setup = MatchSetup(
        decks=[deck, deck], shuffle=ShuffleMode.NONE, forms=[FORMS, FORMS],
        elixir_milli=[10000, 10000], start_tick=engine.rules().deploy_lockout_ticks,
    )
    try:
        engine.reset(1, setup)
    except (ValueError, NotImplementedError) as exc:
        pytest.skip(f"SKIPPED, NOT PASSED: this engine runs no hero or evolved form ({exc})")
    rec = ReplayRecorder(frame_every_tick=True)
    rec.begin(engine, 1, setup)
    t = engine.arena().subtile
    state = engine.state()
    for card, (tx, ty) in (("Musketeer", (9, 10)), ("Cannon", (4, 8))):
        state = engine.state()
        cmds = [
            DeployCommand(team, state.players[team].hand.index(ids[card]),
                          *to_engine(engine.arena(), team, tx * t + t // 2, ty * t + t // 2))
            for team in TEAMS
        ]
        assert [r.status for r in engine.step(cmds, 1)] == [DeployStatus.OK] * 2
        rec.record_frame(engine)
    pressed = False
    checked = 0
    for _ in range(120):
        state = engine.state()
        frame = rec.trace.frames[-1]
        assert frame.tick == state.tick
        assert frame.evo == [p.evo for p in state.players], (frame.evo, state.players)
        assert frame.abilities == [p.abilities for p in state.players], frame.abilities
        checked += 1
        cmds = []
        if not pressed and engine.check_deploy(DeployCommand(0, HAND_SIZE, 0, 0)) == 0:
            cmds = [DeployCommand(0, HAND_SIZE, 0, 0)]
            pressed = True
        engine.step(cmds, 1)
        rec.record_frame(engine)
    assert pressed, "the hero's button never came on"
    frames = rec.trace.frames
    assert any(f.abilities[0] and f.abilities[0][0][1] == 1 for f in frames), (
        "no frame shows the hero's charge spent after its press"
    )
    assert all(f.evo[0] for f in frames), "a frame lost the evolved Cannon's row"
    assert checked > 100


def test_a_frame_recorded_before_the_form_rows_existed_still_decodes():
    """Trailing and defaulted, as ``next_cards`` and ``projectiles`` are: an older frame is a
    shorter array and decodes with no rows, not invented ones."""
    before = [30, [], [5000, 5000], [0, 0], [[0, 1, 2, 3], [0, 1, 2, 3]], "abc", [], [4, 5], []]
    frame = msgspec.msgpack.decode(msgspec.msgpack.encode(before), type=TraceFrame)
    assert (frame.tick, frame.next_cards) == (30, [4, 5])
    assert (frame.evo, frame.abilities) == ([], [])
