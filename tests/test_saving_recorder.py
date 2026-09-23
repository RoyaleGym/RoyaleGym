"""``SavingReplayRecorder`` writes completed traces, so a rollout worker can produce them.

WHY IT EXISTS AT ALL
    ``ReplayRecorder`` already takes ``on_finish``, which is the better tool for a script
    and is unreachable from a training run: the recorder is an ``EnvFactory`` COMPONENT,
    given as an importable class plus JSON kwargs, and a callable is not something JSON can
    express. A worker that receives a pickled recipe can be handed a class and a directory
    and cannot be handed a function.

WHAT THESE PIN, AND WHY EACH IS A ROLLOUT-WORKER PROBLEM RATHER THAN A SCRIPT PROBLEM
    A script runs once, alone, and its reader is a person who waits for it. A rollout worker
    is one of several processes writing into a directory that something else may be reading
    at the same moment, for hours, unattended. Each test below is one of the ways that
    difference bites:

    * two workers must not write the same filename,
    * a reader must never see a half-written trace,
    * a bad path must fail when the config loads, not after the first episode of a
      five-hour run,
    * and pruning must not delete another worker's files.

    The last one is the one that would have been easiest to get wrong and hardest to
    notice: eight workers each keeping "the newest four" of a shared directory would each
    be deleting the others' newest, and the directory would look plausibly populated
    throughout.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from royalegym.mock_engine import MockEngine
from royalegym.protocol import MatchSetup
from royalegym.replay import SavingReplayRecorder, load_trace

DECK = list(range(8))


def play_one(recorder: SavingReplayRecorder, engine: MockEngine, seed: int = 0) -> None:
    """One short battle through the recorder's own hooks, as the env drives them."""
    setup = MatchSetup(decks=[DECK, DECK])
    engine.reset(seed=seed, setup=setup)
    recorder.begin(engine, seed, setup)
    for _ in range(3):
        engine.step([], 10)
        recorder.record_frame(engine)
        recorder.record_step(engine.state().tick, 10, [], [])
    recorder.end(engine)


def test_it_saves_every_nth_completed_trace_and_not_the_others(tmp_path):
    rec = SavingReplayRecorder(out_dir=str(tmp_path), every=3, keep=10)
    engine = MockEngine()
    counts = []
    for i in range(7):
        play_one(rec, engine, seed=i)
        counts.append(len(list(tmp_path.glob("*.msgpack"))))
    assert counts == [0, 0, 1, 1, 1, 2, 2], (
        f"saved after episodes {counts}; with every=3 a file should appear on the 3rd and "
        "6th completed trace and on no other"
    )
    assert rec.completed == 7


def test_a_saved_trace_loads_back(tmp_path):
    """A file that cannot be read is not a saved trace, however plausible its size."""
    rec = SavingReplayRecorder(out_dir=str(tmp_path), every=1)
    play_one(rec, MockEngine())
    written = list(tmp_path.glob("*.msgpack"))
    assert len(written) == 1
    trace = load_trace(written[0])
    assert trace.result is not None
    assert trace.frames, "the trace loaded but carries no frames"


def test_the_filename_carries_the_pid_so_two_workers_cannot_collide(tmp_path):
    """Workers are separate PROCESSES with separate counters.

    Without the pid, worker A's 50th and worker B's 50th are the same name and one
    silently replaces the other -- in a directory that still looks populated.
    """
    rec = SavingReplayRecorder(out_dir=str(tmp_path), every=1)
    play_one(rec, MockEngine())
    name = next(iter(tmp_path.glob("*.msgpack"))).name
    assert name.startswith(f"{os.getpid()}-"), (
        f"{name!r} does not begin with this process's pid, so a second worker writing its "
        "own episode of the same ordinal would overwrite it"
    )


def test_no_partial_file_is_left_where_a_reader_would_find_it(tmp_path):
    """The write is atomic: a temporary name, then a rename.

    A viewer polling this directory would otherwise open a half-written trace, and msgpack
    does not fail loudly on a truncated one.
    """
    rec = SavingReplayRecorder(out_dir=str(tmp_path), every=1)
    play_one(rec, MockEngine())
    leftovers = [p.name for p in tmp_path.iterdir() if not p.name.endswith(".msgpack")]
    assert leftovers == [], f"a non-final file was left behind: {leftovers}"


def test_pruning_keeps_the_newest_and_deletes_only_its_own(tmp_path):
    """Two recorders share a directory, as two rollout workers would.

    Each must keep its own newest `keep` and touch nothing of the other's. A recorder that
    pruned the DIRECTORY would delete its neighbour's newest files, and the directory would
    still look healthy.
    """
    mine = SavingReplayRecorder(out_dir=str(tmp_path), every=1, keep=2)
    theirs = SavingReplayRecorder(out_dir=str(tmp_path), every=1, keep=2)
    engine = MockEngine()
    for i in range(4):
        play_one(mine, engine, seed=i)
    theirs.saved.append(tmp_path / "other-worker-0001.msgpack")
    theirs.saved[-1].write_bytes(b"not mine")

    assert len(mine.saved) == 2, f"kept {len(mine.saved)} of its own, expected 2"
    for p in mine.saved:
        assert p.is_file(), f"{p.name} is tracked as kept but is not on disk"
    assert (tmp_path / "other-worker-0001.msgpack").is_file(), (
        "the other recorder's file was deleted; a shared directory would have workers "
        "pruning each other"
    )


def test_a_bad_directory_fails_when_the_config_loads_not_after_an_episode(tmp_path):
    """A five-hour run should not discover its output path is unusable at the first end().

    The mirror of the failure train hit from the other side: a publisher that reported
    "3601 frames, 0 sent" and looked like it worked throughout.
    """
    blocker = tmp_path / "a-file"
    blocker.write_text("not a directory")
    with pytest.raises(OSError, match="a-file"):
        SavingReplayRecorder(out_dir=str(blocker / "under-a-file"))


@pytest.mark.parametrize(("kwargs", "expect"), [
    ({"every": 0}, "would save nothing"),
    ({"keep": 0}, "would delete what it just wrote"),
])
def test_settings_that_would_silently_produce_nothing_are_refused(tmp_path, kwargs, expect):
    """The message is asserted, not just the type: each of these fails in a way whose
    symptom is an EMPTY directory, so the error has to say which setting caused it."""
    with pytest.raises(ValueError, match=expect):
        SavingReplayRecorder(out_dir=str(tmp_path), **kwargs)


def test_every_argument_is_expressible_in_json(tmp_path):
    """The whole reason this class exists rather than `on_finish`.

    A recorder is an EnvFactory component: an importable class plus JSON kwargs. If any
    argument needed a Python object, a rollout worker could not be given one -- which is
    exactly why `on_finish` is unreachable from a config.
    """
    import inspect
    import json

    sig = inspect.signature(SavingReplayRecorder.__init__)
    kwargs = {n: p.default for n, p in sig.parameters.items()
              if n not in ("self", "out_dir") and p.default is not inspect.Parameter.empty}
    kwargs["out_dir"] = str(tmp_path)
    json.dumps(kwargs)  # raises if any default is not JSON-expressible
    rebuilt = SavingReplayRecorder(**json.loads(json.dumps(kwargs)))
    assert rebuilt.out_dir == tmp_path


# ---------------------------------------------------------------------------
# What a recorded FRAME carries. Not about saving, but about whether a replay can be
# honest: a trace without next_cards has to report the cycle as unknown, and guessing it
# would be worse than saying so.


def test_a_recorded_frame_carries_the_next_card_each_seat_will_draw():
    """Graded against the ENGINE's own PlayerState, not against the recorder.

    Comparing a recorded frame with the recorder's own idea of the frame would be the
    shape that hid a transposition in the learner's viser sink for a day: both sides read
    the same source, so they move together and agree about anything.
    """
    engine = MockEngine()
    setup = MatchSetup(decks=[DECK, DECK])
    engine.reset(seed=3, setup=setup)
    # every=10**9 so nothing is written: this test is about the FRAME, not the saving.
    out = Path(os.environ.get("TMP", ".")) / "royalegym-frame-test"
    rec = SavingReplayRecorder(out_dir=str(out), every=10**9)
    rec.begin(engine, 3, setup)
    engine.step([], 10)
    rec.record_frame(engine)
    trace = rec.trace
    assert trace is not None
    frame = trace.frames[-1]
    from_engine = [p.next_card for p in engine.state().players]
    assert frame.next_cards == from_engine, (
        f"the frame says {frame.next_cards} and the engine says {from_engine}"
    )
    assert len(frame.next_cards) == len(engine.state().players)
    # Non-vacuity: an all-empty list would satisfy the comparison above if the engine also
    # reported nothing, and then this test would pass while carrying no information.
    assert from_engine, "the engine reported no players, so the comparison above is empty"


def test_a_frame_recorded_before_next_cards_existed_still_decodes():
    """`array_like=True` means positional fields, so an older trace is a SHORTER array.

    The field is trailing and defaulted for that reason, exactly as `spells` was before it.
    A trace recorded yesterday must still load, and it must load saying [] rather than
    inventing a cycle.
    """
    import msgspec

    from royalegym.replay import TraceFrame

    before = [30, [], [5000, 5000], [0, 0], [[0, 1, 2, 3], [0, 1, 2, 3]], "abc", []]
    frame = msgspec.msgpack.decode(msgspec.msgpack.encode(before), type=TraceFrame)
    assert frame.tick == 30
    assert frame.next_cards == [], (
        "an older frame decoded with something other than an empty list, so a replay of it "
        "would show a cycle the recording never contained"
    )
