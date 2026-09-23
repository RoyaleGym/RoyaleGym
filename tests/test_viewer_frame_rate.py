"""The viewer gets one frame per ENGINE TICK, not one per decision.

WHY THIS EXISTS
    The owner reported the viewer as low frame rate. It was not a rendering problem and
    not a network one: the env published once per decision. The arithmetic is exact rather
    than a guess. A decision is ``decision_ms`` of game time rounded up to whole engine
    ticks, so at the default 500 ms over a 50 ms tick a decision is ten ticks, and the
    viewer received 2 frames a second however fast the engine ran. The viewer's own target
    is 20 fps and its draw loop caps at 60, so one frame per tick arrives with headroom.

    Found by train, who could work around it for a tool of their own through the recorder
    hook, and could not for ``royalelearn play --viser``, which builds its own env.

WHAT THESE PIN, AND WHY EACH ONE
    The count, because that is the defect. The no-viewer path, because making every
    training run publish ten times as often for nobody to watch would be a worse bug than
    the one being fixed. The events, because a play repeated on all ten ticks would show
    up ten times in the viewer's event log. And the recorder, because a recording must not
    change depending on whether somebody happened to have the viewer open.
"""

from __future__ import annotations

import pytest

from royalegym.env import ClashParallelEnv
from royalegym.mock_engine import MockEngine


class SpyPublisher:
    """Stands in for ViserPublisher and counts what the env sends it."""

    def __init__(self, attached: bool = True) -> None:
        self.attached = attached
        self.frames: list[tuple[int, int]] = []  # (tick, number of event lines)

    def publish(self, state, cards, arena, decks, events=()) -> None:
        # Two details copied from the real ViserPublisher rather than invented, because
        # a fake that is more permissive than the thing it stands in for tests a system
        # that does not exist. `events` is optional, since reset() publishes an opening
        # frame without any; and a DETACHED publisher sends nothing, which is what makes
        # the no-viewer path cost nothing.
        if not self.attached:
            return
        self.frames.append((state.tick, len(events)))


class CountingEngine(MockEngine):
    """A MockEngine that records how many times it was stepped and with what tick count."""

    def __init__(self) -> None:
        super().__init__()
        self.step_calls: list[int] = []

    def step(self, commands, ticks):
        self.step_calls.append(ticks)
        return super().step(commands, ticks)


def build(publisher: SpyPublisher | None, engine: CountingEngine | None = None):
    env = ClashParallelEnv(engine=engine or CountingEngine(), viser=publisher)
    env.reset(seed=0)
    # reset() publishes an opening frame. Dropped here so the counts below are about
    # one DECISION, which is what the defect was about.
    publisher.frames.clear()
    return env


def noop_actions(env) -> dict:
    from royalegym.action import NOOP

    return dict.fromkeys(env.agents, NOOP)


def test_an_attached_viewer_gets_one_frame_per_engine_tick() -> None:
    spy = SpyPublisher(attached=True)
    env = build(spy)
    ticks = env.decision_ticks
    assert ticks > 1, (
        f"decision_ticks is {ticks}, so one frame per decision and one per tick are the "
        "same number and this test cannot tell them apart"
    )
    env.step(noop_actions(env))
    assert len(spy.frames) == ticks, (
        f"a decision is {ticks} engine ticks and the viewer got {len(spy.frames)} frames. "
        f"One per decision is the defect this exists to catch."
    )
    seen = [tick for tick, _ in spy.frames]
    assert len(set(seen)) == ticks, f"frames repeat a tick rather than advancing: {seen}"


def test_the_final_frame_of_a_decision_is_not_published_twice() -> None:
    """_advance publishes every tick; step() must then not publish the last one again."""
    spy = SpyPublisher(attached=True)
    env = build(spy)
    env.step(noop_actions(env))
    ticks = [tick for tick, _ in spy.frames]
    assert len(ticks) == len(set(ticks)), f"a tick was published twice: {ticks}"


def test_nothing_changes_for_a_run_with_no_viewer_attached() -> None:
    """Ten times the publishing for nobody watching would be worse than the original bug."""
    engine = CountingEngine()
    spy = SpyPublisher(attached=False)
    env = ClashParallelEnv(engine=engine, viser=spy)
    env.reset(seed=0)
    engine.step_calls.clear()
    env.step(noop_actions(env))
    assert spy.frames == [], "a detached viewer was sent frames"
    assert engine.step_calls == [env.decision_ticks], (
        f"with no viewer attached the engine should be stepped ONCE for the whole "
        f"decision, and it was stepped {engine.step_calls}. Per-tick stepping for a run "
        "nobody is watching is a cost with no benefit."
    )


def test_a_play_appears_on_one_tick_and_not_on_all_of_them() -> None:
    """Repeating the deploy results on every tick would show each play ten times."""
    spy = SpyPublisher(attached=True)
    env = build(spy)
    acted = False
    for _ in range(40):
        actions = {}
        for agent in env.agents:
            mask = env._masks[agent]
            playable = [i for i in range(1, len(mask)) if mask[i]]
            actions[agent] = playable[0] if playable else 0
        spy.frames.clear()
        env.step(actions)
        with_events = [n for _, n in spy.frames if n]
        if with_events:
            acted = True
            assert len(with_events) == 1, (
                f"the deploys were published on {len(with_events)} ticks of one decision. "
                "They happened on one tick and belong on one tick, or the viewer's event "
                "log shows every play ten times."
            )
            assert spy.frames[0][1] > 0, (
                "the events were not on the FIRST tick of the decision, which is where "
                "the engine evaluated the commands"
            )
            break
        if not env.agents:
            break
    assert acted, "no decision in 40 steps produced an accepted play, so events went untested"


@pytest.mark.parametrize("attached", [True, False])
def test_a_recorder_gets_the_same_frames_whoever_is_watching(attached: bool) -> None:
    """A recording must not depend on whether the viewer happened to be open.

    This is the trap in the fix: the per-tick loop already existed FOR the recorder, so
    the easy version reuses it wholesale and a recorder that asked for one frame per
    decision silently starts getting ten as soon as somebody attaches a viewer.
    """

    class CountingRecorder:
        frame_every_tick = False

        def __init__(self) -> None:
            self.frames = 0
            self.steps = 0

        def begin(self, *a, **k) -> None:
            pass

        def record_frame(self, engine) -> None:
            self.frames += 1

        def record_step(self, *a, **k) -> None:
            self.steps += 1

        def end(self, engine) -> None:
            pass

    rec = CountingRecorder()
    env = ClashParallelEnv(
        engine=CountingEngine(), viser=SpyPublisher(attached=attached), recorder=rec
    )
    env.reset(seed=0)
    rec.frames = 0
    rec.steps = 0
    env.step(noop_actions(env))
    assert (rec.frames, rec.steps) == (1, 1), (
        f"a recorder with frame_every_tick False got {rec.frames} frames for one decision "
        f"with the viewer attached={attached}. It must get exactly one either way."
    )
