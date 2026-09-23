"""royalegym.viser: nothing leaves the env until a viewer says hello; then one datagram per
ENGINE TICK that RoyaleViser's StreamSource decodes as a sound Frame (RoyaleViser installed)."""

from __future__ import annotations

import time

import pytest

from royalegym import viser as viser_mod
from royalegym.env import ClashParallelEnv
from royalegym.viser import ViserPublisher

sources = pytest.importorskip("royaleviser.sources")
model = pytest.importorskip("royaleviser.model")


def test_publisher_to_stream_source_round_trip() -> None:
    pub = ViserPublisher(port=0)
    env = ClashParallelEnv(viser=pub)
    env.reset(seed=1)
    assert pub.sent == 0  # no viewer: nothing sent
    assert not pub.attached
    viewer = sources.StreamSource(*pub.address)  # sends the heartbeat
    time.sleep(0.05)
    pub._last_poll = 0.0  # heartbeats are polled at most once a second
    env.step({"blue": 0, "red": 0})
    # One datagram per ENGINE TICK, not per decision. This said 1 until the viewer
    # was found to be running at 2 fps: the env published once per decision, which at
    # decision_ms 500 over a 50 ms tick is ten ticks and so two frames a second
    # however fast the engine ran. Compared against decision_ticks rather than a
    # literal 10, because a literal would silently stop meaning 'every tick' the day
    # decision_ms or tick_ms moves.
    assert pub.sent == env.decision_ticks
    frame = None
    for _ in range(50):
        time.sleep(0.01)
        if (frame := viewer.frame()) is not None:
            break
    assert frame is not None
    assert model.problems(frame) == []
    assert frame.tick == env.battle_state.tick
    assert frame.units_per_tile == env.engine.arena().subtile
    assert [u.uid for u in frame.units] == [e.uid for e in env.battle_state.entities]
    viewer.close()
    env.close()


def test_a_publisher_names_its_run_in_every_frame_and_omits_it_when_unnamed(monkeypatch) -> None:
    """Two runs on one machine share the viewer's fixed ports, so a frame has to say which
    run it came from; the viewer cannot otherwise tell whose battle it is drawing."""
    monkeypatch.setenv(viser_mod.ENV_VAR, "127.0.0.1:0")
    monkeypatch.setenv(viser_mod.RUN_ENV_VAR, " train-hog26 ")
    named = ViserPublisher.from_env()
    monkeypatch.delenv(viser_mod.RUN_ENV_VAR, raising=False)
    unnamed = ViserPublisher.from_env()
    assert named is not None
    assert unnamed is not None
    assert (named.run, unnamed.run) == ("train-hog26", "")

    for pub, expected in ((named, "train-hog26"), (unnamed, None)):
        sent: list[dict] = []
        pub.publish_dict = sent.append  # type: ignore[method-assign]
        env = ClashParallelEnv(viser=pub)
        env.reset(seed=1)
        pub._peer = ("127.0.0.1", 9)  # pretend a viewer said hello
        pub._last_hello = pub._last_poll = time.monotonic()
        env.step({"blue": 0, "red": 0})
        assert sent, "nothing was published while attached"
        assert sent[0]["meta"].get("run") == expected
        env.close()


def test_a_second_publisher_on_a_taken_port_says_it_is_a_second_run() -> None:
    """A bare OSError here reads as a broken machine and means "a run is already streaming".

    On Windows the raw error is WinError 10048, "Only one usage of each socket address is
    normally permitted", which names neither training nor another run nor a remedy. A
    reader meets it at the moment they follow the published example with a run already
    going, and the published example is what told them to use that port. Viser 1
    reproduced it and found it had also been read as "the example is broken" in a gate run.

    Bound on an EPHEMERAL port rather than the documented 9870, so this test cannot fail
    because some other session on this machine happens to be streaming -- and cannot pass
    by accident for the same reason.
    """
    first = ViserPublisher(port=0)
    host, port = first.address
    try:
        with pytest.raises(OSError, match="already using it") as caught:
            ViserPublisher(host=host, port=port)
        message = str(caught.value)
        assert str(port) in message, f"the error does not name the port it failed on: {message}"
        for expected in ("already", "second run", "ROYALEVISER"):
            assert expected in message, (
                f"the error does not mention {expected!r}, so a reader still cannot tell a "
                f"second run from a broken machine: {message}"
            )
        # The cause is kept rather than swallowed: whoever is debugging a genuinely odd
        # bind failure still needs the operating system's own words.
        assert caught.value.__cause__ is not None, (
            "the original OSError was discarded, so a bind failure that is NOT a second "
            "run now has no diagnosis at all"
        )
    finally:
        first.close()
