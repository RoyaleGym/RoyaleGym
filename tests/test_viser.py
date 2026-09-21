"""royalegym.viser: nothing leaves the env until a viewer says hello; then one datagram per
publish that RoyaleViser's StreamSource decodes as a sound Frame (RoyaleViser installed)."""

from __future__ import annotations

import time

import pytest

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
    assert pub.sent == 1
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
