"""royalegym.viser: nothing leaves the env until a viewer says hello; then one datagram per
ENGINE TICK that RoyaleViser's StreamSource decodes as a sound Frame (RoyaleViser installed)."""

from __future__ import annotations

import time

import pytest

from royalegym import viser as viser_mod
from royalegym.env import ClashParallelEnv
from royalegym.mock_engine import MockEngine
from royalegym.protocol import MatchSetup
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available
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
        # No shell's syntax (cmd's `set NAME=value` sets nothing in PowerShell or bash), and
        # no fixed port to try next: the taken one may be any.
        assert "ROYALEVISER=" not in message, message
        assert "9872" not in message, message
        # The cause is kept rather than swallowed: whoever is debugging a genuinely odd
        # bind failure still needs the operating system's own words.
        assert caught.value.__cause__ is not None, (
            "the original OSError was discarded, so a bind failure that is NOT a second "
            "run now has no diagnosis at all"
        )
    finally:
        first.close()


def test_the_special_forms_rows_decode_as_the_viewer_reads_them() -> None:
    """``evo`` and ``abilities``, the engine's rows with NAMES where the engine has ids,
    through RoyaleViser's own decoder. MockEngine has no forms, so its frames carry these
    rows empty and the round trip above cannot see them; a player that has both is built
    here. A card id left where the viewer reads a name fails ``decode_frame`` HERE, in the
    repo that builds the row, and not later in a viewer's red."""
    import msgspec

    eng = MockEngine()
    ids = {c.name: c.card_id for c in eng.cards()}
    deck = [ids[n] for n in ("Musketeer", "Cannon", "Knight", "Archer", "Giant", "Minions",
                             "Fireball", "Zap")]
    forms = [[2, 1, 0, 0, 0, 0, 0, 0], [0] * 8]
    eng.reset(1, MatchSetup(decks=[deck, deck]))
    state = eng.state()
    # A hero's row as every engine sends it, and a champion's with the columns the champion
    # engine appends (card_id, cooldown_ticks): the viewer's four columns either way.
    blue = msgspec.structs.replace(
        state.players[0], abilities=[[1, 0, 2], [0, 0, 1, ids["Knight"], 40]],
        evo=[[ids["Cannon"], 2, 1, 3]],
    )
    state = msgspec.structs.replace(state, players=[blue, state.players[1]])
    names = {c.card_id: c.name for c in eng.cards()}
    d = viser_mod.frame_dict(
        state, names.__getitem__, eng.arena().subtile, [deck, deck], forms=forms
    )
    frame = model.decode_frame(msgspec.msgpack.encode(d))
    assert model.problems(frame) == []
    assert [tuple(r) for r in frame.players[0].abilities] == [
        ("Musketeer", 1, 0, 2), ("Knight", 0, 0, 1)
    ]
    # The cooldown on the viewer's own key (RoyaleViser 856abbe): -1 where the row does not
    # say. Its length is checked against the rows by model.problems above.
    assert frame.players[0].ability_cooldowns == [-1, 40]
    assert frame.players[1].ability_cooldowns == []
    assert [tuple(r) for r in frame.players[0].evo] == [("Cannon", 2, 1)]
    # The slip it exists for: an id where the viewer reads a name does not decode.
    d["players"][0]["evo"] = [[ids["Cannon"], 2, 1]]
    with pytest.raises(msgspec.ValidationError):
        model.decode_frame(msgspec.msgpack.encode(d))


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_a_battle_with_an_evolution_reaches_the_viewer_with_the_engines_own_rows() -> None:
    """The engine's ``evo`` rows, untouched: [card, plays, next evolved, cycle length], four
    columns since RoyaleSim d925aa8 (2026-10-01). The test above builds its row by hand, and
    with three columns it hid that ``frame_dict`` unpacked three: every battle with an
    evolved card crashed the live viewer, on every published release, until 0.1.16."""
    import msgspec

    eng = RustEngine()
    ids = {c.name: c.card_id for c in eng.cards()}
    deck = [ids[n] for n in ("Skeletons", "Barbarians", "Knight", "Archer", "Giant",
                             "Minions", "Fireball", "Zap")]
    forms = [[1, 1, 0, 0, 0, 0, 0, 0], [0] * 8]
    eng.reset(1, MatchSetup(decks=[deck, deck], forms=forms))
    state = eng.state()
    rows = state.players[0].evo
    assert [len(r) for r in rows] == [4, 4], f"the engine's evo rows changed shape: {rows}"
    names = {c.card_id: c.name for c in eng.cards()}
    d = viser_mod.frame_dict(
        state, names.__getitem__, eng.arena().subtile, [deck, deck], forms=forms
    )
    frame = model.decode_frame(msgspec.msgpack.encode(d))
    assert model.problems(frame) == []
    assert sorted(r[0] for r in frame.players[0].evo) == ["Barbarians", "Skeletons"]
    assert [tuple(r[1:]) for r in frame.players[0].evo] == [tuple(r[1:3]) for r in rows]


def test_the_publisher_can_send_a_busy_battle_frame_on_every_os():
    """macOS caps one UDP datagram at the send buffer (9216 bytes by default), so a busy
    battle's frame failed to send there. The publisher raises its buffer past UDP's 64 KB."""
    import socket

    plain = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    default = plain.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    plain.close()
    pub = ViserPublisher(port=0)
    try:
        raised = pub._sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        assert raised > default, (raised, default)
        assert raised >= 1 << 16, raised
        listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listener.bind(("127.0.0.1", 0))
        try:
            assert pub._sock.sendto(b"x" * 60_000, listener.getsockname()) == 60_000
        finally:
            listener.close()
    finally:
        pub._sock.close()


@pytest.mark.parametrize("switch", ["1", "true", "on", "yes", "TRUE"])
def test_a_plain_on_switch_streams_to_the_default_address(monkeypatch, switch):
    """ROYALEVISER=1 means "stream", at the default address. It was read as host '' port 1, so
    a run bound 127.0.0.1:1: refused on Linux and macOS (a port below 1024), and on Windows
    bound where no viewer ever looks."""
    monkeypatch.setattr(viser_mod, "PORT", 0)  # the default, made ephemeral for the test
    monkeypatch.setenv(viser_mod.ENV_VAR, switch)
    pub = ViserPublisher.from_env()
    try:
        assert pub is not None
        assert pub.address[0] == viser_mod.HOST
        assert pub.address[1] != 1
    finally:
        pub._sock.close()


def test_only_an_address_in_use_is_reported_as_in_use():
    """Any other bind failure says what happened, not "something is already using it"."""
    with pytest.raises(OSError, match=r"cannot stream from 192.0.2.1") as caught:
        ViserPublisher(host="192.0.2.1", port=0)  # TEST-NET-1: never an address of this machine
    assert "already using it" not in str(caught.value)
    assert "192.0.2.1" in str(caught.value)
