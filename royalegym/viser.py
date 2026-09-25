"""The engine side of RoyaleViser: publish battle state over UDP only while a viewer watches.

The viewer's rule: a separate process, never in the tick loop, zero cost when nobody is
watching. This module is the whole of what royalegym knows about the viewer; it imports
nothing from RoyaleViser (dependency direction stays RoyaleLearn -> RoyaleGym -> RoyaleSim)
and nothing graphical.

    from royalegym.env import ClashParallelEnv, ClashSelfPlayVecEnv
    from royalegym.viser import ViserPublisher
    env = ClashParallelEnv(viser=ViserPublisher())        # 127.0.0.1:9870
    # or, for self-play:  set ROYALEVISER=127.0.0.1:9870, then
    vec = ClashSelfPlayVecEnv(8)                          # binds ONE, watches game 0
    # then, in another process:  python -m royaleviser --stream 127.0.0.1:9870

WHO READS ROYALEVISER
    ``ViserPublisher.from_env()``, and only ``ClashSelfPlayVecEnv`` calls it.
    ``ClashParallelEnv`` publishes when it is HANDED a publisher and reads no
    environment variable of its own: a viewer watches ONE battle and has one fixed
    port, so N envs each building their own from the variable is ``OSError 10048``,
    which is what self-play with more than one game used to raise. The decision
    belongs to whoever knows how many battles there are.

PROTOCOL
    The viewer sends the heartbeat datagram HELLO to (host, port) once a second while it is
    open. ``publish`` costs one clock read while nobody has said hello in ATTACH_TIMEOUT_S:
    the socket is polled for heartbeats at most once a second (a non-blocking recvfrom).
    While attached, every call encodes one frame with msgspec msgpack and sends ONE datagram
    to the last heartbeat's address. A datagram over MAX_DATAGRAM bytes (UDP over IPv4) is
    resent with every unit's ``path`` emptied, then counted in ``dropped`` if still too big.
    Measured 2026-09-20 (MockEngine): 2.6 KB per frame with 12 units + 6 towers, 11 KB with 60
    entities, far under the limit.

WIRE FORM
    ``frame_dict`` is royaleviser.model.Frame as a plain dict with that dataclass's field
    names, so ``royaleviser.model.decode_frame`` reads the datagram without a converter and
    ``royaleviser.sources.TraceSource`` draws a trace with the same unit/spell rows. The
    field list is repeated here on purpose (../RoyaleViser/tests/test_sources.py round-trips a
    published frame through the viewer's decoder and its contract check, so a drift fails
    there). Positions stay in engine subtiles; ``units_per_tile`` says how many per tile.
"""

from __future__ import annotations

import os
import socket
import time
from collections.abc import Callable, Sequence
from typing import Any

import msgspec

from .protocol import (
    EMPTY_CARD,
    FIRER_TOWER,
    BattleState,
    CardInfo,
    EntityKind,
    EntityState,
    PlayerState,
    ProjectileState,
    SpellState,
    status_of,
)

HOST = "127.0.0.1"
PORT = 9870
HELLO = b"royaleviser 1"  # the viewer's heartbeat (royaleviser.sources.STREAM_HELLO)
ATTACH_TIMEOUT_S = 3.0  # no heartbeat for this long: detached, nothing is sent
POLL_INTERVAL_S = 1.0  # how often the socket is looked at for heartbeats
MAX_DATAGRAM = 65507
ENV_VAR = "ROYALEVISER"  # host:port; read once by ClashParallelEnv when no publisher is given
RUN_ENV_VAR = "ROYALEVISER_RUN"  # names the run whose frames these are (see from_env)
EVENTS_KEPT = 200  # event lines carried in every frame (newest last)

TOWER_NAMES = {EntityKind.KING_TOWER: "KingTower", EntityKind.PRINCESS_TOWER: "PrincessTower"}
TEAM_NAMES = ("Blue", "Red")


def names_of(cards: Sequence[CardInfo]) -> Callable[[int], str]:
    """card id -> name for one catalogue; unknown ids print as ``#<id>`` (never silent)."""
    table = {c.card_id: c.name for c in cards}
    return lambda card_id: table.get(card_id, f"#{card_id}")


def unit_dict(e: EntityState, name_of: Callable[[int], str]) -> dict[str, Any]:
    """royaleviser.model.Unit as a dict. Path is not in an engine state.

    ``footprint`` is the engine's own box for a building or tower, passed through as it
    came: None for a troop and for an engine that reports no box, so the viewer draws
    its marked stand-in rather than a box made up here.

    NOT REPORTED IS None, NEVER A VALUE. ``target``, ``direction`` and ``state`` were
    hard-coded None until the engine exported them (2026-09-24). An engine that still
    exports nothing decodes to EntityState's "did not say" defaults, and each maps back
    to None here: no target, a zero facing, and attack_phase -1. A zero facing in
    particular must not reach the viewer as [0, 0], because the viewer normalises the
    direction's length and a zero vector has none.

    ``status`` is the buff list as the engine names it, names whole: the viewer splits
    sim's "|"-joined names itself.
    """
    name = TOWER_NAMES.get(e.kind) if e.card_id == EMPTY_CARD else None
    return {
        "uid": e.uid,
        "team": e.team,
        "kind": e.kind,
        "name": name if name is not None else name_of(e.card_id),
        "x": e.x,
        "y": e.y,
        "hp": e.hp,
        "max_hp": e.max_hp,
        "radius": e.radius,
        "flying": e.flying,
        "deploy_ticks": e.deploy_ticks,
        "stun_ticks": e.stun_ticks,
        "target": e.target_uid if e.target_uid >= 0 else None,
        "path": [],
        "direction": list(e.facing) if e.facing != (0, 0) else None,
        "state": e.attack_phase if e.attack_phase >= 0 else None,
        "status": [[buff, ms_left] for buff, ms_left in e.buffs],
        "extra": {
            "tower_slot": e.tower_slot,
            "knockback_ticks": e.knockback_ticks,
            "shield": e.shield,
            # None when the engine did not report it (status_of); bits in protocol.STATUS_*
            "status_flags": status_of(e),
        },
        "footprint": list(e.footprint) if e.footprint is not None else None,
    }


def projectile_dict(p: ProjectileState, name_of: Callable[[int], str]) -> dict[str, Any]:
    """One projectile in flight, for the viewer. ENGINE frame, subtiles, like units.

    ``name`` is what fired it: the card, "tower" for a crown tower (firer -1), and None
    when the engine does not know (firer -2, a projectile restored from a snapshot older
    than the field). Sim keeps -2 apart from -1 so that "unknown" never reads as "a tower
    fired this", and mapping every negative id to "tower" would undo exactly that.
    """
    return {
        "team": p.team,
        "x": p.x,
        "y": p.y,
        "aim_x": p.aim_x,
        "aim_y": p.aim_y,
        "target": p.target_uid if p.target_uid >= 0 else None,
        "splash": p.splash,
        "name": (
            name_of(p.firer_card_id) if p.firer_card_id >= 0
            else "tower" if p.firer_card_id == FIRER_TOWER
            else None
        ),
    }


def spell_dict(s: SpellState, name_of: Callable[[int], str]) -> dict[str, Any]:
    return {
        "team": s.team,
        "name": name_of(s.card_id),
        "motion": s.motion,
        "x": s.x,
        "y": s.y,
        "aim_x": s.aim_x,
        "aim_y": s.aim_y,
        "extra": {
            "delay_ticks": s.delay_ticks,
            "travelled": s.travelled,
            "length": s.length,
            "hits": s.hits,
        },
    }


def player_dict(
    p: PlayerState, name_of: Callable[[int], str], deck: Sequence[int] | None
) -> dict[str, Any]:
    """royaleviser.model.Player: the engine knows everything but the cycle beyond next_card.

    An empty hand slot (EMPTY_CARD) is the empty string: known to be empty, not unknown.
    """
    return {
        "team": p.team,
        "elixir_milli": p.elixir_milli,
        "elixir_known": True,
        "hand": [name_of(c) if c != EMPTY_CARD else "" for c in p.hand],
        "hand_known": True,
        "next_card": name_of(p.next_card) if p.next_card != EMPTY_CARD else None,
        "cycle": [],
        "deck": [name_of(c) for c in deck] if deck is not None else [],
        "deck_known": deck is not None,
        "crowns": p.crowns,
        "tower_hp": list(p.tower_hp),
        "tower_max_hp": list(p.tower_max_hp),
        "king_active": p.king_active,
    }


def frame_dict(
    state: BattleState,
    name_of: Callable[[int], str],
    units_per_tile: int,
    decks: Sequence[Sequence[int]] | None = None,
    events: Sequence[str] = (),
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The viewer's Frame for one BattleState (see WIRE FORM)."""
    return {
        "tick": state.tick,
        "tick_ms": state.tick_ms,
        "units_per_tile": units_per_tile,
        "players": [
            player_dict(p, name_of, decks[p.team] if decks is not None else None)
            for p in state.players
        ],
        "units": [unit_dict(e, name_of) for e in state.entities],
        "spells": [spell_dict(s, name_of) for s in state.spells],
        "projectiles": [projectile_dict(p, name_of) for p in state.projectiles],
        "overtime": state.overtime,
        "game_over": state.game_over,
        "winner": state.winner,
        "crowns": [p.crowns for p in state.players],
        "events": list(events),
        "meta": dict(meta or {}),
    }


def play_event(tick: int, team: int, name: str, x: int, y: int, units_per_tile: int) -> str:
    """One events-panel line for an accepted deploy: "t120 Blue plays Knight (3.5, 14.5)"."""
    at = f"({_tiles(x, units_per_tile)}, {_tiles(y, units_per_tile)})"
    return f"t{tick} {TEAM_NAMES[team]} plays {name} {at}"


def _tiles(v: int, units_per_tile: int) -> str:
    """Integer units -> tiles rounded to a tenth, without a float."""
    tenths = (v * 10 + units_per_tile // 2) // units_per_tile
    return f"{tenths // 10}.{tenths % 10}"


class ViserPublisher:
    """Sends frames to an attached viewer; detached, ``publish`` is one clock read.

    ``port`` 0 binds an ephemeral port (tests); ``address`` is the bound (host, port).
    ``publish(state, cards, arena)`` builds the frame dict, adds the spawn/death lines it
    derives from the previous published state (only while attached, so a viewer that
    attaches mid-battle does not see every unit "spawn"), and sends it.
    ``publish_dict`` sends a ready dict (royaleviser.sources.Publisher's path).
    """

    def __init__(self, host: str = HOST, port: int = PORT, run: str = "") -> None:
        self.run = run
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.bind((host, port))
        except OSError as exc:
            # A BARE OSError HERE READS AS A BROKEN MACHINE AND MEANS "a run is already
            # streaming". On Windows it is WinError 10048, "Only one usage of each socket
            # address is normally permitted", which says nothing about training, about
            # another run, or about what to do. A reader meets it at the exact moment they
            # follow the published example with a run already going, and the published
            # example is what told them to use this port. Viser 1 reproduced it and found
            # it had also been read as "the example is broken" in a gate run.
            #
            # The port itself stays fixed, deliberately: discovery needs a known address,
            # so the reader sets one string and points a viewer at the same string. An
            # ephemeral port would need a discovery file, which is more machinery than
            # this should carry.
            self._sock.close()
            raise OSError(
                f"cannot stream to {host}:{port} because something is already using it, "
                "which is almost always another run of your own: ONE run holds this port "
                "for as long as it is streaming. A second run on the same machine needs "
                "its own, so set ROYALEVISER=127.0.0.1:9872 (any free port) and point the "
                f"viewer at the same number. The original error was: {exc}"
            ) from exc
        self._sock.setblocking(False)
        self.address: tuple[str, int] = self._sock.getsockname()[:2]
        self._peer: tuple[str, int] | None = None
        self._last_hello = 0.0
        self._last_poll = 0.0
        self.seq = 0
        self.sent = 0
        self.dropped = 0  # frames too big even without paths
        self._events: list[str] = []
        self._seen: dict[int, tuple[int, str]] = {}  # uid -> (team, name) of the last frame
        self._names: tuple[int, Callable[[int], str]] | None = None  # (id(cards), name_of)

    @classmethod
    def from_env(cls) -> ViserPublisher | None:
        """A publisher for ROYALEVISER=host:port, or None when the variable is unset/empty.

        ``ROYALEVISER_RUN`` names the run these frames come from. The ports are fixed, so
        two runs on one machine reach the same viewer and it cannot tell whose frames it
        is drawing beside whose learning panel; the name in every frame is what lets it
        say. Unset is the empty string, and then no ``run`` key is sent at all.
        """
        spec = os.environ.get(ENV_VAR, "").strip()
        if not spec:
            return None
        host, _, port = spec.rpartition(":")
        if not port.isdigit():
            raise ValueError(f"{ENV_VAR}={spec!r} is not host:port")
        return cls(host or HOST, int(port), os.environ.get(RUN_ENV_VAR, "").strip())

    def _poll(self, now: float) -> None:
        self._last_poll = now
        while True:
            try:
                data, addr = self._sock.recvfrom(64)
            except (BlockingIOError, ConnectionResetError, OSError):
                # WSAECONNRESET on Windows means an earlier send hit a closed viewer port.
                return
            if data == HELLO:
                self._peer = (addr[0], addr[1])
                self._last_hello = now

    @property
    def attached(self) -> bool:
        now = time.monotonic()
        if now - self._last_poll >= POLL_INTERVAL_S:
            self._poll(now)
        return self._peer is not None and now - self._last_hello < ATTACH_TIMEOUT_S

    def _name_of(self, cards: Sequence[CardInfo]) -> Callable[[int], str]:
        if self._names is None or self._names[0] != id(cards):
            self._names = (id(cards), names_of(cards))
        return self._names[1]

    def publish(
        self,
        state: BattleState,
        cards: Sequence[CardInfo],
        arena: Any,
        decks: Sequence[Sequence[int]] | None = None,
        events: Sequence[str] = (),
        meta: dict[str, Any] | None = None,
    ) -> bool:
        """One datagram for ``state`` if a viewer is attached. Returns whether one was sent."""
        if not self.attached:
            self._seen.clear()
            return False
        name_of = self._name_of(cards)
        self._note_units(state, name_of, arena.subtile)
        self._events.extend(events)
        del self._events[:-EVENTS_KEPT]
        m = {"source": "engine", "seq": self.seq, "wall_us": time.time_ns() // 1000}
        if self.run:
            m["run"] = self.run
        m.update(meta or {})
        d = frame_dict(state, name_of, arena.subtile, decks, self._events, m)
        return self.publish_dict(d)

    def _note_units(
        self, state: BattleState, name_of: Callable[[int], str], units_per_tile: int
    ) -> None:
        """Spawn/death lines by uid diffing with the previous published state (crown towers
        never spawn; a destroyed one is a death)."""
        now: dict[int, tuple[int, str]] = {}
        for e in state.entities:
            name = TOWER_NAMES.get(e.kind) if e.card_id == EMPTY_CARD else None
            now[e.uid] = (e.team, name if name is not None else name_of(e.card_id))
        if self._seen:
            for e in state.entities:
                if e.uid not in self._seen and e.card_id != EMPTY_CARD:
                    self._events.append(
                        f"t{state.tick} spawn {TEAM_NAMES[e.team]} {now[e.uid][1]}"
                        f" ({_tiles(e.x, units_per_tile)}, {_tiles(e.y, units_per_tile)})"
                    )
            for uid, (team, name) in self._seen.items():
                if uid not in now:
                    self._events.append(f"t{state.tick} death {TEAM_NAMES[team]} {name}")
        self._seen = now

    def publish_dict(self, d: dict[str, Any]) -> bool:
        """Send one frame dict to the attached viewer (the size policy lives here)."""
        if self._peer is None:
            return False
        data = msgspec.msgpack.encode(d)
        if len(data) > MAX_DATAGRAM:
            for u in d["units"]:
                u["path"] = []
            data = msgspec.msgpack.encode(d)
            if len(data) > MAX_DATAGRAM:
                self.dropped += 1
                return False
        try:
            self._sock.sendto(data, self._peer)
        except OSError:
            return False  # the viewer went away between its heartbeat and this send
        self.seq += 1
        self.sent += 1
        return True

    def close(self) -> None:
        self._sock.close()
        self._peer = None
