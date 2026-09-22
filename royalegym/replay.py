"""Battle traces: record, save, load, and VERIFY by re-simulation.

A trace is two things at once:
  * a replay in the sense calibration.json means -- seed + initial setup + the
    command log + periodic state hashes -- which ``verify_trace`` re-simulates
    and checks hash by hash; and
  * enough per-frame entity data (positions, hp, radius) for ``render.py`` to
    draw the battle as a self-contained HTML page without an engine
    (``python -m royalegym.render trace.msgpack -o battle.html``).

Encoding is msgspec msgpack (``.msgpack``, compact) or JSON (``.json``, for
eyeballing). Per-entity rows use ``array_like`` structs, so a frame is a list of
short arrays; the header names the columns so the viewer reads them by name
(render.py refuses a trace that lacks a column it draws).
State hashes are hex strings, because a u64 does not survive a JavaScript Number.

SPELLS (2026-09-13). A frame also records the engine's live spell objects
(``BattleState.spells``) and, through ``EntityState``, each entity's stun and
knockback timers; the header names the spell columns in ``spell_fields``. Both are
trailing fields with defaults, so a version-1 trace recorded before they existed still
decodes (with no spells and zero timers) and the version number did not move. A
MockEngine trace has no spell objects: its spells resolve inside a tick.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgspec

from .protocol import (
    CardInfo,
    DeployCommand,
    DeployResult,
    Engine,
    EntityState,
    MatchSetup,
    SpellState,
)

if TYPE_CHECKING:
    from .state_mutator import Snapshot

TRACE_FORMAT = "royalegym-trace"
TRACE_VERSION = 1


class TraceStep(msgspec.Struct, array_like=True):
    tick: int  # engine tick before the commands were applied
    ticks: int  # ticks requested
    commands: list[DeployCommand]
    statuses: list[int]


class TraceFrame(msgspec.Struct, array_like=True):
    tick: int
    entities: list[EntityState]
    elixir_milli: list[int]
    crowns: list[int]
    hands: list[list[int]]
    state_hash: str
    spells: list[SpellState] = []  # live spell objects (trailing: older traces decode)


class TraceHeader(msgspec.Struct):
    format: str
    version: int
    seed: int
    setup: MatchSetup | None
    snapshot: bytes | None
    frame_every_tick: bool
    tick_ms: int
    subtile: int
    tiles_x: int
    tiles_y: int
    half: int
    grid: list[list[int]]
    bridges_half_cols: list[tuple[int, int]]
    water_half_rows: tuple[int, int]
    cards: list[CardInfo]
    entity_fields: list[str]
    frame_fields: list[str]
    step_fields: list[str]
    spell_fields: list[str] = []  # SpellState columns; empty in a pre-spell trace


class TraceResult(msgspec.Struct):
    winner: int
    crowns: list[int]
    final_tick: int
    final_hash: str


class Trace(msgspec.Struct):
    header: TraceHeader
    steps: list[TraceStep]
    frames: list[TraceFrame]
    result: TraceResult | None = None


def _hex(h: int) -> str:
    return f"{h & ((1 << 64) - 1):016x}"


def _frame(engine: Engine) -> TraceFrame:
    s = engine.state()
    return TraceFrame(
        tick=s.tick,
        entities=list(s.entities),
        elixir_milli=[p.elixir_milli for p in s.players],
        crowns=[p.crowns for p in s.players],
        hands=[list(p.hand) for p in s.players],
        state_hash=_hex(engine.state_hash()),
        spells=list(s.spells),
    )


class ReplayRecorder:
    """Attach to an env (``recorder=``). Keeps the most recent trace in ``trace``.

    ``frame_every_tick=True`` records a frame per engine tick (smooth viewing,
    and the tightest determinism check); False records one per decision.
    ``on_finish`` is called with each completed trace, e.g. to save it.
    """

    def __init__(self, frame_every_tick: bool = True, on_finish: Any = None) -> None:
        self.frame_every_tick = frame_every_tick
        self.on_finish = on_finish
        self.trace: Trace | None = None
        self._open = False

    def begin(self, engine: Engine, seed: int, init: MatchSetup | Snapshot) -> None:
        a = engine.arena()
        setup = init if isinstance(init, MatchSetup) else None
        snapshot = None if setup is not None else init.blob  # type: ignore[union-attr]
        header = TraceHeader(
            format=TRACE_FORMAT,
            version=TRACE_VERSION,
            seed=seed,
            setup=setup,
            snapshot=snapshot,
            frame_every_tick=self.frame_every_tick,
            tick_ms=engine.state().tick_ms,
            subtile=a.subtile,
            tiles_x=a.tiles_x,
            tiles_y=a.tiles_y,
            half=a.half,
            grid=a.grid,
            bridges_half_cols=list(a.bridges_half_cols),
            water_half_rows=a.water_half_rows,
            cards=list(engine.cards()),
            entity_fields=list(EntityState.__struct_fields__),
            frame_fields=list(TraceFrame.__struct_fields__),
            step_fields=list(TraceStep.__struct_fields__),
            spell_fields=list(SpellState.__struct_fields__),
        )
        self.trace = Trace(header=header, steps=[], frames=[_frame(engine)])
        self._open = True

    def record_frame(self, engine: Engine) -> None:
        if self._open and self.trace is not None:
            self.trace.frames.append(_frame(engine))

    def record_step(
        self,
        tick: int,
        ticks: int,
        commands: list[DeployCommand],
        results: list[DeployResult],
    ) -> None:
        if self._open and self.trace is not None:
            self.trace.steps.append(
                TraceStep(tick, ticks, list(commands), [r.status for r in results])
            )

    def end(self, engine: Engine) -> Trace | None:
        if not self._open or self.trace is None:
            return None
        s = engine.state()
        self.trace.result = TraceResult(
            winner=s.winner,
            crowns=[p.crowns for p in s.players],
            final_tick=s.tick,
            final_hash=_hex(engine.state_hash()),
        )
        self._open = False
        if self.on_finish is not None:
            self.on_finish(self.trace)
        return self.trace


def save_trace(trace: Trace, path: str | Path) -> Path:
    p = Path(path)
    data = msgspec.json.encode(trace) if p.suffix == ".json" else msgspec.msgpack.encode(trace)
    p.write_bytes(data)
    return p


def load_trace(path: str | Path) -> Trace:
    p = Path(path)
    raw = p.read_bytes()
    if p.suffix == ".json":
        return msgspec.json.decode(raw, type=Trace)
    return msgspec.msgpack.decode(raw, type=Trace)


def verify_trace(trace: Trace, engine: Engine) -> list[str]:
    """Re-simulate ``trace`` on ``engine`` and list every divergence (empty = identical).

    Checks the deploy statuses of every step, every recorded frame hash, and the
    final hash. The first divergence is usually the informative one; later ones
    are consequences.
    """
    h = trace.header
    problems: list[str] = []
    if h.setup is not None:
        engine.reset(h.seed, h.setup)
    elif h.snapshot is not None:
        engine.load_state(h.snapshot)
    else:
        return ["trace has neither setup nor snapshot"]
    frames = iter(trace.frames)

    def check_frame() -> None:
        want = next(frames, None)
        if want is None:
            problems.append(f"tick {engine.state().tick}: more frames simulated than recorded")
            return
        got = _hex(engine.state_hash())
        if got != want.state_hash:
            problems.append(f"tick {want.tick}: hash {got} != recorded {want.state_hash}")

    check_frame()
    for i, st in enumerate(trace.steps):
        if engine.state().tick != st.tick:
            problems.append(f"step {i}: engine at tick {engine.state().tick}, trace says {st.tick}")
        if h.frame_every_tick:
            results = engine.step(st.commands, 1)
            check_frame()
            for _ in range(st.ticks - 1):
                if engine.state().game_over:
                    break
                engine.step([], 1)
                check_frame()
        else:
            results = engine.step(st.commands, st.ticks)
            check_frame()
        got_status = [r.status for r in results]
        if got_status != st.statuses:
            problems.append(f"step {i}: statuses {got_status} != recorded {st.statuses}")
    if trace.result is not None and _hex(engine.state_hash()) != trace.result.final_hash:
        problems.append("final hash differs")
    return problems
