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

import os
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
    # The card table the recording engine read (its ``card_table_stamp()``). An engine
    # fills the ones it states; "" is "not stated", which is also what an older trace
    # decodes with. RustEngine: the first three. MockEngine: vintage and loaded hash.
    cards_vintage: str = ""
    cards_json_fnv1a64: str = ""
    cards_json_hash_source: str = ""
    cards_loaded_fnv1a64: str = ""


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


# The TraceHeader fields an engine's ``card_table_stamp()`` can fill.
CARD_TABLE_FIELDS = (
    "cards_vintage",
    "cards_json_fnv1a64",
    "cards_json_hash_source",
    "cards_loaded_fnv1a64",
)


def _card_table_fields(engine: Engine) -> dict[str, str]:
    """The header's card-table fields, from an engine that states its table.

    ``card_table_stamp`` is not part of ``protocol.Engine``, so an engine without one
    records a trace with the fields left "".
    """
    stamp = getattr(engine, "card_table_stamp", None)
    if stamp is None:
        return {}
    return {k: str(v) for k, v in stamp().items() if k in CARD_TABLE_FIELDS}


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
            **_card_table_fields(engine),
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


class SavingReplayRecorder(ReplayRecorder):
    """A recorder that WRITES completed traces, so a rollout worker can produce them.

    WHY THIS EXISTS RATHER THAN ``on_finish``
        ``ReplayRecorder`` already takes ``on_finish``, and for a script that is the better
        tool. It is unreachable from a training run: the recorder is an ``EnvFactory``
        COMPONENT, given as an importable class plus JSON kwargs, and a callable is not
        something JSON can express. So a worker that gets a pickled recipe can be handed
        this class and a directory, and cannot be handed a function. Every argument here is
        a string, an int or a bool for that reason.

        Asked for by train, who had built and measured the rest: a 3-minute battle is 3,601
        frames and 1.6 MB on disk, and their viewer plays one to a real viewer at wall clock
        in 57 MB. Watching a live run cannot work -- `time/collection` is 4.35 s of a 62.94 s
        iteration, and those 4.35 seconds hold about 82,000 engine ticks, so it is a firehose
        between silences rather than slow gameplay. Replaying a saved trace is the form that
        works, and the training run producing them was the one missing piece.

    WHAT IT COSTS THE RUN
        One write every ``every`` completed episodes, and nothing else. Recording itself is
        whatever ``frame_every_tick`` already costs; this adds serialisation of a finished
        trace, not per-tick work.

    THREE THINGS IT DOES THAT A NAIVE VERSION WOULD NOT, each because a rollout worker is
    not a script:
        * **The PID is in the name.** Workers are separate PROCESSES with separate counters,
          so two of them would otherwise write the same ``000050`` file and one would win.
        * **The write is atomic**, via a temporary file and ``os.replace``. A viewer reading
          the directory while a worker writes would otherwise find a half-written trace, and
          msgpack does not fail loudly on one.
        * **The directory is created and tested at CONSTRUCTION**, so a bad path fails when
          the config is loaded rather than after the first episode of a five-hour run. A
          recorder that silently saves nothing is the failure train hit from the other side:
          a publisher that reported "3601 frames, 0 sent" and looked like it worked.
    """

    def __init__(
        self,
        out_dir: str,
        every: int = 50,
        keep: int = 4,
        frame_every_tick: bool = True,
    ) -> None:
        super().__init__(frame_every_tick=frame_every_tick)
        if every < 1:
            raise ValueError(f"every must be >= 1, got {every}: 0 would save nothing quietly")
        if keep < 1:
            raise ValueError(f"keep must be >= 1, got {keep}: 0 would delete what it just wrote")
        self.out_dir = Path(out_dir)
        self.every = every
        self.keep = keep
        # Fail here, not after the first episode. mkdir raises on an unusable path.
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.completed = 0
        self.saved: list[Path] = []

    def end(self, engine: Engine) -> Trace | None:
        trace = super().end(engine)
        if trace is None:
            return None
        self.completed += 1
        if self.completed % self.every == 0:
            self.saved.append(self._write(trace))
            self._prune()
        return trace

    def _write(self, trace: Trace) -> Path:
        """Atomically, because a viewer may be reading this directory as we write."""
        tick = trace.result.final_tick if trace.result is not None else 0
        name = f"{os.getpid()}-{self.completed:06d}-tick{tick}.msgpack"
        final = self.out_dir / name
        tmp = final.with_suffix(".msgpack.part")
        tmp.write_bytes(msgspec.msgpack.encode(trace))
        os.replace(tmp, final)
        return final

    def _prune(self) -> None:
        """Keep the newest ``keep`` of OUR OWN files.

        Only this process's, because another worker's traces are not ours to delete and a
        directory shared by eight workers would otherwise have each of them deleting the
        others' newest files.
        """
        while len(self.saved) > self.keep:
            old = self.saved.pop(0)
            old.unlink(missing_ok=True)
