#!/usr/bin/env python3
r"""Showcase tile: record a battle, re-run it, and show that the hash check can fail.

Everything on the figure is measured in the run that draws it. One battle is recorded with
``ReplayRecorder``, written with ``save_trace``, read back with ``load_trace``, and handed
to ``verify_trace`` together with a brand-new engine; the hashes drawn are the real ones,
read out of the trace and out of the re-simulation as it happens.

The second strip is the control. The same trace is re-run again with one thing changed --
the first deploy command in the log is moved one tile sideways -- and the figure shows
which tick hashes that changes. Without it, a row of green checks would look the same whether the
comparison was strict or vacuous. The perturbation is chosen by a fixed rule (first command,
one tile, toward the middle of the arena), not searched for, and whatever it produces is
what gets drawn.
"""

from __future__ import annotations

import pathlib
import tempfile

import make_media as M

SEED = 0
DECK = ("Knight", "Archer", "Giant", "Minions", "Fireball", "Cannon", "Zap", "Musketeer")
NOOP_PROB = 0.7
HASH_CHARS = 8  # how much of each 16-hex-digit hash the tile shows, if that much separates them


class _WatchedEngine:
    """A real engine that also keeps every ``state_hash`` ``verify_trace`` asks it for.

    ``verify_trace`` returns only the divergences, so this is how the re-run's own hash
    values reach the figure. Every call is forwarded untouched; nothing is faked.
    """

    def __init__(self, inner):
        self._inner = inner
        self.hashes: list[int] = []

    def state_hash(self) -> int:
        h = self._inner.state_hash()
        self.hashes.append(h)
        return h

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _record(path: pathlib.Path):
    """Play one whole battle at a fixed seed and deck, one recorded frame per tick."""
    import numpy as np

    from royalegym import ClashParallelEnv, DefaultStateMutator, RandomLegalOpponent, RustEngine
    from royalegym.replay import ReplayRecorder, save_trace

    engine = RustEngine()
    by_name = {c.name: c.card_id for c in engine.cards()}
    deck = [by_name[n] for n in DECK]
    rec = ReplayRecorder(frame_every_tick=True)
    env = ClashParallelEnv(engine=engine, recorder=rec,
                           state_mutator=DefaultStateMutator(decks=[deck, deck]))
    obs, _ = env.reset(seed=SEED)
    rng, policy = np.random.default_rng(SEED), RandomLegalOpponent(noop_prob=NOOP_PROB)
    while env.agents:
        obs, *_ = env.step({a: policy.act(obs[a], obs[a]["action_mask"], rng)
                            for a in env.agents})
    save_trace(rec.trace, path)
    return rec.trace


def _rerun(trace):
    """``verify_trace`` on a fresh engine. Returns its hashes as hex, and its complaints.

    The one thing that must not happen quietly is a short re-run: ``verify_trace`` guards
    the "more frames simulated than recorded" direction but not the other one, and fewer
    hashes coming back would otherwise shrink the denominator instead of failing.
    """
    from royalegym import RustEngine
    from royalegym.replay import verify_trace

    watched = _WatchedEngine(RustEngine())
    problems = verify_trace(trace, watched)
    hexes = [f"{h & ((1 << 64) - 1):016x}" for h in watched.hashes]
    if len(hexes) < len(trace.frames):
        raise RuntimeError(f"re-run produced {len(hexes)} hashes for {len(trace.frames)} "
                           "recorded frames")
    return hexes[:len(trace.frames)], problems


def _moved_copy(trace):
    """A copy of the trace with its first deploy command moved one tile sideways.

    Fixed rule, no search: the first command in the log, one tile toward the middle of the
    arena so the target stays on the board. Returns the copy, the tick that command is
    played on, and how far it moved in whole tiles, measured off the copy rather than assumed.
    """
    import msgspec

    copy = msgspec.msgpack.decode(msgspec.msgpack.encode(trace), type=type(trace))
    step = next(s for s in copy.steps if s.commands)
    cmd = step.commands[0]
    sub = copy.header.subtile
    middle = (copy.header.tiles_x * sub) // 2
    dx = sub if cmd.x < middle else -sub
    step.commands = [msgspec.structs.replace(cmd, x=cmd.x + dx), *step.commands[1:]]
    return copy, step.tick, abs(step.commands[0].x - cmd.x) // sub


def _runs(idx: list[int]):
    """Consecutive integers grouped into ``(first, last)`` runs."""
    out: list[list[int]] = []
    for i in idx:
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return [(a, b) for a, b in out]


def _cut(a: str, b: str) -> int:
    """How many leading hex digits to show so two different hashes look different."""
    if a == b:
        return HASH_CHARS
    first = next(i for i in range(len(a)) if a[i] != b[i])
    return max(HASH_CHARS, first + 1)


def _fit(d, text, font_size, max_w, mono=False):
    """Largest font at or below ``font_size`` whose ``text`` fits ``max_w``."""
    size = font_size
    while size > 34:
        f = M.theme_font(size, mono=mono)
        if d.textlength(text, font=f) <= max_w:
            return f
        size -= 2
    return M.theme_font(size, mono=mono)


def draw(out_path: pathlib.Path) -> str:
    """Record, re-run twice (faithfully and with one deploy moved), then draw the two."""
    from royalegym.replay import load_trace

    with tempfile.TemporaryDirectory() as tmp:
        trace_path = pathlib.Path(tmp) / "battle.msgpack"
        _record(trace_path)
        size_bytes = trace_path.stat().st_size
        trace = load_trace(trace_path)  # everything below is what came back off disk

    recorded = [f.state_hash for f in trace.frames]
    n = len(recorded)
    same, problems = _rerun(trace)
    moved_trace, moved_tick, moved_tiles = _moved_copy(trace)
    moved, moved_problems = _rerun(moved_trace)

    # Full 16-digit comparisons. The tile truncates only for drawing, and _cut keeps the
    # truncation from ever showing two different hashes as the same eight characters.
    same_diff = [i for i in range(n) if same[i] != recorded[i]]
    moved_diff = [i for i in range(n) if moved[i] != recorded[i]]
    ok = not same_diff and not problems
    deploys = sum(len(s.commands) for s in trace.steps)
    decks = trace.header.setup.decks
    ticks = [f.tick for f in trace.frames]

    # The row of hashes is the first tick where the moved run parts from the recording --
    # a rule, not a pick, and the tile says which tick it is.
    row_i = moved_diff[0] if moved_diff else n - 1
    row_tick = ticks[row_i]
    k = _cut(recorded[row_i], moved[row_i])

    # ---------------------------------------------------------------- draw
    W, H = 1000, 640
    im, d = M.canvas(W, H)
    pad = 40
    inner = W - 2 * pad

    head_n, head_rest = f"{len(same_diff)}", f" of {n} hashes differ"
    f_head = _fit(d, head_n + head_rest, 68, inner)
    y = 22
    d.text((pad, y), head_n, font=f_head, fill=M.GREEN if ok else M.RED)
    d.text((pad + d.textlength(head_n, font=f_head), y), head_rest, font=f_head, fill=M.TEXT)
    y += f_head.size + 10

    # Two lines that the strips below cannot say for themselves: what one strip covers, and
    # what the recording held in the first place.
    for text, size in ((f"each strip: one battle, tick {ticks[0]} to {ticks[-1]}, "
                        "a hash every tick", 36),
                       (f"the recording holds the seed, {len(decks)}x{len(decks[0])} decks "
                        f"and {deploys} deploys", 34)):
        f = _fit(d, text, size, inner)
        d.text((pad, y), text, font=f, fill=M.DIM)
        y += f.size + 6
    y += 14

    # ---- the two strips: every tick of the battle, in order, left to right
    f_lbl = M.theme_font(34)
    bar_h = 26

    def strip(label: str, diff: list[int], tint):
        nonlocal y
        d.text((pad, y), label, font=f_lbl, fill=M.TEXT)
        count = f"{len(diff)} differ"
        d.text((W - pad - d.textlength(count, font=f_lbl), y), count, font=f_lbl, fill=tint)
        y += f_lbl.size + 6
        d.rounded_rectangle([pad, y, W - pad, y + bar_h], radius=7, fill=M.GREEN)
        for a, b in _runs(diff):
            x0 = pad + inner * a / n
            x1 = max(x0 + 7, pad + inner * (b + 1) / n)  # a short run still has to be seen
            d.rectangle([x0, y, x1, y + bar_h], fill=M.RED)
        y += bar_h

    strip("same commands", same_diff, M.GREEN if ok else M.RED)
    y += 22
    strip(f"one deploy moved {moved_tiles} tile", moved_diff, M.RED if moved_diff else M.DIM)
    if moved_diff:
        runs = _runs(moved_diff)
        mx = pad + inner * moved_diff[0] / n  # a pointer, so a short run is not widened to be seen
        d.polygon([(mx - 11, y + 16), (mx + 11, y + 16), (mx, y + 2)], fill=M.RED)
        span = (f"differs at ticks {ticks[runs[0][0]]}-{ticks[runs[0][1]]}, then matches "
                "again to the end"
                if len(runs) == 1 and runs[0][1] < n - 1
                else f"differs from tick {ticks[moved_diff[0]]} on")
    else:
        span = "the moved deploy changed no hash at all"
    y += 18
    f_span = _fit(d, span, 34, inner)
    d.text((pad, y), span, font=f_span, fill=M.TEXT)
    y += f_span.size + 16

    # ---- the hashes themselves, at the tick the two re-runs disagree on
    f_mono = M.theme_font(40, mono=True)
    rows = [("same commands", same[row_i]), (f"moved {moved_tiles} tile", moved[row_i])]
    panel_h = 16 + f_lbl.size + 8 + len(rows) * 52 + 8
    d.rounded_rectangle([pad, y, W - pad, y + panel_h], radius=14, fill=M.PANEL,
                        outline=M.BORDER, width=2)
    cw = d.textlength("0", font=f_mono)
    x_lbl = pad + 22
    x_rec = pad + 320
    x_run = x_rec + cw * (k + 2.4)
    ty = y + 14
    d.text((x_lbl, ty), f"at tick {row_tick}", font=f_lbl, fill=M.TEXT)
    d.text((x_rec, ty), "recorded", font=f_lbl, fill=M.TEXT)
    d.text((x_run, ty), "re-run", font=f_lbl, fill=M.TEXT)
    ty += f_lbl.size + 8
    for label, got in rows:
        agrees = got == recorded[row_i]
        d.text((x_lbl, ty), label, font=f_lbl, fill=M.DIM)
        d.text((x_rec, ty), recorded[row_i][:k], font=f_mono, fill=M.BLUE)
        d.text((x_run, ty), got[:k], font=f_mono, fill=M.BLUE if agrees else M.RED)
        mx, my = x_run + cw * (k + 1.2), ty + 8
        if agrees:  # drawn, so no font has to own the glyph
            d.line([(mx, my + 18), (mx + 12, my + 32), (mx + 34, my)],
                   fill=M.GREEN, width=7, joint="curve")
        else:
            d.line([(mx, my), (mx + 32, my + 32)], fill=M.RED, width=7)
            d.line([(mx + 32, my), (mx, my + 32)], fill=M.RED, width=7)
        ty += 52
    y += panel_h + 12

    foot = f"verify_trace() -> {problems!r} · {len(moved_problems)} lines once moved"
    f_foot = _fit(d, foot, 34, inner, mono=True)
    d.text((pad, y), foot, font=f_foot, fill=M.TEXT if ok else M.RED)
    if y + f_foot.size > H:  # a layout that ran off the bottom is not a figure
        raise RuntimeError(f"the tile needs {y + f_foot.size} px of {H}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)
    return (f"{n} frame hashes compared, {len(same_diff)} differed; verify_trace returned "
            f"{problems!r}; battle from reset seed {SEED} (trace header seed "
            f"{trace.header.seed}) to tick {trace.result.final_tick}, {deploys} deploys, "
            f"trace {size_bytes} bytes; moving the deploy at tick {moved_tick} by "
            f"{moved_tiles} tile changed {len(moved_diff)} hashes and drew "
            f"{len(moved_problems)} verify_trace lines; hash row at tick {row_tick}")
