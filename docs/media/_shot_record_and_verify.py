#!/usr/bin/env python3
r"""Showcase tile: record a battle, re-run it on a fresh engine, compare every hash.

Everything on the figure is measured in the run that draws it: the battle is recorded
with ``ReplayRecorder``, saved with ``save_trace``, and then handed to ``verify_trace``
together with a brand-new engine. The hashes shown are the real ones, read out of the
recorded trace and out of the re-simulation as it happens.
"""

from __future__ import annotations

import pathlib
import tempfile

import make_media as M

SEED = 0
DECK = ("Knight", "Archer", "Giant", "Minions", "Fireball", "Cannon", "Zap", "Musketeer")
NOOP_PROB = 0.7
HASH_CHARS = 8  # how much of each 16-hex-digit hash the tile shows


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
    from royalegym import (ClashParallelEnv, DefaultStateMutator, RandomLegalOpponent,
                           RustEngine)
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
    """Record, save, verify, then draw the comparison. Returns the numbers on the tile."""
    from royalegym import RustEngine
    from royalegym.replay import load_trace, verify_trace

    with tempfile.TemporaryDirectory() as tmp:
        trace_path = pathlib.Path(tmp) / "battle.msgpack"
        _record(trace_path)
        size_bytes = trace_path.stat().st_size
        trace = load_trace(trace_path)  # round-trip: verify what came back off disk

    watched = _WatchedEngine(RustEngine())
    problems = verify_trace(trace, watched)

    recorded = [f.state_hash for f in trace.frames]
    rerun = [f"{h & ((1 << 64) - 1):016x}" for h in watched.hashes[:len(recorded)]]
    n = min(len(recorded), len(rerun))
    # The headline counts hash comparisons only; verify_trace also checks every deploy
    # status, and its whole return value is printed underneath, so nothing hides.
    n_diff = sum(1 for i in range(n) if recorded[i] != rerun[i])
    ok = n_diff == 0 and not problems
    final_tick = trace.result.final_tick
    mb = size_bytes / 1_000_000

    rows_idx = [0, n // 2, n - 1]
    rows = [(trace.frames[i].tick, recorded[i][:HASH_CHARS], rerun[i][:HASH_CHARS])
            for i in rows_idx]

    # ---------------------------------------------------------------- draw
    W, H = 1000, 620
    im, d = M.canvas(W, H)
    pad = 40
    inner = W - 2 * pad

    head_n, head_rest = f"{n_diff}", f" of {n} hashes differ"
    f_head = _fit(d, head_n + head_rest, 74, inner)
    y = 34
    d.text((pad, y), head_n, font=f_head, fill=M.GREEN if ok else M.RED)
    d.text((pad + d.textlength(head_n, font=f_head), y), head_rest, font=f_head, fill=M.TEXT)
    y += f_head.size + 16

    f_sub = _fit(d, "re-run on a fresh engine", 40, inner)
    d.text((pad, y), "re-run on a fresh engine", font=f_sub, fill=M.DIM)
    y += f_sub.size + 34

    # table
    f_lbl = M.theme_font(38)
    d.text((pad, y), f"{len(rows)} of {n} frames shown", font=f_lbl, fill=M.DIM)
    y += 50

    panel_h = 50 + len(rows) * 68 + 22
    d.rounded_rectangle([pad, y, W - pad, y + panel_h], radius=14, fill=M.PANEL,
                        outline=M.BORDER, width=2)
    f_mono = M.theme_font(42, mono=True)
    cw = d.textlength("0", font=f_mono)
    x_tick = pad + 26
    x_rec = x_tick + cw * 5.4
    x_run = x_rec + cw * (HASH_CHARS + 1.6)
    x_ok = x_run + cw * (HASH_CHARS + 1.4)

    ty = y + 14
    d.text((x_tick, ty), "tick", font=f_lbl, fill=M.DIM)
    d.text((x_rec, ty), "recorded", font=f_lbl, fill=M.DIM)
    d.text((x_run, ty), "re-run", font=f_lbl, fill=M.DIM)
    ty += 56

    for tick, a, b in rows:
        d.text((x_tick, ty), str(tick), font=f_mono, fill=M.TEXT)
        d.text((x_rec, ty), a, font=f_mono, fill=M.BLUE)
        d.text((x_run, ty), b, font=f_mono, fill=M.BLUE if a == b else M.RED)
        if a == b:  # a drawn tick, so no font has to own the glyph
            cx, cy = x_ok + 14, ty + 26
            d.line([(cx, cy), (cx + 12, cy + 14), (cx + 34, cy - 18)],
                   fill=M.GREEN, width=7, joint="curve")
        else:
            d.line([(x_ok + 4, ty + 6), (x_ok + 36, ty + 38)], fill=M.RED, width=7)
            d.line([(x_ok + 36, ty + 6), (x_ok + 4, ty + 38)], fill=M.RED, width=7)
        ty += 68

    y += panel_h + 30
    foot = f"verify_trace() -> {problems!r}  ·  {mb:.1f} MB trace"
    f_foot = _fit(d, foot, 38, inner, mono=True)
    d.text((pad, y), foot, font=f_foot, fill=M.TEXT if ok else M.RED)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)
    return (f"{n} frame hashes compared, {n_diff} differed; verify_trace returned "
            f"{problems!r}; battle seed {SEED} to tick {final_tick}, "
            f"trace {size_bytes} bytes ({mb:.1f} MB); rows at ticks "
            f"{[r[0] for r in rows]}")
