#!/usr/bin/env python3
"""The four state mutators the repo ships, each one actually reset and measured.

Everything on the picture is computed in ``draw``. One environment per mutator is built
on a fresh engine, ``reset(seed=0)`` once, and the board it lands on is read straight
back off ``env.battle_state``: the clock, the hit points left on all six crown towers,
and how many non-tower entities are standing. Nothing is typed in.

The snapshot row is a real save. A default battle is played 120 env steps, the engine is
asked for its bytes and its state hash, and ``SnapshotStateMutator`` loads those bytes
into a brand new engine; the line under the table reports whether the reloaded hash came
back equal, which is the only evidence on the picture that "exact" is the right word.

The mid-game row is ONE draw from the ranges that mutator is configured with, not a
picked-out example: its tick, elixir and per-tower damage are sampled from the env's
seeded generator.
"""

from __future__ import annotations

import pathlib

import make_media as M
import numpy as np

from royalegym import ClashParallelEnv, RandomLegalOpponent, RustEngine
from royalegym.protocol import BLUE, RED, SpawnSpec
from royalegym.state_mutator import (
    DefaultStateMutator,
    MidGameStateMutator,
    ScriptedBoardStateMutator,
    SnapshotStateMutator,
)

# The deck and seed the README's Try-it battle uses, so this figure is that battle.
DECK = ("Knight", "Archer", "Giant", "Minions", "Fireball", "Cannon", "Zap", "Musketeer")
SEED = 0

LEAD_IN_STEPS = 120      # env steps played before the snapshot is taken
NOOP_PROB = 0.55         # chance a seat passes on a lead-in step

# MidGameStateMutator's ranges. Every number the row shows is drawn from these by the
# env's seeded generator, so the row is a sample of the curriculum, not a chosen board.
MID_TICKS = (1200, 2400)
MID_ELIXIR = (3000, 9000)
MID_HP_PCT = (35, 85)
MID_DOWN_PROB = 0.34

SCRIPT_ELIXIR = 4000     # the hand-built board's starting elixir, both seats


# --------------------------------------------------------------------------- measuring


def _deck(engine: RustEngine) -> list[int]:
    """The example deck, looked up BY NAME: a card id is a position in a catalogue."""
    ids = {c.name: c.card_id for c in engine.cards()}
    missing = [n for n in DECK if n not in ids]
    if missing:
        raise RuntimeError(f"this build's catalogue has no {missing}")
    return [ids[n] for n in DECK]


def _facts(env: ClashParallelEnv) -> dict[str, int]:
    """What the board looks like the instant reset() returns."""
    s = env.battle_state
    return {
        "tick": s.tick,
        "tick_ms": s.tick_ms,
        "hp": sum(sum(p.tower_hp) for p in s.players),
        "units": sum(1 for e in s.entities if e.tower_slot < 0),
    }


def _reset(mutator) -> tuple[ClashParallelEnv, dict[str, int]]:
    env = ClashParallelEnv(engine=RustEngine(), state_mutator=mutator)
    env.reset(seed=SEED)
    return env, _facts(env)


def _tile_center(arena, tx: int, ty: int) -> tuple[int, int]:
    return tx * arena.subtile + arena.subtile // 2, ty * arena.subtile + arena.subtile // 2


def _hand_built(engine: RustEngine, deck: list[int]) -> ScriptedBoardStateMutator:
    """A defensive drill: a push already walking in, and two defenders to meet it."""
    ids = {c.name: c.card_id for c in engine.cards()}
    arena = engine.arena()
    spawns = [
        SpawnSpec(BLUE, ids["Knight"], *_tile_center(arena, 3, 12)),
        SpawnSpec(BLUE, ids["Musketeer"], *_tile_center(arena, 4, 10)),
        SpawnSpec(RED, ids["Giant"], *_tile_center(arena, 3, 19)),
        SpawnSpec(RED, ids["Minions"], *_tile_center(arena, 5, 20)),
    ]
    return ScriptedBoardStateMutator(
        spawns=spawns, decks=[deck, deck],
        elixir_milli=[SCRIPT_ELIXIR, SCRIPT_ELIXIR],
    )


def _play_in(deck: list[int]) -> tuple[bytes, int, dict[str, int]]:
    """Play a default battle a while, then hand back its bytes and its state hash."""
    engine = RustEngine()
    env = ClashParallelEnv(engine=engine, state_mutator=DefaultStateMutator(decks=[deck, deck]))
    obs, _ = env.reset(seed=SEED)
    rng, policy = np.random.default_rng(SEED), RandomLegalOpponent(noop_prob=NOOP_PROB)
    for _ in range(LEAD_IN_STEPS):
        if not env.agents:
            break
        obs, *_ = env.step({a: policy.act(obs[a], obs[a]["action_mask"], rng)
                            for a in env.agents})
    return engine.save_state(), engine.state_hash(), _facts(env)


def _clock(tick: int, tick_ms: int) -> str:
    secs = round(tick * tick_ms / 1000)
    return f"{secs // 60}:{secs % 60:02d}"


# --------------------------------------------------------------------------- drawing


def _right(d, x: int, y: int, text: str, font, fill) -> None:
    d.text((x - d.textlength(text, font=font), y), text, font=font, fill=fill)


def draw(out_path: pathlib.Path) -> str:
    """Reset one env per mutator, draw what each one landed on, save it."""
    # ---- measure -------------------------------------------------------------
    probe = RustEngine()
    deck = _deck(probe)

    fresh_env, fresh = _reset(DefaultStateMutator(decks=[deck, deck]))
    full_hp = fresh["hp"]      # a fresh start is the 100% the other rows are read against
    fresh_towers = list(fresh_env.battle_state.players[BLUE].tower_hp)

    _, mid = _reset(MidGameStateMutator(
        tick_range=MID_TICKS, elixir_milli_range=MID_ELIXIR,
        max_tower_hp=(fresh_towers[0], fresh_towers[1]),
        tower_hp_percent=MID_HP_PCT, tower_down_prob=MID_DOWN_PROB,
        decks=[deck, deck]))

    _, scripted = _reset(_hand_built(probe, deck))

    blob, saved_hash, saved = _play_in(deck)
    snap_env, snapshot = _reset(SnapshotStateMutator([blob]))
    hash_matched = snap_env.engine.state_hash() == saved_hash

    rows = [
        ("Default", fresh),
        ("MidGame", mid),
        ("ScriptedBoard", scripted),
        ("Snapshot", snapshot),
    ]
    for _, f in rows:
        f["pct"] = round(100 * f["hp"] / full_hp)
        f["clock"] = _clock(f["tick"], f["tick_ms"])
    n = len(rows)
    distinct = len({(f["tick"], f["hp"], f["units"]) for _, f in rows})

    # ---- draw ----------------------------------------------------------------
    # Sized for a three-across README table: nothing under 34 px, headline 68 px.
    W, H = 1000, 640
    im, d = M.canvas(W, H)
    f_head = M.theme_font(68)
    f_sub = M.theme_font(37)
    f_col = M.theme_font(34)
    f_name = M.theme_font(43)
    f_num = M.theme_font(47)
    f_foot = M.theme_font(36)

    L, R = 40, W - 40
    d.text((L, 18), f"{n} starts, {distinct} different boards", font=f_head, fill=M.TEXT)
    d.text((L, 102), f"one reset each, seed {SEED}", font=f_sub, fill=M.DIM)

    x_name, x_clock = L, 500
    bar0, bar1, x_pct = 536, 742, 890
    x_units = R

    hy = 178
    d.text((x_name, hy), "STATE MUTATOR", font=f_col, fill=M.DIM)
    _right(d, x_clock, hy, "CLOCK", f_col, M.DIM)
    d.text((bar0, hy), "TOWER HP", font=f_col, fill=M.DIM)
    _right(d, x_units, hy, "UNITS", f_col, M.DIM)
    d.line([(L, hy + 44), (R, hy + 44)], fill=M.BORDER, width=2)

    top, row_h = 240, 82
    for i, (name, f) in enumerate(rows):
        y = top + i * row_h
        if i:
            d.line([(L, y - 12), (R, y - 12)], fill=M.PANEL, width=2)
        d.text((x_name, y + 2), name, font=f_name, fill=M.BLUE)
        _right(d, x_clock, y, f["clock"], f_num, M.TEXT)

        pct = f["pct"]
        colour = M.GREEN if pct >= 100 else M.AMBER
        d.rounded_rectangle([bar0, y + 16, bar1, y + 42], radius=13, fill=M.PANEL)
        w = round((bar1 - bar0) * min(pct, 100) / 100)
        if w > 4:
            d.rounded_rectangle([bar0, y + 16, bar0 + w, y + 42], radius=13, fill=colour)
        _right(d, x_pct, y, f"{pct}%", f_num, colour)
        _right(d, x_units, y, str(f["units"]), f_num, M.TEXT if f["units"] else M.DIM)

    foot = ("Snapshot reload matched the saved hash" if hash_matched
            else "Snapshot reload MISSED the saved hash")
    d.text((L, H - 62), foot, font=f_foot, fill=M.GREEN if hash_matched else M.RED)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)

    detail = "; ".join(f"{name} tick {f['tick']} ({f['clock']}), towers {f['hp']} "
                       f"({f['pct']}% of a fresh {full_hp}), {f['units']} units"
                       for name, f in rows)
    return (f"{n} mutators reset on seed {SEED}, {distinct}/{n} distinct "
            f"(tick, tower hp, units): {detail}. Snapshot taken after {LEAD_IN_STEPS} "
            f"env steps at tick {saved['tick']}, {len(blob)} bytes, hash "
            f"{saved_hash:016x}, reload matched={hash_matched}")
