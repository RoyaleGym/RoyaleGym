#!/usr/bin/env python3
"""Four starting boards, one state mutator each, drawn from what reset() left behind.

Everything on the picture is measured in ``draw``. One ``ClashParallelEnv`` per mutator
is built on its own engine, ``reset(seed=SEED)`` is called once, and the board it lands
on is read straight out of ``env.battle_state``: the clock, every crown tower's hp and
max hp, and the position of every non-tower entity. Those coordinates are mapped into
the arena rectangle the engine reports, so each column is a plan of a real board and a
destroyed tower is a hole where the engine says the tower was. The column headings are
the mutator classes' own ``__name__`` with the shared suffix cut off.

Nothing is counted on the picture, because a count of the units a figure itself placed
is the figure reading back its own input. The boards are drawn instead.

The MidGame column is ONE draw from the ranges that mutator is configured with, not a
picked-out board. The column says so, and the band it was drawn from is printed at the
foot of the picture, read back off the mutator object rather than off the constants.

The Snapshot column is a round trip. A default battle is played until the lead-in step
budget runs out, the engine is asked for its bytes and its state hash, and
``SnapshotStateMutator`` loads those bytes into a brand new engine. Two things are then
measured and both are reported: the reloaded engine's hash equals the saved one, and
that same reloaded state stepped a single tick no longer does. The second one is there
because the first, on its own, would also pass for a hash that cannot tell two states
apart -- and then it would be evidence of nothing.
"""

from __future__ import annotations

import pathlib

import make_media as M
import numpy as np

from royalegym import ClashParallelEnv, RandomLegalOpponent, RustEngine
from royalegym.protocol import BLUE, RED, SpawnSpec, TowerSlot
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

# This figure's arguments to MidGameStateMutator -- not that class's defaults, which
# start a clean board (tower_hp_percent (100, 100), tower_down_prob 0.0). Every number
# the MidGame column shows is drawn from these by the env's seeded generator.
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


def _facts(env: ClashParallelEnv, max_hp: dict[int, int]) -> dict:
    """The board the instant reset() returns: clock, all six towers, every unit.

    ``max_hp`` is the fresh board's per-slot maximum, needed only to size a tower that
    is already destroyed -- a destroyed tower has no entity to read a maximum off.
    """
    s = env.battle_state
    alive = {(e.team, e.tower_slot): e for e in s.entities if e.tower_slot >= 0}
    towers = {}
    for team, player in enumerate(s.players):
        for slot in TowerSlot:
            e = alive.get((team, slot))
            hp = player.tower_hp[slot]
            if e is not None and e.hp != hp:
                raise RuntimeError(f"tower {team}/{slot}: entity {e.hp}, player {hp}")
            towers[(team, slot)] = {
                "hp": hp,
                "max": e.max_hp if e is not None else max_hp[slot],
                "pos": (e.x, e.y) if e is not None else None,
            }
    units = [(e.team, e.x, e.y) for e in s.entities if e.tower_slot < 0]
    return {"tick": s.tick, "tick_ms": s.tick_ms, "towers": towers, "units": units}


def _reset(mutator) -> ClashParallelEnv:
    env = ClashParallelEnv(engine=RustEngine(), state_mutator=mutator)
    env.reset(seed=SEED)
    return env


def _tile_center(arena, tx: int, ty: int) -> tuple[int, int]:
    return tx * arena.subtile + arena.subtile // 2, ty * arena.subtile + arena.subtile // 2


def _hand_built(engine: RustEngine, deck: list[int]) -> ScriptedBoardStateMutator:
    """A defensive drill: a push already walking in, and two defenders to meet it.

    One spec is one unit on the board, not one card's worth: ``protocol.spawn_violation``
    bounds a spec's hp by the PER-UNIT ``CardInfo.hitpoints``, and both engines spawn a
    single entity per spec (MockEngine ``_new_battle``). So the Minions spec below is one
    minion, and that is why the picture draws the board rather than counting it.
    """
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


def _play_in(deck: list[int]) -> tuple[bytes, int, int]:
    """Play a default battle a while, then hand back its bytes, hash and step count."""
    engine = RustEngine()
    env = ClashParallelEnv(engine=engine, state_mutator=DefaultStateMutator(decks=[deck, deck]))
    obs, _ = env.reset(seed=SEED)
    rng, policy = np.random.default_rng(SEED), RandomLegalOpponent(noop_prob=NOOP_PROB)
    played = 0
    for _ in range(LEAD_IN_STEPS):
        if not env.agents:
            break
        obs, *_ = env.step({a: policy.act(obs[a], obs[a]["action_mask"], rng)
                            for a in env.agents})
        played += 1
    return engine.save_state(), engine.state_hash(), played


def _clock(tick: int, tick_ms: int) -> str:
    secs = round(tick * tick_ms / 1000)
    return f"{secs // 60}:{secs % 60:02d}"


def _short(mutator) -> str:
    """The mutator's own class name, minus the suffix all four of them share."""
    return type(mutator).__name__.removesuffix("StateMutator")


# --------------------------------------------------------------------------- drawing


def _dim(colour, f: float) -> tuple[int, int, int]:
    return tuple(int(c * f) for c in colour)


def _centre(d, cx: int, y: int, text: str, font, fill) -> None:
    d.text((cx - d.textlength(text, font=font) / 2, y), text, font=font, fill=fill)


def _fits(d, text: str, font, width: int) -> bool:
    return d.textlength(text, font=font) <= width


def _board(d, arena, facts: dict, colour: dict[int, tuple], x: int, y: int,
           w: int, h: int) -> None:
    """One starting board, to scale: the arena, its six tower sites, its units."""
    def px(ex: int, ey: int) -> tuple[float, float]:
        return x + w * ex / arena.width, y + h * ey / arena.height

    d.rounded_rectangle([x, y, x + w, y + h], radius=12, fill=M.PANEL)

    half = arena.subtile // arena.half          # subtiles per half-cell
    top = y + h * (arena.water_half_rows[0] * half) / arena.height
    bot = y + h * ((arena.water_half_rows[1] + 1) * half) / arena.height
    d.rectangle([x, top, x + w, bot], fill=_dim(M.RIVER, 0.30))
    for bx in arena.bridge_centers_x():
        cx, _ = px(bx, 0)
        d.rectangle([cx - w * 0.035, top, cx + w * 0.035, bot], fill=_dim(M.BRIDGE, 0.55))

    for (team, slot), t in sorted(facts["towers"].items()):
        pos = t["pos"] or _tower_site(arena, team, slot)
        cx, cy = px(*pos)
        size = 17 if slot == TowerSlot.KING else 14
        box = [cx - size, cy - size, cx + size, cy + size]
        team_c = colour[team]
        if t["hp"] <= 0:                         # destroyed: an empty socket
            d.rounded_rectangle(box, radius=5, outline=_dim(team_c, 0.45), width=3)
            d.line([box[0] + 6, box[1] + 6, box[2] - 6, box[3] - 6], fill=M.RED, width=4)
            d.line([box[0] + 6, box[3] - 6, box[2] - 6, box[1] + 6], fill=M.RED, width=4)
            continue
        d.rounded_rectangle(box, radius=5, fill=M.BG, outline=_dim(team_c, 0.5), width=3)
        fill_h = (box[3] - box[1] - 6) * min(1.0, t["hp"] / t["max"])
        d.rounded_rectangle([box[0] + 3, box[3] - 3 - fill_h, box[2] - 3, box[3] - 3],
                            radius=4, fill=team_c)

    for team, ex, ey in facts["units"]:
        cx, cy = px(ex, ey)
        d.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=colour[team],
                  outline=M.BG, width=3)


def _tower_site(arena, team: int, slot: int) -> tuple[int, int]:
    """Where a tower stands, for the towers that are not standing any more."""
    if slot == TowerSlot.KING:
        return arena.king_centers[team]
    return arena.princess_centers[team][slot - 1]


def draw(out_path: pathlib.Path) -> str:
    """Reset one env per mutator, draw the board each one landed on, save it."""
    # ---- measure -------------------------------------------------------------
    probe = RustEngine()
    deck = _deck(probe)
    arena = probe.arena()

    fresh_m = DefaultStateMutator(decks=[deck, deck])
    fresh_env = _reset(fresh_m)
    full = {slot: hp for slot, hp in enumerate(fresh_env.battle_state.players[BLUE].tower_hp)}
    fresh = _facts(fresh_env, full)

    mid_m = MidGameStateMutator(
        tick_range=MID_TICKS, elixir_milli_range=MID_ELIXIR,
        max_tower_hp=(full[TowerSlot.KING], full[TowerSlot.LEFT]),
        tower_hp_percent=MID_HP_PCT, tower_down_prob=MID_DOWN_PROB,
        decks=[deck, deck])
    mid = _facts(_reset(mid_m), full)

    script_m = _hand_built(probe, deck)
    scripted = _facts(_reset(script_m), full)

    blob, saved_hash, played = _play_in(deck)
    snap_m = SnapshotStateMutator([blob])
    snap_env = _reset(snap_m)
    snapshot = _facts(snap_env, full)
    reload_matched = snap_env.engine.state_hash() == saved_hash

    # The control. Without it, a hash that returned the same number for every state
    # would pass the line above, and the picture would be footing "exact" on nothing.
    ticked = RustEngine()
    ticked.load_state(blob)
    ticked.step([], 1)
    tick_differs = ticked.state_hash() != saved_hash

    cols = [(_short(m), f) for m, f in
            ((fresh_m, fresh), (mid_m, mid), (script_m, scripted), (snap_m, snapshot))]
    n = len(cols)
    down = sum(1 for _, f in cols for t in f["towers"].values() if t["hp"] <= 0)

    # ---- draw ----------------------------------------------------------------
    # Sized for a three-across README table: nothing under 34 px, headline 66 px.
    colour = {BLUE: M.BLUE, RED: M.RED}
    W, H = 1000, 640
    im, d = M.canvas(W, H)
    f_head = M.theme_font(66)
    f_name = M.theme_font(40)
    f_clock = M.theme_font(42)
    f_note = M.theme_font(34)

    L, R = 36, W - 36
    words = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}
    d.text((L, 2), f"{words.get(n, n)} starts, one reset each", font=f_head, fill=M.TEXT)

    notes = {
        _short(mid_m): ("one draw", "from a band"),
        _short(snap_m): ("played in,", "then saved"),
    }
    col_w = (R - L) / n
    bw, bh = 141, 250
    board_y = 140
    for i, (name, f) in enumerate(cols):
        cx = int(L + col_w * (i + 0.5))
        _centre(d, cx, 86, name, f_name, M.BLUE)
        _board(d, arena, f, colour, int(cx - bw / 2), board_y, bw, bh)
        _centre(d, cx, board_y + bh + 6, _clock(f["tick"], f["tick_ms"]), f_clock, M.TEXT)
        for j, line in enumerate(notes.get(name, ())):
            _centre(d, cx, board_y + bh + 60 + j * 36, line, f_note, M.DIM)

    lo, hi = (_clock(t, fresh["tick_ms"]) for t in mid_m.tick_range)
    band = (f"seed {SEED} · MidGame band {lo}-{hi}, "
            f"{mid_m.hp_pct[0]}-{mid_m.hp_pct[1]}% tower hp")
    def _check(matched: bool, differs: bool) -> str:
        return (f"saved bytes reloaded: hash {'matched' if matched else 'MISSED'} "
                f"· one tick on: {'differs' if differs else 'SAME TOO'}")

    check = _check(reload_matched, tick_differs)
    # Every wording this line can take has to fit, not just today's. A figure that
    # crashes on the failing branch cannot report the failure it exists to report.
    for line in (band, *(_check(a, b) for a in (True, False) for b in (True, False))):
        if not _fits(d, line, f_note, R - L):
            raise RuntimeError(f"footer does not fit at 34 px: {line!r}")
    d.line([(L, H - 104), (R, H - 104)], fill=M.PANEL, width=2)
    d.text((L, H - 92), band, font=f_note, fill=M.DIM)
    d.text((L, H - 48), check, font=f_note,
           fill=M.GREEN if reload_matched and tick_differs else M.RED)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)

    detail = "; ".join(
        f"{name} tick {f['tick']} ({_clock(f['tick'], f['tick_ms'])}), "
        f"towers {[f['towers'][(t, s)]['hp'] for t in (BLUE, RED) for s in TowerSlot]}, "
        f"{len(f['units'])} units" for name, f in cols)
    return (f"{n} mutators reset on seed {SEED}, {down} tower(s) already down: {detail}. "
            f"Snapshot taken after {played} env steps, {len(blob)} bytes, hash "
            f"{saved_hash:016x}, reload matched={reload_matched}, "
            f"one tick on differs={tick_differs}")
