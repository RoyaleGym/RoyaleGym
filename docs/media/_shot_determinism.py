#!/usr/bin/env python3
r"""Figure: same seed, same battle.

One scripted battle is played three times and the per-tick state hash is compared:

  run 1   a fresh ``royalesim.Battle``
  run 2   a second, independently constructed ``Battle``, same seed, same commands
  run 3   a third ``Battle`` that never played the opening: it is handed run 1's
          snapshot bytes mid-battle with ``.load()`` and carries on from there

The deploy script is not typed in. It is produced by a fourth throwaway battle that
walks a seeded RNG over hand slots and arena positions and keeps whatever
``check_deploy`` accepts, so the plan is a legal, lively battle and is the same list of
commands on every machine. The three runs then replay that one list.

The picture is drawn for a README tile: it has to stay readable when GitHub shrinks it
into a table cell a few hundred pixels wide. So it carries one claim, four rows of the
comparison around the point where run 3 joins, and nothing else. The full run is
summarised in the return value.

Nothing here reads a screen or a recording. Everything on the picture is counted in
this run.
"""

from __future__ import annotations

import pathlib
import sys

_HERE = str(pathlib.Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import make_media as M  # noqa: E402

# The battle. Short enough to draw, long enough that units cross the river, meet and
# fight: at 20 ticks a second, 600 ticks is the first 30 seconds of a match.
SEED = 20260922
TICKS = 600
SNAPSHOT_AT = 240  # where run 3 is handed the bytes and takes over
DECK = ("Knight", "Archer", "Giant", "Minions", "Fireball", "Cannon", "Zap",
        "Musketeer")
SHUFFLE_MIRRORED = 2
R_OK = 0

HEX_SHOWN = 9  # leading hex digits of the 16 a state hash has; the rest is elided


# --------------------------------------------------------------------------- engine


def _new_battle(slot_of_k):
    import royalesim

    # The deck is named, never numbered: a card id is a position in the catalogue and
    # the catalogue is whatever the build's card table holds. Naming eight cards pins
    # the ids to 0..7 here.
    return royalesim.Battle(list(DECK), slot_of_k, None, None)


def _plan(slot_of_k, arena_w: int, arena_h: int) -> dict[int, list[tuple]]:
    """A legal deploy script, computed by playing a throwaway battle.

    Each tick, each side gets one chance to play: a seeded RNG picks a hand slot and a
    point in the arena, ``check_deploy`` says whether the engine would take it, and only
    the accepted ones go into the script. The throwaway battle is stepped alongside so
    elixir, hands and occupancy stay honest.
    """
    import numpy as np

    rng = np.random.default_rng(SEED)
    b = _new_battle(slot_of_k)
    deck = list(range(len(DECK)))
    b.reset(SEED, [deck, deck], SHUFFLE_MIRRORED, 0, None, None, [])

    script: dict[int, list[tuple]] = {}
    for tick in range(TICKS):
        cmds = []
        for team in (0, 1):
            if rng.random() > 0.22:
                continue
            slot = int(rng.integers(0, 4))
            # Own half, away from the very edges. Anything the engine refuses -- the
            # river, a tower's footprint, the wrong side of the arena, no elixir -- is
            # simply dropped, so no illegal command ever reaches the script.
            lo, hi = (0.08, 0.46) if team == 0 else (0.54, 0.92)
            x = int(rng.uniform(0.08, 0.92) * arena_w)
            y = int(rng.uniform(lo, hi) * arena_h)
            if b.check_deploy(team, slot, x, y) == R_OK:
                cmds.append((team, slot, x, y))
        b.step(cmds, 1)
        if cmds:
            script[tick] = cmds
    return script


def _run(slot_of_k, script, ticks: int) -> tuple[list[int], bytes]:
    """Play the script from tick 0 and return one hash per tick, plus the snapshot."""
    b = _new_battle(slot_of_k)
    deck = list(range(len(DECK)))
    b.reset(SEED, [deck, deck], SHUFFLE_MIRRORED, 0, None, None, [])
    hashes, blob = [], b""
    for tick in range(ticks):
        b.step(script.get(tick, []), 1)
        hashes.append(b.state_hash())
        if tick == SNAPSHOT_AT - 1:
            blob = bytes(b.save())
    return hashes, blob


def _resume(slot_of_k, script, blob: bytes, ticks: int) -> list[int]:
    """A battle that never played the opening: load the bytes and carry on."""
    b = _new_battle(slot_of_k)
    b.load(blob)
    return [(b.step(script.get(t, []), 1), b.state_hash())[1]
            for t in range(SNAPSHOT_AT, ticks)]


def _liveliness(slot_of_k, script, ticks: int) -> tuple[int, int]:
    """Evidence that the matching hashes are not the equality of two idle battles.

    Returns the crown-tower hit points actually LOST between the first tick and the last,
    and how many non-tower entities are alive at the end.

    An earlier version of this compared each tower's hit points against its maximum from
    the card table, and reported 4768. That is not damage: a tower does not start at the
    level ladder's maximum, so the figure was publishing a difference between two
    different things and calling it harm done. Measure start against end instead, and if
    the answer is zero, say zero.
    """
    import json

    b = _new_battle(slot_of_k)
    deck = list(range(len(DECK)))
    b.reset(SEED, [deck, deck], SHUFFLE_MIRRORED, 0, None, None, [])
    first = json.loads(bytes(b.state_json()))
    for t in range(ticks):
        b.step(script.get(t, []), 1)
    last = json.loads(bytes(b.state_json()))
    lost = sum(a - b_ for p, q in zip(first["players"], last["players"])
               for a, b_ in zip(p["tower_hp"], q["tower_hp"]))
    alive = sum(1 for e in last["entities"] if e[2] == 0) if last.get("entities") else 0
    return lost, alive


# --------------------------------------------------------------------------- glyphs


def _check(d, x: int, y: int, size: int, colour) -> None:
    """A tick mark drawn, not typed: the theme fonts have no U+2713."""
    w = int(size * 0.18) or 2
    d.line([(x, y + size * 0.52), (x + size * 0.36, y + size * 0.84),
            (x + size * 0.98, y + size * 0.14)], fill=colour, width=w, joint="curve")


def _cross(d, x: int, y: int, size: int, colour) -> None:
    w = int(size * 0.18) or 2
    d.line([(x + size * 0.1, y + size * 0.12), (x + size * 0.9, y + size * 0.88)],
           fill=colour, width=w)
    d.line([(x + size * 0.9, y + size * 0.12), (x + size * 0.1, y + size * 0.88)],
           fill=colour, width=w)


def _vdots(d, cx: int, cy: int, colour, r: int = 4, gap: int = 13) -> None:
    for k in (-1, 0, 1):
        d.ellipse([cx - r, cy + k * gap - r, cx + r, cy + k * gap + r], fill=colour)


# --------------------------------------------------------------------------- drawing


def draw(out_path: pathlib.Path) -> str:
    """Compute the data, draw the picture, save it, return a one-line summary."""
    from PIL import Image, ImageDraw

    from royalegym import RustEngine

    engine = RustEngine()
    arena = engine.arena()
    slot_of_k = engine.slot_of_k

    script = _plan(slot_of_k, arena.width, arena.height)
    n_deploys = sum(len(v) for v in script.values())

    h1, blob = _run(slot_of_k, script, TICKS)
    h2, _ = _run(slot_of_k, script, TICKS)
    h3 = _resume(slot_of_k, script, blob, TICKS)
    hp_lost, n_alive = _liveliness(slot_of_k, script, TICKS)

    # The counts. Nothing below is written into a string literal.
    n_diff_12 = sum(1 for a, b in zip(h1, h2) if a != b)
    third = {SNAPSHOT_AT + i + 1: h for i, h in enumerate(h3)}
    n_diff_13 = sum(1 for i, h in enumerate(h3) if h != h1[SNAPSHOT_AT + i])
    n_resumed = len(h3)
    n_distinct = len(set(h1))
    total_cmp = TICKS + n_resumed
    total_diff = n_diff_12 + n_diff_13

    # Four rows out of the battle, chosen at the handoff: the tick before run 3 exists,
    # the two it joins on, and the last tick of the battle. The figure says so.
    picked = [SNAPSHOT_AT, SNAPSHOT_AT + 1, SNAPSHOT_AT + 2, TICKS]
    rows: list = [picked[0], picked[1], picked[2], None, picked[3]]

    # ---------------------------------------------------------------- measure
    #
    # Nothing on this canvas is below 34 px. The README shows it in a cell a few hundred
    # pixels wide, so every size here is chosen to survive a 2.8x downscale.
    f_big = M.theme_font(72)
    f_sub = M.theme_font(34)
    f_head = M.theme_font(34)
    f_mono = M.theme_font(38, mono=True)
    f_tick = M.theme_font(34, mono=True)
    f_foot = M.theme_font(34)

    ruler = ImageDraw.Draw(Image.new("RGB", (8, 8)))

    def w_of(text, font) -> int:
        return int(ruler.textlength(text, font=font))

    def cell(h: int) -> str:
        return f"{h:016x}"[:HEX_SHOWN] + "…"

    hex_w = max(w_of(cell(h), f_mono) for h in h1)
    tick_w = max(w_of(str(t), f_tick) for t in picked)
    mark_w = 42
    pad = 44
    gap = 30
    tick_x = pad
    c1 = tick_x + tick_w + gap
    c2 = c1 + hex_w + gap
    c3 = c2 + hex_w + gap
    mark_x = c3 + hex_w + gap
    table_r = mark_x + mark_w

    # Headline: one number and four words.
    # "checks", not "hashes": total_cmp is the number of comparisons made, and more
    # hash values than that were produced to make them.
    hl_a, hl_b = str(total_diff), f" of {total_cmp} hash checks differ"
    w_hl = w_of(hl_a, f_big) + w_of(hl_b, f_big)

    # The support line. Every candidate is true; the widest one that fits wins.
    busy = f"{n_deploys} cards played · {n_alive} units alive at the end"
    if hp_lost:
        busy += f" · {hp_lost} tower HP gone"
    subs = [f"Seed {SEED} · {busy}", f"Seed {SEED} · {busy}", busy]
    foot = [f"Run 3 joins at tick {SNAPSHOT_AT + 1} from run 1's snapshot.",
            f"{len(picked)} rows of {TICKS}, picked at that handoff."]

    limit = 1000 - 2 * pad
    sub = next((s for s in subs if w_of(s, f_sub) <= limit), subs[-1])
    W = min(1000, max(table_r, pad + w_hl, pad + w_of(sub, f_sub),
                      *(pad + w_of(f, f_foot) for f in foot)) + pad)

    row_h, dots_h, head_h = 58, 40, 50
    y_hl = 34
    y_sub = y_hl + 86
    y_tab = y_sub + 62
    y_rows = y_tab + head_h
    y_foot = y_rows + 3 * row_h + dots_h + row_h + 26
    H = y_foot + 2 * 44 + 18

    # ---------------------------------------------------------------- draw
    im, d = M.canvas(W, H)

    good = M.GREEN if total_diff == 0 else M.RED
    d.text((pad, y_hl), hl_a, font=f_big, fill=good)
    d.text((pad + w_of(hl_a, f_big), y_hl), hl_b, font=f_big, fill=M.TEXT)
    d.text((pad, y_sub), sub, font=f_sub, fill=M.DIM)

    d.text((tick_x, y_tab), "tick", font=f_head, fill=M.DIM)
    d.text((c1, y_tab), "run 1", font=f_head, fill=M.BLUE)
    d.text((c2, y_tab), "run 2", font=f_head, fill=M.BLUE)
    d.text((c3, y_tab), "run 3", font=f_head, fill=M.AMBER)
    d.line([(pad, y_tab + head_h - 12), (table_r, y_tab + head_h - 12)],
           fill=M.BORDER, width=2)

    ry = y_rows
    for i, tick in enumerate(rows):
        if tick is None:
            mid = ry + dots_h // 2
            for cx in (tick_x + tick_w // 2, c1 + hex_w // 2, c2 + hex_w // 2,
                       c3 + hex_w // 2, mark_x + mark_w // 2):
                _vdots(d, cx, mid, M.DIM)
            ry += dots_h
            continue
        a, b = h1[tick - 1], h2[tick - 1]
        c = third.get(tick)
        same = (a == b) and (c is None or c == a)
        if not same:
            d.rectangle([pad - 8, ry - 6, table_r + 8, ry + row_h - 10],
                        fill=(84, 32, 32))
        elif i % 2 == 0:
            d.rectangle([pad - 8, ry - 6, table_r + 8, ry + row_h - 10], fill=M.PANEL)
        col = M.TEXT if same else M.RED
        d.text((tick_x, ry + 2), str(tick), font=f_tick, fill=M.DIM)
        d.text((c1, ry), cell(a), font=f_mono, fill=col)
        d.text((c2, ry), cell(b), font=f_mono, fill=col)
        if c is None:
            dash = "—"
            d.text((c3 + (hex_w - w_of(dash, f_mono)) // 2, ry), dash, font=f_mono,
                   fill=(96, 96, 112))
        else:
            d.text((c3, ry), cell(c), font=f_mono, fill=col)
        (_check if same else _cross)(d, mark_x, ry + 4, mark_w,
                                     M.GREEN if same else M.RED)
        # Where run 3 is handed the bytes: between the last row without it and the
        # first row with it.
        if c is not None and third.get(tick - 1) is None:
            d.line([(pad - 8, ry - 7), (table_r + 8, ry - 7)], fill=M.AMBER, width=3)
        ry += row_h

    for k, line in enumerate(foot):
        d.text((pad, y_foot + k * 44), line, font=f_foot, fill=M.DIM)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)

    verdict = "all identical" if total_diff == 0 else "MISMATCH"
    return (f"{verdict}: {TICKS} ticks played twice ({n_diff_12} differing), "
            f"{n_resumed} ticks replayed from a snapshot taken at tick {SNAPSHOT_AT} "
            f"({n_diff_13} differing), {total_cmp} hash comparisons; seed {SEED}, "
            f"{n_deploys} scripted deploys, {n_alive} units alive at tick {TICKS}, "
            f"{hp_lost} tower hit points lost; canvas {W}x{H}")
