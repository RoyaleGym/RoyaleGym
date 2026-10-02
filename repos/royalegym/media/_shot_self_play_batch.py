#!/usr/bin/env python3
"""One bot playing itself: the batch a self-play vector environment hands a policy.

``ClashSelfPlayVecEnv(num_games=N)`` runs N battles and exposes them as ``2 * N``
rows, slot ``2 * g`` being game ``g``'s blue seat and ``2 * g + 1`` its red seat. So
a single shared policy gets both players' experience out of one forward pass, and
the leading axis of every observation key is twice the number of battles.

Nothing here is typed in. The environment is built, reset and stepped for real; the
keys and shapes are printed off the arrays that come back; the mover is named off the
object that moved; and the tower numbers behind the bars are read out of the
observation vector by name, through ``vector_offsets``, never by a counted index.

What the picture is for, and what it deliberately does not say
-------------------------------------------------------------
The claim being drawn is the own-frame one: the two rows of a battle hold the *same*
three towers with own and enemy swapped. Each row is a pair of bars, its own towers
above the enemy's.

The colour is the part that has to be able to come out wrong, so it is taken off the
data and not off the seat. Within a battle, the triple row ``2 * g`` calls its own is
painted blue and the triple row ``2 * g + 1`` calls its own is painted red, and then
all four bars in the column are looked up by the triple they actually hold. Only the
first bar is blue by definition; the other three have to earn their colour. Rows in
one absolute frame would put the blue triple on top of both rows and the column would
read as two parallel bars instead of a cross. A triple belonging to neither row comes
out amber. Bar length is a second, independent signal: the crossed pair are equal
lengths, and the four battles are visibly different from each other.

The bars carry no scale and no number on purpose. The field is ``tower_hp`` over the
player's level-scaled nominal maximum, which is not the ``max_hp`` of the tower
entities standing in the arena: measured at reset, a full-health king reads 0.785 and
a full-health princess 0.852, never 1.0. So the value is not "percent of health left",
a full row is not a full tower, and an earlier version of this figure printed it as a
health percentage and was wrong. The bars are comparable to each other, which is all
the mirror claim needs, and the footer names them after the observation field rather
than after health.

The mirror is checked, not asserted, and checked so that it can fail. Every one of
the ``2N * (2N - 1)`` ordered row pairs is compared on all three towers from both
sides, and the figure only claims the swap when the mirroring pairs are exactly the
within-battle ones and no others. Two guards keep that from being a free pass: the
battles are stepped well past reset first, because at reset all rows are equal and
every pair mirrors; and the rows' own-tower triples must be pairwise distinct, so a
pair cannot mirror by coincidence. If either guard fails the footer says so instead
of claiming the swap. The counts it prints, ``only 8 of 56``, are what makes the
check readable as a check.

Sized for a README tile: drawn at 900 px, shown at about 360, so nothing on it is
smaller than 34 px. The height is computed from the number of observation keys, so a
fifth key grows the canvas rather than sliding the strip under the footer.
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import make_media as M  # noqa: E402  (palette and fonts, shared with the other figures)

# The battle the README documents: a named deck, so the size of the card catalogue
# cannot change which cards are dealt, and a fixed seed.
SEED = 0
DECK = ("Knight", "Archer", "Giant", "Minions", "Fireball", "Cannon", "Zap", "Musketeer")
NUM_GAMES = 4
NOOP_PROB = 0.7
# Far enough in that the four battles have pulled apart: at reset every tower is at
# full health and all eight rows read the same, which would make the mirror true of
# every pair and say nothing. This is the tick the strip is drawn at, and the figure
# names it as a single tick.
STEPS = 240

W = 900
MARGIN = 30

F_HEAD = 64
F_KEY = 36
F_IDX = 36
F_SMALL = 34  # the floor: nothing on this canvas is smaller

ROW_H = 50     # one observation key
CHIP_H = 56    # one batch row: two bars
BAR_H = 16

SEAT_FILL = ((20, 58, 64), (62, 28, 20))       # blue seat, red seat
SEAT_EDGE = (M.BLUE, M.RED)


def _build(np):
    """A real vector environment, stepped for real. Returns what the figure draws."""
    from royalegym import (
        ClashParallelEnv,
        ClashSelfPlayVecEnv,
        DefaultStateMutator,
        RandomLegalOpponent,
        RustEngine,
    )
    from royalegym.obs import vector_offsets

    engine = RustEngine()
    # By name, never by number: a card id is a position in the catalogue, and the
    # positions move between card tables.
    by_name = {c.name: c.card_id for c in engine.cards()}
    ids = [by_name[n] for n in DECK]
    num_cards = len(engine.cards())

    def make():
        return ClashParallelEnv(engine=RustEngine(),
                                state_mutator=DefaultStateMutator(decks=[ids, ids]))

    # viser=None: this figure is not watching a battle, and binding the publisher's
    # port here would fight whatever else is running.
    venv = ClashSelfPlayVecEnv(num_games=NUM_GAMES, env_fn=make, viser=None)
    obs, _ = venv.reset(seed=SEED)

    rng, mover = np.random.default_rng(SEED), RandomLegalOpponent(noop_prob=NOOP_PROB)
    steps = 0
    for _ in range(STEPS):
        mask = obs["action_mask"]
        acts = np.array([mover.act({k: v[i] for k, v in obs.items()}, mask[i], rng)
                         for i in range(mask.shape[0])])
        obs, *_ = venv.step(acts)
        steps += 1

    keys = list(venv.single_observation_space.spaces)
    arrays = [(k, tuple(obs[k].shape)) for k in keys]

    off = vector_offsets(num_cards)
    vec = obs["vector"]
    own = vec[:, off["own_tower_hp"]]
    foe = vec[:, off["enemy_tower_hp"]]

    n = venv.num_envs
    # Every ordered pair, on all three towers from both sides. The claim is not that
    # the adjacent pairs mirror -- it is that they are the only ones that do.
    mirror = [[i != j and np.array_equal(own[i], foe[j]) and np.array_equal(foe[i], own[j])
               for j in range(n)] for i in range(n)]
    want = {(2 * g + k, 2 * g + 1 - k) for g in range(venv.num_games) for k in (0, 1)}
    found = {(i, j) for i in range(n) for j in range(n) if mirror[i][j]}
    paired = found == want
    # Guard against a mirror that holds because everything is equal: if two rows had
    # the same three towers, a pair could mirror without meaning anything.
    spread = len({tuple(own[i]) for i in range(n)}) == n

    # One bar per (row, own/enemy): how long it is, and which of the battle's two
    # tower triples it holds. The identity is what gets coloured, so the colours are
    # read off the data instead of off the seat the row belongs to.
    rows = [[(float(own[i].mean()), tuple(own[i])), (float(foe[i].mean()), tuple(foe[i]))]
            for i in range(n)]
    return dict(games=venv.num_games, envs=n, arrays=arrays, rows=rows, paired=paired,
                spread=spread, mover=type(mover).__name__, steps=steps,
                found=len(found), pairs=n * (n - 1))


def _shape_text(d, x, y, shape, font, lead_fill):
    """``(8, 20, 32, 18)`` with the leading axis in its own colour. Returns the width."""
    head, tail = f"({shape[0]}", "".join(f", {s}" for s in shape[1:]) + ")"
    d.text((x, y), "(", font=font, fill=M.DIM)
    x2 = x + d.textlength("(", font=font)
    d.text((x2, y), head[1:], font=font, fill=lead_fill)
    x3 = x2 + d.textlength(head[1:], font=font)
    d.text((x3, y), tail, font=font, fill=M.TEXT)
    return x3 + d.textlength(tail, font=font) - x


def draw(out_path: pathlib.Path) -> str:
    import numpy as np

    r = _build(np)
    games, envs, arrays, rows = r["games"], r["envs"], r["arrays"], r["rows"]

    head_f = M.theme_font(F_HEAD)
    mono = M.theme_font(F_KEY, mono=True)
    idx_f = M.theme_font(F_IDX, mono=True)
    small = M.theme_font(F_SMALL)

    head = f"{games} battles make {envs} batch rows"
    foot = ["top bar: own towers. colour: whose towers they are",
            f"{r['mover']} moves, one tick at step {r['steps']}"]
    if r["paired"] and r["spread"]:
        foot.append(f"only {r['found']} of {r['pairs']} row pairs mirror: the two rows of a battle")
    elif not r["spread"]:
        foot.append(f"step {r['steps']}: rows are not all distinct, the mirror says little")
    else:
        foot.append(f"step {r['steps']}: the two seats did NOT mirror")

    # ---------------------------------------------------- height, from the contents
    probe = M.canvas(1, 1)[1]
    head_h = probe.textbbox((0, 0), head, font=head_f)[3] + 16
    panel_y = 12 + head_h
    panel_h = ROW_H * len(arrays) + 18
    strip_y = panel_y + panel_h + 14
    label_h = 38
    strip_h = label_h + 2 * CHIP_H + 8
    foot_y = strip_y + strip_h + 12
    H = foot_y + 40 * (len(foot) - 1) + probe.textbbox((0, 0), foot[-1], font=small)[3] + 14

    im, d = M.canvas(W, H)
    d.text((MARGIN, 12), head, font=head_f, fill=M.TEXT)

    # --------------------------------------------- every key, with its leading axis
    d.rounded_rectangle((MARGIN, panel_y, W - MARGIN, panel_y + panel_h), 12,
                        fill=M.PANEL, outline=M.BORDER, width=2)
    key_w = max(d.textlength(k, font=mono) for k, _ in arrays)
    shape_x = MARGIN + 22 + key_w + 40
    for i, (key, shape) in enumerate(arrays):
        yy = panel_y + 9 + i * ROW_H + 4
        d.text((MARGIN + 22, yy), key, font=mono, fill=M.DIM)
        _shape_text(d, shape_x, yy, shape, mono, M.AMBER)

    # ------------------------------------------- the rows, two bars each, by battle
    gw = (W - 2 * MARGIN) / games
    idx_w = d.textlength("0", font=idx_f)
    for g in range(games):
        gx = MARGIN + g * gw
        d.text((gx + 8, strip_y), f"battle {g + 1}", font=small, fill=M.DIM)
        # The two triples this battle's rows call their own, in seat order. Every bar
        # in the column is coloured by which of the two it actually holds, so only the
        # first bar is a colour by definition; the other three have to earn theirs. A
        # bar holding neither comes out amber, and nothing on the column lines up.
        owners = {rows[2 * g + s][0][1]: SEAT_EDGE[s] for s in (0, 1)}
        for k in (0, 1):
            i = 2 * g + k
            cy = strip_y + label_h + k * (CHIP_H + 8)
            right = gx + gw - 16
            d.rounded_rectangle((gx + 4, cy, right, cy + CHIP_H), 10,
                                fill=SEAT_FILL[k], outline=SEAT_EDGE[k], width=3)
            d.text((gx + 16, cy + 8), str(i), font=idx_f, fill=SEAT_EDGE[k])
            bx = gx + 16 + idx_w + 16
            full = right - 16 - bx
            for slot, (value, triple) in enumerate(rows[i]):
                by = cy + 9 + slot * (BAR_H + 6)
                d.rounded_rectangle((bx, by, bx + full * value, by + BAR_H), 3,
                                    fill=owners.get(triple, M.AMBER))

    for i, line in enumerate(foot):
        d.text((MARGIN, foot_y + i * 40), line, font=small, fill=M.DIM)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)
    shapes = "; ".join(f"{k}{s}" for k, s in arrays)
    return (f"num_games={games} num_envs={envs}; {shapes}; "
            f"mover={r['mover']} steps={r['steps']}; "
            f"mirroring ordered row pairs: {r['found']} of {r['pairs']}, "
            f"exactly the within-battle ones: {r['paired']}; own triples distinct: "
            f"{r['spread']}; bar values (own, foe) = "
            f"{[(round(a[0], 3), round(b[0], 3)) for a, b in rows]}")
