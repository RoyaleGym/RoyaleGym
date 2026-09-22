#!/usr/bin/env python3
"""One bot playing itself: the batch a self-play vector environment hands a policy.

``ClashSelfPlayVecEnv(num_games=N)`` runs N battles and exposes them as ``2 * N``
rows, slot ``2 * g`` being game ``g``'s blue seat and ``2 * g + 1`` its red seat. So
a single shared policy gets both players' experience out of one forward pass, and
the leading axis of every observation key is twice the number of battles.

Nothing here is typed in. The environment is built, reset and stepped for real; the
shapes and dtypes are printed off the arrays that come back; and the tower-health
numbers on the pair strip are read out of the observation vector by name, through
``vector_offsets``, never by a counted index.

The pairing claim is checked rather than asserted. Blue's ``own_tower_hp`` and red's
``enemy_tower_hp`` are the same three towers seen from the two seats, so the two rows
of one battle mirror each other exactly. The module compares every ordered pair of
rows and only draws the strip if the mirror holds for the adjacent pairs and for no
others; if it ever stops holding, the strip says so instead of claiming it.

Sized for a README tile: drawn at 900 px, shown at about 360, so nothing on it is
smaller than 34 px.
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
# full health and all eight rows read the same, which would make the mirror true but
# say nothing. This is the step the strip is drawn at, and the figure names it.
STEPS = 240

W, H = 900, 640
MARGIN = 30

F_HEAD = 64
F_SUB = 36
F_KEY = 36
F_CHIP = 36
F_SMALL = 34  # the floor: nothing on this canvas is smaller

SEAT_FILL = ((20, 58, 64), (62, 28, 20))       # blue seat, red seat
SEAT_EDGE = (M.BLUE, M.RED)


def _build(np):
    """A real vector environment, stepped for real. Returns what the figure draws."""
    from royalegym import (ClashParallelEnv, ClashSelfPlayVecEnv, DefaultStateMutator,
                           RandomLegalOpponent, RustEngine)
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

    rng, policy = np.random.default_rng(SEED), RandomLegalOpponent(noop_prob=NOOP_PROB)
    for _ in range(STEPS):
        mask = obs["action_mask"]
        acts = np.array([policy.act({k: v[i] for k, v in obs.items()}, mask[i], rng)
                         for i in range(mask.shape[0])])
        obs, *_ = venv.step(acts)

    keys = list(venv.single_observation_space.spaces)
    arrays = [(k, tuple(obs[k].shape), str(obs[k].dtype)) for k in keys]

    off = vector_offsets(num_cards)
    vec = obs["vector"]
    own = vec[:, off["own_tower_hp"]]
    foe = vec[:, off["enemy_tower_hp"]]

    n = venv.num_envs
    mirror = [[i != j and np.array_equal(own[i], foe[j]) and np.array_equal(foe[i], own[j])
               for j in range(n)] for i in range(n)]
    want = {(2 * g + k, 2 * g + 1 - k) for g in range(NUM_GAMES) for k in (0, 1)}
    found = {(i, j) for i in range(n) for j in range(n) if mirror[i][j]}
    paired = found == want

    pct = [(int(round(own[i].mean() * 100)), int(round(foe[i].mean() * 100)))
           for i in range(n)]
    distinct = len({p[0] for p in pct}) == n
    return venv.num_games, venv.num_envs, arrays, pct, paired, distinct


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

    games, envs, arrays, pct, paired, distinct = _build(np)

    im, d = M.canvas(W, H)
    mono = M.theme_font(F_KEY, mono=True)
    chip_f = M.theme_font(F_CHIP, mono=True)
    small = M.theme_font(F_SMALL)
    small_m = M.theme_font(F_SMALL, mono=True)

    d.text((MARGIN, 26), f"{games} battles make {envs} batch rows",
           font=M.theme_font(F_HEAD), fill=M.TEXT)
    d.text((MARGIN, 100), "one policy, both seats", font=M.theme_font(F_SUB), fill=M.DIM)

    # ------------------------------------------------- every key, as it came back
    y = 168
    row_h = 56
    d.rounded_rectangle((MARGIN, y - 14, W - MARGIN, y - 14 + row_h * len(arrays) + 10),
                        12, fill=M.PANEL, outline=M.BORDER, width=2)
    key_w = max(d.textlength(k, font=mono) for k, _, _ in arrays)
    shape_x = MARGIN + 22 + key_w + 34
    for i, (key, shape, dtype) in enumerate(arrays):
        yy = y + i * row_h
        d.text((MARGIN + 22, yy), key, font=mono, fill=M.DIM)
        _shape_text(d, shape_x, yy, shape, mono, M.AMBER)
        d.text((W - MARGIN - 22 - d.textlength(dtype, font=small_m), yy + 2), dtype,
               font=small_m, fill=M.GREEN)

    # ------------------------------------------------- the rows, paired by battle
    y = y - 14 + row_h * len(arrays) + 10 + 26
    gw = (W - 2 * MARGIN) / games
    ch = 52
    # widest value actually drawn, so a 100 % tower cannot overflow its column
    num_w = max(d.textlength(str(v), font=chip_f) for pair in pct for v in pair)
    for g in range(games):
        gx = MARGIN + g * gw
        d.text((gx + 8, y), f"battle {g + 1}", font=small, fill=M.DIM)
        for k in (0, 1):
            i = 2 * g + k
            cy = y + 44 + k * (ch + 8)
            d.rounded_rectangle((gx + 6, cy, gx + gw - 18, cy + ch), 10,
                                fill=SEAT_FILL[k], outline=SEAT_EDGE[k], width=3)
            d.text((gx + 20, cy + 6), str(i), font=chip_f, fill=SEAT_EDGE[k])
            right = gx + gw - 30
            for slot, value in enumerate(pct[i]):
                text = str(value)
                x = right - num_w * (2 - slot) - 22 * (1 - slot)
                d.text((x + num_w - d.textlength(text, font=chip_f), cy + 6), text,
                       font=chip_f, fill=M.TEXT)

    foot = (f"row, own, foe: mean tower hp %, step {STEPS}" if paired
            else f"step {STEPS}: the two seats did NOT mirror")
    d.text((MARGIN, H - 54), foot, font=small, fill=M.DIM)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)
    shapes = "; ".join(f"{k}{s} {t}" for k, s, t in arrays)
    return (f"num_games={games} num_envs={envs}; {shapes}; "
            f"mirror pairs exactly adjacent: {paired}; own% all distinct: {distinct}; "
            f"own/foe mean tower hp % at step {STEPS}: {pct}")
