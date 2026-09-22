#!/usr/bin/env python3
"""RoyaleGym: the legal-move mask on the first step of the Try-it battle.

One figure. It plays the battle the README documents, takes the action mask straight out
of the first observation, and draws it: one tile board per hand card, lit where the move
is legal. Every number on the canvas is measured in this run -- the total, the per-card
counts, the elixir costs and the card names all come from the env that was just reset.

Nothing runs at import time. Call ``draw(out_path)``.
"""

from __future__ import annotations

import pathlib

import make_media as M

# The same eight cards and the same seed the README's Try-it block uses, so this figure is
# about that battle and not about a different one. Cards are named, never numbered: a card
# id is a position in the catalogue and the positions move between card tables.
DECK = ("Knight", "Archer", "Giant", "Minions", "Fireball", "Cannon", "Zap", "Musketeer")
SEED = 0

W, H = 1000, 640
MARGIN = 30
TILE = 10           # px per board tile; 18 x 32 tiles makes a 180 x 320 board
LIT = M.BLUE
DARK = (52, 52, 68)


def _measure() -> dict:
    """Reset the Try-it battle and read the legal-move mask for both seats."""
    import numpy as np

    from royalegym import ClashParallelEnv, DefaultStateMutator, RustEngine

    engine = RustEngine()
    by_name = {c.name: c.card_id for c in engine.cards()}
    deck = [by_name[n] for n in DECK]
    env = ClashParallelEnv(engine=RustEngine(),
                           state_mutator=DefaultStateMutator(decks=[deck, deck]))
    obs, _ = env.reset(seed=SEED)

    parser = env.action_parser
    nx, ny = parser.nx, parser.ny
    per = nx * ny
    state = env.battle_state

    seats = {}
    for team, agent in enumerate(env.agents):
        mask = np.asarray(obs[agent]["action_mask"])
        cards = []
        for slot, card_id in enumerate(state.players[team].hand):
            info = parser.cards[card_id]
            grid = mask[1 + slot * per: 1 + (slot + 1) * per].reshape(ny, nx)
            cards.append({"name": info.name, "elixir": int(info.elixir),
                          "count": int(grid.sum()), "grid": grid})
        seats[agent] = {"total": int(mask.sum()), "noop": int(mask[0]), "cards": cards}

    return {"seats": seats, "n_actions": int(parser.space.n), "nx": nx, "ny": ny,
            "tick": int(state.tick), "agents": list(env.agents)}


def _board(d, x0: int, y0: int, grid) -> None:
    """One 18 x 32 tile board, drawn with the acting player's own king at the bottom.

    Solid blocks, no gaps between tiles: the lit region has to stay one readable shape
    when the whole figure is shrunk to a third of its width.
    """
    ny, nx = grid.shape
    d.rectangle([x0 - 2, y0 - 2, x0 + nx * TILE + 1, y0 + ny * TILE + 1], fill=DARK,
                outline=M.BORDER, width=2)
    for ty in range(ny):
        row = grid[ty]
        # ty counts away from the acting player's own king, so row 0 is drawn last.
        top = y0 + (ny - 1 - ty) * TILE
        tx = 0
        while tx < nx:
            if not row[tx]:
                tx += 1
                continue
            run = tx
            while run < nx and row[run]:
                run += 1
            d.rectangle([x0 + tx * TILE, top, x0 + run * TILE - 1, top + TILE - 1],
                        fill=LIT)
            tx = run


def draw(out_path: pathlib.Path) -> str:
    """Compute, draw, save to out_path, return a one-line summary of the numbers on it."""
    m = _measure()
    first, second = m["agents"][0], m["agents"][1]
    a, b = m["seats"][first], m["seats"][second]
    total, n_actions = a["total"], m["n_actions"]
    same = a["total"] == b["total"]

    im, d = M.canvas(W, H)
    f_head = M.theme_font(64)
    f_sub = M.theme_font(38)
    f_name = M.theme_font(36)
    f_count = M.theme_font(48)
    f_sum = M.theme_font(38, mono=True)

    d.text((MARGIN, 14), f"{total} of {n_actions} moves legal", font=f_head, fill=M.TEXT)
    # The boards below are the first seat's. The second seat's total is stated rather than
    # drawn, and it is the measured one either way.
    also = "also" if same else "instead"
    d.text((MARGIN, 96), f"Lit = legal at tick {m['tick']}. "
           f"{second.capitalize()} {also} {b['total']}.", font=f_sub, fill=(196, 196, 208))

    # One board per hand card, evenly spread across the canvas.
    n = len(a["cards"])
    board_w = m["nx"] * TILE
    span = W - 2 * MARGIN
    step = (span - board_w) / (n - 1) if n > 1 else 0
    y_name, y_count, y_board = 150, 192, 250

    for i, card in enumerate(a["cards"]):
        x0 = int(MARGIN + i * step)
        mid = x0 + board_w // 2
        # Card name, then its elixir cost in the game's own amber badge.
        label, cost = card["name"], str(card["elixir"])
        r = 21
        w_label = d.textlength(label + " ", font=f_name)
        left = mid - (w_label + 2 * r) / 2
        d.text((left, y_name), label, font=f_name, fill=M.TEXT)
        cx, cy = left + w_label + r, y_name + 23
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=M.AMBER)
        d.text((cx - d.textlength(cost, font=f_name) / 2, y_name), cost, font=f_name,
               fill=M.BG)
        count = str(card["count"])
        d.text((mid - d.textlength(count, font=f_count) / 2, y_count), count,
               font=f_count, fill=LIT)
        _board(d, x0, y_board, card["grid"])

    # The arithmetic, so a reader can check the headline: no-op plus the four counts.
    terms = " + ".join(str(c["count"]) for c in a["cards"])
    line = f"no-op {a['noop']} + {terms} = {total}"
    font = f_sum
    while d.textlength(line, font=font) > span and font.size > 34:
        font = M.theme_font(font.size - 2, mono=True)
    d.text(((W - d.textlength(line, font=font)) / 2, y_board + m["ny"] * TILE + 16),
           line, font=font, fill=M.TEXT)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)

    parts = ", ".join(f"{c['name']} {c['elixir']}e {c['count']}" for c in a["cards"])
    return (f"{total} of {n_actions} legal at tick {m['tick']} for {first} "
            f"({'red matches' if same else 'seats DIFFER: ' + str(b['total'])}); "
            f"no-op {a['noop']} + {terms} = {total}; {parts}; "
            f"board {m['nx']} x {m['ny']}")
