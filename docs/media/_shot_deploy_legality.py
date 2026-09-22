#!/usr/bin/env python3
r"""One figure: how many tiles Blue may drop a Giant on, before and after a tower falls.

``Battle.check_deploy`` is asked about the centre of every tile of the action grid, twice
-- once on a fresh battle, once with one of Red's princess towers set to zero hit points.
The two boards are painted with the reason code that came back.

Nothing here is typed in. The arena size, the tile counts, the card name, which princess
tower is the one taken down, which side of the screen that puts it on, and every number in
the headline, the legend and the footer are read off the engine while the picture is drawn.

The card is a troop, and the rule the picture shows is the troop rule. Buildings are held
to their own half and most spells may go anywhere, so a fallen tower moves neither. The
figure says which card it asked about rather than trying to state that on the canvas.
"""

from __future__ import annotations

import json
import pathlib

import make_media as M

# The deck is named rather than drawn at random so the picture is the same on every
# checkout: the card in hand slot 0 is what every tile is being asked about.
DECK = ["Giant", "Knight", "Archers", "Musketeer", "Fireball", "Arrows", "Minions",
        "Goblins"]
HAND_SLOT = 0
TEAM = 0

W, H = 1000, 640
TILE = 12           # pixels per arena tile; even, so half-tile spans land on whole pixels
LEGEND_ROWS = 3     # the largest codes get a row; the footer says how many were left out

# Type is sized for a README tile roughly a third of the page wide, so the figure is read
# at about a third of this size. Nothing here goes below 34.
F_HEAD, F_SUB, F_CAP, F_CODE, F_NUM, F_FOOT = 64, 38, 36, 34, 40, 34

# One flat colour per refusal code. The legal tiles keep the viewer's grass checkerboard
# and the water keeps the viewer's river colour, so the board still reads as a board.
NO_DEPLOY_COL = (146, 108, 214)
OUT_OF_TERRITORY_COL = (150, 60, 55)
OCCUPIED_COL = (120, 120, 135)
INK = (24, 24, 32)


def _battle(tower_hp):
    import royalesim

    b = royalesim.Battle(card_names=DECK, slot_of_k=[[0, 1, 2], [0, 1, 2]])
    b.reset(seed=1, decks=[list(range(8))] * 2, shuffle=0, start_tick=0,
            elixir_milli=[10_000, 10_000], tower_hp=tower_hp, spawns=[])
    return b


def _sweep(b, tiles_x, tiles_y):
    """Ask check_deploy about the centre of every tile. Returns codes[ty][tx]."""
    import royalesim

    t = royalesim.SUBTILE
    return [[b.check_deploy(TEAM, HAND_SLOT, (tx * 2 + 1) * t // 2, (ty * 2 + 1) * t // 2)
             for tx in range(tiles_x)]
            for ty in range(tiles_y)]


def _tile_colour(name, tx, ty):
    if name == "OK":
        return M.GRASS if (tx + ty) % 2 == 0 else M.GRASS2
    if name == "WATER":
        return M.RIVER
    if name == "NO_DEPLOY":
        return NO_DEPLOY_COL
    if name == "OUT_OF_TERRITORY":
        return OUT_OF_TERRITORY_COL
    if name == "OCCUPIED":
        return OCCUPIED_COL
    return M.DIM


def _draw_board(d, x0, y0, names, bridges, river_rows):
    """The mask itself, plus the two bridges so the board is recognisable."""
    ny, nx = len(names), len(names[0])
    for ty in range(ny):
        for tx in range(nx):
            d.rectangle([x0 + tx * TILE, y0 + ty * TILE,
                         x0 + (tx + 1) * TILE - 1, y0 + (ty + 1) * TILE - 1],
                        fill=_tile_colour(names[ty][tx], tx, ty))
    # A bridge is two tiles wide but sits on half-tile boundaries, so it is drawn from the
    # arena's own float spans. Rounding them to whole tiles put the old outlines one tile
    # off, and asymmetrically, on a mirror-symmetric arena.
    for br in bridges:
        d.rectangle([x0 + int(br["x_min"] * TILE), y0 + min(river_rows) * TILE,
                     x0 + int(br["x_max"] * TILE) - 1,
                     y0 + (max(river_rows) + 1) * TILE - 1],
                    outline=M.BRIDGE, width=2)
    d.rectangle([x0 - 2, y0 - 2, x0 + nx * TILE + 1, y0 + ny * TILE + 1],
                outline=M.BORDER, width=2)


def _draw_tower(d, x0, y0, cx, cy, dead):
    """A Red princess tower at tile centre (cx, cy), crossed out when it is at zero hp."""
    r = 1.6 * TILE
    px, py = x0 + cx * TILE, y0 + cy * TILE
    d.ellipse([px - r, py - r, px + r, py + r],
              fill=OCCUPIED_COL if dead else M.RED, outline=M.TEXT, width=2)
    if dead:
        for sx in (-1, 1):
            d.line([px - r * 0.7 * sx, py - r * 0.7, px + r * 0.7 * sx, py + r * 0.7],
                   fill=M.TEXT, width=3)


def _draw_gained(d, x0, y0, gained):
    """Outline the tiles whose answer changed, so the count in the headline is visible."""
    for (tx, ty) in gained:
        px, py = x0 + tx * TILE, y0 + ty * TILE
        if (tx, ty - 1) not in gained:
            d.line([px, py, px + TILE, py], fill=INK, width=3)
        if (tx, ty + 1) not in gained:
            d.line([px, py + TILE, px + TILE, py + TILE], fill=INK, width=3)
        if (tx - 1, ty) not in gained:
            d.line([px, py, px, py + TILE], fill=INK, width=3)
        if (tx + 1, ty) not in gained:
            d.line([px + TILE, py, px + TILE, py + TILE], fill=INK, width=3)


def draw(out_path: pathlib.Path) -> str:
    import royalesim

    reasons = list(royalesim.DEPLOY_REASONS)
    arena = json.loads(royalesim.EMBEDDED_ARENA_JSON)
    nx, ny = arena["tiles"]
    river_rows = arena["river_tile_rows"]
    bridges = arena["bridges"]

    # Board one: the fresh battle, every tower standing.
    base = _battle(None)
    state = json.loads(base.state_json())
    card_name = DECK[state["players"][TEAM]["hand"][HAND_SLOT]]
    full_hp = [list(p["tower_hp"]) for p in state["players"]]

    # Board two: the same battle with one of Red's princess towers already at zero. Which
    # of the two it is, and which side of the screen that puts it on, is read off the
    # engine's own tower positions rather than assumed. "Left" is the viewer's left, which
    # is the side the picture shows it on.
    enemy = 1 - TEAM
    pos = base.tower_positions()[enemy]
    princess_k = [k for k in range(len(full_hp[enemy])) if pos[k][0] != pos[0][0]]
    dead_k = min(princess_k, key=lambda k: pos[k][0])
    # Named in the VIEWER's frame, which is the frame the picture is drawn in. In Red's
    # own frame this tower is on the other side, and saying "Red's left" put the words and
    # the cross on opposite sides of the board.
    side = "left" if pos[dead_k][0] < nx * royalesim.SUBTILE / 2 else "right"
    hurt_hp = [list(row) for row in full_hp]
    hurt_hp[enemy][dead_k] = 0
    hurt = _battle(hurt_hp)

    sweeps = [_sweep(base, nx, ny), _sweep(hurt, nx, ny)]
    boards = [[[reasons[c] for c in row] for row in s] for s in sweeps]
    counts = []
    for names in boards:
        c = {}
        for row in names:
            for n in row:
                c[n] = c.get(n, 0) + 1
        counts.append(c)
    legal = [c.get("OK", 0) for c in counts]
    gained = {(tx, ty) for ty in range(ny) for tx in range(nx)
              if boards[0][ty][tx] != "OK" and boards[1][ty][tx] == "OK"}

    returned = sorted(set(counts[0]) | set(counts[1]),
                      key=lambda n: -max(counts[0].get(n, 0), counts[1].get(n, 0)))
    shown = returned[:LEGEND_ROWS]

    im, d = M.canvas(W, H)
    f_head = M.theme_font(F_HEAD)
    f_sub = M.theme_font(F_SUB)
    f_cap = M.theme_font(F_CAP)
    f_code = M.theme_font(F_CODE, mono=True)
    f_num = M.theme_font(F_NUM)
    f_foot = M.theme_font(F_FOOT)

    x0 = 36
    head = "%d → %d legal tiles" % (legal[0], legal[1])
    d.text((x0, 10), head, font=f_head, fill=M.TEXT)
    d.text((x0 + d.textlength(head, font=f_head) + 24, 10),
           "+%d" % (legal[1] - legal[0]), font=f_head, fill=M.GREEN)
    d.text((x0, 92), "Blue's %s, after the %s-hand enemy princess falls" % (card_name, side),
           font=f_sub, fill=M.DIM)

    board_w, board_h = nx * TILE, ny * TILE
    cap_y, board_y = 148, 192
    xs = (x0, x0 + board_w + 26)
    for i, (names, bx) in enumerate(zip(boards, xs)):
        d.text((bx, cap_y), "%d legal" % legal[i], font=f_cap,
               fill=M.GREEN if i else M.TEXT)
        _draw_board(d, bx, board_y, names, bridges, river_rows)
        for k in princess_k:
            _draw_tower(d, bx, board_y, pos[k][0] / royalesim.SUBTILE,
                        pos[k][1] / royalesim.SUBTILE, dead=bool(i) and k == dead_k)
    # The tiles that opened, on the board where they opened, labelled with their count.
    gx = xs[1]
    _draw_gained(d, gx, board_y, gained)
    gxs = [t[0] for t in gained]
    gys = [t[1] for t in gained]
    d.text((gx + (max(gxs) + 1) * TILE + 10,
            board_y + (min(gys) + max(gys) + 1) * TILE / 2 - F_NUM * 0.62),
           "+%d" % len(gained), font=f_num, fill=M.GREEN)

    # Legend: the codes exactly as DEPLOY_REASONS spells them, with both tile counts.
    lx = xs[1] + board_w + 34
    d.rectangle([lx - 16, board_y - 2, W - 24, board_y + board_h],
                fill=M.PANEL, outline=M.BORDER, width=2)
    d.text((lx, board_y + 14), "Battle.check_deploy", font=f_code, fill=M.TEXT)
    row_h = (board_h - 66) // LEGEND_ROWS
    y = board_y + 64
    for name in shown:
        a, b = counts[0].get(name, 0), counts[1].get(name, 0)
        sw = [lx, y + 2, lx + 30, y + 32]
        if name == "OK":
            d.rectangle(sw, fill=M.GRASS)
            d.rectangle([lx + 15, y + 2, lx + 30, y + 17], fill=M.GRASS2)
            d.rectangle([lx, y + 17, lx + 15, y + 32], fill=M.GRASS2)
        else:
            d.rectangle(sw, fill=_tile_colour(name, 0, 0))
        d.rectangle(sw, outline=M.BORDER, width=1)
        d.text((lx + 42, y), name, font=f_code, fill=M.TEXT)
        if a == b:
            d.text((lx + 42, y + 40), "%d" % a, font=f_num, fill=M.DIM)
        else:
            # Green when the change went Blue's way: more legal tiles, fewer refusals.
            better = (b > a) if name == "OK" else (b < a)
            txt = "%d → " % a
            d.text((lx + 42, y + 40), txt, font=f_num, fill=M.DIM)
            d.text((lx + 42 + d.textlength(txt, font=f_num), y + 40), "%d" % b,
                   font=f_num, fill=M.GREEN if better else M.RED)
        y += row_h

    d.text((x0, board_y + board_h + 12),
           "%d tile centres per board; %d of %d codes shown."
           % (nx * ny, len(shown), len(returned)), font=f_foot, fill=M.DIM)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)

    return ("%d x %d action grid, Blue playing %s: legal %d -> %d tiles (+%d) after Red's "
            "%s princess tower is set to 0 hp, the tiles gained being rows %d-%d, cols "
            "%d-%d; %s"
            % (nx, ny, card_name, legal[0], legal[1], legal[1] - legal[0], side,
               min(gys), max(gys), min(gxs), max(gxs),
               ", ".join("%s %d->%d" % (n, counts[0].get(n, 0), counts[1].get(n, 0))
                         for n in returned)))
