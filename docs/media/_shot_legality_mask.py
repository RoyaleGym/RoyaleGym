#!/usr/bin/env python3
"""RoyaleGym: the SHAPE of the legal-move mask on the opening step of the Try-it battle.

The README tile beside this figure carries the totals. This figure carries what a total
cannot: where on the board each card in hand may actually be put, and the fact that the
answer is a different shape for a spell, a building and a troop. Four boards, one per hand
card, lit where the move is legal, plus the rule class each shape comes from -- read off the
card table, not typed in here.

The one line of text under the boards is the check: every deploy action in the space is put
through the engine's own ``check_deploy`` and compared with the mask the observation handed
the policy. Those are two separate implementations of the placement rules (a numpy oracle in
``royalegym/action.py`` and the Rust core), so a mismatch is a real failure and not a
restatement. To show the comparison can fail at all, the same comparison is run a second time
against the mask shifted one row, and that count is printed beside the first.

Every number, name and rule class on the canvas is measured in this run. Nothing runs at
import time. Call ``draw(out_path)``.
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # importable from anywhere, like the sibling _shot_ modules
    sys.path.insert(0, str(HERE))

import make_media as M  # noqa: E402  (palette and fonts, shared with the other figures)

# The same eight cards and the same seed the README's Try-it block uses, so this figure is
# about that battle and not about a different one. Cards are named, never numbered: a card
# id is a position in the catalogue and the positions move between card tables.
DECK = ("Knight", "Archer", "Giant", "Minions", "Fireball", "Cannon", "Zap", "Musketeer")
SEED = 0
SEAT = "blue"  # whose boards these are; it is printed on the canvas, never left implied

# 900 shown at about 360 in a README table cell is a 2.5x downscale, so the 34 px floor
# lands at 13.6 px and the headline at 25. Every size below is chosen against that.
W, H = 900, 640
MARGIN = 30
TILE = 9            # px per board tile; 18 x 32 tiles makes a 162 x 288 board
F_FLOOR = 34        # nothing on this canvas is smaller

LIT = M.BLUE
OFF = (46, 46, 60)  # illegal
BOARD_EDGE = M.BORDER


def _blend(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


RIVER_OFF = _blend(OFF, M.RIVER, 0.34)  # illegal AND in the water band


def _measure() -> dict:
    """Reset the Try-it battle and read one seat's mask, then re-check it against the engine."""
    import numpy as np

    from royalegym import ClashParallelEnv, DefaultStateMutator, RustEngine
    from royalegym.env import AGENT_TEAM
    from royalegym.protocol import EMPTY_CARD, DeployStatus, Placement

    # One engine: the ids the deck is resolved against, the card table whose names and rule
    # classes are drawn, and the check_deploy the mask is graded against are then all the
    # same build's.
    engine = RustEngine()
    cards = list(engine.cards())
    by_name = {c.name: c.card_id for c in cards}
    deck = [by_name[n] for n in DECK]
    env = ClashParallelEnv(engine=engine,
                           state_mutator=DefaultStateMutator(decks=[deck, deck]))
    obs, _ = env.reset(seed=SEED)

    parser = env.action_parser
    state = env.battle_state
    assert SEAT in env.agents, f"{SEAT} is not a seat of this env"
    team = AGENT_TEAM[SEAT]
    player = state.players[team]
    # The boards come from obs[SEAT] and the hand from players[team]. Nothing in the API
    # promises those are the same seat, and if they were not, every board would be labelled
    # with the other player's card and the picture would still look right. Check it.
    assert player.team == team, "players[] is not indexed by team; the hand is the wrong seat's"

    n_actions = int(parser.space.n)
    n_slots, ny, nx = parser.mask_plane_shape()
    mask = np.asarray(obs[SEAT]["action_mask"], dtype=bool)
    planes = mask[1:].reshape(n_slots, ny, nx)  # the parser's own documented layout

    # check_deploy reads the engine's own state, so the sweep is only a comparison with the
    # mask if the state the mask was built from is that state.
    assert state == engine.state(), "battle_state is not the engine's current state"
    verdict = np.zeros(n_actions, dtype=bool)
    for action in range(1, n_actions):
        cmd = parser.parse(action, state, team)
        verdict[action] = engine.check_deploy(cmd) == int(DeployStatus.OK)
    disagree = int((mask[1:] != verdict[1:]).sum())
    # The same comparison against a mask that is wrong in the mildest way there is: every
    # board slid one row towards the enemy. If this came back 0 too, the comparison above
    # would be worth nothing.
    nudged = int((np.roll(planes, 1, axis=1).reshape(-1) != verdict[1:]).sum())

    elixir_milli = int(player.elixir_milli)
    hand = []
    for slot, card_id in enumerate(player.hand):
        if card_id == EMPTY_CARD:  # the mask leaves an empty slot dark; draw no board for it
            continue
        info = cards[card_id]
        hand.append({"slot": slot, "name": info.name, "elixir": int(info.elixir),
                     "rule": Placement(info.placement).name.lower().replace("_", " "),
                     "afford": info.elixir * 1000 <= elixir_milli,
                     "count": int(planes[slot].sum()), "grid": planes[slot]})

    # The same four cards sit in the other seat's hand in a different order, so the seats can
    # only be compared card by card. Not drawn -- the picture cannot evidence it -- but said
    # in the return line, where it can be checked.
    other = env.agents[1] if env.agents[0] == SEAT else env.agents[0]
    o_team = AGENT_TEAM[other]
    o_planes = np.asarray(obs[other]["action_mask"], dtype=bool)[1:].reshape(n_slots, ny, nx)
    o_hand = list(state.players[o_team].hand)
    seat_diff, seat_cmp = 0, 0
    for c in hand:
        cid = player.hand[c["slot"]]
        if cid in o_hand:
            seat_diff += int((c["grid"] != o_planes[o_hand.index(cid)]).sum())
            seat_cmp += c["grid"].size

    arena = engine.arena()
    half = arena.hy // ny
    lo, hi = arena.water_half_rows
    water_rows = range(lo // half, hi // half + 1)

    return {"hand": hand, "n_actions": n_actions, "noop": int(mask[0]), "nx": nx, "ny": ny,
            "total": int(mask.sum()), "tick": int(state.tick), "tiles": nx * ny,
            "elixir": elixir_milli / 1000.0, "disagree": disagree, "nudged": nudged,
            "checked": n_actions - 1, "water_rows": set(water_rows), "other": other,
            "seat_diff": seat_diff, "seat_cmp": seat_cmp,
            "skipped": len(player.hand) - len(hand)}


def _board(d, x0: int, y0: int, grid, water_rows) -> None:
    """One 18 x 32 tile board, drawn with the acting player's own king at the bottom.

    Solid blocks, no gaps between tiles: the lit region has to stay one readable shape when
    the whole figure is shrunk to a third of its width. Illegal tiles inside the water band
    are tinted, so a reader can see that the lit region on a troop board stops at the river
    rather than at an arbitrary line.
    """
    ny, nx = grid.shape
    d.rectangle([x0 - 2, y0 - 2, x0 + nx * TILE + 1, y0 + ny * TILE + 1], fill=OFF,
                outline=BOARD_EDGE, width=2)
    for ty in range(ny):
        row = grid[ty]
        # ty counts away from the acting player's own king, so row 0 is drawn at the bottom.
        top = y0 + (ny - 1 - ty) * TILE
        wet = ty in water_rows
        tx = 0
        while tx < nx:
            lit = bool(row[tx])
            if not lit and not wet:
                tx += 1
                continue
            run = tx
            while run < nx and bool(row[run]) == lit:
                run += 1
            d.rectangle([x0 + tx * TILE, top, x0 + run * TILE - 1, top + TILE - 1],
                        fill=LIT if lit else RIVER_OFF)
            tx = run


def _fit(d, text: str, size: int, width: float, mono: bool = False):
    """The largest font at or under ``size``, down to the floor, whose text fits ``width``."""
    while size > F_FLOOR:
        font = M.theme_font(size, mono=mono)
        if d.textlength(text, font=font) <= width:
            return font
        size -= 2
    return M.theme_font(F_FLOOR, mono=mono)


def _centre(d, text, font, cx, y, fill):
    d.text((cx - d.textlength(text, font=font) / 2.0, y), text, font=font, fill=fill)


def draw(out_path: pathlib.Path) -> str:
    """Compute, draw, save to out_path, return a one-line summary of what is on it."""
    m = _measure()
    hand = m["hand"]
    span = W - 2 * MARGIN

    im, d = M.canvas(W, H)
    f_head = M.theme_font(64)
    f_name = M.theme_font(36)
    f_rule = M.theme_font(F_FLOOR)
    f_num = M.theme_font(F_FLOOR, mono=True)

    # The headline is the question the four boards answer. The counts the README tile quotes
    # are not repeated here: a number drawn into a picture is the copy that goes stale.
    d.text((MARGIN, 8), "Where each card may go", font=f_head, fill=M.TEXT)

    # This is one moment of one battle, and at this moment the elixir bar gates nothing, so
    # every dark tile below is a placement rule. Say both rather than let the picture imply
    # that a mask is only ever about territory.
    broke = [c["name"] for c in hand if not c["afford"]]
    sub = (f"{SEAT.capitalize()}'s own frame, opening step. "
           + (f"{', '.join(broke)} priced out." if broke else "Nothing priced out."))
    f_sub = _fit(d, sub, 38, span)
    d.text((MARGIN, 96), sub, font=f_sub, fill=_blend(M.DIM, M.TEXT, 0.45))

    n = len(hand)
    board_w, board_h = m["nx"] * TILE, m["ny"] * TILE
    step = (span - board_w) / (n - 1) if n > 1 else 0
    y_name, y_rule, y_board = 150, 196, 244

    for i, card in enumerate(hand):
        x0 = int(MARGIN + i * step)
        cx = x0 + board_w / 2.0
        _centre(d, card["name"], _fit(d, card["name"], 36, board_w + 40), cx, y_name, M.TEXT)
        # The rule class, straight off the card table: it is why two of these boards are the
        # same shape and the other two are not.
        _centre(d, card["rule"], _fit(d, card["rule"], F_FLOOR, board_w + 40), cx, y_rule,
                M.AMBER)
        _board(d, x0, y_board, card["grid"], m["water_rows"])
        # Two boards can differ by a handful of tiles and look identical at README size, so
        # the count carries what the shape cannot.
        _centre(d, f"{card['count']}/{m['tiles']}", f_num, cx, y_board + board_h + 14, LIT)

    # The check, and the evidence that the check can fail.
    line = (f"all {m['checked']} match the engine     "
            f"shifted one row, {m['nudged']} do not")
    f_chk = _fit(d, line, 36, span)
    if m["disagree"]:
        line = f"{m['disagree']} of {m['checked']} DISAGREE with the engine"
        f_chk = _fit(d, line, 36, span)
    _centre(d, line, f_chk, W / 2.0, 582, M.TEXT if not m["disagree"] else M.RED)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)

    parts = ", ".join(f"{c['name']} {c['elixir']}e {c['rule']} {c['count']}" for c in hand)
    return (f"tick {m['tick']}, {SEAT} at {m['elixir']:.1f} elixir: {m['total']} of "
            f"{m['n_actions']} legal = no-op {m['noop']} + "
            f"{' + '.join(str(c['count']) for c in hand)}; {parts}; "
            f"engine check_deploy over {m['checked']} actions: {m['disagree']} disagree, "
            f"{m['nudged']} after shifting the mask one row; vs {m['other']} card for card: "
            f"{m['seat_diff']} of {m['seat_cmp']} tiles differ; "
            f"empty hand slots skipped {m['skipped']}; board {m['nx']} x {m['ny']}; "
            f"canvas {W} x {H}")
