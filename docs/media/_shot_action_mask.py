#!/usr/bin/env python3
"""The action mask of one real decision, drawn as one board per card in hand.

The action space is ``Discrete(2305)``: index 0 is the no-op and index
``1 + slot * 576 + y * 18 + x`` plays hand slot ``slot`` at tile ``(x, y)`` in the
acting seat's own frame. So the mask minus its first entry is four 18 x 32 pictures,
one per hand card, and that is exactly what this draws.

Nothing here is typed in. The battle is played forward from a fixed seed with both
sides picking uniformly among their own legal moves, the step to draw is chosen by a
rule (the first step at which exactly one card in hand has no legal tile), and every
number on the picture is counted off the mask that step produced.

The figure is sized for a README tile: it is drawn at 900 px and has to stay readable
when a table cell shrinks it to about 360, so the smallest type on it is 34 px and
almost everything the earlier version said has been moved to the docs.
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import make_media as M  # noqa: E402  (palette and fonts, shared with the other figures)

# The same battle the other figures and the README quote: a named deck, so the card
# catalogue's size cannot change which cards are dealt, and a fixed seed.
SEED = 0
# The same example deck the docs use, and for the same reason: every card is in the
# thin slice, the 18 whose behaviour is checked against recordings.
DECK = ("Knight", "Archer", "Giant", "Minions", "Fireball", "Cannon", "Zap", "Musketeer")
NOOP_PROB = 0.7
MAX_STEPS = 60

# Drawing. 900 x 636 shown at 360 is a 2.5x downscale, so 34 px type lands at 13 px and
# the headline at 24. Every size below is chosen against that, not against full size.
W, H = 900, 636
MARGIN = 30
TILE = 9

F_HEAD = 62
F_SUB = 34
F_NAME = 36
F_SMALL = 34  # the floor: nothing on this canvas is smaller

LIT = M.BLUE
LIT2 = (62, 180, 193)  # the checker is low-contrast on purpose: it has to survive the
OFF = (26, 26, 34)     # downscale in the page without moire
OFF2 = (33, 33, 43)
DEAD = (206, 48, 74)   # the tint on a card with no legal tile at all


def _blend(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _centre(d, text, font, cx, y, fill):
    d.text((cx - d.textlength(text, font=font) / 2.0, y), text, font=font, fill=fill)


def _pick_step(env, obs, np):
    """Play forward and stop at the first decision worth drawing.

    Worth drawing means exactly one of the hand cards has an empty plane and the others
    do not, so the figure shows both halves of the rule: one card priced out, the rest
    limited by territory. Returns ``(obs, step, chosen)``.

    There is no fallback to an earlier step. The caller reads the tick, the elixir and
    the hand out of ``env.battle_state``, which is only ever the LATEST step, so handing
    back an older observation would pair one step's mask with another step's numbers.
    If no step matches, the last one is returned with ``chosen`` false and the figure
    says something weaker about it.
    """
    from royalegym import RandomLegalOpponent

    rng = np.random.default_rng(SEED)
    policy = RandomLegalOpponent(noop_prob=NOOP_PROB)
    step = 0
    for _ in range(MAX_STEPS):
        acts = {a: policy.act(obs[a], obs[a]["action_mask"], rng) for a in env.agents}
        obs, *_ = env.step(acts)
        step += 1
        if not env.agents:
            break
        planes = obs["blue"]["mask_planes"]
        per = [int(planes[s].sum()) for s in range(planes.shape[0])]
        if per.count(0) == 1:
            return obs, step, True
    return obs, step, False


def _board(np, plane, arena, dead=False):
    """One mask plane as a PIL image, the seat's own side at the bottom.

    Illegal tiles are dark; the river band and the two bridges are tinted where they are
    illegal, so the reader can tell which way up the board is without a caption saying
    so. A plane with no legal tile at all is tinted towards red.
    """
    from PIL import Image

    off, off2 = OFF, OFF2
    if dead:
        off, off2 = _blend(OFF, DEAD, 0.20), _blend(OFF2, DEAD, 0.20)

    ny, nx = plane.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    checker = (xx + yy) % 2 == 0
    img = np.zeros((ny, nx, 3), dtype=np.uint8)
    on = plane.astype(bool)
    img[on & checker] = LIT
    img[on & ~checker] = LIT2
    img[~on & checker] = off
    img[~on & ~checker] = off2

    half = arena.hy // ny  # half-rows per tile
    lo, hi = arena.water_half_rows
    river = np.zeros(ny, dtype=bool)
    river[lo // half: hi // half + 1] = True
    bridge = np.zeros(nx, dtype=bool)
    for c0, c1 in arena.bridges_half_cols:
        bridge[c0 // half: c1 // half + 1] = True
    wet = river[:, None] & ~on
    img[wet & ~bridge[None, :]] = _blend(off, M.RIVER, 0.34)
    img[wet & bridge[None, :]] = _blend(off, M.BRIDGE, 0.36)

    im = Image.fromarray(img[::-1])  # y = 0 is the seat's own edge; draw it at the bottom
    return im.resize((nx * TILE, ny * TILE), Image.NEAREST)


def draw(out_path: pathlib.Path) -> str:
    import numpy as np

    from royalegym import ClashParallelEnv, DefaultStateMutator, RustEngine
    from royalegym.env import AGENT_TEAM

    engine = RustEngine()
    cards = engine.cards()
    by_name = {c.name: c.card_id for c in cards}
    deck = [by_name[n] for n in DECK]
    env = ClashParallelEnv(engine=engine,
                           state_mutator=DefaultStateMutator(decks=[deck, deck]))
    obs, _ = env.reset(seed=SEED)
    obs, step, chosen = _pick_step(env, obs, np)

    mask = obs["blue"]["action_mask"]
    planes = obs["blue"]["mask_planes"]
    n_slots, ny, nx = planes.shape
    arena = engine.arena()
    state = env.battle_state

    # The mask drawn is blue's; so must the hand and the elixir under it be. Seat order
    # is an engine detail, so look the seat up rather than assuming players[0].
    team = AGENT_TEAM["blue"]
    me = state.players[team]
    assert me.team == team, "players[] is not indexed by team; the hand would be the wrong seat's"

    # The mask planes really are the flat mask minus index 0, so say so by checking it.
    assert np.array_equal(planes.reshape(-1), mask[1:]), "mask_planes is not the flat mask"

    per = [int(planes[s].sum()) for s in range(n_slots)]
    tiles = nx * ny
    total = int(mask.shape[0])
    legal = int(mask.sum())
    noop = int(mask[0])
    elixir = me.elixir_milli / 1000.0
    hand = [cards[c] for c in me.hand]
    tick_ms = env.decision_ms / env.decision_ticks
    seconds = state.tick * tick_ms / 1000.0

    # The one card with no legal tile, and whether the bar is actually why. `per[s] == 0`
    # alone does not prove affordability is the cause, so compare the price to the bar
    # before saying it is.
    zero = [i for i, c in enumerate(hand) if per[i] == 0]
    priced_out = [i for i in zero if hand[i].elixir > elixir]

    im, d = M.canvas(W, H)
    f_head = M.theme_font(F_HEAD)
    f_sub = M.theme_font(F_SUB)
    f_name = M.theme_font(F_NAME)
    f_num = M.theme_font(F_SMALL, mono=True)
    f_note = M.theme_font(F_SMALL)

    d.text((MARGIN, 14), "%d of %d moves legal" % (legal, total), font=f_head, fill=M.TEXT)
    # Brighter than DIM: grey at 34 px is fine at full size and grey at 14 px is not.
    d.text((MARGIN, 94), "lit tiles are the legal placements", font=f_sub,
           fill=_blend(M.DIM, M.TEXT, 0.4))

    bw, bh = nx * TILE, ny * TILE
    gap = (W - 2 * MARGIN - n_slots * bw) // (n_slots - 1)
    name_y, board_y = 148, 200
    for s in range(n_slots):
        x0 = MARGIN + s * (bw + gap)
        cx = x0 + bw / 2.0
        card, count, dead = hand[s], per[s], per[s] == 0
        colour = M.RED if dead else M.TEXT
        _centre(d, "%s %d" % (card.name, card.elixir), f_name, cx, name_y, colour)
        im.paste(_board(np, planes[s], arena, dead), (x0, board_y))
        d.rectangle([x0 - 1, board_y - 1, x0 + bw, board_y + bh],
                    outline=M.RED if dead else M.BORDER, width=2)
        _centre(d, "%d/%d" % (count, tiles), f_num, cx, board_y + bh + 12, colour)

    # The arithmetic behind the headline, so it can be checked off the picture.
    d.text((MARGIN, 540),
           "%d = %d no-op + %s" % (legal, noop, " + ".join(str(c) for c in per)),
           font=f_num, fill=M.TEXT)

    # This step was searched for, so say so, and say what was searched for.
    if chosen and len(priced_out) == 1:
        card = hand[priced_out[0]]
        note = "chosen step: %s needs %d, bar has %.1f" % (card.name, card.elixir, elixir)
        colour = M.AMBER
    elif chosen:
        note = "chosen step: one card has no legal tile"
        colour = M.AMBER
    else:
        note = "step %d of this battle, %.1f elixir" % (step, elixir)
        colour = M.DIM
    d.text((MARGIN, 586), note, font=f_note, fill=colour)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)
    return ("step %d, tick %d (%.1f s), %.3f elixir: %d of %d legal = %d no-op + %s; "
            "hand %s; no legal tile: %s; priced out: %s; canvas %dx%d"
            % (step, state.tick, seconds, elixir, legal, total, noop,
               " + ".join(str(c) for c in per),
               ", ".join("%s/%d" % (c.name, c.elixir) for c in hand),
               ", ".join(hand[i].name for i in zero) or "none",
               ", ".join(hand[i].name for i in priced_out) or "none", W, H))
