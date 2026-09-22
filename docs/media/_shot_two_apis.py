#!/usr/bin/env python3
"""Two standard interfaces over one battle, both built for real and compared.

The README tile says you can drive both seats through PettingZoo or one seat through
Gymnasium. This draws that claim after checking it: a ``ClashParallelEnv`` and a
``gym.make`` env are both constructed, both reset, and the three things that ought to
match are compared field by field before anything is written on the canvas.

Nothing is typed in. The Gymnasium id is read out of ``gymnasium.registry`` rather
than quoted, the one drawn is the one that registers, builds and emits no warning, the
action-space size comes off the spaces themselves, and the opening board is compared
entity by entity between the two envs.

WHAT IS NOT CLAIMED. The two envs do not land on the same battle from the same
``seed=``: ``ClashGymEnv.reset`` draws its own sub-seed from the seed it is given, so
the opening HANDS differ even though the board does not. The figure therefore says
"same opening board", which is what was measured, and leans for "one battle" on the
thing that is structurally true instead -- the Gymnasium env holds a
``ClashParallelEnv`` and steps it.

Sized for a README tile: drawn at 1000 px and shown at about 360, so the smallest type
is 34 px and the headline is 66.
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import make_media as M  # noqa: E402  (palette and fonts, shared with the other figures)

# The same battle the README documents: a named deck so the card catalogue's size
# cannot change which cards are dealt, and a fixed seed.
SEED = 0
DECK = ("Knight", "Archer", "Giant", "Minions", "Fireball", "Cannon", "Zap", "Musketeer")

W, H = 1000, 640
MARGIN = 36

F_HEAD = 66
F_API = 50
F_CHIP = 40
F_ROW = 38
F_ID = 36
F_FOOT = 36
F_SMALL = 34  # the floor: nothing on this canvas is smaller

PANEL_Y, PANEL_H = 100, 274
MERGE_Y = 380          # where the two panels join into one
BAR_Y, BAR_H = 412, 146
CHIP_W, CHIP_H = 190, 66


def _blend(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _centre(d, text, font, cx, y, fill):
    d.text((cx - d.textlength(text, font=font) / 2.0, y), text, font=font, fill=fill)


def _dashed(d, box, colour, width=4, dash=13):
    """A dashed outline, which is how a seat someone else drives is drawn."""
    x0, y0, x1, y1 = box
    for x in range(int(x0), int(x1), dash * 2):
        d.line([x, y0, min(x + dash, x1), y0], fill=colour, width=width)
        d.line([x, y1, min(x + dash, x1), y1], fill=colour, width=width)
    for y in range(int(y0), int(y1), dash * 2):
        d.line([x0, y, x0, min(y + dash, y1)], fill=colour, width=width)
        d.line([x1, y, x1, min(y + dash, y1)], fill=colour, width=width)


def _check(d, x, y, size, colour):
    """A tick drawn from two strokes, so no font has to own the glyph."""
    d.line([x, y + size * 0.55, x + size * 0.38, y + size * 0.92], fill=colour, width=6)
    d.line([x + size * 0.38, y + size * 0.92, x + size, y + size * 0.08],
           fill=colour, width=6)


def _merge(d, xs, y0, y1, cx, colour, width=6, head=13):
    """Two stems joined into one arrow: the two APIs meeting on one env.

    A pair of long diagonals turned into grey hairlines at a third of the size, so
    this is drawn as a short bracket instead, which keeps its shape when shrunk.
    """
    mid = (y0 + y1) / 2.0
    for x in xs:
        d.line([x, y0, x, mid], fill=colour, width=width)
    d.line([min(xs), mid, max(xs), mid], fill=colour, width=width)
    d.line([cx, mid, cx, y1 - head], fill=colour, width=width)
    d.polygon([(cx, y1), (cx - head, y1 - head), (cx + head, y1 - head)], fill=colour)


def _pick_gym_id(gym, mutator_fn):
    """The registered id that builds, runs on the real engine and does not warn.

    There are three ids and one of them warns on purpose, because it does not say which
    engine you got. Rather than quoting an id, every ``royalegym/`` id in the registry
    is built and the warnings it raises are counted; the ones that warn are dropped and
    the quietest surviving id whose env holds a ``RustEngine`` is the one drawn.
    """
    import warnings

    from royalegym.rust_engine import RustEngine

    tried = []
    for env_id in sorted(k for k in gym.registry if k.startswith("royalegym/")):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                env = gym.make(env_id, state_mutator=mutator_fn())
            except Exception as exc:  # an id that cannot build here is just skipped
                tried.append((env_id, None, type(exc).__name__))
                continue
        warned = len(caught)
        rust = isinstance(env.unwrapped.parallel.engine, RustEngine)
        tried.append((env_id, warned, "rust" if rust else "other engine"))
        if warned == 0 and rust:
            return env_id, env, tried
        env.close()
    raise RuntimeError(f"no quiet Rust id in the registry: {tried}")


def _board_key(state):
    """Everything on the board, in an order that does not depend on entity ids.

    Uids and hands come out of the engine's generator, and the two envs do not share
    one, so neither belongs in a comparison of the BOARD.
    """
    ents = sorted((e.team, e.tower_slot, e.kind, e.card_id, e.x, e.y, e.hp, e.max_hp)
                  for e in state.entities)
    players = [(p.team, p.elixir_milli, tuple(p.tower_hp), p.crowns) for p in state.players]
    return (state.tick, tuple(ents), tuple(players))


def draw(out_path: pathlib.Path) -> str:
    import gymnasium as gym

    import royalegym  # noqa: F401  (importing it is what registers the gym ids)
    from royalegym import ClashParallelEnv, DefaultStateMutator, RustEngine
    from royalegym.protocol import TowerSlot

    engine = RustEngine()
    by_name = {c.name: c.card_id for c in engine.cards()}
    deck = [by_name[n] for n in DECK]

    def mutator():
        return DefaultStateMutator(decks=[deck, deck])

    # ---- the two interfaces, both built and both reset ----------------------------
    par = ClashParallelEnv(engine=engine, state_mutator=mutator())
    par_obs, _ = par.reset(seed=SEED)
    par_state = par.battle_state

    gym_id, gym_env, tried = _pick_gym_id(gym, mutator)
    gym_obs, _ = gym_env.reset(seed=SEED)
    gym_state = gym_env.unwrapped.parallel.battle_state

    # ---- what the two agree on, measured rather than asserted in prose -------------
    seats_par = list(par.possible_agents)
    gym_seat = gym_env.unwrapped.agent
    gym_other = gym_env.unwrapped.other
    n_seats_par, n_seats_gym = len(seats_par), 1

    par_space = par.action_space(seats_par[0])
    same_space = par_space == gym_env.action_space
    n_actions = int(par_space.n)

    par_keys = sorted(par_obs[gym_seat].keys())
    gym_keys = sorted(gym_obs.keys())
    same_keys = par_keys == gym_keys
    same_shapes = same_keys and all(
        par_obs[gym_seat][k].shape == gym_obs[k].shape for k in gym_keys)
    n_keys = len(gym_keys)

    wraps = type(gym_env.unwrapped.parallel).__name__
    same_class = wraps == type(par).__name__

    same_board = _board_key(par_state) == _board_key(gym_state)
    n_towers = sum(1 for e in par_state.entities if e.tower_slot >= 0)
    king_hp = par_state.players[0].tower_hp[TowerSlot.KING]
    # The hands do NOT match, and saying so here keeps the caption honest.
    hands_match = [list(p.hand) for p in par_state.players] == \
                  [list(p.hand) for p in gym_state.players]

    if not (same_space and same_keys and same_shapes and same_class and same_board):
        raise RuntimeError(
            f"the two APIs disagree: space={same_space} keys={same_keys} "
            f"shapes={same_shapes} class={same_class} board={same_board}")

    ns, _ = gym_id.split("/", 1)
    namespace, short_id = ns + "/", gym_id.split("/", 1)[1]

    # ---- the drawing ---------------------------------------------------------------
    im, d = M.canvas(W, H)
    f_head = M.theme_font(F_HEAD)
    f_api = M.theme_font(F_API)
    f_chip = M.theme_font(F_CHIP)
    f_row = M.theme_font(F_ROW)
    f_id = M.theme_font(F_ID, mono=True)
    f_small = M.theme_font(F_SMALL)
    f_foot = M.theme_font(F_FOOT)
    # DIM grey is fine at 34 px and gone at 13; every secondary colour here is lifted.
    bright = _blend(M.DIM, M.TEXT, 0.45)
    brighter = _blend(M.DIM, M.TEXT, 0.65)

    d.text((MARGIN, 12), "Drive %d seats, or just %d" % (n_seats_par, n_seats_gym),
           font=f_head, fill=M.TEXT)

    pw = (W - 2 * MARGIN - 24) // 2
    panels = (
        (MARGIN, "PettingZoo", ("", "ClashParallelEnv"),
         ((seats_par[0], True), (seats_par[1], True)), ("your bot", "your bot")),
        (MARGIN + pw + 24, "Gymnasium", (namespace, short_id),
         ((gym_seat, True), (gym_other, False)), ("your bot", "scripted")),
    )
    seat_colour = {"blue": M.BLUE, "red": M.RED}
    for px, name, ids, chips, chip_subs in panels:
        d.rounded_rectangle([px, PANEL_Y, px + pw, PANEL_Y + PANEL_H], radius=14,
                            fill=M.PANEL, outline=M.BORDER, width=3)
        cx = px + pw / 2.0
        _centre(d, name, f_api, cx, PANEL_Y + 10, M.TEXT)
        for i, line in enumerate(ids):
            if line:
                _centre(d, line, f_id, cx, PANEL_Y + 70 + i * 40, brighter)
        gap = 18
        cw = 2 * CHIP_W + gap
        cy = PANEL_Y + 150
        for i, (seat, mine) in enumerate(chips):
            x0 = cx - cw / 2.0 + i * (CHIP_W + gap)
            box = [x0, cy, x0 + CHIP_W, cy + CHIP_H]
            col = seat_colour.get(seat, M.TEXT)
            if mine:
                d.rounded_rectangle(box, radius=10, fill=col)
                _centre(d, seat, f_chip, x0 + CHIP_W / 2.0, cy + 10, M.BG)
            else:
                d.rounded_rectangle(box, radius=10, fill=M.BG)
                _dashed(d, box, col)
                _centre(d, seat, f_chip, x0 + CHIP_W / 2.0, cy + 10, col)
            _centre(d, chip_subs[i], f_small, x0 + CHIP_W / 2.0, cy + CHIP_H + 10, bright)

    # The two stems join into one before they reach the bar: the whole point of the tile.
    bar_top = BAR_Y
    _merge(d, [px + pw / 2.0 for px, *_ in panels], MERGE_Y, bar_top - 2, W / 2.0,
           M.BORDER)

    d.rounded_rectangle([MARGIN, bar_top, W - MARGIN, bar_top + BAR_H], radius=14,
                        fill=M.PANEL, outline=M.GREEN, width=3)
    rows = ("same %s underneath" % wraps,
            "same Discrete(%d)" % n_actions,
            "same %d observation keys" % n_keys)
    for i, row in enumerate(rows):
        ry = bar_top + 14 + i * 44
        _check(d, MARGIN + 26, ry + 4, 26, M.GREEN)
        d.text((MARGIN + 74, ry), row, font=f_row, fill=M.TEXT)

    _centre(d, "Same opening board: %d towers, kings %d hp" % (n_towers, king_hp),
            f_foot, W / 2.0, bar_top + BAR_H + 14, brighter)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)
    gym_env.close()

    return ("seats: PettingZoo %s (%d) vs Gymnasium %s +%s scripted (%d); id %s chosen "
            "from %s; Discrete(%d) on both; obs keys %s, shapes equal; gym env wraps %s; "
            "opening board identical (%d towers, king %d hp); opening hands match: %s; "
            "canvas %dx%d"
            % (seats_par, n_seats_par, gym_seat, gym_other, n_seats_gym, gym_id,
               [t[0] for t in tried], n_actions, gym_keys, wraps, n_towers, king_hp,
               hands_match, W, H))
