#!/usr/bin/env python3
"""Two standard interfaces over one battle: who supplies which seat, and a mirrored run.

The README tile says you can drive both seats through PettingZoo or one seat through
Gymnasium. The picture's job is the SHAPE of that -- two entry points, two seats each,
and one of the four seats filled by the env itself rather than by you -- which is the
part a sentence keeps having to re-explain.

WHAT IS MEASURED, and how it could fail. A ``ClashGymEnv`` is built and reset, and the
seed its ``reset`` drew for the ``ClashParallelEnv`` underneath is read back off that
env and handed to a second, independent ``ClashParallelEnv`` with its own engine. Both
are then stepped through their own API with the same pair of actions per step -- mine
for the learner seat, and the ones the opponent object actually returned for the other,
recorded as it returns them -- and after every step the two BattleState objects are
compared whole (``==`` on the struct: entities, uids, hands, elixir, towers, spells;
nothing pruned) along with the observation each API hands back for the learner seat.

That comparison can fail, and the figure shows it failing: the same battle is replayed a
third time with the two seats' actions exchanged, and the step at which it parts company
with the mirrored run is drawn beside a cross. A comparison that could not tell those
two runs apart would not be worth drawing.

Nothing is typed in. The seat names, their count, the class names, the opponent keyword
and the number of steps, cards and the divergence step all come out of the objects.
The one claim the picture makes without evidence beside it is the shared class name in
the bar, which is structure, not a measurement: ``ClashGymEnv`` builds a
``ClashParallelEnv``, so no tick is drawn against it.

Sized for a README tile: drawn at 1000 px and shown at about 360, so the smallest type
is 34 px and the headline is 66.
"""

from __future__ import annotations

import inspect
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
STEPS = 60        # the cap on the mirrored run; it stops early if the battle ends
NOOP_PROB = 0.5   # both seats: often enough that the run is not 60 no-ops

W, H = 1000, 640
MARGIN = 36

F_HEAD = 66
F_API = 50
F_CHIP = 40
F_ROW = 36
F_ID = 36
F_SMALL = 34  # the floor: nothing on this canvas is smaller

PANEL_Y, PANEL_H = 104, 264
CHIP_W, CHIP_H = 174, 66
BAR_Y, BAR_H = 404, 166


def _blend(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _centre(d, text, font, cx, y, fill):
    d.text((cx - d.textlength(text, font=font) / 2.0, y), text, font=font, fill=fill)


def _dashed(d, box, colour, width=4, dash=13):
    """A dashed outline, which is how a seat the env fills is drawn."""
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


def _cross(d, x, y, size, colour):
    """The tick's opposite, for the run that is supposed to come out different."""
    d.line([x, y + size * 0.1, x + size, y + size * 0.9], fill=colour, width=6)
    d.line([x + size, y + size * 0.1, x, y + size * 0.9], fill=colour, width=6)


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


def _opponent_kwarg(cls) -> str:
    """The name of the constructor slot that takes the other seat's policy.

    Read off the signature rather than quoted, by looking for the parameter whose
    annotation mentions the ``Opponent`` protocol.
    """
    params = inspect.signature(cls.__init__).parameters
    names = [n for n, p in params.items() if "Opponent" in str(p.annotation)]
    if len(names) != 1:
        raise RuntimeError(f"cannot tell which {cls.__name__} slot takes an opponent: {names}")
    return names[0]


class _Recorded:
    """The opponent the env was given, with the action it returns kept.

    The other seat's action has to be known to replay the step through the other API,
    and asking the policy again would not be the same thing: it is stochastic and it
    draws from the env's generator. So the action is taken as it is handed over.
    """

    def __init__(self, inner):
        self.inner = inner
        self.actions: list[int] = []

    def act(self, obs, mask, rng) -> int:
        action = int(self.inner.act(obs, mask, rng))
        self.actions.append(action)
        return action


def _obs_equal(a, b) -> bool:
    import numpy as np

    return sorted(a) == sorted(b) and all(np.array_equal(a[k], b[k]) for k in a)


def draw(out_path: pathlib.Path) -> str:
    import gymnasium as gym
    import numpy as np
    from pettingzoo.utils.env import ParallelEnv

    from royalegym import (
        ClashGymEnv,
        ClashParallelEnv,
        DefaultStateMutator,
        RandomLegalOpponent,
        RustEngine,
    )
    from royalegym.protocol import DeployStatus

    engine = RustEngine()
    by_name = {c.name: c.card_id for c in engine.cards()}
    deck = [by_name[n] for n in DECK]

    def mutator():
        return DefaultStateMutator(decks=[deck, deck])

    # ---- the two interfaces, both built and both reset ----------------------------
    opp_kwarg = _opponent_kwarg(ClashGymEnv)
    opponent = _Recorded(RandomLegalOpponent(noop_prob=NOOP_PROB))
    gym_env = ClashGymEnv(engine=RustEngine(), state_mutator=mutator(),
                          **{opp_kwarg: opponent})
    gym_obs, _ = gym_env.reset(seed=SEED)

    # The battle the Gymnasium env chose for itself. Its reset draws a sub-seed from the
    # seed it is given, so seed=SEED on a second env would be a DIFFERENT battle; the
    # seed that was actually used is read back off the env underneath and passed on.
    sub_seed = getattr(gym_env.parallel, "_np_random_seed", None)
    if sub_seed is None:
        raise RuntimeError("cannot read back the seed the Gymnasium env drew for itself")
    par = ClashParallelEnv(engine=engine, state_mutator=mutator())
    par_obs, _ = par.reset(seed=sub_seed)

    # The labels on the two panels, each true of the object drawn under it.
    if not isinstance(par, ParallelEnv):
        raise RuntimeError("the left panel says PettingZoo and the object is not one")
    if not isinstance(gym_env, gym.Env):
        raise RuntimeError("the right panel says Gymnasium and the object is not one")

    seats = list(par.possible_agents)
    mine, theirs = gym_env.agent, gym_env.other
    if sorted(seats) != sorted([mine, theirs]):
        raise RuntimeError(f"the Gymnasium seats {mine}/{theirs} are not {seats}")
    n_seats_par = len(seats)
    # One of the two seats is yours through the Gymnasium API; the loop below checks
    # that the env really did fill the other one, every step.
    n_seats_gym = len([s for s in seats if s == mine])

    # ---- the mirrored run ----------------------------------------------------------
    rng = np.random.default_rng(SEED)
    policy = RandomLegalOpponent(noop_prob=NOOP_PROB)
    my_actions: list[int] = []
    mirror: list[object] = [par.battle_state]
    cards = 0
    if par.battle_state != gym_env.parallel.battle_state:
        raise RuntimeError("the two envs did not start on the same battle")
    for _ in range(STEPS):
        action = int(policy.act(gym_obs, gym_obs["action_mask"], rng))
        my_actions.append(action)
        gym_obs, _, term, trunc, info = gym_env.step(action)
        par_obs, *_ = par.step({mine: action, theirs: opponent.actions[-1]})
        cards += sum(int(s) == DeployStatus.OK for s in
                     (info["deploy_status"], info["opponent_deploy_status"]))
        if not _obs_equal(par_obs[mine], gym_obs):
            raise RuntimeError(f"observations differ at step {len(my_actions)}")
        if par.battle_state != gym_env.parallel.battle_state:
            raise RuntimeError(f"states differ at step {len(my_actions)}")
        mirror.append(par.battle_state)
        if term or trunc:
            break
    n_steps = len(my_actions)
    if len(opponent.actions) != n_steps:
        raise RuntimeError("the env did not fill the other seat on every step")
    if cards == 0:
        raise RuntimeError("no card was played: the run compares two idle battles")
    # An equality that cannot say no proves nothing, so it is asked something it must
    # refuse: the same board seen from the two seats is not the same observation.
    if _obs_equal(par_obs[mine], par_obs[theirs]):
        raise RuntimeError("the observation comparison cannot tell the two seats apart")

    # ---- the control: the same comparison, on a run it must NOT pass ----------------
    # If the actions had landed on the other seat -- the one mistake a matching pair of
    # states could plausibly hide -- this is where the two would part company.
    swapped = ClashParallelEnv(engine=RustEngine(), state_mutator=mutator())
    swapped.reset(seed=sub_seed)
    split = None
    for i, action in enumerate(my_actions, start=1):
        swapped.step({theirs: action, mine: opponent.actions[i - 1]})
        if swapped.battle_state != mirror[i]:
            split = i
            break
    if split is None:
        raise RuntimeError("swapping the seats changed nothing: the comparison is blind")

    # ---- the drawing ---------------------------------------------------------------
    im, d = M.canvas(W, H)
    f_head = M.theme_font(F_HEAD)
    f_api = M.theme_font(F_API)
    f_chip = M.theme_font(F_CHIP)
    f_row = M.theme_font(F_ROW)
    f_id = M.theme_font(F_ID, mono=True)
    f_small = M.theme_font(F_SMALL)
    # DIM grey is fine at 34 px and gone at 13; every secondary colour here is lifted.
    bright = _blend(M.DIM, M.TEXT, 0.45)
    brighter = _blend(M.DIM, M.TEXT, 0.65)

    d.text((MARGIN, 10), "Drive %d seats, or just %d" % (n_seats_par, n_seats_gym),
           font=f_head, fill=M.TEXT)

    pw = (W - 2 * MARGIN - 24) // 2
    shared = type(par).__name__
    panels = (
        (MARGIN, "PettingZoo", shared,
         [(s, True) for s in seats]),
        (MARGIN + pw + 24, "Gymnasium", type(gym_env).__name__,
         [(s, s == mine) for s in seats]),
    )
    seat_colour = {"blue": M.BLUE, "red": M.RED}
    for px, name, cls, chips in panels:
        d.rounded_rectangle([px, PANEL_Y, px + pw, PANEL_Y + PANEL_H], radius=14,
                            fill=M.PANEL, outline=M.BORDER, width=3)
        cx = px + pw / 2.0
        _centre(d, name, f_api, cx, PANEL_Y + 8, M.TEXT)
        _centre(d, cls, f_id, cx, PANEL_Y + 78, brighter)
        gap = 18
        cw = len(chips) * CHIP_W + (len(chips) - 1) * gap
        cy = PANEL_Y + 146
        for i, (seat, yours) in enumerate(chips):
            x0 = cx - cw / 2.0 + i * (CHIP_W + gap)
            box = [x0, cy, x0 + CHIP_W, cy + CHIP_H]
            col = seat_colour.get(seat, M.TEXT)
            if yours:
                d.rounded_rectangle(box, radius=10, fill=col)
                _centre(d, seat, f_chip, x0 + CHIP_W / 2.0, cy + 10, M.BG)
                sub, sub_col = "you", bright
            else:
                d.rounded_rectangle(box, radius=10, fill=M.BG)
                _dashed(d, box, col)
                _centre(d, seat, f_chip, x0 + CHIP_W / 2.0, cy + 10, col)
                sub, sub_col = opp_kwarg + "=", col
            # Centred on its chip, but never over the panel edge: the widest of these
            # labels is a keyword and the outer chip is close to the border.
            half = d.textlength(sub, font=f_small) / 2.0
            scx = min(max(x0 + CHIP_W / 2.0, px + 14 + half), px + pw - 14 - half)
            _centre(d, sub, f_small, scx, cy + CHIP_H + 10, sub_col)

    # The two stems join into one before they reach the bar: the whole point of the tile.
    _merge(d, [px + pw / 2.0 for px, *_ in panels], PANEL_Y + PANEL_H, BAR_Y - 2,
           W / 2.0, M.BORDER)

    d.rounded_rectangle([MARGIN, BAR_Y, W - MARGIN, BAR_Y + BAR_H], radius=14,
                        fill=M.PANEL, outline=M.BORDER, width=3)
    _centre(d, shared, f_id, W / 2.0, BAR_Y + 8, brighter)
    rows = ((_check, M.GREEN, "obs + state match: %d random legal steps, %d cards"
             % (n_steps, cards)),
            (_cross, M.AMBER, "swap the seats and they split at step %d" % split))
    for i, (glyph, colour, row) in enumerate(rows):
        ry = BAR_Y + 62 + i * 48
        glyph(d, MARGIN + 26, ry + 4, 26, colour)
        d.text((MARGIN + 74, ry), row, font=f_row, fill=M.TEXT)

    _centre(d, "one battle: the seed the Gymnasium env drew, passed on",
            f_small, W / 2.0, BAR_Y + BAR_H + 12, brighter)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)
    gym_env.close()
    par.close()
    swapped.close()

    return ("seats %s: PettingZoo drives %d, Gymnasium %d + %s from a %s; states compared "
            "whole (BattleState ==, uids and hands included) and observations key by key "
            "for %d steps with %d cards played, all equal; control: the same actions on "
            "the swapped seats differ at step %d; canvas %dx%d"
            % (seats, n_seats_par, n_seats_gym, theirs,
               type(opponent.inner).__name__, n_steps, cards, split, W, H))
