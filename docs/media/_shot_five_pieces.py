"""Showcase tile: the five swappable pieces of ``ClashParallelEnv``.

Everything on the canvas is read back from the package at draw time: the slot names
come from the constructor signature, the default class of each slot comes from an env
that was handed nothing, and the "+N more" counts come from walking every module in
``royalegym`` for concrete subclasses of that slot's base class.
"""

from __future__ import annotations

import importlib
import inspect
import pathlib
import pkgutil
import warnings

import make_media as M

# The constructor parameters that take a swappable piece, in signature order, with the
# deprecated aliases (``terminal_conditions``, ``state_setter``) left out. Checked
# against the live signature below so a rename cannot leave this stale.
PIECE_SLOTS = (
    "obs_builder",
    "action_parser",
    "reward_fn",
    "termination_cond",
    "truncation_cond",
    "state_mutator",
)

# slot -> (attribute the env stores it on, base class, plain-English phrase, colour key)
ROLES = (
    ("obs_builder", "obs_builder", "what it sees", "BLUE"),
    ("action_parser", "action_parser", "what a move means", "BRIDGE"),
    ("reward_fn", "reward_fn", "what it's paid for", "GREEN"),
    ("state_mutator", "state_mutator", "how a match starts", "GRASS2"),
    ("termination_cond", "termination", "when it ends", "RED"),
)

W, H = 1000, 640


def _bases():
    from royalegym.action import ActionParser
    from royalegym.done_condition import DoneCondition
    from royalegym.obs import ObsBuilder
    from royalegym.reward import RewardFunction
    from royalegym.state_mutator import StateMutator

    return {
        "obs_builder": ObsBuilder,
        "action_parser": ActionParser,
        "reward_fn": RewardFunction,
        "termination_cond": DoneCondition,
        "truncation_cond": DoneCondition,
        "state_mutator": StateMutator,
    }


def _concrete(base) -> list[str]:
    """Every instantiable subclass of ``base`` that ships inside ``royalegym``."""
    import royalegym

    found: set[str] = set()
    for mod_info in pkgutil.iter_modules(royalegym.__path__):
        mod = importlib.import_module(f"royalegym.{mod_info.name}")
        for obj in vars(mod).values():
            if (
                inspect.isclass(obj)
                and issubclass(obj, base)
                and obj is not base
                and not inspect.isabstract(obj)
                and obj.__module__.startswith("royalegym.")
            ):
                found.add(obj.__name__)
    return sorted(found)


def measure() -> dict:
    from royalegym.env import ClashParallelEnv

    sig = inspect.signature(ClashParallelEnv.__init__)
    for slot in PIECE_SLOTS:
        if slot not in sig.parameters:
            raise AssertionError(f"ClashParallelEnv no longer takes {slot!r}")
        if sig.parameters[slot].default is not None:
            raise AssertionError(f"{slot} no longer defaults to None")

    # An env handed nothing at all: the defaults are whatever it kept.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "no engine chosen" notice; irrelevant here
        env = ClashParallelEnv()

    bases = _bases()
    rows = []
    for slot, attr, phrase, colour in ROLES:
        got = getattr(env, attr)
        if got is None:
            raise AssertionError(f"{slot} left {attr} empty")
        shipped = _concrete(bases[slot])
        default = type(got).__name__
        if default not in shipped:
            raise AssertionError(f"default {default} is not a shipped {bases[slot].__name__}")
        rows.append(
            {
                "slot": slot,
                "phrase": phrase,
                "default": default,
                "others": len(shipped) - 1,
                "colour": colour,
            }
        )

    # The sixth slot: an episode-length cap the env leaves switched off.
    if env.truncation is not None:
        raise AssertionError("truncation_cond now has a default")

    return {
        "rows": rows,
        "slots": len(PIECE_SLOTS),
        "alternatives": sum(r["others"] for r in rows),
    }


def draw(out_path: pathlib.Path) -> str:
    data = measure()
    rows = data["rows"]
    pieces = len(rows)
    alts = data["alternatives"]

    im, d = M.canvas(W, H)
    colours = {k: getattr(M, k) for k in ("BLUE", "BRIDGE", "GREEN", "GRASS2", "RED")}

    f_head = M.theme_font(66)
    f_sub = M.theme_font(36)
    f_phrase = M.theme_font(38)
    f_class = M.theme_font(36, mono=True)
    f_badge = M.theme_font(34)
    f_foot = M.theme_font(36)

    pad = 40
    d.text((pad, 22), f"{pieces} pieces, {alts} alternatives", font=f_head, fill=M.TEXT)
    d.text((pad, 106), "what you get if you pass nothing", font=f_sub, fill=M.DIM)
    d.line((pad, 164, W - pad, 164), fill=M.BORDER, width=2)

    # Columns sized from the real strings, so a longer class name cannot collide.
    phrase_w = max(d.textlength(r["phrase"], font=f_phrase) for r in rows)
    class_w = max(d.textlength(r["default"], font=f_class) for r in rows)
    x_phrase = pad + 20
    x_class = x_phrase + phrase_w + 20
    badge_right = W - pad - 10
    badge_w = max(d.textlength(f"+{r['others']} more", font=f_badge) for r in rows)
    if x_class + class_w > badge_right - badge_w - 28:
        raise AssertionError("columns collide; shorten a phrase or drop the badge word")

    # A hairline between the class column and the count, so a long class name and a
    # right-aligned "+N more" cannot read as one string once the tile is shrunk.
    x_rule = badge_right - badge_w - 14

    top, row_h, gap = 176, 72, 8
    for i, r in enumerate(rows):
        y = top + i * (row_h + gap)
        accent = colours[r["colour"]]
        d.rounded_rectangle((pad, y, W - pad, y + row_h), radius=10, fill=M.PANEL)
        d.rounded_rectangle((pad, y, pad + 12, y + row_h), radius=5, fill=accent)

        mid = y + row_h // 2
        bb = d.textbbox((0, 0), r["phrase"], font=f_phrase)
        d.text((x_phrase, mid - (bb[1] + bb[3]) // 2), r["phrase"], font=f_phrase, fill=M.TEXT)
        bb = d.textbbox((0, 0), r["default"], font=f_class)
        d.text((x_class, mid - (bb[1] + bb[3]) // 2), r["default"], font=f_class, fill=accent)

        d.line((x_rule, y + 16, x_rule, y + row_h - 16), fill=M.BORDER, width=2)

        badge = f"+{r['others']} more"
        bw = d.textlength(badge, font=f_badge)
        bb = d.textbbox((0, 0), badge, font=f_badge)
        d.text((badge_right - bw, mid - (bb[1] + bb[3]) // 2), badge, font=f_badge, fill=M.DIM)

    foot = f"A {data['slots']}th slot caps episode length. Off by default."
    d.text((pad, H - 58), foot, font=f_foot, fill=M.DIM)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)

    listed = ", ".join(f"{r['slot']}={r['default']}(+{r['others']})" for r in rows)
    return (
        f"{pieces} pieces / {alts} shipped alternatives; {listed}; "
        f"{data['slots']} piece slots on ClashParallelEnv.__init__, truncation_cond default None"
    )
