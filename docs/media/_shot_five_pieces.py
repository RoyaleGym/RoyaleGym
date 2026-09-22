"""Showcase tile: the swappable pieces of ``ClashParallelEnv``.

Nothing on the canvas is a list kept by hand.

The slots come from the constructor signature: every parameter annotated with one of
the piece base classes is probed by building an env that passes only that parameter,
and two parameters whose instance lands on the same attribute are one slot, which is
how the deprecated aliases fold away instead of being named here. A slot the drawing
has no row for stops the build.

The default in each row is whatever an env handed nothing kept on that attribute.

The dots are the shipped concrete subclasses the slot will take, and "will take" is the
constructor's own answer rather than a rule written here: every candidate that can be
built without being handed real data is built and offered to the constructor, and the
ones it turns away are dropped. A candidate that cannot be built that cheaply inherits
the answer the constructor already gave for its branch of the class hierarchy, which is
the nearest thing to asking that does not involve inventing data for it.

That is what keeps the episode-cap conditions out of the "when it ends" row. The
refusals are required to be non-empty: a constructor that had stopped checking would
agree to everything, which would draw a wider row and mean nothing, so that case stops
the build instead of passing for evidence.

Only the plain-English phrase and the accent colour of each row are written here.
"""

from __future__ import annotations

import importlib
import inspect
import pathlib
import pkgutil

import make_media as M

# attribute the env stores the piece on -> (plain-English phrase, colour key).
# The phrasing and the colour are the only editorial choices on the tile; every slot
# discovered below must appear here or ``measure`` raises.
ROLES = {
    "obs_builder": ("what it sees", "BLUE"),
    "action_parser": ("what a move means", "BRIDGE"),
    "reward_fn": ("what it's paid for", "GREEN"),
    "state_mutator": ("how a match starts", "GRASS2"),
    "termination": ("when it ends", "RED"),
    "truncation": ("how long it may run", "AMBER"),
}

W, H = 1000, 640


def _bases() -> dict[str, type]:
    from royalegym.action import ActionParser
    from royalegym.done_condition import DoneCondition
    from royalegym.obs import ObsBuilder
    from royalegym.reward import RewardFunction
    from royalegym.state_mutator import StateMutator

    return {c.__name__: c for c in (ObsBuilder, ActionParser, RewardFunction,
                                    DoneCondition, StateMutator)}


def _concrete(base: type) -> list[type]:
    """Every instantiable subclass of ``base`` that ships inside ``royalegym``."""
    import royalegym

    found: set[type] = set()
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
                found.add(obj)
    return sorted(found, key=lambda c: c.__name__)


def _build(cls: type):
    """A throwaway instance of ``cls``, or None if it cannot be built cheaply.

    Only ever used as a probe: it is handed to the constructor to see whether the
    constructor takes it, and then dropped. No value it carries reaches the canvas.
    """
    try:
        return cls()
    except TypeError:
        pass
    kwargs: dict[str, object] = {}
    for p in list(inspect.signature(cls.__init__).parameters.values())[1:]:
        if p.default is not inspect.Parameter.empty:
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        ann = str(p.annotation)
        if ann.startswith(("Sequence", "list", "tuple[")) or "Mapping" in ann:
            kwargs[p.name] = []
        elif ann == "int":
            kwargs[p.name] = 1
        elif ann == "float":
            kwargs[p.name] = 1.0
        else:
            return None
    try:
        return cls(**kwargs)
    except Exception:
        return None


def _pieces(env, bases) -> dict[str, object]:
    """The env attributes that hold a piece, by name."""
    tup = tuple(bases.values())
    return {k: v for k, v in vars(env).items()
            if v is None or isinstance(v, tup)}


def _lands_on(slot: str, value, bases) -> list[str] | None:
    """Where ``value`` ends up when it is the only thing passed to the constructor.

    Returns the attribute names holding the exact object passed (``None`` when the
    constructor refuses it). A ``TypeError`` is a refusal; anything else means the
    probe object was unusable and the answer is unknown, reported as an empty list.
    """
    from royalegym.env import ClashParallelEnv

    wanted = set(map(id, value)) if isinstance(value, list) else {id(value)}
    try:
        env = ClashParallelEnv(**{slot: value})
    except TypeError:
        return None
    except Exception:
        return []
    return sorted(k for k, v in _pieces(env, bases).items() if id(v) in wanted)


def _branches(cls: type, base: type) -> set[type]:
    """The abstract classes between ``base`` and ``cls``: what a refusal generalises to."""
    return {c for c in cls.__mro__
            if c is not cls and c is not base
            and issubclass(c, base) and inspect.isabstract(c)}


def measure() -> dict:
    from royalegym.env import ClashParallelEnv

    bases = _bases()
    sig = inspect.signature(ClashParallelEnv.__init__)

    # 1. Candidate slots: every parameter that is annotated with a piece base class.
    candidates: list[tuple[str, type, bool]] = []
    for name, p in sig.parameters.items():
        ann = str(p.annotation)
        for base_name, base in bases.items():
            if base_name in ann:
                candidates.append((name, base, ann.startswith("Sequence")))
                break
    if not candidates:
        raise AssertionError("no piece parameters found on ClashParallelEnv.__init__")

    # 2. What each candidate accepts, and where the thing it accepted ended up. Two
    #    parameters that land on the same attribute are one slot under two names.
    pool = {bn: [i for i in (_build(c) for c in _concrete(b)) if i is not None]
            for bn, b in bases.items()}
    canonical: dict[str, str] = {}   # attribute -> first parameter name that reaches it
    for name, base, seq in candidates:
        where: list[str] | None = None
        for probe in pool[base.__name__]:
            got = _lands_on(name, [probe] if seq else probe, bases)
            if got:
                where = got
                break
        if where is None:
            raise AssertionError(f"{name} took none of the shipped {base.__name__}s")
        if len(where) != 1:
            raise AssertionError(f"{name} set {where}; one parameter, one attribute")
        canonical.setdefault(where[0], name)

    missing = [a for a in canonical if a not in ROLES]
    if missing:
        raise AssertionError(f"ClashParallelEnv grew {missing}; the tile has no row for it")

    # 3. The defaults: an env handed nothing at all.
    env = ClashParallelEnv()
    defaults = _pieces(env, bases)

    # 4. What each slot will take: every shipped subclass of its base, offered to the
    #    constructor one at a time. A class that cannot be built cheaply gets the answer
    #    the constructor already gave for its branch of the hierarchy.
    refused: dict[str, set[type]] = {}
    rows = []
    for attr, slot in canonical.items():
        base = dict((n, b) for n, b, _ in candidates)[slot]
        seq = dict((n, s) for n, _, s in candidates)[slot]
        turned_away: set[type] = set()
        verdicts: list[tuple[type, bool | None]] = []
        for cls in _concrete(base):
            probe = _build(cls)
            ok: bool | None = None
            if probe is not None:
                got = _lands_on(slot, [probe] if seq else probe, bases)
                if got is None:
                    ok = False
                    turned_away |= _branches(cls, base)
                elif got:
                    ok = got == [attr]
            verdicts.append((cls, ok))
        refused[attr] = turned_away
        takes = [cls.__name__ for cls, ok in verdicts
                 if ok or (ok is None and not any(issubclass(cls, r) for r in turned_away))]
        got = defaults.get(attr)
        default = type(got).__name__ if got is not None else None
        # The default leads its own row, so the bright dot lines up down the tile and the
        # rest of the strip reads as the depth of the bench behind it.
        takes.sort(key=lambda n: (n != default, n))
        if default is not None and default not in takes:
            raise AssertionError(f"{attr} defaults to {default}, which it will not take")
        phrase, colour = ROLES[attr]
        rows.append({"attr": attr, "slot": slot, "phrase": phrase, "colour": colour,
                     "default": default, "takes": takes})
    order = list(ROLES)
    rows.sort(key=lambda r: order.index(r["attr"]))

    # The refusals are the only reason any row is narrower than "every shipped subclass".
    # A constructor that had stopped checking would hand back agreement everywhere, which
    # would look like the same picture and mean nothing, so that case stops the build.
    if not any(refused.values()):
        raise AssertionError("no slot refused any shipped class; the probe proves nothing")

    return {"rows": rows, "refused": {k: sorted(c.__name__ for c in v)
                                      for k, v in refused.items() if v}}


def _dot(d, cx, cy, r, *, fill=None, outline=None):
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=outline,
              width=0 if fill else 3)


def draw(out_path: pathlib.Path) -> str:
    data = measure()
    rows = data["rows"]

    im, d = M.canvas(W, H)
    colours = {k: getattr(M, k) for k in ("BLUE", "BRIDGE", "GREEN", "GRASS2", "RED", "AMBER")}

    f_head = M.theme_font(62)
    f_sub = M.theme_font(34)
    f_class = M.theme_font(34, mono=True)
    f_leg = M.theme_font(34)

    pad = 36
    on = sum(1 for r in rows if r["default"] is not None)
    d.text((pad, 12), f"{on} pieces set, {len(rows) - on} left off", font=f_head, fill=M.TEXT)
    d.text((pad, 92), "the class each slot uses if you pass nothing", font=f_sub, fill=M.DIM)
    d.line((pad, 142, W - pad, 142), fill=M.BORDER, width=2)

    # Columns sized from the real strings. A longer class name spends the phrase column's
    # slack first and then tightens the dot pitch; nothing here is a measurement taken once
    # and written down, so the tile reflows for a rename instead of colliding or refusing.
    labels = {r["attr"]: (r["default"] or "(none)") for r in rows}
    class_w = max(d.textlength(t, font=f_class) for t in labels.values())
    most = max(len(r["takes"]) for r in rows)
    x_phrase = pad + 16
    for size in (36, 34):  # 34 is the floor the tile is legible at when it is shrunk
        f_phrase = M.theme_font(size)
        phrase_w = max(d.textlength(r["phrase"], font=f_phrase) for r in rows)
        x_class = x_phrase + phrase_w + 20
        x_dots = x_class + class_w + 26
        pitch = int((W - pad - x_dots) // most)
        if pitch >= 14:
            break
    if pitch < 14:
        widest = max(labels.values(), key=len)
        raise AssertionError(
            f"{widest!r} has outgrown the row (dot pitch would be {pitch}px); "
            f"shorten a phrase in ROLES to give the class column more room"
        )
    pitch = min(pitch, 26)
    r_dot = max(5, pitch // 2 - 3)

    top, row_h, gap = 154, 64, 6
    for i, r in enumerate(rows):
        y = top + i * (row_h + gap)
        accent = colours[r["colour"]]
        off = r["default"] is None
        d.rounded_rectangle((pad, y, W - pad, y + row_h), radius=10, fill=M.PANEL)
        d.rounded_rectangle((pad, y, pad + 12, y + row_h), radius=5,
                            fill=M.BORDER if off else accent)

        mid = y + row_h // 2
        bb = d.textbbox((0, 0), r["phrase"], font=f_phrase)
        d.text((x_phrase, mid - (bb[1] + bb[3]) // 2), r["phrase"], font=f_phrase,
               fill=M.DIM if off else M.TEXT)
        label = labels[r["attr"]]
        bb = d.textbbox((0, 0), label, font=f_class)
        d.text((x_class, mid - (bb[1] + bb[3]) // 2), label, font=f_class,
               fill=M.BORDER if off else accent)

        for j, name in enumerate(r["takes"]):
            cx = x_dots + r_dot + j * pitch
            _dot(d, cx, mid, r_dot, fill=accent if name == r["default"] else M.BORDER)

    y_leg = H - 46
    x = pad + 6
    _dot(d, x, y_leg, r_dot, fill=M.TEXT)
    x += r_dot + 12
    d.text((x, y_leg - 22), "the default", font=f_leg, fill=M.DIM)
    x += d.textlength("the default", font=f_leg) + 40 + r_dot
    _dot(d, x, y_leg, r_dot, fill=M.BORDER)
    x += r_dot + 12
    d.text((x, y_leg - 22), "also ships for that slot", font=f_leg, fill=M.DIM)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)

    listed = "; ".join(
        f"{r['slot']}->{r['attr']}={r['default']} of {len(r['takes'])} ({', '.join(r['takes'])})"
        for r in rows
    )
    return f"{len(rows)} piece slots; {listed}; refused: {data['refused']}"
