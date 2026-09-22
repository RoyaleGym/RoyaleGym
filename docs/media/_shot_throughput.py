#!/usr/bin/env python3
r"""Figure: how many engine ticks a second one core does, and what that buys.

Nothing here is quoted from a doc. ``draw`` runs ``tools/throughput.py`` in the RoyaleSim
checkout a handful of times and draws the MEDIAN run, not the fastest one: a best-of-N is
biased high by construction, and this machine is usually busy with other work. Every run is
long enough to be seconds rather than noise, and all of them are plotted so the spread is on
the face of the picture.

The one derived quantity is battles an hour. It falls straight out of the measured rate and
the battle length the tool prints on its own first line, and the figure shows the arithmetic
rather than hiding it, because with a 3,600-tick battle and a 3,600-second hour the two
numbers are the same numeral and a reader who is not told why will assume a bug.

The figure is drawn for a README table cell: nothing on it is smaller than 34 px, so it
still reads when the tile is 360 px wide.
"""

from __future__ import annotations

import pathlib
import re
import statistics
import subprocess
import sys

import make_media as M

RUNS = 5        # how many times the tool is run; the MEDIAN transcript is the one drawn
BATTLES = 25    # per run, so a single run takes seconds rather than a fraction of one
TOOL = "tools/throughput.py"
SECONDS_PER_HOUR = 3600  # the unit definition, used by both the sum and its caption

W, H = 1000, 620
MARGIN = 48
SOFT = (186, 186, 198)  # a touch brighter than DIM: the 34 px lines must hold at 360 px


def _repo() -> pathlib.Path:
    """The RoyaleSim checkout, as the media script already locates its siblings."""
    d = M.SIBLING["RoyaleSim"]
    if not (d / TOOL).exists():
        raise RuntimeError(f"{TOOL} is not in the RoyaleSim checkout next to RoyaleGym")
    return d


def _bold(size: int):
    """A bold face if the machine has one; small type survives downscaling far better."""
    from PIL import ImageFont

    for name in ("segoeuib.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return M.theme_font(size)


def _run_once(repo: pathlib.Path) -> str:
    tool = TOOL.replace("/", "\\") if sys.platform == "win32" else TOOL
    p = subprocess.run([sys.executable, tool, "--battles", str(BATTLES)],
                       cwd=repo, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{TOOL} exited {p.returncode}:\n{p.stderr.strip()}")
    return p.stdout.replace("\r\n", "\n").rstrip("\n")


def _num(text: str, pattern: str) -> float:
    m = re.search(pattern, text)
    if not m:
        raise RuntimeError(f"cannot find {pattern!r} in the tool's output:\n{text}")
    return float(m.group(1).replace(",", ""))


def _median_run(repo: pathlib.Path) -> tuple[str, list[float]]:
    """Run the tool RUNS times; return the transcript of the median run and every rate."""
    out = [_run_once(repo) for _ in range(RUNS)]
    rates = [_num(t, r"([\d,]+) ticks/s") for t in out]
    mid = sorted(rates)[len(rates) // 2]          # RUNS is odd, so this is a real run
    return out[rates.index(mid)], rates


def _row(d, x: int, base: int, parts) -> int:
    """Draw ``(text, font, colour)`` pieces along one baseline; return the x it ended at."""
    for text, font, colour in parts:
        d.text((x, base), text, font=font, fill=colour, anchor="ls")
        x += d.textlength(text, font=font)
    return x


def draw(out_path: pathlib.Path) -> str:
    repo = _repo()
    transcript, rates = _median_run(repo)

    rate = _num(transcript, r"([\d,]+) ticks/s")
    battles = _num(transcript, r"([\d,]+) battles x")
    steps = _num(transcript, r"x ([\d,]+) steps")
    per_step = _num(transcript, r"steps x ([\d,]+) ticks")
    total_ticks = _num(transcript, r"= ([\d,]+) ticks in")
    seconds = _num(transcript, r"ticks in ([\d.]+) s")
    ticks_per_battle = steps * per_step                     # the tool's own battle length
    battles_per_hour = rate * SECONDS_PER_HOUR / ticks_per_battle

    im, d = M.canvas(W, H)

    # ---- the headline: one number, four words ---------------------------------------
    f_hero = _bold(128)
    f_unit = _bold(56)
    f_note = M.theme_font(36)
    hero = f"{rate:,.0f}"
    base = 178
    wn = d.textlength(hero, font=f_hero)
    d.text((MARGIN, base), hero, font=f_hero, fill=M.BLUE, anchor="ls")
    d.text((MARGIN + wn + 24, base - 46), "ticks/s", font=f_unit, fill=M.TEXT, anchor="ls")
    d.text((MARGIN + wn + 26, base), "on one core", font=f_note, fill=SOFT, anchor="ls")

    # ---- what it buys, with the arithmetic shown so the repeat is not a mystery ------
    f_eq = _bold(52)
    f_small = M.theme_font(34)
    top, bot = 226, 362
    d.rounded_rectangle([MARGIN, top, W - MARGIN, bot], radius=14,
                        fill=M.PANEL, outline=M.BORDER, width=2)
    d.rectangle([MARGIN, top + 2, MARGIN + 8, bot - 2], fill=M.BLUE)
    _row(d, MARGIN + 34, top + 66, [
        ("= ", f_eq, M.DIM),
        (f"{battles_per_hour:,.0f}", f_eq, M.BLUE),
        (" whole battles an hour", f_eq, M.TEXT),
    ])
    d.text((MARGIN + 34, bot - 26),
           f"a battle is {ticks_per_battle:,.0f} ticks, an hour {SECONDS_PER_HOUR:,} seconds",
           font=f_small, fill=SOFT, anchor="ls")

    # ---- the spread: every run, with the drawn one marked ----------------------------
    lo, hi = min(rates), max(rates)
    x0, x1 = MARGIN + 78, W - MARGIN - 78
    axis = 486
    d.text((MARGIN, 432), f"all {len(rates)} runs, the median one drawn above",
           font=f_small, fill=SOFT, anchor="ls")
    d.line([x0, axis, x1, axis], fill=M.BORDER, width=3)
    for r in rates:
        cx = x0 + (x1 - x0) * ((r - lo) / (hi - lo) if hi > lo else 0.5)
        drawn = abs(r - rate) < 0.5
        rad, col = (17, M.BLUE) if drawn else (11, M.DIM)
        d.ellipse([cx - rad, axis - rad, cx + rad, axis + rad], fill=col,
                  outline=M.BG, width=3)
    d.text((x0, axis + 50), f"{lo:,.0f}", font=f_small, fill=SOFT, anchor="ms")
    d.text((x1, axis + 50), f"{hi:,.0f}", font=f_small, fill=SOFT, anchor="ms")

    # ---- provenance ------------------------------------------------------------------
    d.text((MARGIN, H - 26),
           f"{TOOL} · {battles:,.0f} battles a run · {seconds:.1f} s each",
           font=f_small, fill=SOFT, anchor="ls")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)

    spread = ", ".join(f"{r:,.0f}" for r in sorted(rates))
    return (f"throughput: median of {len(rates)} runs {rate:,.0f} ticks/s "
            f"(all runs: {spread}; mean {statistics.mean(rates):,.0f}); "
            f"{total_ticks:,.0f} ticks a run in {seconds:.2f} s; "
            f"{ticks_per_battle:,.0f} ticks a battle from the tool's own line, so "
            f"{battles_per_hour:,.0f} battles an hour; canvas {W}x{H}")
