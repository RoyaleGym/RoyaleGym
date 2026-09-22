#!/usr/bin/env python3
"""README tile: a whole battle becomes ONE self-contained HTML page.

The trace is the battle the READMEs quote, ``royalegym.render`` turns it into a page, the
page's bytes are written to a scratch file, and everything drawn here comes back out of
that file.

THE PICTURE is the battle itself, decoded out of the written page: the crown-tower HP of
both sides across every recorded frame, with the moments a tower fell marked. It is decoded
out of the bytes that were written, not read off the trace object still in memory, and
"all N ticks" is checked against the decoded tick column rather than taken from the page's
own ``frame_every_tick`` flag. The team names are the page's own winner table.

THE ZERO is counted, not asserted: every tag attribute a browser would fetch from
(``src``, ``href``, ``srcset``, ``poster``, ``formaction``, ``background``, ``action``,
``xlink:href``, ``data-src``), every CSS ``url(`` and ``@import``, every absolute or
protocol-relative URL, and the JavaScript calls that reach the network (``fetch(``,
``XMLHttpRequest``, ``WebSocket``, ``EventSource``, ``import(``, ``importScripts(``,
``sendBeacon(``, ``navigator.serviceWorker``, and assignment to a ``.src`` property).

THE SCAN IS CHECKED, because a scanner that finds nothing anywhere would also print zero:
three references -- an ``<img src>``, an ``@import`` and a ``fetch()`` -- are planted in a
copy of the same page, the same scan is run over it, and what it finds is printed on the
tile beside the zero.

The battle recording itself is ``make_media.record_battle``'s, which is recorded once and
cached in ``.work``; the page, the plot and every count on the tile are produced in this
run from that recording.
"""

from __future__ import annotations

import pathlib
import re
import sys
import tempfile
from html.parser import HTMLParser

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import make_media as M  # noqa: E402

W, H = 1000, 640
PAD = 40

# Attributes a browser would issue a request for if they named something outside the file.
FETCH_ATTRS = ("src", "href", "srcset", "poster", "action", "formaction", "background",
               "xlink:href", "data-src")

# The ways a script reaches the network. Without these the zero would rest on a scan that
# structurally cannot see the most obvious way for a page to phone home.
JS_CALLS = (r"\bfetch\s*\(", r"\bXMLHttpRequest\b", r"\bWebSocket\b", r"\bEventSource\b",
            r"\bimport\s*\(", r"\bimportScripts\s*\(", r"\bsendBeacon\s*\(",
            r"\bnavigator\.serviceWorker\b", r"\.src\s*=")

# Planted in a copy of the page so the scan has to find something. One per family:
# an attribute, a stylesheet import, a script fetch. None of them is an absolute URL or a
# ``url(``, so each is counted exactly once and the total is the number planted.
PLANTS = ('<img src="board.png">',
          '<style>@import "theme.css";</style>',
          '<script>fetch("frames.json")</script>')


class _Refs(HTMLParser):
    """Every fetchable attribute in the page, with the tag that carries it."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.refs: list[tuple[str, str, str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        for k, v in attrs:
            if k.lower() in FETCH_ATTRS:
                self.refs.append((tag, k.lower(), v or ""))

    handle_startendtag = handle_starttag


def _measure(text: str) -> dict[str, int]:
    """Count everything in ``text`` that points outside the file it lives in."""
    p = _Refs()
    p.feed(text)
    p.close()
    return {
        "attrs": len(p.refs),
        "css_url": len(re.findall(r"url\(", text)),
        "imports": len(re.findall(r"@import", text)),
        "absolute": len(re.findall(r"https?://", text)),
        "protocol_relative": len(re.findall(r"(?<![:/\w])//[A-Za-z0-9]", text)),
        "js": sum(len(re.findall(p, text)) for p in JS_CALLS),
    }


def _plant(text: str) -> str:
    """The same page with one reference of each family added, to make the scan fail."""
    blob = "".join(PLANTS)
    return text.replace("</body>", blob + "</body>", 1) if "</body>" in text else text + blob


def _fit(d, text: str, size: int, limit: int, mono: bool = False):
    """The largest font at or below ``size`` whose line fits in ``limit`` pixels."""
    while size > 34:
        f = M.theme_font(size, mono=mono)
        if d.textlength(text, font=f) <= limit:
            return f
        size -= 2
    return M.theme_font(size, mono=mono)


def _tower_hp(view) -> tuple[list[list[int]], list[tuple[int, int]]]:
    """Per-frame crown-tower HP for each team, and the frames where a tower fell.

    Columns are found by name in the page's own ``entity_fields`` list, and the tower kinds
    by name in its ``kinds`` table, so nothing here assumes a column order.
    """
    ef = list(view["entity_fields"])
    ff = list(view["frame_fields"])
    i_kind, i_team, i_hp = ef.index("kind"), ef.index("team"), ef.index("hp")
    i_ents, i_crowns = ff.index("entities"), ff.index("crowns")
    kinds = view["kinds"]
    towers = {kinds["KING_TOWER"], kinds["PRINCESS_TOWER"]}

    series: list[list[int]] = [[], []]
    falls: list[tuple[int, int]] = []  # (frame index, the team that lost the tower)
    crowns = None
    for i, f in enumerate(view["frames"]):
        hp = [0, 0]
        for e in f[i_ents]:
            if e[i_kind] in towers:
                hp[e[i_team]] += max(0, e[i_hp])
        series[0].append(hp[0])
        series[1].append(hp[1])
        c = list(f[i_crowns])
        if crowns is not None and c != crowns:
            # Whoever's crown count went up knocked the other side's tower down.
            for t in (0, 1):
                if c[t] > crowns[t]:
                    falls.append((i, 1 - t))
        crowns = c
    return series, falls


def _plot(d, series, falls, box, vmax: int) -> list[tuple[int, int]]:
    """Draw both HP lines across the plot box; return each line's end point."""
    x0, y0, x1, y1 = box
    width = x1 - x0
    n = len(series[0])
    span = y1 - y0

    def xy(col: int, v: float) -> tuple[float, float]:
        return x0 + col, y1 - (v / vmax) * span

    d.line([(x0, y1), (x1, y1)], fill=M.BORDER, width=2)
    for i, _team in falls:
        x = x0 + i * width // n
        d.line([(x, y0), (x, y1)], fill=M.BORDER, width=2)

    ends = []
    for team, colour in ((0, M.BLUE), (1, M.RED)):
        pts = []
        for col in range(width):
            a, b = col * n // width, max(col * n // width + 1, (col + 1) * n // width)
            chunk = series[team][a:b]
            pts.append(xy(col, sum(chunk) / len(chunk)))
        d.line(pts, fill=colour, width=5, joint="curve")
        ends.append(pts[-1])

    for i, team in falls:
        x = x0 + i * width // n
        cy = xy(0, series[team][i])[1]
        d.ellipse([x - 11, cy - 11, x + 11, cy + 11], fill=M.TEXT)
        d.ellipse([x - 7, cy - 7, x + 7, cy + 7], fill=(M.BLUE, M.RED)[team])
    if falls:
        x = x0 + falls[0][0] * width // n
        d.text((x + 16, y1 - 46), "a tower falls", font=M.theme_font(34), fill=M.DIM)
    return [(int(px), int(py)) for px, py in ends]


def draw(out_path: pathlib.Path) -> str:
    """Render the replay page, measure it, and draw the tile. Returns a one-line summary."""
    from royalegym.render import extract_view, parse_page, render_html
    from royalegym.replay import load_trace

    trace = load_trace(M.record_battle())
    page = render_html(trace)

    # Real bytes on disk, not len() of a string: that is the file someone double-clicks.
    # Bytes in and bytes out, because write_text() would translate newlines on Windows and
    # the tile would then say one thing here and another on Linux.
    with tempfile.TemporaryDirectory() as tmp:
        f = pathlib.Path(tmp) / "replay.html"
        f.write_bytes(page.encode("utf-8"))
        total = f.stat().st_size
        text = f.read_bytes().decode("utf-8")
        fname = f.name

    view = extract_view(text)
    frames = len(view["frames"])
    ff = list(view["frame_fields"])
    ticks = [fr[ff.index("tick")] for fr in view["frames"]]
    # "all N ticks" is read off the decoded tick column, not taken from the header's
    # frame_every_tick flag: a strided page would fail this and say the weaker thing.
    every = ticks == list(range(ticks[0], ticks[-1] + 1))

    # Names for the two sides out of the page's own enum table, not chosen here.
    wins = view.get("winners", {})
    names = [str(wins.get(str(t), f"team {t}")) for t in (0, 1)]
    series, falls = _tower_hp(view)
    vmax = max(max(series[0]), max(series[1]))

    # How the bytes split: the embedded recording vs everything else in the file.
    body = [b for a, b in parse_page(text).scripts if a.get("id") == "replay-data"]
    data_bytes = len(body[0].encode("utf-8"))

    m = _measure(text)
    outside = sum(m.values())
    planted, found = len(PLANTS), sum(_measure(_plant(text)).values()) - outside
    ok = outside == 0 and found >= planted

    im, d = M.canvas(W, H)

    # ---- headline: the one number that is not in the caption
    y = 16
    l1 = f"One file, {total / 1048576:.1f} MB"
    d.text((PAD, y), l1, font=_fit(d, l1, 72, W - 2 * PAD), fill=M.TEXT)

    sub = (f"one battle, all {frames} ticks inside it" if every
           else f"one battle, {frames} of its frames inside it")
    d.text((PAD, 118), sub, font=_fit(d, sub, 38, W - 2 * PAD), fill=M.DIM)

    # ---- the file, with the battle that is in it drawn from its own bytes
    d.rounded_rectangle([PAD, 176, W - PAD, 470], radius=14, fill=M.PANEL,
                        outline=M.BORDER, width=3)
    tag = M.theme_font(34, mono=True)  # the name of the file that was written and measured
    d.text((W - PAD - 24 - d.textlength(fname, font=tag), 192), fname, font=tag, fill=M.DIM)
    cap = "crown tower HP, decoded back out of it"
    limit = W - 2 * PAD - 72 - d.textlength(fname, font=tag)
    d.text((PAD + 24, 190), cap, font=_fit(d, cap, 36, limit), fill=M.DIM)

    px0, px1 = PAD + 24, W - PAD - 132
    ends = _plot(d, series, falls, (px0, 246, px1, 398), vmax)
    for (_ex, ey), colour, name in zip(ends, (M.BLUE, M.RED), names, strict=True):
        d.text((px1 + 14, ey - 20), name, font=M.theme_font(34), fill=colour)

    t0, t1 = f"tick {ticks[0]}", f"tick {ticks[-1]}"
    fnt = M.theme_font(34)
    d.text((px0, 410), t0, font=fnt, fill=M.DIM)
    d.text((px1 - d.textlength(t1, font=fnt), 410), t1, font=fnt, fill=M.DIM)

    # ---- and nothing of it points anywhere else
    y = 486
    # The swatch answers both questions: nothing found here, and the scan can find things.
    d.rectangle([PAD, y + 8, PAD + 34, y + 42],
                fill=M.GREEN if ok else (M.AMBER if outside == 0 else M.RED))
    l3 = f"{outside} links or fetches out of the file"
    d.text((PAD + 52, y), l3, font=_fit(d, l3, 46, W - 2 * PAD - 52),
           fill=M.GREEN if outside == 0 else M.RED)

    y = 552
    all_ = "all " if found >= planted else ""
    l4 = f"the same scan, {planted} planted in a copy of it: {all_}{found} found"
    d.text((PAD, y), l4, font=_fit(d, l4, 34, W - 2 * PAD),
           fill=M.DIM if found >= planted else M.AMBER)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)
    return (f"{out_path.name}: {total} bytes ({total / 1048576:.2f} MB), {frames} frames "
            f"tick {ticks[0]}-{ticks[-1]}, recording {data_bytes} B + the rest "
            f"{total - data_bytes} B, tower HP {series[0][-1]} v {series[1][-1]}, "
            f"{len(falls)} towers fell, refs out {m}, planted {planted} found {found}.")
