#!/usr/bin/env python3
"""README tile: a whole battle becomes ONE self-contained HTML page.

Everything on the picture is measured in this run. The trace is the battle the READMEs
quote, ``royalegym.render`` turns it into a page, the page is written to a scratch file,
and the numbers come from that file: its size on disk, the frames decoded back out of it,
and the count of references to anything outside it.

"Nothing external" is not asserted, it is counted: every tag attribute a browser would
fetch from (``src``, ``href``, ``srcset``, ``poster``, ``xlink:href``, ``action``), every
CSS ``url(`` and ``@import``, and every absolute or protocol-relative URL in the text.
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


def _measure(text: str) -> dict[str, object]:
    """Count everything on the page that points outside the page."""
    p = _Refs()
    p.feed(text)
    p.close()
    return {
        "attrs": p.refs,
        "css_url": len(re.findall(r"url\(", text)),
        "imports": len(re.findall(r"@import", text)),
        "absolute": len(re.findall(r"https?://", text)),
        "protocol_relative": len(re.findall(r"(?<![:/\w])//[A-Za-z0-9]", text)),
    }


def _fit(d, text: str, size: int, limit: int, mono: bool = False):
    """The largest font at or below ``size`` whose line fits in ``limit`` pixels."""
    while size > 34:
        f = M.theme_font(size, mono=mono)
        if d.textlength(text, font=f) <= limit:
            return f
        size -= 2
    return M.theme_font(size, mono=mono)


def draw(out_path: pathlib.Path) -> str:
    """Render the replay page, measure it, and draw the tile. Returns a one-line summary."""
    from royalegym.render import extract_view, parse_page, render_html
    from royalegym.replay import load_trace

    trace = load_trace(M.record_battle())
    page = render_html(trace)

    # Real bytes on disk, not len() of a string: that is the file a reader double-clicks.
    with tempfile.TemporaryDirectory() as tmp:
        f = pathlib.Path(tmp) / "replay.html"
        f.write_text(page, encoding="utf-8")
        total = f.stat().st_size
        text = f.read_text(encoding="utf-8")

    view = extract_view(text)
    frames = len(view["frames"])
    # Frames ride as arrays; the column order is the header's own frame_fields list.
    ti = list(view["frame_fields"]).index("tick")
    first_tick = view["frames"][0][ti]
    last_tick = view["frames"][-1][ti]

    # How the bytes split: the embedded recording vs the viewer that draws it.
    body = [b for a, b in parse_page(text).scripts if a.get("id") == "replay-data"]
    data_bytes = len(body[0].encode("utf-8"))
    code_bytes = total - data_bytes

    m = _measure(text)
    n_attr = len(m["attrs"])
    n_src = sum(1 for _, k, _ in m["attrs"] if k in ("src", "srcset", "data-src", "poster"))
    n_href = sum(1 for _, k, _ in m["attrs"] if k in ("href", "xlink:href"))
    n_url = int(m["css_url"]) + int(m["imports"])
    n_abs = int(m["absolute"]) + int(m["protocol_relative"])

    im, d = M.canvas(W, H)

    # ---- headline: the one thing to take away
    y = 40
    l1 = f"One file, {total / 1048576:.1f} MB"
    d.text((PAD, y), l1, font=_fit(d, l1, 78, W - 2 * PAD), fill=M.TEXT)
    y += 92
    outside = n_attr + n_url + n_abs
    ok = outside == 0
    l2 = f"{outside} files to fetch"
    d.text((PAD, y), l2, font=_fit(d, l2, 78, W - 2 * PAD), fill=M.GREEN if ok else M.RED)
    y += 116

    # ---- the picture: the file's bytes, split by what they are
    bar_h = 64
    bar_w = W - 2 * PAD
    cut = PAD + max(10, round(bar_w * data_bytes / total))
    d.rectangle([PAD, y, cut, y + bar_h], fill=M.BLUE)
    d.rectangle([cut, y, PAD + bar_w, y + bar_h], fill=M.AMBER)
    d.rectangle([PAD, y, PAD + bar_w, y + bar_h], outline=M.BORDER, width=2)
    y += bar_h + 26

    rows = [
        (M.BLUE, f"the recording   {data_bytes / 1048576:.2f} MB"),
        (M.AMBER, f"viewer code   {code_bytes / 1024:.0f} KB"),
    ]
    for colour, label in rows:
        d.rectangle([PAD, y + 10, PAD + 34, y + 44], fill=colour)
        d.text((PAD + 52, y), label, font=_fit(d, label, 46, bar_w - 52), fill=M.TEXT)
        y += 64

    y += 26
    line = f"{frames} frames, tick {first_tick} to {last_tick}"
    d.text((PAD, y), line, font=_fit(d, line, 46, bar_w), fill=M.TEXT)
    y += 66

    ev = f"{n_src} src, {n_href} href, {n_url} url( ), {n_abs} http"
    d.text((PAD, y), ev, font=_fit(d, ev, 42, bar_w), fill=M.GREEN if ok else M.RED)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)
    detail = "" if ok else f" EXTERNAL: {m['attrs'][:4]}"
    return (f"{out_path.name}: {total} bytes ({total / 1048576:.2f} MB), {frames} frames "
            f"tick {first_tick}-{last_tick}, recording {data_bytes} B + viewer {code_bytes} B, "
            f"fetchable attrs {n_attr}, css url/@import {n_url}, absolute urls {n_abs}.{detail}")
