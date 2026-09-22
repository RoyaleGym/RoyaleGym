#!/usr/bin/env python3
r"""The spread of the calibration ledger: how well each engine constant is known.

Everything on the picture is read out of the two files at draw time:

    RoyaleSim/data/calibration.json   every entry's status, hence the whole tally
    RoyaleSim/docs/calibration.md     the status vocabulary, the order it ranks the
                                      words in, and which words it says rank together

Nothing is retyped here. The only strings this module owns are the headline words
around the computed numbers and the word "or" that joins two status words in a band.

The figure is drawn for the README of RoyaleSim (see FIGURES in make_media.py), so the
one path it prints, data/calibration.json, is relative to the repo that hosts it, and is
a tracked file.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import make_media as M  # noqa: E402  (palette and fonts live there)

SIM = HERE.parents[2] / "RoyaleSim"
LEDGER = SIM / "data" / "calibration.json"
LEDGER_DOC = SIM / "docs" / "calibration.md"
LEDGER_REL = "data/calibration.json"  # as a RoyaleSim reader would follow it

W, H = 1000, 640
BANDS = 4  # rows the reader gets; the vocabulary is folded down to this many


# ------------------------------------------------------------------ reading the files


def _walk_statuses(node, out):
    """Count every entry that is a dict carrying both 'status' and 'value'."""
    if isinstance(node, dict):
        if "status" in node and "value" in node:
            out[node["status"]] = out.get(node["status"], 0) + 1
            return
        for v in node.values():
            _walk_statuses(v, out)
    elif isinstance(node, list):
        for v in node:
            _walk_statuses(v, out)


def _vocabulary(md_text):
    """The status words in the order the doc ranks them, with each one's gloss.

    The doc has several `word` | meaning tables; the one wanted is under the heading
    that says the statuses are listed in increasing order of trust. Rows are taken in
    file order, which is that ordering.
    """
    lines = md_text.splitlines()
    start = next(i for i, ln in enumerate(lines)
                 if ln.startswith("#") and "order of trust" in ln)
    order, gloss = [], {}
    for line in lines[start + 1:]:
        if line.startswith("#"):
            break
        m = re.match(r"^\|\s*`([a-z_]+)`\s*\|\s*(.+?)\s*\|\s*$", line)
        if not m:
            continue
        word, meaning = m.group(1), m.group(2)
        if word in gloss:
            continue
        order.append(word)
        gloss[word] = re.sub(r"[`*]", "", meaning)
    return order, gloss


def _ranks_with(md_text, order, gloss):
    """word -> the word the doc says it ranks with, for every such statement.

    Two places say it. A row of the vocabulary table can say so in its own meaning
    ("ranked with `guess`"), and the prose under the table can say so about a row that
    sits out of order ("`owner_ruling` ranks **with** `measured`"). Both are read, so a
    picture that merges two statuses is merging them because the doc does.
    """
    alias = {}
    for word in order:
        m = re.search(r"rank(?:s|ed)? with `?([a-z_]+)`?", gloss[word])
        if m and m.group(1) in order and m.group(1) != word:
            alias[word] = m.group(1)
    for m in re.finditer(r"`([a-z_]+)`\s+ranks\s+\*\*with\*\*\s+`([a-z_]+)`", md_text):
        low, high = m.group(1), m.group(2)
        if low in order and high in order and low != high:
            alias[low] = high
    return alias


def _bands(tally, order, alias, most):
    """The tally folded down to at most `most` rows, in decreasing order of trust.

    Two folds happen, and both are visible on the picture. First the words the doc says
    rank together become one row, carrying both names. Then, if that still leaves more
    rows than fit, the least-trusted rows are joined until they do -- and again the row
    keeps both names, so nothing is hidden behind a word this module invented.
    """
    unknown = [w for w in tally if w not in order]
    if unknown:
        raise ValueError(f"status not in the doc's vocabulary: {unknown}")

    groups = {}  # head word -> [member words, most trusted first]
    for word in sorted(tally, key=order.index, reverse=True):
        head = word
        while head in alias:
            head = alias[head]
        groups.setdefault(head, []).append(word)

    # Inside a row the bigger word goes first, so the name the reader sees first is the
    # one most of the row's keys carry.
    rows = [(sorted(ws, key=lambda w: tally[w], reverse=True),
             sum(tally[w] for w in ws))
            for _, ws in sorted(groups.items(), key=lambda kv: order.index(kv[0]),
                                reverse=True)]
    while len(rows) > most:
        (ws_a, n_a), (ws_b, n_b) = rows[-2], rows[-1]
        rows[-2:] = [(ws_a + ws_b, n_a + n_b)]
    return rows


# ------------------------------------------------------------------ drawing helpers


def _status_colour(words, order):
    """Red for a placeholder through to green for an observation.

    The position on the ramp is the mean of the band's own places in the doc's
    increasing-trust ordering, so the colours cannot say something the doc does not.
    """
    if len(order) < 2:
        return M.DIM
    rank = sum(order.index(w) for w in words) / len(words) / (len(order) - 1)
    lo, mid, hi = M.RED, M.AMBER, M.GREEN
    a, b, t = (lo, mid, rank / 0.5) if rank < 0.5 else (mid, hi, (rank - 0.5) / 0.5)
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _fit(d, text, width, sizes, **kw):
    """The largest of `sizes` at which the text fits, and its font."""
    font = M.theme_font(sizes[-1], **kw)
    for size in sizes:
        f = M.theme_font(size, **kw)
        if d.textlength(text, font=f) <= width:
            return f
        font = f
    return font


def draw(out_path: pathlib.Path) -> str:
    """Compute the data, draw the picture, save it, return the numbers that landed on it."""
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    md = LEDGER_DOC.read_text(encoding="utf-8")
    order, gloss = _vocabulary(md)
    alias = _ranks_with(md, order, gloss)

    tally = {}
    _walk_statuses(ledger, tally)
    total = sum(tally.values())
    rows = _bands(tally, order, alias, BANDS)
    assert sum(n for _, n in rows) == total, "the bands lost a key"

    labels = [" or ".join(w.replace("_", " ") for w in ws) for ws, _ in rows]
    colours = [_status_colour(ws, order) for ws, _ in rows]
    counts = [n for _, n in rows]

    # ------------------------------------------------------------------ canvas
    im, d = M.canvas(W, H)
    pad = 40
    inner = W - 2 * pad

    head = f"{total} constants, graded"
    f_head = _fit(d, head, inner, (78, 72, 66, 64))
    d.text((pad, 34), head, font=f_head, fill=M.TEXT)

    sub = "by how well each one is known"
    f_sub = _fit(d, sub, inner, (42, 38, 36, 34))
    d.text((pad, 128), sub, font=f_sub, fill=M.DIM)

    # the spread, as one bar of `total` keys
    bar_y, bar_h, gap = 198, 62, 4
    x = pad
    widths = []
    for n in counts:
        widths.append(round((inner - gap * (len(counts) - 1)) * n / total))
    widths[-1] += inner - gap * (len(counts) - 1) - sum(widths)
    for w, col in zip(widths, colours):
        d.rounded_rectangle((x, bar_y, x + w, bar_y + bar_h), 10, fill=col)
        x += w + gap

    # one row per band: a badge with the count, the status words beside it
    y = 298
    bw, bh, step = 128, 62, 72
    f_count = M.theme_font(44, mono=True)
    f_share = M.theme_font(36)
    for label, n, col in zip(labels, counts, colours):
        d.rounded_rectangle((pad, y, pad + bw, y + bh), 14, fill=col)
        txt = str(n)
        d.text((pad + (bw - d.textlength(txt, font=f_count)) / 2, y + 4), txt,
               font=f_count, fill=M.BG)
        share = f"{n / total:.0%}"
        sw = d.textlength(share, font=f_share)
        d.text((W - pad - sw, y + 12), share, font=f_share, fill=M.DIM)
        f_lab = _fit(d, label, inner - bw - 50 - sw, (40, 38, 36, 34))
        d.text((pad + bw + 28, y + 8), label, font=f_lab, fill=M.TEXT)
        y += step

    foot = f"{LEDGER_REL} · {len(tally)} status words, {len(rows)} bands"
    f_foot = _fit(d, foot, inner, (36, 34))
    d.text((pad, H - 50), foot, font=f_foot, fill=M.DIM)

    im.save(out_path)
    parts = ", ".join(f"{lab}:{n}" for lab, n in zip(labels, counts))
    raw = ", ".join(f"{w}:{tally[w]}" for w in sorted(tally, key=order.index,
                                                      reverse=True))
    return (f"{total} keys over {len(tally)} status words ({raw}); drawn as "
            f"{len(rows)} bands -- {parts}; merges the doc states: {alias}")


if __name__ == "__main__":  # pragma: no cover - convenience only
    print(draw(pathlib.Path(sys.argv[1])))
