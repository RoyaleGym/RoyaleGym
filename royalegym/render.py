"""Standalone HTML replay viewer: a battle trace -> ONE self-contained .html file.

WHAT IT IS FOR
    This project is judged by behaviour you can watch, and
    the trace ``replay.py`` records is the only artefact that carries a battle out
    of a process. This turns one into a page that opens by double-click: no
    server, no network, no install. Inline CSS and JS, one <canvas>, the trace
    data embedded as JSON in a ``<script type="application/json">`` element.

WHAT IT SHOWS
    The 18 x 32 board at half-tile resolution (grass, each side's half tinted,
    water, bridges, no-deploy cells hatched), every entity -- crown towers and
    buildings as squares, troops as circles, flying troops ringed with a shadow --
    sized by its collision radius and coloured by team, HP bars, deploy timers
    (translucent with a dashed outline), deploy commands as markers (rejected ones
    grey, with the DeployStatus name), per-team elixir / crowns / hand, the state
    hash of the frame, and the result. SPELLS: every live spell object
    the engine recorded -- a projectile in flight or an airborne Log as a team-coloured
    dot with a dashed line to a cross at its landing point, a rolling Log as a bar
    with a line to its roll end, an area effect as a dashed ring -- labelled with the
    card; stunned entities get a yellow ring, and the tooltip shows stun and
    knockback ticks. Controls: timeline scrubber, play/pause,
    frame step, speed, flip the board to Red's seat, optional interpolation, and a
    hover tooltip per entity. Keys: Space, Left/Right, Home/End.

WHICH ARENA IT DRAWS, AND WHY
    The grid is taken from the TRACE HEADER, because that is the arena the battle
    was actually simulated on (an engine copies ``Engine.arena()`` into the header,
    and both engines load it from data/derived/arena.json). It is then compared
    with the current arena.json through ``protocol.Arena.load`` -- the one Python
    arena loader, not a second one -- and any difference is shown as a warning
    banner in the page, never silently: a stale trace drawn on a regenerated arena
    would put units in the river.

FLOATS
    Only the JavaScript scales integers to pixels. Nothing here feeds back into a
    simulation.

WHAT IT CANNOT DO
    Show anything the trace did not record: troop projectiles, targets, attack timers
    and paths are not in ``EntityState``. A spell's hit radius is not recorded
    (``SpellState`` has none), so spell markers are drawn at a FIXED size that says
    nothing about what the spell hits, and a trace recorded before the engine
    modelled spells carries none. With ``frame_every_tick=False`` frames are
    one per decision, and motion between them is a jump (or a straight-line guess
    if interpolation is switched on, which is labelled as such).

USAGE
    python -m royalegym.render trace.msgpack -o battle.html [--stride N] [--title T]
    exit codes: 0 written, 1 the trace is not renderable, 2 usage error.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections.abc import Sequence
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import msgspec

from .protocol import (
    BIT_LANE_LEFT,
    BIT_LANE_RIGHT,
    BIT_NO_DEPLOY,
    BIT_WATER,
    Arena,
    DeployStatus,
    EntityKind,
    SpellMotion,
    Winner,
    data_dir,
    default_calibration,
)
from .replay import TRACE_FORMAT, TRACE_VERSION, Trace, load_trace

VIEW_FORMAT = "royalegym-replay-view"
VIEW_VERSION = 1
DATA_ELEMENT_ID = "replay-data"

# Columns the viewer reads by name. A trace that lacks one cannot be drawn honestly.
REQUIRED_ENTITY_FIELDS = (
    "uid",
    "team",
    "kind",
    "card_id",
    "tower_slot",
    "x",
    "y",
    "hp",
    "max_hp",
    "radius",
    "flying",
    "deploy_ticks",
)
REQUIRED_FRAME_FIELDS = ("tick", "entities", "elixir_milli", "crowns", "hands", "state_hash")
# Needed only when a frame actually carries spell objects (a pre-spell trace has none).
REQUIRED_SPELL_FIELDS = ("team", "card_id", "motion", "x", "y", "aim_x", "aim_y")


class RenderError(ValueError):
    """The trace cannot be rendered faithfully."""


def _arena_warnings(trace: Trace, arena: Arena | None) -> tuple[list[str], str]:
    """Compare the trace's arena with data/derived/arena.json; return (warnings, vintage)."""
    h = trace.header
    vintage = "unknown"
    try:
        current = arena if arena is not None else Arena.load(default_calibration())
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return [
            f"current arena.json could not be loaded ({exc}); drawn from the trace only"
        ], vintage
    try:
        # Provenance only -- geometry always goes through Arena.load above.
        raw = json.loads((data_dir() / "derived" / "arena.json").read_text(encoding="utf-8"))
        vintage = str(raw.get("provenance", {}).get("vintage", "unknown"))
    except (OSError, ValueError):
        pass
    diffs = []
    for name, got, want in (
        ("subtile", h.subtile, current.subtile),
        ("tiles_x", h.tiles_x, current.tiles_x),
        ("tiles_y", h.tiles_y, current.tiles_y),
        ("half", h.half, current.half),
        ("water_half_rows", list(h.water_half_rows), list(current.water_half_rows)),
        (
            "bridges_half_cols",
            [list(b) for b in h.bridges_half_cols],
            [list(b) for b in current.bridges_half_cols],
        ),
        ("grid", h.grid, current.grid),
    ):
        if got != want:
            diffs.append(name)
    if diffs:
        return [
            "the trace was recorded on a different arena than the current "
            f"data/derived/arena.json (differs in: {', '.join(diffs)}); the board is drawn "
            "from the trace"
        ], vintage
    return [], vintage


def build_view(trace: Trace, arena: Arena | None = None, stride: int = 1) -> dict[str, Any]:
    """The JSON payload the page embeds. Raises ``RenderError`` on an unrenderable trace.

    ``stride`` keeps every stride-th frame (the first and last are always kept), for
    long ``frame_every_tick`` traces; deploy markers are unaffected.
    """
    h = trace.header
    if h.format != TRACE_FORMAT:
        raise RenderError(f"not a royalegym trace: format {h.format!r}")
    if h.version != TRACE_VERSION:
        raise RenderError(f"trace version {h.version} is not {TRACE_VERSION}; update render.py")
    missing = [f for f in REQUIRED_ENTITY_FIELDS if f not in h.entity_fields]
    missing += [f"frame.{f}" for f in REQUIRED_FRAME_FIELDS if f not in h.frame_fields]
    if missing:
        raise RenderError(f"trace lacks columns the viewer needs: {missing}")
    if not trace.frames:
        raise RenderError("trace has no frames: there is nothing to draw")
    if stride < 1:
        raise RenderError("stride must be >= 1")
    if len(h.grid) != h.tiles_y * h.half or any(len(r) != h.tiles_x * h.half for r in h.grid):
        raise RenderError("trace header grid does not match its tile counts")
    if any(f.spells for f in trace.frames):
        need = [f"spell.{c}" for c in REQUIRED_SPELL_FIELDS if c not in h.spell_fields]
        if need:
            raise RenderError(f"trace lacks columns the viewer needs: {need}")

    frames = trace.frames
    if stride > 1:
        keep = list(range(0, len(frames), stride))
        if keep[-1] != len(frames) - 1:
            keep.append(len(frames) - 1)
        frames = [frames[i] for i in keep]

    # Card played by each command: the hand as it stood at the step's start tick.
    hands_at: dict[int, list[list[int]]] = {}
    for f in trace.frames:
        hands_at.setdefault(f.tick, f.hands)
    deploys = []
    for st in trace.steps:
        hands = hands_at.get(st.tick)
        for cmd, status in zip(st.commands, st.statuses, strict=True):
            card = -1
            if (
                hands is not None
                and cmd.team in (0, 1)
                and 0 <= cmd.hand_slot < len(hands[cmd.team])
            ):
                card = hands[cmd.team][cmd.hand_slot]
            deploys.append(
                {
                    "tick": st.tick,
                    "team": cmd.team,
                    "slot": cmd.hand_slot,
                    "card": card,
                    "x": cmd.x,
                    "y": cmd.y,
                    "status": status,
                }
            )

    warnings, vintage = _arena_warnings(trace, arena)
    result = msgspec.to_builtins(trace.result) if trace.result is not None else None
    return {
        "format": VIEW_FORMAT,
        "version": VIEW_VERSION,
        "source": {
            "trace_format": h.format,
            "trace_version": h.version,
            "seed": str(h.seed),  # a u64 does not survive a JavaScript Number
            "frame_every_tick": h.frame_every_tick,
            "tick_ms": h.tick_ms,
            "frames_recorded": len(trace.frames),
            "stride": stride,
            "arena_vintage": vintage,
        },
        "arena": {
            "subtile": h.subtile,
            "tiles_x": h.tiles_x,
            "tiles_y": h.tiles_y,
            "half": h.half,
            "grid": h.grid,
            "water_half_rows": list(h.water_half_rows),
            "bridges_half_cols": [list(b) for b in h.bridges_half_cols],
            "bits": {
                "LANE_LEFT": BIT_LANE_LEFT,
                "LANE_RIGHT": BIT_LANE_RIGHT,
                "NO_DEPLOY": BIT_NO_DEPLOY,
                "WATER": BIT_WATER,
            },
        },
        "cards": [msgspec.to_builtins(c) for c in h.cards],
        "entity_fields": list(h.entity_fields),
        "frame_fields": list(h.frame_fields),
        "spell_fields": list(h.spell_fields),
        "motions": {m.name: int(m) for m in SpellMotion},
        "kinds": {k.name: int(k) for k in EntityKind},
        "statuses": {int(s): s.name for s in DeployStatus},
        "winners": {int(w): w.name for w in Winner},
        "frames": msgspec.to_builtins(frames),
        "deploys": deploys,
        "result": result,
        "warnings": warnings,
    }


def _embed_json(payload: dict[str, Any]) -> str:
    """JSON safe inside a <script> element.

    ``<`` only ever occurs inside JSON strings, where ``\\u003c`` is the same
    character, so escaping it makes ``</script>`` and ``<!--`` impossible in the
    element's text without changing the decoded data. Card names are the only
    free text today; the render test plants a hostile one.
    """
    text = msgspec.json.encode(payload).decode("utf-8")
    return text.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def render_html(
    trace: Trace | str | Path,
    *,
    title: str | None = None,
    stride: int = 1,
    arena: Arena | None = None,
) -> str:
    """The complete, self-contained HTML document for ``trace``."""
    t = load_trace(trace) if isinstance(trace, (str, Path)) else trace
    payload = build_view(t, arena=arena, stride=stride)
    if title is None:
        title = f"royalegym replay - seed {t.header.seed}"
    subs = {"TITLE": html.escape(title), "DATA_ID": DATA_ELEMENT_ID, "DATA": _embed_json(payload)}
    # One pass over the template: substituted text (a title or a card name that
    # happens to contain "__DATA__") is never itself re-substituted.
    return _PLACEHOLDER.sub(lambda m: subs[m.group(1)], _TEMPLATE)


def write_html(
    trace: Trace | str | Path,
    path: str | Path,
    *,
    title: str | None = None,
    stride: int = 1,
    arena: Arena | None = None,
) -> Path:
    p = Path(path)
    p.write_text(render_html(trace, title=title, stride=stride, arena=arena), encoding="utf-8")
    return p


class _ScriptCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.scripts: list[tuple[dict[str, str | None], str]] = []
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self._in_script: dict[str, str | None] | None = None
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        self.tags.append((tag, a))
        if tag == "script":
            self._in_script = a
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._in_script is not None:
            self.scripts.append((self._in_script, "".join(self._buf)))
            self._in_script = None

    def handle_data(self, data: str) -> None:
        if self._in_script is not None:
            self._buf.append(data)


def parse_page(text: str) -> _ScriptCollector:
    """Parse a rendered page with the stdlib HTML parser (what a browser's tokenizer sees)."""
    p = _ScriptCollector()
    p.feed(text)
    p.close()
    return p


def extract_view(text: str) -> dict[str, Any]:
    """Decode the JSON payload embedded in a rendered page. Raises if it is not exactly one."""
    found = [body for attrs, body in parse_page(text).scripts if attrs.get("id") == DATA_ELEMENT_ID]
    if len(found) != 1:
        raise RenderError(f"expected one #{DATA_ELEMENT_ID} element, found {len(found)}")
    view = json.loads(found[0])
    if not isinstance(view, dict) or view.get("format") != VIEW_FORMAT:
        raise RenderError("embedded payload is not a royalegym replay view")
    return view


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m royalegym.render", description="Render a royalegym battle trace to HTML."
    )
    ap.add_argument(
        "trace", type=Path, help="trace written by replay.save_trace (.msgpack or .json)"
    )
    ap.add_argument("-o", "--out", type=Path, required=True, help="output .html path")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame (default 1)")
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)
    if not args.trace.is_file():
        print(f"no such trace: {args.trace}", file=sys.stderr)
        return 2
    try:
        trace = load_trace(args.trace)
        out = write_html(trace, args.out, title=args.title, stride=args.stride)
    except (RenderError, msgspec.DecodeError, msgspec.ValidationError) as exc:
        print(f"cannot render {args.trace}: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(trace.frames)} frames)")
    return 0


_PLACEHOLDER = re.compile(r"__(TITLE|DATA_ID|DATA)__")

# The page. No external URL of any kind may appear in it (tests/test_env_render.py).
_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --blue:#2f6fe4; --red:#e0483c; --ink:#1d232b; --muted:#66707c; --panel:#f4f5f7;
  --line:#d9dde3; }
* { box-sizing: border-box; }
body { margin:0; font:13px/1.35 system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
  color:var(--ink); background:#e9ebee; }
header { display:flex; align-items:baseline; gap:12px; padding:8px 14px; background:#fff;
  border-bottom:1px solid var(--line); flex-wrap:wrap; }
header h1 { font-size:15px; margin:0; }
header .meta { color:var(--muted); font-size:12px; }
#warn { display:none; background:#fff3cd; color:#6b4e00; border-bottom:1px solid #e6cf7a;
  padding:6px 14px; }
main { display:flex; gap:12px; padding:10px 14px; align-items:flex-start; flex-wrap:wrap; }
.side { width:190px; background:#fff; border:1px solid var(--line); border-radius:8px;
  padding:10px; }
.side h2 { margin:0 0 6px; font-size:14px; display:flex; justify-content:space-between; }
.side.blue h2 { color:var(--blue); } .side.red h2 { color:var(--red); }
.bar { height:10px; background:#e3e6ea; border-radius:5px; overflow:hidden; margin:3px 0 8px; }
.bar > div { height:100%; background:#c44fd6; }
.hand { display:grid; grid-template-columns:1fr 1fr; gap:4px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:5px; padding:3px 5px;
  font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.card b { float:right; color:#8a3fa0; }
.kv { display:flex; justify-content:space-between; color:var(--muted); font-size:12px; }
.kv span:last-child { color:var(--ink); font-variant-numeric:tabular-nums; }
#boardwrap { flex:1 1 360px; min-width:260px; display:flex; flex-direction:column;
  align-items:center; }
canvas { background:#7fb35b; border-radius:6px; box-shadow:0 1px 4px rgba(0,0,0,.25);
  touch-action:none; }
#controls { display:flex; gap:8px; align-items:center; width:100%; max-width:760px;
  margin-top:8px; flex-wrap:wrap; }
#controls button { font:inherit; padding:4px 10px; border:1px solid var(--line);
  background:#fff; border-radius:5px; cursor:pointer; min-width:34px; }
#controls button:hover { background:var(--panel); }
#scrub { flex:1 1 240px; }
#clock { font-variant-numeric:tabular-nums; min-width:150px; text-align:right; }
#result { margin-top:6px; font-weight:600; min-height:18px; }
#tip { position:fixed; pointer-events:none; background:rgba(20,24,30,.92); color:#fff;
  padding:5px 7px; border-radius:5px; font-size:12px; display:none; white-space:pre;
  z-index:5; }
footer { color:var(--muted); font-size:11px; padding:4px 14px 12px; }
.legend { display:flex; gap:10px; flex-wrap:wrap; font-size:11px; color:var(--muted);
  margin-top:6px; }
.sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:3px;
  vertical-align:-1px; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <span class="meta" id="meta"></span>
</header>
<div id="warn"></div>
<main>
  <section class="side blue" id="side0"></section>
  <div id="boardwrap">
    <canvas id="board" width="360" height="640"></canvas>
    <div id="controls">
      <button id="play" title="Play / pause (Space)" aria-label="Play or pause">&#9654;</button>
      <button id="prev" title="Previous frame (Left)"
        aria-label="Previous frame">&#9664;&#9664;</button>
      <button id="next" title="Next frame (Right)" aria-label="Next frame">&#9654;&#9654;</button>
      <input id="scrub" type="range" min="0" max="0" value="0" aria-label="Timeline">
      <select id="speed" title="Playback speed" aria-label="Playback speed">
        <option value="0.25">0.25x</option><option value="0.5">0.5x</option>
        <option value="1" selected>1x</option><option value="2">2x</option>
        <option value="4">4x</option><option value="8">8x</option>
      </select>
      <label><input type="checkbox" id="flip" aria-label="Red at bottom"> Red at bottom</label>
      <label title="straight-line guess between recorded frames">
        <input type="checkbox" id="interp" aria-label="Interpolate"> interpolate</label>
      <span id="clock"></span>
    </div>
    <div id="result"></div>
    <div class="legend">
      <span><i class="sw" style="background:#4aa3df"></i>water</span>
      <span><i class="sw" style="background:#b98b5a"></i>bridge</span>
      <span><i class="sw" style="background:#5d7f4a"></i>no-deploy (hatched)</span>
      <span>square = tower/building, circle = troop, ring = flying, dashed = deploying</span>
      <span>spell: dot + dashed line to its landing cross, bar = rolling Log, dashed ring =
        area; marker size is NOT the hit radius; yellow ring = stunned</span>
    </div>
  </div>
  <section class="side red" id="side1"></section>
</main>
<footer id="foot"></footer>
<div id="tip"></div>
<script type="application/json" id="__DATA_ID__">__DATA__</script>
<script>
(function () {
  "use strict";
  var D = JSON.parse(document.getElementById("__DATA_ID__").textContent);
  var A = D.arena, S = A.subtile, TX = A.tiles_x, TY = A.tiles_y, HC = A.half;
  var BITS = A.bits;
  var EF = {}, FF = {}, SF = {};
  D.entity_fields.forEach(function (n, i) { EF[n] = i; });
  D.frame_fields.forEach(function (n, i) { FF[n] = i; });
  (D.spell_fields || []).forEach(function (n, i) { SF[n] = i; });
  var MOTION = D.motions || {};
  var FR = D.frames, N = FR.length, TICK_MS = D.source.tick_ms;
  var KIND = D.kinds, CARDS = D.cards;
  var TEAM = ["#2f6fe4", "#e0483c"], TEAM_DARK = ["#173f8f", "#8e2219"];
  var NAMES = ["Blue", "Red"];
  var cv = document.getElementById("board"), ctx = cv.getContext("2d");
  var scrub = document.getElementById("scrub"), playBtn = document.getElementById("play");
  var speedSel = document.getElementById("speed"), flipBox = document.getElementById("flip");
  var interpBox = document.getElementById("interp"), tip = document.getElementById("tip");
  var cur = 0, playing = false, simTick = FR[0][FF.tick], lastT = null;
  var scale = 20, arenaLayer = null, drawn = [];
  scrub.max = String(N - 1);

  function tick(i) { return FR[i][FF.tick]; }
  function cardName(id) { return id >= 0 && id < CARDS.length ? CARDS[id].name : ""; }
  function fmtTime(t) {
    var ms = t * TICK_MS, s = Math.floor(ms / 1000);
    return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0") + "." +
      String(Math.floor((ms % 1000) / 100));
  }
  function px(x, y) {
    var tx = x / S, ty = y / S;
    if (flipBox.checked) { tx = TX - tx; ty = TY - ty; }
    return [tx * scale, (TY - ty) * scale];
  }

  function buildArena() {
    var c = document.createElement("canvas");
    c.width = cv.width; c.height = cv.height;
    var g = c.getContext("2d"), dpr = cv.width / (TX * scale);
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    var lo = A.water_half_rows[0], hi = A.water_half_rows[1], cell = S / HC;
    var hatch = document.createElement("canvas"); hatch.width = 8; hatch.height = 8;
    var hg = hatch.getContext("2d");
    hg.strokeStyle = "rgba(40,60,30,0.45)"; hg.lineWidth = 1.2;
    hg.beginPath(); hg.moveTo(0, 8); hg.lineTo(8, 0); hg.stroke();
    var pat = g.createPattern(hatch, "repeat");
    for (var hy = 0; hy < A.grid.length; hy++) {
      for (var hx = 0; hx < A.grid[hy].length; hx++) {
        var v = A.grid[hy][hx];
        var p0 = px(hx * cell, hy * cell), p1 = px((hx + 1) * cell, (hy + 1) * cell);
        var x0 = Math.min(p0[0], p1[0]), y0 = Math.min(p0[1], p1[1]);
        var w = Math.abs(p1[0] - p0[0]), h = Math.abs(p1[1] - p0[1]);
        var tileChecker = (Math.floor(hx / HC) + Math.floor(hy / HC)) % 2;
        if (v & BITS.WATER) g.fillStyle = "#4aa3df";
        else if (hy >= lo && hy <= hi) g.fillStyle = "#b98b5a";
        else g.fillStyle = tileChecker ? "#86b962" : "#8fc26a";
        g.fillRect(x0, y0, w + 0.5, h + 0.5);
        if (!(v & BITS.WATER) && (hy < lo || hy > hi)) {
          g.fillStyle = hy < lo ? "rgba(47,111,228,0.10)" : "rgba(224,72,60,0.10)";
          g.fillRect(x0, y0, w + 0.5, h + 0.5);
        }
        if (v & BITS.NO_DEPLOY) {
          g.fillStyle = "rgba(40,60,30,0.18)"; g.fillRect(x0, y0, w + 0.5, h + 0.5);
          g.fillStyle = pat; g.fillRect(x0, y0, w + 0.5, h + 0.5);
        }
      }
    }
    g.strokeStyle = "rgba(0,0,0,0.08)"; g.lineWidth = 1;
    for (var i = 0; i <= TX; i++) {
      g.beginPath(); g.moveTo(i * scale + 0.5, 0);
      g.lineTo(i * scale + 0.5, TY * scale); g.stroke();
    }
    for (var j = 0; j <= TY; j++) {
      g.beginPath(); g.moveTo(0, j * scale + 0.5);
      g.lineTo(TX * scale, j * scale + 0.5); g.stroke();
    }
    arenaLayer = c;
  }

  function layout() {
    var wrap = document.getElementById("boardwrap");
    var availH = Math.max(320, window.innerHeight - 190);
    var availW = Math.max(240, wrap.clientWidth - 4);
    scale = Math.max(8, Math.floor(Math.min(availW / TX, availH / TY)));
    var dpr = window.devicePixelRatio || 1;
    cv.style.width = TX * scale + "px"; cv.style.height = TY * scale + "px";
    cv.width = Math.round(TX * scale * dpr); cv.height = Math.round(TY * scale * dpr);
    buildArena();
    draw();
  }

  function hpColor(f) { return f > 0.6 ? "#3ecf4a" : f > 0.3 ? "#f2c230" : "#f0503c"; }

  function entitiesAt() {
    var rows = FR[cur][FF.entities];
    if (!interpBox.checked || cur + 1 >= N) return rows;
    var t0 = tick(cur), t1 = tick(cur + 1);
    if (t1 <= t0) return rows;
    var a = Math.min(1, Math.max(0, (simTick - t0) / (t1 - t0)));
    var next = {};
    FR[cur + 1][FF.entities].forEach(function (r) { next[r[EF.uid]] = r; });
    return rows.map(function (r) {
      var n = next[r[EF.uid]];
      if (!n) return r;
      var c = r.slice();
      c[EF.x] = r[EF.x] + (n[EF.x] - r[EF.x]) * a;
      c[EF.y] = r[EF.y] + (n[EF.y] - r[EF.y]) * a;
      return c;
    });
  }

  function drawEntity(r) {
    var p = px(r[EF.x], r[EF.y]), rad = Math.max(3, r[EF.radius] / S * scale);
    var team = r[EF.team], kind = r[EF.kind], flying = r[EF.flying];
    var deploying = r[EF.deploy_ticks] > 0, square = kind !== KIND.TROOP;
    ctx.save();
    if (deploying) ctx.globalAlpha = 0.5;
    if (flying) {
      ctx.fillStyle = "rgba(0,0,0,0.22)";
      ctx.beginPath(); ctx.ellipse(p[0], p[1] + rad * 0.9, rad, rad * 0.45, 0, 0, 7); ctx.fill();
    }
    ctx.fillStyle = TEAM[team]; ctx.strokeStyle = TEAM_DARK[team]; ctx.lineWidth = 1.5;
    if (deploying) ctx.setLineDash([3, 2]);
    ctx.beginPath();
    if (square) ctx.rect(p[0] - rad, p[1] - rad, 2 * rad, 2 * rad);
    else ctx.arc(p[0], p[1], rad, 0, 2 * Math.PI);
    ctx.fill(); ctx.stroke();
    if (flying) {
      ctx.setLineDash([]); ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(p[0], p[1], rad + 2, 0, 2 * Math.PI); ctx.stroke();
    }
    ctx.setLineDash([]);
    if (EF.stun_ticks !== undefined && r[EF.stun_ticks] > 0) {
      ctx.globalAlpha = 1; ctx.strokeStyle = "#ffd400"; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.arc(p[0], p[1], rad + 3.5, 0, 2 * Math.PI); ctx.stroke();
    }
    var label = kind === KIND.KING_TOWER ? "K" : kind === KIND.PRINCESS_TOWER ? "P" :
      cardName(r[EF.card_id]).slice(0, 2);
    if (rad >= 7 && label) {
      ctx.fillStyle = "#fff"; ctx.font = "600 " + Math.round(Math.min(rad, 14)) + "px system-ui";
      ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(label, p[0], p[1] + 0.5);
    }
    ctx.restore();
    var frac = r[EF.max_hp] > 0 ? Math.max(0, Math.min(1, r[EF.hp] / r[EF.max_hp])) : 0;
    var bw = Math.max(16, 2 * rad), bx = p[0] - bw / 2, by = p[1] - rad - 7;
    ctx.fillStyle = "rgba(0,0,0,0.55)"; ctx.fillRect(bx - 1, by - 1, bw + 2, 5);
    ctx.fillStyle = hpColor(frac); ctx.fillRect(bx, by, bw * frac, 3);
    if (square && kind !== KIND.BUILDING) {
      ctx.fillStyle = "#fff"; ctx.font = "600 10px system-ui"; ctx.textAlign = "center";
      ctx.textBaseline = "bottom"; ctx.fillText(String(r[EF.hp]), p[0], by - 2);
    }
    drawn.push({ p: p, rad: rad, r: r });
  }

  function spellsAt() {
    if (FF.spells === undefined) return [];
    return FR[cur][FF.spells] || [];
  }

  function drawSpell(q) {
    var team = q[SF.team], motion = q[SF.motion];
    var p = px(q[SF.x], q[SF.y]), aim = px(q[SF.aim_x], q[SF.aim_y]);
    var m = Math.max(4, scale * 0.35);
    ctx.save();
    ctx.strokeStyle = TEAM_DARK[team]; ctx.fillStyle = TEAM[team]; ctx.lineWidth = 2;
    if (motion !== MOTION.AREA) {
      ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(aim[0], aim[1]); ctx.stroke();
      ctx.setLineDash([]);
    }
    if (motion === MOTION.FLIGHT || motion === MOTION.AIRBORNE) {
      ctx.beginPath(); ctx.moveTo(aim[0] - m, aim[1] - m); ctx.lineTo(aim[0] + m, aim[1] + m);
      ctx.moveTo(aim[0] - m, aim[1] + m); ctx.lineTo(aim[0] + m, aim[1] - m); ctx.stroke();
      ctx.beginPath(); ctx.arc(p[0], p[1], m * 0.7, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    } else if (motion === MOTION.ROLLING) {
      ctx.fillStyle = "#7a4a1f";
      ctx.fillRect(p[0] - scale, p[1] - m * 0.45, 2 * scale, m * 0.9);
      ctx.strokeRect(p[0] - scale, p[1] - m * 0.45, 2 * scale, m * 0.9);
    } else {
      ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.arc(p[0], p[1], scale, 0, 2 * Math.PI); ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.fillStyle = TEAM_DARK[team]; ctx.font = "600 10px system-ui";
    ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    ctx.fillText(cardName(q[SF.card_id]), p[0], p[1] - m - 1);
    ctx.restore();
    drawn.push({ p: p, rad: m, spell: q });
  }

  function drawDeploys() {
    var now = tick(cur), window_ = Math.max(1, Math.round(1000 / TICK_MS));
    D.deploys.forEach(function (d) {
      if (d.tick > now || now - d.tick > window_) return;
      var p = px(d.x, d.y), ok = d.status === 0, fade = 1 - (now - d.tick) / (window_ + 1);
      ctx.save(); ctx.globalAlpha = Math.max(0.15, fade);
      ctx.strokeStyle = ok ? TEAM[d.team] : "#555"; ctx.lineWidth = 2;
      if (!ok) ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.arc(p[0], p[1], scale * 0.6, 0, 2 * Math.PI); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = ok ? TEAM_DARK[d.team] : "#333"; ctx.font = "600 11px system-ui";
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      var text = cardName(d.card) + (ok ? "" : " x " + (D.statuses[d.status] || d.status));
      ctx.fillText(text, p[0], p[1] + scale * 0.65);
      ctx.restore();
    });
  }

  function side(team, f) {
    var el = document.getElementById("side" + team);
    var elixir = f[FF.elixir_milli][team], crowns = f[FF.crowns][team];
    var towers = { 0: "-", 1: "-", 2: "-" };
    f[FF.entities].forEach(function (r) {
      if (r[EF.team] === team && r[EF.tower_slot] >= 0) towers[r[EF.tower_slot]] = r[EF.hp];
    });
    var hand = f[FF.hands][team].map(function (c) {
      var card = c >= 0 && c < CARDS.length ? CARDS[c] : null;
      return '<div class="card">' + (card ? esc(card.name) + "<b>" + card.elixir + "</b>" :
        "(empty)") + "</div>";
    }).join("");
    el.innerHTML = "<h2><span>" + NAMES[team] + "</span><span>" + "&#9819;".repeat(crowns) +
      "<small> " + crowns + "</small></span></h2>" +
      '<div class="kv"><span>elixir</span><span>' + (elixir / 1000).toFixed(2) + "</span></div>" +
      '<div class="bar"><div style="width:' + Math.min(100, elixir / 100) + '%"></div></div>' +
      '<div class="hand">' + hand + "</div>" +
      '<div class="kv" style="margin-top:8px"><span>king</span><span>' + towers[0] +
      "</span></div>" +
      '<div class="kv"><span>own-left princess</span><span>' + towers[1] + "</span></div>" +
      '<div class="kv"><span>own-right princess</span><span>' + towers[2] + "</span></div>";
  }

  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (ch) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch];
    });
  }

  function draw() {
    if (!arenaLayer) return;
    var dpr = cv.width / (TX * scale);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(arenaLayer, 0, 0);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    var f = FR[cur], rows = entitiesAt();
    drawn = [];
    var order = rows.slice().sort(function (a, b) {
      return (a[EF.flying] - b[EF.flying]) || (a[EF.y] - b[EF.y]) || (a[EF.uid] - b[EF.uid]);
    });
    order.forEach(drawEntity);
    spellsAt().forEach(drawSpell);
    drawDeploys();
    side(0, f); side(1, f);
    scrub.value = String(cur);
    document.getElementById("clock").textContent = "tick " + tick(cur) + "  " +
      fmtTime(tick(cur)) + "  (" + (cur + 1) + "/" + N + ")";
    playBtn.innerHTML = playing ? "&#10074;&#10074;" : "&#9654;";
    var res = document.getElementById("result");
    if (cur === N - 1 && D.result) {
      var w = D.result.winner, c = D.result.crowns;
      res.textContent = (w === 2 ? "Draw" : w === 0 ? "Blue wins" : w === 1 ? "Red wins" :
        "Episode cut before the game ended") + "  (crowns " + c[0] + " - " + c[1] +
        ")   final hash " + D.result.final_hash;
    } else {
      res.textContent = "state hash " + f[FF.state_hash];
    }
  }

  function seek(i) {
    cur = Math.max(0, Math.min(N - 1, i));
    simTick = tick(cur);
    draw();
  }

  function frameLoop(t) {
    if (playing) {
      if (lastT !== null) {
        simTick += (t - lastT) * Number(speedSel.value) / TICK_MS;
        while (cur + 1 < N && tick(cur + 1) <= simTick) cur++;
        if (cur === N - 1) playing = false;
      }
      lastT = t;
      draw();
    } else {
      lastT = null;
    }
    window.requestAnimationFrame(frameLoop);
  }

  function toggle() {
    if (!playing && cur === N - 1) seek(0);
    playing = !playing;
    draw();
  }

  playBtn.addEventListener("click", toggle);
  document.getElementById("prev").addEventListener("click", function () {
    playing = false; seek(cur - 1);
  });
  document.getElementById("next").addEventListener("click", function () {
    playing = false; seek(cur + 1);
  });
  function onScrub() { playing = false; seek(Number(scrub.value)); }
  scrub.addEventListener("input", onScrub);
  scrub.addEventListener("change", onScrub);
  flipBox.addEventListener("change", function () { buildArena(); draw(); });
  interpBox.addEventListener("change", draw);
  window.addEventListener("resize", layout);
  window.addEventListener("keydown", function (e) {
    // A focused control already handles these keys natively (Space clicks a button,
    // arrows move the slider); handling them here too would act twice.
    var tag = e.target && e.target.tagName;
    if (tag === "BUTTON" || tag === "INPUT" || tag === "SELECT") return;
    // e.key first: synthetic and some platform events leave e.code empty.
    var k = e.key || e.code;
    if (k === " " || k === "Spacebar" || k === "Space") { e.preventDefault(); toggle(); }
    else if (k === "ArrowLeft") { playing = false; seek(cur - 1); }
    else if (k === "ArrowRight") { playing = false; seek(cur + 1); }
    else if (k === "Home") { playing = false; seek(0); }
    else if (k === "End") { playing = false; seek(N - 1); }
  });
  cv.addEventListener("mousemove", function (e) {
    var b = cv.getBoundingClientRect(), mx = e.clientX - b.left, my = e.clientY - b.top;
    var hit = null;
    for (var i = drawn.length - 1; i >= 0; i--) {
      var d = drawn[i], dx = mx - d.p[0], dy = my - d.p[1];
      if (dx * dx + dy * dy <= Math.max(d.rad, 6) * Math.max(d.rad, 6)) { hit = d; break; }
    }
    if (!hit) { tip.style.display = "none"; return; }
    if (hit.spell) {
      var q = hit.spell, mname = Object.keys(MOTION).filter(function (k) {
        return MOTION[k] === q[SF.motion];
      })[0];
      tip.textContent = cardName(q[SF.card_id]) + "  (" + NAMES[q[SF.team]] + " spell, " +
        String(mname).toLowerCase() + ")\ntile (" + (q[SF.x] / S).toFixed(2) + ", " +
        (q[SF.y] / S).toFixed(2) + ")\naim (" + (q[SF.aim_x] / S).toFixed(2) + ", " +
        (q[SF.aim_y] / S).toFixed(2) + ")" +
        (SF.delay_ticks !== undefined && q[SF.delay_ticks] > 0 ?
          "\ndelay " + q[SF.delay_ticks] + " ticks" : "") +
        (SF.length !== undefined && q[SF.length] > 0 ? "\nrolled " +
          (q[SF.travelled] / S).toFixed(2) + " / " + (q[SF.length] / S).toFixed(2) +
          " tiles, " + q[SF.hits] + " hit" : "");
      tip.style.display = "block";
      tip.style.left = (e.clientX + 14) + "px"; tip.style.top = (e.clientY + 10) + "px";
      return;
    }
    hit = hit.r;
    var kindName = Object.keys(KIND).filter(function (k) { return KIND[k] === hit[EF.kind]; })[0];
    tip.textContent = (cardName(hit[EF.card_id]) || kindName) + "  (" + NAMES[hit[EF.team]] +
      ", uid " + hit[EF.uid] + ")\nhp " + hit[EF.hp] + " / " + hit[EF.max_hp] +
      "\ntile (" + (hit[EF.x] / S).toFixed(2) + ", " + (hit[EF.y] / S).toFixed(2) + ")" +
      "\nradius " + (hit[EF.radius] / S).toFixed(3) + " tiles" +
      (hit[EF.flying] ? "\nflying" : "") +
      (hit[EF.deploy_ticks] > 0 ? "\ndeploying: " + hit[EF.deploy_ticks] + " ticks left" : "") +
      (EF.stun_ticks !== undefined && hit[EF.stun_ticks] > 0 ?
        "\nstunned: " + hit[EF.stun_ticks] + " ticks left" : "") +
      (EF.knockback_ticks !== undefined && hit[EF.knockback_ticks] > 0 ?
        "\nknockback: " + hit[EF.knockback_ticks] + " ticks left" : "");
    tip.style.display = "block";
    tip.style.left = (e.clientX + 14) + "px"; tip.style.top = (e.clientY + 10) + "px";
  });
  cv.addEventListener("mouseleave", function () { tip.style.display = "none"; });

  var src = D.source;
  document.getElementById("meta").textContent = "seed " + src.seed + " | " + N + " frames" +
    (src.stride > 1 ? " (every " + src.stride + " of " + src.frames_recorded + ")" : "") +
    " | " + (src.frame_every_tick ? "every tick" : "one frame per decision") +
    " | tick " + TICK_MS + " ms | " + D.deploys.length + " deploy commands";
  if (D.warnings.length) {
    var warn = document.getElementById("warn");
    warn.style.display = "block";
    warn.textContent = "Warning: " + D.warnings.join(" | ");
  }
  document.getElementById("foot").textContent = "trace " + src.trace_format + " v" +
    src.trace_version + " | arena vintage: " + src.arena_vintage +
    " | positions are recorded engine state; interpolation, when on, is a straight-line guess" +
    " | Clash Royale data is evidence, not spec";
  layout();
  window.requestAnimationFrame(frameLoop);
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
