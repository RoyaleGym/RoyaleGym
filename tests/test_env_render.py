"""The HTML replay viewer: one self-contained page, valid embedded data, no network.

What these tests can see: that the page parses the way a browser's tokenizer
would split it, that the embedded JSON decodes to exactly the recorded battle, that
nothing in the page names an external resource, and (with node on PATH) that the
inline script is syntactically valid JavaScript, and (with node) that the script
RUNS -- every frame drawn, the tooltip hit-tested over the board -- against a stub DOM
whose canvas records draw calls, spell objects and stunned units included. What they
cannot see: whether the canvas drawing LOOKS right. That was checked by eye in a
browser (troops) and is not re-checked here.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import msgspec
import numpy as np
import pytest

from _decks import card_ids
from royalegym import render
from royalegym.done_condition import GameOverCondition, StepLimitCondition
from royalegym.env import ClashParallelEnv
from royalegym.mock_engine import MockEngine
from royalegym.protocol import (
    Arena,
    Calibration,
    DeployStatus,
    EntityKind,
    EntityState,
    SpellMotion,
    SpellState,
)
from royalegym.render import RenderError, build_view, extract_view, parse_page, render_html
from royalegym.replay import ReplayRecorder, load_trace, save_trace, verify_trace
from royalegym.selfplay import RandomLegalOpponent
from royalegym.state_mutator import DefaultStateMutator

#: Resolved by NAME, like the rest of the suite: the catalogue renumbers whenever a card
#: becomes loadable, so a literal id list quietly names different cards.
ALL_TYPES = card_ids(
    ("Knight", "Giant", "Minions", "Valkyrie", "Cannon", "Fireball", "Zap", "Log"), MockEngine()
)


#: 80 rather than 60 steps. The recording has to CONTAIN what the viewer tests look at,
#: and a random policy stopped deploying the flier within 60 steps when the engine's
#: starting elixir changed on 2026-09-22: more elixir at tick 0 means a different card is
#: affordable first, so the same seed plays a different battle. The recording is checked
#: for its contents rather than assumed, in test_the_recorded_battle_is_real_evidence.
RECORD_STEPS = 80


def _record(steps: int = RECORD_STEPS, frame_every_tick: bool = True):
    rec = ReplayRecorder(frame_every_tick=frame_every_tick)
    env = ClashParallelEnv(
        recorder=rec,
        state_mutator=DefaultStateMutator(decks=[ALL_TYPES, ALL_TYPES[::-1]]),
        termination_cond=GameOverCondition(),
        truncation_cond=StepLimitCondition(steps),
    )
    obs, _ = env.reset(seed=2026)
    rng = np.random.default_rng(0)
    opp = RandomLegalOpponent(noop_prob=0.6)
    while env.agents:
        obs, *_ = env.step({a: opp.act(obs[a], obs[a]["action_mask"], rng) for a in env.agents})
    assert rec.trace is not None
    return rec.trace


TRACE = _record()


def external_references(page: str) -> list[str]:
    """Anything that would make the page reach outside itself."""
    parsed = parse_page(page)
    found = [
        f"<{tag} {k}=...>" for tag, attrs in parsed.tags for k in attrs if k in ("src", "href")
    ]
    found += [f"<{tag}>" for tag, _ in parsed.tags if tag in ("link", "iframe", "img", "object")]
    found += re.findall(r"(?i)(?:https?:|wss?:|ftp:)//|@import|url\(|fetch\(|XMLHttpRequest", page)
    return found


def test_the_recorded_battle_is_real_evidence():
    # The viewer is only tested if the trace has something to show.
    assert verify_trace(TRACE, MockEngine()) == []
    assert len(TRACE.frames) > 100
    statuses = [s for st in TRACE.steps for s in st.statuses]
    assert statuses.count(DeployStatus.OK) >= 4
    troops = [e for f in TRACE.frames for e in f.entities if e.tower_slot < 0]
    assert any(e.deploy_ticks > 0 for e in troops)
    assert any(e.flying for e in troops)


def test_page_is_one_self_contained_document():
    page = render_html(TRACE)
    parsed = parse_page(page)
    tags = [t for t, _ in parsed.tags]
    assert tags.count("canvas") == 1
    assert tags.count("html") == 1
    assert tags.count("style") == 1
    kinds = [attrs.get("type") for attrs, _ in parsed.scripts]
    assert kinds == ["application/json", None]  # the data, then the code; nothing else
    assert external_references(page) == []


def test_plant_external_reference_checker_fires():
    page = render_html(TRACE)
    planted = page.replace(
        "</head>", '<link rel="stylesheet" href="https://example.invalid/x.css"></head>', 1
    )
    assert planted != page
    found = external_references(planted)
    assert "<link>" in found
    assert "<link href=...>" in found


def test_embedded_json_is_exactly_the_recorded_battle():
    view = extract_view(render_html(TRACE))
    assert view["entity_fields"] == list(EntityState.__struct_fields__)
    assert view["frames"] == msgspec.to_builtins(TRACE.frames)
    assert view["cards"] == msgspec.to_builtins(TRACE.header.cards)
    arena = Arena.load(Calibration.load())
    assert view["arena"]["grid"] == arena.grid
    assert (view["arena"]["tiles_x"], view["arena"]["tiles_y"]) == (18, 32)
    assert view["warnings"] == []
    assert view["source"]["seed"] == str(TRACE.header.seed)
    n_commands = sum(len(st.commands) for st in TRACE.steps)
    assert len(view["deploys"]) == n_commands
    assert view["result"] == msgspec.to_builtins(TRACE.result)


def test_deploy_markers_name_the_card_that_was_in_the_slot():
    view = extract_view(render_html(TRACE))
    by_tick = {f.tick: f for f in TRACE.frames}
    checked = 0
    for d in view["deploys"]:
        if d["status"] != DeployStatus.OK:
            continue
        # The engine's own record of what it spent: the card left the hand slot.
        assert by_tick[d["tick"]].hands[d["team"]][d["slot"]] == d["card"]
        spawned = [
            e
            for e in by_tick[d["tick"] + 1].entities
            if e.team == d["team"] and e.card_id == d["card"] and e.deploy_ticks > 0
        ]
        spell = TRACE.header.cards[d["card"]].count == 0
        assert spell or spawned, d
        checked += 1
    assert checked >= 4


def test_hostile_card_name_cannot_break_out_of_the_data_element():
    hostile = '</script><script>window.pwned=1</script><!-- "quote" & amp \u2028'
    cards = list(TRACE.header.cards)
    cards[0] = msgspec.structs.replace(cards[0], name=hostile)
    trace = msgspec.structs.replace(
        TRACE, header=msgspec.structs.replace(TRACE.header, cards=cards)
    )
    page = render_html(trace, title="</title><script>alert(1)</script>")
    assert len(parse_page(page).scripts) == 2
    assert extract_view(page)["cards"][0]["name"] == hostile
    assert "<script>alert(1)" not in page


def test_plant_unescaped_embedding_is_caught_by_the_script_count(monkeypatch):
    """Regression plant: embed the JSON without escaping ``<``; the gate must see it."""
    hostile = "</script><script>window.pwned=1</script>"
    cards = list(TRACE.header.cards)
    cards[0] = msgspec.structs.replace(cards[0], name=hostile)
    trace = msgspec.structs.replace(
        TRACE, header=msgspec.structs.replace(TRACE.header, cards=cards)
    )
    monkeypatch.setattr(render, "_embed_json", lambda p: msgspec.json.encode(p).decode())
    page = render_html(trace)
    assert "</script><script>window.pwned=1</script>" in page  # the plant took effect
    assert len(parse_page(page).scripts) != 2
    with pytest.raises((RenderError, ValueError)):
        extract_view(page)


def test_stride_keeps_first_and_last_frames():
    view = build_view(TRACE, stride=7)
    ticks = [f[0] for f in view["frames"]]
    assert ticks[0] == TRACE.frames[0].tick
    assert ticks[-1] == TRACE.frames[-1].tick
    assert len(ticks) == -(-len(TRACE.frames) // 7) + (1 if (len(TRACE.frames) - 1) % 7 else 0)


def test_per_decision_traces_render_too():
    trace = _record(steps=20, frame_every_tick=False)
    view = extract_view(render_html(trace))
    assert len(view["frames"]) == 21
    assert view["source"]["frame_every_tick"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda t: msgspec.structs.replace(t, frames=[]), "no frames"),
        (
            lambda t: msgspec.structs.replace(
                t, header=msgspec.structs.replace(t.header, format="something-else")
            ),
            "not a royalegym trace",
        ),
        (
            lambda t: msgspec.structs.replace(
                t, header=msgspec.structs.replace(t.header, version=99)
            ),
            "version",
        ),
        (
            lambda t: msgspec.structs.replace(
                t,
                header=msgspec.structs.replace(
                    t.header, entity_fields=[f for f in t.header.entity_fields if f != "radius"]
                ),
            ),
            "radius",
        ),
    ],
    ids=["no-frames", "format", "version", "missing-column"],
)
def test_unrenderable_traces_are_refused(mutate, message):
    with pytest.raises(RenderError, match=message):
        build_view(mutate(TRACE))


def test_arena_drift_is_shown_not_hidden():
    grid = [list(r) for r in TRACE.header.grid]
    grid[0][0] ^= 32  # one half-cell of water that the current arena.json does not have
    trace = msgspec.structs.replace(TRACE, header=msgspec.structs.replace(TRACE.header, grid=grid))
    view = extract_view(render_html(trace))
    assert len(view["warnings"]) == 1
    assert "grid" in view["warnings"][0]
    assert view["arena"]["grid"] == grid  # drawn from what the battle ran on


def test_cli_round_trip_through_msgpack_and_json(tmp_path, capsys):
    for suffix in (".msgpack", ".json"):
        src = save_trace(TRACE, tmp_path / f"t{suffix}")
        out = tmp_path / f"t{suffix}.html"
        assert render.main([str(src), "-o", str(out), "--stride", "2"]) == 0
        view = extract_view(out.read_text(encoding="utf-8"))
        assert view["frames"][0] == msgspec.to_builtins(load_trace(src).frames[0])
    assert "wrote" in capsys.readouterr().out
    assert render.main([str(tmp_path / "missing.msgpack"), "-o", str(tmp_path / "x.html")]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert render.main([str(bad), "-o", str(tmp_path / "y.html")]) == 1


NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node not on PATH: inline JS syntax NOT checked")
def test_inline_script_is_valid_javascript(tmp_path):
    page = render_html(TRACE)
    (code,) = [body for attrs, body in parse_page(page).scripts if "type" not in attrs]
    js = tmp_path / "viewer.js"
    js.write_text(code, encoding="utf-8")
    run = subprocess.run([NODE, "--check", str(js)], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    # Plant: the same checker must reject a broken script.
    js.write_text(code.replace("function draw()", "function draw("), encoding="utf-8")
    run = subprocess.run([NODE, "--check", str(js)], capture_output=True, text=True, check=False)
    assert run.returncode != 0


# --- running the viewer, spells included ---------------------------------------------

# A stub DOM just rich enough for the viewer: every element accepts any property and
# listener, every canvas context method is a recorded no-op. The harness runs the
# page script, scrubs through every frame, sweeps the mouse over the board on each
# spell frame and prints what it saw as JSON. It proves the code RUNS and reaches the
# spell and stun paths; it says nothing about how the drawing looks.
_NODE_HARNESS = r"""
const fs = require("fs");
const [dataPath, codePath] = process.argv.slice(2);
const DATA = fs.readFileSync(dataPath, "utf8");
const calls = {};
function ctx2d() {
  return new Proxy({}, {
    get(target, k) {
      if (k in target) return target[k];
      return function () { calls[k] = (calls[k] || 0) + 1; return { addColorStop() {} }; };
    },
    set(target, k, v) { target[k] = v; return true; },
  });
}
const elements = {};
function el(id) {
  if (!elements[id]) {
    const handlers = {};
    elements[id] = {
      id, style: {}, value: "0", checked: false, textContent: "", innerHTML: "",
      width: 360, height: 640, clientWidth: 400, max: "0",
      getContext: () => ctx2d(), getBoundingClientRect: () => ({ left: 0, top: 0 }),
      addEventListener: (ev, fn) => { (handlers[ev] = handlers[ev] || []).push(fn); },
      fire: (ev, arg) => (handlers[ev] || []).forEach((fn) => fn(arg || {})),
    };
  }
  return elements[id];
}
elements["replay-data"] = { textContent: DATA };
global.document = { getElementById: el, createElement: () => el("tmp" + Math.random()) };
const winHandlers = {};
global.window = {
  innerHeight: 800, devicePixelRatio: 1, requestAnimationFrame: () => 0,
  addEventListener: (ev, fn) => { winHandlers[ev] = fn; },
};
eval(fs.readFileSync(codePath, "utf8"));
const D = JSON.parse(DATA);
const FF = {}; D.frame_fields.forEach((n, i) => { FF[n] = i; });
const scrub = el("scrub"), board = el("board"), tip = el("tip");
let spellTips = 0, stunTips = 0, framesDrawn = 0, spellFrames = 0;
for (let i = 0; i < D.frames.length; i++) {
  scrub.value = String(i); scrub.fire("input"); framesDrawn++;
  if (!(D.frames[i][FF.spells] || []).length) continue;
  spellFrames++;
  if (spellFrames > 3) continue;  // sweep the tooltip over the first three only
  for (let x = 0; x < 400; x += 2) {
    for (let y = 0; y < 700; y += 2) {
      board.fire("mousemove", { clientX: x, clientY: y });
      if (tip.style.display === "block") {
        if (tip.textContent.includes(" spell, ")) spellTips++;
        if (tip.textContent.includes("stunned: ")) stunTips++;
      }
    }
  }
}
console.log(JSON.stringify({ framesDrawn, spellFrames, spellTips, stunTips, calls }));
"""


def _trace_with_spells():
    """TRACE with spell objects and a stunned unit written into its later frames.

    Synthetic, like tests/test_env_obs.py's spell states: MockEngine records no spell
    objects (its spells resolve inside a tick). One of each motion, both teams, and a
    Log whose end point lies past the arena edge."""
    t_ = TRACE.header.subtile
    cid = {c.name: c.card_id for c in TRACE.header.cards}
    rows = [
        SpellState(
            0, cid["Fireball"], SpellMotion.FLIGHT, 9 * t_, 6 * t_, 14 * t_, 26 * t_, 2, 0, 0, 0
        ),
        SpellState(
            1,
            cid["Log"],
            SpellMotion.ROLLING,
            4 * t_,
            12 * t_,
            4 * t_,
            -3 * t_,
            0,
            9 * t_,
            11 * t_,
            1,
        ),
        SpellState(0, cid["Zap"], SpellMotion.AREA, 3 * t_, 20 * t_, 3 * t_, 20 * t_, 0, 0, 0, 0),
        SpellState(
            1, cid["Log"], SpellMotion.AIRBORNE, 12 * t_, 24 * t_, 12 * t_, 22 * t_, 0, 0, 0, 0
        ),
    ]
    frames = []
    for k, f in enumerate(TRACE.frames):
        if k < len(TRACE.frames) // 2:
            frames.append(f)
            continue
        ents = [
            msgspec.structs.replace(e, stun_ticks=5) if e.kind == EntityKind.TROOP else e
            for e in f.entities
        ]
        frames.append(msgspec.structs.replace(f, spells=rows, entities=ents))
    assert any(e.kind == EntityKind.TROOP for f in frames[len(frames) // 2 :] for e in f.entities)
    return msgspec.structs.replace(TRACE, frames=frames)


def run_viewer(tmp_path, trace, code_edit=None) -> dict:
    page = render_html(trace)
    (data,) = [body for attrs, body in parse_page(page).scripts if attrs.get("id")]
    (code,) = [body for attrs, body in parse_page(page).scripts if "type" not in attrs]
    if code_edit is not None:
        edited = code_edit(code)
        assert edited != code, "plant did not land"
        code = edited
    (tmp_path / "data.json").write_text(data, encoding="utf-8")
    (tmp_path / "viewer.js").write_text(code, encoding="utf-8")
    (tmp_path / "harness.js").write_text(_NODE_HARNESS, encoding="utf-8")
    run = subprocess.run(
        [
            NODE,
            str(tmp_path / "harness.js"),
            str(tmp_path / "data.json"),
            str(tmp_path / "viewer.js"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode != 0:
        return {"error": run.stderr[-2000:]}
    return json.loads(run.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(NODE is None, reason="node not on PATH: the viewer is NOT executed")
def test_viewer_runs_every_frame_and_shows_spells_and_stuns(tmp_path):
    trace = _trace_with_spells()
    got = run_viewer(tmp_path, trace)
    print(f"viewer run: {got}")
    assert "error" not in got, got["error"]
    assert got["framesDrawn"] == len(trace.frames)
    assert got["spellFrames"] == sum(1 for f in trace.frames if f.spells) > 0
    assert got["spellTips"] > 0, "no spell tooltip was ever shown"
    assert got["stunTips"] > 0, "no stunned-unit tooltip was ever shown"
    # A pre-spell trace (no spell rows, no spell_fields) must still run.
    old = msgspec.structs.replace(
        TRACE, header=msgspec.structs.replace(TRACE.header, spell_fields=[])
    )
    got_old = run_viewer(tmp_path, old)
    assert "error" not in got_old, got_old["error"]
    assert got_old["spellFrames"] == 0


@pytest.mark.skipif(NODE is None, reason="node not on PATH: the viewer is NOT executed")
def test_plant_viewer_spell_path_that_throws_is_caught(tmp_path):
    """Plant: the spell drawing reads a column the payload does not name. The harness
    must fail (a throwing draw) -- a check that only parsed the script would not."""
    got = run_viewer(
        tmp_path,
        _trace_with_spells(),
        lambda code: code.replace(
            "var team = q[SF.team], motion", "var team = q[SF.team].x.y, motion"
        ),
    )
    assert "error" in got, f"PLANT DID NOT LAND: {got}"
    assert "TypeError" in got["error"], got["error"]
