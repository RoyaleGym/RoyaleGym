#!/usr/bin/env python3
r"""Generate the pictures and videos the four public READMEs embed.

Everything here is drawn from the engine or from the repos' own data. Nothing outside this
checkout is involved and nothing needs a display, so it runs anywhere the engine is built.

    ..\.venv\Scripts\python RoyaleGym\docs\media\make_media.py --list
    ..\.venv\Scripts\python RoyaleGym\docs\media\make_media.py            # everything
    ..\.venv\Scripts\python RoyaleGym\docs\media\make_media.py determinism ledger

It writes into the sibling repos' ``docs/media/`` folders, which is where the READMEs look
for them. The four repos are cloned side by side into one folder, which every install
recipe already assumes. A sibling that is not there is skipped and said so.

What it needs: the venv, ``royalesim`` built, ``royalegym`` and ``royaleviser`` installed,
and Pillow. The videos also need RoyaleViser's ``media`` extra, which supplies ffmpeg;
without it the videos are skipped and the stills are still written.

Everything it produces is reproducible. The battle is played on a fixed seed with a named
deck, and the viewer's capture freezes the two numbers on screen that would otherwise move
with the wall clock. Running this twice gives the same bytes, except for the two shots
whose subject is time: ``throughput`` times the engine on this machine, and
``live-training-env`` shows a real frame rate.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SIBLING = {name: ROOT / name for name in ("RoyaleSim", "RoyaleGym", "RoyaleLearn", "RoyaleViser")}
WORK = HERE / ".work"  # traces and other intermediates, gitignored

# The seed and the deck the READMEs quote for the Try-it battle. Keeping both means the
# video and the printed output on the page are the same battle.
#
# The deck is named, not drawn at random, and that is load-bearing. An environment with no
# state mutator deals a random 8-card deck out of the card catalogue, and the catalogue's
# size depends on which card table the checkout built. Two machines then play different
# battles from the same seed. Naming eight cards that exist in every table makes the battle
# a function of this file instead.
TRY_IT_SEED = 0
TRY_IT_DECK = ("Knight", "Archer", "Giant", "Minions", "Fireball", "Cannon", "Zap",
               "Musketeer")

TILE_PX = 22       # the mp4: the whole window stays under 1024 px wide
GIF_TILE_PX = 16   # a gif pays for every pixel in every frame


class Skip(Exception):
    """This one cannot be made here, and the reason is worth printing."""


def say(msg: str) -> None:
    print(msg, flush=True)


def out_dir(repo: str) -> Path:
    d = SIBLING[repo] / "docs" / "media"
    if not d.parent.exists():
        raise Skip(f"{repo} is not cloned next to RoyaleGym")
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- the battle


def record_battle(seed: int = TRY_IT_SEED, *, noop_prob: float = 0.7,
                  deck: tuple[str, ...] = TRY_IT_DECK, name: str = "battle") -> Path:
    """Play one whole battle and leave a trace of it, one frame per engine tick.

    Both players choose at random among their legal moves, which is already a valid
    opponent, so no policy is needed. Recording every tick makes the environment step the
    engine one tick at a time instead of ten at once. That is right here and wrong in a
    training run.
    """
    import numpy as np

    from royalegym import ClashParallelEnv, DefaultStateMutator, RandomLegalOpponent, RustEngine
    from royalegym.replay import ReplayRecorder, save_trace

    WORK.mkdir(parents=True, exist_ok=True)
    dest = WORK / f"{name}.msgpack"
    if dest.exists():
        say(f"    reusing {dest.name}")
        return dest

    engine = RustEngine()
    # By name, never by number: a card id is a position in the catalogue, and the
    # positions move between card tables.
    by_name = {c.name: c.card_id for c in engine.cards()}
    ids = [by_name[n] for n in deck]
    rec = ReplayRecorder(frame_every_tick=True)
    env = ClashParallelEnv(engine=engine, recorder=rec,
                           state_mutator=DefaultStateMutator(decks=[ids, ids]))
    obs, _ = env.reset(seed=seed)
    rng, policy = np.random.default_rng(seed), RandomLegalOpponent(noop_prob=noop_prob)
    t0 = time.perf_counter()
    while env.agents:
        obs, *_ = env.step({a: policy.act(obs[a], obs[a]["action_mask"], rng)
                            for a in env.agents})
    save_trace(rec.trace, dest)
    s = env.battle_state
    say(f"    seed {seed}: winner {s.winner}, crowns {[p.crowns for p in s.players]}, "
        f"{len(rec.trace.frames)} frames, {time.perf_counter() - t0:.1f} s to record")
    return dest


def trace(name: str = "battle"):
    from royaleviser.sources import open_source
    return open_source(str(record_battle(name=name)))


def shot(source, dest: Path, **kw) -> None:
    """One capture, with its size printed so an oversized gif is caught here."""
    from royaleviser.capture import capture

    try:
        files = capture(source, str(dest), **kw)
    except ImportError as e:  # the media extra, which carries ffmpeg
        raise Skip(f"{dest.name}: {e}") from None
    total = sum(f.stat().st_size for f in files) / 1e6
    warn = "   TOO BIG for a README, keep a gif under about 6 MB" if (
        dest.suffix == ".gif" and total > 6) else ""
    say(f"    {dest.name}  {total:.2f} MB{warn}")


# --------------------------------------------------------------------------- the palette
#
# The viewer's own colours, so a figure drawn here sits beside a screenshot of the window
# without clashing.

BG = (30, 30, 40)
PANEL = (40, 40, 54)
BORDER = (100, 100, 120)
TEXT = (255, 255, 255)
DIM = (150, 150, 160)
BLUE = (71, 204, 218)
RED = (224, 73, 41)
GREEN = (50, 200, 50)
AMBER = (220, 200, 50)
GRASS = (188, 195, 55)
GRASS2 = (217, 215, 47)
RIVER = (106, 230, 237)
BRIDGE = (255, 175, 120)


def theme_font(size: int, mono: bool = False):
    from PIL import ImageFont

    names = (["consola.ttf", "DejaVuSansMono.ttf", "cour.ttf"] if mono
             else ["segoeui.ttf", "DejaVuSans.ttf", "arial.ttf"])
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def canvas(w: int, h: int):
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (w, h), BG)
    return im, ImageDraw.Draw(im)


def title(d, x: int, y: int, text: str, sub: str = "") -> int:
    d.text((x, y), text, font=theme_font(34), fill=TEXT)
    if sub:
        d.text((x, y + 46), sub, font=theme_font(21), fill=DIM)
        return y + 84
    return y + 52


# --------------------------------------------------------------------------- the shots


def clip(src, stem: Path, *, tick0: int, tick1: int, mp4_every: int, gif_every: int,
         fps: int = 20, crop: str = "left", view=None, compare=None) -> None:
    """A gif and an mp4 of the same stretch of battle.

    The gif is the file the README points at, because GitHub plays a gif inside an ``img``
    tag and will not play an mp4 there. It is drawn smaller and sampled sparser so it stays
    polite on a phone. The mp4 is the good copy for anyone who clicks through.
    """
    for suffix, every, scale in ((".mp4", mp4_every, TILE_PX), (".gif", gif_every, GIF_TILE_PX)):
        shot(src, stem.with_suffix(suffix), ticks=(tick0, tick1, every), scale=scale,
             crop=crop, fps=fps, view=view, compare=compare)


def _view(**kw):
    from royaleviser.render import ViewState
    return ViewState(**kw)


def whole_battle() -> None:
    """RoyaleGym: one whole battle, end to end, in the viewer."""
    src = trace()
    clip(src, out_dir("RoyaleGym") / "whole-battle",
         tick0=0, tick1=src.length, mp4_every=5, gif_every=20)


def battle_page() -> None:
    """RoyaleSim: the busiest minute of a battle, scrubbed tick by tick.

    Deliberately not the same clip as RoyaleGym's whole-battle tile. That one is about the
    match from end to end; this one is about the engine's per-tick detail, so it runs at
    close to real time over the stretch with the most units on the board.
    """
    src = trace()
    t = _busiest_tick(src)
    clip(src, out_dir("RoyaleSim") / "battle-page",
         tick0=max(0, t - 300), tick1=min(src.length, t + 300),
         mp4_every=1, gif_every=3)


def _busiest_tick(src) -> int:
    """The tick with the most units alive, found by sweeping the trace."""
    best, best_n = 0, -1
    for i in range(0, src.length, 5):
        src.seek(i)
        f = src.frame()
        if f is not None and len(f.units) > best_n:
            best, best_n = f.tick, len(f.units)
    say(f"    busiest tick: t{best}, {best_n} units on the board")
    return best


def cards_and_spells() -> None:
    """RoyaleSim: a spell landing on a crowd, then the finish."""
    src = trace(name="crowd")
    t = _spell_on_crowd(src)
    clip(src, out_dir("RoyaleSim") / "cards-and-spells",
         tick0=max(0, t - 60), tick1=min(src.length, t + 140),
         mp4_every=1, gif_every=2, crop="board")


def _spell_on_crowd(src) -> int:
    """The tick where the most units are on the board with a spell in the air.

    Chosen from the trace rather than scripted, so the clip is a thing the engine did
    rather than a thing arranged for the camera.
    """
    best, best_n = 0, -1
    for i in range(src.length):
        src.seek(i)
        f = src.frame()
        if f is None or not f.spells:
            continue
        if len(f.units) > best_n:
            best, best_n = f.tick, len(f.units)
    if best_n < 0:
        raise Skip("no spell was played in this battle")
    say(f"    busiest spell tick: t{best}, {best_n} units on the board")
    return best


def engine_trace() -> None:
    """RoyaleViser: a trace opened at tick 900, with a unit pinned in the inspector."""
    src = trace()
    src.seek(src.index_at_tick(900))
    f = src.frame()
    troops = [u for u in f.units if getattr(u, "kind", None) == 0]
    view = _view(seat=0, hover_uid=troops[0].uid if troops else None)
    d = out_dir("RoyaleViser")
    shot(src, d / "engine-trace.png", ticks=(900, 901, 1), scale=24, crop="full", view=view)
    clip(src, d / "engine-trace", tick0=900, tick1=1500, mp4_every=2, gif_every=4, view=view)


def compare_ghost() -> None:
    """RoyaleViser: the second recording ghosted onto the first."""
    from royaleviser.sources import open_source

    fx = SIBLING["RoyaleViser"] / "tests" / "fixtures"
    a, b = fx / "frames-synthetic-A.jsonl.gz", fx / "frames-synthetic-B.jsonl.gz"
    if not a.exists():
        raise Skip("the scripted battle's recordings are not in this checkout")
    clip(open_source(str(a)), out_dir("RoyaleViser") / "compare-ghost",
         tick0=200, tick1=1000, mp4_every=2, gif_every=4, crop="board",
         compare=open_source(str(b)))


def replay_scrubbed() -> None:
    """RoyaleViser: the scripted battle that ships with the tests, played back."""
    from royaleviser.sources import open_source

    fx = SIBLING["RoyaleViser"] / "tests" / "fixtures" / "frames-synthetic-A.jsonl.gz"
    if not fx.exists():
        raise Skip("the scripted battle's recordings are not in this checkout")
    src = open_source(str(fx))
    clip(src, out_dir("RoyaleViser") / "replay-scrubbed-4x",
         tick0=0, tick1=2400, mp4_every=4, gif_every=8)


ENV_PUBLISHER = r"""
import os, sys, numpy as np
from royalegym import (ClashParallelEnv, ClashSelfPlayVecEnv, DefaultStateMutator,
                       RandomLegalOpponent, RustEngine)
engine = RustEngine()
by_name = {c.name: c.card_id for c in engine.cards()}
deck = [by_name[n] for n in %r]
def make():
    return ClashParallelEnv(engine=RustEngine(),
                            state_mutator=DefaultStateMutator(decks=[deck, deck]))
venv = ClashSelfPlayVecEnv(num_games=4, env_fn=make)   # binds the publisher, watches game 0
obs, _ = venv.reset(seed=%d)
rng, policy = np.random.default_rng(%d), RandomLegalOpponent(noop_prob=0.7)
print("publishing", flush=True)
for _ in range(%d):
    mask = obs["action_mask"]
    acts = np.array([policy.act({k: v[i] for k, v in obs.items()}, mask[i], rng)
                     for i in range(mask.shape[0])])
    obs, *_ = venv.step(acts)
"""


def live_training_env() -> None:
    """RoyaleViser: a real environment streaming in one process, the viewer in another.

    Not the scripted battle and not the stand-in learner. The environment here is the real
    one, stepping a batch of four self-play battles on the engine, and the frames reach the
    viewer over the same UDP socket a training run would use. The learning panel stays empty
    because nothing in this shot trains.

    This is the one shot whose on-screen timing is real rather than frozen, because the
    frame rate is the thing being shown.
    """
    import os
    import subprocess
    import sys
    import tempfile

    from royaleviser.render import ViewState
    from royaleviser.sources import StreamSource

    host, port = "127.0.0.1", 9873
    env = dict(os.environ, ROYALEVISER=f"{host}:{port}",
               SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "pub.py"
        script.write_text(ENV_PUBLISHER % (TRY_IT_DECK, TRY_IT_SEED, TRY_IT_SEED, 400),
                          encoding="utf-8")
        pub = subprocess.Popen([sys.executable, str(script)], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            # The environment sends nothing until a viewer says hello, so the viewer has to
            # be the one that waits.
            from royaleviser.app import run

            dest = out_dir("RoyaleViser") / "live-training-env.png"
            src = StreamSource(host, port)
            code = run(src, ViewState(seat=0), seconds=10.0, shot=str(dest), scale=TILE_PX)
            say(f"    viewer exited {code}; {dest.name} "
                f"{dest.stat().st_size / 1e6:.2f} MB" if dest.exists()
                else f"    viewer exited {code} with no shot")
        finally:
            pub.terminate()
            try:
                pub.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pub.kill()


def site_hero() -> None:
    """The documentation site's own hero, which has to be its own file.

    MkDocs copies only what is under its docs_dir, and the site's docs_dir is
    docs/site/pages, so it cannot reach docs/media. Rather than keep a second copy of the
    README's 2.7 MB gif, the site gets a smaller one: the same battle, drawn at a smaller
    scale and sampled sparser, because a docs front page is read on a phone more often
    than a README is.
    """
    src = trace()
    d = SIBLING["RoyaleGym"] / "docs" / "site" / "pages" / "media"
    d.mkdir(parents=True, exist_ok=True)
    shot(src, d / "whole-battle.gif", ticks=(0, src.length, 24), scale=12, crop="left", fps=18)


# The drawn figures live one to a module, so several people can work on them at once.
FIGURES = {
    "deploy_legality": ("RoyaleSim", "deploy-legality.png"),
    "determinism": ("RoyaleSim", "determinism.png"),
    "throughput": ("RoyaleSim", "throughput.png"),
    "snapshots": ("RoyaleSim", "snapshots.png"),
    "ledger": ("RoyaleSim", "ledger.png"),
    "action_mask": ("RoyaleLearn", "ppo-learner.png"),
    # RoyaleGym's showcase grid: eight of the nine tiles on its front page. The ninth,
    # battle-in-viewer.png, is a screenshot of the viewer and nothing here remakes it.
    "two_apis": ("RoyaleGym", "two-apis.png"),
    "legality_mask": ("RoyaleGym", "legality-mask.png"),
    "self_play_batch": ("RoyaleGym", "self-play-batch.png"),
    "five_pieces": ("RoyaleGym", "five-pieces.png"),
    "start_anywhere": ("RoyaleGym", "start-anywhere.png"),
    "record_and_verify": ("RoyaleGym", "record-and-verify.png"),
    "replay_page": ("RoyaleGym", "replay-page.png"),
    "hidden_information": ("RoyaleGym", "hidden-information.png"),
}


def unreachable() -> list[str]:
    """Every _shot_ module on disk that nothing in FIGURES can reach.

    Eight of these sat unreachable for hours. The modules existed, the READMEs embedded
    the placeholders they were written to replace, and --list named neither, so nothing
    anywhere said the work was undone. A registry that cannot report its own gaps is
    worse than no registry.
    """
    on_disk = {q.stem[len("_shot_"):] for q in HERE.glob("_shot_*.py")}
    return sorted(on_disk - set(FIGURES))


def _figure(name: str):
    repo, filename = FIGURES[name]

    def run() -> None:
        mod = importlib.import_module(f"_shot_{name}")
        dest = out_dir(repo) / filename
        summary = mod.draw(dest)
        say(f"    {dest.name}  {dest.stat().st_size / 1e6:.2f} MB")
        if summary:
            say(f"      {summary}")
    run.__doc__ = f"{repo}: {filename}"
    return run


SHOTS = {
    "whole-battle": whole_battle,
    "battle-page": battle_page,
    "cards-and-spells": cards_and_spells,
    "engine-trace": engine_trace,
    "compare-ghost": compare_ghost,
    "replay-scrubbed": replay_scrubbed,
    "live-training-env": live_training_env,
    "site-hero": site_hero,
    **{n.replace("_", "-"): _figure(n) for n in FIGURES},
}

# Placeholders this script does not make yet, and why. The README keeps a placeholder SVG
# for each, and the caption on it says what the real one has to show.
_LONG_RUN = "the trainer runs, but no run has gone enough iterations to photograph"
NOT_MADE = {
    "measured-routes": "wants a recorded real battle drawn beside the engine; nothing here"
                       " draws one yet",
    "contact-law": "wants a recorded real battle drawn beside the engine; nothing here draws"
                   " one yet",
    "training-run": f"wants a long training run filmed live; {_LONG_RUN}",
    "rollout-workers": "wants steps per second at several worker counts on an idle machine;"
                       " nothing sweeps them yet",
    "frozen-pool-ladder": f"wants a pool of many frozen snapshots; {_LONG_RUN}",
    "checkpoints": f"wants a long run's curve with a resumed run laid on it; {_LONG_RUN}",
    "metrics-sink": f"wants a long run's metrics dashboard; {_LONG_RUN}",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("shots", nargs="*", help="which to make; all of them by default")
    ap.add_argument("--list", action="store_true", help="name them and stop")
    args = ap.parse_args(argv)

    if args.list:
        for name, fn in SHOTS.items():
            print(f"  {name:20s} {(fn.__doc__ or '').splitlines()[0]}")
        print("\nnot made here:")
        for name, why in NOT_MADE.items():
            print(f"  {name:20s} {why}")
        orphans = unreachable()
        if orphans:
            print("")
            print("WARNING: _shot_ modules nothing can reach, so nothing regenerates them:")
            for name in orphans:
                print(f"  _shot_{name}.py")
        return 0

    sys.path.insert(0, str(HERE))
    wanted = args.shots or list(SHOTS)
    unknown = [s for s in wanted if s not in SHOTS]
    if unknown:
        ap.error(f"unknown: {', '.join(unknown)}. Try --list.")

    failed = []
    for name in wanted:
        say(f"{name}")
        t0 = time.perf_counter()
        try:
            SHOTS[name]()
        except Skip as e:
            say(f"    skipped: {e}")
        except Exception:
            traceback.print_exc()
            failed.append(name)
        else:
            say(f"    {time.perf_counter() - t0:.1f} s")
    if failed:
        say(f"\nFAILED: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
