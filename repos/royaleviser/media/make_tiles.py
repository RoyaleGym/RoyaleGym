#!/usr/bin/env python3
"""The guide's `tile-*.png` crops and `viewer-learning.png`, every one regenerable.

    ..\\..\\..\\.venv\\Scripts\\python docs\\media\\make_tiles.py            # every tile
    ..\\..\\..\\.venv\\Scripts\\python docs\\media\\make_tiles.py compare    # one of them

Every tile here is a crop of the real window drawn on `tests/fixtures`, the scripted battle
as the two seats would have recorded it. No screen and no clock are involved: the window is
rendered off-display and the same source and tick give the same bytes, so a regenerated tile
is a clean diff or no diff at all.

That is the point of the file. These two tiles were hand-taken screenshots of a real match,
which made them the only pictures in the repo that nobody else could reproduce and that no
change to the viewer would ever update. A picture of a window is worth keeping only while it
still shows what the window does.

A tile is a CROP because the window is 1082 x 832 and a README thumbnail of the whole thing
shows nothing. The crops are named regions rather than magic numbers: see `TILES`.

Three more were hand-taken until 2026-10-01 and are made here now, on the current window:

- `event-log` and `inspector` are crops of the window on the engine battle RoyaleGym's
  `docs/media/make_media.py` records (the battle behind `engine-trace` and `viewer-trace.png`),
  at the same tick with the same unit pinned. They need the engine, from the sibling checkouts.
- `live-stream` and `learning` are the window ATTACHED over UDP: `tests/run_stream.py` streams
  the scripted battle and a scripted learner's status, so nothing in them comes from a real
  training run. `learning` is the whole window, written to `docs/viewer-learning.png`.

A caption that quotes something drawn in one of these pictures (an fps, a unit's name) is
updated in the same commit as the picture.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import pygame  # noqa: E402

from royaleviser.app import KEYS, App  # noqa: E402
from royaleviser.render import ViewState  # noqa: E402
from royaleviser.sources import CaptureSource  # noqa: E402

SCALE = 24


def fixtures() -> tuple[Path, Path]:
    import synthetic  # the generator that also writes them; tests check the two agree

    return synthetic.fixture_paths()


def window(
    main: Path, other: Path | None, tick: int, *, scrub: bool = False, **view_kw
) -> pygame.Surface:
    """The whole window, drawn on ``main`` (and ``other`` ghosted) at ``tick``.

    ``scrub`` plays every frame up to ``tick`` first, which is what a person watching does and
    the only way the compare totals are the battle's rather than one frame's.
    """
    pygame.init()
    srcs = [CaptureSource(main)]
    if other is not None:
        srcs.append(CaptureSource(other))
    app = App(srcs, ViewState(**view_kw), scale=SCALE)
    app.renderer.help_lines = KEYS
    end = srcs[0].index_at_tick(tick)
    for i in range(end + 1) if scrub else (end,):
        srcs[0].seek(i)
        app.pull()
    app.draw(0.0)
    for s in srcs:
        s.close()
    return app.renderer.surface.copy()


def crop(surface: pygame.Surface, rect: tuple[int, int, int, int], to: Path) -> None:
    out = pygame.Surface((rect[2], rect[3]))
    out.blit(surface, (0, 0), rect)
    pygame.image.save(out, str(to))
    print(f"{to.name}: {rect[2]}x{rect[3]} from {rect[:2]}")


def compare() -> None:
    """The compare panel at the end of the battle, with the two recordings' agreement on it.

    Both fixtures are the same scripted battle recorded from the two seats, so the honest
    number for them is zero: what the panel is showing is that it CAN say so tick by tick.
    """
    a, b = fixtures()
    import synthetic

    surf = window(a, b, synthetic.LAST_TICK, scrub=True)
    crop(surf, (460, 320, 528, 297), HERE / "tile-compare.png")


def paths_and_targets() -> None:
    """Units with the route they walk and the thing they attack, both drawn."""
    a, _ = fixtures()
    surf = window(a, None, 162, show_paths=True, show_targets=True)
    crop(surf, (345, 225, 432, 243), HERE / "tile-paths-and-targets.png")


ENGINE_TICK = 900  # make_media's engine-trace tick: these tiles show the same moment


def engine_window() -> tuple[pygame.Surface, str]:
    """The window on make_media's engine battle at ENGINE_TICK with its first troop pinned,
    exactly as ``engine_trace`` there pins it; and the pinned unit's name, for the caption."""
    import importlib.util

    path = REPO.parent / "RoyaleGym" / "docs" / "media" / "make_media.py"
    if not path.exists():
        raise SystemExit(f"the engine tiles need RoyaleGym beside this repo ({path})")
    spec = importlib.util.spec_from_file_location("make_media", path)
    make_media = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_media)
    make_media.WORK = HERE / ".work"  # its recorded battle stays in this repo, gitignored
    pygame.init()
    src = make_media.trace()
    src.seek(src.index_at_tick(ENGINE_TICK))
    frame = src.frame()
    troops = [u for u in frame.units if u.kind == 0]
    pinned = troops[0] if troops else None
    app = App([src], ViewState(seat=0, hover_uid=pinned.uid if pinned else None), scale=SCALE)
    app.renderer.help_lines = KEYS
    app.pull()
    app.draw(0.0)
    src.close()
    return app.renderer.surface.copy(), pinned.name if pinned else ""


def event_log() -> None:
    """The event list: plays, spawns and deaths, with their tick and tile."""
    surf, _ = engine_window()
    crop(surf, EVENT_LOG_RECT, HERE / "tile-event-log.png")


def inspector() -> None:
    """Every raw field of the pinned unit."""
    surf, name = engine_window()
    crop(surf, INSPECTOR_RECT, HERE / "tile-inspector.png")
    print(f"tile-inspector.png pins {name}")


def stream_shot(to: Path) -> pygame.Surface:
    """The whole window attached over UDP to run_stream's scripted publisher and learner."""
    import run_stream

    if run_stream.main(["--seconds", "6", "--shot", str(to)]) != 0:
        raise SystemExit(f"run_stream wrote no picture to {to}")
    return pygame.image.load(str(to))


def live_stream() -> None:
    """The window on a stream: LIVE, its frame rate, events arriving."""
    tmp = HERE / ".live-stream-full.png"
    try:
        crop(stream_shot(tmp), LIVE_STREAM_RECT, HERE / "tile-live-stream.png")
    finally:
        tmp.unlink(missing_ok=True)


def learning() -> None:
    """The whole window with the learning panel filled by run_stream's scripted learner."""
    stream_shot(REPO / "docs" / "viewer-learning.png")
    print("viewer-learning.png: the whole window")


# The crops, chosen on the 1082 x 832 window at scale 24 (theme.layout).
EVENT_LOG_RECT = (0, 140, 480, 270)  # the status block and its events, and the board's edge
INSPECTOR_RECT = (522, 5, 560, 315)  # the board's right half and the pinned unit's fields
LIVE_STREAM_RECT = (0, 140, 480, 270)  # LIVE, the frame rate, the events, the board's edge

TILES = {
    "compare": compare,
    "paths-and-targets": paths_and_targets,
    "event-log": event_log,
    "inspector": inspector,
    "live-stream": live_stream,
    "learning": learning,
}


def main(argv: list[str]) -> int:
    names = argv or list(TILES)
    unknown = [n for n in names if n not in TILES]
    if unknown:
        print(f"unknown tile(s): {', '.join(unknown)}; have {', '.join(TILES)}", file=sys.stderr)
        return 2
    for name in names:
        TILES[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
