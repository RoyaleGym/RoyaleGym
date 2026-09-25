# The viewer

[![repo](https://img.shields.io/badge/repo-RoyaleViser-8957e5?style=flat-square&logo=python&logoColor=white)](https://github.com/RoyaleGym/RoyaleViser)
![no engine build](https://img.shields.io/badge/engine%20build-not%20needed-2ea043?style=flat-square)
![capture](https://img.shields.io/badge/capture-png%20%C2%B7%20mp4%20%C2%B7%20gif-8957e5?style=flat-square)
![draw cost](https://img.shields.io/badge/draw-2--6%20ms%20per%20frame-2ea043?style=flat-square)
![unwatched](https://img.shields.io/badge/unwatched-193%20ns%20per%20step-555?style=flat-square)

**This is the one you can run right now.** Two recordings of a scripted battle are committed with
the tests, about 20 KB each, so they came down with your clone. You need the shared virtual
environment and one `pip install`. No engine build. No Rust. No game files. No recordings of
your own.

That makes it the friendliest way to see whether any of this works on your machine before you
commit an evening to the rest of the setup.

Once you are past that, it is how you see what your bot did instead of guessing it from numbers.
It draws one picture per tick, which is the game's own 50 ms step, so 20 pictures for every
second of the battle.

It always runs as its own separate program. Your training run never waits for it. When nobody is
watching, it costs that run 193 nanoseconds a step.

## Run it

From the RoyaleViser folder, after the venv and `pip install -e RoyaleViser`:

=== "Windows"

    ```
    ..\.venv\Scripts\python -m royaleviser tests\fixtures\frames-synthetic-A.jsonl.gz --compare tests\fixtures\frames-synthetic-B.jsonl.gz --speed 4 --seconds 8
    ```

=== "macOS and Linux"

    !!! warning "UNVERIFIED"
        Nobody has run this from a clean install yet. Only the Windows form of the command was
        run for this page; this is the same command with the other platform's paths.

    ```
    ../.venv/bin/python -m royaleviser tests/fixtures/frames-synthetic-A.jsonl.gz --compare tests/fixtures/frames-synthetic-B.jsonl.gz --speed 4 --seconds 8
    ```

A window opens and plays the battle at four times speed. After eight seconds it closes itself and
prints how long each picture took to draw. Here is what that printed on a 4-core laptop with
several other jobs running:

```
royaleviser: 250 draws, mean 6.47 ms, max 467.37 ms
```

250 pictures at 6.47 ms each on average. A replay needs 20 a second, which is 50 ms each, so
there is a lot of room to spare. The 467 ms is the very first draw, which builds the board and
loads the fonts, and it happens once.

Two things you will see in the window. The second recording is drawn on top of the first as
hollow ghosts, and since these two recordings are identical the ghosts sit exactly on the units
and never step off. The compare panel finishes at `395 ticks compared, 0 differ`, under a
`GAME OVER  Blue wins` banner.

!!! note "That battle is a script, not something the engine played"
    One frame of these fixtures stands for six ticks of the script, so the units move at six
    times their scripted speed. It is an honest look at the window and a poor look at how the
    engine actually plays. For that, open a trace.

Drop `--seconds` and the window stays open until you close it.

**Keys.** Space plays and pauses. The arrow keys step one frame at a time. Clicking a unit pins
it, and clicking the bar seeks. `c` turns the compare ghost on and off. `s` saves a PNG. `h`
lists every key there is.

**No screen?** Set `SDL_VIDEODRIVER=dummy` and no window opens at all. `--seconds` and `--shot`
keep working. That is how you drive the viewer over SSH or on a build machine. The run above was
done that way.

## The three things it can open

Same command, three kinds of source.

| Source | What it is | What you need |
|---|---|---|
| A **recording** | one line of JSON per frame, 20 frames a second, of a real match | this package only |
| A **trace** | a battle saved from the engine, one frame per tick if you ask for it | this package plus RoyaleGym, and RoyaleSim's data folder |
| A **stream** | a program running right now, one frame per environment step | this package only, plus the running program |

The filenames below are examples. Put your own in their place. The command you ran above is the
same command with a file that ships with the repo.

The three lines open the three kinds of source: a recording, like the two that ship in
`tests/fixtures`; a trace saved from the engine; and an environment running right now.

```
python -m royaleviser frames-my-match.jsonl.gz
python -m royaleviser battle.msgpack --start-tick 900
python -m royaleviser --stream 127.0.0.1:9870
```

For the stream, set `ROYALEVISER=127.0.0.1:9870` before your training run starts and change no
code. Nothing is sent until a viewer says hello, and sending stops three seconds after the last
viewer goes away, so you can leave it switched on in a run nobody is watching.

A viewer shows one battle at a time, so when you run eight games at once the vectorised
environment picks game 0 and hands the sender to it. Eight games all reaching for one port is an
error, not eight streams.

Recordings of real matches are private, so the repo does not ship one. The three tests that need
one skip, and they say out loud that they skipped, so nobody mistakes a skip for a pass.

## Save a picture, with no window at all

This is how you get an image for your own write-up, your pull request or your Discord post.
`capture` writes what the window would show straight to a file. No window opens, no display is
needed, and the clock on screen is frozen so the same source gives you the same bytes.

```python
from royaleviser.capture import capture
from royaleviser.sources import open_source

out = capture(open_source("tests/fixtures/frames-synthetic-A.jsonl.gz"),
              "shot.png", ticks=(120, 121, 1), scale=24, crop="left")
for p in out:
    print(p.name, p.stat().st_size, "bytes")
```

```
shot.png 67707 bytes
```

That was run from the RoyaleViser folder with no display driver set at all, twice, and gave
exactly the same file size both times.

A clip is the same call with a different suffix. Ticks 0 to 400, every eighth tick, at half the
pixel size:

```python
out = capture(open_source("tests/fixtures/frames-synthetic-A.jsonl.gz"),
              "clip.gif", ticks=(0, 400, 8), scale=16, crop="left")
```

```
clip.gif 256233 bytes
```

A quarter of a megabyte for the scripted battle. The whole signature:

```
capture(source, out, *, ticks=None, scale=24, view=None, crop="full", fps=20,
        compare=None, theme=DEFAULT, live_timing=False) -> list[Path]
```

- The suffix picks the format. `.png` writes one file per frame, numbered `name-0000.png` when
  the range holds more than one. `.mp4` and `.gif` encode the range at `fps`.
- `ticks` is `(start, stop, step)` in battle ticks, which is the clock the window shows, not
  frame numbers. Leave it out and you get the whole source.
- `crop` is `full`, `board`, `left`, or a rectangle. `left` drops the inspector column, which is
  usually what you want for a picture of the board.
- `compare` ghosts a second source at the same tick. `view` carries the seat, the overlays, and
  `hover_uid`, which pins a unit in the inspector.
- PNG needs nothing extra. mp4 and gif are encoded by ffmpeg, which arrives with the optional
  `media` extra. The gif above was made with that extra installed.

## When you would touch it

- You want to see what your bot actually did. A number that says the bot lost does not tell you
  it walked a Giant into a Fireball.
- You want to see where the engine and the real game disagree. Open two recordings of one match
  and the second is drawn as hollow ghosts on the first. On a tick where they differ, the ghost
  steps off the unit.
- You want a picture or a clip for a write-up.
- You want to watch a training run while it is running, with your loss and your ladder rating in
  a panel beside the board.
- You want a different front end. The viewer hands you one frame object that anything else can
  draw from.

## When you would not

| You want to | Go here instead |
|---|---|
| change what the bot does | [The environments](environments.md) |
| fix how a unit behaves | [The engine](engine.md) |
| run the training loop | [The learner](learner.md) |
| share a battle with someone who has installed nothing | RoyaleGym's replay page, one self-contained HTML file you double-click |

## What it will not show you, and why

Every gap here is the source not carrying the information. None of them is the window refusing
to draw something.

- A stream carries one frame per environment step, which is 10 ticks by default, not one frame
  per tick. Want every tick? Record a trace with `frame_every_tick=True` and open that.
- A recording gives no unit a radius and no flying flag, so every unit is drawn at one size and
  air units look like ground units.
- A trace or a stream gives no unit a path, so the path overlay draws nothing. What each unit is
  aiming at, its status effects and the shots in flight do come through, but only with an engine,
  RoyaleGym and RoyaleViser new enough to carry them. All three gained this late on 2026-09-24, so
  an older copy of any of the three, or a trace saved with one, shows none of them.
- Spells in a recording arrive as projectiles, apart from the few whose effect is tagged with
  its card. There are no rage, poison or freeze zones.
- A recording holds only the recording player's hand. The opponent's shows as
  `hand: not in this source`.
- The learning panel has been filled by a real training run, and RoyaleViser's README shows that
  picture: a laptop run on 2026-09-22, caught at iteration 33. Two tiles in it are not data.
  `ELO vs pool` reads 1200 and `win rate` reads 0.0 % because a learner that is still training is
  never played against the pool under its own name, so those two read the same whatever the run is
  doing. Everything else on the panel is what the learner reported that minute. A status arrives
  once per training iteration rather than once per step, so the panel shows how old the last one is
  beside its heading.

## Where the detail is

- [RoyaleViser's README](https://github.com/RoyaleGym/RoyaleViser) for the whole picture,
  including how to save a battle from the engine and how to send your training numbers.
- [`docs/internals.md`](https://github.com/RoyaleGym/RoyaleViser/blob/main/docs/internals.md)
  for the frame model, the three formats, the stream protocol, every command-line flag and the
  performance table.
