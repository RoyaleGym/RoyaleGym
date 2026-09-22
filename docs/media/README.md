# README media, and the generator for all four repos

The generator lives here: [`make_media.py`](make_media.py). It writes into RoyaleSim,
RoyaleGym, RoyaleLearn and RoyaleViser, because the four repos are cloned side by side into
one folder and every install recipe already assumes that. A sibling that is not there is
skipped and says so.

```
.venv\Scripts\python RoyaleGym\docs\media\make_media.py --list
.venv\Scripts\python RoyaleGym\docs\media\make_media.py whole-battle determinism
.venv\Scripts\python RoyaleGym\docs\media\make_media.py                # everything
```

Run it from the folder that holds the repos. It needs the engine built, `royalegym` and
`royaleviser` installed, and Pillow. The videos also need RoyaleViser's `media` extra, which
supplies ffmpeg; without it the videos are skipped and the stills are still written.

**It is reproducible.** The battle is played on a fixed seed with a named deck, and
RoyaleViser's `capture()` freezes the two numbers on screen that move with the wall clock, so
running it twice gives the same bytes. One shot is a deliberate exception and says so.

**The deck is named on purpose.** An environment with no state mutator deals a random 8-card
deck out of the card catalogue, and the catalogue's size depends on which card table the
checkout built. Two machines then play different battles from the same seed. Naming eight
cards makes the battle a function of this file. The cards are looked up by name and never by
number, because a card id is a position in the catalogue and the positions move between
tables.

Each video is a `.gif` and an `.mp4` of the same thing. The README points at the gif, because
GitHub plays a gif inside an `<img>` tag and will not play an mp4 there.

## What it makes, and where each one goes

| Shot | Repo | What it shows |
|---|---|---|
| `whole-battle` | RoyaleGym | One whole battle end to end in the viewer, both players choosing at random among their legal moves. |
| `battle-page` | RoyaleSim | The busiest stretch of the same battle, found by sweeping the trace for the tick with the most units alive. One frame per engine tick. |
| `cards-and-spells` | RoyaleSim | A spell landing on a crowd, found the same way: the tick with a spell in the air and the most units on the board. |
| `deploy-legality` | RoyaleSim | The arena coloured by what `check_deploy` answers, before and after an enemy princess tower falls. |
| `determinism` | RoyaleSim | One seed run twice and resumed once from a snapshot, with the per-tick hashes side by side. |
| `throughput` | RoyaleSim | A real transcript of `tools/throughput.py`, the median of five runs, with the spread of all five. |
| `snapshots` | RoyaleSim | One saved position loaded into several engines and played on differently. |
| `ledger` | RoyaleSim | All 148 engine constants graded by how well each one is known. |
| `action-mask` | RoyaleLearn | The legality mask of one real decision, drawn as one board per hand card. |
| `engine-trace` | RoyaleViser | A trace opened at tick 900, with a unit pinned in the inspector. |
| `compare-ghost` | RoyaleViser | The second recording of the scripted battle ghosted onto the first. |
| `replay-scrubbed` | RoyaleViser | The scripted battle that ships with the tests, played back. |
| `live-training-env` | RoyaleViser | A real environment streaming over UDP to the viewer in another process. This is the shot whose on-screen timing is real, because the frame rate is the thing being shown. |
| `site-hero` | the docs site | A smaller copy of the whole battle, for `docs/site`. MkDocs copies only what is under its own docs folder, so the site cannot reach this one. |

## What it does not make

A *placeholder* is a generated SVG whose alt text begins "Image placeholder:" or "Video
placeholder:" and says what the real picture must show. Seven are left across the four repos,
in two groups.

**Two in RoyaleSim need a recording of a real battle beside the engine** (`measured-routes`,
`contact-law`). Those recordings are private, so neither can be made from a public checkout,
and neither is a scripting problem. They need a decision about what a public picture of a real
recording may contain. They are also the two tiles carrying the project's strongest claims, so
they are worth doing properly rather than approximating.

**Five in RoyaleLearn need a training run** (`training-run`, `rollout-workers`,
`frozen-pool-ladder`, `checkpoints`, `metrics-sink`). The harness is not written yet. There is
nothing to photograph and no honest way to fake it.

`make_media.py --list` prints both lists with the reason for each.

## The drawn figures

Six of the shots are drawn rather than captured, one to a module, `_shot_*.py`. Each one
exposes `draw(out_path)` and computes everything on it from the engine at the moment it runs.

They are drawn at about 1000 pixels wide and have to survive being shrunk to 360, which is
roughly what a three-column README table gives them on a desktop. That is the whole of their
design constraint: nothing smaller than 34 pixel type, one headline, and very few elements. An
earlier set was drawn at 1280 with 18 pixel labels and was unreadable in the place it was
going to be seen. If you add one, render it, shrink it to 360 wide, and look at it before you
call it done.
