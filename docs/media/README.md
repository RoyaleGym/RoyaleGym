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
running it twice gives the same bytes. Two shots are deliberate exceptions, because time is
what they show: `throughput` times the engine on the machine that runs it, and
`live-training-env` shows a real frame rate.

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
| `two-apis` | RoyaleGym | One battle through PettingZoo, which drives both seats, and through Gymnasium, which drives one. The two runs are compared after every step, and a run with the seats swapped shows the comparison can fail. |
| `legality-mask` | RoyaleGym | Where each card in hand may go on the opening step, one board per card. Every move is checked against the engine's `check_deploy`, and again with the mask shifted one row to show the check can fail. |
| `self-play-batch` | RoyaleGym | A self-play batch: N battles become 2N rows, and the two rows of a battle hold the same towers with own and enemy swapped. |
| `five-pieces` | RoyaleGym | The swappable pieces of `ClashParallelEnv`, read off its constructor, with the default and the shipped choices for each. |
| `start-anywhere` | RoyaleGym | Four starting boards, one per state mutator, drawn from what `reset()` left behind. |
| `record-and-verify` | RoyaleGym | A battle recorded, saved, loaded and re-run on a fresh engine with every tick hash matching, beside a copy with one deploy moved a tile, where they stop matching. |
| `replay-page` | RoyaleGym | A whole battle as one HTML page, decoded back out of the written file, and a count of the outside references it makes, which is zero. |
| `hidden-information` | RoyaleGym | A to-scale map of one seat's observation vector: its own hand, what it has worked out about the enemy, and the enemy hand `Reveal(enemy_hand=True)` adds. |
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

**Two in RoyaleSim need a recording of a real battle drawn beside the engine**
(`measured-routes`, `contact-law`). Nothing here draws one yet. They are also the two tiles
carrying the project's strongest claims, so they are worth doing properly rather than
approximating.

**Five in RoyaleLearn need real runs of the trainer** (`training-run`, `rollout-workers`,
`frozen-pool-ladder`, `checkpoints`, `metrics-sink`). RoyaleLearn's trainer exists and trains.
What is missing is a run long enough to photograph. A rating curve, a ladder of frozen
opponents, a resumed run lying on the original's curve and a metrics dashboard all need a run
of many iterations, and so far runs have been a few iterations long. `rollout-workers` needs
something slightly different: steps per second measured at several worker counts on an idle
machine, and nothing sweeps that yet. There is no honest way to fake any of them.

`make_media.py --list` prints all seven with the reason for each.

Some pictures in the READMEs are not made here, and nothing else in the four repos remakes
them: the `family.svg` diagrams, which are drawn by hand, and a few screenshots of the viewer
(`battle-in-viewer.png` here, `engine-battle-viewer.png` in RoyaleSim, `self-play-env.png` in
RoyaleLearn and the `tile-*.png` stills in RoyaleViser).

## The drawn figures

Fourteen of the shots are drawn rather than captured, one to a module, `_shot_*.py`. Each one
exposes `draw(out_path)` and computes everything on it at the moment it runs, from the engine
or from the repos' own files.

They are drawn at about 1000 pixels wide and have to survive being shrunk to 360, which is
roughly what a three-column README table gives them on a desktop. That is the whole of their
design constraint: nothing smaller than 34 pixel type, one headline, and very few elements. An
earlier set was drawn at 1280 with 18 pixel labels and was unreadable in the place it was
going to be seen. If you add one, render it, shrink it to 360 wide, and look at it before you
call it done.
