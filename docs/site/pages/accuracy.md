# How accurate is the engine

<p align="center">
  <img alt="Measured" src="https://img.shields.io/badge/accuracy-measured%2C%20not%20guessed-0b7285?style=flat-square">
  <img alt="Position within a quarter tile, towers left out" src="https://img.shields.io/badge/position%20match%2C%20no%20towers-56.5%25-orange?style=flat-square">
  <img alt="Hitpoints exact, towers left out" src="https://img.shields.io/badge/hitpoints%20exact%2C%20no%20towers-81.4%25-2ea043?style=flat-square">
  <img alt="Corpus" src="https://img.shields.io/badge/corpus-67%20recorded%20battles-555?style=flat-square">
  <img alt="Moving" src="https://img.shields.io/badge/expect%20it%20to-keep%20moving-8957e5?style=flat-square">
</p>

You get a number, and you get the list of things the number is hiding. This page is the short
version. It says how close the engine is to the real game, where it is closest, where it is
worst, and what a bot author should do about that.

The short answer: good, not perfect, and measured.

## How it is checked

Real matches are recorded. The same match is then replayed in the engine, and every unit's
position and hitpoints are compared against the recording, on every tick. A tick is 50 ms of
game time, so a three minute battle is 3,600 comparisons per unit.

There are 67 recorded battles behind the numbers below, all from the run of 2026-09-21. 25 are
scored from start to finish. The other 42 are scored up to the first card the engine cannot play
yet.

!!! note "You cannot re-run this one yourself"
    The recordings of real matches are private, so the accuracy measurement is not something a
    reader can reproduce from a clone. That is a fact about the project and we would rather say
    it than dress it up. The speed numbers elsewhere on this site you can re-run in a minute.
    This one you cannot.

## The headline

How often the engine agrees with the recording:

| how often the engine agrees | counting towers | towers left out | from the run at |
|---|---|---|---|
| a unit is within a quarter of a tile of where it really was | 85.1% | 56.5% | `d872d792711934c2` |
| a unit's hitpoints are exactly right | 81.7%, the run before it | 79.5% | `d872d792711934c2` |

Both right-hand figures are from the run at `d872d792711934c2`. The one cell that is not is marked
where it sits, not in a note above the table, because a reader who arrives here from a link never
sees a note above the table. The position figure moved on 2026-09-22 when the spawner emission
point was corrected, from 49.4%, over the same 73 fixtures and the same harness; hitpoints barely
moved, 79.4% to 79.5%, which is the useful contrast. Correcting where a spawner puts its units
changed where units stand and not how much damage they took.

The right-hand column is the one to look at. Towers do not move and there are six of them in
every battle, so counting them flatters the result. A tower that sits still is not evidence that
the engine walks units correctly.

"Within a quarter of a tile" is a tight bar. A tile is one square of the arena grid, the unit
everything on the board is positioned in, so a quarter of one is a short distance. A unit can be
in the right place to your eye and still be counted as a miss here.

## One unit is close. A crowd is not

<div class="grid cards" markdown>

-   __A Knight on its own__

    ---

    Within a quarter of a tile 72.9% of the time. While it walks to a tower untouched, it is on
    the game's own path, to a fiftieth of a tile, on 73.3% of those ticks. Close to solved.

-   __Goblins__

    ---

    40.6%. Cheap units arriving three or four at once are where the engine and the game come
    apart.

-   __Skeletons__

    ---

    46.0%. Same story. The more bodies touching each other, the worse it gets.

</div>

So the gap is not spread evenly. It is almost all in swarms. Every figure in this section is
from the run at build `d872d792711934c2`, over 73 recorded battles.

## What goes wrong first

The same run that produces the table above also reports what went wrong first in every battle.
That is the useful table, because it says what to fix. Measured at build `d872d792711934c2`:

| cause of the first divergence | battles | share of all the misses, in battles that went wrong this way first |
|---|---|---|
| when a unit dies | 16 | 31.2% |
| where a spawner or a multi-unit card puts its units | 10 | 29.4% |
| attack timing | 10 | 20.5% |
| how units push each other apart on contact | 12 | 18.9% |

**Do not compare this table with the one it replaced.** Contact was 31.0% and is now 18.9%;
death was 21.9% and is now 31.2%. Contact did not improve. Each battle is counted against
whatever went wrong FIRST in it, so correcting the largest cause changes the scene every other
cause is measured on, and a row-by-row comparison of two runs measures nothing.

19 of the 67 battles in the corpus never diverge at all, though most of those are short ones.

These numbers are from 2026-09-21, not the target. The target is that a swarm fight does not diverge
either. Expect the table to move.

## What this means for your bot

Your bot learns the engine. It does not learn the game. Those are the same thing only where the
engine is right.

!!! warning "This section is judgement, not measurement"
    Nobody has trained a bot here and then tested it against the real game. The paragraphs below
    are what we expect from the table above, not something anyone has measured. When somebody
    does run that test, this section gets replaced with what actually happened.

Where the engine is closest, a bot's habits should carry over. Pushing a single tank down a lane,
choosing which lane, spending elixir at the right moment, defending a lone unit with a building:
those lean on walking, routes and timing, and those are the parts measured closest to the game.

Where the engine is furthest, they may not carry over. A bot that learns exactly how a pile of
Skeletons flows around a Knight is learning a crowd behaviour that is 46.0% right. A habit built
on that can be worth nothing, or worse than nothing, in a real match. That 46.0% is from the
run at build `d872d792711934c2`.

The practical version, three lines:

- Before you rely on a specific interaction, look at the table. If it lives in the swarm rows,
  treat what your bot learns about it as a guess.
- Do not tune a reward function against a single exact interaction. Reward the outcome, not the
  choreography.
- Check again in a month. Two of the five causes are being worked on, and the numbers on this
  page are a snapshot.

## An honest summary

This engine is not as accurate as running the real game, which is true by definition. What it
gives you instead is that it is fast, it runs anywhere, it needs no game files, and it tells you
exactly how wrong it is and where.

## Read next

- [The engine](pieces/engine.md) for what the engine does and how you drive it.
- [`docs/replay-parity.md`](https://github.com/RoyaleGym/RoyaleSim/blob/main/docs/replay-parity.md)
  in RoyaleSim's own docs. That page has the per-card breakdown, what is scored, what is not
  scored, and the exact commands. Everything on this page is a summary of it.
- [Troubleshooting](troubleshooting.md) if something on your machine is not doing what this site
  says it should.
