# How accurate is the engine

<p align="center">
  <img alt="Measured" src="https://img.shields.io/badge/accuracy-measured%2C%20not%20guessed-0b7285?style=flat-square">
  <img alt="Position within a quarter tile, towers left out" src="https://img.shields.io/badge/position%20match%2C%20no%20towers-98.8%25-2ea043?style=flat-square">
  <img alt="Hitpoints exact, towers left out" src="https://img.shields.io/badge/hitpoints%20exact%2C%20no%20towers-99.3%25-2ea043?style=flat-square">
  <img alt="Corpus" src="https://img.shields.io/badge/corpus-73%20recordings%20of%2040%20matches-555?style=flat-square">
  <img alt="76 basic cards and one hero" src="https://img.shields.io/badge/cards-76%20basic%20%2B%201%20hero-orange?style=flat-square">
</p>

You get a number, and you get the list of things the number is hiding. This page is the short
version. It says how close the engine is to the real game, where it is closest, where it is
furthest, and what a bot author should do about that.

The short answer: very close on the battles it's measured on, and those use basic cards and one hero.

## How it is checked

Real matches are recorded. The same match is then replayed in the engine, given only the
recorded card plays, decks, card levels and towers, and every unit's position and hitpoints are
compared against the recording, on every tick. A tick is 50 ms of game time, so a three minute
battle is 3,600 comparisons per unit.

The numbers below are from one run on royalesim 0.1.10 (RoyaleSim commit `8e6d7cc`, engine build
`04a8183e395b1e31`), scored on 2026-10-03, over 73 recordings of 40 matches (31 matches were
recorded from both sides). 67 recordings play, covering 37 matches, and all 67 are scored from start
to finish. The other 6 start in the middle of a battle. The percentages are over recordings, so a
match recorded from both sides counts twice.

!!! warning "What these numbers cover"
    - **Basic cards, and one hero.** The recordings use 76 basic cards and one hero (the Hero
      Musketeer, played 3 times), and no champions or evolutions, so those aren't measured here.
    - **Battles the work has seen.** These are the recordings every change to the engine has
      been checked against for weeks. The numbers say how well the engine agrees with them, not
      how it does on battles it was never checked against. No figure for new battles is
      published yet.

!!! note "You cannot re-run this one yourself"
    The recordings of real matches are private, so the accuracy measurement is not something a
    reader can reproduce from a clone. That is a fact about the project and we would rather say
    it than dress it up. The speed numbers elsewhere on this site you can re-run in a minute.
    This one you cannot.

## The headline

How often the engine agrees with the recording:

| how often the engine agrees | counting towers | towers left out |
|---|---|---|
| a unit is within a quarter of a tile of where it really was | 99.5% | 98.8% |
| a unit's hitpoints are exactly right | 98.8% | 99.3% |

That's 602,830 unit-ticks without towers and 1,556,259 with them, all from the run above.

The right-hand column is the one to look at. Towers do not move and there are six of them in
every battle, so counting them flatters the result. A tower that sits still is not evidence that
the engine walks units correctly.

"Within a quarter of a tile" is a tight bar. A tile is one square of the arena grid, the unit
everything on the board is positioned in, so a quarter of one is a short distance. A unit can be
in the right place to your eye and still be counted as a miss here.

The run this page showed before, on 2026-09-22/23 at engine build `d872d792711934c2`, gave 56.5%
and 79.5% without towers. Don't read the two as one jump: that run scored only 270,972 unit-ticks,
because 40 of its recordings stopped at the first card that engine couldn't play yet. Each figure
belongs to its own run.

## Single units and swarms

<div class="grid cards" markdown>

-   __A Knight on its own__

    ---

    Within a quarter of a tile 97.9% of the time, and its hitpoints exactly right 98.9%. While
    it walks to a tower untouched, it is on the game's own path, to a fiftieth of a tile, on 98.3%
    of those ticks.

-   __Goblins__

    ---

    Within a quarter of a tile 99.5% of the time, hitpoints exactly right 99.8%. Cheap units
    arriving three or four at once.

-   __Skeletons__

    ---

    Within a quarter of a tile 98.6% of the time, hitpoints exactly right 99.7%. Many small
    bodies touching each other.

</div>

Swarms were the weak spot of the earlier run (Goblins 40.6%, Skeletons 46.0% there). In this run
they are as close as a single unit.

## What goes wrong first

The same run also reports what went wrong first in every recording that went wrong at all, which
says what to look at next. The cause is the scorer's label for the first unit to drift more than a
tile, not a diagnosis. Towers left out:

| cause of the first divergence, as the scorer labels it | recordings (matches) | share of all the misses, in recordings that went wrong this way first |
|---|---|---|
| attack timing | 5 (3) | 37.6% |
| how units push each other apart on contact | 4 (2) | 31.1% |
| walking | 5 (3) | 30.4% |
| when a unit dies | 2 (1) | 0.6% |

51 of the 67 recordings, 28 of the 37 matches, never diverge at all; they hold the other 0.3% of
the misses. Where units are put by a spawner or a multi-unit card,
the top cause in the earlier run, no longer starts any battle's divergence.

**Do not compare this table row by row with the one it replaced.** Each battle is counted against
whatever went wrong FIRST in it, so fixing one cause changes the scene every other cause is
measured on.

## Matched recently

From RoyaleSim's changelog, 0.1.4 to 0.1.10, each change measured in the game. All of them are in
the run above.

- **The end of a level overtime.** Play stops, the board clears down to the crown towers, and the towers lose the
  same health each tick until one falls. That tower's side loses (0.1.4).
- **Card levels.** Each side, and each card in a deck, can have its own level, as in a real match (0.1.7).
- **Champions.** A champion's ability is one use per deploy, except the Boss Bandit's (0.1.10).
- **What a unit reports.** A unit's card is the card whose play put it on the board: a Tombstone's Skeletons report
  the Tombstone (0.1.5).
- **What a player can see.** A Prince's run-up and an Inferno's ramp, where a unit under ground will come up, how long
  a hero's or champion's ability has left, a Clone's copy, and each side's deck and forms (0.1.8, 0.1.9).
- **Battle logic.** Knockback and death pushback, kamikaze contact, troops' sight of crown towers, target ties, jump
  landings, champion dashes, placement ties, and more in each release.

## What this means for your bot

Your bot learns the engine. It does not learn the game. Those are the same thing only where the
engine is right.

!!! warning "This section is judgement, not measurement"
    Our own bots, built with RoyaleGym, play the real ladder: the account went from about 300 to
    2,765 trophies in its first three days. That's one deck and a few bots, so it says a bot made
    here can win real matches, not which habits carry over. The paragraphs below are what we
    expect from the table above; we haven't published a measurement of them.

With basic cards, the engine and the game agree on almost every tick of the battles measured, so
a bot's habits with those cards should carry over: pushing a tank down a lane, choosing which
lane, spending elixir at the right moment, defending with a building.

Champions and evolutions aren't in the measurement, and only one hero is. They are built from
measurements of the game too, but nobody has scored whole battles with them yet, so what a bot
learns about them is less certain.

The practical version, three lines:

- If your deck uses champions, heroes or evolutions, treat what your bot learns about them as a
  guess until they are measured.
- Do not tune a reward function against a single exact interaction. Reward the outcome, not the
  choreography.
- Check again in a month. The numbers on this page are a snapshot of royalesim 0.1.10.

## An honest summary

This engine is not as accurate as running the real game, which is true by definition. What it
gives you instead is that it is fast, it runs anywhere, it needs no game files, and it tells you
exactly how wrong it is and where.

## Read next

- [The engine](pieces/engine.md) for what the engine does and how you drive it.
- [`docs/replay-parity.md`](https://github.com/RoyaleGym/RoyaleSim/blob/main/docs/replay-parity.md)
  in RoyaleSim's own docs, with the per-card breakdown, what is scored, what is not scored, and
  the commands, for an earlier run.
- [Troubleshooting](troubleshooting.md) if something on your machine is not doing what this site
  says it should.
