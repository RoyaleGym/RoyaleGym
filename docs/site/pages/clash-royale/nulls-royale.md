# Playing on Nulls Royale

[Nulls Royale](https://nulls.gg/servers/nulls-royale/) is a fan-run server for Clash Royale.
With [RoyaleBot](https://github.com/RoyaleGym/RoyaleBot), a bot you made with RoyaleGym plays
real matches there, card by card. Two steps: train on the version Nulls runs, then play.

## 1. Train on the Nulls version

Nulls Royale runs game version 16.635 (its page offers 16.635.1). RoyaleGym plays the current
game's rules by default, but its default card values are an older version's, and a few cards
differ. To train on the cards your bot will meet on Nulls, pick the 16.635 card table:

```python
from royalegym import RustEngine, make_env

NULLS = {"cards.CARD_TABLE": "client160635007_20261103"}
for overrides in (None, NULLS):
    env = make_env(engine=RustEngine(calibration_overrides=overrides))
    musketeers = next(card for card in env.engine.cards() if card.name == "ThreeMusketeers")
    print("16.635" if overrides else "default", musketeers.hitpoints)
```

```
default 883
16.635 906
```

`pip install --upgrade "royalegym[all]"` gets you an engine that has this table.

!!! tip "In your quickstart.py"
    In `build_env`, change `RustEngine()` to
    `RustEngine(calibration_overrides={"cards.CARD_TABLE": "client160635007_20261103"})`. It
    changes the battles, so it needs a new bot: change `save_dir` to a new name too.

**How close is it?** The table is the game's own 16.635 card data. We haven't checked it against
battles recorded on Nulls yet, and Nulls can change cards in its own updates.

**Nulls' previous version, 15.535.** `make_env(client="15.535.29")` switches both the card values
and the game mechanics to version 15.535's. We checked that setup against 750 short test scenes
recorded on 15.535. The engine put 99.6% of units within a quarter tile of where the game had
them and got 99.9% of hitpoints exactly right, and 684 scenes matched on every tick (RoyaleSim
0.1.32, and the same scores on every release since, up to 0.1.48). These are short scenes, not
whole battles: over a whole battle, small differences add up.

## 2. Play it with RoyaleBot

1. Download the Nulls Royale app (the `.apk` file) yourself from
   [Nulls' page](https://nulls.gg/servers/nulls-royale/). RoyaleBot supports Nulls Royale
   16.635.1, the version that page has now, and the earlier 15.535.13.
2. Follow [RoyaleBot's setup](https://github.com/RoyaleGym/RoyaleBot#1-prepare-the-game). It
   prepares your copy of the app, then `royalebot-play --bot <your bot's folder>` plays your
   bot in your matches.

Put **AI** or **Bot** in your in-game name, so the people you play know.
