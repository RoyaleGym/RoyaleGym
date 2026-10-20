# Playing on Nulls Royale

[Nulls Royale](https://nulls.gg/servers/nulls-royale/) is a fan-run server for Clash Royale.
With [RoyaleBot](https://github.com/RoyaleGym/RoyaleBot), a bot you made with RoyaleGym plays
real matches there, card by card. Two steps: train on the version Nulls runs, then play.

## 1. Train on the Nulls version

Nulls Royale runs game version 15.535. RoyaleGym plays today's version by default, and a few
cards and rules differ between the two. To train on the rules your bot will meet on Nulls, pass
`client="15.535.29"`:

```python
from royalegym import make_env

for client in (None, "15.535.29"):
    env = make_env(client=client)
    fire_spirits = next(card for card in env.engine.cards() if card.name == "FireSpirits")
    print(client or "today", fire_spirits.hitpoints)
```

```
today 215
15.535.29 217
```

That one setting switches both the card values and the game mechanics to version 15.535's. It
needs royalegym 0.1.24 or newer: `pip install --upgrade "royalegym[all]"`.

!!! tip "In your quickstart.py"
    In `build_env`, change `RustEngine()` to
    `RustEngine(calibration_overrides=client_overrides("15.535.29"))`, and add
    `from royalegym.rust_engine import client_overrides` to the imports at the top of
    `build_env`. It changes the battles, so it needs a new bot: change `save_dir` to a new name
    too.

**How close is it?** We checked this setup against 750 short test scenes recorded on version
15.535. The engine put 99.6% of units within a quarter tile of where the game had them and got
99.9% of hitpoints exactly right, and 684 scenes matched on every tick (RoyaleSim 0.1.32, and the same
scores on every release since, up to 0.1.40).
These are short scenes, not whole battles: over a whole battle, small differences add up. Nulls
can also change cards in its own updates.

## 2. Play it with RoyaleBot

1. Download the Nulls Royale app (the `.apk` file) yourself from
   [Nulls' page](https://nulls.gg/servers/nulls-royale/). RoyaleBot supports Nulls Royale
   15.535.13, the version that page has now.
2. Follow [RoyaleBot's setup](https://github.com/RoyaleGym/RoyaleBot#1-prepare-the-game). It
   prepares your copy of the app, then `royalebot-play --bot <your bot's folder>` plays your
   bot in your matches.

Put **AI** or **Bot** in your in-game name, so the people you play know.
