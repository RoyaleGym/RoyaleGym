# RoyaleSim

[RoyaleSim](https://github.com/RoyaleGym/RoyaleSim) is the battle engine. It
plays a whole Clash Royale battle (elixir, hands, troops walking and fighting,
spells, towers, overtime and crowns) with no game or phone involved. It's written in
Rust and installs as a Python module, with prebuilt files for Windows, Linux and macOS.

Most of the time you never touch it directly: `RustEngine()` in RoyaleGym drives it for you.

## Same Seed, Same Battle

The engine uses whole numbers only, so the same seed and the same moves give exactly the same
battle, on any computer. A battle that went wrong once can be replayed to go wrong again, which
makes bugs in a bot much easier to find.

## Using the Engine Directly

`RustEngine` is the engine as RoyaleGym sees it. You can drive it yourself: start a battle,
play cards, move the clock forward, and save or load the whole battle at any moment.

```python
from royalegym import MatchSetup, RustEngine, deck_ids
from royalegym.protocol import DeployCommand

engine = RustEngine()
deck = deck_ids(["Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Cannon", "Musketeer"],
                engine.cards())
engine.reset(seed=1, setup=MatchSetup(decks=[deck, deck]))
engine.step([], ticks=100)  # 5 seconds pass; nobody plays

# Blue plays the first card in its hand, in the middle of its own half.
results = engine.step([DeployCommand(team=0, hand_slot=0, x=162_000, y=144_000)], ticks=1)
print("accepted" if results[0].status == 0 else "refused")

# Save the battle, play on, then go back and play the same 10 seconds again.
saved = engine.save_state()
engine.step([], ticks=200)
first_time = engine.state_hash()
engine.load_state(saved)
engine.step([], ticks=200)
print(engine.state_hash() == first_time)
```

```
accepted
True
```

Positions here are in the engine's own units: 18,000 to a tile, with Blue's side at the bottom.
`engine.state()` returns the whole battle, the same `state` that reward functions and
observation builders read.

## How Close Is It to the Real Game?

Close, and still improving. The engine is measured against recordings of real battles, and every
number in it records whether it was measured or is still a guess. The engine is the one part of
the project still expected to change in big ways; RoyaleGym's interface on top of it stays the
same.

The engine's own documentation, with how it works inside and how accurate it is, is in the
[RoyaleSim repository](https://github.com/RoyaleGym/RoyaleSim).
