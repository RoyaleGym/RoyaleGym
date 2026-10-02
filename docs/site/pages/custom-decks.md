# Custom Decks

A battle deals each side eight cards. By default `make_env` gives both sides the same starter
deck. This page shows how to pick your own.

## Name the eight cards

Pass the card names as a list. Both sides get that deck.

```python
from royalegym import make_env

env = make_env(deck=["Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Cannon", "Musketeer"])
obs, info = env.reset(seed=0)
names = {c.card_id: c.name for c in env.engine.cards()}
print("Blue's hand:", [names[c] for c in env.battle_state.players[0].hand])
print("cards you can use:", len(names))
```

```
Blue's hand: ['Cannon', 'Zap', 'Fireball', 'Archer']
cards you can use: 136
```

To see every name, print `[c.name for c in env.engine.cards()]`. A name the engine does not know
is refused, and the error says so.

Always use names, never numbers. A card's number is only its place in the list, and new cards
are added to the end of that list in later versions.

## Give each side its own deck

Pass two lists: Blue's deck, then Red's.

```python
blue = ["Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Cannon", "Musketeer"]
red = ["HogRider", "Valkyrie", "Skeletons", "IceSpirits", "Fireball", "Log", "Cannon", "Musketeer"]
env = make_env(deck=[blue, red])
env.reset(seed=0)
print("Red's hand:", [names[c] for c in env.battle_state.players[1].hand])
```

```
Red's hand: ['Fireball', 'Valkyrie', 'Log', 'Cannon']
```

For a fresh random deck every battle, pass `deck="random"`.

## Evolutions and heroes

Some cards have an evolved form or a hero form. Name them, and they are played that way on both
sides. The card must be in the deck.

```python
env = make_env(deck=blue, evolved=["Knight"], heroes=["Musketeer"])
env.reset(seed=0)
me = env.battle_state.players[0]
print("evolution counters:", [[names[c], plays, nxt] for c, plays, nxt in me.evo])
print("ability buttons:", len(me.abilities), "| moves:", env.action_parser.n_actions)
```

```
evolution counters: [['Knight', 0, 0]]
ability buttons: 1 | moves: 2308
```

A hero has an ability button, so its deck gets extra moves for pressing it: 2308 here instead of
2305. A champion card, such as the Golden Knight, brings a button too.

These outputs were run on 2026-10-01 on engine build `650078fef1aca217`. Hands depend on the
engine version, so yours may differ.

Next: [Rewards](rewards.md), to change what your bot is paid for.
