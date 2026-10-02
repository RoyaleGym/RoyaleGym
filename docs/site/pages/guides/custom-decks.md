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

Some cards have an evolved form or a hero form. Name them, and both sides get them. The card must
be in the deck. An evolved Knight plays as its evolution every third time you play it (most cards;
a few cycle faster). A hero always plays as its hero form.

```python
env = make_env(deck=blue, evolved=["Knight"], heroes=["Musketeer"])
env.reset(seed=0)
me = env.battle_state.players[0]
print("ability buttons:", len(me.abilities), "| moves:", env.action_parser.n_actions)
```

```
ability buttons: 1 | moves: 2308
```

A deck with a hero or a champion gets three extra moves, one per button slot, so 2308 instead of
2305. Here only one slot holds a button. A slot with no button is never allowed.

These outputs were run on 2026-10-01 on engine build `a581356680589e2c`. Hands depend on the
engine version, so yours may differ.

Next: [Rewards](rewards.md), to change what your bot is paid for.
