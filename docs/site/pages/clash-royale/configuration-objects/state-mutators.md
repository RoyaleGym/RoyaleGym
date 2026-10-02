# State Mutators

A state mutator decides how each battle starts: which decks each side gets, and whether the
battle starts from the beginning or partway through. The environment asks it for a new start
every time a battle ends.

## Choosing Decks

`DefaultStateMutator` starts a normal battle from the first second. Give it the decks by card
name, Blue's first:

```python
from royalegym import DefaultStateMutator, make_env

deck = ["Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Cannon", "Musketeer"]
env = make_env(state_mutator=DefaultStateMutator(decks=[deck, deck]))
obs, info = env.reset(seed=0)

names = {card.card_id: card.name for card in env.engine.cards()}
print([names[card] for card in env.battle_state.players[0].hand])
```

```text
['Cannon', 'Zap', 'Fireball', 'Archer']
```

Each side's deck is shuffled at the start of the battle, so the opening hand changes with the
seed. With no decks, `DefaultStateMutator()` deals each side eight random cards every battle.

`make_env(deck=...)` is a shortcut for the same thing: `deck=` takes one deck for both sides,
two decks, or `"random"`.

### Evolutions and heroes

Some cards have an evolved form or a hero form. The easiest way to use them is `make_env`:

```python
from royalegym import make_env

env = make_env(evolved=["Knight"], heroes=["Musketeer"])
print(env.action_space("blue"))
```

```
Discrete(2308)
```

Both sides get the forms you name, and the cards must be in the deck. An evolved card plays as
its evolution each time it has been played enough times to charge up, as in the game. A hero
always plays as its hero, and its ability button adds moves (here, 2308 instead of 2305). A card
that has no such form in the engine is refused with an error that says so.

`DefaultStateMutator(decks=..., forms=...)` does the same with one number per card: `0` the card
itself, `1` its evolution, `2` its hero.

## The Ones That Come With RoyaleGym

| Mutator | Starts each battle |
|---|---|
| `DefaultStateMutator(decks=...)` | From the first second, with the decks you give, or random ones. |
| `MidGameStateMutator(...)` | Partway through, with random elixir and tower damage. Good for practising endings. |
| `ScriptedBoardStateMutator(spawns=...)` | With units already on the board, for drills like "defend this push". |
| `SnapshotStateMutator(snapshots)` | From exact saved moments of earlier battles. |
| `DeckCurriculumStateMutator(deck, ...)` | With your deck on one or both sides, against a pool of other decks. |
| `WeightedStateMutator([(mutator, weight), ...])` | From one of several mutators, picked at random by weight each battle. |

One difference from RLGym: you don't chain mutators one after another. A mutator describes the
whole start of a battle, and `WeightedStateMutator` picks between them.

## How They Work

A state mutator has one method:

```py
# Called before every battle. `rng` is a numpy random generator, seeded by env.reset(seed=...).
# `cards` is the engine's card list. Return a MatchSetup describing the start.
def build(self, rng, cards): ...
```

A `MatchSetup` holds the two decks as card ids, and optionally the starting clock, elixir,
tower health and units on the board. `deck_ids(names, cards)` turns card names into ids.

## Creating Your Own

This one deals each side one of two decks at random, every battle:

```python
from royalegym import MatchSetup, StateMutator, deck_ids


class TwoDecksMutator(StateMutator):
    """Deals each side one of two decks at random, every battle."""

    DECKS = [
        ["Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Cannon", "Musketeer"],
        ["HogRider", "Valkyrie", "Skeletons", "IceSpirits", "Fireball", "Log", "Cannon", "Musketeer"],
    ]

    def build(self, rng, cards):
        blue = self.DECKS[rng.integers(2)]
        red = self.DECKS[rng.integers(2)]
        return MatchSetup(decks=[deck_ids(blue, cards), deck_ids(red, cards)])
```

```python
from royalegym import make_env

env = make_env(state_mutator=TwoDecksMutator())
names = {card.card_id: card.name for card in env.engine.cards()}
for seed in range(3):
    env.reset(seed=seed)
    print([names[card] for card in env.battle_state.players[1].hand])
```

```text
['Skeletons', 'Log', 'HogRider', 'Musketeer']
['IceSpirits', 'HogRider', 'Valkyrie', 'Skeletons']
['Fireball', 'Giant', 'Knight', 'Musketeer']
```

The same seed always gives the same start, so a battle can be repeated exactly.
