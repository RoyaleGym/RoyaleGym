# Observation Builders

An observation builder decides what the bot sees. After every step it turns the battle into
numbers, one set for each seat, and hands them to the bot.

## What the Bot Sees by Default

`SpatialObsBuilder` gives each seat four things:

| Key | Shape | What it holds |
|---|---|---|
| `spatial` | 20 x 32 x 18 | The arena as a grid of tiles, in 20 layers: where each side's troops, buildings and towers are, their health, spells in the air, water and the river. |
| `vector` | one long list | Everything that isn't a place on the board: your hand and the next card, your elixir, the cards you've seen the opponent play, tower health, crowns and the clock. |
| `action_mask` | 2305 | Which moves are legal right now. See [Action Parsers](action-parsers.md). |
| `mask_planes` | 4 x 32 x 18 | The same mask, as one grid per card in your hand. |

```python
from royalegym import make_env

env = make_env()
obs, info = env.reset(seed=0)
print(obs["blue"]["spatial"].shape)
print(env.obs_builder.channel_names()[:5])
```

```
(20, 32, 18)
['own_ground_troops', 'own_air_troops', 'own_buildings', 'own_towers', 'own_hp']
```

`env.obs_builder.channel_names()` lists all 20 layers.

### It only sees what a player could see

Like you on your phone, the bot sees its own side at the bottom of the arena, its own hand and
its own elixir. It doesn't see the opponent's hand or elixir bar. Instead, the builder counts
what the opponent has played and how much elixir they have probably had since, the way a good
player keeps count in their head.

You can lift that limit for experiments with `Reveal`, which adds the hidden values as extra
inputs. A bot trained that way expects them, so it can't play a fair battle afterwards.

## Options

| Option | What it adds |
|---|---|
| `SpatialObsBuilder(card_identity=True)` | Which card each unit came from, as its own `card_ids` grid, and the last card the opponent played. |
| `SpatialObsBuilder(evolutions=True)` | Which of your cards are evolved, and where evolved units stand. |
| `SpatialObsBuilder(heroes=True)` | Where each side's heroes stand, and which of the opponent's heroes still have their ability to use. |
| `SpatialObsBuilder(spell_identity=True)` | Which spell is where, as its own `spell_ids` grid: where each spell is now and where it lands. Needs `card_identity=True` and `spell_aim_after_ticks`. |
| `SpatialObsBuilder(unit_status=True)` | Shields, units under a Rage or slowed by cold, and units already locked on a crown tower. |
| `SpatialObsBuilder(reveal=Reveal(...))` | Hidden information, such as the opponent's real elixir. For experiments only. |

`Reveal` comes from `royalegym`. Turn on what you want the bot to see:
`Reveal(enemy_elixir=True)`, `Reveal(enemy_hand=True)`, and also `enemy_next_card`,
`enemy_deck` and `enemy_spell_aim`. For example,
`obs_builder = SpatialObsBuilder(reveal=Reveal(enemy_elixir=True))`.

`EntityListObsBuilder()` gives a list of units, one row each, instead of a grid. RoyaleLearn
can't train on it.

!!! note "Which builders RoyaleLearn can train"
    RoyaleLearn's trainer reads `SpatialObsBuilder`'s four keys, with any of the options in the
    table. A builder of your own works with the environment and with a trainer you write or
    bring yourself.

!!! tip "In your quickstart.py"
    The builder is the `obs_builder = ...` line in `build_env`. Changing what the bot sees
    needs a new bot: change `save_dir` to a new name too.

## How They Work

Every observation builder has these methods:

```python
# Called once when the environment is made. The engine and the action parser are ready.
def bind(self, engine, action_parser): ...

# Called at the start of every battle.
def reset(self, state): ...

# What the observations look like, as a gymnasium Dict space.
def observation_space(self): ...

# What `team` sees in `state`. Always include the action mask.
def build(self, state, team, action_mask): ...
```

`bind` and `reset` already do the setup every builder needs, so you only write
`observation_space` and `build`.

## Creating Your Own

Here is a builder that sees almost nothing: the six towers' health and its own elixir.

```python
import numpy as np
from gymnasium import spaces
from royalegym import ObsBuilder


class TowersAndElixirObs(ObsBuilder):
    """Sees only the six towers' health and its own elixir."""

    def observation_space(self):
        return spaces.Dict({
            "vector": spaces.Box(0.0, 1.0, shape=(7,), dtype=np.float32),
            "action_mask": self.mask_space,  # set up for you in bind()
        })

    def build(self, state, team, action_mask):
        me, foe = state.players[team], state.players[1 - team]
        towers = [hp / top for hp, top in zip(me.tower_hp, me.tower_max_hp)]
        towers += [hp / top for hp, top in zip(foe.tower_hp, foe.tower_max_hp)]
        elixir = me.elixir_milli / 10_000  # elixir is counted in thousandths
        return {
            "vector": np.array(towers + [elixir], dtype=np.float32),
            "action_mask": action_mask,
        }
```

```python
from royalegym import make_env

env = make_env(obs_builder=TowersAndElixirObs())
obs, info = env.reset(seed=0)
print(obs["blue"]["vector"])
```

```
[1.  1.  1.  1.  1.  1.  0.6]
```

Each seat sees its own towers first, then the enemy's. The battle starts with every tower at
full health and 6 elixir.

## What a Builder Can Read

`state` is the whole battle at one moment. The parts you'll use most:

| Field | What it is |
|---|---|
| `state.tick` | Time since the battle started, in ticks of 50 ms. |
| `state.players[team]` | One side: `elixir_milli` (thousandths of an elixir), `hand` and `next_card` (card ids), `crowns`, `tower_hp` and `tower_max_hp` (king, left, right). |
| `state.entities` | Every unit, building and tower on the board: `team`, `kind`, `card_id`, `x`, `y`, `hp`, `max_hp`, `flying`. |
| `state.game_over`, `state.winner` | Whether the battle is over, and who won. |

Positions are in the engine's own units, 18,000 to a tile, with Blue at the bottom. To see a
position from one seat's side, use `to_own(arena, team, x, y)`.
Card ids are only positions in the engine's card list: `engine.cards()[card_id].name` gives the
name. A unit's `card_id` is the card whose play put it on the board: the Skeletons a Tombstone
makes report the Tombstone, and an evolved or hero unit reports its base card.
