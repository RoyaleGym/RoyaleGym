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
| `SpatialObsBuilder(evolution_progress=True)` | How far each card in your hand has got toward its next evolved play. Needs `evolutions=True`. |
| `SpatialObsBuilder(heroes=True)` | Where each side's heroes stand, and which of the opponent's heroes still have their ability to use. Champions don't count as heroes. |
| `SpatialObsBuilder(spell_identity=True)` | Which card made each live spell object, as its own `spell_ids` grid: a cast spell, or an effect a troop or building left (an evolution's fireworks or bombs, a death bomb). It shows where each one is now and where it lands. Needs `card_identity=True` and `spell_aim_after_ticks`. |
| `SpatialObsBuilder(unit_status=True)` | Shields, units under a Rage or slowed by cold, and units already locked on a crown tower. |
| `SpatialObsBuilder(card_status=True)` | Each card's evolution and hero status: all of yours from the start, and the opponent's as you see them played. |
| `SpatialObsBuilder(unit_actions=True)` | What units are doing: charging, winding up or using an ability, Clone copies, and where a tunneller will come up. |
| `SpatialObsBuilder(unit_identity=True)` | What each unit is, as its own `unit_ids` grid: a skeleton reads Skeleton whether the Witch or a Tombstone made it. `card_ids` says which card it came from. |
| `SpatialObsBuilder(effect_identity=True)` | With `spell_identity`, keeps cast spells apart from the effects troops and buildings leave: `spell_ids` grows from 4 layers to 8, with spell cards in the first four and those effects (an evolution's bombs or fireworks, a death bomb) in the last four. |
| `SpatialObsBuilder(enemy_queue=True)` | The opponent's next card and the three after it, in order, worked out from its plays the way a player watching them would. A place it can't know yet reads "not deduced". |
| `SpatialObsBuilder(button_index=True)` | Each of your ability buttons in the order the actions press them: which card it belongs to, and whether it's ready, used or cooling down. |
| `SpatialObsBuilder(elixir_income=True)` | The opponent's counted elixir also takes in what the board pays: an Elixir Collector's elixir, and the elixir it pays when it dies, and what an Elixir Golem's pieces pay when they die. Needs royalegym 0.1.26 or newer: `pip install --upgrade "royalegym[all]"`. |
| `SpatialObsBuilder(unit_velocity=True)` | How far each unit has moved since the bot last looked, in tiles per 5 ticks, as eight more layers: each side's units moving right, left, forward and back. A unit that has just appeared shows no movement yet, and towers show none. With RoyaleLearn, list the eight layers (the last eight of `channel_names()`) in `rollout.codec_float16_planes` so their fractions are kept. Needs royalegym 0.1.29 or newer. |
| `SpatialObsBuilder(own_button_pending=True)` | Which of your ability buttons has a press waiting to happen, when the environment delays commands (`command_delay_ticks`). Needs an action parser with `ability_buttons=True`, and royalegym 0.1.29 or newer. |
| `SpatialObsBuilder(own_waiting_plays=True)` | Your own plays still waiting to happen under a command delay, soonest first, up to two: where each lands, how many ticks are left, and what it costs. Needs royalegym 0.1.29 or newer. |
| `SpatialObsBuilder(enemy_card_cycle=True)` | For each card, how many plays until it's back in the opponent's hand, where their plays so far show it, and how many plays since they last played it. Needs royalegym 0.1.29 or newer. |
| `SpatialObsBuilder(reveal=Reveal(...))` | Hidden information, such as the opponent's real elixir. For experiments only. |

`Reveal` comes from `royalegym`. Turn on what you want the bot to see:
`Reveal(enemy_elixir=True)`, `Reveal(enemy_hand=True)`, and also `enemy_next_card`,
`enemy_deck` and `enemy_spell_aim`. For example,
`obs_builder = SpatialObsBuilder(reveal=Reveal(enemy_elixir=True))`.

`EntityListObsBuilder()` gives a list of units, one row each, instead of a grid. RoyaleLearn
can't train on it. It takes the table's options that add to the vector (`evolutions`,
`evolution_progress`, `card_status`, `button_index`, `enemy_queue`, `elixir_income`,
`own_button_pending`, `own_waiting_plays` and `enemy_card_cycle`), and six of its own that
add columns to each unit's row:

| Option | What each row gains |
|---|---|
| `EntityListObsBuilder(unit_motion=True)` | How far the unit has moved since the bot last looked, as `unit_velocity` measures it. |
| `EntityListObsBuilder(unit_stats=True)` | Its full hitpoints, its shield, and where its target stands. |
| `EntityListObsBuilder(unit_status=True)` | One column per status flag: under ground, invisible, hidden, evolved, a hero, a Clone copy, an ability winding up or running, charged, and grounded. |
| `EntityListObsBuilder(unit_actions=True)` | Its charge, its ability timer, which part of its attack it's in, which way it faces, and where a tunneller will come up. |
| `EntityListObsBuilder(unit_buffs=True)` | Up to six buffs on the unit, the one with the most time left first, and how long each has left (10 seconds reads 1). Which buff each one is goes in a new `entity_buffs` key, 0 for an empty slot. `buff_names=[...]` sets which id each buff gets. |
| `EntityListObsBuilder(unit_extras=True)` | Eight more columns: a spawner's clock and how many units its wave still has to put down; whether the unit is standing before a dash or dashing, and the ticks until it goes; how much its buffs speed up or slow its walking and its attacks (0 is normal, -1 frozen); and whether it is leaping the river. |

`unit_motion`, `unit_stats`, `unit_status` and `unit_actions` need royalegym 0.1.29 or newer, and
`unit_buffs` and `unit_extras` need 0.1.30 or newer.

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
