# Action Parsers

An action parser turns the bot's choice into something the game understands: which card to
play, and where. It also builds the **action mask**, the list of moves that are legal right now.

## The Default: 2305 Moves

`TileActionParser` gives the bot 2305 moves, numbered from 0:

- **0** means wait.
- **1 to 2304** mean "play the card in hand slot `slot` on tile (`x`, `y`)", where
  `action = 1 + slot * 576 + y * 18 + x`.

The arena is 18 tiles wide and 32 tiles long. Tiles are counted from the bot's own side: `x`
goes from 0 to 17, left to right, and `y` from 0 at the bot's own back edge to 31 at the
enemy's. Both seats use the same numbers, so one bot can play either side.

You don't have to do the arithmetic yourself:

```python
from royalegym import make_env

env = make_env()
parser = env.action_parser

action = parser.encode(2, 9, 10)  # hand slot 2, tile x=9, y=10
print(action)
print(parser.decode(action))
```

```
1342
(2, 9, 10)
```

## The Action Mask

Most moves are not legal most of the time: the card costs more elixir than the bot has, the
tile is on the enemy's side of the river, it's water, or the battle has only just begun. Every
observation carries `action_mask`, one number per move: 1 if the move is legal right now, 0 if
not. A move the mask forbids is played as "wait".

`mask_planes` is the same mask laid out as a grid, one 32 x 18 layer per hand slot. Here is how
many tiles each card in Blue's hand could be played on, five seconds into a battle:

```python
from royalegym import make_env

env = make_env()
obs, info = env.reset(seed=0)
for _ in range(10):  # nobody can play a card in the first few seconds
    obs, rewards, terminated, truncated, info = env.step({"blue": 0, "red": 0})

names = {card.card_id: card.name for card in env.engine.cards()}
hand = env.battle_state.players[0].hand
planes = obs["blue"]["mask_planes"]  # [hand slot, y, x]
for slot in range(4):
    print(names[hand[slot]], int(planes[slot].sum()))
```

```text
Cannon 240
Zap 576
Fireball 576
Archer 247
```

Spells can go anywhere, so they have all 576 tiles. Troops and buildings fit only on your own
half. RoyaleLearn always picks from the legal moves, so you only need the mask if you write
your own bot or trainer.

## Ability Buttons

Heroes and champions have an ability button. `TileActionParser(ability_buttons=True)` adds one
move per button slot after the 2305 tile moves, so there are 2308. A button move is legal only
while that ability can be used. `make_env(heroes=[...])` turns the buttons on for you.

## Other Options

| Option | What it does |
|---|---|
| `TileActionParser(ability_buttons=True)` | Adds the hero and champion ability buttons. |
| `TileActionParser(buildings="taps_where_the_building_stays")` | Only offers building taps where the building lands on the tile you chose. By default the game moves a building that doesn't fit to the nearest place it does. Not for RoyaleLearn: its start-up check refuses it when the deck has a building. |
| `TileActionParser(ui_buttons=True)` | Makes the mask match the game's screen. While a hero or champion is on the board, its ability button sits over a back corner of its owner's side, and a card dropped under the button isn't placed. The mask then refuses those tiles for every card. One button covers the corner at x 0-1, and a second the corner at x 15-17 (`action.UI_BUTTON_TILES` lists the tiles). Off by default. |
| `HalfTileActionParser()` | Half-tile precision: 9217 moves instead of 2305. |

!!! note "Which parsers RoyaleLearn can train"
    RoyaleLearn's trainer is built for `TileActionParser`'s grid of moves, with its default
    buildings setting, with or without ability buttons. Any other action parser, including
    your own, works with the environment and with a trainer you write or bring yourself. With
    another parser in `build_env`, training stops with `spatial planes are (32, 18) tiles,
    tiles says (64, 36)` or `the observation space has no 'mask_planes' key`: that's this limit.

!!! tip "In your quickstart.py"
    The parser is the `action_parser = ...` line in `build_env`. Changing it changes the bot's
    moves, so it needs a new bot: change `save_dir` to a new name too.

## How They Work

Every action parser has these methods:

```python
# Called once when the environment is made. Read what you need from the engine here.
def bind(self, engine): ...

# How many moves there are, as a gymnasium Discrete space.
@property
def space(self): ...

# One number per move: 1 if it is legal for `team` in `state`, 0 if not.
def action_mask(self, state, team): ...

# The card play for one move, or None to wait.
def parse(self, action, state, team): ...
```

## Creating Your Own

Here is a parser with only 13 moves: wait, or play a card in one of three spots. It borrows
`TileActionParser` to work out which of its moves are legal.

```python
import numpy as np
from gymnasium import spaces
from royalegym import ActionParser, TileActionParser


class FewSpotsActionParser(ActionParser):
    """Plays a card in one of three spots: in front of either bridge, or behind the king tower."""

    SPOTS = [(3, 13), (14, 13), (8, 0)]  # (tile x, tile y), seen from the bot's own side

    def bind(self, engine):
        super().bind(engine)
        self.tiles = TileActionParser()  # it works out what is legal for us
        self.tiles.bind(engine)

    @property
    def space(self):
        return spaces.Discrete(1 + 4 * len(self.SPOTS))  # wait, or a card in a spot

    def _tile_action(self, action):
        slot, spot = divmod(action - 1, len(self.SPOTS))
        x, y = self.SPOTS[spot]
        return self.tiles.encode(slot, x, y)

    def action_mask(self, state, team):
        tile_mask = self.tiles.action_mask(state, team)
        mask = np.zeros(self.space.n, dtype=np.int8)
        mask[0] = 1  # waiting is always allowed
        for action in range(1, self.space.n):
            mask[action] = tile_mask[self._tile_action(action)]
        return mask

    def parse(self, action, state, team):
        if action == 0:
            return None  # wait
        return self.tiles.parse(self._tile_action(action), state, team)
```

Pass it to the environment like any other piece:

```python
from royalegym import make_env

env = make_env(action_parser=FewSpotsActionParser())
obs, info = env.reset(seed=0)
print(env.action_space("blue"))
```

```
Discrete(13)
```
