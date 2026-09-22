# What the bot sees and does

<p align="center">
  <img alt="Action space" src="https://img.shields.io/badge/action%20space-2305%20moves-0b7285?style=flat-square">
  <img alt="Legality" src="https://img.shields.io/badge/illegal%20moves-masked%20out-2ea043?style=flat-square">
  <img alt="Perspective" src="https://img.shields.io/badge/perspective-own%20king%20at%20the%20bottom-555?style=flat-square">
  <img alt="Hidden information" src="https://img.shields.io/badge/opponent's%20hand-hidden%20by%20default-8957e5?style=flat-square">
  <img alt="Blocks" src="https://img.shields.io/badge/code%20blocks%20on%20this%20page-all%20run-2ea043?style=flat-square">
</p>

In plain words, before any numbers.

Your bot sees the board and its own hand. Every half second it chooses one of two
things: wait, or play one card from its hand on one tile. That is the whole contract.

It is told which of those moves are legal before it picks, so it never has to learn
that you cannot drop a Knight in the river.

The rest of this page is the exact detail, all of it printed by programs that were run.

!!! warning "Your widths will not match these widths"
    Several numbers below depend on how many cards are in your card catalogue, and that
    is decided when you build the engine. The machine that produced this page had 95
    cards. A clean public checkout builds the 2018 card table and will report something
    else. Nothing in RoyaleGym or RoyaleLearn types these widths in. They are read from
    the environment at startup, and you should do the same. Never treat a width on this
    page as a constant of the project.

## What one observation actually is

It is a dictionary of four numpy arrays. This program prints the real shapes and types.

```python
import numpy as np
from royalegym import ClashParallelEnv, DefaultStateMutator, RustEngine

engine = RustEngine()
by_name = {c.name: c.card_id for c in engine.cards()}
deck = [by_name[n] for n in ("Knight", "Archer", "Giant", "Minions",
                             "Fireball", "Cannon", "Zap", "Musketeer")]

env = ClashParallelEnv(engine=engine,
                       state_mutator=DefaultStateMutator(decks=[deck, deck]))
obs, info = env.reset(seed=0)

for name, part in obs["blue"].items():
    print(f"{name:12} shape {str(tuple(part.shape)):14} dtype {part.dtype}")

print("cards in this catalogue:", len(engine.cards()))
print("legal actions right now:", int(obs["blue"]["action_mask"].sum()), "of", obs["blue"]["action_mask"].size)
```

```
spatial      shape (20, 32, 18)   dtype float32
vector       shape (1177,)        dtype float32
action_mask  shape (2305,)        dtype int8
mask_planes  shape (4, 32, 18)    dtype int8
cards in this catalogue: 95
legal actions right now: 1605 of 2305
```

The deck is named card by card on purpose. With no deck named, each side is dealt eight
random cards out of your catalogue, so the same seed gives a different battle on a
different machine and none of these numbers reproduce.

<div class="grid cards" markdown>

- __`spatial`, 20 planes of 32 by 18__

    ---

    The board as a picture. One value per tile per plane. Feed it to a convolutional
    network.

- __`vector`, one flat row__

    ---

    Everything that is not on the board: your hand, your cycle, elixir, tower health,
    the clock.

- __`action_mask`, 2305 flags__

    ---

    One flag per possible move, 1 if it is legal right now.

- __`mask_planes`, 4 by 32 by 18__

    ---

    The same mask with the wait action removed, shaped like the board so a
    convolutional network can read it.

</div>

### The 20 planes

```python
from royalegym import RustEngine, SpatialObsBuilder, TileActionParser

engine = RustEngine()
parser = TileActionParser()
parser.bind(engine)
builder = SpatialObsBuilder()
builder.bind(engine, parser)

for i, (name, static) in enumerate(builder.spatial_layout()):
    print(f"{i:2}  {name:20} {'same every step' if static else ''}")
```

```
 0  own_ground_troops
 1  own_air_troops
 2  own_buildings
 3  own_towers
 4  own_hp
 5  enemy_ground_troops
 6  enemy_air_troops
 7  enemy_buildings
 8  enemy_towers
 9  enemy_hp
10  own_deploying
11  enemy_deploying
12  water                same every step
13  no_deploy            same every step
14  enemy_troop_zone
15  own_spells
16  enemy_spells
17  own_spell_aim
18  own_stunned
19  enemy_stunned
```

"Own" and "enemy" are from the point of view of whoever is choosing. Red's board is
turned 180 degrees before it is written down, so both seats always see their own king
at the bottom. That is what lets one bot play both sides of a match.

`own_deploying` is units that have been played but are still in their deploy animation
and cannot act yet. `enemy_troop_zone` is where the opponent could put a troop right
now, which is about their options, not yours.

The flat `vector` holds your hand, the cost of each card in it, whether you can afford
it, where you are in your eight card cycle, tower health on both sides, crowns, the
clock and the elixir rate. Every slot is listed in
[observation-spec.md](https://github.com/RoyaleGym/RoyaleGym/blob/main/docs/observation-spec.md).

## The action space, and the arithmetic

There are 2305 possible moves. Here is where that comes from.

The arena is 18 tiles across and 32 tiles deep, so there are 18 times 32, which is 576
tiles. You have four cards in hand. Four cards on 576 tiles is 2304 moves. Add one for
doing nothing and you get 2305.

Index 0 is the wait action. Index `1 + slot * 576 + y * 18 + x` plays hand slot `slot`
on tile `(x, y)`, in your own frame.

```python
import numpy as np
from royalegym import (ClashParallelEnv, DefaultStateMutator, RandomLegalOpponent,
                       RustEngine, TileActionParser)

engine = RustEngine()
by_name = {c.name: c.card_id for c in engine.cards()}
deck = [by_name[n] for n in ("Knight", "Archer", "Giant", "Minions",
                             "Fireball", "Cannon", "Zap", "Musketeer")]

env = ClashParallelEnv(engine=engine,
                       state_mutator=DefaultStateMutator(decks=[deck, deck]))
obs, info = env.reset(seed=0)
print("action space:", env.action_space("blue"))

parser = TileActionParser()
parser.bind(engine)
print("no-op is index", parser.noop())
print("hand slot 0 on tile (9, 5) is index", parser.encode(0, 9, 5))
print("index 2304 decodes to (slot, x, y) =", parser.decode(2304))

rng, policy = np.random.default_rng(0), RandomLegalOpponent(noop_prob=0.7)
for step in range(1, 61):
    actions = {a: policy.act(obs[a], obs[a]["action_mask"], rng) for a in env.agents}
    obs, reward, terminated, truncated, info = env.step(actions)
    if step in (1, 20, 60):
        mask = obs["blue"]["action_mask"]
        print(f"step {step:3}  legal actions {int(mask.sum()):5}"
              f"  elixir {env.battle_state.players[0].elixir_milli / 1000:.2f}")
```

```
action space: Discrete(2305)
no-op is index 0
hand slot 0 on tile (9, 5) is index 100
index 2304 decodes to (slot, x, y) = (3, 17, 31)
```

Check it by hand, because it is the kind of thing you want to be sure of. Slot 0, tile
(9, 5): `1 + 0 * 576 + 5 * 18 + 9` is `1 + 0 + 90 + 9`, which is 100. And 2304 is the
very last index, so it should be the last hand slot on the far corner tile: slot 3,
tile (17, 31). Both agree.

??? note "Why one big list of 2305 and not three separate choices"
    You could ask the bot for a card, then an x, then a y. Three small choices are
    easier to learn, but they cannot be masked properly. Whether a move is legal depends
    on the card and the tile together. A Fireball can go anywhere, a Giant cannot go in
    the river, and a building cannot go in the opponent's half. Masking each choice on
    its own can only rule out a whole card or a whole column, so the bot would still be
    allowed to ask for a Knight on the enemy king.

    One joint list of 2305 can say exactly which pairs are allowed, and 2305 is a small
    number for a network. `HalfTileActionParser` ships beside it at `Discrete(9217)`, at
    half tile resolution, so the trade can be measured rather than argued about.

Timing is not a separate choice either. A decision happens every 500 ms of game time,
and waiting is just picking index 0.

## The legality mask, and why it saves you so much

The second half of that program printed this:

```
step   1  legal actions  1605  elixir 5.18
step  20  legal actions     1  elixir 2.57
step  60  legal actions   459  elixir 3.71
```

Read the middle line. At step 20 exactly one action was legal, and that one is the wait
action. The player had 2.57 elixir and the four cards in hand were Giant, Cannon, Minions
and Archer, the cheapest of them 3. The deck does hold a 2 cost card, Zap, and it was not in
hand at that moment, which is the whole point: what you can afford depends on the four cards
you happen to be holding, not on the eight you chose. Without a mask your bot would spend
thousands of steps discovering that by being refused.

The count moves with your elixir, with your hand, with the towers still standing and
with the buildings already on the board. The mask covers elixir, which half of the
arena you may play in, water, the river, the footprint of buildings already down, and
the rectangle around each enemy crown tower that is still alive.

For comparison, RoyaleGym's
[architecture.md](https://github.com/RoyaleGym/RoyaleGym/blob/main/docs/architecture.md)
reports 691 legal actions of 2305 on the first step with default random decks on the
Rust engine, and 1235 on `MockEngine`. This page got 1605 with the named deck above.
Different deck, different card table, different count. The number is not a property of
the game.

Two details worth trusting the project for:

- The mask is worked out by a separate piece of Python, not asked of the engine. The
  test suite then compares the two rulings exhaustively, for every card and every
  position. A wrong mask fails a test instead of quietly ruining a week of training.
- `action_masks()` returns it in the form sb3-contrib's MaskablePPO expects, so a
  standard masked policy works with no glue.

If you do get a move past the mask, the engine refuses it, the step becomes a wait, and
the reason appears in `info["deploy_status"]`.

## Hidden information

By default your bot sees what a person watching the match could write down. Nothing
more.

- The opponent's hand is not in the observation.
- Neither is their deck, beyond the cards you have watched them play.
- Their elixir **is** there, but as a count, not a peek. The observation pays for every
  card you saw them play and adds the regeneration everyone knows about, exactly the way
  a good player counts in their head. It matches the bar the engine keeps, and a test
  plays battles out and checks it every step from both seats. `MatchMemory.exact` says
  so when it cannot be, so a run can assert on it.

You can turn any of that off, for a curriculum or for debugging. It is called a reveal,
and the important part is how it is done.

```python
from royalegym import (ClashParallelEnv, DefaultStateMutator, RustEngine,
                       SpatialObsBuilder)
from royalegym.obs import Reveal

engine = RustEngine()
by_name = {c.name: c.card_id for c in engine.cards()}
deck = [by_name[n] for n in ("Knight", "Archer", "Giant", "Minions",
                             "Fireball", "Cannon", "Zap", "Musketeer")]


def widths(builder):
    env = ClashParallelEnv(engine=engine, obs_builder=builder,
                           state_mutator=DefaultStateMutator(decks=[deck, deck]))
    obs, info = env.reset(seed=0)
    return obs["blue"]["vector"].shape[0], obs["blue"]["spatial"].shape[0]


fair = widths(SpatialObsBuilder())
cheat = widths(SpatialObsBuilder(reveal=Reveal(enemy_hand=True)))
print("fair             vector", fair[0], " spatial channels", fair[1])
print("enemy_hand shown vector", cheat[0], " spatial channels", cheat[1])

env = ClashParallelEnv(engine=engine,
                       obs_builder=SpatialObsBuilder(reveal=Reveal(enemy_hand=True)),
                       state_mutator=DefaultStateMutator(decks=[deck, deck]))
print("recorded in the config:", env.config()["obs_builder"]["params"]["reveal"])
```

```
fair             vector 1177  spatial channels 20
enemy_hand shown vector 1561  spatial channels 20
recorded in the config: {'enemy_elixir': False, 'enemy_hand': True, 'enemy_next_card': False, 'enemy_deck': False, 'enemy_spell_aim': False}
```

The vector got wider. It did not gain four slots of zeros that quietly fill in when you
flip a switch. A fair observation and a cheating one are not even the same size, so you
cannot train with one and evaluate with the other by accident. And
`ClashParallelEnv.config()` writes the reveal down, so a checkpoint always says whether
the bot was allowed to look.

The five reveals are `enemy_elixir`, `enemy_hand`, `enemy_next_card`, `enemy_deck` and
`enemy_spell_aim`. `enemy_elixir` is the one exception to the width rule: the slot
already exists and holds the counted value, so the reveal swaps the source instead of
adding a slot.

## Read next

- [Writing a reward function](rewards.md), which is the piece you are expected to write.
- [The environments](pieces/environments.md), for the other swappable pieces.
- [observation-spec.md](https://github.com/RoyaleGym/RoyaleGym/blob/main/docs/observation-spec.md)
  for every channel and every vector slot, with its range and whether it is fair.
- [architecture.md](https://github.com/RoyaleGym/RoyaleGym/blob/main/docs/architecture.md)
  for the action space, the mask and the layer boundaries.
