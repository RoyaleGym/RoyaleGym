# Observations and Actions

Every half second of game time, each side's bot sees the board and picks one move: wait, or play
one card from its hand on one tile. This page shows what it sees, how the moves are numbered, and
the mask that says which moves are allowed.

## What the bot sees

An observation is a dictionary of numpy arrays, one per seat.

```python
from royalegym import make_env

env = make_env()
obs, info = env.reset(seed=0)
for name, part in obs["blue"].items():
    print(f"{name:12} shape {part.shape}")
```

```
spatial      shape (20, 32, 18)
vector       shape (1675,)
action_mask  shape (2305,)
mask_planes  shape (4, 32, 18)
```

- `spatial` is the board as a picture: 20 layers over the 32 by 18 tiles. Feed it to a
  convolutional network.
- `vector` is everything else: your hand, elixir, tower health and the clock.
- `action_mask` has one flag per move, 1 if it is allowed now.
- `mask_planes` is the same mask laid out like the board, one layer per hand slot.

Each side sees the board from its own end, with its king at the bottom. The width of `vector`
depends on how many cards the engine knows, so read it from the space. Don't type it in.

## The 2305 moves

The arena has 18 by 32 tiles, which is 576. You hold four cards, so there are 4 x 576 = 2304 ways
to play one. Waiting makes 2305.

```python
parser = env.action_parser
print("moves:", parser.n_actions)
print("wait is move", parser.noop())
print("hand slot 0 on tile (9, 5) is move", parser.encode(0, 9, 5))
print("move 2304 is (slot, x, y)", parser.decode(2304))
```

```
moves: 2305
wait is move 0
hand slot 0 on tile (9, 5) is move 100
move 2304 is (slot, x, y) (3, 17, 31)
```

Move `1 + slot * 576 + y * 18 + x` plays that hand slot on tile `(x, y)`. Check it: 1 + 0 + 90 +
9 = 100.

## The mask

The mask rules out every move you can't make right now: cards you can't afford and tiles that card
can't go on. Your bot only picks among allowed moves, so it never wastes time learning the
rules.

```python
for step in range(1, 11):
    wait = {a: parser.noop() for a in env.agents}
    obs, rewards, terminated, truncated, info = env.step(wait)
    if step in (1, 9, 10):
        print(f"step {step:2}: {int(obs['blue']['action_mask'].sum())} legal moves")
```

```
step  1: 1 legal moves
step  9: 1640 legal moves
step 10: 1640 legal moves
```

A battle opens with a short wait when no card can be played, so at first only the wait is
allowed. Then play opens. The count changes with your elixir, your hand and the board.

If a move gets past the mask anyway, the engine refuses it and the step becomes a wait.

## Ability buttons

A hero or a champion has an ability button. A deck with one gets three extra moves, one per
button slot. Move `2305 + k` presses button `k`.

```python
env = make_env(heroes=["Musketeer"])
obs, info = env.reset(seed=0)
print("moves:", env.action_parser.n_actions)
print("buttons ready:", obs["blue"]["ability_ready"])
```

```
moves: 2308
buttons ready: [0 0 0]
```

`ability_ready` has one flag per button. No button is ready at the start. The mask allows a press
only when the button is ready and you have the elixir for it.

These outputs were run on 2026-10-01 on engine build `a581356680589e2c`.

Next: [Custom Decks](custom-decks.md), to choose the cards.
