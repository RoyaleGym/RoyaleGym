# Quick Start

One file trains a bot, plays a battle with it, and saves that battle so you can watch it. It
runs on an ordinary computer, no graphics card needed.

Save this as `quickstart.py` (it is also in the repo as `examples/quickstart.py`) and run
`python quickstart.py`:

```py
--8<-- "../../examples/quickstart.py"
```

It prints one line each time the bot learns: how many steps it has played, how many battles
finished, crowns per battle, and how long it took. At the end it tells you who won the battle
it saved, and how to watch it.

On a CPU this takes a while: leave it running. Training longer makes a better bot, and running the
file again carries on from where it stopped.

## What each part does

1. **`build_env`** builds one battle. `reward=TowerHPReward()` pays your bot for damage to the
   enemy's towers and charges it for damage to its own. This is the line to change for a
   different bot.
2. **`Learner(...)`** is the trainer. It plays `n_envs` battles at once against a bot that
   picks random allowed moves (`opponent="random"`), and learns from them.
3. **`learn(total_steps=...)`** trains. A step is one decision, half a second of game time. More
   steps take longer and make a better bot. Run the file again and it carries on from where it
   stopped.
4. **`save`** and **`load_policy`** keep the bot and load it back.
5. **`play_battle`** plays one battle, your bot as Blue against the random bot, and saves it.

## Watch it

```
royaleviser my_bot_battle.msgpack
```

A window opens and plays the battle. Space pauses, the arrow keys step. More in
[Watch Your Bot](watch-your-bot.md).

## Change one thing

The quickest way to a different bot is to change what it is paid for. Swap `TowerHPReward()`
for another reward, or write your own: see [Rewards](guides/rewards.md).

## What next

- [Rewards](guides/rewards.md): what your bot wants.
- [Observations and Actions](guides/observations-and-actions.md): what it sees and the moves it can make.
- [Custom Decks](guides/custom-decks.md): play with your own cards.
