# Cloning a Bot

A clone is a bot that learned by copying moves. You give it games that someone else played, and
its network learns to pick the moves they picked. That's how to start a bot that plays like a
person: a bot that learns only by trial and error, starting from nothing, doesn't get far.

You can clone two kinds of player:

- **People**, from thousands of real games. Start here.
- **A bot**: one of the scripted bots that come with RoyaleGym, or a bot you trained.

Either way, you can then train your clone further by trial and error, starting from what it
copied instead of from nothing.

This page builds on the [Quick Start](../quickstart.md): do that first. Everything here comes with
`royalegym[all]` from [Install](../install.md), version 0.1.6 or newer
(`pip install --upgrade "royalegym[all]"` updates an older one). The cloning is part of
[RoyaleImitate](../resources/royaleimitate.md).

## Clone Human Players

### The Games You Copy

[IL_Replay](https://huggingface.co/datasets/VanguardX101/IL_Replay) is a public set of about
252,000 real matches on Hugging Face, most of them ranked (Path of Legends) games. For each match
it has both players' decks and every card they played: which card, on which tile, and when. It
doesn't say who the players were or how good they were.

RoyaleImitate replays each match in RoyaleGym's engine, card play by card play. Every half second
of the match, for both players, your clone gets one example to learn from: what that player could
see at that moment, and what they did. Either they played a card on a tile, or they waited.

### 1. Make the Clone

In your terminal, in the `royale` folder with `(venv)` showing, make a file called
`clone_humans.py` the same way you made `quickstart.py` (`notepad clone_humans.py` on Windows,
`touch clone_humans.py` and `open -e clone_humans.py` on a Mac, `nano clone_humans.py` on Linux),
and paste this into it:

```python title="clone_humans.py"
from royalegym import make_env
from royaleimitate import clone, from_replays
from royalelearn import Learner


def build_env():
    return make_env()


if __name__ == "__main__":
    learner = Learner(build_env, save_dir="runs/my_clone")
    demos = from_replays(learner, "runs/human-demos", matches=1000)
    clone(learner, demos, "runs/human_clone")
    print("Your clone is in runs/human_clone")
```

Run it with `python clone_humans.py`. It prints a line every 100 matches, then a summary:

```text
100 matches written, 45 skipped
200 matches written, 74 skipped
...
1000 matches written, 360 skipped
1000 matches written (830608 rows); skipped: 195 play refused by the engine, 154 card not in the catalogue, 8 card not in hand, 3 not an eight-card deck
Your clone is in runs/human_clone
```

On our computer, replaying the matches took about 11 minutes (that part runs on the processor),
and cloning them on its RTX 4070 Ti about an hour and a half more, while another program shared the card.
Cloning prints nothing while it works, so let it run.

Here is what each line does:

- `Learner(build_env, ...)` gives the clone its network and what it sees. It doesn't train here,
  so nothing goes in its `save_dir`. Use the same `build_env` you'll train with later, so the
  clone sees what your bot will see.
- `from_replays` downloads the games as it needs them, about 5,000 matches (15 MB) at a time, into
  the Hugging Face cache, so a second run doesn't download them again. It replays 1,000 matches and
  writes every example from them into `runs/human-demos`, about 140 MB in all.
- It skips a match it can't replay, most often because RoyaleGym's engine doesn't have one of its
  cards, or refuses one of its plays. The summary says how many it skipped, and why. With
  `make_env()`, about 1 in 4 matches were skipped when we ran it. Skipped matches don't count
  towards the 1,000.
- `clone` trains the network to make the same moves as the players. It keeps about one match in
  twenty aside to check the copy, stops once the copy stops getting better, and saves the clone in
  `runs/human_clone`.

The matches keep their players' own decks, but your clone plays the deck `build_env` gives it.
With `make_env()`, that's the Quick Start's deck on both sides. For your own deck, change
`build_env` to `make_env(deck=[...])`, with eight names as in
[Use Your Own Deck](../quickstart.md#8-use-your-own-deck), before you make the clone. The other
files below import this `build_env`, so they use the same deck. Some simple decks to start with:

```python
hog_2_6 = ["HogRider", "Musketeer", "Cannon", "IceGolemite", "IceSpirits", "Skeletons", "Fireball", "Log"]
balloon = ["Balloon", "Miner", "IceGolemite", "Musketeer", "Skeletons", "BombTower", "Snowball", "BarbLog"]
royal_giant = ["RoyalGiant", "Fisherman", "Hunter", "Ghost", "ElectroSpirit", "Skeletons", "Fireball", "BarbLog"]
```

`BarbLog` is the Barbarian Barrel, `Snowball` the Giant Snowball and `Ghost` the Royal Ghost; the
[names that differ](../cheatsheets/game-values.md#names-that-differ-from-the-game) are all listed.

`matches=1000` is how many matches it copies. More matches take longer to replay and to clone,
and take more disk space.

If it stops before it prints `Your clone is in runs/human_clone`, for example because you pressed
Ctrl+C or the computer restarted, it can't carry on. Delete the `runs/human-demos` folder, and
`runs/human_clone` if it's there, and run it again. Otherwise it stops with an error that says the
folder `already holds files`. To make a second clone next to the first, give it new folders
instead, such as `"runs/human-demos-2"` and `"runs/human_clone_2"`.

### 2. Watch Your Clone

The clone loads like any bot you trained. Make `watch_clone.py` next to `clone_humans.py`:

```python title="watch_clone.py"
import random

from royalegym import play_battle
from royalelearn import Learner

from clone_humans import build_env

bot = Learner.load_policy("runs/human_clone")
battle = play_battle(build_env(), blue=bot, red="push", seed=random.randrange(1_000_000),
                     save_to="clone_battle.msgpack")
print("Crowns:", battle.crowns[0], "-", battle.crowns[1])
```

```
python watch_clone.py
royaleviser clone_battle.msgpack
```

Your clone plays blue, at the bottom, against the push bot, which plays every card as far forward
as it can. Run it a few times: each battle goes differently.

Load your clone the default way, as here. People wait far more often than they play, so waiting
is a clone's single most likely move almost every half second: with `greedy=True` it never plays a
card.

### 3. Test Your Clone

To see how good it is, play it against every scripted bot. Make `test_clone.py` next to the
others. It takes a few minutes:

```python title="test_clone.py"
from royalegym import CallableOpponent, evaluate, ladder
from royalelearn import Learner

from clone_humans import build_env

bot = Learner.load_policy("runs/human_clone")
me = CallableOpponent(lambda obs, mask: bot(obs))
for name, opponent in ladder():
    print(evaluate(me, opponent, build_env, games=50, names=("my clone", name)).summary())
```

This is what we got. Yours will differ a little:

```text
my clone vs noop: 50-0-0 over 50 games. win rate 100.0% (92.9% to 100.0% at 95%) -- my clone is better. seat gap +0.0%, mean 2227 ticks.
my clone vs random: 47-3-0 over 50 games. win rate 94.0% (83.8% to 97.9% at 95%) -- my clone is better. seat gap +4.0%, mean 3469 ticks.
my clone vs first-affordable: 48-2-0 over 50 games. win rate 96.0% (86.5% to 98.9% at 95%) -- my clone is better. seat gap +0.0%, mean 3598 ticks.
my clone vs defend: 45-5-0 over 50 games. win rate 90.0% (78.6% to 95.7% at 95%) -- my clone is better. seat gap -12.0%, mean 3984 ticks.
my clone vs push: 44-6-0 over 50 games. win rate 88.0% (76.2% to 94.4% at 95%) -- my clone is better. seat gap +16.0%, mean 3630 ticks.
my clone vs patient: 40-10-0 over 50 games. win rate 80.0% (67.0% to 88.8% at 95%) -- my clone is better. seat gap -16.0%, mean 3074 ticks.
```

Each line says how many battles it won, lost and drew, and which bot is clearly better, or that
it's "too close to call". Ours beat every one of them, playing 23 to 32 cards a battle: it spends
its elixir, defends and attacks. The scripted bots are simple, though, so it's a start, not a strong
player. More about reading these lines is on
[Opponents](configuration-objects/opponents.md#who-is-better).

### 4. Train From Your Clone

Your clone can go on learning by trial and error, starting from what it copied. Make
`train_from_clone.py` next to the others:

```python title="train_from_clone.py"
from royaleimitate.artifacts import artifact_digest
from royalelearn import Learner

from clone_humans import build_env

if __name__ == "__main__":
    start = {"path": "runs/human_clone", "sha256": artifact_digest("runs/human_clone")}
    learner = Learner(build_env, save_dir="runs/from_clone", extensions={"warm_start": {"init": start}})
    learner.learn(total_steps=1_000_000_000)  # stop it with Ctrl+C whenever you like
```

It trains like the [Quick Start](../quickstart.md) and saves in `runs/from_clone`. `sha256` is a
fingerprint of the clone's folder, so the run starts from exactly that clone. Give both `Learner`s
the same settings: the clone's network has to fit the one you train.

Training a clone further is harder than it sounds. It can forget what it copied as it chases its
reward, and get worse instead of better. So test it: in `test_clone.py`, change
`runs/human_clone` to `runs/from_clone`, run it again, and compare its lines with your clone's. To
keep the new bot's moves close to the clone's while it learns, see the stay-close penalty in
[RoyaleImitate](../resources/royaleimitate.md).

### What a Clone Can't Do

- **It copies what people did, not why.** It learns the moves people made in situations like the
  one it's in. It can't plan, and it copies their mistakes along with everything else.
- **It plays like the players it copied.** IL_Replay doesn't say how good the players were, so
  your clone plays like a mix of them, not like the best of them.
- **It plays evolutions and heroes as plain cards.** The replay plays every card in its normal
  form and leaves out hero ability presses, so your clone never learns them.
- **The replay is RoyaleGym's battle, not the real one.** A match records the plays, not the
  board. Between plays, RoyaleGym's engine works out what happens, and it isn't exactly the real
  game (see [How accurate is the engine](../accuracy.md)). So now and then a unit ends up somewhere
  it didn't in the real match, and the player's next move answers a board your clone never saw.

## Clone a Bot

You can also clone a bot: one of the [scripted bots](configuration-objects/opponents.md#the-scripted-bots),
by name, or a bot you trained. The bot you copy is called the teacher.

| Name | What it does |
|---|---|
| `"push"` | Plays every card as far forward as it can. |
| `"defend"` | Only plays on its own half, in front of its king tower. |
| `"patient"` | Waits until it can afford three cards, then plays forward. |
| `"first_affordable"` | Plays the first card it can afford, as soon as it can. |
| `"random"` | Waits most of the time; otherwise plays a random legal card. |

Make a file called `clone_my_bot.py`:

```python title="clone_my_bot.py"
from royalegym import make_env
from royaleimitate import clone, record
from royalelearn import Learner


def build_env():
    return make_env()


if __name__ == "__main__":
    learner = Learner(build_env, save_dir="runs/my_clone")
    demos = record(learner, "push", "runs/demos", battles=200)
    clone(learner, demos, "runs/clone")
    print("Your clone is in runs/clone")
```

Run it with `python clone_my_bot.py`. Recording the 200 battles took about two minutes on our
computer. The cloning after it uses your graphics card, like training does.

- `record` lets the teacher play both sides of 200 battles and writes down every move it makes,
  in `runs/demos`. Record at least 100: `clone` keeps about one battle in twenty aside to check
  the copy, and stops with an error if none was set aside.
- `clone` works as it does for human players, and saves the clone in `runs/clone`.

To clone a bot you trained, pass it instead of `"push"`. It has to be a bot for the same battles,
with the same deck and the same observation, as the one you're cloning into:

```python
    teacher = Learner.load_policy("runs/my_bot")
    demos = record(learner, teacher, "runs/demos", battles=200)
```

Then watch, test and train your clone as in steps 2 to 4 above, with `runs/clone` in place of
`runs/human_clone`, and `from clone_my_bot import build_env` in place of
`from clone_humans import build_env`.

A clone of a scripted bot plays a lot like its teacher, but not exactly: it learned from 200
battles, not from the teacher's rules. It's only as good as its teacher.
