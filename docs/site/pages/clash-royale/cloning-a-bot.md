# Cloning a Bot

A clone is a new bot that learned by copying another bot's moves. You let a teacher play some
battles, and your bot's network learns to pick the moves the teacher picked. That takes minutes,
not hours, and you can watch the clone play before you do any reinforcement learning.

Everything here comes with `royalegym[all]` from [Install](../install.md), version 0.1.4 or newer.
The clone is part of [RoyaleImitate](../resources/royaleimitate.md).

## 1. Pick a Teacher

The teacher can be one of the [scripted bots](configuration-objects/opponents.md#the-scripted-bots),
by name:

| Name | What it does |
|---|---|
| `"push"` | Plays every card as far forward as it can. |
| `"defend"` | Only plays on its own half, in front of its king tower. |
| `"patient"` | Waits until it can afford three cards, then plays forward. |
| `"first_affordable"` | Plays the first card it can afford, as soon as it can. |
| `"random"` | Waits most of the time; otherwise plays a random legal card. |

Or it can be a bot you trained: `Learner.load_policy("runs/my_bot")`. It has to be a bot for the
same battles, with the same deck and the same observation, as the one you're cloning into.

## 2. Record and Clone

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

Here is what each line does:

- `Learner(build_env, ...)` gives the clone its network and its battles. It doesn't train here.
  Use the same `build_env` you train with, so the clone sees what your bot will see.
- `record` lets the teacher play both sides of 200 battles and writes down every move it makes,
  in `runs/demos`.
- `clone` trains the network to make the same moves. It keeps about one battle in twenty aside to
  check the copy, stops once the copy stops getting better, and saves the clone in `runs/clone`.
  It needs at least 100 battles.

To clone a bot you trained, pass it instead of `"push"`:

```python
    teacher = Learner.load_policy("runs/my_bot")
    demos = record(learner, teacher, "runs/demos", battles=200)
```

## 3. Watch Your Clone

The clone loads like any bot you trained. Make `watch_clone.py` next to `clone_my_bot.py`:

```python title="watch_clone.py"
from royalegym import play_battle
from royalelearn import Learner

from clone_my_bot import build_env

bot = Learner.load_policy("runs/clone")
battle = play_battle(build_env(), blue=bot, red="push", save_to="clone_battle.msgpack")
print("Crowns:", battle.crowns[0], "-", battle.crowns[1])
```

```
python watch_clone.py
royaleviser clone_battle.msgpack
```

Your clone plays blue, at the bottom, against its own teacher. It should play a lot like the
teacher, but not exactly: it learned from 200 battles, not from the teacher's rules.

To see how good it is, test it against the scripted bots the way the FAQ's
[How do I know how good my bot is?](../faq.md#how-do-i-know-how-good-my-bot-is) does, with
`runs/clone` in place of `runs/quickstart`.

## 4. Train From Your Clone

Your clone can go on learning with reinforcement learning, starting from what it copied instead
of from nothing. Make `train_from_clone.py` next to the others:

```python title="train_from_clone.py"
from royaleimitate.artifacts import artifact_digest
from royalelearn import Learner

from clone_my_bot import build_env

if __name__ == "__main__":
    start = {"path": "runs/clone", "sha256": artifact_digest("runs/clone")}
    learner = Learner(build_env, save_dir="runs/from_clone", extensions={"warm_start": {"init": start}})
    learner.learn(total_steps=1_000_000_000)  # stop it with Ctrl+C whenever you like
```

It trains like the [Quick Start](../quickstart.md) and saves in `runs/from_clone`. Watch it the
same way, with `runs/from_clone` in place of `runs/clone`.

`sha256` is a fingerprint of the clone's folder, so the run starts from exactly that clone. Give
both `Learner`s the same settings: the clone's network has to fit the one you train. To keep the
new bot's moves close to the clone's while it learns, see [RoyaleImitate](../resources/royaleimitate.md).

## Clone Human Players

The teacher can also be people. [IL_Replay](https://huggingface.co/datasets/VanguardX101/IL_Replay)
is a public set of real ladder games on Hugging Face, and RoyaleImitate can replay them in your
engine and clone the players. It comes with `royalegym[all]` from version 0.1.6.

Make `clone_humans.py`. It's step 2 with `from_replays` in place of `record`:

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

- `from_replays` downloads the games as it needs them, about 5,000 matches (15 MB) at a time, into
  the Hugging Face cache, so a second run doesn't download them again.
- It replays each match in your engine with both players' card plays. Every decision of both
  players becomes something for the clone to copy: the card and the tile they played, or waiting.
- It skips a match when your engine doesn't have one of its cards or refuses one of its plays,
  and prints how many it skipped. With `make_env()`'s cards, about 7 in 10 matches went through
  when we tried it, each in under a second.

Then watch your clone and train from it as in steps 3 and 4, with `runs/human_clone` in place of
`runs/clone`.
