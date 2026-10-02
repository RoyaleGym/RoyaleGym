# Quick Start

In this guide you'll train your first bot, watch it play, and then change its deck and what it
learns. Install RoyaleGym first: see [Install](install.md).

<!-- known issue, v0.1.0 only: remove when v0.1.1 ships -->
!!! warning "Right now: Windows only"
    In this first version (v0.1.0), the quickstart only runs on Windows. On Linux and macOS it
    installs, but stops with an error as it starts. A fix is coming in the next version.

## 1. Make the Quickstart File

In your terminal, in the `royale` folder with `(venv)` showing, open a new empty file in a text
editor:

=== "Windows"

    ```powershell
    notepad quickstart.py
    ```

    Notepad asks whether to create a new file. Click **Yes**.

=== "macOS"

    ```bash
    touch quickstart.py
    open -e quickstart.py
    ```

=== "Linux"

    ```bash
    nano quickstart.py
    ```

Now copy the whole file below (the copy button is in its top right corner), paste it into the
editor, and save (Ctrl+S, or Cmd+S on a Mac). Because you made it from the terminal, it's
already in the right folder with the right name.

```py
--8<-- "examples/quickstart.py"
```

You don't need to understand all of it yet. The parts you'll change first are the `deck` line
and the `reward_fn` lines, and this page shows you how.

!!! tip "Don't double-click it"
    Double-clicking `quickstart.py` runs it in a window that closes straight away. To change
    it, open it with `notepad quickstart.py` (or your editor) again. To run it, use the
    terminal, as below.

## 2. Run It

```
python quickstart.py
```

## 3. What You'll See

First it prints one line about the run. Then, each time the bot has learned from a batch of
battles, it prints an `update` line:

```text
training in runs\quickstart on RustEngine: 32 battles at once, a line every 16,384 steps
update     1  steps      16,384  battles   31  crowns  +0.71 a battle     71 s
update     2  steps      32,768  battles   37  crowns  +0.43 a battle     56 s
update     3  steps      49,152  battles   50  crowns  +1.00 a battle     36 s
```

That's from an RTX 4070 Ti. Your numbers will be different: they change from run to run, and a
slower graphics card or processor takes more seconds per update.

The first `update` line takes the longest, because it includes starting up. As long as no error
appeared, it's working. Now and then it also prints a `checkpoint` line: that's it saving its
progress.

| Column | What it means |
|---|---|
| `update` | How many times the bot has learned so far. |
| `steps` | How many moves it has made in all, across every battle. |
| `battles` | How many battles finished since the last line. |
| `crowns` | Crowns it took minus crowns it lost, per battle. It goes from -3 (it loses every battle 0-3) to +3 (it wins every battle 3-0). A `-` means no battle finished yet. |
| `s` | How many seconds that update took. |

`crowns` is the number to watch. If it goes up over time, your bot is learning.

**Is it using my graphics card?** If it weren't, it would have printed a warning that starts
with "No GPU that torch can use was found". No warning means it's on the graphics card. On
Windows you can watch it work in Task Manager > Performance > GPU.

## 4. How Long Should I Leave It?

As long as you like. There's no point where it's "done": it keeps slowly getting better, and you
can stop and carry on whenever you want. Bots for games like this usually need many hours of
training before they play well.

In the quickstart, your bot plays a bot that makes random moves. A high `crowns` number means it
beats that random bot, which is a start but not the same as being good. Once `crowns` stays high,
your bot has learned most of what the random bot can teach it. Then try `opponent="self"`, so it
plays against copies of itself instead. Against itself, `crowns` stays near zero, because both
sides get better together; [watch it play](#5-watch-it-play) to see how it's doing.

!!! note "Leaving it running overnight"
    - Keep a laptop plugged in, with the lid open.
    - Turn off sleep while it trains (on Windows: Settings > System > Power).
    - Loud fans are normal. It's using the graphics card and processor fully.
    - Playing games on the same computer slows training down, and can make it run out of
      graphics memory.
    - If the computer restarts, for example for an update, run `python quickstart.py` again.
      It carries on from its last saved `checkpoint`.

## 5. Watch It Play

While it trains, open a **second** terminal, turn on the virtual environment, and start the
viewer:

=== "Windows"

    ```powershell
    cd royale
    .\venv\Scripts\Activate.ps1
    royaleviser
    ```

=== "macOS and Linux"

    ```bash
    cd royale
    source venv/bin/activate
    royaleviser
    ```

A window opens and shows one of the battles your bot is playing, live: your bot on one side,
its opponent on the other.

![The viewer: one side's hand and elixir top left, the other's bottom left, the arena in the middle](media/viewer.png){ width="100%" }

Press **Space** to pause, **h** to see every key, and **q** to close it. The viewer is a
separate program: pausing or closing it doesn't pause or stop training, and you can open it
again any time.

## 6. Stop and Carry On

To stop, click the training terminal and press **Ctrl+C** once. It prints:

```text
Ctrl-C: stopping after this update, with a checkpoint. Press it again to stop now.
```

It finishes the update it's on, saves, and stops. That can take a little while, so wait for the
`checkpoint` line. Your bot is saved.

If you press Ctrl+C a second time, or close the window, it stops at once. You then keep
everything up to its last `checkpoint` line.

To carry on, run the same command again:

```
python quickstart.py
```

Its first line now ends with `carrying on from` and the checkpoint it loaded. Everything it has
learned is in the `runs\quickstart` folder. To start again from nothing, delete that folder.

## 7. Watch a Whole Battle Later

After you've stopped training, you can make your bot play one battle against the random bot and
watch the whole thing. Make a second file, `watch.py`, the same way as in step 1
(`notepad watch.py` on Windows), and paste this into it:

```py
from royalegym import play_battle
from royalelearn import Learner

from quickstart import build_env

bot = Learner.load_policy("runs/quickstart")
battle = play_battle(build_env(), blue=bot, red="random", save_to="my_bot_battle.msgpack")
print("Crowns:", battle.crowns[0], "-", battle.crowns[1])
```

Run it, then open the battle in the viewer:

```
python watch.py
royaleviser my_bot_battle.msgpack
```

`Learner.load_policy("runs/quickstart")` loads the newest save of your bot. Your bot plays blue,
at the bottom. Run `watch.py` again for a new battle each time.

## 8. Use Your Own Deck

Open `quickstart.py` again (`notepad quickstart.py`) and find this line in `build_env`:

```py
    deck = ["Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Cannon", "Musketeer"]
```

Replace the eight names with yours. Keep each name inside quotes, with commas between them, and
keep the spaces at the start of the line. For example, Hog 2.6:

```py
    deck = ["HogRider", "Musketeer", "Cannon", "IceGolemite", "IceSpirits", "Skeletons", "Fireball", "Log"]
```

Names are written without spaces, and some differ from the game: `IceGolemite` is the Ice Golem,
`IceSpirits` the Ice Spirit and `Log` The Log. The table of [names that differ](cheatsheets/game-values.md#names-that-differ-from-the-game)
lists the rest. To search for a name, run this (change `ice` to part of the name you want):

```
python -c "from royalegym import RustEngine; print([c.name for c in RustEngine().cards() if 'ice' in c.name.lower()])"
```

```text
['IceWizard', 'IceSpirits', 'IceGolemite', 'SuperIceGolemite']
```

Then change `save_dir="runs/quickstart"` to a new name, such as `save_dir="runs/hog"`.
Otherwise it quietly carries on training your old bot with the new deck, instead of starting a
new one.

### Evolutions and heroes

Add a `forms` line under your deck, with one number per card in the same order: `0` the normal
card, `1` its evolution, `2` its hero. Then pass it to `DefaultStateMutator`:

```py
    deck = ["HogRider", "Musketeer", "Cannon", "IceGolemite", "IceSpirits", "Skeletons", "Fireball", "Log"]
    forms = [0, 1, 0, 0, 0, 1, 0, 0]  # evolved Musketeer and evolved Skeletons

    state_mutator = DefaultStateMutator(decks=[deck, deck], forms=[forms, forms])
```

A hero has an ability button, so with a hero (a `2`) also change the action parser line to:

```py
    action_parser = TileActionParser(ability_buttons=True)
```

If a card has no evolution or hero in RoyaleGym yet, you get an error that says so.

### Good to know about decks

- Both sides play the deck you give, so your bot learns a mirror match.
- Every card and tower is level 11, the tournament standard. The towers are the normal Princess
  Towers; there's no tower troop to choose.
- If you give the two sides different decks (`decks=[my_deck, other_deck]`), your bot plays
  both sides at random, so it learns both decks.

## 9. Change What It Learns

The **reward** is how you tell your bot what you want. After every move it gets a score, and it
learns to do whatever makes that score high. In `quickstart.py` it's these lines:

```py
    reward_fn = CombinedReward([
        (WinLossReward(), 1.0),  # +1 for a win, -1 for a loss
        (CrownReward(), 0.2),    # for each crown taken, minus each crown lost
        (TowerHPReward(), 0.1),  # tower damage done, minus tower damage taken
    ])
```

Each line inside the brackets is one thing it gets paid for, and the number is how much that
counts. For example, to also charge it for wasting elixir by sitting at 10, do two things:

1. Add `ElixirLeakPenalty` to the end of the import line above, which becomes:

    ```py
        from royalegym import CombinedReward, CrownReward, TowerHPReward, WinLossReward, ElixirLeakPenalty
    ```

2. Add one line inside the brackets:

    ```py
        reward_fn = CombinedReward([
            (WinLossReward(), 1.0),
            (CrownReward(), 0.2),
            (TowerHPReward(), 0.1),
            (ElixirLeakPenalty(), 0.01),  # -0.01 for each move it spends at full elixir
        ])
    ```

Keep the number next to `WinLossReward` the biggest one. The others are hints that help your bot
find its way to a win. If a hint is worth more than winning, the bot learns to chase the hint
instead.

Change `save_dir` to a new name here too, then run it again. The rewards that come with
RoyaleGym, and how to write your own, are on [Reward Functions](clash-royale/configuration-objects/reward-functions.md).

## Next

- [FAQ](faq.md): the questions everyone asks.
- [It Doesn't Work](it-doesnt-work.md): error messages and what to do about them.
- [Training an Agent](clash-royale/training-an-agent.md): when you're ready to go further.
