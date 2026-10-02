# It Doesn't Work

Find the message you see, then do what it says underneath. The messages are grouped by when
they happen. If yours isn't here, ask on [Discord](https://discord.gg/4D2BS5JBHP) and paste the
**whole** message, plus the command you ran.

## While Installing

### `python` is not recognized, or the Microsoft Store opens

You see one of these on Windows:

```text
python : The term 'python' is not recognized as the name of a cmdlet, function, script file, or operable program.
```

```text
Python was not found; run without arguments to install from the Microsoft Store, or disable this shortcut from Settings > Manage App Execution Aliases.
```

Python isn't installed, or Windows can't find it. Run the installer from
[python.org](https://www.python.org/downloads/windows/) again and **tick "Add python.exe to
PATH"** on its first screen. Then close PowerShell, open a new one, and try again.

### `command not found: python` or `externally-managed-environment` (Mac, Linux)

You see one of these on a Mac or Linux:

```text
zsh: command not found: python
Command 'python' not found, did you mean: command 'python3'
error: externally-managed-environment
```

The virtual environment isn't on, so there's no plain `python` or `pip`. Turn it on:
`cd royale`, then `source venv/bin/activate`. Your prompt then starts with `(venv)`, and
`python` and `pip` work.

### `running scripts is disabled on this system`

You see this when you turn on the virtual environment on Windows, if you skipped the first
command in step 3 of [Install](install.md#3-make-a-folder-for-your-bot):

```text
.\venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system.
```

PowerShell blocks scripts until you allow them for your account. Run this once, and type `Y`
when it asks:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

It only changes a setting for your own account, needs no admin password, and lets PowerShell run
scripts that are on your own computer. Then run `.\venv\Scripts\Activate.ps1` again.

### `zsh: no matches found: royalegym[all]`

You left out the quotes. Copy the install line from [Install](install.md#5-install-royalegym)
whole, quotes included.

### `requires a different Python`

You see something like:

```text
ERROR: Package 'royalegym' requires a different Python: 3.11.9 not in '>=3.12'
```

Your Python is too old. Install Python 3.13 or 3.12 (step 1 of [Install](install.md)), then
make the virtual environment again (step 3).

### `Failed to build 'pygame'`, or no version of `pygame`

You see one of these in step 5, often after a wall of red text:

```text
Failed to build 'pygame'
ERROR: Could not find a version that satisfies the requirement pygame>=2.6
```

Your Python is 3.14 or newer, and the viewer can't install on it yet. Install Python 3.13 (step
1 of [Install](install.md)), make the virtual environment again with it (step 3), and repeat
steps 4 and 5.

### `Could not find a version that satisfies the requirement torch`

- **On a Mac with an Intel processor**, the message ends with `(from versions: 2.2.0, 2.2.1,
  2.2.2)`. PyTorch no longer makes a version for Intel Macs that RoyaleGym can use, so it doesn't
  install there. You need a Mac with Apple silicon, or a Windows or Linux computer.
- **Anywhere else**, your Python is probably newer than PyTorch supports yet. Check with
  `python --version`, install Python 3.13 or 3.12, make the virtual environment again (step 3),
  and repeat step 4.

### `Could not find a version that satisfies the requirement royalegym`

`pip` can't find RoyaleGym. It isn't on PyPI yet, so `pip install "royalegym[all]"` on its own
doesn't work: the line needs its `--find-links` part. Copy the whole install line from step 5 of
[Install](install.md#5-install-royalegym) with its copy button.

### `pip` is not recognized

Put `python -m` in front of `pip`: `python -m pip install ...` works the same as `pip install ...`.

## When You Run Something

### `No module named 'royalegym'`

```text
ModuleNotFoundError: No module named 'royalegym'
```

The virtual environment isn't on. Your prompt should start with `(venv)`. Go to your folder and
turn it on:

=== "Windows"

    ```powershell
    cd royale
    .\venv\Scripts\Activate.ps1
    ```

=== "macOS and Linux"

    ```bash
    cd royale
    source venv/bin/activate
    ```

### `can't open file ... quickstart.py`

```text
can't open file 'C:\Users\you\royale\quickstart.py': [Errno 2] No such file or directory
```

Python can't find the file. Either you're in a different folder, or the file has another name.

- Run `dir` (Windows) or `ls` (Mac, Linux) to see the files in this folder. If `quickstart.py`
  isn't there, go to your folder with `cd royale`, or make the file again from the terminal as
  in [step 1 of the Quick Start](quickstart.md#1-make-the-quickstart-file).
- If you see `quickstart.py.txt`, rename it: `ren quickstart.py.txt quickstart.py` on Windows,
  `mv quickstart.py.txt quickstart.py` on a Mac or Linux.

### `Cannot find path ... royale\royale` or `cd: no such file or directory: royale`

You ran `cd royale` while you were already in the `royale` folder. Nothing broke: your prompt
already shows `royale`, so skip the `cd` line.

### `SyntaxError`, `IndentationError` or `NameError` after you changed the file

```text
SyntaxError: invalid syntax
IndentationError: unexpected indent
NameError: name 'HogRider' is not defined
```

A change you made to `quickstart.py` broke how Python reads it. The message says which line.
The usual causes:

- A card name lost its quotes: write `"HogRider"`, not `HogRider`.
- Curly quotes (`“ ”`) instead of straight ones (`"`). Python says `invalid character '“'`.
  Retype them. On a Mac, turn off Edit > Substitutions > Smart Quotes in TextEdit.
- A comma is missing between two names, or a bracket `[` `]` or `(` `)` is missing.
- The spaces at the start of a line changed. Every line inside `build_env` starts with exactly
  four spaces, and the lines inside `CombinedReward([` ... `])` with eight.
- A name you added isn't imported: add it to the `from royalegym import ...` line.
- A class you pasted from another page (your own reward, for example) is below `build_env` or
  missing. Paste it above the line `def build_env():`.

If you can't find it, copy the file from the [Quick Start](quickstart.md) again and redo your
change.

### `unexpected character after line continuation character` on line 1 (Mac)

```text
SyntaxError: unexpected character after line continuation character
```

TextEdit saved the file as formatted text, not plain text. Delete it with `rm quickstart.py`,
then make it again with `touch quickstart.py` and `open -e quickstart.py`, and paste again. Or
in TextEdit, choose Format > Make Plain Text before you save.

### `The battle engine ... is not installed`

The engine wasn't installed with RoyaleGym. Run the install line from step 5 of
[Install](install.md#5-install-royalegym) again. If the message shows a `pip` line of its own,
don't use that one: it lacks the `--find-links` part.

### `names '...', which this engine's catalogue of ... cards does not have`

```text
ValueError: decks[0] names 'Archers', which this engine's catalogue of 136 cards does not have (close: Archer, SuperArcher, EliteArcher). A deck can only use cards engine.cards() lists.
```

A card name in your deck is spelled the way the game shows it, not the way RoyaleGym writes it.
Use one of the names after `close:`. Names have no spaces (only `Elixir Collector` keeps its
space), and a few differ: `Archer` is the Archers card, `Log` is The Log. The full list is in
[Game Values](cheatsheets/game-values.md#names-that-differ-from-the-game). Code that uses
`make_env(deck=...)` says `no card named Archers` instead; the fix is the same.

### `... has no evolution that loads` or `... has no loadable hero form`

```text
ValueError: HogRider has no evolution that loads
ValueError: Log has no loadable hero form: the table carries none
```

That card has no evolution, or no hero, in RoyaleGym yet. In your `forms` line, change its `1`
or `2` back to `0`, and play the normal card. (With `make_env`, take it out of `evolved=[...]`
or `heroes=[...]`.) The cards that have one are listed in
[Game Values](cheatsheets/game-values.md#evolutions-and-heroes).

### `build_env must be a function defined at the top level of a module`

The trainer finds `build_env` again by its name, so it must be a normal `def build_env():` at the
left edge of your file. Not inside another function, and not a `lambda`.

### A long red message ending in `KeyboardInterrupt`

You pressed Ctrl+C twice, or while it was still starting up. That's how Python says "you stopped
me", and nothing is broken. Your bot keeps everything up to its last `checkpoint` line; run the
same command again to carry on. Next time, press Ctrl+C once and wait: it saves before it stops.

### `holds a run started with another network, environment or opponent setup`

```text
runs\quickstart holds a run started with another network, environment or opponent setup, so this one cannot carry it on. Use the settings it was started with, or give this run another save_dir. What differs:
```

The bot in that folder was trained under something you've since changed, and it can't carry on
under the new one. The usual causes: the opponent (`"random"` to `"self"`), `n_envs`, the network
size, what the bot sees (the observation line), its moves (for example `ability_buttons=True`),
or the software under it: a new graphics card setup, a new PyTorch, or a RoyaleGym update.

Give `save_dir` a new name to start a new bot, or put back what you changed to carry on the old
one. Your old bot stays in its folder either way.

### `already holds a run`

```text
runs\quickstart already holds a run (metric rows). A fresh start here would begin at iteration 1 and write over its checkpoints.
```

The run in that folder stopped before its first `checkpoint` line, so there's nothing to carry
on. Delete the folder (`Remove-Item -Recurse runs\quickstart` in PowerShell,
`rm -r runs/quickstart` on a Mac or Linux) and run it again.

If the message instead says `Pass resume=True to carry it on`, you set `resume=False` and the
folder already has a bot in it. Give `save_dir` a new name, or remove `resume=False`.

### `holds a run with no checkpoint yet` or `is not a saved bot`

```text
runs\quickstart holds a run with no checkpoint yet: train it longer
runs\hog is not a saved bot, a run's folder or one of its checkpoints
```

`watch.py` couldn't load a bot. The first means training hasn't saved yet: let it run until it
prints a `checkpoint` line, or stop it once with Ctrl+C, which saves. The second means the folder
name in `watch.py` is wrong: use the same name as `save_dir` in `quickstart.py`.

### The computer gets very slow, or runs out of memory

Training uses a lot of memory (RAM): the quickstart's settings can use around 10 GB, and more on
a computer without a graphics card. Close other big programs. If it still runs out, make it
smaller in `quickstart.py`: lower `n_envs` (for example to `8`) and `steps_per_update`,
`ppo_batch_size` and `ppo_minibatch_size` (for example to `4_096`, `4_096` and `512`), with a
new `save_dir`.

## The Graphics Card

### PyTorch can't see my graphics card

The trainer prints:

```text
No GPU that torch can use was found, so this trains on the CPU, which is much slower. With an NVIDIA card, install torch with CUDA (see Install).
```

**On a Mac, this is normal.** A Mac has no NVIDIA card, so it always trains on the processor.
Don't run the commands below on a Mac.

Elsewhere, training works, but very slowly.

=== "Windows"

    This almost always means the wrong PyTorch got installed. Swap it for the one that uses
    your card:

    ```powershell
    pip uninstall -y torch
    pip install torch --index-url https://download.pytorch.org/whl/cu128
    ```

    Still not working? Update your NVIDIA driver with the NVIDIA App or from
    [nvidia.com/drivers](https://www.nvidia.com/drivers), then restart your computer.

=== "Linux"

    Run `nvidia-smi`. The driver version in its top line must be 580 or newer. To update it on
    Ubuntu, run `sudo ubuntu-drivers install`, then restart.

Check that this now prints `True`:

```
python -c "import torch; print(torch.cuda.is_available())"
```

GTX 10-series and older cards don't work with this PyTorch, so on those it trains on the
processor. After you fix the graphics card, a bot you started on the processor can't carry on:
start a new one with a new `save_dir`.

### Not enough graphics memory

You see this when training starts, or it stops with `CUDA out of memory`:

```text
one minibatch of 2048 peaked at ... MB of device memory and only ... MB was free
torch.OutOfMemoryError: CUDA out of memory.
```

Close games and other programs that use the graphics card. If it still happens, lower
`ppo_minibatch_size` in `quickstart.py`. A starting point: `1_024` for a card with 8 GB, `512`
for 6 GB, `256` for 4 GB. The message calls it `ppo.minibatch_size`; it's the same setting. Leave
`doctor.vram_headroom_mb` alone.

### Training is very slow

First check that PyTorch can see your graphics card (above).

The battles take turns on one processor core, so the processor never shows near 100% in Task
Manager. That's normal. Playing more battles at once (`n_envs`) lets the graphics card work on
many of them at the same time, which is why the quickstart plays 32. A faster processor core
helps more than more cores. `n_envs` is fixed for a run: to try another number, use a new
`save_dir`.

## The Viewer

### `royaleviser` is not recognized, or `command not found: royaleviser`

The virtual environment isn't on (see [No module named 'royalegym'](#no-module-named-royalegym)),
or the viewer isn't installed. Run the install line from step 5 of
[Install](install.md#5-install-royalegym) again.

### The viewer window stays empty

The terminal says:

```text
royaleviser: waiting for a run on 127.0.0.1:9870 (to open a saved battle instead: royaleviser FILE)
```

The viewer is waiting for a training run to watch. Start training in another terminal, with
`viser=True` in your `Learner(...)`. The quickstart has it on already. The battle appears once
training starts playing.

### `royaleviser runs/` shows no battle

The terminal says something like `royaleviser: opening runs\quickstart\metrics.jsonl`. Training
doesn't save battles, so the viewer opened a training log instead. Save a battle with
[`watch.py`](quickstart.md#7-watch-a-whole-battle-later) and open that file:
`royaleviser my_bot_battle.msgpack`.

### `cannot stream to 127.0.0.1:9870 because something is already using it`

Only one training run at a time can stream to the viewer, and another one already is. Stop the
other run, or set `viser=False` in one of them. If you put a `ViserPublisher` in your
`build_env`, take it out and use `viser=True` on the `Learner` instead.

### No window appears at all

The viewer needs a screen. On a computer with no screen, such as a server you reach over SSH:

- Save a battle with [`watch.py`](quickstart.md#7-watch-a-whole-battle-later), copy
  `my_bot_battle.msgpack` to a computer with a screen, and open it there with
  `royaleviser my_bot_battle.msgpack`.
- Or make a video of it on the server (on Linux, add `SDL_VIDEODRIVER=dummy` in front):

    ```
    python -c "from royaleviser.capture import capture; from royaleviser.sources import open_source; capture(open_source('my_bot_battle.msgpack'), 'my_bot.mp4')"
    ```
