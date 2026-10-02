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
[python.org](https://www.python.org/downloads/) again and **tick "Add python.exe to PATH"** on
its first screen. Then close PowerShell, open a new one, and try again.

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

You left out the quotes. Type it with them: `pip install "royalegym[all]"`.

### `requires a different Python`

You see something like:

```text
ERROR: Package 'royalegym' requires a different Python: 3.11.9 not in '>=3.12'
```

Your Python is too old. Install 3.12 or newer from [python.org](https://www.python.org/downloads/),
then make the virtual environment again (step 3 of [Install](install.md)).

### `Could not find a version that satisfies the requirement torch`

You see it in step 4 of [Install](install.md#4-install-pytorch-for-your-graphics-card). Your
Python is probably newer than PyTorch supports yet. Check with `python --version`. Install
Python 3.14, 3.13 or 3.12 instead, make the virtual environment again (step 3), and repeat
step 4.

### `Could not find a version that satisfies the requirement royalegym`

`pip` can't find RoyaleGym. It isn't on PyPI yet, so the install line needs `--find-links` and
the address of the release files: see step 5 of [Install](install.md#5-install-royalegym).
If you typed `RELEASES_URL` literally, that's a placeholder, not an address.

### `pip` is not recognized

Use `python -m pip` instead of `pip`, for example `python -m pip install "royalegym[all]"`.

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
  `mv quickstart.py.txt quickstart.py` on a Mac.

### `Cannot find path ... royale\royale`

You ran `cd royale` while you were already in the `royale` folder. Nothing broke: your prompt
already ends in `royale`, so skip the `cd` line.

### `SyntaxError`, `IndentationError` or `NameError` after you changed the file

```text
SyntaxError: invalid syntax
IndentationError: unexpected indent
NameError: name 'HogRider' is not defined
```

A change you made to `quickstart.py` broke how Python reads it. The message says which line.
The usual causes:

- A card name lost its quotes: write `"HogRider"`, not `HogRider`.
- A comma is missing between two names, or a bracket `[` `]` or `(` `)` is missing.
- The spaces at the start of a line changed. Every line inside `build_env` starts with exactly
  four spaces, and the lines inside `CombinedReward([` ... `])` with eight.
- A name you added isn't imported: add it to the `from royalegym import ...` line.

If you can't find it, copy the file from the [Quick Start](quickstart.md) again and redo your
change.

### On a Mac, Python shows an error on line 1

The file was saved as formatted text, not plain text. Make it again from the terminal with
`touch quickstart.py` and `open -e quickstart.py`, as in
[step 1 of the Quick Start](quickstart.md#1-make-the-quickstart-file). Or in TextEdit, choose
Format > Make Plain Text before you save.

### `The battle engine (royalesim) is not installed`

The engine wasn't installed with RoyaleGym. Install it:

```
pip install "royalegym[all]"
```

### `no card named ...`

```text
ValueError: no card named Archers in this engine's catalogue; the names are engine.cards()[i].name, e.g. 'Knight', 'MiniPekka', 'Fireball'
```

A card name in your deck is spelled the way the game shows it, not the way RoyaleGym writes it.
Names have no spaces, and a few differ: `Archer` is the Archers card, `Log` is The Log. The full
list is in [Game Values](cheatsheets/game-values.md#cards).

### `... has no evolution that loads`

That card's evolution isn't in the engine yet. Take it out of `evolved=[...]` and play the
normal card.

### `build_env must be a function defined at the top level of a module`

The trainer finds `build_env` again by its name, so it must be a normal `def build_env():` at the
left edge of your file. Not inside another function, and not a `lambda`.

### A long red message ending in `KeyboardInterrupt`

You pressed Ctrl+C twice, or while it was still starting up. That's how Python says "you stopped
me", and nothing is broken. Your bot keeps everything up to its last `checkpoint` line; run the
same command again to carry on. Next time, press Ctrl+C once and wait: it saves before it stops.

### `already holds a run`

```text
runs\my_bot already holds a run. Pass resume=True to carry it on, or give this run another save_dir.
```

You set `resume=False`, and that folder already has a bot in it. Give `save_dir` a new name to
start a new bot, or remove `resume=False` to carry on the old one.

## The Graphics Card

### PyTorch can't see my graphics card

The trainer prints:

```text
No GPU that torch can use was found, so this trains on the CPU, which is much slower. With an NVIDIA card, install torch with CUDA (see Install).
```

Training works, but very slowly. On Windows this almost always means the wrong PyTorch got
installed. Swap it for the one that uses your card:

```
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Check that this now prints `True`:

```
python -c "import torch; print(torch.cuda.is_available())"
```

Still `False`? Update your NVIDIA driver from [nvidia.com/drivers](https://www.nvidia.com/drivers),
restart your computer, and check again. A Mac has no NVIDIA card, so on a Mac it is always
`False`.

### Not enough graphics memory

Either the trainer refuses to start and says the minibatch won't fit in your graphics card's
memory, or training stops with:

```text
torch.OutOfMemoryError: CUDA out of memory.
```

Close games and other programs that use the graphics card. If it still happens, halve
`ppo_minibatch_size` in `quickstart.py`, for example from `2_048` to `1_024`. The quickstart's
setting fits a card with 12 GB; a card with less memory needs a smaller number.

### Training is very slow

First check that PyTorch can see your graphics card (above). If it can, raise `n_envs`, the
number of battles played at once: the battles run on your processor, and they are usually what
holds training back. Close other heavy programs while you train.

## The Viewer

### `royaleviser` is not recognized

The virtual environment isn't on (see [No module named 'royalegym'](#no-module-named-royalegym)),
or the viewer isn't installed. Install it with `pip install "royalegym[all]"`.

### The viewer window stays empty

The terminal says:

```text
royaleviser: waiting for a run on 127.0.0.1:9870 (to open a saved battle instead: royaleviser FILE)
```

The viewer is waiting for a training run to watch. Start training in another terminal, with
`viser=True` in your `Learner(...)`. The quickstart has it on already. The battle appears once
training starts playing.

### `cannot stream to 127.0.0.1:9870 because something is already using it`

Only one training run at a time can stream to the viewer, and another one already is. Stop the
other run, or set `viser=False` in one of them.

### No window appears at all

The viewer needs a screen. On a server with no display, it can't open a window.
