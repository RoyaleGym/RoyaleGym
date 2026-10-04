# Install

This takes about ten minutes, most of it waiting for downloads. Do the steps in order. After
each one, check that you see what the page says you'll see before you go on.

**How to run a command:** type it, or paste it, then press Enter. To paste: Ctrl+V or a
right-click in PowerShell on Windows, Cmd+V on a Mac, Ctrl+Shift+V in a Linux terminal. When a
box has several lines, do them one at a time.

If anything goes wrong, look up the message on [It Doesn't Work](it-doesnt-work.md).

## 1. Install Python

You need **Python 3.12, 3.13 or 3.14**. The newest, 3.14, is a good choice.

=== "Windows"

    1. Go to [python.org/downloads/windows](https://www.python.org/downloads/windows/).
    2. Under **Stable Releases**, find the newest **Python 3.14** (or 3.13 or 3.12) and click
       **Download Windows installer (64-bit)**. If a version newer than 3.14 is listed above it,
       skip that one.
    3. Run the installer. On its first screen, **tick "Add python.exe to PATH"** at the bottom.
       Leave the other box as it is.
    4. Click **Install Now**. When it finishes, you can ignore the "Disable path length limit"
       button and click **Close**.

    Missed the PATH box? Uninstall Python (Settings > Apps > Installed apps > Python >
    Uninstall) and run the installer again.

=== "macOS"

    You need a Mac with Apple silicon (M1 or newer). On a Mac with an Intel processor, the
    PyTorch that RoyaleGym needs no longer installs.

    Go to [python.org/downloads/macos](https://www.python.org/downloads/macos/), download the
    **macOS 64-bit universal2 installer** for the newest Python 3.14 (or 3.13 or 3.12), and run it.
    Skip anything newer than 3.14.

=== "Linux"

    Open a terminal (Ctrl+Alt+T on Ubuntu). On Ubuntu 24.04, which comes with Python 3.12, run:

    ```bash
    sudo apt update
    sudo apt install python3 python3-venv python3-pip
    ```

    `sudo` asks for your password, and nothing shows while you type it. That's normal. Older
    versions of Ubuntu come with an older Python: install Python 3.12, 3.13 or 3.14 with your
    package manager first, or upgrade Ubuntu.

## 2. Open a Terminal

A terminal is the window where you type commands.

=== "Windows"

    Press the Windows key, type `PowerShell`, and press Enter. Then check Python:

    ```powershell
    python --version
    ```

=== "macOS"

    Press Cmd+Space, type `Terminal`, and press Enter. Then check Python:

    ```bash
    python3 --version
    ```

=== "Linux"

    Open your terminal app. Then check Python:

    ```bash
    python3 --version
    ```

You should see something like:

```text
Python 3.14.8
```

3.12 and 3.13 are fine too. If it shows another version, you have more than one Python and the
terminal found the wrong one first. Name the version in step 3 instead, for example
`py -3.14 -m venv venv` on Windows or `python3.14 -m venv venv` on a Mac or Linux. After that, inside the virtual
environment, plain `python` is the right one.

## 3. Make a Folder for Your Bot

This makes a folder called `royale` and a **virtual environment** inside it. A virtual
environment is a private copy of Python just for this project, so nothing you install here
clashes with anything else on your computer.

=== "Windows"

    First, allow PowerShell to turn on virtual environments. Run this once, and type `Y` if it
    asks:

    ```powershell
    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
    ```

    It only changes a setting for your own account, and it doesn't need an admin password.
    It lets PowerShell run scripts that are on your own computer, like the one that turns the
    virtual environment on. Without it, you get a red "running scripts is disabled" error.

    Then make the folder and the virtual environment:

    ```powershell
    mkdir royale
    cd royale
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```

    `mkdir` prints a small table, and the `venv` line prints nothing for a few seconds. That's
    normal.

=== "macOS"

    ```bash
    mkdir royale
    cd royale
    python3 -m venv venv
    source venv/bin/activate
    ```

=== "Linux"

    ```bash
    mkdir royale
    cd royale
    python3 -m venv venv
    source venv/bin/activate
    ```

Your prompt now starts with `(venv)`. That means the virtual environment is on, and anything
you install goes into it. On Windows it looks like the first line below, on a Mac like the
second:

```text
(venv) PS C:\Users\you\royale>
(venv) you@your-mac royale %
```

From here on, plain `python` and `pip` work on every system, because the virtual environment
is on. Your folder is `C:\Users\<your name>\royale` on Windows, and `royale` in your home folder
on a Mac or Linux.

!!! note "Every time you open a new terminal"
    Run just these two lines. Don't run `mkdir` or the `venv` line again.

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

    If your prompt already shows `royale`, skip the `cd` line. If the prompt doesn't start
    with `(venv)`, Python won't find RoyaleGym.

## 4. Install PyTorch for Your Graphics Card

PyTorch is the library that runs your bot's brain (a neural network) on the graphics card.

=== "Windows"

    ```powershell
    pip install torch --index-url https://download.pytorch.org/whl/cu128
    ```

    Don't skip this. On Windows, plain `pip install torch` gets a version that can't use your
    graphics card.

    You **don't** need to install CUDA or anything else from NVIDIA. This one command brings
    what PyTorch needs. Just make sure your NVIDIA driver is up to date, for example with the
    NVIDIA App. It works with GTX 16-series and RTX 20-series cards and newer.

    It's a big download, about 3 GB, and `pip` can sit on one line for several minutes while it
    downloads. That's normal. When it's done, the last line starts with `Successfully installed`.

=== "macOS"

    Nothing to do here: the next step installs PyTorch for you. Macs don't have NVIDIA cards,
    and RoyaleGym doesn't use Apple's graphics chip, so training runs on the processor.

=== "Linux"

    Nothing to do here: the next step installs a PyTorch that already works with NVIDIA cards.
    It needs NVIDIA driver 580 or newer. Check yours with `nvidia-smi`; the version is in the top
    line. To update it on Ubuntu, run `sudo ubuntu-drivers install`, then restart.

If `pip` prints a notice that "a new release of pip is available", you can ignore it.

## 5. Install RoyaleGym

```
pip install "royalegym[all]"
```

Copy the whole line with the copy button, quotes included. `[all]` means "with every part": the
engine, the environments, the trainer, the viewer and the extras. You don't need Git or Rust.

!!! note "An early release"
    New versions come often while the engine is matched to the real game. For a project you want
    to keep the same, stay on the version you installed. To update later, see
    [What happens when Clash Royale updates?](faq.md#what-happens-when-clash-royale-updates)

You'll see a lot of `Downloading` and `Installing` lines. On Linux this step also downloads
PyTorch, about 3 GB, so it can sit on one line for several minutes. It ends with a line that
starts with `Successfully installed` and lists everything it installed, `royalegym` among them.

## 6. Check That It Works

```
python -c "import royalegym, torch; print('RoyaleGym', royalegym.__version__, '| graphics card:', torch.cuda.is_available())"
```

You should see:

```text
RoyaleGym 0.1.8 | graphics card: True
```

The version number may be newer. If it says `graphics card: False` on a computer with an NVIDIA
card, see [PyTorch can't see my graphics card](it-doesnt-work.md#pytorch-cant-see-my-graphics-card).
On a Mac, `False` is expected.

All of this takes about 5 GB of disk space on Windows and Linux, most of it PyTorch, and much
less on a Mac.

You're ready. Next: the [Quick Start](quickstart.md).
