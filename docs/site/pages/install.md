# Install

You need Python 3.12 or newer. Nothing else: no Rust, no game files.

```
pip install "royalegym[all]" --find-links RELEASES_URL
```

<!-- Until PyPI: RELEASES_URL = the RoyaleSim GitHub Releases page (Gym/Sim send the exact URL).
     After the owner OKs PyPI, the line is just: pip install "royalegym[all]" -->

That installs all five pieces. Check it worked:

```
python -c "import royalegym; print(royalegym.__version__)"
```

## Only some of it

`[all]` is everything. If you want less, pick the pieces you need:

| Extra | Adds |
|---|---|
| `royalegym[sim]` | the battle engine (needed to play battles) |
| `royalegym[learn]` | the trainer (pulls in PyTorch, a large download) |
| `royalegym[viser]` | the viewer window |
| `royalegym[imitate]` | learning from replays of real players |

## Use a virtual environment

So nothing clashes with other Python projects, install into a fresh one:

=== "Windows"

    ```
    python -m venv royale-env
    royale-env\Scripts\activate
    ```

=== "macOS and Linux"

    ```
    python3 -m venv royale-env
    source royale-env/bin/activate
    ```

Then run the install line above.

## It didn't work

See [Troubleshooting](troubleshooting.md). If your problem isn't there, ask on [Discord](https://discord.gg/4D2BS5JBHP)
and paste the whole error.

Next: [Quick Start](quick-start.md)
