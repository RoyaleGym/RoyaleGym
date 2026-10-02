# RoyaleGym

[![CI](https://github.com/RoyaleGym/RoyaleGym/actions/workflows/suite.yml/badge.svg)](https://github.com/RoyaleGym/RoyaleGym/actions/workflows/suite.yml)

**Make a Clash Royale bot in Python.** You write what your bot should want; this gives it the
battles, what it sees, the moves it can make, and a trainer.

<p align="center"><img src="docs/site/pages/media/whole-battle.gif" width="100%" alt="A whole battle between two random players, shown in the viewer"></p>

## Install

    pip install "royalegym[all]"

Python 3.12 or newer. No Rust, no game files. (Until the first release, see the
[Install page](docs/site/pages/install.md) for the exact line.)

## Try it

```python
from royalegym import make_env, play_battle

battle = play_battle(make_env(), blue="random", red="random", save_to="first_battle.msgpack")
print("winner", battle.winner, "crowns", battle.crowns)
```

Then watch it: `royaleviser first_battle.msgpack`. To train a bot, run
[`examples/quickstart.py`](examples/quickstart.py).

## Next

- Quick Start, Guides and FAQ: [the docs](docs/site/pages/index.md)
- How it works inside (Advanced): [docs/architecture.md](docs/architecture.md)
- Questions: [Discord](https://discord.gg/4D2BS5JBHP)

MIT licensed. See [LICENSE](LICENSE).
