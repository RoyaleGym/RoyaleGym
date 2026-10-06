<p align="center"><img src="docs/site/pages/assets/royalegym-mark.png" width="128" alt="The RoyaleGym logo: a blue crown shield with a white dumbbell on it, outlined in gold"></p>

<h1 align="center">RoyaleGym</h1>

<p align="center"><a href="https://github.com/RoyaleGym/RoyaleGym/actions/workflows/suite.yml"><img alt="CI" src="https://github.com/RoyaleGym/RoyaleGym/actions/workflows/suite.yml/badge.svg"></a> <img alt="License" src="https://img.shields.io/github/license/RoyaleGym/RoyaleGym?style=flat-square&color=555"> <img alt="Python" src="https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-3776AB?style=flat-square&logo=python&logoColor=white"> <a href="https://royalegym.github.io/RoyaleGym/"><img alt="Docs" src="https://img.shields.io/badge/docs-royalegym.github.io-8957e5?style=flat-square&logo=readthedocs&logoColor=white"></a> <a href="https://discord.gg/8BxaHHVT2u"><img alt="Discord" src="https://img.shields.io/discord/1534663823204552886?style=flat-square&logo=discord&logoColor=white&label=discord&color=5865F2"></a> <img alt="Last commit" src="https://img.shields.io/github/last-commit/RoyaleGym/RoyaleGym?style=flat-square&color=555"></p>

<p align="center"><img alt="APIs: Gymnasium and PettingZoo" src="https://img.shields.io/badge/APIs-Gymnasium%20%2B%20PettingZoo-0b7285?style=flat-square"> <img alt="Engine: Rust, deterministic" src="https://img.shields.io/badge/engine-Rust%2C%20deterministic-DEA584?style=flat-square&logo=rust&logoColor=white"> <img alt="Tick: 50 ms, 20 per second" src="https://img.shields.io/badge/tick-50%20ms%2C%2020%20per%20second-555?style=flat-square"> <img alt="Action space: 2305 moves" src="https://img.shields.io/badge/action%20space-2305%20moves-555?style=flat-square"></p>

**Make a Clash Royale bot in Python.** You write what your bot should want; this gives it the
battles, what it sees, the moves it can make, and a trainer. For a bot that plays like a person,
start by cloning human players from real games
([Cloning a Bot](https://royalegym.github.io/RoyaleGym/clash-royale/cloning-a-bot/)): a bot that
learns only by trial and error, from nothing, doesn't get far.

<p align="center"><img src="docs/site/pages/media/whole-battle.gif" width="100%" alt="A whole battle between two random players, shown in the viewer"></p>

## Install

    pip install "royalegym[all]"

Python 3.12. No Rust, no game files. On Windows with an NVIDIA card, install PyTorch first
([Install](https://royalegym.github.io/RoyaleGym/install/), step 4).

## Try it

```python
from royalegym import make_env, play_battle

battle = play_battle(make_env(), blue="random", red="random", save_to="first_battle.msgpack")
print("winner", battle.winner, "crowns", battle.crowns)
```

Save it as `first_battle.py` and run `python first_battle.py` (a few seconds). `winner` is 0 for
Blue, 1 for Red and 2 for a draw. Then watch it: `royaleviser first_battle.msgpack`. To train a bot,
follow the [Quick Start](https://royalegym.github.io/RoyaleGym/quickstart/): it gives you `quickstart.py`.

## Next

- Quick Start, guides and FAQ: [the docs](https://royalegym.github.io/RoyaleGym/)
- A bot that copies human players: [Cloning a Bot](https://royalegym.github.io/RoyaleGym/clash-royale/cloning-a-bot/)
- How it works inside (Advanced): [architecture](https://royalegym.github.io/RoyaleGym/repos/royalegym/architecture/)
- Questions: [Discord](https://discord.gg/8BxaHHVT2u)

MIT licensed. See [LICENSE](LICENSE).
