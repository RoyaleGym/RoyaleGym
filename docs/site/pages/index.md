# Introduction

**RoyaleGym lets you make a Clash Royale bot on your own computer.** Your bot plays practice
battles, thousands of them, and slowly learns to win. You decide what it gets rewarded for. It
works out the rest by trial and error.

![What a battle looks like in the viewer: both hands, both elixir bars, and the arena with every unit](media/viewer.png){ width="100%" }

## Quick Answers

**Do I need the game, a phone or an emulator?**
No. The battles run in RoyaleSim, a copy of the game's battle engine that runs on your computer.
You don't need the game, any game files, or an account.

**Can my bot play the real game for me?**
No. It trains in the simulator, and that's all it does.

**Can I get banned for this?**
RoyaleGym never touches the game, your account or Supercell's servers. There's nothing to ban.

**What computer do I need?**
Windows 10 or 11, Linux, or a Mac, with Python 3.12 or newer. To train a bot you also want an
NVIDIA graphics card (GTX 16-series, RTX 20-series or newer). Without one, everything still works, but
training is too slow to be much fun.

**Do I need to know machine learning?**
No. If you can copy a file, run it, and change a line of Python, you can train a bot. The
[Quick Start](quickstart.md) walks you through it.

**How long until my bot is good?**
There's no fixed answer. It depends on your graphics card, your settings and what you reward.
Bots for games like this usually train for many hours, often days, before they play well. Your
first run is about watching it learn, not about getting a strong bot.

**Is it free?**
Yes. It's open source, under the MIT license.

**Is it the real game?**
Close. The engine is checked against recordings of real battles, and most cards behave the same.
Some details are still being matched, so the engine keeps improving. Your bot's code won't need
to change when it does.

## What You'll Do

1. [Install it](install.md). About ten minutes, most of it waiting for downloads.
2. [Run the Quick Start](quickstart.md). One file that trains a bot and shows you its progress.
3. [Watch your bot play](quickstart.md#5-watch-it-play) in the viewer.
4. Change something, like [its deck](quickstart.md#8-use-your-own-deck) or
   [what it's rewarded for](quickstart.md#9-change-what-it-learns), and train again.

Stuck? Look in the [FAQ](faq.md) and [It Doesn't Work](it-doesnt-work.md). Still stuck? Ask on
[Discord](https://discord.gg/4D2BS5JBHP). Nobody minds beginner questions there.

## What's Inside

You install one thing, `royalegym`, and it brings these along. You'll mostly use the first two
by name.

| Part | What it does |
|---|---|
| **RoyaleGym** | Sets up the battles: what your bot sees, what moves it can make, and what it's rewarded for. |
| **RoyaleLearn** | The trainer. It runs the battles and teaches your bot. |
| **RoyaleSim** | The battle engine. Elixir, cards, troops, spells, towers, overtime. |
| **RoyaleViser** | A window that shows a battle, so you can watch your bot. |
| **RoyaleImitate** | Optional. Starts a new bot from one you already trained. |

If you know RLGym for Rocket League: this is the same idea, split the same way. RoyaleSim is
RocketSim, RoyaleGym is RLGym, RoyaleLearn is RLGym-PPO and RoyaleViser is RLViser.

RoyaleGym is a fan project. It is not affiliated with, endorsed or sponsored by Supercell.
