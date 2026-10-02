# FAQ

The questions everyone asks. No question is too basic: if yours isn't here, ask on
[Discord](https://discord.gg/4D2BS5JBHP). If you're looking at an error message, go to
[It Doesn't Work](it-doesnt-work.md) instead.

## The Basics

### What is this?

A way to make a Clash Royale bot on your own computer. Your bot plays thousands of practice
battles and learns from them. You decide what it's rewarded for, like winning or taking towers,
and it works out how to get there.

### Do I need the game or a phone?

No. The battles run in a copy of the game's battle engine, on your computer. You don't need the
game, any game files or an account.

### Can my bot play the real game?

No. It trains in the simulator, and that's all it does.

### Will this get me banned?

RoyaleGym never touches the game, your account or Supercell's servers, so there's nothing to ban.

### What computer do I need?

- Windows 10 or 11, Linux, or a Mac with Apple silicon (M1 or newer), with Python 3.12, 3.13 or
  3.14.
- About 5 GB of free disk space, most of it for PyTorch.
- 16 GB of memory (RAM) or more. Training with the quickstart's settings can use around 10 GB.
- To train at a useful speed, an NVIDIA graphics card: a GTX 16-series, an RTX 20-series, or
  anything newer. GTX 10-series and older cards don't work with the PyTorch it uses.
- A fast processor helps too, because the battles run on it.

Intel Macs can't install the PyTorch RoyaleGym needs, so RoyaleGym doesn't run on them.

If you have a PC with an NVIDIA card and a Mac, use the PC.

### Can I use it without an NVIDIA graphics card, or on a Mac?

Yes, but training runs on the processor instead, which is far slower. RoyaleGym doesn't use
Apple's graphics chip, so a Mac always trains on its processor. Everything else (running
battles, writing rewards, the viewer) works the same.

### Can I use Google Colab or a rented GPU?

We haven't tested Colab. A rented Linux machine with an NVIDIA card works like any Linux
computer: follow [Install](install.md) in its terminal. It has no screen, so watch your bot by
saving battles with `watch.py` and opening them on your own computer (see
[No window appears at all](it-doesnt-work.md#no-window-appears-at-all)).

### Do I need to know Python or machine learning?

You need to be able to run a Python file and change a line in it. You don't need to know any
machine learning. The [Quick Start](quickstart.md) shows you everything step by step, and the
[terms page](cheatsheets/rl-terms.md) explains the words in plain language.

### Is it free?

Yes. It's open source, under the MIT license.

## Training

### I ran the quickstart. What am I looking at?

One line each time the bot learns from a batch of battles. The number to watch is `crowns`:
crowns taken minus crowns lost, per battle. If it goes up over time, your bot is learning. It
jumps up and down from line to line, so look at the trend over 20 lines or more. It starts above
zero because the random bot waits 9 moves out of 10, and near +3 your bot beats it 3-0 almost
every time. Every column is explained in [What You'll See](quickstart.md#3-what-youll-see).

### How long until my bot is any good?

There's no fixed answer. It depends on your graphics card, your settings and your reward. Bots
for games like this usually need many hours of training, often days, before they play well.
Leave it running overnight, and watch it play now and then to see how it's doing.

### How do I watch it play?

While it trains, open a second terminal, turn on the virtual environment and run `royaleviser`.
A window shows one of its battles live. See [Watch It Play](quickstart.md#5-watch-it-play).

### How do I stop training and carry on later?

Press **Ctrl+C** once in the training terminal (Control+C on a Mac too), and wait. It finishes
the update it's on, saves, and stops. Run the same command again to carry on where it stopped.
See [Stop and Carry On](quickstart.md#6-stop-and-carry-on).

### How do I start over?

Delete the `runs\quickstart` folder (`Remove-Item -Recurse runs\quickstart` in PowerShell,
`rm -r runs/quickstart` on a Mac or Linux), or change `save_dir` in `quickstart.py` to a new
name.

### Is it OK to leave it running overnight, or for days?

Yes. It's like running a demanding game for that long: the fans get loud, and that's normal.
Keep a laptop plugged in with the lid open, and turn off sleep while it trains. If the computer
restarts, run the script again and it carries on from its last save. There's a checklist in
[How Long Should I Leave It?](quickstart.md#4-how-long-should-i-leave-it)

### How much disk space does a bot take?

Not much. Each save is a few megabytes, and it keeps only the newest three. Training against
itself (`opponent="self"`) also keeps a few older versions of your bot to play against.

### Can I train two bots at once?

Yes, if your computer can keep up: give each its own `save_dir`. Only one of them can stream to
the viewer at a time, so set `viser=False` in the other.

### Why does my bot do nothing?

A brand new bot picks its moves almost at random, and waiting is one of its choices. So at
first it often waits a lot, or plays cards in odd places. That's normal, and it changes as it
learns.

If it still mostly waits after hours of training, look at your reward. If it only gets paid for
winning, it hardly ever sees a reward at all, so it has nothing to learn from. Rewards for crowns
and tower damage, like the quickstart's, pay out much more often.

### Why does my bot spam one card?

It found something your reward pays for, and it's doing it as often as it can. Look at what the
reward pays: a reward for playing cards, for example, teaches it to play cards, any cards,
anywhere. Change the reward, or train a new bot against copies of itself (`opponent="self"`, with
a new `save_dir`), so a trick that beats a weak opponent stops working.

### Why does `crowns` stay near zero?

If it plays against itself (`opponent="self"`), both sides get better together, so neither wins
more. That's expected. Watch it play to see whether it's improving. After a few million steps,
the file `metrics.jsonl` in your `save_dir` also records a rating, `ladder/rating_above_v0`: how
much stronger it is than its first saved version, in Elo points. +100 means it wins about 64% of
its games against that first version.

### How do I make it use my deck?

Change the `deck = [...]` line in `quickstart.py` to your eight cards, and use a new `save_dir`
there and in `watch.py`. See [Use Your Own Deck](quickstart.md#8-use-your-own-deck). Card names
have no spaces, except `Elixir Collector`; the full list is in
[Game Values](cheatsheets/game-values.md#cards).

### Can it use evolutions, heroes and champions?

Most of them: 42 evolutions, 16 heroes and every champion. The lists are in
[Game Values](cheatsheets/game-values.md#evolutions-and-heroes), and
[Evolutions and heroes](quickstart.md#evolutions-and-heroes) shows how to add them to your deck.

### What level are the cards? Can I pick a tower troop?

Every card and tower is level 11, the tournament standard. The towers are the normal Princess
Towers; tower troops aren't in RoyaleGym.

### Can the opponent use a different deck?

Yes: `DefaultStateMutator(decks=[my_deck, other_deck])`. Your bot then plays both sides at
random, so it learns both decks. In the quickstart, both sides use your deck.

### What's a reward, and how do I change it?

The reward is a score your bot gets after every move. It learns to do whatever makes that score
high, so the reward is how you tell it what you want. In `quickstart.py` it's the `reward_fn`
lines. See [Change What It Learns](quickstart.md#9-change-what-it-learns), and
[Reward Functions](clash-royale/configuration-objects/reward-functions.md) to write your own.

### What do all the settings in `Learner(...)` mean?

Each line in `quickstart.py` has a short comment. The [RoyaleLearn](resources/royalelearn.md)
page lists every setting and its default, and the [terms page](cheatsheets/rl-terms.md) explains
the words. You can train a good bot without touching most of them.

### How do I make training faster?

First make sure PyTorch can see your graphics card (see
[It Doesn't Work](it-doesnt-work.md#pytorch-cant-see-my-graphics-card)).

The battles take turns on one processor core, so the processor won't show near 100%, and a
faster core helps more than more cores. Playing more battles at once (`n_envs`) lets the
graphics card work on many of them together, which is why the quickstart plays 32. `n_envs` is
fixed for a run: to try another number, start a new bot with a new `save_dir`.

### Can I see graphs of how it's doing?

Yes, with Weights & Biases, a free website for charts. Make an account at
[wandb.ai](https://wandb.ai), run `pip install wandb` and then `wandb login` once (with the
virtual environment on), and set `log_to_wandb=True` in `quickstart.py`.

## Playing

### What can I do with my bot once it's trained?

- Watch it play: live while it trains, or [a whole battle later](quickstart.md#7-watch-a-whole-battle-later).
- Test it against the simple bots that come with RoyaleGym (see below).
- Keep training it with a new reward. Training it against itself starts a new bot in a new
  `save_dir`.
- Keep a copy of its folder in `runs`: that's your bot, with everything it has learned.

### How do I share my bot, or a battle?

Send your bot's folder from `runs`. Your friend loads it with `Learner.load_policy("that folder")`
and plays it with the same deck, as in `watch.py`. They can't carry on training it on their
computer. To share a battle, send the `.msgpack` file that `watch.py` saves; they open it with
`royaleviser FILE`. To make a video of it, see
[No window appears at all](it-doesnt-work.md#no-window-appears-at-all).

### Can I play against my bot?

Not yet. You can watch it play against other bots, or against itself.

### How do I know how good my bot is?

Watch it play, and test it against the simple bots that come with RoyaleGym. Save this as
`how_good.py` next to `quickstart.py` and run `python how_good.py`. It takes a few minutes:

```py
from royalegym import CallableOpponent, evaluate, ladder
from royalelearn import Learner

from quickstart import build_env

bot = Learner.load_policy("runs/quickstart")
me = CallableOpponent(lambda obs, mask: bot(obs))
for name, opponent in ladder():
    print(evaluate(me, opponent, build_env, games=50, names=("my bot", name)).summary())
```

Each line says how many battles it won and whether it's clearly better or "too close to call".
A good bot clearly beats every bot on the list, not just the random one. More about this on
[Opponents](clash-royale/configuration-objects/opponents.md#who-is-better).

## The Game

### Is it exactly like the real game?

Close, but not exact. The engine is checked against recordings of real battles, and most cards
behave the same. Some details are still being matched, and the engine keeps improving.

### Which cards are in it?

Most cards you can use in ladder battles. The list is in
[Game Values](cheatsheets/game-values.md#cards).

### How do I remove everything?

First, with the virtual environment on, run `pip cache purge`: `pip` keeps its own copy of
everything it downloaded, PyTorch included, outside your folder. Then delete your `royale`
folder. That removes RoyaleGym, PyTorch and your bots. If you don't need Python any more,
uninstall it too (on Windows: Settings > Apps > Installed apps > Python > Uninstall).

### What happens when Clash Royale updates?

New cards and balance changes reach RoyaleGym in a new version of the engine. To update, find
the newest version on the [Releases page](https://github.com/RoyaleGym/RoyaleGym/releases), and
run this with that version in place of `v0.1.3` at the end. Updating from v0.1.1 or earlier? Run
`pip uninstall -y pygame` first: the viewer now uses pygame-ce in its place.

```
pip install --upgrade royalegym royalesim royalelearn royaleviser royaleimitate --find-links https://github.com/RoyaleGym/RoyaleGym/releases/expanded_assets/v0.1.3
```

It updates the five RoyaleGym packages and leaves your PyTorch alone. Your code keeps working,
but a bot can't carry on training across an update: start a new one with a new `save_dir`.
