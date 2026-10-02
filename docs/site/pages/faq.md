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

- Windows 10 or 11, Linux or a Mac, with Python 3.12, 3.13 or 3.14.
- About 5 GB of free disk space, most of it for PyTorch.
- To train at a useful speed, an NVIDIA graphics card: a GTX 16-series, an RTX 20-series, or
  anything newer.
- A processor with more cores helps too, because the battles run on it.

If you have a PC with an NVIDIA card and a Mac, use the PC.

### Can I use it without an NVIDIA graphics card, or on a Mac?

Yes, but training runs on the processor instead, which is far slower. RoyaleGym doesn't use
Apple's graphics chip, so a Mac always trains on its processor. Everything else (running
battles, writing rewards, the viewer) works the same. On a laptop, a smaller `n_envs`, such as 8,
keeps it from using every core.

### Do I need to know Python or machine learning?

You need to be able to run a Python file and change a line in it. You don't need to know any
machine learning. The [Quick Start](quickstart.md) shows you everything step by step, and the
[terms page](cheatsheets/rl-terms.md) explains the words in plain language.

### Is it free?

Yes. It's open source, under the MIT license.

## Training

### I ran the quickstart. What am I looking at?

One line each time the bot learns from a batch of battles. The number to watch is `crowns`:
crowns taken minus crowns lost, per battle. If it goes up over time, your bot is learning. Every
column is explained in [What You'll See](quickstart.md#3-what-youll-see).

### How long until my bot is any good?

There's no fixed answer. It depends on your graphics card, your settings and your reward. Bots
for games like this usually need many hours of training, often days, before they play well.
Leave it running overnight, and watch it play now and then to see how it's doing.

### How do I watch it play?

While it trains, open a second terminal, turn on the virtual environment and run `royaleviser`.
A window shows one of its battles live. See [Watch It Play](quickstart.md#5-watch-it-play).

### How do I stop training and carry on later?

Press **Ctrl+C** once in the training terminal, and wait. It finishes the update it's on, saves,
and stops. Run the same command again to carry on where it stopped. See
[Stop and Carry On](quickstart.md#6-stop-and-carry-on).

### How do I start over?

Delete the `runs\quickstart` folder, or change `save_dir` in `quickstart.py` to a new name.

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
anywhere. Change the reward, or let it train against copies of itself (`opponent="self"`), so a
trick that beats a weak opponent stops working.

### Why does `crowns` stay near zero?

If it plays against itself (`opponent="self"`), both sides get better together, so neither wins
more. That's expected. Watch it play to see whether it's improving. After a few million steps,
the file `metrics.jsonl` in your `save_dir` also records a rating (`ladder/rating_above_v0`):
how much stronger it is than its first saved version.

### How do I make it use my deck?

Change the `deck = [...]` line in `quickstart.py` to your eight cards. See
[Use Your Own Deck](quickstart.md#8-use-your-own-deck). Card names have no spaces; the full list
is in [Game Values](cheatsheets/game-values.md#cards).

### Can it use evolutions, heroes and champions?

Most evolutions and hero forms, and every champion. See
[Evolutions and heroes](quickstart.md#evolutions-and-heroes) to add them to your deck. If one
isn't in RoyaleGym yet, you get an error that says so.

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

Then look at your processor. The battles run on it, and they are usually what holds training
back. Open Task Manager > Performance > CPU (Activity Monitor on a Mac) while it trains. If the
processor isn't near 100%, raise `n_envs`, the number of battles played at once, in steps (for
example from 32 to 48, then 64) and watch again. If you raise it too far, training just gets
slower, or runs out of memory.

## Playing

### What can I do with my bot once it's trained?

- Watch it play: live while it trains, or [a whole battle later](quickstart.md#7-watch-a-whole-battle-later).
- Test it against the simple bots that come with RoyaleGym (see the next question).
- Keep training it, with a new reward or against itself.
- Keep a copy of its folder in `runs`: that's your bot, with everything it has learned.

### Can I play against my bot?

Not yet. You can watch it play against other bots, or against itself.

### How do I know how good my bot is?

Watch it play, and test it against the simple bots that come with RoyaleGym. See
[Who Is Better?](clash-royale/configuration-objects/opponents.md#who-is-better).

## The Game

### Is it exactly like the real game?

Close, but not exact. The engine is checked against recordings of real battles, and most cards
behave the same. Some details are still being matched, and the engine keeps improving.

### Which cards are in it?

Most cards you can use in ladder battles. The list is in
[Game Values](cheatsheets/game-values.md#cards).

### How do I remove everything?

Delete your `royale` folder. That removes RoyaleGym, PyTorch and your bots. If you don't need
Python any more, uninstall it too (on Windows: Settings > Apps > Installed apps > Python >
Uninstall).

### What happens when Clash Royale updates?

New cards and balance changes reach RoyaleGym in a new version of the engine. To update, run the
install line from [Install](install.md#5-install-royalegym) again, with `--upgrade` added right
after `pip install`. Your code keeps working, but a bot trained on the old
version may play a little differently on the new one.
