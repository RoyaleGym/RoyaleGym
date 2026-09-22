# The learner

[![repo](https://img.shields.io/badge/repo-RoyaleLearn-3776AB?style=flat-square&logo=python&logoColor=white)](https://github.com/RoyaleGym/RoyaleLearn)
![status](https://img.shields.io/badge/training%20loop-runs%20on%20the%20real%20engine-d29922?style=flat-square)
![tests](https://img.shields.io/badge/tests-green%2C%20and%20growing%20fast-2ea043?style=flat-square)
![torch](https://img.shields.io/badge/torch-optional%20extra-555?style=flat-square)

**RoyaleLearn closed its training loop on 2026-09-22, and nobody has trained a bot with it
yet.** Both halves matter, so here they are in order.

The command runs end to end: rollouts, gradient steps, a checkpoint with its manifest, a
snapshot in the opponent pool, and a `resume` that reloads it in a fresh process. It needs
torch, which the plain install does not pull in.

```
pip install -e "RoyaleLearn[torch]"
python -m royalelearn train --config examples/configs/smoke.json
```

The only run so far was a three-iteration smoke test to prove the loop closes. So nothing is
known about how long a useful run takes, what it costs, or whether the bot it produces is any
good.

Around the loop is the boring half you would otherwise write yourself, and it has been tested
far longer than the loop has: the settings file and its sanity checks, the run identity that a
resume is checked against, the networks, the code that turns an observation into numbers, the
buffer that holds experience, the workers that collect battles, the ladder that rates one
policy against another, the metrics output and the checkpoint store.

!!! tip "You do not have to wait"
    You can train a bot today by pointing an existing library at [the environments](environments.md).
    They expose `action_masks()` in the form sb3-contrib's MaskablePPO expects.
    [Your first bot](../first-bot.md) walks through it.

## What is it for, when it lands

One job: turn battles into a bot that is actually better than the one before it.

<div class="grid cards" markdown>

-   **It plays itself**

    ---

    N battles run at once as 2N player slots, so one bot learns from both sides of every match
    in a single batch. Both seats feed the same policy.

-   **It keeps a ladder of its old selves**

    ---

    Past versions of your bot are kept frozen in a pool. A new version is rated against them,
    and only joins the pool when its win rate clears a gate. That is how you tell real
    improvement from a bot that only beats its own latest quirk.

-   **A resumed run is the same run**

    ---

    A checkpoint holds the policy, the critic, the optimizer, the pool and the seeds. Reload it
    and the curve carries on where it was, rather than restarting somewhere nearby.

-   **You can watch it happen**

    ---

    Set one environment variable and [the viewer](viewer.md) attaches to the training run from
    another window, with your loss and your ladder rating in a panel beside the board.

</div>

## What runs today

The package imports without torch. That matters more than it sounds: the settings and the run
identity are useful on a machine where you have not installed a gigabyte of deep-learning
libraries, and torch is only pulled in when you ask for a name that actually needs it.

```
python -c "import royalelearn, sys; print(royalelearn.RunConfig, 'torch' in sys.modules)"
```

```
<class 'royalelearn.config.RunConfig'> False
```

Under that, everything runs: the engine, the environments, the legal-move mask, seeding from end
to end, the opponent-pool bookkeeping, and the viewer stream.

## The commands that will exist

These are the plan. Do not try them tonight.

!!! note "Run here, not from a clean install"
    These commands were run on a machine that already had everything built. Nobody has yet gone
    from four fresh clones to a training run in one sitting, so the install path around them is
    the untested part, not the commands.

```
python -m royalelearn train --config examples/configs/laptop.json
python -m royalelearn config --profile laptop -o run.json     # writes a config you can edit
python -m royalelearn doctor --config run.json                # first-run checks
python -m royalelearn bench                                   # this machine's throughput
```

`doctor` is the one worth knowing about in advance. It builds one environment, prints the engine
build fingerprint and the observation shapes, checks the legal-move mask against the engine
exhaustively, works out how much memory the run will need, and refuses to start a run that will
not fit. `bench` measures your own machine instead of quoting somebody else's.

There will also be `examples/train_1v1.py`, which is about fifteen lines: load a config, change
a couple of fields, run it.

## Changing the reward

This is the thing most people will want, so here is where it lives. The reward function is in
`royalelearn/rewards.py` and is assembled in `default_potential_reward()`. You change it by
writing a `RewardFunction` subclass, which is RoyaleGym's base class, and naming it in the
config's env block. Naming it in the config means it gets recorded in the checkpoint, so months
later the file still says what the bot was trained to want. `examples/custom_reward.py` will
show it.

[Writing a reward function](../rewards.md) is the page for this, and it works today against the
environments.

!!! warning "Do not re-tune the shipped weights"
    The shipped reward terms are built as potential differences, which is a shape that cannot
    change which strategy is best. It can only change how fast the bot finds it. That property
    is worth keeping.

    If you nudge a weight every time you see a behaviour you dislike, the weight is standing in
    for a term that is missing. Add the term.

## When you would touch it

- You want to train a bot properly, with self-play, a ladder and checkpoints, once the loop
  lands.
- You want a different rating scheme, a different way of sampling opponents from the pool, or
  your metrics somewhere other than where they go by default. Each of those is a base class with
  a default, so you replace one without forking the training loop.
- You want to read the design before it is built. That is the point of
  [`docs/design.md`](https://github.com/RoyaleGym/RoyaleLearn/blob/main/docs/design.md), which
  carries the reasoning rather than just the plan.

## When you would not

| You want to | Go here instead |
|---|---|
| train something today | [Your first bot](../first-bot.md), with an off-the-shelf library |
| change what the bot wants | [Writing a reward function](../rewards.md) |
| change what it sees or what its moves mean | [The environments](environments.md) |
| fix how the battle behaves | [The engine](engine.md) |
| watch a run | [The viewer](viewer.md) |

## Two design decisions you may disagree with

They are written down so you can argue with them in
[the Discord](https://discord.gg/4D2BS5JBHP) rather than guess at them.

**There is deliberately no quick version first.** No throwaway trainer, no baseline learner to
tide people over. The layers underneath were finished to a standard, and a half-finished harness
would be the thing everyone used forever.

**Nothing types a number that the environment already knows.** Every shape, every width, every
field position is read off a running environment at startup. The card catalogue on this machine
went from 65 to 95 cards one morning and no code noticed, which is exactly the point. If you
ever see a page or a config with an observation width written into it, that page is already
wrong.

## Where the detail is

- [RoyaleLearn's README](https://github.com/RoyaleGym/RoyaleLearn) for what is built and what is
  open.
- [`docs/design.md`](https://github.com/RoyaleGym/RoyaleLearn/blob/main/docs/design.md) for the
  pieces, the metric, and the conventions the harness is held to.
- [`docs/harness-spec.md`](https://github.com/RoyaleGym/RoyaleLearn/blob/main/docs/harness-spec.md)
  for the specification itself.
