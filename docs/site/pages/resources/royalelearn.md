# RoyaleLearn

[RoyaleLearn](https://github.com/RoyaleGym/RoyaleLearn) is the trainer: RoyaleGym's RLGym-PPO.
You give it a function that builds your environment, and it trains a bot on it with PPO. It
plays many battles at once, learns on your graphics card, saves checkpoints as it goes, and
carries on from the last one if you stop and start again.

```py
from royalelearn import Learner

learner = Learner(build_env, save_dir="runs/my_bot")
learner.learn(total_steps=1_000_000)
learner.save("runs/my_bot/bot")
```

`build_env` must be a function defined at the top level of your script, not a lambda or a
function inside another function: the trainer finds it again by its name. It returns a
`ClashParallelEnv` with `SpatialObsBuilder` and `TileActionParser`, the defaults.

## Every Setting

Every setting has a default, so you only pass the ones you want to change.

**The run**

| Setting | Default | What it does |
|---|---|---|
| `n_envs` | `8` | Battles played at once. The battles run on your CPU, so this is usually what makes training faster. |
| `device` | `"auto"` | `"auto"` uses your graphics card if PyTorch can see one, and the CPU otherwise. Or `"cuda"`, `"cpu"`. |
| `threads` | half your cores, at most 8 | CPU threads for the network's maths. |
| `save_dir` | `"runs/royalelearn"` | Where checkpoints and logs go. |
| `resume` | `True` | Carry on from the newest checkpoint in `save_dir`. `False` refuses to start if one is there. |
| `opponent` | `"random"` | `"random"`, `"noop"` or `"self"`. See [Opponents](../clash-royale/configuration-objects/opponents.md). |
| `timestep_limit` | none | Where `learn()` stops when you don't give it `total_steps`. |
| `checkpoint_every` | `50_000` | Save a checkpoint every this many decisions. |
| `n_checkpoints_to_keep` | `3` | Older checkpoints are deleted. |
| `seed` | none | Makes a run repeatable on the same computer. |

**The network**

| Setting | Default | What it does |
|---|---|---|
| `trunk_channels` | `32` | How wide the network is. |
| `trunk_blocks` | `2` | How deep it is. |
| `critic_hidden` | `64` | The size of the part that predicts the reward. |

**The update** (the names follow RLGym-PPO's where there is one)

| Setting | Default | What it does |
|---|---|---|
| `steps_per_update` | `1024` | Decisions collected before each update. RLGym-PPO calls it `ts_per_iteration`. |
| `ppo_batch_size` | `steps_per_update // 2` | Decisions used in each update. |
| `ppo_minibatch_size` | `steps_per_update // 8` | Decisions per adjustment. Bigger uses more graphics memory. |
| `ppo_epochs` | `3` | Passes over each batch. |
| `policy_lr` | `2e-4` | Learning rate of the policy. |
| `critic_lr` | `2e-4` | Learning rate of the critic. |
| `ppo_ent_coef` | `0.01`, falling to `0.003` | How much exploring is rewarded. A number keeps it fixed. |
| `ppo_clip_range` | `0.2` | How far one update may move the policy. |
| `gae_gamma` | `0.997`, rising to `0.999` | How much future rewards count. A number keeps it fixed. |
| `gae_lambda` | `0.99` | How the trainer spreads credit for a reward over the moves before it. |
| `standardize_returns` | `True` | Scales rewards to a steady size, which keeps learning stable. |

**Watching**

| Setting | Default | What it does |
|---|---|---|
| `viser` | `False` | Stream one battle to the [viewer](../clash-royale/configuration-objects/viewer.md). |
| `log_to_wandb` | `False` | Chart the run on [Weights & Biases](https://wandb.ai). Needs `pip install wandb`. |
| `wandb_project_name`, `wandb_group_name`, `wandb_run_name` | none | Where the charts go on wandb. |
| `verbose` | `False` | Print the whole start-up report. |

**Add-ons**

| Setting | Default | What it does |
|---|---|---|
| `extensions` | none | Settings for add-ons such as [RoyaleImitate](royaleimitate.md). |

The words in these tables are explained in [Reinforcement Learning Terms](../cheatsheets/rl-terms.md).

## Checkpoints and Carrying On

Press Ctrl+C once and the trainer finishes the update it's on, writes a checkpoint, and returns
from `learn()`, so the rest of your script still runs. A second Ctrl+C stops at once.

Everything a run writes goes in `save_dir`: checkpoints, one line of numbers per update in
`metrics.jsonl`, and the run's settings. Make a `Learner` with the same `save_dir` again, and it
carries on from the newest checkpoint. Its first line then ends with `carrying on from` and the
checkpoint's folder.

- `total_steps` is the run's total, not "this many more". `learn(1_000_000)` on a run that is
  already at 600,000 trains 400,000 more.
- You can change the learning rates, `ppo_epochs`, the batch sizes, `ppo_ent_coef` and
  `gae_gamma` between runs; it carries on and prints what changed.
- If you change the network size or the opponent, it refuses to carry on and says why.
- If you change what's inside `build_env`, such as the deck or the reward, it does **not**
  notice: it carries on training the old bot under the new setup. Use a new `save_dir` when you
  want a new bot.
- `resume=False` refuses a folder that already holds a run. It never deletes anything.

## Using a Trained Bot

`Learner.load_policy(folder)` loads a bot as a function that takes one seat's observation and
returns a move. `folder` can be a run's `save_dir` (its newest checkpoint), one checkpoint folder,
or a folder written by `learner.save(folder)`:

```py
from royalegym import make_env, play_battle
from royalelearn import Learner

bot = Learner.load_policy("runs/my_bot")
battle = play_battle(make_env(), blue=bot, red="random", save_to="my_bot_battle.msgpack")
```

By default the bot picks its moves at random, weighted by how much it likes each one, the
way it trained. `Learner.load_policy(folder, greedy=True)` always plays its favourite legal
move instead.

## More

For every setting there is, and how the trainer works inside, see the
[RoyaleLearn repository](https://github.com/RoyaleGym/RoyaleLearn).
