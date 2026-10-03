# Reinforcement Learning Terms

The words you'll meet while training a bot, in plain language. You don't need any of this to
run the Quick Start, but it helps when you start changing settings.

## The Basics

**Reinforcement learning.** A way to teach a program by trial and error. It tries moves, gets
rewarded or charged for what happens, and slowly learns which moves pay off.

**Agent.** The thing that learns and acts: your bot. A battle has two agents, `"blue"` and `"red"`.

**Environment.** The world the agent acts in. Here, a Clash Royale battle, run by RoyaleSim and
set up by RoyaleGym.

**Step.** One turn of the loop: the agent sees the battle, picks a move, and the battle moves on
by one decision (half a second, by default).

**Episode.** One whole run from start to end. Here, one battle.

**Observation.** What the agent sees at a step: the numbers an [observation builder](../clash-royale/configuration-objects/observation-builders.md)
makes from the battle.

**Action.** The move the agent picks: wait, or play a card on a tile.

**Action mask.** Which actions are legal right now. The agent only picks from these.

**Reward.** The number the agent gets after each step. It learns to make the total as big as it
can. See [Reward Functions](../clash-royale/configuration-objects/reward-functions.md).

**Return.** The total reward from now to the end of the battle, with rewards far in the future
counting a little less than rewards soon.

**Termination and truncation.** Two ways a battle can stop. Terminated: it really ended.
Truncated: it was cut short, so what would have come next still counts. See
[Done Conditions](../clash-royale/configuration-objects/done-conditions.md).

## The Bot's Brain

**Policy.** The part of the bot that picks moves. It's a neural network: observation in, a
chance for each move out.

**Critic** (or value function). A second part that guesses how much reward is still to come from
here. The trainer uses its guesses to judge whether a move turned out better or worse than
expected.

**Neural network.** A big function with many adjustable numbers (weights). Training adjusts the
weights. `trunk_channels` and `trunk_blocks` set how big RoyaleLearn's network is.

**Checkpoint.** A saved copy of the network and the trainer's state, so training can stop and
carry on later.

## The Trainer

**PPO** (Proximal Policy Optimization). The training method RoyaleLearn uses.
It collects a batch of experience, then nudges the policy toward moves that did better than
expected, but only a little at a time, so one lucky batch can't wreck it.

**Timestep.** One decision of one agent. `steps` in the console counts these.

**Steps per update** (`steps_per_update`). How many decisions the trainer collects before it
learns from them.

**Batch and minibatch** (`ppo_batch_size`, `ppo_minibatch_size`). The decisions the trainer
learns from in one update, and the smaller pieces it cuts them into for each adjustment. A
bigger minibatch uses more of your graphics card's memory.

**Epoch** (`ppo_epochs`). One pass over the batch. More epochs squeeze more out of each batch,
but too many make the bot overfit to that batch.

**Learning rate** (`policy_lr`, `critic_lr`). How big each adjustment is. Too big and training
becomes unstable; too small and it crawls.

**Entropy.** How spread out the bot's choices are. High entropy: it tries many different moves.
Low: it always picks its favourite. `ppo_ent_coef` rewards some entropy, so the bot keeps
exploring instead of settling too early.

**Gamma** (`gae_gamma`). How much the bot cares about rewards far in the future compared to
rewards soon. Close to 1 means it plans further ahead.

## Training Against Others

**Opponent.** Whoever sits in the other seat. See [Opponents](../clash-royale/configuration-objects/opponents.md).

**Self-play.** Training against copies of your own bot, so it always has an opponent at its own
level.

**Reward shaping.** Small extra rewards that give hints along the way, like paying for tower
damage, on top of the reward you really care about: winning.

**Overfitting.** Getting good at exactly what it practised, and no better at anything else. A bot
that only ever plays the random opponent can overfit to it.
