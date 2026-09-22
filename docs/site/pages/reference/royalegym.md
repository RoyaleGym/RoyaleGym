# royalegym

The environment layer. This is the package you import to get a battle your bot can play in.

Everything on this page is generated from the docstrings in the code, so it cannot drift from
what the package actually does. If something here is wrong, the docstring is wrong.

If you are looking for where to start rather than what a name does, read
[Your first bot](../first-bot.md) and [Writing a reward function](../rewards.md) first.

## The environments

::: royalegym.env
    options:
      members:
        - ClashParallelEnv
        - ClashGymEnv
        - ClashSelfPlayVecEnv

## The five pieces you swap

These are the parts you write. Each one is a base class with a working default already shipped,
so you can replace one and leave the other four alone.

::: royalegym.reward

::: royalegym.obs

::: royalegym.action

::: royalegym.state_mutator

::: royalegym.done_condition

## The engines

`RustEngine` is the real one. `MockEngine` is a pure-Python stand-in that runs the whole API
without the Rust build, which is useful before you have built anything and useless for anything
about game fidelity.

::: royalegym.rust_engine

::: royalegym.mock_engine

## Recording and watching

::: royalegym.replay

::: royalegym.viser

## Self-play bookkeeping

::: royalegym.selfplay
