# royalelearn

The training harness.

!!! warning "New, and barely tested"

    The training loop closed on 2026-09-22. `python -m royalelearn train` runs end to end, and
    it needs the torch extra. The only run so far was three iterations long, so treat anything
    below that touches the loop itself as young code. See [The learner](../pieces/learner.md)
    for what that means in practice.

Everything on this page is generated from the docstrings in the code, so it shows what exists
rather than what is planned. A name that is missing from this page has not been written yet.

Some of these need torch, which is an optional extra: `pip install -e "RoyaleLearn[torch]"`. The
configuration tree, the run identity and the rollout workers are deliberately torch-free, and
that is a property the package is tested for rather than an accident.

## Configuration and identity

You write a config, and the run's identity is computed from it so that a resume can be checked
against the run it is resuming.

::: royalelearn.config

::: royalelearn.identity

## Rewards

Where the shipped reward function lives, and the shape your own one takes.

::: royalelearn.rewards

## Collecting experience

::: royalelearn.rollout

::: royalelearn.obs_layout

## The learner

::: royalelearn.learn

## Rating and checkpoints

::: royalelearn.ladder

::: royalelearn.checkpoint

## Seeding and determinism

Every random draw in a run descends from one seed tree, so a resumed run lands on the original's
curve.

::: royalelearn.seeding

::: royalelearn.determinism

## Metrics

::: royalelearn.metrics

## The base classes

::: royalelearn.api
