"""An episode can be named instead of counted, so a run can be resumed into.

WHY THIS EXISTS
    ``ClashParallelEnv.reset(seed=None)`` leaves its generator running, on purpose:
    consecutive episodes should not all be the same battle. The consequence is that
    with the autoreset passing no seed, episode 400 of a game is only reachable by
    playing 399 first. A learner that restores its own weights, optimizers and
    schedules byte for byte still diverges from the run it is continuing, and every
    divergence is in the environment.

    ``autoreset_seed_fn`` makes the nth episode of game g the one seeded ``fn(g, n)``,
    reachable directly. ``episode_ordinals`` is the counter to checkpoint and
    ``set_episode_ordinals`` puts it back.

WHAT IT CATCHES
    The hook not being wired into the autoreset at all, a seed that does not in fact
    determine the episode, a default that quietly changed, and an ordinal that does
    not survive a round trip through a checkpoint.

WHAT IT CANNOT CATCH
    Resuming MID-episode. Nothing here does that and the design does not offer it: a
    resumed run continues at the episode the counter names and the partially played
    one is dropped, which is what ``test_the_in_flight_episode_is_dropped_not_replayed``
    pins so the guarantee is not read as more than it is.

    Determinism of anything outside the env -- the policy's own sampling, the order a
    trainer consumes the batch. Those are the learner's to seed.

    Whether the Rust engine honours its seed the way the mock does. These run on the
    mock. The property is about who calls reset with what, which is this file's code.
"""

from __future__ import annotations

import numpy as np
import pytest

from royalegym.done_condition import GameOverCondition, StepLimitCondition
from royalegym.env import ClashParallelEnv, ClashSelfPlayVecEnv
from royalegym.mock_engine import MockEngine
from royalegym.state_mutator import DefaultStateMutator

DECK = [0, 3, 10, 14, 11, 13, 7, 9]
STEPS = 3  # every episode truncates after this many, so all games move in lockstep


def short_env(max_steps: int = STEPS) -> ClashParallelEnv:
    return ClashParallelEnv(
        engine=MockEngine(),
        state_mutator=DefaultStateMutator(decks=[DECK, list(reversed(DECK))]),
        termination_cond=GameOverCondition(),
        truncation_cond=StepLimitCondition(max_steps),
    )


def named(game: int, ordinal: int) -> int:
    """A seed that depends on both arguments, so a hook that drops one shows up."""
    return 1_000_003 * (game + 1) + 7919 * ordinal


def fingerprint(obs) -> bytes:
    """One hashable value for a batched observation."""
    return b"|".join(np.asarray(obs[k]).tobytes() for k in sorted(obs))


def noop(vec: ClashSelfPlayVecEnv) -> np.ndarray:
    return np.zeros(vec.num_envs, dtype=np.int64)


def run(vec: ClashSelfPlayVecEnv, steps: int) -> list[bytes]:
    return [fingerprint(vec.step(noop(vec))[0]) for _ in range(steps)]


# ---------------------------------------------------------------------------
# the default
# ---------------------------------------------------------------------------


class SeedSpy(ClashParallelEnv):
    """Records the seed every reset was called with, autoresets included."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seeds: list[int | None] = []

    def reset(self, seed=None, options=None):
        self.seeds.append(seed)
        return super().reset(seed=seed, options=options)


def spy_env() -> SeedSpy:
    return SeedSpy(
        engine=MockEngine(),
        state_mutator=DefaultStateMutator(decks=[DECK, list(reversed(DECK))]),
        termination_cond=GameOverCondition(),
        truncation_cond=StepLimitCondition(STEPS),
    )


def test_without_the_hook_the_autoreset_passes_no_seed_exactly_as_before() -> None:
    vec = ClashSelfPlayVecEnv(2, spy_env, viser=None)
    vec.reset()
    run(vec, 2 * STEPS)
    for env in vec.envs:
        assert env.seeds[0] is None, "reset() with no seed must stay unseeded"
        assert env.seeds[1:] == [None, None], f"autoreset seeds changed: {env.seeds}"
    vec.close()


def test_with_the_hook_the_autoreset_passes_the_named_seed() -> None:
    vec = ClashSelfPlayVecEnv(2, spy_env, viser=None, autoreset_seed_fn=named)
    vec.reset()
    run(vec, 2 * STEPS)
    for g, env in enumerate(vec.envs):
        assert env.seeds == [named(g, 0), named(g, 1), named(g, 2)], env.seeds
    # And the two games really were given different seeds, or a hook that ignored
    # its game argument would pass this file.
    assert vec.envs[0].seeds != vec.envs[1].seeds
    vec.close()


def test_an_explicit_seed_wins_and_still_consumes_an_ordinal() -> None:
    vec = ClashSelfPlayVecEnv(1, spy_env, viser=None, autoreset_seed_fn=named)
    vec.reset(seed=4242)
    assert vec.envs[0].seeds == [4242]
    assert vec.episode_ordinals == (1,), "the ordinal moves even when it was not used"
    run(vec, STEPS)
    assert vec.envs[0].seeds[1] == named(0, 1), "numbering keeps counting episode starts"
    vec.close()


# ---------------------------------------------------------------------------
# the property the hook exists for
# ---------------------------------------------------------------------------


def test_the_same_function_gives_the_same_episodes() -> None:
    a = ClashSelfPlayVecEnv(2, short_env, viser=None, autoreset_seed_fn=named)
    b = ClashSelfPlayVecEnv(2, short_env, viser=None, autoreset_seed_fn=named)
    first = [fingerprint(a.reset()[0]), *run(a, 3 * STEPS)]
    second = [fingerprint(b.reset()[0]), *run(b, 3 * STEPS)]
    assert first == second
    # Vacuity: consecutive episodes must actually differ, or every comparison in
    # this file is satisfied by an env that ignores its seed entirely.
    assert first[0] != first[STEPS], "episodes 0 and 1 are identical; the seed did nothing"
    a.close()
    b.close()


def test_a_run_resumes_row_for_row_from_the_ordinals_it_checkpointed() -> None:
    """The one that makes the resume guarantee true rather than half true."""
    a = ClashSelfPlayVecEnv(2, short_env, viser=None, autoreset_seed_fn=named)
    a.reset()
    run(a, 3 * STEPS)  # episodes 0-2 played, episode 3 in flight

    saved = a.episode_ordinals
    assert saved == (4, 4), "four episodes started per game by now"

    # The original run plays out the episode it was in and carries on.
    original = run(a, STEPS)[-1:] + run(a, STEPS)

    # A fresh object, told only the ordinals, continues the same row.
    b = ClashSelfPlayVecEnv(2, short_env, viser=None, autoreset_seed_fn=named)
    b.set_episode_ordinals(saved)
    resumed = [fingerprint(b.reset()[0]), *run(b, STEPS)]

    assert resumed == original
    assert len(set(original)) > 1, "a constant stream would make this pass by accident"
    a.close()
    b.close()


def test_without_the_hook_the_same_resume_replays_instead_of_continuing() -> None:
    """The gap the hook closes, measured -- this is what a run does today.

    The plant for the test above. A fresh worker given the same starting seed
    arrives at episode 0, not at the episode the run had reached, because with no
    seed at the autoreset the only route to the nth battle is the n-1 before it.
    """
    a = ClashSelfPlayVecEnv(2, short_env, viser=None)
    a.reset(seed=11)
    run(a, 3 * STEPS)
    original = run(a, STEPS)[-1:] + run(a, STEPS)

    b = ClashSelfPlayVecEnv(2, short_env, viser=None)
    resumed = [fingerprint(b.reset(seed=11)[0]), *run(b, STEPS)]

    assert resumed != original, (
        "an unseeded autoreset appeared to resume; either the episodes do not vary "
        "with the seed or the test above is not measuring what it claims"
    )
    a.close()
    b.close()


def test_the_in_flight_episode_is_dropped_not_replayed() -> None:
    """The limit of the guarantee, pinned so it is not read as more than it is.

    Resuming continues at the episode the counter names. The episode that was half
    played when the checkpoint was taken is not finished and not replayed -- its
    transitions were already in the original run's buffer.
    """
    a = ClashSelfPlayVecEnv(1, short_env, viser=None, autoreset_seed_fn=named)
    a.reset()
    run(a, STEPS + 1)  # one step into episode 1
    assert a.episode_ordinals == (2,)

    b = ClashSelfPlayVecEnv(1, short_env, viser=None, autoreset_seed_fn=named)
    b.set_episode_ordinals(a.episode_ordinals)
    first_after_resume = fingerprint(b.reset()[0])

    c = ClashSelfPlayVecEnv(1, short_env, viser=None, autoreset_seed_fn=named)
    c.set_episode_ordinals((1,))
    episode_one_from_its_start = fingerprint(c.reset()[0])

    assert first_after_resume != episode_one_from_its_start, (
        "resuming replayed the in-flight episode; the counter names the NEXT one"
    )
    a.close()
    b.close()
    c.close()


# ---------------------------------------------------------------------------
# the arguments
# ---------------------------------------------------------------------------


def test_a_function_that_is_not_callable_is_refused_at_construction() -> None:
    with pytest.raises(TypeError, match="fn\\(game_index, episode_ordinal\\)"):
        ClashSelfPlayVecEnv(1, short_env, viser=None, autoreset_seed_fn=7)


@pytest.mark.parametrize("bad", [1.5, "4", None, True], ids=["float", "str", "none", "bool"])
def test_a_function_returning_something_other_than_an_int_is_refused(bad) -> None:
    """``True`` included: bool is an int, and it would silently seed every episode 1."""
    vec = ClashSelfPlayVecEnv(1, short_env, viser=None, autoreset_seed_fn=lambda g, n: bad)
    with pytest.raises(TypeError, match="has to "):
        vec.reset()
    vec.close()


def test_a_numpy_integer_is_accepted_because_that_is_what_a_generator_returns() -> None:
    vec = ClashSelfPlayVecEnv(
        1, short_env, viser=None, autoreset_seed_fn=lambda g, n: np.int64(named(g, n))
    )
    vec.reset()
    run(vec, STEPS)
    assert vec.episode_ordinals == (2,)
    vec.close()


def test_set_episode_ordinals_checks_its_argument() -> None:
    vec = ClashSelfPlayVecEnv(2, short_env, viser=None, autoreset_seed_fn=named)
    with pytest.raises(ValueError, match="one per game"):
        vec.set_episode_ordinals([1])
    with pytest.raises(ValueError, match="cannot be negative"):
        vec.set_episode_ordinals([1, -1])
    vec.set_episode_ordinals([3, 9])
    assert vec.episode_ordinals == (3, 9)
    vec.close()


def test_the_ordinals_are_a_snapshot_not_a_live_handle() -> None:
    """A checkpoint holding a live list would record whatever the run did next."""
    vec = ClashSelfPlayVecEnv(1, short_env, viser=None, autoreset_seed_fn=named)
    vec.reset()
    saved = vec.episode_ordinals
    run(vec, STEPS)
    assert saved == (1,), "the snapshot moved with the run"
    assert vec.episode_ordinals == (2,)
    vec.close()
