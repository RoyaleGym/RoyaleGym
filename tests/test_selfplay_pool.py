"""The opponent pool: Elo, head-to-head, PFSP sampling and the round trip through disk.

WHY THIS EXISTS
    It did not. ``OpponentPool``, ``CallableOpponent``, ``record_result``,
    ``pfsp_weights``, ``expected_score`` and ``PolicySnapshot`` appeared zero times in
    the whole suite, while ``record_result``'s own docstring said "the update is
    zero-sum: the pool's total Elo is conserved, which the tests check". No test
    checked it. A docstring that claims a test exists is worse than one that claims
    nothing, because it stops the next reader from looking.

WHAT IT CATCHES
    An Elo update that is not zero-sum, a rating that moves the wrong way, a PFSP
    weighting that prefers opponents you already beat, sampling that can return the
    learner or ignore its weights, a pool that loses ratings or history when it is
    saved and loaded, and eviction that throws away the wrong snapshot.

WHAT IT CANNOT CATCH
    Whether Elo is the right measure of a bot, or whether PFSP is the right curriculum.
    Both are choices, not properties.

    Anything about actually playing the games. The pool never opens a payload -- it
    stores a string and hands it back -- so what a snapshot IS remains the caller's
    business, and that gap is real: see the test at the bottom.
"""

from __future__ import annotations

import numpy as np
import pytest

from royalegym.selfplay import (
    CallableOpponent,
    OpponentPool,
    expected_score,
)


def pool_with(*ids: str, **kwargs) -> OpponentPool:
    pool = OpponentPool(**kwargs)
    for step, name in enumerate(ids, start=1):
        pool.add(name, step=step * 100)
    return pool


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------


def test_the_elo_update_is_zero_sum() -> None:
    """The claim record_result's docstring makes. Now it is checked.

    Not zero-sum and the pool inflates: every rating drifts up together, nobody's
    number means anything against a snapshot from last week, and the curriculum quietly
    stops preferring hard opponents.
    """
    pool = pool_with("learner", "a", "b")
    before = sum(s.elo for s in pool.snapshots.values())
    for score in (1.0, 0.0, 0.5, 1.0, 0.0):
        pool.record_result("learner", "a", score)
        pool.record_result("a", "b", score)
    after = sum(s.elo for s in pool.snapshots.values())
    assert after == pytest.approx(before), f"pool Elo moved by {after - before}"
    # Vacuity: the ratings really did move, or conservation is trivially true.
    assert pool.snapshots["learner"].elo != pool.initial_elo


def test_winning_raises_your_rating_and_lowers_theirs() -> None:
    pool = pool_with("learner", "a")
    elo_a, elo_b = pool.record_result("learner", "a", 1.0)
    assert elo_a > pool.initial_elo
    assert elo_b < pool.initial_elo
    assert pool.snapshots["learner"].games == 1
    assert pool.snapshots["a"].games == 1


def test_beating_someone_stronger_is_worth_more() -> None:
    """The point of Elo. Equal gains for unequal wins would make the number a game count."""
    weak = pool_with("learner", "a")
    weak.snapshots["a"].elo = 1000.0
    strong = pool_with("learner", "b")
    strong.snapshots["b"].elo = 1600.0
    gain_vs_weak = weak.record_result("learner", "a", 1.0)[0] - weak.initial_elo
    gain_vs_strong = strong.record_result("learner", "b", 1.0)[0] - strong.initial_elo
    assert gain_vs_strong > gain_vs_weak


def test_expected_score_is_symmetric_and_even_at_equal_ratings() -> None:
    assert expected_score(1200.0, 1200.0) == pytest.approx(0.5)
    assert expected_score(1400.0, 1200.0) + expected_score(1200.0, 1400.0) == pytest.approx(1.0)
    assert expected_score(1600.0, 1200.0) > 0.9


def test_a_score_that_is_not_a_result_is_refused() -> None:
    pool = pool_with("learner", "a")
    with pytest.raises(ValueError, match=r"0, 0\.5 or 1"):
        pool.record_result("learner", "a", 0.75)


# ---------------------------------------------------------------------------
# head-to-head and PFSP
# ---------------------------------------------------------------------------


def test_an_unplayed_matchup_is_an_even_prior() -> None:
    """Beta(1,1): no games means no opinion, not a zero win rate.

    Returning 0.0 would make every unplayed opponent look maximally hard and PFSP
    would chase whoever it had never met instead of whoever beats it.
    """
    assert pool_with("learner", "a").win_rate("learner", "a") == 0.5


def test_the_win_rate_moves_towards_the_results_without_reaching_certainty() -> None:
    pool = pool_with("learner", "a")
    for _ in range(10):
        pool.record_result("learner", "a", 1.0)
    rate = pool.win_rate("learner", "a")
    assert rate == pytest.approx(11 / 12), rate  # ten wins and the prior, not 1.0
    assert rate < 1.0, "ten wins in a row is not certainty"
    # The two directions are complements, so one player's confidence is the other's doubt.
    assert pool.win_rate("a", "learner") == pytest.approx(1.0 - rate)


def test_pfsp_prefers_the_opponents_you_lose_to() -> None:
    """The whole purpose. Weights that do not order by difficulty are a uniform sample
    with extra steps."""
    pool = pool_with("learner", "easy", "hard")
    for _ in range(8):
        pool.record_result("learner", "easy", 1.0)
        pool.record_result("learner", "hard", 0.0)
    w = pool.pfsp_weights("learner", ["easy", "hard"])
    assert w.sum() == pytest.approx(1.0)
    assert w[1] > w[0], f"hard {w[1]} should outweigh easy {w[0]}"


def test_the_variance_weighting_prefers_a_coin_flip() -> None:
    pool = pool_with("learner", "even", "hopeless")
    for _ in range(8):
        pool.record_result("learner", "even", 1.0)
        pool.record_result("learner", "even", 0.0)
        pool.record_result("learner", "hopeless", 0.0)
    w = pool.pfsp_weights("learner", ["even", "hopeless"], weighting="variance")
    assert w[0] > w[1]


def test_an_unknown_weighting_is_refused_rather_than_silently_uniform() -> None:
    pool = pool_with("learner", "a")
    with pytest.raises(ValueError, match="unknown PFSP weighting"):
        pool.pfsp_weights("learner", ["a"], weighting="vibes")


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["pfsp", "uniform", "latest"])
def test_sampling_never_returns_the_learner(strategy: str) -> None:
    """Self-play against yourself-right-now is not self-play, it is a mirror."""
    pool = pool_with("learner", "a", "b")
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert pool.sample(rng, strategy=strategy).snapshot_id != "learner"


def test_latest_is_the_newest_snapshot_by_step() -> None:
    pool = pool_with("learner", "old", "new")
    assert pool.sample(np.random.default_rng(0), strategy="latest").snapshot_id == "new"


def test_pfsp_sampling_actually_follows_its_weights() -> None:
    """Draw many and check the hard opponent comes up more.

    Without this, ``sample`` could compute weights and ignore them -- the weights test
    above would still pass, and the curriculum would be uniform.
    """
    pool = pool_with("learner", "easy", "hard")
    for _ in range(12):
        pool.record_result("learner", "easy", 1.0)
        pool.record_result("learner", "hard", 0.0)
    rng = np.random.default_rng(7)
    drawn = [pool.sample(rng, strategy="pfsp").snapshot_id for _ in range(400)]
    assert drawn.count("hard") > drawn.count("easy") * 2


def test_an_empty_pool_says_so() -> None:
    with pytest.raises(LookupError, match="no opponents"):
        OpponentPool().sample(np.random.default_rng(0))


def test_an_unknown_strategy_is_refused() -> None:
    pool = pool_with("learner", "a")
    with pytest.raises(ValueError, match="unknown strategy"):
        pool.sample(np.random.default_rng(0), strategy="best")


# ---------------------------------------------------------------------------
# registry and persistence
# ---------------------------------------------------------------------------


def test_registering_the_same_id_twice_is_refused() -> None:
    pool = pool_with("a")
    with pytest.raises(KeyError, match="already registered"):
        pool.add("a", step=1)


def test_eviction_drops_the_oldest_and_never_the_learner() -> None:
    pool = OpponentPool(max_size=2)
    pool.add("learner", step=0)
    for i in range(4):
        pool.add(f"s{i}", step=i * 10)
    assert "learner" in pool.snapshots
    kept = sorted(s for s in pool.snapshots if s != "learner")
    assert kept == ["s2", "s3"], kept


def test_a_saved_pool_comes_back_with_its_ratings_and_its_history(tmp_path) -> None:
    pool = pool_with("learner", "a", "b")
    for _ in range(5):
        pool.record_result("learner", "a", 1.0)
        pool.record_result("learner", "b", 0.0)
    path = tmp_path / "pool.json"
    pool.save(path)

    back = OpponentPool.load(path)  # a classmethod: it returns a pool, it does not fill one
    assert {s: round(v.elo, 9) for s, v in back.snapshots.items()} == {
        s: round(v.elo, 9) for s, v in pool.snapshots.items()
    }
    assert back.win_rate("learner", "a") == pool.win_rate("learner", "a")
    assert back.win_rate("learner", "b") == pool.win_rate("learner", "b")
    # The head-to-head really was non-trivial, or this compares two 0.5s.
    assert back.win_rate("learner", "a") != back.win_rate("learner", "b")


# ---------------------------------------------------------------------------
# the gap between a snapshot and something you can play
# ---------------------------------------------------------------------------


def test_a_sampled_snapshot_is_not_yet_an_opponent_and_callable_bridges_it() -> None:
    """``sample`` hands back a record, not something with ``act``. That is the design.

    The pool never opens a payload, so it cannot build a policy for you. What it can do
    is name one, and ``CallableOpponent`` is the adapter from your function to the
    ``Opponent`` the envs take. This test exists to pin the seam, because "PFSP picked
    you an opponent" and "you have an opponent" are different sentences and the
    distance between them is the caller's to cover.
    """
    pool = pool_with("learner", "a")
    snap = pool.sample(np.random.default_rng(0))
    assert not hasattr(snap, "act")
    assert snap.payload is None  # nothing was stored, and the pool invented nothing

    policies = {"a": lambda obs, mask: int(np.flatnonzero(mask)[0])}
    opponent = CallableOpponent(policies[snap.snapshot_id])
    mask = np.zeros(10, dtype=np.int8)
    mask[3] = 1
    assert opponent.act({}, mask, np.random.default_rng(0)) == 3


def test_a_frozen_policy_cannot_smuggle_an_illegal_action_through() -> None:
    """An older policy meeting a newer mask picks a move that is no longer legal.

    It becomes a no-op, which the env logs, rather than reaching the engine.
    """
    opponent = CallableOpponent(lambda obs, mask: 5)
    mask = np.zeros(10, dtype=np.int8)
    mask[3] = 1
    assert opponent.act({}, mask, np.random.default_rng(0)) != 5
