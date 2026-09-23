"""Comparing two bots: the seat split, the error bar, and what counts as a result.

WHY THIS EXISTS
    "Is bot A better than bot B" had no answer in this package. The Elo bookkeeping
    shipped, and the loop that produces a result for it did not, so everyone was
    expected to write their own -- and the hand-written version goes wrong in the same
    three ways every time: it plays A in one seat, it reports a bare percentage, and it
    counts a battle the step limit cut short as a draw.

WHAT IT CATCHES
    An uneven seat split, a result that changes when nothing random should have, a
    verdict declared on a sample that cannot support it, a Wilson interval that is not
    one, and an unfinished battle being scored as half a win.

WHAT IT CANNOT CATCH
    Whether the bots are any good, or whether the environment is a fair test of them.
    And it uses MockEngine: this is about the counting, not about the game.
"""

from __future__ import annotations

import numpy as np
import pytest

from _decks import MIXED_DECK, card_ids
from royalegym import (
    ClashParallelEnv,
    DefaultStateMutator,
    MockEngine,
    NoopOpponent,
    RandomLegalOpponent,
)
from royalegym.done_condition import GameOverCondition, StepLimitCondition
from royalegym.evaluate import MatchResult, evaluate, wilson

DECK = card_ids(MIXED_DECK, MockEngine())


def env_fn(max_steps: int = 700):
    def build() -> ClashParallelEnv:
        return ClashParallelEnv(
            engine=MockEngine(),
            state_mutator=DefaultStateMutator(decks=[DECK, list(reversed(DECK))]),
            termination_cond=GameOverCondition(),
            truncation_cond=StepLimitCondition(max_steps),
        )

    return build


class AlwaysFirstCard:
    """Deterministic and weak: the lowest legal action that is not the no-op.

    Skipping the no-op matters. Index 0 IS the no-op, so "the lowest legal action"
    never plays a card, every battle ends 0-0 at the full time, and a comparison
    against a bot that also does nothing is six draws. That is what the first version
    of this fixture did, and the vacuity guard in the mirror test below is what caught
    it -- the mirror property held perfectly over a set of games in which nothing
    happened.
    """

    def act(self, obs, mask, rng):
        del obs, rng
        legal = np.flatnonzero(mask)
        legal = legal[legal != 0]
        return int(legal[0]) if legal.size else 0


# ---------------------------------------------------------------------------
# the seat
# ---------------------------------------------------------------------------


def test_both_seats_get_the_same_number_of_games() -> None:
    """The split is the whole reason this is not a for loop.

    The two seats are not interchangeable in this engine, so an uneven split puts one
    bot in the stronger chair more often and the difference arrives in the total
    looking like skill.
    """
    result = evaluate(AlwaysFirstCard(), NoopOpponent(), env_fn(), games=6, seed=0)
    assert len(result.by_seat) == 2
    assert {s.seat for s in result.by_seat} == {"blue", "red"}
    assert result.by_seat[0].games == result.by_seat[1].games == 3
    assert result.games == 6


def test_an_odd_number_of_games_is_rounded_up_not_split_unevenly() -> None:
    result = evaluate(AlwaysFirstCard(), NoopOpponent(), env_fn(), games=5, seed=0)
    assert result.games == 6
    assert result.by_seat[0].games == result.by_seat[1].games


def test_fewer_than_two_games_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 2 games"):
        evaluate(NoopOpponent(), NoopOpponent(), env_fn(), games=1)


def test_swapping_the_two_bots_mirrors_the_result() -> None:
    """A beating B by n is B losing to A by n. If it is not, the seats are not balanced.

    The strongest check in this file: it compares two whole evaluations rather than a
    number against itself, and an asymmetry anywhere in the seat handling breaks it.
    """
    forward = evaluate(AlwaysFirstCard(), NoopOpponent(), env_fn(), games=6, seed=3)
    reverse = evaluate(NoopOpponent(), AlwaysFirstCard(), env_fn(), games=6, seed=3)
    assert forward.wins == reverse.losses
    assert forward.losses == reverse.wins
    assert forward.draws == reverse.draws
    assert forward.undecided == reverse.undecided
    # Vacuity: a pairing where nothing was ever decided would satisfy all of that.
    assert forward.decided > 0


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_the_same_seed_gives_the_same_answer() -> None:
    kwargs = dict(games=4, seed=11)
    first = evaluate(RandomLegalOpponent(0.5), NoopOpponent(), env_fn(), **kwargs)
    again = evaluate(RandomLegalOpponent(0.5), NoopOpponent(), env_fn(), **kwargs)
    assert (first.wins, first.losses, first.draws) == (again.wins, again.losses, again.draws)
    assert first.mean_ticks == again.mean_ticks


# ---------------------------------------------------------------------------
# what counts as a result
# ---------------------------------------------------------------------------


def test_a_battle_the_step_limit_cut_short_is_undecided_not_drawn() -> None:
    """The one that quietly ruins comparisons.

    With a short limit nothing finishes. Counted as draws, every pair of bots looks
    evenly matched, and the shorter the limit the more evenly matched they look.
    """
    result = evaluate(AlwaysFirstCard(), NoopOpponent(), env_fn(max_steps=5), games=4, seed=0)
    assert result.undecided == 4
    assert result.draws == 0
    assert result.decided == 0
    assert result.better is None
    # And the rates do not pretend: nothing finished, so there is nothing to average.
    assert result.score_rate == 0.5


def test_a_decided_game_is_counted_for_the_winner() -> None:
    """Vacuity guard for the file: some pairing really does produce wins."""
    result = evaluate(RandomLegalOpponent(0.5), NoopOpponent(), env_fn(), games=6, seed=1)
    assert result.decided > 0
    assert result.wins + result.losses + result.draws + result.undecided == result.games


# ---------------------------------------------------------------------------
# the error bar
# ---------------------------------------------------------------------------


def test_a_narrow_sample_will_not_name_a_winner() -> None:
    """Two games won 1-1 is not evidence, and the verdict says so."""
    result = MatchResult(games=2, wins=1, losses=1, draws=0, names=("a", "b"))
    low, high = result.interval
    assert low < 0.5 < high
    assert result.better is None


def test_a_wide_enough_margin_does_name_one() -> None:
    result = MatchResult(games=100, wins=70, losses=30, draws=0, names=("a", "b"))
    assert result.better == "a"
    assert MatchResult(games=100, wins=30, losses=70, draws=0, names=("a", "b")).better == "b"


def test_the_same_percentage_on_a_smaller_sample_is_not_a_verdict() -> None:
    """70% of 10 and 70% of 100 are the same number and not the same evidence.

    This is the property a bare win rate hides, and the reason the interval is on the
    result object rather than left to the reader.
    """
    assert MatchResult(games=10, wins=7, losses=3, draws=0, names=("a", "b")).better is None
    assert MatchResult(games=100, wins=70, losses=30, draws=0, names=("a", "b")).better == "a"


def test_wilson_stays_inside_zero_and_one_at_the_extremes() -> None:
    """Where the normal approximation goes negative and makes a sweep look conclusive."""
    low, high = wilson(10, 10)
    assert 0.0 < low < 1.0
    assert high == pytest.approx(1.0)
    low, high = wilson(0, 10)
    assert low == pytest.approx(0.0)
    assert 0.0 < high < 1.0


def test_wilson_matches_a_worked_value() -> None:
    """A textbook case, so the formula is checked against arithmetic done elsewhere."""
    low, high = wilson(70, 100, 0.95)
    assert low == pytest.approx(0.6041, abs=0.001)
    assert high == pytest.approx(0.7817, abs=0.001)


def test_no_games_is_total_ignorance_not_a_coin_flip() -> None:
    assert wilson(0, 0) == (0.0, 1.0)


def test_an_unsupported_confidence_is_refused_rather_than_approximated() -> None:
    with pytest.raises(ValueError, match="confidence must be one of"):
        wilson(5, 10, 0.975)


# ---------------------------------------------------------------------------
# the two rates are different questions
# ---------------------------------------------------------------------------


def test_the_win_rate_and_the_score_rate_answer_different_questions() -> None:
    """One counts wins among decided games; the other scores a draw as half."""
    result = MatchResult(games=10, wins=4, losses=2, draws=4)
    assert result.win_rate == pytest.approx(4 / 6)
    assert result.score_rate == pytest.approx(6 / 10)


def test_the_summary_says_when_it_cannot_tell() -> None:
    text = MatchResult(games=10, wins=5, losses=5, draws=0, names=("a", "b")).summary()
    assert "too close to call" in text
    assert "a vs b" in text
