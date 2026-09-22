"""Is bot A better than bot B? Answered with an error bar, on both seats.

WHY THIS IS NOT A LOOP YOU WRITE YOURSELF
    It is, and people do, and it is the loop that goes wrong quietly. Three ways:

    THE SEAT. This engine reproduces the real game's frame-keyed behaviour, so the two
    seats are not interchangeable -- the shipped pathfinder can send a Red unit and its
    rotated Blue twin along different equal-cost routes. Play A as Blue every game and
    part of what you measure is the colour. ``evaluate`` plays every matchup twice,
    once each way round, and reports the halves separately so a seat effect is visible
    rather than baked into the total.

    THE ERROR BAR. 100 games decided 55-45 is not evidence that A is better: the
    interval covers a coin. A bare win rate invites the reader to believe a difference
    the sample cannot support, and the whole point of asking is to decide something.
    ``MatchResult.interval`` is a Wilson interval, and ``better`` is None until it
    clears 0.5.

    THE DRAWS, AND THE GAMES THAT NEVER FINISHED. Overtime can end level, and counting
    a draw as half a win silently changes the question from "does A win more" to "does
    A score more" -- both numbers are here and they are named differently. Worse, a
    battle the step limit cut short is not a draw at all, and scoring it as one drags
    every comparison towards 50% in proportion to how short the limit was. Those are
    counted as ``undecided`` and left out of both rates.

WHAT THIS DOES NOT DO
    Train anything, own a pool, or open a checkpoint. It takes two ``Opponent`` objects
    -- anything with ``act(obs, mask, rng)`` -- and plays them. ``OpponentPool`` is
    where results are recorded over time; this answers one question about one pair.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .env import AGENTS, ClashParallelEnv
from .protocol import Winner
from .selfplay import Opponent

#: Anything that builds a fresh environment. A FACTORY, because an environment carries
#: per-battle state and evaluation plays many battles.
EnvBuilder = Callable[[], ClashParallelEnv]


@dataclass(frozen=True)
class SeatResult:
    """One side of the pairing: A's record while A sat in one particular seat."""

    seat: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    undecided: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws + self.undecided


@dataclass(frozen=True)
class MatchResult:
    """What came of playing A against B.

    ``win_rate`` counts only decided games. ``score_rate`` counts a draw as half, which
    is the Elo convention -- they are different questions and a single "win rate" that
    quietly means one of them is how a comparison becomes an argument.
    """

    games: int
    wins: int
    losses: int
    draws: int
    undecided: int = 0
    by_seat: tuple[SeatResult, ...] = ()
    mean_ticks: float = 0.0
    mean_crown_margin: float = 0.0
    confidence: float = 0.95
    names: tuple[str, str] = ("A", "B")
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def decided(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        """Share of DECIDED games A won. 0.5 when nothing was decided."""
        return self.wins / self.decided if self.decided else 0.5

    @property
    def score_rate(self) -> float:
        """A's score per FINISHED game with a draw worth half. The Elo convention.

        Undecided games are left out rather than counted as halves. A battle the step
        limit cut short is not a draw -- nobody held on for it -- and scoring it like
        one drags every comparison towards 50% in proportion to how short the limit
        was, which reads as "these bots are evenly matched" and means "you did not let
        them finish".
        """
        finished = self.wins + self.losses + self.draws
        return (self.wins + 0.5 * self.draws) / finished if finished else 0.5

    @property
    def interval(self) -> tuple[float, float]:
        """Wilson score interval for ``win_rate``, at ``confidence``.

        Wilson rather than the normal approximation because the normal one is wrong
        exactly where it matters: few games, or a rate near 0 or 1, where it can hand
        back a bound below zero and make a sweep look conclusive.
        """
        return wilson(self.wins, self.decided, self.confidence)

    @property
    def better(self) -> str | None:
        """The name of the better bot, or None if this many games cannot say.

        None is the honest answer far more often than people expect, and it is the
        answer a bare percentage hides.
        """
        low, high = self.interval
        if low > 0.5:
            return self.names[0]
        if high < 0.5:
            return self.names[1]
        return None

    @property
    def seat_gap(self) -> float:
        """A's win rate as the first seat minus as the second. 0 if it played both alike.

        A large gap means the measurement is partly about the colour, whatever the
        totals say.
        """
        if len(self.by_seat) != 2:
            return 0.0
        rates = []
        for s in self.by_seat:
            decided = s.wins + s.losses
            rates.append(s.wins / decided if decided else 0.5)
        return rates[0] - rates[1]

    def summary(self) -> str:
        low, high = self.interval
        a, b = self.names
        verdict = f"{self.better} is better" if self.better else "too close to call"
        unfinished = f", {self.undecided} unfinished" if self.undecided else ""
        return (
            f"{a} vs {b}: {self.wins}-{self.losses}-{self.draws}{unfinished} "
            f"over {self.games} games. "
            f"win rate {self.win_rate:.1%} ({low:.1%} to {high:.1%} at "
            f"{self.confidence:.0%}) -- {verdict}. "
            f"seat gap {self.seat_gap:+.1%}, mean {self.mean_ticks:.0f} ticks."
        )


def wilson(wins: int, games: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, clamped to [0, 1]."""
    if games <= 0:
        return (0.0, 1.0)
    z = _z_for(confidence)
    p = wins / games
    denom = 1.0 + z * z / games
    centre = (p + z * z / (2 * games)) / denom
    spread = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _z_for(confidence: float) -> float:
    """Two-sided normal quantile. A small table, because scipy is not a dependency."""
    table = {0.80: 1.281552, 0.90: 1.644854, 0.95: 1.959964, 0.98: 2.326348, 0.99: 2.575829}
    if confidence in table:
        return table[confidence]
    raise ValueError(f"confidence must be one of {sorted(table)}, not {confidence}")


def play_one(
    env: ClashParallelEnv,
    blue: Opponent,
    red: Opponent,
    seed: int,
) -> tuple[Winner | None, int, int]:
    """Play one battle. Returns (winner, ticks, crown margin from BLUE's side).

    ``winner`` is None when the episode was cut short rather than decided, which a
    truncation is: there is no winner, and pretending there is one is how a step limit
    becomes a result.
    """
    players = {"blue": blue, "red": red}
    rngs = {a: np.random.default_rng(seed * 2 + i) for i, a in enumerate(AGENTS)}
    obs, _ = env.reset(seed=seed)
    truncated_out = False
    while env.agents:
        actions = {
            a: int(players[a].act(obs[a], obs[a]["action_mask"], rngs[a])) for a in env.agents
        }
        obs, _, terminated, truncated, _ = env.step(actions)
        truncated_out = any(truncated.values()) and not any(terminated.values())
    state = env.battle_state
    margin = state.players[0].crowns - state.players[1].crowns
    winner = None if truncated_out or not state.game_over else state.winner
    return winner, state.tick, margin


def evaluate(
    a: Opponent,
    b: Opponent,
    env_fn: EnvBuilder,
    *,
    games: int = 100,
    seed: int = 0,
    confidence: float = 0.95,
    names: tuple[str, str] = ("A", "B"),
) -> MatchResult:
    """Play ``a`` against ``b`` and say who is better, or that you cannot tell yet.

    ``games`` is rounded UP to an even number and split evenly between the two seat
    assignments, because an odd split hands one bot more games in the stronger seat and
    the difference turns up in the total as if it were skill.

    Every battle is seeded from ``seed``, so the same call gives the same answer, and
    the two seat halves use the SAME seeds -- the same battle played both ways round,
    which removes the variance between pairings from the comparison rather than
    averaging over it.
    """
    if games < 2:
        raise ValueError("evaluate needs at least 2 games: one for each seat assignment")
    per_seat = (games + 1) // 2
    env = env_fn()
    seats = []
    wins = losses = draws = undecided = 0
    ticks: list[int] = []
    margins: list[int] = []

    for a_is_blue in (True, False):
        seat_wins = seat_losses = seat_draws = seat_undecided = 0
        for g in range(per_seat):
            blue, red = (a, b) if a_is_blue else (b, a)
            winner, tick, margin = play_one(env, blue, red, seed=seed + g)
            ticks.append(tick)
            margins.append(margin if a_is_blue else -margin)
            if winner is None:
                seat_undecided += 1  # cut short, not drawn: no one earned this
            elif winner == Winner.DRAW:
                seat_draws += 1
            elif (winner == Winner.BLUE) == a_is_blue:
                seat_wins += 1
            else:
                seat_losses += 1
        seats.append(
            SeatResult(
                seat="blue" if a_is_blue else "red",
                wins=seat_wins,
                losses=seat_losses,
                draws=seat_draws,
                undecided=seat_undecided,
            )
        )
        wins += seat_wins
        losses += seat_losses
        draws += seat_draws
        undecided += seat_undecided

    env.close()
    return MatchResult(
        games=2 * per_seat,
        wins=wins,
        losses=losses,
        draws=draws,
        undecided=undecided,
        by_seat=tuple(seats),
        mean_ticks=sum(ticks) / len(ticks),
        mean_crown_margin=sum(margins) / len(margins),
        confidence=confidence,
        names=names,
        meta={"seed": seed, "games_per_seat": per_seat},
    )
