"""Scripted opponents to train against, ordered by how well they actually do.

WHY THESE EXIST
    The package shipped two: one that never plays a card and one that picks uniformly
    from legal moves. The project's own tutorial then walks a reader through a twelve
    line bot that three-crowns the random one in ninety six seconds, and says in as
    many words that the random opponent is a floor and not a wall. So the shipped
    training signal ran out in the first hour and there was no next rung.

    These are the next rungs. None of them is clever. They are the strategies a person
    describes in one sentence, which is the level a learner should be expected to pass
    through rather than to start above.

HOW THEY READ THE BOARD
    From the MASK, and nothing else. ``obs["mask_planes"]`` is [slot, y, x] in the
    acting player's own frame, so "forward" is the same direction for both seats and a
    scripted rule cannot accidentally learn the colour. Nothing here decodes the
    observation vector, on purpose: an opponent that indexes into the vector layout
    breaks silently the next time a field is added, and these are meant to be the
    stable thing you measure against.

WHAT THE LADDER ACTUALLY MEASURES, WHICH IS LESS THAN IT LOOKS
    A full round robin on MockEngine, both seats, seed 0. RE-MEASURED 2026-09-23 at 200
    games a pairing, after a calibration change doubled OVERTIME_S from 60 s to 120 s.
    That is worth stating rather than quietly restating the numbers: MockEngine reads
    calibration.json at RUNTIME and is pure Python, so the change moved every battle that
    reaches overtime with no rebuild and no commit in this repo. The 40-game table that
    stood here was measured under the old value::

        noop             vs everything        0-40   beaten by all five   (40 games)
        random           vs first-affordable 114-86  random is better     (57.0%)
        random           vs defend           109-91  too close to call
        random           vs push              99-101 too close to call
        random           vs patient           65-135 patient is better    (32.5%)
        first-affordable vs defend            84-99  too close to call    (17 draws)
        first-affordable vs push              96-101 too close to call
        first-affordable vs patient           75-122 patient is better    (38.1%)
        defend           vs push             101-98  too close to call
        defend           vs patient           79-120 patient is better    (39.7%)
        push             vs patient           65-135 patient is better    (32.5%)

    SIX orderings are established and no more: noop is beneath everything, patient is
    above all four of the middle strategies, and random beats first-affordable. The last
    is the awkward one and is kept because it is what the sample says: the naive policy
    beats the first-affordable heuristic, marginally (lower bound 50.1%), which is a
    reason to distrust first-affordable as a baseline rather than a reason to hide it.

    FORTY GAMES WAS NOT ENOUGH, and that is the substantive change. Three of the patient
    orderings sat inside the interval at 40 games and are comfortably outside it at 200 --
    patient beating random reads 14-26 at forty and 65-135 at two hundred. The old table
    called them "too close to call", which was true of that sample and read as "these are
    the same strength". A wider sample separated them.

    first-affordable, defend and push remain mutually indistinguishable AT 200 GAMES --
    they are different STRATEGIES of similar strength, not steps on a staircase, and this
    file does not pretend otherwise. ``tests/test_opponents.py`` asserts the orderings that
    hold, at the sample size that establishes each, and asserts that the middle three are
    NOT claimed to be ordered.

    THE SEAT GAPS ARE LARGE and are recorded because nothing here explains them: +12.0%
    (random/first-affordable), +13.0% (random/push), +13.5% (defend/patient). A seat gap
    that size on a scripted pairing is not obviously a property of the strategies, and it
    is the same shape as the seat asymmetry that keeps the ``rust-red-leads`` scenario
    marked in tests/test_reward_ground_truth.py. Whether they are one thing is unmeasured.

    That is worth reading twice if you are about to report that your bot improved.
    Three of these look like a progression, are described in increasing order of
    sophistication, and cannot be told apart by forty games. A difference you can name
    is not a difference you have measured.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .action import NOOP


def _planes(obs: dict[str, Any], mask: np.ndarray) -> np.ndarray | None:
    """The [slot, y, x] legality planes, or None if this observation has none."""
    planes = obs.get("mask_planes") if isinstance(obs, dict) else None
    if planes is None:
        return None
    planes = np.asarray(planes)
    return planes if planes.ndim == 3 and planes.any() else None


def _index(planes: np.ndarray, slot: int, y: int, x: int) -> int:
    """The flat action index for one cell of the planes.

    The layout is the parser's: no-op first, then slot-major, then row, then column.
    Spelled out here rather than imported so the arithmetic is visible at the one
    place a reader is likely to copy.
    """
    _, ny, nx = planes.shape
    return 1 + slot * ny * nx + y * nx + x


class FirstAffordableOpponent:
    """Play the lowest-numbered card you can afford, wherever it is legal, at once.

    The tutorial's bot, promoted into the package so a reader has it without
    retyping it. It is the cheapest possible non-trivial strategy: no positioning, no
    timing, no reading the board. It exists as the FLOOR that is not the random
    opponent -- something a learner must beat before any result means anything.
    """

    def act(self, obs: dict[str, Any], mask: np.ndarray, rng: np.random.Generator) -> int:
        del rng
        planes = _planes(obs, mask)
        if planes is None:
            return NOOP
        for slot in range(planes.shape[0]):
            cells = np.argwhere(planes[slot] > 0)
            if cells.size:
                y, x = cells[0]
                return _index(planes, slot, int(y), int(x))
        return NOOP


class DefendOpponent:
    """Only ever place on your own half, as near its middle as allowed.

    One sentence of strategy: never commit to the other side of the river, and stand
    where a unit can meet a push down either lane. The anchor is the centre column, a
    quarter of the way up the board: in front of your own king, just ahead of your
    princess towers. It loses to anything that builds a push, which is the shape a
    learner is supposed to discover.

    Until 2026-09-22 this bot took the lowest, then leftmost, legal cell of its own
    half. So did ``FirstAffordableOpponent``, whose first legal cell is the same
    back corner, and the two chose the same action on every one of 3 208 decisions
    compared on both engines. The ladder had one fewer rung than it listed.
    ``tests/test_opponents.py`` now requires every pair of rungs to differ.
    """

    def act(self, obs: dict[str, Any], mask: np.ndarray, rng: np.random.Generator) -> int:
        del rng
        planes = _planes(obs, mask)
        if planes is None:
            return NOOP
        _, ny, nx = planes.shape
        anchor_y, anchor_x2 = ny // 4, nx - 1  # x doubled, so the centre stays integer
        for slot in range(planes.shape[0]):
            cells = np.argwhere(planes[slot][: ny // 2] > 0)  # own half, own frame
            if cells.size:
                y, x = min(
                    cells.tolist(),
                    key=lambda c: (
                        4 * (c[0] - anchor_y) ** 2 + (2 * c[1] - anchor_x2) ** 2,
                        c[0],
                        c[1],
                    ),
                )
                return _index(planes, slot, int(y), int(x))
        return NOOP


class PushOpponent:
    """Place as far up the board as the rules allow, every time you can.

    The mirror of ``DefendOpponent`` and the other half of the question a bot is meant
    to answer for itself. Aggressive placement is not free: everything it plays lands
    far from its own towers and cannot come back to defend.
    """

    def act(self, obs: dict[str, Any], mask: np.ndarray, rng: np.random.Generator) -> int:
        del rng
        planes = _planes(obs, mask)
        if planes is None:
            return NOOP
        for slot in range(planes.shape[0]):
            cells = np.argwhere(planes[slot] > 0)
            if cells.size:
                y, x = max(cells.tolist(), key=lambda c: (c[0], c[1]))
                return _index(planes, slot, int(y), int(x))
        return NOOP


class PatientOpponent:
    """Hold until you can afford ``ready`` cards at once, then commit forward.

    The first rung with any timing in it. Playing every card the moment it is
    affordable is what the greedy bots do, and it means never having two things on the
    board together; waiting until several are legal at once approximates having saved
    elixir, using only what the mask says.

    That approximation is the interesting part and its limit: the mask says what is
    LEGAL, and a card becomes legal when it is affordable, so counting playable slots
    is a lower bound on elixir and not a reading of it. A cheap hand looks rich.
    """

    def __init__(self, ready: int = 3) -> None:
        if ready < 1:
            raise ValueError("ready must be at least 1 card")
        self.ready = ready

    def act(self, obs: dict[str, Any], mask: np.ndarray, rng: np.random.Generator) -> int:
        del rng
        planes = _planes(obs, mask)
        if planes is None:
            return NOOP
        playable = [s for s in range(planes.shape[0]) if planes[s].any()]
        if len(playable) < self.ready:
            return NOOP
        slot = playable[0]
        cells = np.argwhere(planes[slot] > 0)
        y, x = max(cells.tolist(), key=lambda c: (c[0], c[1]))
        return _index(planes, slot, int(y), int(x))


def ladder() -> tuple[tuple[str, Any], ...]:
    """The rungs, weakest first, as (name, opponent) pairs.

    Roughly by sophistication, which is NOT the same as by strength: see the module
    docstring for the round robin. Only two of the orderings are measured.

    A function rather than a constant because every entry is a fresh object: several
    of these keep no state today and one of them might tomorrow, and a shared instance
    handed to eight environments is the bug this package refuses elsewhere.
    """
    from .selfplay import NoopOpponent, RandomLegalOpponent

    return (
        ("noop", NoopOpponent()),
        ("random", RandomLegalOpponent(0.7)),
        ("first-affordable", FirstAffordableOpponent()),
        ("defend", DefendOpponent()),
        ("push", PushOpponent()),
        ("patient", PatientOpponent(ready=3)),
    )
