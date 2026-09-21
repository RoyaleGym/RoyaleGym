"""Terminal conditions: state -> (terminated, truncated).

``terminated`` means the MDP really ended (the value of the next state is 0);
``truncated`` means we stopped watching (bootstrap from the next state). Mixing
them up biases every value estimate near the cut, so conditions declare which
one they produce.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from .protocol import BattleState


class TerminalCondition(ABC):
    def reset(self, state: BattleState) -> None:
        """Called at episode start. Optional."""
        del state

    @abstractmethod
    def check(self, state: BattleState) -> tuple[bool, bool]:
        """Return (terminated, truncated). Called once per env step."""


class GameOverCondition(TerminalCondition):
    """The engine declared the battle over (king down, time, sudden death)."""

    def check(self, state: BattleState) -> tuple[bool, bool]:
        return state.game_over, False


class StepLimitCondition(TerminalCondition):
    """Truncate after ``max_steps`` env steps (decisions, not ticks)."""

    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self.steps = 0

    def reset(self, state: BattleState) -> None:
        self.steps = 0

    def check(self, state: BattleState) -> tuple[bool, bool]:
        self.steps += 1
        return False, self.steps >= self.max_steps


class TickLimitCondition(TerminalCondition):
    """Truncate once the engine clock reaches ``max_tick`` (absolute tick)."""

    def __init__(self, max_tick: int) -> None:
        self.max_tick = max_tick

    def check(self, state: BattleState) -> tuple[bool, bool]:
        return False, state.tick >= self.max_tick


class FirstCrownCondition(TerminalCondition):
    """Curriculum: terminate as soon as any crown changes hands since reset.

    ``terminated`` (not truncated): in this curriculum task the crown IS the
    outcome, so there is nothing to bootstrap from.
    """

    def __init__(self) -> None:
        self.start = (0, 0)

    def reset(self, state: BattleState) -> None:
        self.start = (state.players[0].crowns, state.players[1].crowns)

    def check(self, state: BattleState) -> tuple[bool, bool]:
        now = (state.players[0].crowns, state.players[1].crowns)
        return now != self.start or state.game_over, False


class AnyCondition(TerminalCondition):
    """OR of several conditions. Every child is checked every step (so counters advance)."""

    def __init__(self, conditions: Sequence[TerminalCondition]) -> None:
        self.conditions = list(conditions)

    def reset(self, state: BattleState) -> None:
        for c in self.conditions:
            c.reset(state)

    def check(self, state: BattleState) -> tuple[bool, bool]:
        term = trunc = False
        for c in self.conditions:
            a, b = c.check(state)
            term |= a
            trunc |= b
        # A real ending dominates a cut on the same step.
        return term, trunc and not term
