"""Done conditions: state -> bool, in one of two roles.

A ``DoneCondition`` answers one question per env step: is this episode over?
Which *kind* of over is decided by the slot the env receives it in, the same
split RLGym v2 makes:

* ``termination_cond``: the MDP really ended (the value of the next state is 0):
  a king tower fell, the clock ran out, a curriculum objective was met.
* ``truncation_cond``: we stopped watching (bootstrap from the next state): a
  step or tick budget was spent.

Mixing them up biases every value estimate near the cut, so the shipped
conditions declare their role by subclassing ``TerminationCondition`` or
``TruncationCondition``, and the env refuses a condition passed in the wrong
slot. Both are thin subclasses of ``DoneCondition`` that add nothing but the
name; ``AnyCondition`` / ``AllCondition`` stay role-neutral so they can combine
conditions in either slot.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from .protocol import BattleState


class DoneCondition(ABC):
    def reset(self, state: BattleState) -> None:
        """Called at episode start. Optional."""
        del state

    def config(self) -> dict[str, object]:
        """Constructor state, JSON-able, for ``ClashParallelEnv.config()``."""
        return {}

    @abstractmethod
    def is_done(self, state: BattleState) -> bool:
        """Whether the episode is over as of ``state``. Called once per env step."""


class TerminationCondition(DoneCondition, ABC):
    """A ``DoneCondition`` for the termination slot: the episode's outcome is decided."""


class TruncationCondition(DoneCondition, ABC):
    """A ``DoneCondition`` for the truncation slot: the episode is cut, not decided."""


class GameOverCondition(TerminationCondition):
    """The engine declared the battle over (king down, time, sudden death)."""

    def is_done(self, state: BattleState) -> bool:
        return state.game_over


class StepLimitCondition(TruncationCondition):
    """Truncate after ``max_steps`` env steps (decisions, not ticks)."""

    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self.steps = 0

    def config(self) -> dict[str, object]:
        return {"max_steps": self.max_steps}

    def reset(self, state: BattleState) -> None:
        self.steps = 0

    def is_done(self, state: BattleState) -> bool:
        self.steps += 1
        return self.steps >= self.max_steps


class TickLimitCondition(TruncationCondition):
    """Truncate once the engine clock reaches ``max_tick`` (absolute tick)."""

    def __init__(self, max_tick: int) -> None:
        self.max_tick = max_tick

    def config(self) -> dict[str, object]:
        return {"max_tick": self.max_tick}

    def is_done(self, state: BattleState) -> bool:
        return state.tick >= self.max_tick


class FirstCrownCondition(TerminationCondition):
    """Curriculum: end as soon as any crown changes hands since reset.

    A termination, not a truncation: in this curriculum task the crown IS the
    outcome, so there is nothing to bootstrap from.
    """

    def __init__(self) -> None:
        self.start = (0, 0)

    def reset(self, state: BattleState) -> None:
        self.start = (state.players[0].crowns, state.players[1].crowns)

    def is_done(self, state: BattleState) -> bool:
        now = (state.players[0].crowns, state.players[1].crowns)
        return now != self.start or state.game_over


class _CompositeCondition(DoneCondition, ABC):
    """Shared plumbing of ``AnyCondition`` / ``AllCondition``."""

    def __init__(self, conditions: Sequence[DoneCondition]) -> None:
        self.conditions = list(conditions)

    def config(self) -> dict[str, object]:
        return {
            "conditions": [
                {"class": type(c).__name__, "params": c.config()} for c in self.conditions
            ]
        }


class AnyCondition(_CompositeCondition):
    """OR of several conditions. Every child is checked every step (so counters advance)."""

    def reset(self, state: BattleState) -> None:
        for c in self.conditions:
            c.reset(state)

    def is_done(self, state: BattleState) -> bool:
        done = False
        for c in self.conditions:
            done |= c.is_done(state)
        return done


class AllCondition(_CompositeCondition):
    """AND of several conditions. Every child is checked every step (so counters advance)."""

    def reset(self, state: BattleState) -> None:
        for c in self.conditions:
            c.reset(state)

    def is_done(self, state: BattleState) -> bool:
        done = True
        for c in self.conditions:
            done &= c.is_done(state)
        return done


# Kept for callers written before the rename.
TerminalCondition = DoneCondition
