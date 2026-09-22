"""Reward functions: (previous state, state, deploy results) -> scalar, per team.

Composable: ``CombinedReward([(WinLossReward(), 1.0), (TowerHPReward(), 0.3)])``.

A WARNING ABOUT WEIGHTS
    A reward coefficient that gets re-tuned every time a new behaviour is
    measured is standing in for a missing reward term. If the agent learns to
    hoard elixir and you respond by nudging ``ElixirLeakPenalty`` up, then down,
    then up again as other behaviours shift, the weight is absorbing something
    the reward cannot express -- find that thing and give it a term (or remove a
    term that is fighting the terminal signal). Weights should settle, not drift.
    The terminal win/loss term is the only one that is the actual objective;
    everything else is shaping, and shaping that is not a difference of a
    potential function can change which policy is optimal.

ZERO-SUM
    Every term here is antisymmetric between seats on a mirrored transition
    (reward_blue == -reward_red), except ``ElixirLeakPenalty``,
    ``IllegalActionPenalty`` and ``PlacementDepthReward``, which score only the
    acting player's own behaviour. Tests check the antisymmetry, because a self-play
    reward that is not zero-sum rewards both players for colluding.

POSITION
    ``DeployResult`` carries the ``x`` and ``y`` a command was evaluated at, in the
    ENGINE frame. Any term that scores WHERE something was played must convert with
    ``protocol.to_own`` first, or it rewards Blue and punishes Red for the same
    placement and the self-play run learns the seat rather than the game.
    ``PlacementDepthReward`` is the worked example.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from fractions import Fraction

from .protocol import (
    BattleState,
    CardInfo,
    DeployResult,
    Engine,
    EntityKind,
    TowerSlot,
    Winner,
    to_own,
)


class RewardFunction(ABC):
    def bind(self, engine: Engine) -> None:
        """Receive static engine data (card catalogue etc.). Optional."""
        del engine

    def reset(self, state: BattleState) -> None:
        """Called at episode start. Optional."""
        del state

    def config(self) -> dict[str, object]:
        """Constructor state, JSON-able, for ``ClashParallelEnv.config()``."""
        return {}

    @abstractmethod
    def get_reward(
        self,
        team: int,
        prev: BattleState,
        state: BattleState,
        results: Sequence[DeployResult],
    ) -> float: ...


class WinLossReward(RewardFunction):
    """+1 on the transition into a win, -1 into a loss, ``draw`` for a draw."""

    def __init__(self, draw: float = 0.0) -> None:
        self.draw = draw

    def config(self) -> dict[str, object]:
        return {"draw": self.draw}

    def get_reward(
        self,
        team: int,
        prev: BattleState,
        state: BattleState,
        results: Sequence[DeployResult],
    ) -> float:
        if not state.game_over or prev.game_over:
            return 0.0
        if state.winner == Winner.DRAW:
            return self.draw
        return 1.0 if state.winner == team else -1.0


class CrownReward(RewardFunction):
    """Change in (own crowns - enemy crowns)."""

    def get_reward(
        self,
        team: int,
        prev: BattleState,
        state: BattleState,
        results: Sequence[DeployResult],
    ) -> float:
        own = state.players[team].crowns - prev.players[team].crowns
        foe = state.players[1 - team].crowns - prev.players[1 - team].crowns
        return float(own - foe)


class TowerHPReward(RewardFunction):
    """Enemy tower HP destroyed minus own tower HP lost, as fractions of max HP.

    This is a difference of a potential (sum of normalised tower HP), so it does
    not change which policy is optimal for the terminal objective -- it only
    densifies the signal.
    """

    def __init__(self, king_weight: float = 1.0, princess_weight: float = 1.0) -> None:
        self.w = {
            TowerSlot.KING: king_weight,
            TowerSlot.LEFT: princess_weight,
            TowerSlot.RIGHT: princess_weight,
        }

    def config(self) -> dict[str, object]:
        return {
            "king_weight": self.w[TowerSlot.KING],
            "princess_weight": self.w[TowerSlot.LEFT],
        }

    def _potential(self, state: BattleState, team: int) -> float:
        p = state.players[team]
        return sum(self.w[s] * p.tower_hp[s] / max(1, p.tower_max_hp[s]) for s in TowerSlot)

    def get_reward(
        self,
        team: int,
        prev: BattleState,
        state: BattleState,
        results: Sequence[DeployResult],
    ) -> float:
        own = self._potential(state, team) - self._potential(prev, team)
        foe = self._potential(state, 1 - team) - self._potential(prev, 1 - team)
        return own - foe


class ElixirTradeReward(RewardFunction):
    """Elixir value of enemy units killed minus own units lost, / MAX-ish scale.

    A unit is valued at card elixir / units summoned by that card. Entities that
    vanish count as killed whether by damage or by lifetime expiry -- a building
    that times out WAS spent elixir, so counting it as a loss is the honest
    accounting, not a bug. Crown towers are excluded (TowerHPReward covers them).

    EXACT ARITHMETIC. Unit values are ``Fraction(elixir, count)`` and the sum is
    exact until the final division. A running float sum of the same values depends
    on entity iteration order, so on a perfectly rotation-mirrored transition (the
    same kills on both sides) it returns +1.1e-19 for one seat and -1.1e-19 for the
    other instead of 0 for both -- measured on both engines by
    tests/test_rust_engine.py's multi-unit rotation test. Harmless to a gradient,
    but it makes "equal rewards on a mirror" an uncheckable property.
    """

    def __init__(self, scale: float = 10.0) -> None:
        self.scale = scale
        self.value: dict[int, Fraction] = {}

    def config(self) -> dict[str, object]:
        return {"scale": self.scale}

    def bind(self, engine: Engine) -> None:
        cards: Sequence[CardInfo] = engine.cards()
        self.value = {c.card_id: Fraction(c.elixir, max(1, c.count)) for c in cards}

    def get_reward(
        self,
        team: int,
        prev: BattleState,
        state: BattleState,
        results: Sequence[DeployResult],
    ) -> float:
        alive = {e.uid for e in state.entities}
        total = Fraction(0)
        for e in prev.entities:
            if e.uid in alive or e.kind in (EntityKind.KING_TOWER, EntityKind.PRINCESS_TOWER):
                continue
            v = self.value.get(e.card_id, Fraction(0))
            total += v if e.team != team else -v
        return float(total) / self.scale


class ElixirLeakPenalty(RewardFunction):
    """-1 per decision step spent sitting at full elixir (regeneration wasted).

    Not zero-sum: both players can leak at once. ``max_milli`` defaults to the
    engine's MAX_MANA via calibration when bound through an env.
    """

    def __init__(self, max_milli: int | None = None) -> None:
        self.max_milli = max_milli

    def config(self) -> dict[str, object]:
        return {"max_milli": self.max_milli}

    def bind(self, engine: Engine) -> None:
        if self.max_milli is None:
            from .protocol import default_calibration

            self.max_milli = default_calibration().int("match.MAX_MANA") * 1000

    def get_reward(
        self,
        team: int,
        prev: BattleState,
        state: BattleState,
        results: Sequence[DeployResult],
    ) -> float:
        assert self.max_milli is not None, "ElixirLeakPenalty used before bind()"
        full = self.max_milli
        leaked = (
            prev.players[team].elixir_milli >= full and state.players[team].elixir_milli >= full
        )
        return -1.0 if leaked else 0.0


class PlacementDepthReward(RewardFunction):
    """How far up the board this team's accepted placements were, in own-frame tiles.

    The template for every positional shaping term, and the reason
    ``DeployResult`` carries coordinates. Positive is forward: a placement in the
    enemy half scores above one behind your own towers, scaled so a placement at the
    far end is 1 and at your own back line is -1. ``weight`` on the aggressive side
    is the knob; a NEGATIVE weight rewards defending at home.

    NOT IN ``default_reward`` AT ALL, on purpose -- not at zero weight, not at any
    weight. Whether pushing or defending is better is the thing a bot is supposed to
    learn, and a shaping term that answers it in advance is the weight-drift this
    module's docstring warns about. It is here to be copied and to prove the
    coordinates arrive, not to be switched on untested.
    ``test_the_positional_term_stays_out_of_the_default_reward`` keeps it out.

    Antisymmetric between the seats on a mirrored transition, like the terms above:
    the own frame flips with the seat, so the same placement scores +d for one
    player and is scored by the other only through its own placements.
    """

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    def bind(self, engine: Engine) -> None:
        self.arena = engine.arena()

    def config(self) -> dict[str, object]:
        return {"weight": self.weight}

    def get_reward(
        self,
        team: int,
        prev: BattleState,
        state: BattleState,
        results: Sequence[DeployResult],
    ) -> float:
        arena = getattr(self, "arena", None)
        if arena is None:
            raise RuntimeError("PlacementDepthReward used before bind()")
        total = 0.0
        for r in results:
            if r.team != team or r.status != 0:
                continue
            _, y_own = to_own(arena, team, r.x, r.y)
            total += (2.0 * y_own / max(1, arena.height)) - 1.0
        return self.weight * total


class IllegalActionPenalty(RewardFunction):
    """-1 for each of this team's commands the engine rejected this step.

    With a correct mask and a masked policy this is always 0; a non-zero value in
    training logs is a mask bug or an unmasked policy, which is why it exists.
    """

    def get_reward(
        self,
        team: int,
        prev: BattleState,
        state: BattleState,
        results: Sequence[DeployResult],
    ) -> float:
        return -float(sum(1 for r in results if r.team == team and r.status != 0))


class CombinedReward(RewardFunction):
    """Weighted sum of terms. ``last_terms`` keeps each weighted term for logging.

    KEYED BY TEAM, because the env calls ``get_reward`` once PER SEAT on the same
    transition. A single flat breakdown was cleared at the top of every call, so
    the first seat's numbers were destroyed by the second and whatever a training
    run logged as "the reward breakdown" was only ever Red's -- while the scalar
    rewards, which are returned rather than stored, were right for both. Read one
    seat's with ``terms_for(team)``.
    """

    def __init__(self, terms: Sequence[tuple[RewardFunction, float]]) -> None:
        self.terms = list(terms)
        self.last_terms: dict[int, dict[str, float]] = {}

    def terms_for(self, team: int) -> dict[str, float]:
        """The weighted breakdown of ``team``'s last reward; empty before the first."""
        return self.last_terms.get(team, {})

    def config(self) -> dict[str, object]:
        return {
            "terms": [
                {"class": type(t).__name__, "weight": w, "params": t.config()}
                for t, w in self.terms
            ]
        }

    def bind(self, engine: Engine) -> None:
        for t, _ in self.terms:
            t.bind(engine)

    def reset(self, state: BattleState) -> None:
        self.last_terms = {}
        for t, _ in self.terms:
            t.reset(state)

    def get_reward(
        self,
        team: int,
        prev: BattleState,
        state: BattleState,
        results: Sequence[DeployResult],
    ) -> float:
        total = 0.0
        breakdown: dict[str, float] = {}
        self.last_terms[team] = breakdown
        for t, w in self.terms:
            v = w * t.get_reward(team, prev, state, results)
            breakdown[type(t).__name__] = v
            total += v
        return total


def default_reward() -> CombinedReward:
    """Terminal objective plus light, potential-style shaping."""
    return CombinedReward(
        [
            (WinLossReward(), 1.0),
            (CrownReward(), 0.2),
            (TowerHPReward(), 0.1),
            (ElixirTradeReward(), 0.02),
        ]
    )
