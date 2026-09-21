"""State setters: how each episode begins.

A setter returns either a ``MatchSetup`` (the engine builds a fresh battle from
it) or a ``Snapshot`` (the engine loads an exact saved state). All randomness
comes from the ``np.random.Generator`` the env passes in, which is itself seeded
from ``env.reset(seed=...)``, so a seeded reset is reproducible end to end.

Curriculum is expressed here, not in the engine: start mid-game with a tower
already down, start from a scripted defensive board, replay a saved position.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .protocol import (
    DECK_SIZE,
    TEAMS,
    CardInfo,
    MatchSetup,
    ShuffleMode,
    SpawnSpec,
    TowerSlot,
)


@dataclass(frozen=True)
class Snapshot:
    """An exact engine state to resume from (``Engine.save_state`` bytes)."""

    blob: bytes


class StateSetter(ABC):
    @abstractmethod
    def build(self, rng: np.random.Generator, cards: Sequence[CardInfo]) -> MatchSetup | Snapshot:
        """Describe the next episode's starting state."""


def random_deck(rng: np.random.Generator, cards: Sequence[CardInfo]) -> list[int]:
    if len(cards) < DECK_SIZE:
        raise ValueError(f"need at least {DECK_SIZE} cards, catalogue has {len(cards)}")
    return [int(c) for c in rng.choice(len(cards), size=DECK_SIZE, replace=False)]


class DefaultStateSetter(StateSetter):
    """A normal battle from tick 0.

    ``decks``: fixed [blue, red] decks; None draws a random 8-card deck per team.
    ``mirror``: both teams get Blue's deck in the same order (ShuffleMode.MIRRORED)
    -- the setting for self-play symmetry checks.
    """

    def __init__(
        self,
        decks: Sequence[Sequence[int]] | None = None,
        shuffle: ShuffleMode = ShuffleMode.INDEPENDENT,
        mirror: bool = False,
    ) -> None:
        self.decks = [list(d) for d in decks] if decks is not None else None
        self.shuffle = shuffle
        self.mirror = mirror

    def _decks(self, rng: np.random.Generator, cards: Sequence[CardInfo]) -> list[list[int]]:
        if self.decks is not None:
            decks = [list(d) for d in self.decks]
        else:
            decks = [random_deck(rng, cards) for _ in TEAMS]
        if self.mirror:
            decks = [list(decks[0]), list(decks[0])]
        return decks

    def build(self, rng: np.random.Generator, cards: Sequence[CardInfo]) -> MatchSetup:
        shuffle = ShuffleMode.MIRRORED if self.mirror else self.shuffle
        return MatchSetup(decks=self._decks(rng, cards), shuffle=int(shuffle))


class MidGameStateSetter(DefaultStateSetter):
    """Start partway through the match with randomised elixir and tower damage.

    Ticks and elixir are given as inclusive integer ranges. ``tower_down_prob``
    is the chance each princess tower starts destroyed (awarding the crown).
    Tower HP ranges are fractions in PERCENT of the engine's default max HP, which
    the setter does not know -- so it is expressed as ``tower_hp_percent`` and
    converted using ``max_tower_hp`` supplied by the caller.
    """

    def __init__(
        self,
        tick_range: tuple[int, int],
        elixir_milli_range: tuple[int, int],
        max_tower_hp: tuple[int, int],  # (king, princess) max HP
        tower_hp_percent: tuple[int, int] = (100, 100),
        tower_down_prob: float = 0.0,
        decks: Sequence[Sequence[int]] | None = None,
        mirror: bool = False,
    ) -> None:
        super().__init__(decks=decks, mirror=mirror)
        self.tick_range = tick_range
        self.elixir_range = elixir_milli_range
        self.max_tower_hp = max_tower_hp
        self.hp_pct = tower_hp_percent
        self.down_prob = tower_down_prob

    def build(self, rng: np.random.Generator, cards: Sequence[CardInfo]) -> MatchSetup:
        base = super().build(rng, cards)
        tick = int(rng.integers(self.tick_range[0], self.tick_range[1] + 1))
        elixir = [int(rng.integers(self.elixir_range[0], self.elixir_range[1] + 1)) for _ in TEAMS]
        hp: list[list[int]] = []
        for _ in TEAMS:
            row = [0, 0, 0]
            for slot in TowerSlot:
                max_hp = self.max_tower_hp[0 if slot == TowerSlot.KING else 1]
                pct = int(rng.integers(self.hp_pct[0], self.hp_pct[1] + 1))
                row[slot] = max(1, max_hp * pct // 100)
                if slot != TowerSlot.KING and rng.random() < self.down_prob:
                    row[slot] = 0
            hp.append(row)
        return MatchSetup(
            decks=base.decks,
            shuffle=base.shuffle,
            start_tick=tick,
            elixir_milli=elixir,
            tower_hp=hp,
        )


class ScriptedBoardStateSetter(DefaultStateSetter):
    """A fixed board: units already on the field (e.g. a defensive drill)."""

    def __init__(
        self,
        spawns: Sequence[SpawnSpec],
        decks: Sequence[Sequence[int]] | None = None,
        elixir_milli: Sequence[int] | None = None,
        tower_hp: Sequence[Sequence[int]] | None = None,
        start_tick: int = 0,
        mirror: bool = False,
    ) -> None:
        super().__init__(decks=decks, mirror=mirror)
        self.spawns = list(spawns)
        self.elixir = list(elixir_milli) if elixir_milli is not None else None
        self.tower_hp = [list(r) for r in tower_hp] if tower_hp is not None else None
        self.start_tick = start_tick

    def build(self, rng: np.random.Generator, cards: Sequence[CardInfo]) -> MatchSetup:
        base = super().build(rng, cards)
        return MatchSetup(
            decks=base.decks,
            shuffle=base.shuffle,
            start_tick=self.start_tick,
            elixir_milli=self.elixir,
            tower_hp=self.tower_hp,
            spawns=list(self.spawns),
        )


class SnapshotStateSetter(StateSetter):
    """Resume from saved positions, sampled uniformly (e.g. mined from replays)."""

    def __init__(self, blobs: Sequence[bytes]) -> None:
        if not blobs:
            raise ValueError("SnapshotStateSetter needs at least one snapshot")
        self.blobs = list(blobs)

    def build(self, rng: np.random.Generator, cards: Sequence[CardInfo]) -> Snapshot:
        return Snapshot(self.blobs[int(rng.integers(len(self.blobs)))])


class WeightedStateSetter(StateSetter):
    """Curriculum mix: pick a child setter by weight each episode.

    ``set_weights`` lets a training loop anneal the mix (e.g. from scripted drills
    toward full games) without rebuilding the env.
    """

    def __init__(self, setters: Sequence[tuple[StateSetter, float]]) -> None:
        self.setters = [s for s, _ in setters]
        self.set_weights([w for _, w in setters])

    def set_weights(self, weights: Sequence[float]) -> None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (len(self.setters),) or (w < 0).any() or w.sum() <= 0:
            raise ValueError("weights must be non-negative, one per setter, not all zero")
        self.p = w / w.sum()

    def build(self, rng: np.random.Generator, cards: Sequence[CardInfo]) -> MatchSetup | Snapshot:
        i = int(rng.choice(len(self.setters), p=self.p))
        return self.setters[i].build(rng, cards)
