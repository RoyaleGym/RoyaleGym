"""State mutators: how each episode begins.

A mutator returns either a ``MatchSetup`` (the engine builds a fresh battle from
it) or a ``Snapshot`` (the engine loads an exact saved state). All randomness
comes from the ``np.random.Generator`` the env passes in, which is itself seeded
from ``env.reset(seed=...)``, so a seeded reset is reproducible end to end.

Curriculum is expressed here, not in the engine: start mid-game with a tower
already down, start from a scripted defensive board, replay a saved position.

The name is RLGym v2's. One difference from RLGym's mutators: a battle here is
described whole (``MatchSetup`` is the complete starting state) and then handed
to the engine, rather than edited in place, so mutators compose by *choice*
(``WeightedStateMutator`` picks one per episode) rather than by chaining, and
there is no ``MutatorSequence``. Variations on a start are subclasses of
``DefaultStateMutator`` that fill in more of the ``MatchSetup``.
"""

from __future__ import annotations

import difflib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .protocol import (
    BLUE,
    DECK_SIZE,
    RED,
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


class StateMutator(ABC):
    @abstractmethod
    def build(self, rng: np.random.Generator, cards: Sequence[CardInfo]) -> MatchSetup | Snapshot:
        """Describe the next episode's starting state."""

    def config(self) -> dict[str, object]:
        """Constructor state, JSON-able, for ``ClashParallelEnv.config()``."""
        return {}


def random_deck(rng: np.random.Generator, cards: Sequence[CardInfo]) -> list[int]:
    if len(cards) < DECK_SIZE:
        raise ValueError(f"need at least {DECK_SIZE} cards, catalogue has {len(cards)}")
    return [int(c) for c in rng.choice(len(cards), size=DECK_SIZE, replace=False)]


class DefaultStateMutator(StateMutator):
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

    def config(self) -> dict[str, object]:
        return {"decks": self.decks, "shuffle": int(self.shuffle), "mirror": self.mirror}

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


class MidGameStateMutator(DefaultStateMutator):
    """Start partway through the match with randomised elixir and tower damage.

    Ticks and elixir are given as inclusive integer ranges. ``tower_down_prob``
    is the chance each princess tower starts destroyed (awarding the crown).
    Tower HP ranges are fractions in PERCENT of the engine's default max HP, which
    the mutator does not know -- so it is expressed as ``tower_hp_percent`` and
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

    def config(self) -> dict[str, object]:
        return {
            **super().config(),
            "tick_range": list(self.tick_range),
            "elixir_milli_range": list(self.elixir_range),
            "max_tower_hp": list(self.max_tower_hp),
            "tower_hp_percent": list(self.hp_pct),
            "tower_down_prob": self.down_prob,
        }

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


class ScriptedBoardStateMutator(DefaultStateMutator):
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

    def config(self) -> dict[str, object]:
        return {
            **super().config(),
            "spawns": len(self.spawns),
            "elixir_milli": self.elixir,
            "tower_hp": self.tower_hp,
            "start_tick": self.start_tick,
        }

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


class SnapshotStateMutator(StateMutator):
    """Resume from saved positions, sampled uniformly (e.g. mined from replays)."""

    def __init__(self, blobs: Sequence[bytes]) -> None:
        if not blobs:
            raise ValueError("SnapshotStateMutator needs at least one snapshot")
        self.blobs = list(blobs)

    def config(self) -> dict[str, object]:
        return {"snapshots": len(self.blobs)}

    def build(self, rng: np.random.Generator, cards: Sequence[CardInfo]) -> Snapshot:
        return Snapshot(self.blobs[int(rng.integers(len(self.blobs)))])


class WeightedStateMutator(StateMutator):
    """Curriculum mix: pick a child mutator by weight each episode.

    ``set_weights`` lets a training loop anneal the mix (e.g. from scripted drills
    toward full games) without rebuilding the env.
    """

    def __init__(self, mutators: Sequence[tuple[StateMutator, float]]) -> None:
        self.mutators = [m for m, _ in mutators]
        self.set_weights([w for _, w in mutators])

    def config(self) -> dict[str, object]:
        return {
            "mutators": [
                {"class": type(m).__name__, "weight": float(p), "params": m.config()}
                for m, p in zip(self.mutators, self.p, strict=True)
            ]
        }

    def set_weights(self, weights: Sequence[float]) -> None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (len(self.mutators),) or (w < 0).any() or w.sum() <= 0:
            raise ValueError("weights must be non-negative, one per mutator, not all zero")
        self.p = w / w.sum()

    def build(self, rng: np.random.Generator, cards: Sequence[CardInfo]) -> MatchSetup | Snapshot:
        i = int(rng.choice(len(self.mutators), p=self.p))
        return self.mutators[i].build(rng, cards)


#: Where ``DeckCurriculumStateMutator`` puts its deck: a fair coin per episode, or one seat.
SEATS = ("either", "blue", "red")
_TEAM_OF_SEAT = {"blue": BLUE, "red": RED}


def deck_ids(names: Sequence[str], cards: Sequence[CardInfo], where: str = "deck") -> list[int]:
    """Card ids for a deck written as card NAMES, looked up in an engine's catalogue.

    A card id is a position in one catalogue, and positions move between card tables,
    so a deck kept as names means the same cards on every engine that has them. A name
    the catalogue does not have raises ValueError naming it: a typo, or a card this
    engine does not simulate.
    """
    index = {c.name: c.card_id for c in cards}
    ids = []
    for name in names:
        if name not in index:
            close = difflib.get_close_matches(name, list(index), n=3)
            hint = f" (close: {', '.join(close)})" if close else ""
            raise ValueError(
                f"{where} names {name!r}, which this engine's catalogue of {len(cards)} "
                f"cards does not have{hint}. A deck can only use cards engine.cards() lists."
            )
        ids.append(int(index[name]))
    return ids


def _deck_names(deck: Sequence[str], where: str) -> tuple[str, ...]:
    # A tuple first, so a deck given as a generator is read once, not used up by the check.
    names = tuple(deck) if not isinstance(deck, str) else None
    if names is None or not all(isinstance(n, str) for n in names):
        raise TypeError(f"{where} must be a list of card names, got {deck!r}")
    if len(names) != DECK_SIZE:
        raise ValueError(f"{where} must have {DECK_SIZE} cards, got {len(names)}")
    if len(set(names)) != DECK_SIZE:
        raise ValueError(f"{where} lists a card twice: {list(names)}")
    return names


def _probability(value: float, name: str) -> float:
    p = float(value)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must be a probability in [0, 1], got {value!r}")
    return p


class DeckCurriculumStateMutator(StateMutator):
    """A deck curriculum: one deck you name, on one seat or both, against a pool.

    Each episode:
      * with probability ``p`` the deck is played. Then with probability ``mirror_p``
        both seats get it in the same order (``ShuffleMode.MIRRORED``). Otherwise it
        goes to ``seat`` and the other seat draws from ``pool``.
      * with probability ``1 - p`` both seats draw from ``pool``, independently.

    ``seat`` is "blue", "red", or "either" (a fair coin each episode). With "either"
    and no mirror, the deck is on each seat in ``p / 2`` of episodes.

    ``pool`` is a list of decks, drawn uniformly (list a deck twice to weight it), or
    None for a random deck of eight different cards from the catalogue. Pool draws use
    ``shuffle``; a mirror always uses ``ShuffleMode.MIRRORED``.

    Decks are card NAMES, looked up in the catalogue the env passes to ``build``. A name
    the catalogue does not have fails the first reset, for every deck in the pool and
    not only the one drawn, so a typo cannot wait hours for its turn.

    ``config()`` is exactly the constructor's keyword arguments, JSON-able, so
    ``from_config(m.config())`` rebuilds a mutator that draws the same episodes from the
    same seed, and a config file can name the class and pass the dict as its kwargs.
    ``set_curriculum`` changes any of them between episodes without rebuilding the env::

        main = ["HogRider", "Musketeer", "Cannon", "Skeletons",
                "Fireball", "Log", "Knight", "Zap"]
        m = DeckCurriculumStateMutator(main, p=1.0, mirror_p=1.0)    # mirror matches
        env = ClashParallelEnv(engine, state_mutator=m)
        ...
        m.set_curriculum(p=0.9, mirror_p=0.0, pool=[deck_a, deck_b])  # then a pool

    ``set_curriculum`` changes this object only. Envs built in other processes (an
    ``EnvFactory`` recipe run in a worker) hold their own copy and need the call too:
    send ``m.config()`` and call ``set_curriculum(**config)`` there.
    """

    def __init__(
        self,
        deck: Sequence[str],
        *,
        p: float = 1.0,
        seat: str = "either",
        mirror_p: float = 0.0,
        pool: Sequence[Sequence[str]] | None = None,
        shuffle: int = ShuffleMode.INDEPENDENT,
    ) -> None:
        self._ids: tuple[tuple[str, ...], list[int], list[list[int]] | None] | None = None
        self._set(deck, p, seat, mirror_p, pool, shuffle)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DeckCurriculumStateMutator:
        """The mutator ``config()`` describes. Also takes the ``{"class", "params"}``
        record ``ClashParallelEnv.config()`` keeps under ``"state_mutator"``."""
        if "params" in config and "class" in config:
            if str(config["class"]).rsplit(".", 1)[-1] != cls.__name__:
                raise ValueError(f"config is for {config['class']}, not {cls.__name__}")
            config = config["params"]
        return cls(**config)

    def config(self) -> dict[str, object]:
        return {
            "deck": list(self.deck),
            "p": self.p,
            "seat": self.seat,
            "mirror_p": self.mirror_p,
            "pool": [list(d) for d in self.pool] if self.pool is not None else None,
            "shuffle": int(self.shuffle),
        }

    def set_curriculum(self, **changes: Any) -> None:
        """Change any constructor argument, by name, from the next episode on.

        All or nothing: a value that is refused leaves the mutator as it was.
        """
        unknown = sorted(set(changes) - set(self.config()))
        if unknown:
            raise TypeError(f"set_curriculum got unknown arguments {unknown}")
        merged: dict[str, Any] = {**self.config(), **changes}
        self._set(**merged)

    def _set(
        self,
        deck: Sequence[str],
        p: float,
        seat: str,
        mirror_p: float,
        pool: Sequence[Sequence[str]] | None,
        shuffle: int,
    ) -> None:
        # Everything is checked before anything is stored, so a refusal changes nothing.
        names = _deck_names(deck, "deck")
        if seat not in SEATS:
            raise ValueError(f"seat must be one of {SEATS}, got {seat!r}")
        if pool is not None:
            if isinstance(pool, str) or len(pool) == 0:
                raise ValueError("pool must be a list of decks, or None for random decks")
            pool_names = tuple(_deck_names(d, f"pool[{i}]") for i, d in enumerate(pool))
        else:
            pool_names = None
        values = (_probability(p, "p"), _probability(mirror_p, "mirror_p"))
        mode = ShuffleMode(int(shuffle))
        self.deck, self.seat, self.pool, self.shuffle = names, seat, pool_names, mode
        self.p, self.mirror_p = values
        self._ids = None

    def _resolve(self, cards: Sequence[CardInfo]) -> tuple[list[int], list[list[int]] | None]:
        """Card ids of the deck and the whole pool, for this catalogue.

        Kept until the catalogue or the curriculum changes, so a reset does not look up
        a large pool again.
        """
        key = tuple(c.name for c in cards)
        if self._ids is None or self._ids[0] != key:
            deck = deck_ids(self.deck, cards, "deck")
            pool = None
            if self.pool is not None:
                pool = [deck_ids(d, cards, f"pool[{i}]") for i, d in enumerate(self.pool)]
            self._ids = (key, deck, pool)
        return self._ids[1], self._ids[2]

    def _from_pool(
        self, rng: np.random.Generator, cards: Sequence[CardInfo], pool: list[list[int]] | None
    ) -> list[int]:
        if pool is None:
            return random_deck(rng, cards)
        return list(pool[int(rng.integers(len(pool)))])

    def build(self, rng: np.random.Generator, cards: Sequence[CardInfo]) -> MatchSetup:
        deck, pool = self._resolve(cards)
        if rng.random() >= self.p:
            both = [self._from_pool(rng, cards, pool) for _ in TEAMS]
            return MatchSetup(decks=both, shuffle=int(self.shuffle))
        if rng.random() < self.mirror_p:
            return MatchSetup(decks=[list(deck), list(deck)], shuffle=int(ShuffleMode.MIRRORED))
        if self.seat == "either":
            team = int(rng.integers(len(TEAMS)))
        else:
            team = _TEAM_OF_SEAT[self.seat]
        decks: list[list[int]] = [[], []]
        decks[team] = list(deck)
        decks[1 - team] = self._from_pool(rng, cards, pool)
        return MatchSetup(decks=decks, shuffle=int(self.shuffle))


# Kept for callers written before the rename.
StateSetter = StateMutator
DefaultStateSetter = DefaultStateMutator
MidGameStateSetter = MidGameStateMutator
ScriptedBoardStateSetter = ScriptedBoardStateMutator
SnapshotStateSetter = SnapshotStateMutator
WeightedStateSetter = WeightedStateMutator
