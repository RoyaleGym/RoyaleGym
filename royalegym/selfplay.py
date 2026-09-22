"""Self-play: opponents, an opponent pool, sampling and Elo bookkeeping.

OPPONENTS
    An ``Opponent`` maps (observation, action mask, rng) -> action. The rng is
    the env's own generator, so a seeded env with a stochastic opponent is still
    reproducible (gymnasium's step-determinism check relies on this).

POOL SAMPLING
    uniform   every snapshot equally likely -- broad, protects against forgetting
    latest    always the newest -- pure self-play, fastest to cycle into rock-
              paper-scissors loops
    pfsp      prioritised fictitious self-play (AlphaStar): weight each snapshot
              by f(P[learner beats it]); ``hard`` f(p)=(1-p)^power focuses on
              opponents the learner still loses to, ``variance`` f(p)=p(1-p)
              focuses on even matchups.
    Win probabilities use a Beta(1,1) prior, so an unplayed snapshot reads 0.5
    instead of dividing by zero or being ignored forever.

ELO
    Standard logistic Elo with base-10 / 400 scale. It is bookkeeping for humans
    and for PFSP diagnostics, not a training signal. Elo assumes transitive
    strength; a Clash meta is not transitive, so read it with that in mind.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import msgspec
import numpy as np

from .action import NOOP


class Opponent(Protocol):
    def act(self, obs: dict[str, Any], mask: np.ndarray, rng: np.random.Generator) -> int: ...


class NoopOpponent:
    """Never plays a card. The easiest possible scripted opponent."""

    def act(self, obs: dict[str, Any], mask: np.ndarray, rng: np.random.Generator) -> int:
        return NOOP


class RandomLegalOpponent:
    """Uniform over legal actions, taking no-op with probability ``noop_prob``.

    Pure uniform over 2305 actions picks no-op ~0% of the time and spams cards
    the instant they are affordable, which is a strange opponent to learn against.
    """

    def __init__(self, noop_prob: float = 0.9) -> None:
        self.noop_prob = noop_prob

    def act(self, obs: dict[str, Any], mask: np.ndarray, rng: np.random.Generator) -> int:
        legal = np.flatnonzero(mask)
        legal = legal[legal != NOOP]
        if legal.size == 0 or rng.random() < self.noop_prob:
            return NOOP
        return int(legal[rng.integers(legal.size)])


class CallableOpponent:
    """Wrap a frozen policy function ``fn(obs, mask) -> action``."""

    def __init__(self, fn: Callable[[dict[str, Any], np.ndarray], int]) -> None:
        self.fn = fn

    def act(self, obs: dict[str, Any], mask: np.ndarray, rng: np.random.Generator) -> int:
        a = int(self.fn(obs, mask))
        if not mask[a]:
            # A frozen policy trained on an older mask must not smuggle illegal
            # actions into the engine; fall back to no-op and let the env log it.
            return NOOP
        return a


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------


class PolicySnapshot(msgspec.Struct):
    snapshot_id: str
    step: int
    payload: str | None = None  # e.g. a checkpoint path; the pool never opens it
    elo: float = 1200.0
    games: int = 0
    meta: dict[str, Any] = {}


class _Record(msgspec.Struct):
    wins: float = 0.0  # draws count 0.5
    games: int = 0


class PoolFile(msgspec.Struct):
    snapshots: list[PolicySnapshot]
    records: dict[str, dict[str, _Record]]  # records[a][b] = a's results vs b
    k_factor: float
    initial_elo: float


def expected_score(elo_a: float, elo_b: float) -> float:
    return float(1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0)))


class OpponentPool:
    """Snapshot registry with sampling strategies and Elo/head-to-head tracking.

    The learner is an ordinary entry (conventionally id ``"learner"``) so its Elo
    is tracked on the same scale as the snapshots it plays.
    """

    def __init__(
        self, k_factor: float = 32.0, initial_elo: float = 1200.0, max_size: int | None = None
    ) -> None:
        self.k = k_factor
        self.initial_elo = initial_elo
        self.max_size = max_size
        self.snapshots: dict[str, PolicySnapshot] = {}
        self.records: dict[str, dict[str, _Record]] = {}

    # -- registry ------------------------------------------------------------

    def add(
        self,
        snapshot_id: str,
        step: int,
        payload: str | None = None,
        meta: dict[str, Any] | None = None,
        elo: float | None = None,
    ) -> PolicySnapshot:
        if snapshot_id in self.snapshots:
            raise KeyError(f"snapshot {snapshot_id!r} already registered")
        snap = PolicySnapshot(
            snapshot_id=snapshot_id,
            step=step,
            payload=payload,
            elo=self.initial_elo if elo is None else elo,
            meta=dict(meta or {}),
        )
        self.snapshots[snapshot_id] = snap
        if self.max_size is not None:
            self._evict()
        return snap

    def _evict(self) -> None:
        assert self.max_size is not None
        ids = [s for s in self.snapshots if s != "learner"]
        while len(ids) > self.max_size:
            oldest = min(ids, key=lambda i: self.snapshots[i].step)
            del self.snapshots[oldest]
            ids.remove(oldest)

    def ensure(self, snapshot_id: str, step: int = 0) -> PolicySnapshot:
        if snapshot_id not in self.snapshots:
            return self.add(snapshot_id, step)
        return self.snapshots[snapshot_id]

    def latest(self, exclude: Sequence[str] = ("learner",)) -> PolicySnapshot:
        cands = [s for s in self.snapshots.values() if s.snapshot_id not in exclude]
        if not cands:
            raise LookupError("pool is empty")
        return max(cands, key=lambda s: (s.step, s.snapshot_id))

    # -- results -------------------------------------------------------------

    def record_result(self, a: str, b: str, score_a: float) -> tuple[float, float]:
        """Record one game. ``score_a`` is 1 (a won), 0.5 (draw) or 0 (b won).

        Returns the new (elo_a, elo_b). The update is zero-sum: the pool's total
        Elo is conserved, checked by
        ``tests/test_selfplay_pool.py::test_the_elo_update_is_zero_sum``. It was not
        checked by anything for the life of this class, while this line said it was --
        which is worse than saying nothing, because it stops the next reader looking.
        """
        if score_a not in (0.0, 0.5, 1.0):
            raise ValueError("score_a must be 0, 0.5 or 1")
        sa, sb = self.ensure(a), self.ensure(b)
        ea = expected_score(sa.elo, sb.elo)
        delta = self.k * (score_a - ea)
        sa.elo += delta
        sb.elo -= delta
        sa.games += 1
        sb.games += 1
        ra = self.records.setdefault(a, {}).setdefault(b, _Record())
        rb = self.records.setdefault(b, {}).setdefault(a, _Record())
        ra.wins += score_a
        ra.games += 1
        rb.wins += 1.0 - score_a
        rb.games += 1
        return sa.elo, sb.elo

    def win_rate(self, a: str, b: str) -> float:
        """P[a beats b] with a Beta(1,1) prior."""
        r = self.records.get(a, {}).get(b)
        if r is None:
            return 0.5
        return (r.wins + 1.0) / (r.games + 2.0)

    # -- sampling ------------------------------------------------------------

    def pfsp_weights(
        self,
        learner: str,
        candidates: Sequence[str],
        weighting: str = "hard",
        power: float = 2.0,
    ) -> np.ndarray:
        p = np.array([self.win_rate(learner, c) for c in candidates], dtype=np.float64)
        if weighting == "hard":
            w = (1.0 - p) ** power
        elif weighting == "variance":
            w = p * (1.0 - p)
        elif weighting == "linear":
            w = 1.0 - p
        else:
            raise ValueError(f"unknown PFSP weighting {weighting!r}")
        if w.sum() <= 0:
            w = np.ones_like(w)
        return w / w.sum()

    def sample(
        self,
        rng: np.random.Generator,
        strategy: str = "pfsp",
        learner: str = "learner",
        weighting: str = "hard",
        power: float = 2.0,
    ) -> PolicySnapshot:
        cands = sorted(s for s in self.snapshots if s != learner)
        if not cands:
            raise LookupError("pool has no opponents")
        if strategy == "latest":
            return self.latest(exclude=(learner,))
        if strategy == "uniform":
            return self.snapshots[cands[int(rng.integers(len(cands)))]]
        if strategy == "pfsp":
            w = self.pfsp_weights(learner, cands, weighting, power)
            return self.snapshots[cands[int(rng.choice(len(cands), p=w))]]
        raise ValueError(f"unknown strategy {strategy!r}")

    # -- persistence ---------------------------------------------------------

    def save(self, path: str | Path) -> None:
        f = PoolFile(
            snapshots=list(self.snapshots.values()),
            records=self.records,
            k_factor=self.k,
            initial_elo=self.initial_elo,
        )
        Path(path).write_bytes(msgspec.json.encode(f))

    @classmethod
    def load(cls, path: str | Path, max_size: int | None = None) -> OpponentPool:
        f = msgspec.json.decode(Path(path).read_bytes(), type=PoolFile)
        pool = cls(k_factor=f.k_factor, initial_elo=f.initial_elo, max_size=max_size)
        pool.snapshots = {s.snapshot_id: s for s in f.snapshots}
        pool.records = f.records
        return pool
