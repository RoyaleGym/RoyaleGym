"""Two ObsBuilder implementations must produce the SAME observation, bit for bit.

WHY THIS EXISTS BEFORE THE SECOND IMPLEMENTATION DOES
    The observation builder is being ported to Rust (job 7), and the Python one stays as
    the reference it is checked against. This harness is written first, on purpose. A
    port's equality test written afterwards is shaped by the port: you discover what the
    two disagree about and then decide, case by case, whether that disagreement is
    acceptable. Written first, the contract is fixed before anyone has an implementation
    to defend.

    The measurement that motivates the port, on build d872d792711934c2, mid-battle with 19
    entities on the board and decision_ticks 10:

        engine.step(10 ticks)   0.097 ms
        engine.state() decode   0.012 ms
        action_mask, per seat   0.024 ms
        obs build,    per seat  0.086 ms
        obs build x2 seats      0.171 ms   = 1.76x the simulation itself

    The observation build costs more than the battle. The digest is quoted with it because
    a throughput figure is a fact about a build, and this one moved three times on
    2026-09-22, twice with no commit in this repo at all.

WHAT THIS CANNOT CATCH, SAID PLAINLY
    A defect the two implementations SHARE. If the Python reference is wrong and the port
    reproduces it faithfully, this passes and says nothing -- that is what an agreement
    test is, and it is why the shipped observation also has tests that grade against
    ``engine.state()`` rather than against another builder. An equality harness is for
    catching a port that DRIFTS, not for deciding the reference is right.

    That limit is the reason the plants below matter. An equality test that has only ever
    compared a thing to itself is the same defect one level up: it passes, it looks like a
    gate, and nobody has seen it refuse anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from royalegym.action import NOOP, TileActionParser
from royalegym.env import ClashParallelEnv
from royalegym.obs import ObsBuilder, SpatialObsBuilder
from royalegym.protocol import TEAMS, BattleState
from royalegym.rust_engine import core_available

pytestmark = pytest.mark.skipif(
    not core_available(), reason="the compiled engine is not built; a skip here is not a pass"
)


def a_corpus(n_steps: int = 80) -> list[tuple[BattleState, dict[int, object]]]:
    """Boards a builder actually meets, each with ITS OWN masks, captured live.

    The masks are taken at the moment the state is, and that is not a convenience. A
    ``TileActionParser`` asks the engine where a building would LAND, so a mask computed
    later describes whichever battle the engine is in then, not the state it was handed.
    Collecting states first and masking afterwards silently grades every board against the
    last one. That is the same mistake as reading a unit's position after the whole step
    instead of at the deploy tick, in a different place on the same day.

    THE GENERAL FORM, which is worth more than either instance: a query that consults LIVE
    state is not a pure function of the state you think you are holding. `action_mask` looks
    like `f(state, team)` and is not. Anything that takes a snapshot and calls such a query
    later is silently asking about a different world, and nothing in the signature says so.

    Taken by PLAYING rather than by constructing, so these are boards the engine produces
    and not ones a test author imagined.
    """
    from royalegym.rust_engine import RustEngine

    env = ClashParallelEnv(engine=RustEngine())
    env.reset(seed=0)
    out: list[tuple[BattleState, dict[int, object]]] = []

    def capture() -> None:
        state = env.engine.state()
        out.append((state, {t: env.action_parser.action_mask(state, t) for t in TEAMS}))

    capture()
    for _ in range(n_steps):
        if not env.agents:
            break
        actions = {}
        for agent in env.agents:
            mask = env._masks[agent]
            legal = [i for i in range(1, len(mask)) if mask[i]]
            actions[agent] = legal[len(legal) // 2] if legal else NOOP
        env.step(actions)
        capture()
    return out


def bound(builder, engine=None):
    """A builder bound as the env binds it; unbound it has no board shape.

    Each builder gets its OWN engine instance by default rather than sharing one. Sharing
    would make an equality result depend on two builders having seen the same engine,
    which is a coupling the port will not have: the Rust builder holds its own handle.
    """
    from royalegym.rust_engine import RustEngine

    parser = TileActionParser()
    eng = engine or RustEngine()
    parser.bind(eng)
    builder.bind(eng, parser)
    return builder


def disagreements(a: ObsBuilder, b: ObsBuilder, corpus) -> list[str]:
    """Every key, every state, BOTH seats. Returns a description per disagreement."""
    found: list[str] = []
    for i, (state, masks) in enumerate(corpus):
        for team in TEAMS:
            mask = masks[team]
            left, right = a.build(state, team, mask), b.build(state, team, mask)
            if set(left) != set(right):
                found.append(f"state {i} seat {team}: keys {sorted(left)} vs {sorted(right)}")
                continue
            for key in sorted(left):
                x, y = np.asarray(left[key]), np.asarray(right[key])
                if x.shape != y.shape:
                    found.append(f"state {i} seat {team} {key}: shape {x.shape} vs {y.shape}")
                elif not np.array_equal(x, y):
                    bad = int(np.count_nonzero(x != y))
                    where = np.argwhere(x != y)[0]
                    found.append(
                        f"state {i} seat {team} {key}: {bad} cells differ, first at "
                        f"{tuple(int(v) for v in where)} ({x[tuple(where)]} vs {y[tuple(where)]})"
                    )
    return found


class Perturbed(SpatialObsBuilder):
    """A reference builder with ONE thing wrong, for checking the harness bites.

    Each mode is a defect a port could plausibly have: an off-by-one in a channel index, a
    unit conversion, a seat frame, a single cell. None of them changes the SHAPE, because
    a shape change is the easy case and would be caught by anything.
    """

    def __init__(self, mode: str, **kw) -> None:
        super().__init__(**kw)
        self.mode = mode

    def build(self, state, team, mask):
        out = dict(super().build(state, team, mask))
        board = np.array(out["spatial"], copy=True)
        if self.mode == "one cell":
            board[0, 0, 0] = board[0, 0, 0] + np.float32(1.0)
        elif self.mode == "swapped channels":
            board[[0, 1]] = board[[1, 0]]
        elif self.mode == "scaled":
            board = board * np.float32(1.0001)
        elif self.mode == "flipped board":
            board = board[:, ::-1, :].copy()
        elif self.mode == "vector":
            vec = np.array(out["vector"], copy=True)
            vec[0] = vec[0] + np.float32(1e-6)
            out["vector"] = vec
        else:  # pragma: no cover - a typo in a parametrize would otherwise pass silently
            raise AssertionError(f"unknown perturbation {self.mode!r}")
        out["spatial"] = board
        return out


def test_the_corpus_reaches_boards_that_differ_from_each_other() -> None:
    """A harness run over one board, repeated, would agree about almost anything.

    Pins that the corpus really spans kickoff to crowded, because every equality result
    below is a statement about the boards it was run on and nothing else.
    """
    corpus = a_corpus()
    assert len(corpus) >= 20, f"the corpus is only {len(corpus)} states"
    counts = {len(s.entities) for s, _ in corpus}
    assert len(counts) >= 5, (
        f"every board in the corpus has one of {sorted(counts)} entities, so this is close "
        "to a single board measured repeatedly and would agree about almost anything"
    )
    assert max(counts) - min(counts) >= 4, (
        f"the corpus spans {min(counts)} to {max(counts)} entities, which is not the range "
        "between an empty board and a crowded one"
    )


def test_a_builder_agrees_with_itself_across_the_whole_corpus() -> None:
    """The baseline the port will slot into. Trivially true TODAY, and that is the point.

    It is here so that when the Rust builder arrives, the thing it must satisfy already
    exists and has not been shaped around it. On its own this proves nothing, which is
    what the perturbation tests below are for.
    """
    corpus = a_corpus()
    found = disagreements(bound(SpatialObsBuilder()), bound(SpatialObsBuilder()), corpus)
    assert not found, "a builder disagreed with itself:\n  " + "\n  ".join(found[:5])


@pytest.mark.parametrize(
    "mode", ["one cell", "swapped channels", "scaled", "flipped board", "vector"]
)
def test_the_harness_catches_a_builder_that_is_wrong(mode: str) -> None:
    """The half that makes the test above mean something.

    Each of these is a defect a port could plausibly ship, and none changes the shape. If
    the harness cannot tell these from the reference, then "the two builders agree" is a
    sentence about nothing.
    """
    corpus = a_corpus()
    found = disagreements(bound(SpatialObsBuilder()), bound(Perturbed(mode)), corpus)
    assert found, (
        f"the harness did not notice a builder perturbed by {mode!r}. An equality check "
        "that cannot refuse is not a gate, and this one is about to be the contract for a "
        "port."
    )


def test_the_harness_compares_both_seats_and_not_only_blue() -> None:
    """A defect on ONE seat is the shape a port most easily ships.

    Checked by perturbing a builder for a single seat: if the harness only ever looked at
    Blue, a Red-only defect would pass. Distinct from the tests above, which perturb both.
    """
    corpus = a_corpus()

    class RedOnly(SpatialObsBuilder):
        def build(self, state, team, mask):
            out = dict(super().build(state, team, mask))
            if team == TEAMS[1]:
                board = np.array(out["spatial"], copy=True)
                board[0, 0, 0] += np.float32(1.0)
                out["spatial"] = board
            return out

    found = disagreements(bound(SpatialObsBuilder()), bound(RedOnly()), corpus)
    assert found, (
        "a defect present only on the RED seat went unnoticed, so this harness is looking "
        "at one seat and reporting on two"
    )
    assert all("seat 1" in f for f in found), (
        f"a red-only perturbation produced disagreements on other seats too: {found[:3]}"
    )
