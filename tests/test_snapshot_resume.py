"""A battle saved at tick T resumes as the battle that went on: same observation, same future.

A start mid-battle is only worth training from if the episode it begins IS the battle it was cut
from. The engine's state alone is not enough: each seat's observation remembers the match
(``MatchMemory``: the cycle, the counted enemy elixir, the cards seen; the enemy's forms; when
each enemy spell's aim became readable), and a resume that re-seeds that memory shows the policy
a different observation than the one it had at T. ``ClashParallelEnv.snapshot()`` saves both,
and a reset from it restores both. These tests play a battle on, resume a copy at T, and compare
every observation, reward and engine state step by step.
"""

from __future__ import annotations

import numpy as np
import pytest

from royalegym.action import TileActionParser
from royalegym.env import ClashParallelEnv
from royalegym.mock_engine import MockEngine
from royalegym.obs import SpatialObsBuilder
from royalegym.protocol import BLUE, RED
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available
from royalegym.state_mutator import (
    DefaultStateMutator,
    Snapshot,
    SnapshotStateMutator,
    load_snapshots,
    save_snapshots,
)

needs_engine = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))

DECKS = [
    ["Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Cannon", "Musketeer"],
    ["Valkyrie", "Goblins", "MiniPekka", "Arrows", "HogRider", "Skeletons", "Log",
     "GoblinBarrel"],
]


def _builder(full: bool) -> SpatialObsBuilder:
    """Every option that keeps a memory across steps, on the engine that can feed them all."""
    if not full:
        return SpatialObsBuilder(card_identity=True)
    return SpatialObsBuilder(
        card_identity=True, spell_aim_after_ticks=20, spell_identity=True, card_status=True,
        heroes=True, unit_status=True, unit_actions=True, evolutions=True,
        evolution_progress=True,
    )


def _env(rust: bool, delay: int = 0, **kw) -> ClashParallelEnv:
    engine = RustEngine(command_delay_ticks=delay) if rust else MockEngine()
    kw.setdefault("state_mutator", DefaultStateMutator(decks=DECKS))
    return ClashParallelEnv(
        engine, obs_builder=_builder(rust),
        action_parser=TileActionParser(ability_buttons=rust), **kw,
    )


def _actions(env: ClashParallelEnv, rng: np.random.Generator) -> dict[str, int]:
    """A legal play for each seat a third of the time, else wait."""
    out = {}
    for agent in env.agents:
        legal = np.flatnonzero(env.action_masks(agent))
        out[agent] = int(rng.choice(legal)) if rng.random() < 0.35 else 0
    return out


def _same_obs(a: dict, b: dict) -> list[str]:
    """The keys, per seat, where two observations differ."""
    return [
        f"{agent}.{key}"
        for agent in a
        for key in a[agent]
        if not np.array_equal(a[agent][key], b[agent][key])
    ]


def _cut(rust: bool, delay: int, steps: int = 70, seed: int = 3):
    """Play ``steps`` decisions, snapshot, then play on: the snapshot, the observation at the
    cut, and the continuation as (actions, obs, rewards, terminated, truncated, state hash)."""
    env = _env(rust, delay)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        env.step(_actions(env, rng))
    snap = env.snapshot()
    at_cut = {a: {k: np.copy(v) for k, v in o.items()} for a, o in env._obs.items()}
    memory = {t: env.obs_builder.memory[t] for t in (BLUE, RED)}
    assert all(m.foe_plays > 0 for m in memory.values()), "vacuous: a seat has seen no play"
    rows = []
    for _ in range(60):
        acts = _actions(env, rng)
        obs, rew, term, trunc, _ = env.step(acts)
        rows.append((acts, obs, rew, term, trunc, env.engine.state_hash()))
        if not env.agents:
            break
    return snap, at_cut, rows


@pytest.mark.parametrize("delay", [0, 7])
@needs_engine
def test_a_reset_from_a_snapshot_sees_and_plays_what_the_battle_did(delay):
    """The first observation after the reset is the one at T, for both seats and every key, and
    every step after it matches the original battle: observation, reward, done flags, engine."""
    snap, at_cut, rows = _cut(rust=True, delay=delay)
    resumed = _env(True, delay)
    obs, _ = resumed.reset(seed=999, options={"snapshot": snap})
    assert _same_obs(obs, at_cut) == []
    for i, (acts, want, rew, term, trunc, digest) in enumerate(rows):
        got, got_rew, got_term, got_trunc, _ = resumed.step(acts)
        assert resumed.engine.state_hash() == digest, f"engine state differs at step {i}"
        assert _same_obs(got, want) == [], f"step {i}"
        assert (got_rew, got_term, got_trunc) == (rew, term, trunc), f"step {i}"


@needs_engine
def test_the_resume_needs_the_memory_and_not_only_the_engine():
    """The null: the same engine state with the memory left out (a plain ``Snapshot(blob)``,
    which re-seeds the memory at T) shows a different first observation, so the test above
    is measuring the memory and not passing on the engine state alone."""
    snap, at_cut, _ = _cut(rust=True, delay=0)
    resumed = _env(True)
    obs, _ = resumed.reset(options={"snapshot": Snapshot(snap.blob)})
    assert _same_obs(obs, at_cut), "the memory made no difference to the observation"


def test_mock_engine_resumes_the_same_way():
    snap, at_cut, rows = _cut(rust=False, delay=0)
    resumed = _env(False)
    obs, _ = resumed.reset(options={"snapshot": snap})
    assert _same_obs(obs, at_cut) == []
    for i, (acts, want, rew, term, trunc, digest) in enumerate(rows):
        got, got_rew, got_term, got_trunc, _ = resumed.step(acts)
        assert resumed.engine.state_hash() == digest, f"engine state differs at step {i}"
        assert _same_obs(got, want) == [], f"step {i}"
        assert (got_rew, got_term, got_trunc) == (rew, term, trunc), f"step {i}"


def test_the_loaded_memory_is_the_saved_memory():
    """Attribute by attribute, for every memory the builder keeps: what ``load_memory`` puts
    back is what ``save_memory`` took, so a field added to a memory later is carried too."""
    env = _env(False)
    env.reset(seed=5)
    rng = np.random.default_rng(5)
    for _ in range(50):
        env.step(_actions(env, rng))
    blob = env.obs_builder.save_memory()
    other = _env(False)
    other.reset(seed=6)
    other.engine.load_state(env.engine.save_state())
    other.obs_builder.load_memory(blob, other.engine.state())
    for t in (BLUE, RED):
        for name, holder in (("memory", "memory"), ("enemy_forms", "enemy_forms")):
            want = vars(getattr(env.obs_builder, holder)[t])
            got = vars(getattr(other.obs_builder, holder)[t])
            assert sorted(want) == sorted(got)
            for key, value in want.items():
                if isinstance(value, np.ndarray):
                    assert np.array_equal(got[key], value), key
                    assert got[key].dtype == value.dtype, key
                else:
                    assert got[key] == value, f"{name}[{t}].{key}"
                    assert type(got[key]) is type(value), f"{name}[{t}].{key}"


def test_the_spell_aim_clock_is_carried():
    """The clock dates enemy spells by sight on an engine that does not say how long a spell
    has flown; its dates are keyed by (team, card, x, y) tuples, which the saved bytes must give
    back as the same keys. RustEngine reports the flight, so the resume tests above never read
    the clock: this checks it directly."""
    make = lambda: ClashParallelEnv(  # noqa: E731
        MockEngine(), obs_builder=SpatialObsBuilder(spell_aim_after_ticks=20),
        state_mutator=DefaultStateMutator(decks=DECKS),
    )
    env = make()
    env.reset(seed=1)
    env.step({"blue": 0, "red": 0})
    clock = env.obs_builder._aim_clock
    clock.starts.update({(RED, 3, 1800, 5400): 4, (BLUE, 11, 900, 900): 9})
    blob = env.obs_builder.save_memory()
    other = make()
    other.reset(seed=2)
    other.engine.load_state(env.engine.save_state())
    other.obs_builder.load_memory(blob, other.engine.state())
    got = other.obs_builder._aim_clock
    assert got.starts == clock.starts
    assert got.tick == clock.tick
    assert all(type(k) is tuple for k in got.starts)


def test_a_memory_from_another_moment_or_catalogue_is_refused():
    env = _env(False)
    env.reset(seed=1)
    for _ in range(5):
        env.step({"blue": 0, "red": 0})
    blob = env.obs_builder.save_memory()
    later = env.engine.state()
    env.step({"blue": 0, "red": 0})
    with pytest.raises(ValueError, match="tick"):
        env.obs_builder.load_memory(blob, env.engine.state())
    narrow = ClashParallelEnv(
        MockEngine(card_names=DECKS[0] + DECKS[1]), obs_builder=_builder(False),
        state_mutator=DefaultStateMutator(decks=DECKS),
    )
    narrow.reset(seed=1)
    with pytest.raises(ValueError, match="catalogue"):
        narrow.obs_builder.load_memory(blob, later)


@needs_engine
def test_a_memory_that_did_not_count_what_this_builder_counts_is_refused():
    """A snapshot taken by a builder without ``card_status`` never counted the enemy's forms,
    and one without ``spell_aim_after_ticks`` never dated the enemy's spells; a builder that
    shows them cannot be handed nothing and show zeros."""
    env = ClashParallelEnv(RustEngine(), obs_builder=SpatialObsBuilder(),
                           state_mutator=DefaultStateMutator(decks=DECKS))
    env.reset(seed=1)
    env.step({"blue": 0, "red": 0})
    snap = env.snapshot()
    for builder, word in ((SpatialObsBuilder(card_status=True), "card_status"),
                          (SpatialObsBuilder(spell_aim_after_ticks=0), "spell_aim_after_ticks")):
        other = ClashParallelEnv(RustEngine(), obs_builder=builder,
                                 state_mutator=DefaultStateMutator(decks=DECKS))
        with pytest.raises(ValueError, match=word):
            other.reset(options={"snapshot": snap})


def _steps_to_done(env: ClashParallelEnv, options: dict) -> tuple[int, bool, bool, dict]:
    env.reset(seed=2, options=options)
    start = env.battle_state.tick
    while True:
        _, _, term, trunc, info = env.step({"blue": 0, "red": 0})
        if term["blue"] or trunc["blue"]:
            return env.battle_state.tick - start, term["blue"], trunc["blue"], info["blue"]


def test_max_ticks_truncates_the_episode_at_the_first_decision_past_it():
    env = _env(False)
    env.reset(seed=1)
    snap = env.snapshot(max_ticks=200)
    ticks, term, trunc, info = _steps_to_done(env, {"snapshot": snap})
    assert (term, trunc) == (False, True)
    assert 200 <= ticks < 200 + env.decision_ticks
    assert info["episode_ticks"] == ticks
    # A reset option wins over the snapshot's own, and works for a fresh battle too.
    ticks, _, trunc, _ = _steps_to_done(env, {"snapshot": snap, "max_ticks": 95})
    assert trunc
    assert 95 <= ticks < 95 + env.decision_ticks
    ticks, _, trunc, _ = _steps_to_done(env, {"max_ticks": 40})
    assert trunc
    assert 40 <= ticks < 40 + env.decision_ticks
    # And the next episode without one runs uncapped again.
    env.reset(seed=3)
    for _ in range(30):
        _, _, term, trunc, _ = env.step({"blue": 0, "red": 0})
        assert not term["blue"]
        assert not trunc["blue"]
    for bad in (0, -5, 2.5, True):
        with pytest.raises(ValueError, match="max_ticks"):
            env.reset(options={"max_ticks": bad})


def test_the_start_names_its_seat_and_tag_in_the_infos():
    env = _env(False)
    env.reset(seed=1)
    snap = env.snapshot(seat=RED, max_ticks=20, tag="defend-bridge")
    _, infos = env.reset(options={"snapshot": snap})
    assert infos["blue"]["start_seat"] == RED
    assert infos["red"]["start_seat"] == RED
    *_, info = _steps_to_done(env, {"snapshot": snap})
    assert info["start_tag"] == "defend-bridge"
    # A battle from a setup names neither.
    _, infos = env.reset(seed=1)
    assert "start_seat" not in infos["blue"]
    *_, info = _steps_to_done(env, {"max_ticks": 20})
    assert "start_tag" not in info


def _bank() -> list[Snapshot]:
    return [
        Snapshot(b"a", seat=BLUE, tag="horde"),
        Snapshot(b"b", seat=RED, tag="horde"),
        Snapshot(b"c", seat=BLUE, tag="overtime"),
        Snapshot(b"d", tag="overtime"),
    ]


def _draws(m: SnapshotStateMutator, n: int = 4000) -> dict[bytes, int]:
    rng = np.random.default_rng(0)
    out: dict[bytes, int] = {}
    for _ in range(n):
        s = m.build(rng, [])
        out[s.blob] = out.get(s.blob, 0) + 1
    return out


def test_the_bank_is_drawn_by_weight_and_by_seat():
    bank = _bank()
    # A seat draws only the starts made for it, and the ones made for either seat.
    assert set(_draws(SnapshotStateMutator(bank, seat="blue"))) == {b"a", b"c", b"d"}
    assert set(_draws(SnapshotStateMutator(bank, seat="red"))) == {b"b", b"d"}
    assert set(_draws(SnapshotStateMutator(bank))) == {b"a", b"b", b"c", b"d"}
    # Weights by tag: every tag in the bank must be named, so none is dropped by a typo.
    m = SnapshotStateMutator(bank, weights={"horde": 3.0, "overtime": 1.0})
    got = _draws(m)
    assert abs((got[b"a"] + got[b"b"]) / 4000 - 0.75) < 0.03
    with pytest.raises(ValueError, match="overtime"):
        SnapshotStateMutator(bank, weights={"horde": 1.0})
    # Weights per start, and changed between episodes.
    m.set_weights([0, 0, 0, 1])
    assert set(_draws(m, 50)) == {b"d"}
    with pytest.raises(ValueError, match="seat"):
        SnapshotStateMutator([Snapshot(b"a", seat=BLUE)], seat="red")
    with pytest.raises(ValueError, match="seat"):
        SnapshotStateMutator(bank, seat="green")


def test_a_bank_round_trips_through_a_file(tmp_path):
    env = _env(False)
    env.reset(seed=4)
    for _ in range(12):
        env.step({"blue": 0, "red": 0})
    bank = [env.snapshot(seat=BLUE, max_ticks=300, tag="t1"), Snapshot(b"raw")]
    path = tmp_path / "bank.snapshots"
    save_snapshots(path, bank)
    assert load_snapshots(path) == bank
    m = SnapshotStateMutator(str(path), seat="blue")
    assert m.config() == {"snapshots": str(path), "seat": "blue", "weights": None}
    (tmp_path / "junk").write_bytes(b"\x93\x01\x02\x03")
    with pytest.raises(ValueError, match="snapshot"):
        load_snapshots(tmp_path / "junk")


def test_a_plain_blob_still_resumes_as_before():
    """``SnapshotStateMutator([bytes])`` and ``options={"snapshot": bytes}``: the engine state,
    with the memory seeded at it, as before this change."""
    env = _env(False)
    env.reset(seed=1)
    blob = env.engine.save_state()
    for start in ({"snapshot": blob}, None):
        other = _env(False, state_mutator=SnapshotStateMutator([blob]))
        _, infos = other.reset(options=start)
        assert other.battle_state.tick == env.battle_state.tick
        assert "start_seat" not in infos["blue"]
