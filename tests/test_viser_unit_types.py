"""The viewer names each unit by its OWN type, not only its card (Viser, 2026-10-10).

A unit's ``name`` is its CARD (``card_id``), so the Witch's Skeletons read "Witch".
``EntityState.unit_type`` indexes ``Engine.unit_types()``; ``unit_dict`` puts that name in
``extra["unit"]`` when it is handed a way to read it and the engine said a type, and leaves the key
out otherwise, so a frame from an engine or a trace that names no types is unchanged. The env
hands the live publisher the engine's names once per battle (``ViserPublisher.set_unit_types``),
so a duck-typed publisher written before them keeps working. Traces carry the names in
``TraceHeader.unit_types`` (since 2026-10-05).

SKIPS
    The engine tests need the engine. Not a pass without it.
"""

from __future__ import annotations

import pytest

from royalegym import viser as viser_mod
from royalegym.env import ClashParallelEnv
from royalegym.mock_engine import MockEngine
from royalegym.protocol import BLUE, MatchSetup, ShuffleMode, SpawnSpec
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available

needs_engine = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))


def _witch_battle():
    """A RustEngine battle with a Blue Witch on the board long enough to have made Skeletons."""
    engine = RustEngine()
    ids = {c.name: c.card_id for c in engine.cards()}
    deck = [ids[n] for n in ("Knight", "Archer", "Giant", "Minions", "Fireball", "Zap",
                             "Musketeer", "Valkyrie")]
    t = engine.arena().subtile
    engine.reset(2, MatchSetup(
        decks=[deck, deck], shuffle=ShuffleMode.NONE,
        spawns=[SpawnSpec(BLUE, ids["Witch"], 9 * t, 6 * t)]))
    for _ in range(60):
        engine.step([], 10)
        state = engine.state()
        names = engine.unit_types()
        if any(e.unit_type >= 0 and names[e.unit_type] != "Witch"
               and e.card_id == ids["Witch"] for e in state.entities):
            return engine, state
    pytest.fail("no unit of the Witch's card with another type in 600 ticks: nothing to check")


@needs_engine
def test_a_unit_carries_its_own_type_when_one_is_said():
    engine, state = _witch_battle()
    cards = {c.card_id: c.name for c in engine.cards()}
    names = list(engine.unit_types())
    of = viser_mod.unit_namer(names)
    d = viser_mod.frame_dict(state, cards.__getitem__, engine.arena().subtile, unit_type_of=of)
    typed = [(u, e) for u, e in zip(d["units"], state.entities, strict=True) if e.unit_type >= 0]
    assert typed, "no unit states a type"
    for u, e in typed:
        assert u["extra"]["unit"] == names[e.unit_type]
    assert any(u["name"] == "Witch" and u["extra"]["unit"] != "Witch" for u, _ in typed), (
        "no Skeleton under the Witch's card: the test shows nothing"
    )
    for u, e in zip(d["units"], state.entities, strict=True):
        if e.unit_type < 0:
            assert "unit" not in u["extra"]


@needs_engine
def test_without_names_a_frame_is_as_before():
    engine, state = _witch_battle()
    cards = {c.card_id: c.name for c in engine.cards()}
    d = viser_mod.frame_dict(state, cards.__getitem__, engine.arena().subtile)
    assert all("unit" not in u["extra"] for u in d["units"])
    assert viser_mod.unit_namer([]) is None
    assert viser_mod.unit_namer(None) is None


def test_an_index_past_the_names_is_left_unnamed():
    of = viser_mod.unit_namer(["Skeleton"])
    assert of(0) == "Skeleton"
    assert of(1) is None


class _Spy:
    """A publisher that records what the env hands it; ``set_unit_types`` optional."""

    def __init__(self, setter: bool) -> None:
        self.attached = True
        self.published = 0
        if setter:
            self.unit_types = None
            self.set_unit_types = lambda names: setattr(self, "unit_types", names)

    def publish(self, state, cards, arena, decks, events=()) -> None:
        self.published += 1

    def close(self) -> None:
        pass


@needs_engine
def test_the_env_hands_the_live_publisher_the_engines_names():
    spy = _Spy(setter=True)
    env = ClashParallelEnv(engine=RustEngine(), viser=spy)
    env.reset(seed=1)
    assert spy.unit_types == list(env.engine.unit_types())
    env.close()


@pytest.mark.parametrize("engine", ["mock", "rust"])
def test_a_publisher_without_the_setter_still_works(engine):
    if engine == "rust" and not core_available():
        pytest.skip(f"SKIPPED, NOT PASSED: {CORE_IMPORT_ERROR}")
    spy = _Spy(setter=False)
    env = ClashParallelEnv(engine=MockEngine() if engine == "mock" else RustEngine(), viser=spy)
    env.reset(seed=1)
    env.step({"blue": 0, "red": 0})
    assert spy.published > 1
    env.close()


def test_the_real_publisher_uses_the_names_it_was_handed(monkeypatch):
    """``ViserPublisher.publish`` builds its frame with the names ``set_unit_types`` gave it."""
    seen = {}
    monkeypatch.setattr(viser_mod, "frame_dict", lambda *a, **k: seen.update(k) or {})
    pub = viser_mod.ViserPublisher(port=0)
    monkeypatch.setattr(type(pub), "attached", property(lambda self: True))
    monkeypatch.setattr(pub, "publish_dict", lambda d: True)
    monkeypatch.setattr(pub, "_note_units", lambda *a: None)
    pub.set_unit_types(["Skeleton"])
    env = ClashParallelEnv()
    env.reset(seed=1)
    pub.publish(env.battle_state, env.engine.cards(), env.engine.arena())
    assert seen["unit_type_of"](0) == "Skeleton"
    env.close()
    pub.close()
