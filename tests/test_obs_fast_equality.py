"""The builder's plane functions are exactly what they were before the vectorized rewrite.

``tests/_obs_reference.py`` holds the per-entity versions verbatim (RoyaleGym 27b2a64). Here the
shipped functions and the reference run on the same states, real RustEngine battles with
evolutions, heroes, champions, tunnellers, buildings, rage, slow, shields, charges and the
Vines, for both seats, and must agree BYTE FOR BYTE: same dtype, same shape, same bytes, or the
same refusal with the same message. A full SpatialObsBuilder with every option a clone reads
(the rg2 inputs: card and unit identity with SAME_UNIT_ALIASES, spell identity with effects,
evolutions, heroes, unit status and actions, card status, button index, enemy queue) is built
over the same battles with the shipped functions and with the reference ones, and every key of
every observation must match. Then forged edge cases: ties on a tile, points past the edge,
refusals.

SKIPS
    Everything here needs the engine. Not a pass without it.
"""

from __future__ import annotations

import msgspec
import numpy as np
import pytest

import _obs_reference as ref
import royalegym.obs as obs_mod
from royalegym.action import TileActionParser
from royalegym.env import ClashParallelEnv
from royalegym.obs import SAME_UNIT_ALIASES, SpatialObsBuilder, unit_vocabulary
from royalegym.protocol import BLUE, RED, STATUS_HERO, EntityKind, ShuffleMode
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available
from royalegym.state_mutator import DefaultStateMutator, deck_ids

pytestmark = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))

DECKS = {
    "a": (["Musketeer", "Wallbreakers", "Ghost", "Miner", "GoblinDrill", "Tesla", "Rage",
           "IceWizard"], [2, 1, 1, 0, 0, 0, 0, 0]),
    "b": (["GoldenKnight", "DarkPrince", "Prince", "ZapMachine", "Clone", "Vines", "Firecracker",
           "Graveyard"], [0, 2, 0, 0, 0, 0, 1, 0]),
    "c": (["Wizard", "Cannon", "Balloon", "InfernoTower", "Freeze", "Tornado", "ElectroWizard",
           "SkeletonArmy"], [2, 1, 0, 0, 0, 0, 0, 1]),
    "d": (["LittlePrince", "GoblinBarrel", "Witch", "RoyalHogs", "Bowler", "Log", "Poison",
           "Goblins"], [0, 1, 0, 1, 2, 0, 0, 0]),
}
PAIRS = (("a", "b"), ("c", "d"), ("b", "c"))
#: Every option a clone reads (Train's rg2 inputs), plus the ones built since.
RG2 = dict(
    card_identity=True, spell_aim_after_ticks=0, spell_identity=True, effect_identity=True,
    evolutions=True, evolution_progress=True, heroes=True, unit_status=True, card_status=True,
    unit_actions=True, unit_identity=True, unit_aliases=dict(SAME_UNIT_ALIASES),
    button_index=True, enemy_queue=True,
)
NAMES = ("entity_channels", "spell_channels", "evolved_channels", "hero_channels",
         "status_channels", "action_channels", "card_id_planes", "unit_id_planes")


def _env(pair, builder=None):
    eng = RustEngine()
    (na, fa), (nb, fb) = DECKS[pair[0]], DECKS[pair[1]]
    mutator = DefaultStateMutator(decks=[deck_ids(na, eng.cards()), deck_ids(nb, eng.cards())],
                                  forms=[fa, fb], shuffle=ShuffleMode.INDEPENDENT)
    return ClashParallelEnv(eng, action_parser=TileActionParser(ability_buttons=True),
                            obs_builder=builder or SpatialObsBuilder(), state_mutator=mutator)


def _battle(env, seed, steps=None):
    """Random legal play, both seats, presses included; yields the state after each step."""
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    yield env.battle_state
    n = 0
    while env.agents and (steps is None or n < steps):
        acts = {}
        for agent in env.agents:
            legal = np.flatnonzero(env.action_masks(agent))
            legal = legal[legal != 0]
            acts[agent] = int(rng.choice(legal)) if len(legal) and rng.random() < 0.5 else 0
        env.step(acts)
        n += 1
        yield env.battle_state


@pytest.fixture(scope="module")
def battles():
    """(engine, states) per pair, played once for the module."""
    out = []
    for i, pair in enumerate(PAIRS):
        env = _env(pair)
        out.append((env.engine, list(_battle(env, 11 + i))))
    return out


def _call(fn, *args, **kw):
    try:
        return "ok", fn(*args, **kw)
    except ValueError as e:
        return "refused", str(e)


def _same(got, want, where):
    assert got[0] == want[0], (where, got[0], want[0], got[1] if got[0] == "refused" else "")
    if got[0] == "refused":
        assert got[1] == want[1], where
        return
    g, w = got[1], want[1]
    assert g.dtype == w.dtype, (where, g.dtype, w.dtype)
    assert g.shape == w.shape, (where, g.shape, w.shape)
    assert g.tobytes() == w.tobytes(), (where, np.argwhere(g != w)[:5].tolist())


def _args(name, engine, state, team, lookup):
    arena = engine.arena()
    if name == "entity_channels":
        return [((state.entities, team, arena), {}),
                ((state.entities, team, arena), {"grounded_said": True})]
    if name in ("spell_channels", "action_channels"):
        return [((state, team, arena), {})]
    if name == "hero_channels":
        champions = frozenset(c.card_id for c in engine.cards() if c.champion)
        return [((state, team, arena), {}), ((state, team, arena, champions), {})]
    if name == "card_id_planes":
        return [((state.entities, team, arena, len(engine.cards())), {})]
    if name == "unit_id_planes":
        n = len(engine.unit_types())
        return [((state.entities, team, arena, n), {}),
                ((state.entities, team, arena, n, lookup), {})]
    return [((state.entities, team, arena), {})]


def test_every_plane_function_matches_the_reference_on_real_battles(battles):
    seen = {name: 0 for name in NAMES}
    for engine, states in battles:
        _, lookup = unit_vocabulary(engine.unit_types(), SAME_UNIT_ALIASES)
        for i, state in enumerate(states):
            for team in (BLUE, RED):
                for name in NAMES:
                    for args, kw in _args(name, engine, state, team, lookup):
                        got = _call(getattr(obs_mod, name), *args, **kw)
                        want = _call(getattr(ref, name), *args, **kw)
                        _same(got, want, (name, i, team))
                        seen[name] += got[0] == "ok"
    assert all(n > 1000 for n in seen.values()), seen


def test_the_battles_reach_what_the_planes_count(battles):
    """Vacuity: the corpus holds the things each plane counts, on both sides."""
    found = {"hero": 0, "evolved": 0, "shield": 0, "stun": 0, "dest": 0, "charge": 0,
             "rage": 0, "deploying": 0, "spells": 0, "building": 0, "tie": 0}
    for engine, states in battles:
        for s in states:
            found["spells"] += bool(s.spells)
            tiles = set()
            for e in s.entities:
                status = max(e.status_flags, 0)
                found["hero"] += bool(status & STATUS_HERO)
                found["evolved"] += bool(status & 8)
                found["shield"] += e.shield > 0
                found["stun"] += e.stun_ticks > 0
                found["dest"] += e.dest_x >= 0
                found["charge"] += e.charge > 0
                found["rage"] += any("Rage" in name for name, _ in e.buffs)
                found["deploying"] += e.deploy_ticks > 0
                found["building"] += e.kind == EntityKind.BUILDING
                tile = (e.team, e.x // engine.arena().subtile, e.y // engine.arena().subtile)
                found["tie"] += tile in tiles
                tiles.add(tile)
    assert all(n >= 10 for n in found.values()), found


def _observe_all(states, engine, builder):
    parser = TileActionParser(ability_buttons=True)
    parser.bind(engine)
    builder.bind(engine, parser)
    builder.reset(states[0])
    out = []
    for s in states:
        for team in (BLUE, RED):
            out.append(builder.build(s, team, parser.action_mask(s, team)))
    return out


def test_a_full_rg2_builder_matches_the_reference_on_every_key(battles, monkeypatch):
    engine, states = battles[0]
    fast = _observe_all(states, engine, SpatialObsBuilder(**RG2))
    for name in NAMES:
        monkeypatch.setattr(obs_mod, name, getattr(ref, name))
    slow = _observe_all(states, engine, SpatialObsBuilder(**RG2))
    assert len(fast) == len(slow) > 200
    for i, (f, s) in enumerate(zip(fast, slow, strict=True)):
        assert f.keys() == s.keys()
        for key in f:
            assert f[key].dtype == s[key].dtype, (i, key)
            assert f[key].tobytes() == s[key].tobytes(), (i, key)


def test_forged_ties_edges_and_refusals_match_the_reference(battles):
    """Several units on one tile (the lowest uid holds an id plane), points past the arena's
    edge, and every refusal (no status bits, a unit type not said, a card outside the
    catalogue, an enemy hero no button row names)."""
    engine, states = battles[1]
    arena = engine.arena()
    base = max(states, key=lambda s: len(s.entities))
    troops = [e for e in base.entities if e.kind == EntityKind.TROOP]
    assert len(troops) >= 3
    pile = [msgspec.structs.replace(e, x=troops[0].x, y=troops[0].y, uid=900 - k)
            for k, e in enumerate(troops[:3])]
    edge = [msgspec.structs.replace(troops[0], uid=950, x=-500, y=arena.height + 900, dest_x=-3,
                                    dest_y=arena.height + 50)]
    cases = {
        "pile": [*base.entities, *pile],
        "edge": [*base.entities, *edge],
        "no_status": [msgspec.structs.replace(base.entities[0], status_flags=-1),
                      *base.entities[1:]],
        "unit_unsaid": [*base.entities, msgspec.structs.replace(troops[1], uid=1, unit_type=-1)],
        "card_outside": [*base.entities,
                         msgspec.structs.replace(troops[1], uid=2, card_id=10_000)],
        "hero_no_row": [*base.entities,
                        msgspec.structs.replace(troops[1], uid=3, team=RED,
                                                status_flags=STATUS_HERO, card_id=0)],
    }
    _, lookup = unit_vocabulary(engine.unit_types(), SAME_UNIT_ALIASES)
    for label, entities in cases.items():
        state = msgspec.structs.replace(base, entities=entities)
        for team in (BLUE, RED):
            for name in NAMES:
                for args, kw in _args(name, engine, state, team, lookup):
                    _same(_call(getattr(obs_mod, name), *args, **kw),
                          _call(getattr(ref, name), *args, **kw), (label, name, team))
