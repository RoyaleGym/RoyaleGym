"""Every spatial plane is a multiple of 1/1000 that float32 holds exactly, as RoyaleImitate stores.

RoyaleImitate's shards keep each spatial plane as uint16 value x 1000 and refuse a row whose
value does not come back exactly (royaleimitate/shards.py ``_exact``): float32 value -> rint(x
1000) -> uint16 -> float32 / 1000 must give the same float32. A plane that held an unrounded
ratio (``own_hp_frac`` did, hp / max hp) made converting human replays fail on almost every
plan. The builder quantises rather than the shard writer, so a live bot reads exactly what the
clone trained on.

SKIPS
    The played-battle test skips only without the engine.
"""

from __future__ import annotations

import msgspec
import numpy as np
import pytest

from royalegym.mock_engine import MockEngine
from royalegym.obs import SpatialObsBuilder, status_channels
from royalegym.protocol import EntityKind, EntityState, MatchSetup
from royalegym.rust_engine import CORE_IMPORT_ERROR, _core, core_available


def stored_exactly(planes: np.ndarray) -> np.ndarray:
    """RoyaleImitate's rule, cell by cell: True where uint16 x 1000 gives the value back."""
    planes = planes.astype(np.float32)
    quantum = np.rint(planes.astype(np.float64) * 1000)
    stored = np.clip(quantum, 0, 65535).astype(np.uint16)
    back = stored.astype(np.float32) / np.float32(1000)
    return (quantum >= 0) & (quantum <= 65535) & (back == planes)


def test_hp_frac_is_permille_and_stored_exactly():
    eng = MockEngine()
    eng.reset(1, MatchSetup(decks=[list(range(8)), list(range(8))]))
    state, a = eng.state(), eng.arena()
    towers = [msgspec.structs.replace(e, status_flags=0) for e in state.entities
              if e.kind in (EntityKind.KING_TOWER, EntityKind.PRINCESS_TOWER)]
    rng = np.random.default_rng(0)
    units = []
    for uid in range(100, 400):
        max_hp = int(rng.integers(1, 9000))
        hp = int(rng.integers(0, max_hp + 1))
        x, y = int(rng.integers(0, a.width)), int(rng.integers(0, a.height))
        units.append(EntityState(uid, int(rng.integers(0, 2)), int(EntityKind.TROOP), 3, -1, x, y,
                                 hp, max_hp, 500, False, 0, status_flags=0))
    for team in (0, 1):
        planes = status_channels(towers + units, team, a)
        bad = ~stored_exactly(planes)
        assert not bad.any(), f"{int(bad.sum())} cells, e.g. {planes[bad][:3]}"
    one = EntityState(1, 0, int(EntityKind.TROOP), 3, -1, 9000, 9000, 1838, 2001, 500, False, 0,
                      status_flags=0)
    frac = status_channels([*towers, one], 0, a)[10].max()
    assert frac == np.float32(919) / np.float32(1000), "1838 / 2001 = 0.91854 -> 919 permille"


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_every_plane_of_a_played_battle_is_stored_exactly():
    """Every optional plane this engine supports, on, over a played battle."""
    from royalegym import make_env

    ship35 = "charge" in getattr(_core, "ENTITY_FIELDS", ())
    builder = SpatialObsBuilder(
        evolutions=True, spell_aim_after_ticks=0, heroes=True, unit_status=True,
        unit_actions=ship35,
    )
    env = make_env(
        deck=["Skeletons", "Musketeer", "Knight", "Archer", "Giant", "Minions", "Fireball", "Zap"],
        evolved=["Skeletons"], heroes=["Musketeer"], obs_builder=builder,
    )
    rng = np.random.default_rng(5)
    obs, _ = env.reset(seed=3)
    names = builder.channel_names()
    steps = 0
    while env.agents and steps < 400:
        for agent in env.agents:
            sp = obs[agent]["spatial"]
            bad = ~stored_exactly(sp)
            if bad.any():
                plane = int(np.argwhere(bad)[0][0])
                pytest.fail(f"{names[plane]} holds {sp[bad][0]!r}, not stored exactly")
        acts = {}
        for a in env.agents:
            legal = np.flatnonzero(obs[a]["action_mask"])[1:]
            acts[a] = int(rng.choice(legal)) if legal.size and rng.random() < 0.5 else 0
        obs, *_ = env.step(acts)
        steps += 1
    assert steps > 100
