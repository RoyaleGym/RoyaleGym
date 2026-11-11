"""``mirror_state`` rotates every engine-frame value, so a seat flip can test what reads them.

The seat-flip gate builds Blue on a state and Red on its mirror, and asks for the same
observation. That holds only if ``mirror_state`` turns EVERY engine-frame quantity 180 degrees.
It did not turn a unit's ``facing``, a tunneller's ``dest_x``/``dest_y``, a waiting play's tap
(``PlayerState.pending``) or the projectiles (2026-11-11), and the gate's states come from
MockEngine, which leaves all four at their defaults, so nothing noticed. A builder reading them
for Red would be wrong while the gate stayed green. Here they are checked on a forged state, and
on played engine states.

SKIPS
    The engine test skips only without the engine.
"""

from __future__ import annotations

import msgspec
import numpy as np
import pytest

from royalegym.mock_engine import MockEngine
from royalegym.protocol import (
    BLUE,
    RED,
    EntityKind,
    EntityState,
    MatchSetup,
    ProjectileState,
    mirror_state,
)
from royalegym.rust_engine import CORE_IMPORT_ERROR, core_available


def _forged():
    eng = MockEngine()
    eng.reset(1, MatchSetup(decks=[list(range(8)), list(range(8))]))
    s, a = eng.state(), eng.arena()
    miner = EntityState(41, BLUE, int(EntityKind.TROOP), 3, -1, 90000, 200000, 100, 100, 500,
                        False, 0, facing=(120, -230), dest_x=70000, dest_y=400000)
    walker = EntityState(43, RED, int(EntityKind.TROOP), 3, -1, 200000, 300000, 100, 100, 500,
                         False, 0, facing=(-256, 0))
    shot = ProjectileState(RED, 150000, 250000, 160000, 120000, 41, 0, 3)
    play = [0, 5, 108000, 162000, 12, 3000]  # a play waiting: card 5, tap (6, 9) tiles
    press = [1, 4, 0, 0, 7, 2000]  # a press waiting: it has no tap
    players = [msgspec.structs.replace(s.players[BLUE], pending=[play, press]), s.players[RED]]
    return a, msgspec.structs.replace(
        s, entities=[*s.entities, miner, walker], projectiles=[shot], players=players
    )


def test_facing_dest_pending_and_projectiles_turn_with_the_board():
    a, s = _forged()
    m = mirror_state(a, s)
    miner = next(e for e in m.entities if e.uid == 41)
    walker = next(e for e in m.entities if e.uid == 43)
    assert miner.facing == (-120, 230)
    assert (miner.dest_x, miner.dest_y) == (a.width - 70000, a.height - 400000)
    assert walker.facing == (256, 0)
    assert (walker.dest_x, walker.dest_y) == (-1, -1), "'none' stays 'none', not width + 1"
    shot = m.projectiles[0]
    assert (shot.team, shot.x, shot.y) == (BLUE, a.width - 150000, a.height - 250000)
    assert (shot.aim_x, shot.aim_y) == (a.width - 160000, a.height - 120000)
    assert (shot.target_uid, shot.splash, shot.firer_card_id) == (41, 0, 3)
    play, press = m.players[RED].pending
    assert play == [0, 5, a.width - 108000, a.height - 162000, 12, 3000]
    assert press == [1, 4, 0, 0, 7, 2000], "a press has no tap to turn"
    assert m.players[BLUE].pending == []


def test_mirroring_twice_is_the_original():
    a, s = _forged()
    assert mirror_state(a, mirror_state(a, s)) == s


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_red_on_a_mirrored_engine_state_sees_what_blue_sees_with_unit_actions():
    """Played engine states (a Miner tunnels, so a landing tile is set): Blue on the state and Red
    on its mirror build the same planes, the ones that read dest among them.

    Not the deploy zone or the mask: they follow the engine's deploy rules, which on real
    (not tile-centred) positions are not seat-symmetric at tile edges (rust_engine's rotation
    probe measures the engine's own asymmetry; the mask agrees with the engine for both seats,
    action.mask_disagreements). The MockEngine gate in test_env_obs.py keeps those exact."""
    from royalegym import make_env
    from royalegym.action import TileActionParser
    from royalegym.obs import SpatialObsBuilder

    deck = ["Miner", "Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Musketeer"]
    env = make_env(deck=deck, obs_builder=SpatialObsBuilder(unit_actions=True, unit_status=True))
    builder = SpatialObsBuilder(unit_actions=True, unit_status=True)
    parser = TileActionParser()
    parser.bind(env.engine)
    builder.bind(env.engine, parser)
    arena = env.engine.arena()
    rng = np.random.default_rng(7)
    obs, _ = env.reset(seed=7)
    names = builder.channel_names()
    tunnelling, dest_shown, bad = 0, 0, []
    for _ in range(400):
        if not env.agents:
            break
        acts = {}
        for agent in env.agents:
            legal = np.flatnonzero(obs[agent]["action_mask"])[1:]
            acts[agent] = int(rng.choice(legal)) if legal.size and rng.random() < 0.5 else 0
        obs, *_ = env.step(acts)
        s = env.battle_state
        if not any(e.dest_x >= 0 for e in s.entities):
            continue
        tunnelling += 1
        m = mirror_state(arena, s)
        ob = builder.build(s, BLUE, parser.action_mask(s, BLUE))["spatial"]
        orr = builder.build(m, RED, parser.action_mask(m, RED))["spatial"]
        bad += [f"tick {s.tick}: {names[i]}" for i in range(len(names))
                if names[i] != "enemy_troop_zone" and not np.array_equal(ob[i], orr[i])]
        dest_shown += int(ob[names.index("own_tunnel_dest")].sum()
                          + ob[names.index("enemy_tunnel_dest")].sum())
    assert tunnelling >= 3, f"vacuous: a Miner tunnelled on only {tunnelling} states"
    assert dest_shown >= 3, "vacuous: no landing tile was drawn"
    assert bad == [], bad[:5]
