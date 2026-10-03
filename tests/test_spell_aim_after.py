"""An enemy spell's landing point, shown only once a player could read it off the screen.

The client never draws an enemy spell's target. A thrown spell (FLIGHT) shows it through its arc
once it has flown for a while, so ``SpatialObsBuilder(spell_aim_after_ticks=k)`` adds the plane
``enemy_spell_aim_seen``: an enemy spell counted at its landing tile once it has FLOWN k ticks,
counted from when it starts moving, not from the throw. A rolling spell's path is drawn on the
ground and an area spell sits on its target, so those count from the first sight. Own spells are
never in this plane (``own_spell_aim`` has them). Off by default.

SKIPS
    The engine-backed test skips without the engine. Not a pass.
"""

from __future__ import annotations

import msgspec
import numpy as np
import pytest

from royalegym import make_env
from royalegym.obs import Reveal, SpatialObsBuilder, spatial_channels
from royalegym.protocol import BLUE, RED, SpellMotion, SpellState, to_own
from royalegym.rust_engine import CORE_IMPORT_ERROR, core_available

needs_engine = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
K = 10


def test_off_by_default_and_placed_before_any_reveal():
    assert "enemy_spell_aim_seen" not in [n for n, _ in spatial_channels()]
    names = [n for n, _ in spatial_channels(Reveal(enemy_spell_aim=True), True, True)]
    fair = [n for n, _ in spatial_channels(None, True)]
    assert names[: len(fair)] == fair
    assert names[len(fair)] == "enemy_spell_aim_seen"
    assert names[-1] == "enemy_spell_aim"
    with pytest.raises(ValueError, match="spell_aim_after_ticks"):
        SpatialObsBuilder(spell_aim_after_ticks=-1)


def _setup():
    env = make_env(obs_builder=SpatialObsBuilder(spell_aim_after_ticks=K))
    env.reset(seed=0)
    builder = env.obs_builder
    plane = builder.channel_names().index("enemy_spell_aim_seen")
    a = env.engine.arena()
    return env, builder, plane, a


def _spell(team, motion, aim, delay=0, card=0):
    return SpellState(team, card, int(motion), aim[0], aim[1], aim[0], aim[1], delay, 0, 0, 0)


def _seen(env, builder, plane, tick, spells, viewer=BLUE):
    state = msgspec.structs.replace(env.battle_state, tick=tick, spells=spells)
    mask = np.ones(env.action_parser.n_actions, dtype=np.int8)
    return builder.build(state, viewer, mask)["spatial"][plane]


def _tile(a, viewer, point):
    ox, oy = to_own(a, viewer, *point)
    return oy // a.subtile, ox // a.subtile


def test_a_thrown_spell_shows_its_target_only_after_k_ticks_of_flight():
    env, builder, plane, a = _setup()
    aim = (a.subtile * 9 + a.subtile // 2, a.subtile * 6 + a.subtile // 2)
    builder.reset(env.battle_state)
    # Seen at tick 100 still waiting 20 ticks to move: it starts flying at 120.
    s = _seen(env, builder, plane, 100, [_spell(RED, SpellMotion.FLIGHT, aim, delay=20)])
    assert s.sum() == 0
    s = _seen(env, builder, plane, 125, [_spell(RED, SpellMotion.FLIGHT, aim)])
    assert s.sum() == 0, "5 ticks of flight is too early"
    s = _seen(env, builder, plane, 120 + K, [_spell(RED, SpellMotion.FLIGHT, aim)])
    assert s.sum() == 1
    assert s[_tile(a, BLUE, aim)] == 1


def test_first_seen_moving_counts_from_that_sight():
    """Never earlier than a player could know: a spell first seen already in flight is dated at
    that sight, which can only show its target later than the truth, not sooner."""
    env, builder, plane, a = _setup()
    aim = (a.subtile * 4, a.subtile * 8)
    builder.reset(env.battle_state)
    flying = [_spell(RED, SpellMotion.FLIGHT, aim)]
    assert _seen(env, builder, plane, 200, flying).sum() == 0
    assert _seen(env, builder, plane, 200 + K - 1, flying).sum() == 0
    assert _seen(env, builder, plane, 200 + K, flying).sum() == 1


def _flying(team, aim, flown, delay=0, card=0):
    """A FLIGHT spell from an engine that reports ``ticks_flown`` (RoyaleSim 0.1.4 on)."""
    return SpellState(
        team, card, int(SpellMotion.FLIGHT), aim[0], aim[1], aim[0], aim[1], delay, 0, 0, 0, flown
    )


def test_an_engine_that_reports_ticks_flown_is_read_exactly():
    """The engine says how long the spell has flown, so nothing is dated by sight: a spell
    first seen already K ticks into its flight shows at once (the clock would have held it
    K more ticks), one tick short of K stays hidden, and a waiting spell stays hidden."""
    env, builder, plane, a = _setup()
    aim = (a.subtile * 4, a.subtile * 8)
    builder.reset(env.battle_state)
    assert _seen(env, builder, plane, 500, [_flying(RED, aim, K)]).sum() == 1
    builder.reset(env.battle_state)
    assert _seen(env, builder, plane, 500, [_flying(RED, aim, K - 1)]).sum() == 0
    builder.reset(env.battle_state)
    assert _seen(env, builder, plane, 500, [_flying(RED, aim, 0, delay=20)]).sum() == 0


def test_a_row_without_ticks_flown_reads_as_not_reported():
    spell = msgspec.json.decode(
        msgspec.json.encode([1, 0, int(SpellMotion.FLIGHT), 10, 20, 10, 20, 0, 0, 0, 0]),
        type=SpellState,
    )
    assert spell.ticks_flown == -1
    assert _flying(RED, (1, 2), 7).ticks_flown == 7


def test_rolling_and_area_spells_count_from_the_first_sight():
    env, builder, plane, a = _setup()
    aim = (a.subtile * 9, a.subtile * 9)
    builder.reset(env.battle_state)
    for motion in (SpellMotion.AIRBORNE, SpellMotion.ROLLING, SpellMotion.AREA):
        assert _seen(env, builder, plane, 300, [_spell(RED, motion, aim)]).sum() == 1, motion


def test_own_spells_are_not_in_this_plane():
    env, builder, plane, a = _setup()
    aim = (a.subtile * 9, a.subtile * 9)
    builder.reset(env.battle_state)
    assert _seen(env, builder, plane, 300, [_spell(BLUE, SpellMotion.ROLLING, aim)]).sum() == 0
    assert _seen(env, builder, plane, 300, [_spell(RED, SpellMotion.ROLLING, aim)], RED).sum() == 0


def test_config_records_k():
    assert SpatialObsBuilder(spell_aim_after_ticks=K).config()["spell_aim_after_ticks"] == K
    assert "spell_aim_after_ticks" not in SpatialObsBuilder().config()


@needs_engine
def test_a_real_goblin_barrel_appears_after_k_ticks_of_its_flight():
    """Red throws a Goblin Barrel at Blue; Blue's plane stays empty until the barrel has flown K
    ticks, then shows its landing tile."""
    deck_blue = ["Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Cannon", "Musketeer"]
    # Red holds no other spell, so the plane can only ever show the barrel.
    deck_red = [
        "GoblinBarrel", "Archer", "Giant", "Minions", "Valkyrie", "Musketeer", "Cannon", "Knight",
    ]
    env = make_env(
        deck=[deck_blue, deck_red], decision_ms=50,
        obs_builder=SpatialObsBuilder(spell_aim_after_ticks=K),
    )
    obs, _ = env.reset(seed=0)
    p = env.action_parser
    plane = env.obs_builder.channel_names().index("enemy_spell_aim_seen")
    names = {c.card_id: c.name for c in env.engine.cards()}
    thrown, flight_start, seen = False, None, []
    for _ in range(1200):
        red = p.noop()
        if not thrown:
            hand = env.battle_state.players[RED].hand
            slot = next((s for s, c in enumerate(hand) if names.get(c) == "GoblinBarrel"), None)
            if slot is not None:
                a = p.encode(slot, 9, 26)  # red's own frame: deep in blue's half
                if obs["red"]["action_mask"][a]:
                    red, thrown = a, True
            else:
                # Cycle the hand: play any legal move of another card until the barrel comes.
                legal = np.flatnonzero(obs["red"]["action_mask"][1 : p.n_tile_actions]) + 1
                red = int(legal[0]) if legal.size else p.noop()
        obs, *_ = env.step({"blue": p.noop(), "red": red})
        state = env.battle_state
        barrels = [s for s in state.spells if s.team == RED and names[s.card_id] == "GoblinBarrel"]
        if barrels and barrels[0].motion == SpellMotion.FLIGHT and barrels[0].delay_ticks == 0:
            flight_start = state.tick if flight_start is None else flight_start
        shown_now = float(obs["blue"]["spatial"][plane].sum())
        seen.append((state.tick, flight_start, shown_now, bool(barrels)))
        if thrown and flight_start is not None and not barrels:
            break
    assert thrown, "the barrel was never thrown"
    assert flight_start is not None, "the barrel never flew"
    shown = [(t, f) for t, f, v, live in seen if v > 0]
    assert shown, "the barrel's target never appeared"
    first_tick, start = shown[0]
    assert first_tick - start >= K - 1, (first_tick, start)  # never earlier than K of flight
    assert all(v == 0 for t, f, v, live in seen if f is None or t - f < K - 1)
