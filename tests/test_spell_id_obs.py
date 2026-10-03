"""Which spell is where: ``SpatialObsBuilder(spell_identity=True)``.

The spell planes in ``spatial`` count spells; they cannot say a Fireball from a Zap. So a key
of its own, ``spell_ids``, uint8 [4, 32, 18], holding catalogue ids exactly as ``card_ids``
does (``CARD_ID_OFFSET + id``, 0 for nothing), so one embedding serves both:
- plane 0, own spells at the tile of their current centre;
- plane 1, enemy spells at the tile of their current centre;
- plane 2, own spells at their aim tile (a player knows its own target);
- plane 3, enemy spells at their aim tile, once a player could read it (``SpellAimClock``,
  the same rule as ``enemy_spell_aim_seen``).
Two spells on one tile of one plane: the lower catalogue id is written, whatever the list
order. It needs ``card_identity=True`` (the vocabulary and its pinned names) and
``spell_aim_after_ticks`` (when an enemy aim is readable). Off by default.

SKIPS
    The engine-backed test skips without the engine. Not a pass.
"""

from __future__ import annotations

import msgspec
import numpy as np
import pytest

from royalegym import ClashParallelEnv, make_env
from royalegym.mock_engine import MockEngine
from royalegym.obs import CARD_ID_EMPTY, CARD_ID_OFFSET, SpatialObsBuilder
from royalegym.protocol import BLUE, RED, SpellMotion, SpellState, to_engine, to_own
from royalegym.rust_engine import CORE_IMPORT_ERROR, core_available

needs_engine = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
K = 10


def test_it_needs_card_identity_and_an_aim_rule():
    with pytest.raises(ValueError, match="spell_identity needs card_identity"):
        SpatialObsBuilder(spell_identity=True, spell_aim_after_ticks=K)
    with pytest.raises(ValueError, match="spell_identity needs spell_aim_after_ticks"):
        SpatialObsBuilder(spell_identity=True, card_identity=True)


def _env(**kw):
    builder = SpatialObsBuilder(card_identity=True, spell_aim_after_ticks=K, **kw)
    env = ClashParallelEnv(engine=MockEngine(), obs_builder=builder)
    env.reset(seed=0)
    return env, builder


def test_off_by_default_nothing_changes():
    _, builder = _env()
    assert "spell_ids" not in builder.observation_space().spaces
    assert "spell_identity" not in builder.config()


def test_the_space_shares_card_ids_vocabulary():
    env, builder = _env(spell_identity=True)
    space = builder.observation_space()
    a = env.engine.arena()
    assert space["spell_ids"].shape == (4, a.tiles_y, a.tiles_x)
    assert space["spell_ids"].dtype == np.uint8
    assert space["spell_ids"].high.max() == space["card_ids"].high.max()
    assert builder.config()["spell_identity"] is True


def _point(a, tile_x, tile_y, viewer=BLUE):
    half = a.subtile // 2
    return to_engine(a, viewer, a.subtile * tile_x + half, a.subtile * tile_y + half)


def _tile(a, viewer, point):
    ox, oy = to_own(a, viewer, *point)
    return oy // a.subtile, ox // a.subtile


def _spell(team, card, motion, at, aim, delay=0, flown=-1):
    return SpellState(team, card, int(motion), at[0], at[1], aim[0], aim[1], delay, 0, 0, 0, flown)


def _ids(env, builder, spells, viewer=BLUE, tick=500):
    state = msgspec.structs.replace(env.battle_state, tick=tick, spells=spells)
    mask = np.ones(env.action_parser.n_actions, dtype=np.int8)
    return builder.build(state, viewer, mask)["spell_ids"]


def test_an_own_thrown_spell_shows_where_it_is_and_where_it_lands():
    env, builder = _env(spell_identity=True)
    a = env.engine.arena()
    at, aim = _point(a, 9, 20), _point(a, 9, 8)
    ids = _ids(env, builder, [_spell(BLUE, 7, SpellMotion.FLIGHT, at, aim, flown=1)])
    assert ids[0][_tile(a, BLUE, at)] == CARD_ID_OFFSET + 7
    assert ids[2][_tile(a, BLUE, aim)] == CARD_ID_OFFSET + 7
    assert (ids[0] != CARD_ID_EMPTY).sum() == 1
    assert (ids[2] != CARD_ID_EMPTY).sum() == 1
    assert not ids[1].any()
    assert not ids[3].any()


def test_an_enemy_thrown_spell_shows_its_target_only_once_readable():
    env, builder = _env(spell_identity=True)
    a = env.engine.arena()
    at, aim = _point(a, 9, 8), _point(a, 9, 20)
    early = _ids(env, builder, [_spell(RED, 7, SpellMotion.FLIGHT, at, aim, flown=K - 1)])
    assert early[1][_tile(a, BLUE, at)] == CARD_ID_OFFSET + 7, "the spell itself is in sight"
    assert not early[3].any(), "its target is not readable yet"
    late = _ids(env, builder, [_spell(RED, 7, SpellMotion.FLIGHT, at, aim, flown=K)])
    assert late[3][_tile(a, BLUE, aim)] == CARD_ID_OFFSET + 7
    assert not late[0].any()
    assert not late[2].any()


def test_an_enemy_area_spell_shows_its_target_from_the_first_sight():
    env, builder = _env(spell_identity=True)
    a = env.engine.arena()
    p = _point(a, 5, 22)
    ids = _ids(env, builder, [_spell(RED, 11, SpellMotion.AREA, p, p)])
    assert ids[1][_tile(a, BLUE, p)] == CARD_ID_OFFSET + 11
    assert ids[3][_tile(a, BLUE, p)] == CARD_ID_OFFSET + 11


def test_red_sees_blues_spell_as_the_enemys_in_its_own_frame():
    env, builder = _env(spell_identity=True)
    a = env.engine.arena()
    p = _point(a, 3, 12)
    ids = _ids(env, builder, [_spell(BLUE, 4, SpellMotion.AREA, p, p)], viewer=RED)
    assert ids[1][_tile(a, RED, p)] == CARD_ID_OFFSET + 4
    assert ids[3][_tile(a, RED, p)] == CARD_ID_OFFSET + 4
    assert not ids[0].any()


def test_a_shared_tile_holds_the_lower_id_whatever_the_order():
    env, builder = _env(spell_identity=True)
    a = env.engine.arena()
    p = _point(a, 6, 6)
    spells = [_spell(RED, 12, SpellMotion.AREA, p, p), _spell(RED, 3, SpellMotion.AREA, p, p)]
    one = _ids(env, builder, spells)
    two = _ids(env, builder, spells[::-1])
    assert np.array_equal(one, two)
    assert one[1][_tile(a, BLUE, p)] == CARD_ID_OFFSET + 3


def test_an_id_outside_the_catalogue_is_refused():
    env, builder = _env(spell_identity=True)
    a = env.engine.arena()
    p = _point(a, 6, 6)
    with pytest.raises(ValueError, match="outside the"):
        _ids(env, builder, [_spell(RED, builder.num_cards, SpellMotion.AREA, p, p)])


@needs_engine
def test_a_real_fireball_is_named_on_both_sides():
    env = make_env(
        deck=["Fireball", "Knight", "Archer", "Giant", "Minions", "Zap", "Cannon", "Musketeer"],
        obs_builder=SpatialObsBuilder(
            card_identity=True, spell_aim_after_ticks=K, spell_identity=True
        ),
        decision_ms=50,  # one tick a step, so the K-tick boundary is seen
    )
    names = {c.name: c.card_id for c in env.engine.cards()}
    fireball = CARD_ID_OFFSET + names["Fireball"]
    p = env.action_parser
    for seed in range(40):  # a hand only cycles by playing, so start with the Fireball in it
        obs, _ = env.reset(seed=seed)
        if names["Fireball"] in env.battle_state.players[BLUE].hand:
            break
    slot = env.battle_state.players[BLUE].hand.index(names["Fireball"])
    for _ in range(400):  # until Blue can pay for it
        if obs["blue"]["action_mask"][p.encode(slot, 9, 26)]:
            break
        obs, *_ = env.step({"blue": p.noop(), "red": p.noop()})
    else:
        pytest.fail("Blue never could cast its Fireball")
    obs, *_ = env.step({"blue": p.encode(slot, 9, 26), "red": p.noop()})
    seen = {"blue_own": 0, "red_enemy": 0, "red_aim": 0}
    for _ in range(200):
        if not env.battle_state.spells:
            break
        seen["blue_own"] += int((obs["blue"]["spell_ids"][0] == fireball).any())
        seen["red_enemy"] += int((obs["red"]["spell_ids"][1] == fireball).any())
        seen["red_aim"] += int((obs["red"]["spell_ids"][3] == fireball).any())
        obs, *_ = env.step({"blue": p.noop(), "red": p.noop()})
    assert seen["blue_own"] > 0, seen
    assert seen["red_enemy"] == seen["blue_own"], seen
    assert 0 < seen["red_aim"] < seen["red_enemy"], f"the aim shows only after {K} ticks: {seen}"
