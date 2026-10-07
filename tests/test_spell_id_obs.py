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


# --- effect_identity: spell cards apart from the effects units leave ------------------------

#: The planes ``effect_identity=True`` appends to ``spell_ids``.
EFFECT_PLANES = ("own_effect_at", "enemy_effect_at", "own_effect_aim", "enemy_effect_aim_seen")


def test_effect_identity_needs_spell_identity():
    with pytest.raises(ValueError, match="effect_identity needs spell_identity"):
        SpatialObsBuilder(card_identity=True, spell_aim_after_ticks=K, effect_identity=True)


def test_effect_identity_appends_four_planes_and_is_off_by_default():
    from royalegym.obs import EFFECT_ID_PLANES

    assert EFFECT_ID_PLANES == EFFECT_PLANES
    _, plain = _env(spell_identity=True)
    assert "effect_identity" not in plain.config()
    env, split = _env(spell_identity=True, effect_identity=True)
    a = env.engine.arena()
    space = split.observation_space()["spell_ids"]
    assert space.shape == (8, a.tiles_y, a.tiles_x)
    assert space.high.max() == split.observation_space()["card_ids"].high.max()
    assert split.config()["effect_identity"] is True


def _merged(split: np.ndarray) -> np.ndarray:
    """The split planes folded back into four, by the lower-id rule of a shared tile."""
    spells, effects = split[:4].astype(int), split[4:].astype(int)
    both = (spells != CARD_ID_EMPTY) & (effects != CARD_ID_EMPTY)
    return np.where(both, np.minimum(spells, effects), np.maximum(spells, effects))


def test_a_spell_cards_object_stays_and_a_troops_or_buildings_goes_to_the_effect_planes():
    """A Fireball (a SPELL card) stays in planes 0-3; a bomb under the Cannon (a BUILDING) and
    an area under the Minions (a TROOP), objects those cards' units left, go to planes 4-7 in
    the same order. Folded back together they are exactly the planes without the split."""
    env, split = _env(spell_identity=True, effect_identity=True)
    a = env.engine.arena()
    ids = {c.name: c.card_id for c in env.engine.cards()}
    fb_at, fb_aim = _point(a, 9, 20), _point(a, 9, 8)
    bomb, area = _point(a, 4, 12), _point(a, 14, 22)
    spells = [
        _spell(BLUE, ids["Fireball"], SpellMotion.FLIGHT, fb_at, fb_aim, flown=1),
        _spell(BLUE, ids["Cannon"], SpellMotion.FLIGHT, bomb, bomb, delay=20, flown=0),
        _spell(RED, ids["Minions"], SpellMotion.PULSING, area, area, delay=50),
    ]
    out = _ids(env, split, spells)
    fb, cn, mn = (CARD_ID_OFFSET + ids[n] for n in ("Fireball", "Cannon", "Minions"))
    assert out.shape[0] == 8
    assert out[0][_tile(a, BLUE, fb_at)] == fb
    assert out[2][_tile(a, BLUE, fb_aim)] == fb
    assert set(np.unique(out[:4]).tolist()) == {CARD_ID_EMPTY, fb}
    assert out[4][_tile(a, BLUE, bomb)] == cn
    assert out[6][_tile(a, BLUE, bomb)] == cn
    assert out[5][_tile(a, BLUE, area)] == mn
    assert out[7][_tile(a, BLUE, area)] == mn
    assert set(np.unique(out[4:]).tolist()) == {CARD_ID_EMPTY, cn, mn}
    plain_env, plain = _env(spell_identity=True)
    assert np.array_equal(_merged(out), _ids(plain_env, plain, spells))


@needs_engine
def test_on_the_engine_spell_cards_stay_and_the_cards_whose_units_leave_effects_move():
    """RoyaleSim's catalogue decides it (CardInfo.card_kind): the Fireball, the Heal and the
    Goblin Barrel are spell cards, and the four whose evolutions Train found in spell_ids
    (Firecracker, Cannon, Elite Barbarians, Princess) are troops and buildings."""
    from royalegym.rust_engine import RustEngine

    eng = RustEngine()
    ids = {c.name: c.card_id for c in eng.cards()}
    stay = [n for n in ("Fireball", "Heal", "GoblinBarrel") if n in ids]
    move = ["Firecracker", "Cannon", "AngryBarbarians", "Princess"]
    assert {"Fireball", "GoblinBarrel"} <= set(stay)
    assert set(move) <= set(ids)
    builder = SpatialObsBuilder(card_identity=True, spell_aim_after_ticks=K, spell_identity=True,
                                effect_identity=True)
    env = ClashParallelEnv(engine=eng, obs_builder=builder)
    env.reset(seed=0)
    a = eng.arena()
    names = [*stay, *move]
    spots = [_point(a, 2 + 2 * i, 12) for i in range(len(names))]
    spells = [_spell(BLUE, ids[n], SpellMotion.PULSING, p, p, delay=50)
              for n, p in zip(names, spots, strict=True)]
    out = _ids(env, builder, spells)
    for n, p in zip(names, spots, strict=True):
        planes = (4, 6) if n in move else (0, 2)
        for plane in range(8):
            want = CARD_ID_OFFSET + ids[n] if plane in planes else CARD_ID_EMPTY
            assert out[plane][_tile(a, BLUE, p)] == want, (n, plane)
