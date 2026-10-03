"""Heroes on the board: ``SpatialObsBuilder(heroes=True)``.

A hero is a card's special form (a hero Ice Golem plays the IceGolemite card), so ``card_ids``
names its card and nothing in the observation said it was the hero. A player sees it: a hero
looks different, and its one ability shows when it fires. So three planes, counted per tile like
the troop planes:
- ``own_hero`` and ``enemy_hero``: the side's hero units;
- ``enemy_hero_unspent``: the enemy's hero units whose one charge is not used yet.
The enemy's readiness is read from its button rows' ``spent`` column only. ``available`` and
``cost`` are not read: a player cannot see the enemy's buttons.

Off by default, and appended after every other optional fair plane, so every existing layout
keeps its plane order.

SKIPS
    The engine-backed tests skip without the engine. Not a pass.
"""

from __future__ import annotations

import msgspec
import numpy as np
import pytest

from royalegym.action import TileActionParser
from royalegym.mock_engine import MockEngine
from royalegym.obs import Reveal, SpatialObsBuilder, hero_channels, spatial_channels
from royalegym.protocol import (
    BLUE,
    HAND_SIZE,
    RED,
    STATUS_EVOLVED,
    STATUS_HERO,
    DeployCommand,
    DeployStatus,
    EntityKind,
    EntityState,
    MatchSetup,
    ShuffleMode,
    ability_row,
    to_engine,
    to_own,
)
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available

needs_engine = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))

HERO_PLANES = ["own_hero", "enemy_hero", "enemy_hero_unspent"]


# --- layout -------------------------------------------------------------------------------


def test_off_by_default_and_after_every_other_optional_fair_plane():
    assert not set(HERO_PLANES) & {n for n, _ in spatial_channels()}
    before = [n for n, _ in spatial_channels(Reveal(enemy_spell_aim=True), True, True)]
    after = [
        n for n, _ in spatial_channels(Reveal(enemy_spell_aim=True), True, True, heroes=True)
    ]
    assert after[: len(before) - 1] == before[:-1], "an existing plane moved"
    assert after[len(before) - 1 : len(before) + 2] == HERO_PLANES
    assert after[-1] == "enemy_spell_aim", "the revealed planes stay last"
    plain = [n for n, _ in spatial_channels()]
    assert [n for n, _ in spatial_channels(None, heroes=True)] == plain + HERO_PLANES


def test_config_records_the_flag():
    assert SpatialObsBuilder(heroes=True).config()["heroes"] is True
    assert "heroes" not in SpatialObsBuilder().config()


# --- the planes, on a forged board ---------------------------------------------------------


def _board():
    eng = MockEngine()
    eng.reset(1, MatchSetup(decks=[list(range(8)), list(range(8))]))
    return eng.state(), eng.arena()


def _unit(uid, team, card, status, point, kind=EntityKind.TROOP):
    return EntityState(
        uid, team, int(kind), card, -1, point[0], point[1], 100, 100, 500, False, 0,
        status_flags=status,
    )


def _with(state, entities, rows=None):
    """``state`` with only ``entities`` on the board (towers kept, their bits reported as 0)
    and, when given, each side's ability rows."""
    towers = [msgspec.structs.replace(e, status_flags=0) for e in state.entities
              if e.kind in (EntityKind.KING_TOWER, EntityKind.PRINCESS_TOWER)]
    players = state.players
    if rows is not None:
        players = [msgspec.structs.replace(p, abilities=r)
                   for p, r in zip(players, rows, strict=True)]
    return msgspec.structs.replace(state, entities=towers + entities, players=players)


def _tile(arena, viewer, point):
    ox, oy = to_own(arena, viewer, *point)
    return oy // arena.subtile, ox // arena.subtile


def _row(card, spent, available=1, cost=2, cooldown=0):
    return [available, spent, cost, card, cooldown]


def test_a_hero_lights_its_own_plane_for_its_side_and_the_enemy_plane_for_the_other():
    state, a = _board()
    at = (a.subtile * 9 + 100, a.subtile * 12 + 100)
    s = _with(state, [_unit(100, BLUE, 5, STATUS_HERO, at)], rows=[[_row(5, 0)], []])
    blue = hero_channels(s, BLUE, a)
    red = hero_channels(s, RED, a)
    assert blue.shape == (3, a.tiles_y, a.tiles_x)
    assert blue[0].sum() == 1
    assert blue[0][_tile(a, BLUE, at)] == 1
    assert blue[1].sum() == 0
    assert blue[2].sum() == 0
    assert red[0].sum() == 0
    assert red[1].sum() == 1
    assert red[1][_tile(a, RED, at)] == 1
    assert red[2].sum() == 1
    assert red[2][_tile(a, RED, at)] == 1


def test_a_spent_charge_clears_only_the_unspent_plane():
    state, a = _board()
    at = (a.subtile * 4 + 100, a.subtile * 20 + 100)
    s = _with(state, [_unit(100, RED, 5, STATUS_HERO, at)], rows=[[], [_row(5, 1)]])
    blue = hero_channels(s, BLUE, a)
    assert blue[1].sum() == 1, "the hero is still on the board"
    assert blue[2].sum() == 0, "its charge is used"


def test_only_spent_is_read_from_the_enemys_rows():
    """Fair: ``available`` (a cooldown, a hero mid-ability), ``cost`` and the cooldown left are
    not on a player's screen for the enemy's buttons, so changing them changes nothing."""
    state, a = _board()
    at = (a.subtile * 4 + 100, a.subtile * 20 + 100)
    hero = [_unit(100, RED, 5, STATUS_HERO, at)]
    plain = hero_channels(_with(state, hero, rows=[[], [_row(5, 0)]]), BLUE, a)
    for row in (_row(5, 0, available=0), _row(5, 0, cost=9), _row(5, 0, cooldown=40)):
        seen = hero_channels(_with(state, hero, rows=[[], [row]]), BLUE, a)
        assert np.array_equal(plain, seen), row


def test_the_own_rows_are_never_read():
    state, a = _board()
    at = (a.subtile * 9 + 100, a.subtile * 12 + 100)
    hero = [_unit(100, BLUE, 5, STATUS_HERO, at)]
    spent = hero_channels(_with(state, hero, rows=[[_row(5, 1)], []]), BLUE, a)
    unspent = hero_channels(_with(state, hero, rows=[[_row(5, 0)], []]), BLUE, a)
    assert np.array_equal(spent, unspent)


def test_evolved_units_plain_units_and_towers_are_not_heroes():
    state, a = _board()
    p = (a.subtile * 9 + 100, a.subtile * 12 + 100)
    s = _with(state, [
        _unit(100, BLUE, 5, STATUS_EVOLVED, p),
        _unit(101, RED, 6, 0, p),
        _unit(102, RED, 6, STATUS_EVOLVED, p),
    ], rows=[[], []])
    assert hero_channels(s, BLUE, a).sum() == 0
    assert hero_channels(s, RED, a).sum() == 0


def test_two_heroes_on_one_tile_count_two():
    state, a = _board()
    p = (a.subtile * 4 + 100, a.subtile * 20 + 100)
    s = _with(state, [_unit(100, RED, 5, STATUS_HERO, p), _unit(101, RED, 6, STATUS_HERO, p)],
              rows=[[], [_row(5, 0), _row(6, 1)]])
    blue = hero_channels(s, BLUE, a)
    assert blue[1][_tile(a, BLUE, p)] == 2
    assert blue[2][_tile(a, BLUE, p)] == 1, "one of the two has used its charge"


def test_an_engine_that_does_not_report_the_bits_is_refused():
    state, a = _board()  # MockEngine reports no status_flags
    with pytest.raises(ValueError, match="does not report which units are heroes"):
        hero_channels(state, BLUE, a)


def test_an_enemy_hero_no_button_names_is_refused():
    """Reading "no row" as "spent" or "unspent" would hand a network a guess that looks like
    an answer."""
    state, a = _board()
    p = (a.subtile * 4 + 100, a.subtile * 20 + 100)
    hero = [_unit(100, RED, 5, STATUS_HERO, p)]
    with pytest.raises(ValueError, match="no ability row names"):
        hero_channels(_with(state, hero, rows=[[], [_row(7, 0)]]), BLUE, a)
    with pytest.raises(ValueError, match="no ability row names"):
        hero_channels(_with(state, hero, rows=[[], [[1, 0, 2]]]), BLUE, a)


def test_the_flag_refuses_an_engine_that_does_not_report_heroes():
    from royalegym import ClashParallelEnv

    env = ClashParallelEnv(engine=MockEngine(), obs_builder=SpatialObsBuilder(heroes=True))
    with pytest.raises(ValueError, match="does not report which units are heroes"):
        env.reset(seed=0)


# --- on the engine ------------------------------------------------------------------------


@needs_engine
def test_a_hero_musketeer_lights_the_planes_and_its_press_clears_the_unspent_one():
    eng = RustEngine()
    ids = {c.name: c.card_id for c in eng.cards()}
    names = ("Musketeer", "Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Cannon")
    if any(n not in ids for n in names):
        pytest.skip(f"this engine's catalogue lacks {[n for n in names if n not in ids]}")
    deck = [ids[n] for n in names]
    eng.reset(1, MatchSetup(
        decks=[deck, deck], shuffle=ShuffleMode.NONE, forms=[[2] + [0] * 7, [0] * 8],
        elixir_milli=[10000, 10000], start_tick=eng.rules().deploy_lockout_ticks,
    ))
    parser = TileActionParser(ability_buttons=True)
    parser.bind(eng)
    builder = SpatialObsBuilder(heroes=True)
    builder.bind(eng, parser)
    names_at = builder.channel_names()
    own, foe, unspent = (names_at.index(n) for n in HERO_PLANES)
    state = eng.state()
    builder.reset(state)

    def planes(state):
        out = []
        for team in (BLUE, RED):
            sp = builder.build(state, team, parser.action_mask(state, team))["spatial"]
            out.append((sp[own].sum(), sp[foe].sum(), sp[unspent].sum()))
        return out

    assert planes(state) == [(0, 0, 0), (0, 0, 0)]
    t = eng.arena().subtile
    slot = state.players[BLUE].hand.index(ids["Musketeer"])
    x, y = to_engine(eng.arena(), BLUE, 9 * t + t // 2, 10 * t + t // 2)
    assert eng.step([DeployCommand(BLUE, slot, x, y)], 1)[0].status == DeployStatus.OK
    pressed = None
    for _ in range(200):
        state = eng.state()
        if parser.action_mask(state, BLUE)[parser.n_tile_actions]:
            pressed = state
            break
        eng.step([], 1)
    assert pressed is not None, f"the hero's button never came on: {state.players[BLUE].abilities}"
    # On the board, its charge unused: Blue sees its own hero, Red an enemy hero, unspent.
    assert planes(pressed) == [(1, 0, 0), (0, 1, 1)]
    cmd = parser.parse(parser.n_tile_actions, pressed, BLUE)
    assert cmd.hand_slot == HAND_SIZE
    assert eng.step([cmd], 1)[0].status == DeployStatus.OK
    state = eng.state()
    assert ability_row(state.players[BLUE].abilities[0]).spent == 1
    hero_alive = any(e.team == BLUE and e.card_id == ids["Musketeer"] for e in state.entities)
    assert hero_alive, "the hero died at once, so this test would show nothing"
    assert planes(state) == [(1, 0, 0), (0, 1, 0)], "the charge is used; the hero still stands"
