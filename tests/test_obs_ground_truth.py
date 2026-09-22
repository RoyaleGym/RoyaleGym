"""Grade the observation against the ENGINE STATE, not against the other seat.

WHY THIS FILE IS SEPARATE FROM tests/test_env_obs.py
    The gate this layer rests on is an AGREEMENT test: obs[blue] on a state equals
    obs[red] on its mirror, bit for bit. It is load-bearing and it is not enough,
    because every defect that is SYMMETRIC in the seats passes it -- two channels
    swapped on both seats, an hp scale wrong on both seats, an off-by-one tile index
    on both seats. The coverage guard beside it only asks whether a plane is
    non-zero somewhere, which also stays true under a swap. An adversarial review of
    this suite found exactly that gap, and found it in a rewrite that had just moved
    every channel index after ``own_buildings``.

    So these checks ask a different question, and none of them compares one seat to
    the other. ``channel_names()`` is a CLAIM about what each plane means; each check
    takes one of those claims and tests it against the engine's own ``state()``,
    using only the published frame rule (``protocol.to_own``) and the arena's integer
    geometry. Nothing here reads the builder's internals. They were written that way
    deliberately, by someone who had not written obs.py, so that they could not
    inherit its assumptions -- which is the property to preserve if they are ever
    rewritten, not the file's location.

WHAT THEY DO NOT COVER, so nobody over-reads them
    MockEngine only, so the spell and stun channels and the entity-list ``spells``
    rows are not touched here -- that engine resolves spells inside a tick and models
    no stun. Those live in tests/test_rust_engine.py and tests/test_rust_spells.py.
    The building count uses one card, so building footprints other than a Cannon's
    are untested. And every check here is ``SpatialObsBuilder``;
    ``EntityListObsBuilder`` has no ground-truth check of this kind yet.
"""

from __future__ import annotations

import numpy as np
import pytest

from royalegym.action import TileActionParser
from royalegym.mock_engine import MockEngine
from royalegym.obs import SpatialObsBuilder
from royalegym.protocol import (
    BIT_WATER,
    BLUE,
    RED,
    DeployCommand,
    EntityKind,
    MatchSetup,
    ShuffleMode,
    to_own,
)

# The channel's own documented meaning: "sum of own entity hp / 1000".
HP_PER_UNIT = 1000.0
DECK_NAMES = ("Knight", "Minions", "Cannon", "Giant", "Archer", "Goblins", "Skeletons", "Musketeer")


@pytest.fixture(scope="module")
def catalogue():
    probe = MockEngine()
    return {c.name: c for c in probe.cards()}, probe.arena()


def fresh(catalogue, seed=5, tower_hp=None):
    cards, _ = catalogue
    deck = [cards[n].card_id for n in DECK_NAMES]
    engine = MockEngine()
    engine.reset(
        seed,
        MatchSetup(
            decks=[deck, deck],
            shuffle=ShuffleMode.NONE,
            elixir_milli=[10000, 10000],
            tower_hp=tower_hp,
        ),
    )
    return engine


def own_tile(arena, team, x, y):
    """The tile a point falls in, in the team's own frame -- the protocol's rule, here."""
    ox, oy = to_own(arena, team, x, y)
    return (
        min(max(oy // arena.subtile, 0), arena.tiles_y - 1),
        min(max(ox // arena.subtile, 0), arena.tiles_x - 1),
    )


def observe(engine, team):
    """(observation, channel name -> index) for one seat, with nothing cached."""
    parser = TileActionParser()
    parser.bind(engine)
    builder = SpatialObsBuilder()
    builder.bind(engine, parser)
    state = engine.state()
    obs = builder.build(state, team, parser.action_mask(state, team))
    return obs, {name: i for i, name in enumerate(builder.channel_names())}


def place(engine, team, card_id, tx, ty, ticks=1):
    """Play a card at the centre of engine tile (tx, ty). Returns the deploy status."""
    st = engine.state()
    subtile = engine.arena().subtile
    slot = next((i for i, c in enumerate(st.players[team].hand) if c == card_id), None)
    assert slot is not None, f"card {card_id} is not in {team}'s hand"
    results = engine.step(
        [
            DeployCommand(
                team=team,
                hand_slot=slot,
                x=tx * subtile + subtile // 2,
                y=ty * subtile + subtile // 2,
            )
        ],
        ticks,
    )
    return results[0].status


def troops_of(engine, team):
    return [e for e in engine.state().entities if e.team == team and e.kind == EntityKind.TROOP]


# --- G1 / G2: one known unit, one known cell, one known value -------------------


def test_a_lone_troop_lands_in_exactly_the_tile_the_frame_rule_puts_it_in(catalogue):
    """Catches an off-by-one tile index on BOTH seats, which a flip gate cannot."""
    cards, arena = catalogue
    engine = fresh(catalogue)
    assert place(engine, BLUE, cards["Knight"].card_id, 6, 5) == 0
    troops = troops_of(engine, BLUE)
    assert len(troops) == 1, "the fixture must leave exactly one troop to find"
    obs, channel = observe(engine, BLUE)
    plane = obs["spatial"][channel["own_ground_troops"]]
    want = own_tile(arena, BLUE, troops[0].x, troops[0].y)
    hits = [tuple(p) for p in np.argwhere(plane > 0)]
    assert hits == [want], f"engine puts it at {want}; the plane is non-zero at {hits}"
    assert plane[want] == 1


def test_the_hp_channel_is_hit_points_over_a_thousand(catalogue):
    """Catches a wrong scale on both seats, which a flip gate cannot."""
    cards, arena = catalogue
    engine = fresh(catalogue)
    assert place(engine, BLUE, cards["Knight"].card_id, 6, 5) == 0
    troop = troops_of(engine, BLUE)[0]
    obs, channel = observe(engine, BLUE)
    at = own_tile(arena, BLUE, troop.x, troop.y)
    assert troop.hp > 0
    assert float(obs["spatial"][channel["own_hp"]][at]) == pytest.approx(
        troop.hp / HP_PER_UNIT, abs=1e-6
    )


# --- G3: each kind in its own channel, crown towers out of the buildings --------


def test_air_ground_buildings_and_crown_towers_are_four_different_channels(catalogue):
    """The channel list promises a crown tower is NOT counted in own_buildings."""
    cards, _ = catalogue
    # One Blue princess starts down, so the tower count is 2 and cannot collide with
    # the air count of 3. Without that, Minions and the three crown towers both read
    # 3 and a swap between those two channels would pass this test.
    engine = fresh(catalogue, tower_hp=[[2400, 0, 1400], [2400, 1400, 1400]])
    place(engine, BLUE, cards["Minions"].card_id, 6, 5)
    place(engine, BLUE, cards["Cannon"].card_id, 11, 5)
    obs, channel = observe(engine, BLUE)
    mine = [e for e in engine.state().entities if e.team == BLUE]
    want = {
        "own_air_troops": sum(1 for e in mine if e.kind == EntityKind.TROOP and e.flying),
        "own_ground_troops": sum(
            1 for e in mine if e.kind == EntityKind.TROOP and not e.flying
        ),
        "own_buildings": sum(1 for e in mine if e.kind == EntityKind.BUILDING),
        "own_towers": sum(
            1
            for e in mine
            if e.kind in (EntityKind.KING_TOWER, EntityKind.PRINCESS_TOWER)
        ),
    }
    # No two counts equal, or a swap between those two channels passes by coincidence.
    assert len(set(want.values())) == len(want), f"fixture counts collide: {want}"
    for kind in ("own_air_troops", "own_buildings", "own_towers"):
        assert want[kind], f"vacuous: the fixture has no {kind}"
    # own_ground_troops is deliberately 0 here: nothing may leak into it either.
    assert want["own_ground_troops"] == 0
    got = {name: float(obs["spatial"][channel[name]].sum()) for name in want}
    assert got == {k: float(v) for k, v in want.items()}


# --- G4: own vs enemy, with the sides deliberately unequal ---------------------


def test_own_and_enemy_channels_with_the_two_sides_unequal(catalogue):
    """Unequal on purpose: with the same count each side, a swap hides."""
    cards, _ = catalogue
    engine = fresh(catalogue)
    place(engine, BLUE, cards["Knight"].card_id, 6, 5)
    place(engine, RED, cards["Knight"].card_id, 6, 26)
    place(engine, RED, cards["Archer"].card_id, 11, 26)
    counts = {team: len(troops_of(engine, team)) for team in (BLUE, RED)}
    assert counts[BLUE] != counts[RED], f"fixture is symmetric: {counts}"
    for team in (BLUE, RED):
        obs, channel = observe(engine, team)
        assert float(obs["spatial"][channel["own_ground_troops"]].sum()) == counts[team]
        assert float(obs["spatial"][channel["enemy_ground_troops"]].sum()) == counts[1 - team]


# --- G5: the static channels are the arena's, not the battle's -----------------


def test_the_water_channel_is_the_arenas_water(catalogue):
    _, arena = catalogue
    obs, channel = observe(fresh(catalogue), BLUE)
    water = obs["spatial"][channel["water"]]
    half = arena.half
    bad = []
    for ty in range(arena.tiles_y):
        for tx in range(arena.tiles_x):
            wet = sum(
                1
                for dy in range(half)
                for dx in range(half)
                if arena.grid[ty * half + dy][tx * half + dx] & BIT_WATER
            )
            want = wet / (half * half)
            if abs(float(water[ty, tx]) - want) > 1e-6:
                bad.append((ty, tx, float(water[ty, tx]), want))
    assert bad == [], f"{len(bad)} of {arena.tiles_y * arena.tiles_x} tiles differ, first {bad[0]}"
    assert 0 < water.sum() < water.size, "vacuous: the arena is all water or none"


# --- G6: a timer channel that has to go back down ------------------------------


def test_own_deploying_follows_the_deploy_timer(catalogue):
    cards, _ = catalogue
    engine = fresh(catalogue)
    place(engine, BLUE, cards["Giant"].card_id, 6, 5, ticks=1)
    obs, channel = observe(engine, BLUE)
    during = float(obs["spatial"][channel["own_deploying"]].sum())
    engine_during = sum(1 for e in engine.state().entities if e.team == BLUE and e.deploy_ticks)
    engine.step([], 200)
    obs, channel = observe(engine, BLUE)
    after = float(obs["spatial"][channel["own_deploying"]].sum())
    engine_after = sum(1 for e in engine.state().entities if e.team == BLUE and e.deploy_ticks)
    assert during == engine_during
    assert after == engine_after
    assert during != after, "vacuous: the timer never ran out, so nothing was graded"
