"""Observation builders: seat-flip exactness, documentation, space containment."""

from __future__ import annotations

import msgspec
import numpy as np
import pytest

from royalegym import obs as obs_mod
from royalegym.action import TileActionParser
from royalegym.mock_engine import MockEngine
from royalegym.obs import (
    SPATIAL_CHANNELS,
    EntityListObsBuilder,
    SpatialObsBuilder,
    vector_fields,
)
from royalegym.protocol import (
    BLUE,
    RED,
    DeployCommand,
    DeployStatus,
    EntityKind,
    MatchSetup,
    ShuffleMode,
    SpawnSpec,
    SpellMotion,
    SpellState,
    mirror_state,
    to_own,
)

DECK = [0, 3, 10, 14, 11, 13, 7, 9]
CHANNEL = {name: i for i, (name, _) in enumerate(SPATIAL_CHANNELS)}


def _states(n_steps: int = 400):
    """A spread of real battle states from a random two-sided rollout.

    Two capture rules, and the second one exists because the first was not enough:

    * every 40 steps, for a spread over the match clock; and
    * right after every step in which the engine ACCEPTED a deploy.

    With only the periodic rule this set contained no
    entity still in its deploy timer, so ``own_deploying`` / ``enemy_deploying``
    were never exercised and the seat-flip test certified nothing about them. The
    engine and the builder were both fine -- measured: every troop spawns with
    ``deploy_ticks`` = 20 at TICK_MS 50, and the builder writes channels 8/9. The
    sampler was blind: elixir-limited random play accepts ~20 deploys in 400
    steps, a deploy timer is visible for ~7 three-tick steps, and every troop
    deploy of seed 8 (steps 4, 94, 151, 191, 245, 249, 336, 339, 395) fell between
    the 40-step samples. The coverage guard below caught it; keep the guard.
    """
    eng = MockEngine()
    parser = TileActionParser()
    parser.bind(eng)
    s = eng.arena().subtile
    eng.reset(
        8,
        MatchSetup(
            decks=[DECK, DECK[::-1]],
            elixir_milli=[9000, 4000],
            tower_hp=[[2400, 900, 1400], [2400, 0, 1100]],
            # an asymmetric board, so a flip bug cannot hide behind symmetry
            spawns=[SpawnSpec(BLUE, 0, s * 3, s * 9), SpawnSpec(RED, 7, s * 12, s * 20)],
        ),
    )
    rng = np.random.default_rng(8)
    out = [eng.state()]
    for i in range(n_steps):
        st = eng.state()
        cmds = []
        for team in (BLUE, RED):
            legal = np.flatnonzero(parser.action_mask(st, team))[1:]
            if legal.size and rng.random() < 0.3:
                cmds.append(parser.parse(int(rng.choice(legal)), st, team))
        results = eng.step(cmds, 3)
        if i % 40 == 0 or any(r.status == DeployStatus.OK for r in results):
            out.append(eng.state())
        if eng.state().game_over:
            break
    return eng, parser, out


ENG, PARSER, MOCK_STATES = _states()

SPELL_STATUS_CHANNELS = [
    "own_spells",
    "enemy_spells",
    "own_spell_aim",
    "enemy_spell_aim",
    "own_stunned",
    "enemy_stunned",
]


def _with_spells_and_stuns(state, k: int):
    """``state`` with live spell objects and stunned units written in.

    WHY SYNTHETIC. MockEngine resolves every spell inside a tick and models no stun
    (mock_engine.py WHAT IT IS NOT), so no MockEngine state carries what channels
    15..20 and the ``spells`` array read. These rows have the Rust core's shape
    (py.rs state_json) and are deliberately asymmetric -- different cards, motions,
    tiles and timers per team -- so the seat-flip and containment tests below
    exercise them. tests/test_rust_engine.py checks the same channels on RustEngine
    battles, where they are real.
    """
    a = ENG.arena()
    t = a.subtile
    cid = {c.name: c.card_id for c in ENG.cards()}
    spells = [
        # Blue Fireball in flight from its king toward Red's right lane, 2 ticks of delay.
        SpellState(
            BLUE,
            cid["Fireball"],
            SpellMotion.FLIGHT,
            9 * t,
            6 * t + k * t,
            14 * t,
            26 * t,
            2,
            0,
            0,
            0,
        ),
        # Red Log rolling toward Blue's back line; its end point lies past the arena edge.
        SpellState(
            RED,
            cid["Log"],
            SpellMotion.ROLLING,
            4 * t + t // 2,
            12 * t,
            4 * t + t // 2,
            -3 * t,
            0,
            9 * t,
            11 * t + k,
            1,
        ),
        # Blue Zap sitting on a tile; Red Goblin Barrel still airborne.
        SpellState(BLUE, cid["Zap"], SpellMotion.AREA, 3 * t, 20 * t, 3 * t, 20 * t, 0, 0, 0, 0),
        SpellState(
            RED,
            cid["GoblinBarrel"],
            SpellMotion.AIRBORNE,
            12 * t,
            24 * t - k * t,
            6 * t,
            8 * t,
            0,
            0,
            0,
            0,
        ),
    ]
    stunned = {BLUE: 3 + k, RED: 9}
    ents = []
    for e in state.entities:
        if e.kind == EntityKind.TROOP and e.team in stunned:
            ents.append(msgspec.structs.replace(e, stun_ticks=stunned.pop(e.team)))
        else:
            ents.append(e)
    assert not stunned, "fixture needs a troop on each team"
    return msgspec.structs.replace(state, spells=spells[: 2 + k], entities=ents)


def _spell_states(states):
    both = [
        s
        for s in states
        if {e.team for e in s.entities if e.kind == EntityKind.TROOP} == {BLUE, RED}
    ]
    assert len(both) >= 3, "vacuous: too few Mock states with troops on both teams"
    return [_with_spells_and_stuns(both[i * (len(both) - 1) // 2], i) for i in range(3)]


STATES = MOCK_STATES + _spell_states(MOCK_STATES)


def flip_mismatches(builder_cls, states=STATES) -> list[str]:
    b = builder_cls()
    b.bind(ENG, PARSER)
    bad = []
    arena = ENG.arena()
    for i, s in enumerate(states):
        m = mirror_state(arena, s)
        mask_b = PARSER.action_mask(s, BLUE)
        mask_r = PARSER.action_mask(m, RED)
        if not np.array_equal(mask_b, mask_r):
            bad.append(f"state {i}: action mask")
        ob = b.build(s, BLUE, mask_b)
        orr = b.build(m, RED, mask_r)
        for k in ob:
            if not np.array_equal(ob[k], orr[k]):
                bad.append(f"state {i}: {k}")
    return bad


@pytest.mark.parametrize("builder_cls", [SpatialObsBuilder, EntityListObsBuilder])
def test_red_on_the_mirrored_state_sees_exactly_what_blue_sees(builder_cls):
    assert len(STATES) >= 5
    assert flip_mismatches(builder_cls) == []


@pytest.mark.parametrize("builder_cls", [SpatialObsBuilder, EntityListObsBuilder])
def test_plant_y_only_flip_is_caught(builder_cls, monkeypatch):
    def y_only(arena, team, x, y):
        return (x, y) if team == BLUE else (x, arena.height - y)

    monkeypatch.setattr(obs_mod, "to_own", y_only)
    assert flip_mismatches(builder_cls) != []


def test_the_two_seats_do_not_see_the_same_thing_on_an_asymmetric_board():
    """Guards against a flip test that passes because obs ignore the board."""
    b = SpatialObsBuilder()
    b.bind(ENG, PARSER)
    s = STATES[0]
    ob = b.build(s, BLUE, PARSER.action_mask(s, BLUE))
    orr = b.build(s, RED, PARSER.action_mask(s, RED))
    assert not np.array_equal(ob["spatial"], orr["spatial"])
    assert not np.array_equal(ob["vector"], orr["vector"])


def unseen_channels(states) -> list[str]:
    """Spatial channels that are all-zero, from Blue's seat, in every state given."""
    b = SpatialObsBuilder()
    b.bind(ENG, PARSER)
    seen = np.zeros(len(SPATIAL_CHANNELS), dtype=bool)
    for s in states:
        o = b.build(s, BLUE, PARSER.action_mask(s, BLUE))
        seen |= o["spatial"].reshape(len(SPATIAL_CHANNELS), -1).any(axis=1)
    return [SPATIAL_CHANNELS[i][0] for i in np.flatnonzero(~seen)]


def test_every_spatial_channel_is_documented_and_vector_layout_adds_up():
    b = SpatialObsBuilder()
    b.bind(ENG, PARSER)
    space = b.observation_space()
    assert space["spatial"].shape == (len(SPATIAL_CHANNELS), 32, 18)
    assert space["vector"].shape == (sum(n for _, n in vector_fields(len(ENG.cards()))),)
    # Vacuity guard: every channel carries information in at least one sampled
    # state, so the flip and containment tests above exercise all of them.
    assert unseen_channels(STATES) == []


def test_plant_sample_set_without_deploying_entities_trips_the_coverage_guard():
    """Regression plant: put back the shipped defect (no state has a deploy timer
    running) and require the guard to name exactly the two channels it hid."""
    stripped = [
        msgspec.structs.replace(
            s, entities=[msgspec.structs.replace(e, deploy_ticks=0) for e in s.entities]
        )
        for s in STATES
    ]
    # The plant must have changed something, or it grades the unmodified set.
    assert stripped != STATES
    assert unseen_channels(stripped) == ["own_deploying", "enemy_deploying"]


def test_plant_sample_set_without_spells_or_stuns_trips_the_coverage_guard():
    """The MockEngine states alone carry no spell object and no stun: the guard must
    name exactly the six channels the synthetic spell states exist to exercise."""
    assert MOCK_STATES != STATES, "plant did not land: the spell states are missing"
    assert all(not s.spells for s in MOCK_STATES)
    assert unseen_channels(MOCK_STATES) == SPELL_STATUS_CHANNELS


def spell_channel_errors(builder: SpatialObsBuilder) -> list[str]:
    """Channels 15..20, cell by cell, against the rows, for both seats and every
    synthetic spell state."""
    builder.bind(ENG, PARSER)
    a = ENG.arena()
    errors = []
    for i, s in enumerate(STATES[len(MOCK_STATES) :]):
        for seat in (BLUE, RED):
            sp = builder.build(s, seat, PARSER.action_mask(s, seat))["spatial"]
            want = {k: np.zeros((32, 18)) for k in SPELL_STATUS_CHANNELS}

            def cell(x, y, seat=seat):
                ox, oy = to_own(a, seat, x, y)
                return min(max(oy // a.subtile, 0), 31), min(max(ox // a.subtile, 0), 17)

            for q in s.spells:
                side = "own" if q.team == seat else "enemy"
                want[f"{side}_spells"][cell(q.x, q.y)] += 1
                want[f"{side}_spell_aim"][cell(q.aim_x, q.aim_y)] += 1
            for e in s.entities:
                if e.stun_ticks:
                    want["own_stunned" if e.team == seat else "enemy_stunned"][cell(e.x, e.y)] += 1
            errors += [
                f"state {i} seat {seat}: {k}"
                for k, g in want.items()
                if not np.array_equal(sp[CHANNEL[k]], g)
            ]
    return errors


def test_spell_and_stun_channels_mark_the_right_team_at_the_right_own_frame_tile():
    assert spell_channel_errors(SpatialObsBuilder()) == []


def test_plant_spell_aim_read_in_the_engine_frame_is_caught(monkeypatch):
    """Plant: the aim point rasterised without the seat rotation. Aimed at both the
    cell-by-cell check and the seat-flip test."""

    def engine_frame_aim(state, team, arena):
        moved = [
            msgspec.structs.replace(q, aim_x=arena.width - q.aim_x, aim_y=arena.height - q.aim_y)
            if team == RED
            else q
            for q in state.spells
        ]
        return orig(msgspec.structs.replace(state, spells=moved), team, arena)

    orig = obs_mod.spell_channels
    assert spell_channel_errors(SpatialObsBuilder()) == [], "baseline must be green"
    monkeypatch.setattr(obs_mod, "spell_channels", engine_frame_aim)
    assert obs_mod.spell_channels is engine_frame_aim, "plant did not land"
    errors = spell_channel_errors(SpatialObsBuilder())
    assert errors, "PLANT DID NOT LAND: an engine-frame aim point passed the cell check"
    assert all("spell_aim" in m for m in errors), errors
    assert flip_mismatches(SpatialObsBuilder), "PLANT DID NOT LAND on the seat-flip test"


# --- the deploying channels, cell by cell ------------------------------------------


def _both_teams_deploying():
    """Blue and Red each drop a Knight on the same tick, at non-mirrored tiles."""
    eng = MockEngine()
    a = eng.arena()
    s = a.subtile
    knight = next(c.card_id for c in eng.cards() if c.name == "Knight")
    deck = [knight] + [c for c in DECK if c != knight]
    eng.reset(
        0,
        MatchSetup(decks=[deck, deck], shuffle=ShuffleMode.NONE, elixir_milli=[10000, 10000]),
    )
    blue_xy = (s * 4 + s // 2, s * 10 + s // 2)  # engine tile (4, 10)
    red_xy = (s * 11 + s // 2, s * 25 + s // 2)  # engine tile (11, 25) -- not blue's mirror
    res = eng.step([DeployCommand(BLUE, 0, *blue_xy), DeployCommand(RED, 0, *red_xy)], 1)
    assert [r.status for r in res] == [DeployStatus.OK, DeployStatus.OK]
    return eng, eng.state()


def deploy_channel_errors(builder: SpatialObsBuilder, eng, state) -> list[str]:
    """Where own_deploying / enemy_deploying disagree with the entity list, per seat."""
    parser = TileActionParser()
    parser.bind(eng)
    builder.bind(eng, parser)
    a = eng.arena()
    deploying = [e for e in state.entities if e.deploy_ticks > 0]
    errors = []
    if {e.team for e in deploying} != {BLUE, RED}:
        return ["state does not have a deploying entity on both teams"]
    for seat in (BLUE, RED):
        sp = builder.build(state, seat, parser.action_mask(state, seat))["spatial"]
        want = {"own_deploying": np.zeros((32, 18)), "enemy_deploying": np.zeros((32, 18))}
        for e in deploying:
            assert e.kind == EntityKind.TROOP
            ox, oy = to_own(a, seat, e.x, e.y)
            key = "own_deploying" if e.team == seat else "enemy_deploying"
            want[key][oy // a.subtile, ox // a.subtile] += 1
        for key, grid in want.items():
            if not np.array_equal(sp[CHANNEL[key]], grid):
                errors.append(f"seat {seat}: {key}")
    return errors


def test_deploying_channels_mark_the_right_team_at_the_right_own_frame_tile():
    eng, state = _both_teams_deploying()
    assert deploy_channel_errors(SpatialObsBuilder(), eng, state) == []


def test_plant_swapped_deploying_channels_are_caught():
    class Swapped(SpatialObsBuilder):
        def build(self, state, team, action_mask):
            o = super().build(state, team, action_mask)
            sp = o["spatial"]
            own, foe = CHANNEL["own_deploying"], CHANNEL["enemy_deploying"]
            sp[[own, foe]] = sp[[foe, own]]
            return o

    eng, state = _both_teams_deploying()
    assert deploy_channel_errors(Swapped(), eng, state) == [
        "seat 0: own_deploying",
        "seat 0: enemy_deploying",
        "seat 1: own_deploying",
        "seat 1: enemy_deploying",
    ]


@pytest.mark.parametrize("builder_cls", [SpatialObsBuilder, EntityListObsBuilder])
def test_observations_are_inside_their_space(builder_cls):
    b = builder_cls()
    b.bind(ENG, PARSER)
    space = b.observation_space()
    for s in STATES:
        for team in (BLUE, RED):
            o = b.build(s, team, PARSER.action_mask(s, team))
            assert o in space


def test_own_tower_hp_is_own_frame_for_red():
    b = SpatialObsBuilder()
    b.bind(ENG, PARSER)
    s = STATES[0]
    v_red = b.build(s, RED, PARSER.action_mask(s, RED))["vector"]
    n = len(ENG.cards()) + 1
    own_towers = 2 + 4 * n + 4 + 4 + n
    # Red started with its own-LEFT princess destroyed.
    assert v_red[own_towers + 1] == 0.0
    assert v_red[own_towers + 2] == np.float32(1100 / 1400)


# --- entity list order: engine-private, so no observation may depend on it -----------
#
# Summing float32 hp per tile in ``state.entities`` order costs bit-equality between
# the seats: the Rust engine lists entities by (reused) storage slot, so an exactly
# rotation-mirrored battle gives obs[blue] != obs[red] by 1 ulp in channels own_hp /
# enemy_hp -- measured on a Rust rollout at tick 4069. A sort key that omits
# deploy_ticks (or max_hp, radius, flying) leaves rows that tie on the key in engine
# order. These tests build mirrored states whose two seats list their entities in
# DIFFERENT orders, and shuffle the list, and require bit-identical obs.

# Blue's three same-tile units from that state (x, y, hp) and the order Rust listed
# Red's rotated twins in.
GAME30_BLUE = [(225000, 163800, 91), (232200, 178200, 5), (217800, 178200, 91)]
GAME30_RED_ORDER = [0, 2, 1]
ORDER_TRIALS = 40


def _mirrored_order_state():
    """A state equal to its own rotation mirror AS A SET, with Red's twins listed in a
    different order from Blue's: the game-30 triple, plus two stacked twins differing
    only in deploy_ticks (the EntityListObsBuilder tie)."""
    eng = MockEngine()
    parser = TileActionParser()
    parser.bind(eng)
    eng.reset(1, MatchSetup(decks=[DECK, DECK], shuffle=ShuffleMode.NONE))
    base = eng.state()
    a = eng.arena()
    gob = next(c for c in eng.cards() if c.name == "Goblins")

    def ent(uid, team, x, y, hp, deploy=0):
        return msgspec.structs.replace(
            base.entities[0],
            uid=uid,
            team=team,
            kind=int(EntityKind.TROOP),
            card_id=gob.card_id,
            tower_slot=-1,
            x=x,
            y=y,
            hp=hp,
            max_hp=gob.hitpoints,
            radius=gob.radius,
            flying=False,
            deploy_ticks=deploy,
        )

    blue = [ent(100 + 2 * i, BLUE, x, y, hp) for i, (x, y, hp) in enumerate(GAME30_BLUE)]
    red = [
        ent(101 + 2 * i, RED, a.width - GAME30_BLUE[i][0], a.height - GAME30_BLUE[i][1], hp)
        for i, (_, _, hp) in ((i, GAME30_BLUE[i]) for i in GAME30_RED_ORDER)
    ]
    sx, sy = 5 * a.subtile, 11 * a.subtile
    twins_b = [ent(200, BLUE, sx, sy, 50, 0), ent(202, BLUE, sx, sy, 50, 7)]
    twins_r = [
        ent(203, RED, a.width - sx, a.height - sy, 50, 7),
        ent(201, RED, a.width - sx, a.height - sy, 50, 0),
    ]
    # Live spells, rotation-mirrored, Red's listed in a different order, with a pair
    # that ties on (enemy, y, x, motion, card) and differs only in aim and hits.
    fb = next(c.card_id for c in eng.cards() if c.name == "Fireball")
    spells_b = [
        SpellState(BLUE, fb, SpellMotion.FLIGHT, sx, sy, 9 * a.subtile, 25 * a.subtile, 1, 0, 0, 0),
        SpellState(BLUE, fb, SpellMotion.FLIGHT, sx, sy, 4 * a.subtile, 25 * a.subtile, 1, 0, 0, 0),
        SpellState(BLUE, fb, SpellMotion.FLIGHT, sx, sy, 4 * a.subtile, 25 * a.subtile, 1, 0, 0, 3),
    ]
    spells_r = [
        msgspec.structs.replace(
            q,
            team=RED,
            x=a.width - q.x,
            y=a.height - q.y,
            aim_x=a.width - q.aim_x,
            aim_y=a.height - q.aim_y,
        )
        for q in reversed(spells_b)
    ]
    state = msgspec.structs.replace(
        base,
        entities=[*base.entities, *blue, *red, *twins_b, *twins_r],
        spells=[*spells_b, *spells_r],
    )
    return eng, parser, state


def _as_set(state, arena, rotate):
    def row(e):
        x, y, team = (
            (arena.width - e.x, arena.height - e.y, 1 - e.team)
            if rotate
            else (
                e.x,
                e.y,
                e.team,
            )
        )
        return (team, e.kind, e.card_id, e.tower_slot, x, y, e.hp, e.max_hp, e.deploy_ticks)

    def spell(q):
        if rotate:
            q = msgspec.structs.replace(
                q,
                team=1 - q.team,
                x=arena.width - q.x,
                y=arena.height - q.y,
                aim_x=arena.width - q.aim_x,
                aim_y=arena.height - q.aim_y,
            )
        return tuple(msgspec.structs.astuple(q))

    return sorted(row(e) for e in state.entities), sorted(spell(q) for q in state.spells)


def entity_order_mismatches(builder_cls) -> list[str]:
    """Seat and permutation differences of ``builder_cls`` on the mirrored state."""
    eng, parser, state = _mirrored_order_state()
    a = eng.arena()
    assert _as_set(state, a, False) == _as_set(state, a, True), "fixture is not a mirror"
    b = builder_cls()
    b.bind(eng, parser)
    mask_b, mask_r = parser.action_mask(state, BLUE), parser.action_mask(state, RED)
    reference = b.build(state, BLUE, mask_b)
    rng = np.random.default_rng(30)
    bad = []
    for trial in range(ORDER_TRIALS):
        ents = list(state.entities)
        spells = list(state.spells)
        if trial:
            rng.shuffle(ents)
            rng.shuffle(spells)
        s = msgspec.structs.replace(state, entities=ents, spells=spells)
        for seat, mask in ((BLUE, mask_b), (RED, mask_r)):
            o = b.build(s, seat, mask)
            bad += [
                f"trial {trial} seat {seat}: {k}"
                for k in o
                if not np.array_equal(o[k], reference[k])
            ]
    return bad


@pytest.mark.parametrize("builder_cls", [SpatialObsBuilder, EntityListObsBuilder])
def test_observation_does_not_depend_on_entity_list_order(builder_cls):
    assert entity_order_mismatches(builder_cls) == []


def _float_order_channels(entities, team, arena):
    """The shipped defect: hp accumulated as float32 in list order."""
    sp = np.zeros((obs_mod.ENTITY_CHANNELS, arena.tiles_y, arena.tiles_x), dtype=np.float32)
    for e in entities:
        ox, oy = to_own(arena, team, e.x, e.y)
        tx = min(max(ox // arena.subtile, 0), arena.tiles_x - 1)
        ty = min(max(oy // arena.subtile, 0), arena.tiles_y - 1)
        base = 0 if e.team == team else 4
        sp[base + (1 if e.flying else 0) if e.kind == EntityKind.TROOP else base + 2, ty, tx] += 1
        sp[base + 3, ty, tx] += e.hp / obs_mod.HP_SCALE
        if e.deploy_ticks > 0:
            sp[8 if e.team == team else 9, ty, tx] += 1
    return sp


def test_plant_float_order_hp_accumulation_is_caught(monkeypatch):
    """Regression plant: put the list-order float32 sum back; the gate must see the
    seats (or two orderings) differ in the spatial tensor."""
    assert entity_order_mismatches(SpatialObsBuilder) == [], "baseline must be green"
    monkeypatch.setattr(obs_mod, "entity_channels", _float_order_channels)
    assert obs_mod.entity_channels is _float_order_channels, "plant did not land"
    bad = entity_order_mismatches(SpatialObsBuilder)
    assert bad, "PLANT DID NOT LAND: float-order hp sums went unseen"
    assert all(m.endswith(": spatial") for m in bad)


def test_plant_six_field_entity_sort_key_is_caught(monkeypatch):
    """Regression plant: a six-field sort key (enemy, y, x, kind, card, hp)."""
    assert entity_order_mismatches(EntityListObsBuilder) == [], "baseline must be green"

    def six_fields(row):
        return row[:6]

    monkeypatch.setattr(obs_mod, "entity_row_key", six_fields)
    assert obs_mod.entity_row_key is six_fields, "plant did not land"
    bad = entity_order_mismatches(EntityListObsBuilder)
    assert bad, "PLANT DID NOT LAND: deploy_ticks tie kept engine order unseen"
    assert all(m.endswith(": entities") for m in bad)


def test_plant_spell_sort_key_without_aim_and_hits_is_caught(monkeypatch):
    """Plant: spell rows sorted by (enemy, y, x, motion, card) only -- the tied pair in
    the fixture then keeps engine list order, which differs between seats."""
    assert entity_order_mismatches(EntityListObsBuilder) == [], "baseline must be green"

    def five_fields(row):
        return row[:5]

    monkeypatch.setattr(obs_mod, "spell_row_key", five_fields)
    assert obs_mod.spell_row_key is five_fields, "plant did not land"
    bad = entity_order_mismatches(EntityListObsBuilder)
    assert bad, "PLANT DID NOT LAND: a spell tie kept engine order unseen"
    assert all(m.endswith(": spells") for m in bad), bad
