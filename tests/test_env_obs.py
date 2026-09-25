"""Observation builders: seat-flip exactness, documentation, space containment."""

from __future__ import annotations

import msgspec
import numpy as np
import pytest

from _decks import MIXED_DECK, card_id, card_ids
from royalegym import obs as obs_mod
from royalegym.action import TileActionParser
from royalegym.done_condition import StepLimitCondition
from royalegym.env import AGENT_TEAM, AGENTS, ClashParallelEnv
from royalegym.mock_engine import MockEngine
from royalegym.obs import (
    DECK_SIZE,
    EntityListObsBuilder,
    Reveal,
    SpatialObsBuilder,
    spatial_channels,
    vector_fields,
    vector_layout,
    vector_offsets,
)
from royalegym.protocol import (
    BLUE,
    HAND_SIZE,
    RED,
    DeployCommand,
    DeployStatus,
    EntityKind,
    MatchSetup,
    ShuffleMode,
    SpawnSpec,
    SpellMotion,
    SpellState,
    TowerSlot,
    mirror_state,
    to_own,
)
from royalegym.state_mutator import DefaultStateMutator, SnapshotStateMutator

_CARDS = MockEngine().cards()
DECK = card_ids(MIXED_DECK, _CARDS)
# The FAIR channel set. Every reveal appends its channels after these, so a fair
# channel's index is the same whatever the Reveal (obs.py, vector_layout).
CHANNEL = {name: i for i, (name, _) in enumerate(spatial_channels())}
# Everything an enabled Reveal opens, used to build "the most revealing builder"
# wherever a test wants to exercise a revealed channel or slot.
ALL_REVEALED = Reveal(
    enemy_elixir=True,
    enemy_hand=True,
    enemy_next_card=True,
    enemy_deck=True,
    enemy_spell_aim=True,
)


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
            spawns=[
                SpawnSpec(BLUE, card_id("Knight", _CARDS), s * 3, s * 9),
                SpawnSpec(RED, card_id("Minions", _CARDS), s * 12, s * 20),  # flying
            ],
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

# The spell/status channels a FAIR builder writes. ``enemy_spell_aim`` is not one
# of them: where the opponent's spell will land is a Reveal (obs.py).
SPELL_STATUS_CHANNELS = [
    "own_spells",
    "enemy_spells",
    "own_spell_aim",
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


def unseen_channels(states, reveal: Reveal | None = None) -> list[str]:
    """Spatial channels that are all-zero, from Blue's seat, in every state given."""
    b = SpatialObsBuilder(reveal=reveal)
    b.bind(ENG, PARSER)
    names = b.channel_names()
    seen = np.zeros(len(names), dtype=bool)
    for s in states:
        o = b.build(s, BLUE, PARSER.action_mask(s, BLUE))
        seen |= o["spatial"].reshape(len(names), -1).any(axis=1)
    return [names[i] for i in np.flatnonzero(~seen)]


@pytest.mark.parametrize("reveal", [None, ALL_REVEALED])
def test_every_spatial_channel_is_documented_and_vector_layout_adds_up(reveal):
    b = SpatialObsBuilder(reveal=reveal)
    b.bind(ENG, PARSER)
    space = b.observation_space()
    channels = spatial_channels(reveal)
    assert b.channel_names() == [name for name, _ in channels]
    assert space["spatial"].shape == (len(channels), 32, 18)
    assert space["vector"].shape == (
        sum(n for _, n in vector_fields(len(ENG.cards()), reveal)),
    )
    # Vacuity guard: every channel carries information in at least one sampled
    # state, so the flip and containment tests above exercise all of them.
    assert unseen_channels(STATES, reveal) == []


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
    """The spell and status channels, cell by cell, against the rows, for both seats
    and every synthetic spell state.

    Only the channels THIS builder writes: a fair builder has no
    ``enemy_spell_aim`` plane at all, and the reveal-on builder does, so the same
    check grades both without either being given a pass.
    """
    builder.bind(ENG, PARSER)
    a = ENG.arena()
    index = {name: i for i, name in enumerate(builder.channel_names())}
    checked = [k for k in (*SPELL_STATUS_CHANNELS, "enemy_spell_aim") if k in index]
    errors = []
    for i, s in enumerate(STATES[len(MOCK_STATES) :]):
        for seat in (BLUE, RED):
            sp = builder.build(s, seat, PARSER.action_mask(s, seat))["spatial"]
            want = {k: np.zeros((32, 18)) for k in checked}

            def cell(x, y, seat=seat):
                ox, oy = to_own(a, seat, x, y)
                return min(max(oy // a.subtile, 0), 31), min(max(ox // a.subtile, 0), 17)

            for q in s.spells:
                side = "own" if q.team == seat else "enemy"
                want[f"{side}_spells"][cell(q.x, q.y)] += 1
                if f"{side}_spell_aim" in want:
                    want[f"{side}_spell_aim"][cell(q.aim_x, q.aim_y)] += 1
            for e in s.entities:
                if e.stun_ticks:
                    want["own_stunned" if e.team == seat else "enemy_stunned"][cell(e.x, e.y)] += 1
            errors += [
                f"state {i} seat {seat}: {k}"
                for k, g in want.items()
                if not np.array_equal(sp[index[k]], g)
            ]
    return errors


@pytest.mark.parametrize("reveal", [None, ALL_REVEALED])
def test_spell_and_stun_channels_mark_the_right_team_at_the_right_own_frame_tile(reveal):
    assert spell_channel_errors(SpatialObsBuilder(reveal=reveal)) == []


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
    # The plant moves an ENEMY aim point, so it has to be graded by a builder that
    # writes one: the fair builder does not, and would pass the plant for the right
    # reason. tests below hold that channel's absence separately.
    def builder():
        return SpatialObsBuilder(reveal=ALL_REVEALED)

    assert spell_channel_errors(builder()) == [], "baseline must be green"
    monkeypatch.setattr(obs_mod, "spell_channels", engine_frame_aim)
    assert obs_mod.spell_channels is engine_frame_aim, "plant did not land"
    errors = spell_channel_errors(builder())
    assert errors, "PLANT DID NOT LAND: an engine-frame aim point passed the cell check"
    assert all("spell_aim" in m for m in errors), errors
    assert flip_mismatches(builder), "PLANT DID NOT LAND on the seat-flip test"


# --- the deploying channels, cell by cell ------------------------------------------


def _both_teams_deploying():
    """Blue and Red each drop a Knight on the same tick, at non-mirrored tiles."""
    eng = MockEngine()
    a = eng.arena()
    s = a.subtile
    knight = card_id("Knight", eng)
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
    own = v_red[vector_offsets(len(ENG.cards()))["own_tower_hp"]]
    # Red started with its own-LEFT princess destroyed.
    assert own[1] == 0.0
    assert own[2] == np.float32(1100 / 1400)


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
    fb = card_id("Fireball", eng)
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
        base = 0 if e.team == team else obs_mod.TEAM_STRIDE
        if e.kind == EntityKind.TROOP:
            sp[base + (1 if e.flying else 0), ty, tx] += 1
        elif e.kind == EntityKind.BUILDING:
            sp[base + 2, ty, tx] += 1
        else:
            sp[base + 3, ty, tx] += 1
        sp[base + 4, ty, tx] += e.hp / obs_mod.HP_SCALE
        if e.deploy_ticks > 0:
            sp[10 if e.team == team else 11, ty, tx] += 1
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


# --- the layout: widths, and what a Reveal does to them -----------------------


def test_the_fair_vector_is_12n_plus_37_wide_and_the_layout_says_so():
    """The width, asserted rather than derived.

    The spec this rewrite was built to called it 12n + 36. It is 12n + 37, and the
    extra slot is real: the fair block is the old 5n + 30 with the reveal-gated
    enemy-elixir slot kept (it now holds the COUNT) plus 7n + 7 of new features --
    deck n, cycle 6-8 3(n+1), last card n+1, cards seen n, possible hand n, and
    three scalars (ticks since own play, elixir leaked, enemy plays). The test is
    the arithmetic; the docstring is only the reason.
    """
    n = len(ENG.cards())
    fields = vector_layout(n)
    assert all(f.fair for f in fields)
    assert sum(f.size for f in fields) == 12 * n + 37
    b = SpatialObsBuilder()
    b.bind(ENG, PARSER)
    assert b.vec_size == 12 * n + 37
    assert b.observation_space()["vector"].shape == (12 * n + 37,)


def test_the_layout_is_self_describing_and_gapless():
    """Every slot belongs to exactly one named field, for both builders."""
    n = len(ENG.cards())
    for reveal in (None, ALL_REVEALED):
        fields = vector_layout(n, reveal)
        offsets = vector_offsets(n, reveal)
        assert [f.key for f in fields] == list(offsets)
        assert len({f.key for f in fields}) == len(fields), "duplicate field key"
        at = 0
        for f in fields:
            assert offsets[f.key] == slice(at, at + f.size), f.key
            at += f.size
        assert at == sum(s for _, s in vector_fields(n, reveal))


@pytest.mark.parametrize(
    "field",
    ["enemy_elixir", "enemy_hand", "enemy_next_card", "enemy_deck", "enemy_spell_aim"],
)
def test_a_reveal_adds_slots_and_never_moves_a_fair_one(field):
    """Rule (a): an enabled field ADDS; it is never present-but-zero.

    ``enemy_elixir`` is the stated exception -- it swaps the SOURCE of a slot that
    already holds the counted value -- so it is the one field whose width does not
    grow, and the test says so explicitly rather than skipping it.
    """
    n = len(ENG.cards())
    fair = vector_offsets(n)
    one = Reveal(**{field: True})
    revealed = vector_offsets(n, one)
    for key, sl in fair.items():
        assert revealed[key] == sl, f"{field} moved the fair field {key}"
    added = [k for k in revealed if k not in fair]
    fair_width = sum(f.size for f in vector_layout(n))
    width = sum(f.size for f in vector_layout(n, one))
    if field == "enemy_elixir":
        assert added == []
        assert width == fair_width
        # The slot's DESCRIPTION flips from the count to the read value, which is
        # the only way a reader can tell which one a run used. (dict() keys here
        # are whole "key: doc" sentences, so a bare `in` on the dict is vacuous.)
        docs = " ".join(name for name, _ in vector_fields(n, one))
        assert "counted from the plays seen" not in docs
        assert "read from the state" in docs
        fair_docs = " ".join(name for name, _ in vector_fields(n))
        assert "counted from the plays seen" in fair_docs
    elif field == "enemy_spell_aim":
        assert added == []  # it adds a CHANNEL, not a slot
        assert width == fair_width
    else:
        assert added
        assert width > fair_width
        assert all(not f.fair for f in vector_layout(n, one) if f.key in added)


def test_only_a_reveal_writes_the_enemy_spell_aim_channel():
    fair = SpatialObsBuilder()
    fair.bind(ENG, PARSER)
    assert "enemy_spell_aim" not in fair.channel_names()
    assert "own_spell_aim" in fair.channel_names(), "the player chose where to throw their own"
    seen = SpatialObsBuilder(reveal=Reveal(enemy_spell_aim=True))
    seen.bind(ENG, PARSER)
    assert seen.channel_names() == [*fair.channel_names(), "enemy_spell_aim"]
    # And the fair builder really has no plane carrying it: on a state with an
    # enemy spell in flight, nothing it writes equals that spell's aim tile.
    s = STATES[-1]
    enemy_aims = [q for q in s.spells if q.team != BLUE]
    assert enemy_aims, "fixture must carry an enemy spell"
    fair_sp = fair.build(s, BLUE, PARSER.action_mask(s, BLUE))["spatial"]
    seen_sp = seen.build(s, BLUE, PARSER.action_mask(s, BLUE))["spatial"]
    assert np.array_equal(fair_sp, seen_sp[: fair_sp.shape[0]])
    assert seen_sp[-1].any(), "the revealed channel must carry something here"


def test_the_entity_list_builder_hides_the_enemy_aim_point_too():
    """The reveal APPENDS two columns; the fair rows never carry the enemy aim."""
    s = STATES[-1]
    built = {}
    for reveal in (None, Reveal(enemy_spell_aim=True)):
        b = EntityListObsBuilder(reveal=reveal)
        b.bind(ENG, PARSER)
        built[bool(reveal)] = (b, b.build(s, BLUE, PARSER.action_mask(s, BLUE))["spells"])
    (fair_b, fair), (seen_b, seen) = built[False], built[True]
    assert seen.shape[1] == fair.shape[1] + 2, "a reveal changes the width (rule (a))"
    assert seen_b.channel_names()[-2:] == ["spell.enemy_aim_x_own", "spell.enemy_aim_y_own"]
    assert fair_b.observation_space() != seen_b.observation_space()
    enemy = seen[:, 2] > 0
    own_rows = (seen[:, 1] > 0) & (seen[:, 0] > 0)
    assert enemy.any(), "fixture must carry an enemy spell row"
    assert own_rows.any(), "fixture must carry an own spell row"
    # The appended columns carry the enemy aim and nothing else.
    assert seen[enemy][:, -2:].any()
    assert not seen[own_rows][:, -2:].any()
    # The fair columns 9, 10 are the viewer's OWN aim, on own rows only.
    assert fair[own_rows][:, 9:11].any()
    assert not fair[enemy][:, 9:11].any()
    # And the fair block is byte-identical with the reveal on: nothing moved.
    assert np.array_equal(fair, seen[:, : fair.shape[1]])


def _two_enemy_spells(aim_x_a, aim_x_b):
    """One state with two enemy spells alike in everything a fair viewer can see,
    differing only in where they are going -- and in delay and hits, which ARE
    visible and must therefore not be ordered by the hidden aim."""
    fb = card_id("Fireball", ENG)
    t = ENG.arena().subtile
    return msgspec.structs.replace(
        MOCK_STATES[0],
        spells=[
            SpellState(RED, fb, SpellMotion.FLIGHT, 5 * t, 11 * t, aim_x_a, 9 * t, 3, 0, 0, 1),
            SpellState(RED, fb, SpellMotion.FLIGHT, 5 * t, 11 * t, aim_x_b, 9 * t, 7, 0, 0, 2),
        ],
    )


def test_the_fair_spell_rows_do_not_change_when_only_the_hidden_aim_changes():
    """Row ORDER is observable, so it may not be decided by hidden data.

    With the aim ranked early in the sort key, these two states -- identical in every
    visible field -- gave delay columns [0.03, 0.07] and [0.07, 0.03]. The aim is
    last in the key now, so it can only order rows whose visible fields all tie, and
    those rows write the same numbers.
    """
    t = ENG.arena().subtile
    a = _two_enemy_spells(2 * t, 14 * t)
    b = _two_enemy_spells(14 * t, 2 * t)
    fair = EntityListObsBuilder()
    fair.bind(ENG, PARSER)
    ra = fair.build(a, BLUE, PARSER.action_mask(a, BLUE))["spells"]
    rb = fair.build(b, BLUE, PARSER.action_mask(b, BLUE))["spells"]
    assert np.array_equal(ra, rb), (ra[:2, 11], rb[:2, 11])
    # Vacuity: the two states really do differ, and a builder allowed to see the aim
    # really does tell them apart.
    assert a.spells != b.spells
    seen = EntityListObsBuilder(reveal=Reveal(enemy_spell_aim=True))
    seen.bind(ENG, PARSER)
    sa = seen.build(a, BLUE, PARSER.action_mask(a, BLUE))["spells"]
    sb = seen.build(b, BLUE, PARSER.action_mask(b, BLUE))["spells"]
    assert not np.array_equal(sa, sb)


def test_every_motion_sets_one_motion_bit_and_an_unknown_motion_is_refused():
    """Every engine motion sets exactly one of the four motion bits, so a spells row keeps
    its width when the engine adds a motion. A fuse reads as a flight (it acts at its aim
    after its delay), so a Rage bottle is not the Rage area it releases. A code SpellMotion
    does not name raises: written as nothing, it would be a row with no motion bit set."""
    bit = {
        SpellMotion.FLIGHT: "motion_flight",
        SpellMotion.AIRBORNE: "motion_airborne",
        SpellMotion.ROLLING: "motion_rolling",
        SpellMotion.AREA: "motion_area",
        SpellMotion.PULSING: "motion_area",
        SpellMotion.FUSE: "motion_flight",
        SpellMotion.STRIKES: "motion_area",
        SpellMotion.SCHEDULED: "motion_area",
    }
    assert set(bit) == set(SpellMotion), "a motion this test does not name"
    t = ENG.arena().subtile
    zap = card_id("Zap", ENG)
    col = {n: i for i, n in enumerate(obs_mod.SPELL_FEATURE_NAMES)}
    motion_cols = [col[n] for n in ("motion_flight", "motion_airborne", "motion_rolling")]
    motion_cols.append(col["motion_area"])
    b = EntityListObsBuilder()
    b.bind(ENG, PARSER)

    def row(motion):
        spell = SpellState(BLUE, zap, motion, 3 * t, 20 * t, 3 * t, 20 * t, 0, 0, 0, 0)
        s = msgspec.structs.replace(MOCK_STATES[0], spells=[spell])
        return b.build(s, BLUE, PARSER.action_mask(s, BLUE))["spells"][0]

    for motion in SpellMotion:
        got = row(motion)
        assert got[motion_cols].sum() == 1, (motion, got[motion_cols])
        assert got[col[bit[motion]]] == 1, (motion, got[motion_cols])
    for unknown in (len(SpellMotion), -1):
        with pytest.raises(ValueError, match=rf"spell motion {unknown} "):
            row(unknown)


def test_an_unknown_motion_is_refused_even_on_a_row_past_max_spells():
    """The refusal reads every spell, not only the rows the builder keeps: a check on the
    kept rows alone would pass until the unknown spell sorted into them."""
    t = ENG.arena().subtile
    zap = card_id("Zap", ENG)
    b = EntityListObsBuilder(max_spells=2)
    b.bind(ENG, PARSER)
    own = [
        SpellState(BLUE, zap, SpellMotion.AREA, 3 * t, y * t, 3 * t, y * t, 0, 0, 0, 0)
        for y in (2, 3)
    ]
    # An enemy row sorts after both own rows, so it is the one dropped.
    enemy = SpellState(RED, zap, len(SpellMotion), 9 * t, 20 * t, 9 * t, 20 * t, 0, 0, 0, 0)
    s = msgspec.structs.replace(MOCK_STATES[0], spells=[*own, enemy])
    with pytest.raises(ValueError, match=rf"spell motion {len(SpellMotion)} "):
        b.build(s, BLUE, PARSER.action_mask(s, BLUE))
    # Control: the same three rows with a known motion build, and the enemy row is dropped.
    ok = msgspec.structs.replace(s, spells=[*own, msgspec.structs.replace(enemy, motion=0)])
    rows = b.build(ok, BLUE, PARSER.action_mask(ok, BLUE))["spells"]
    assert rows.shape[0] == 2
    assert not rows[:, 2].any(), "the enemy row should be the one past max_spells"


def test_plant_the_aim_ranked_early_in_the_spell_key_leaks_it(monkeypatch):
    """Put the defect back: rank the hidden aim before the visible fields."""
    t = ENG.arena().subtile
    a = _two_enemy_spells(2 * t, 14 * t)
    b = _two_enemy_spells(14 * t, 2 * t)

    def aim_first(row):
        # (enemy, y, x, motion, card, delay, trav, length, hits, ay, ax) -> aim early
        return (*row[:5], *row[9:11], *row[5:9])

    fair = EntityListObsBuilder()
    fair.bind(ENG, PARSER)
    before = fair.build(a, BLUE, PARSER.action_mask(a, BLUE))["spells"]
    assert np.array_equal(
        before, fair.build(b, BLUE, PARSER.action_mask(b, BLUE))["spells"]
    ), "baseline must be green"
    monkeypatch.setattr(obs_mod, "spell_row_key", aim_first)
    assert obs_mod.spell_row_key is aim_first, "plant did not land"
    ra = fair.build(a, BLUE, PARSER.action_mask(a, BLUE))["spells"]
    rb = fair.build(b, BLUE, PARSER.action_mask(b, BLUE))["spells"]
    assert not np.array_equal(ra, rb), (
        "PLANT DID NOT LAND: an aim-first key left the fair rows unchanged"
    )


# --- crown towers are their own channels --------------------------------------


def test_crown_towers_are_counted_apart_from_buildings():
    b = SpatialObsBuilder()
    b.bind(ENG, PARSER)
    names = b.channel_names()
    assert "own_towers" in names
    assert "enemy_towers" in names
    assert "own_troop_zone" not in names, "the mask already says this"
    assert "own_building_zone" not in names
    assert "enemy_troop_zone" in names, "not in any mask, so it stays"
    for s in STATES:
        for seat in (BLUE, RED):
            sp = b.build(s, seat, PARSER.action_mask(s, seat))["spatial"]
            for side, team in (("own", seat), ("enemy", 1 - seat)):
                towers = sum(
                    1
                    for e in s.entities
                    if e.team == team
                    and e.kind in (EntityKind.KING_TOWER, EntityKind.PRINCESS_TOWER)
                )
                builds = sum(
                    1 for e in s.entities if e.team == team and e.kind == EntityKind.BUILDING
                )
                assert sp[CHANNEL[f"{side}_towers"]].sum() == towers
                assert sp[CHANNEL[f"{side}_buildings"]].sum() == builds
    # Vacuity: the sampled states really do have towers standing.
    assert b.build(STATES[0], BLUE, PARSER.action_mask(STATES[0], BLUE))["spatial"][
        CHANNEL["own_towers"]
    ].sum() > 0


# --- mask planes ---------------------------------------------------------------


@pytest.mark.parametrize("builder_cls", [SpatialObsBuilder, EntityListObsBuilder])
def test_mask_planes_are_the_flat_mask_without_the_no_op(builder_cls):
    b = builder_cls()
    b.bind(ENG, PARSER)
    for s in STATES[:6]:
        for seat in (BLUE, RED):
            mask = PARSER.action_mask(s, seat)
            o = b.build(s, seat, mask)
            assert o["mask_planes"].shape == (HAND_SIZE, 32, 18)
            assert o["mask_planes"].dtype == np.int8
            assert np.array_equal(o["mask_planes"].reshape(-1), mask[1:])
            # And the planes index the way the action space encodes: plane,row,col
            # is exactly the action that plays that slot on that tile.
            for slot in range(HAND_SIZE):
                for ty, tx in ((0, 0), (5, 9), (31, 17)):
                    assert o["mask_planes"][slot, ty, tx] == mask[PARSER.encode(slot, tx, ty)]
    assert any(
        b.build(s, BLUE, PARSER.action_mask(s, BLUE))["mask_planes"].any() for s in STATES[:6]
    ), "vacuous: no legal placement in any sampled state"


# --- what the builder remembers across a match --------------------------------
#
# The fair vector is stateful: the opponent's elixir is COUNTED, not read, and the
# cycle features accumulate. Three things have to hold and each has a test here:
# the count is exact (against the value the reveal reads), the memory follows the
# game's cycle rule, and nothing survives a reset.


def _rollout(env, steps, seed, deploy_prob=0.4, on_step=None):
    """Random legal two-sided play through the env, resetting when an episode ends."""
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        if not env.agents:
            obs, _ = env.reset()
            continue
        acts = {}
        for a in env.agents:
            legal = np.flatnonzero(obs[a]["action_mask"])[1:]
            acts[a] = (
                int(rng.choice(legal)) if legal.size and rng.random() < deploy_prob else 0
            )
        obs, _, _, _, _ = env.step(acts)
        if on_step is not None:
            on_step(env, obs)


# Ticks between two scripted plays: long enough that the bar is back at the cap
# (a card costs at most MAX_MANA, and 1x regeneration is one elixir per 56 ticks).
TICKS_PER_PLAY = 400


def _counting_env(**kwargs):
    return ClashParallelEnv(
        engine=MockEngine(),
        state_mutator=DefaultStateMutator(decks=[DECK, DECK[::-1]]),
        truncation_cond=StepLimitCondition(200),
        **kwargs,
    )


def test_the_counted_enemy_elixir_equals_the_bar_the_engine_keeps():
    """The whole reason the count is allowed in the fair set: it is EXACT.

    Both directions are checked -- the count against the state, and the fair
    vector slot against the same slot of a ``Reveal(enemy_elixir=True)`` builder --
    because the first is the invariant and the second is what a policy sees.
    """
    env = _counting_env()
    fair = env.obs_builder
    cheat = SpatialObsBuilder(reveal=Reveal(enemy_elixir=True))
    cheat.bind(env.engine, env.action_parser)
    slot = vector_offsets(len(env.engine.cards()))["enemy_elixir"]
    errors = []
    seen = {"spent": 0, "at_cap": 0}
    last = {"tick": -1}

    def check(e, obs):
        st = e.battle_state
        if st.tick < last["tick"]:  # a reset happened: the cheat builder resets too
            cheat.reset(st)
        last["tick"] = st.tick
        for agent, team in AGENT_TEAM.items():
            mem = fair.memory[team]
            want = st.players[1 - team].elixir_milli
            if mem.enemy_elixir_milli() != want:
                errors.append(f"tick {st.tick} seat {team}: counted {mem.enemy_elixir_milli()} "
                              f"!= {want}")
            if mem.own_elixir_milli() != st.players[team].elixir_milli:
                errors.append(f"tick {st.tick} seat {team}: own bar drifted")
            v = cheat.build(st, team, e.action_masks(agent).astype(np.int8))["vector"]
            if not np.array_equal(v[slot], obs[agent]["vector"][slot]):
                errors.append(f"tick {st.tick} seat {team}: fair slot != revealed slot")
            if st.players[team].elixir_milli >= 1000 * fair.max_mana:
                seen["at_cap"] += 1
        seen["spent"] = max(seen["spent"], fair.memory[BLUE].foe_plays)

    # Two games, because the count has two ways to be wrong and each needs one:
    # a BUSY game exercises paying for plays seen, a QUIET one exercises the cap
    # (1x regeneration is one elixir per 56 ticks, so a bar only fills while nobody
    # is spending).
    _rollout(env, 220, seed=3, deploy_prob=0.4, on_step=check)
    _rollout(env, 150, seed=4, deploy_prob=0.03, on_step=check)
    assert errors[:5] == [], f"{len(errors)} mismatches, first few shown"
    assert seen["spent"] >= 5, seen
    assert seen["at_cap"] >= 1, "no seat ever sat at full elixir: the cap is untested"


class _UncappedLaw:
    """``ElixirLaw`` with the cap removed, for the plant below."""

    def __init__(self, law):
        self._law = law

    def __getattr__(self, name):
        return getattr(self._law, name)

    def advance(self, fine, spent, t0, t1, regular, overtime):
        fine, leaked = self._law.advance(fine, spent, t0, t1, regular, overtime)
        return fine + leaked, leaked


def test_plant_a_counter_that_ignores_the_cap_is_caught():
    """Without the clamp the count runs away as soon as a player sits at full.

    Graded on a QUIET game (few plays), because that is when a bar sits at the cap
    and regeneration is actually thrown away; a busy game spends the elixir before
    the clamp ever bites and the plant would pass for the wrong reason.
    """
    env = _counting_env()
    for mem in env.obs_builder.memory.values():
        mem.law = _UncappedLaw(mem.law)
    bad = []

    def check(e, obs):
        st = e.battle_state
        for team in (BLUE, RED):
            mem = e.obs_builder.memory[team]
            if mem.enemy_elixir_milli() != st.players[1 - team].elixir_milli:
                bad.append(st.tick)

    _rollout(env, 120, seed=3, deploy_prob=0.05, on_step=check)
    assert bad, "PLANT DID NOT LAND: an uncapped count still matched the engine"


def test_the_cycle_behind_the_hand_is_deduced_from_the_plays_alone():
    """Positions 6-8 start unknown and are learned one per play (obs.py MatchMemory)."""
    eng = MockEngine()
    parser = TileActionParser()
    parser.bind(eng)
    b = SpatialObsBuilder()
    b.bind(eng, parser)
    eng.reset(
        4,
        MatchSetup(decks=[DECK, DECK], shuffle=ShuffleMode.NONE, elixir_milli=[10000, 10000]),
    )
    b.reset(eng.state())
    mem = b.memory[BLUE]
    assert mem.own_cycle[0] == DECK[4], "position 5 is next_card and is always known"
    assert mem.own_cycle[1:] == [-1, -1, -1], "6-8 are not knowable at the start"
    known = [3]
    for _ in range(3):
        st = eng.state()
        legal = np.flatnonzero(parser.action_mask(st, BLUE))[1:]
        per = parser.nx * parser.ny
        slot0 = [a for a in legal if (a - 1) // per == 0]
        assert slot0, "hand slot 0 must be playable"
        eng.step([parser.parse(int(slot0[0]), st, BLUE)], TICKS_PER_PLAY)
        b.build(eng.state(), BLUE, parser.action_mask(eng.state(), BLUE))
        known.append(sum(1 for c in mem.own_cycle if c == -1))
    assert known == [3, 2, 1, 0], f"one position learned per play, got {known}"
    # And what it learned is the engine's actual queue.
    queue = list(eng._sim().queues[BLUE])
    assert mem.own_cycle == queue
    assert set(np.flatnonzero(mem.own_deck)) == set(DECK)


def test_enemy_possible_hand_follows_the_eight_card_cycle_rule():
    """A card played is out of hand for exactly four more plays, and no longer."""
    eng = MockEngine()
    parser = TileActionParser()
    parser.bind(eng)
    b = SpatialObsBuilder()
    b.bind(eng, parser)
    n = len(eng.cards())
    eng.reset(
        5,
        MatchSetup(decks=[DECK, DECK], shuffle=ShuffleMode.NONE, elixir_milli=[10000, 10000]),
    )
    b.reset(eng.state())
    mem = b.memory[BLUE]  # Blue watching Red
    assert mem.enemy_possible_hand().all(), "nothing seen yet: every card is possible"
    played = []
    for _ in range(6):
        st = eng.state()
        legal = np.flatnonzero(parser.action_mask(st, RED))[1:]
        per = parser.nx * parser.ny
        slot0 = [a for a in legal if (a - 1) // per == 0]
        assert slot0
        played.append(st.players[RED].hand[0])
        eng.step([parser.parse(int(slot0[0]), st, RED)], TICKS_PER_PLAY)
        b.build(eng.state(), BLUE, parser.action_mask(eng.state(), BLUE))
        possible = mem.enemy_possible_hand()
        out = set(played[-(DECK_SIZE - HAND_SIZE) :])
        assert set(np.flatnonzero(~possible)) == out, played
        assert len(out) <= DECK_SIZE - HAND_SIZE
        assert possible.sum() == n - len(out)
    # Playing hand slot 0 every time walks the whole cycle, so the card played
    # first comes round and is played AGAIN on play 6 -- which is the rule working,
    # not an accident. The card that has come back into hand by now is played[1]:
    # five plays old, and not replayed since.
    assert played[5] == played[0], "fixture should have cycled all the way round"
    assert mem.enemy_possible_hand()[played[1]]
    assert not mem.enemy_possible_hand()[played[5]]
    assert set(np.flatnonzero(mem.foe_seen)) == set(played)


def test_two_episodes_in_the_same_env_do_not_leak_state_into_each_other():
    """The strongest form: the same seed replayed in the SAME env is bit-identical.

    A memory that survived ``reset`` would make the second episode's vector differ
    from the first's on exactly the features that accumulate, so this grades every
    one of them at once.
    """
    env = _counting_env()

    def run(seed):
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        frames = [{a: obs[a]["vector"].copy() for a in AGENTS}]
        for _ in range(40):
            acts = {}
            for a in env.agents:
                legal = np.flatnonzero(obs[a]["action_mask"])[1:]
                acts[a] = int(rng.choice(legal)) if legal.size and rng.random() < 0.5 else 0
            obs, *_ = env.step(acts)
            frames.append({a: obs[a]["vector"].copy() for a in AGENTS})
        return frames

    first = run(11)
    _rollout(env, 60, seed=99)  # a different episode in between, to dirty the memory
    second = run(11)
    assert len(first) == len(second)
    bad = [
        (i, a)
        for i, (f, s) in enumerate(zip(first, second, strict=True))
        for a in AGENTS
        if not np.array_equal(f[a], s[a])
    ]
    assert bad == []
    # and the memory really is empty right after a reset
    env.reset(seed=11)
    for team in (BLUE, RED):
        mem = env.obs_builder.memory[team]
        assert mem.foe_plays == 0
        assert not mem.foe_seen.any()
        assert mem.foe_recent == []
        assert mem.leak_fine == 0
        assert mem.own_last_card == -1
        assert mem.own_cycle[1:] == [-1, -1, -1]


def test_a_builder_whose_reset_is_never_called_still_cannot_leak(monkeypatch):
    """The second guard, on its own.

    ``ObsBuilder.reset`` is one guard; the other is inside ``MatchMemory.observe``,
    which re-seeds when the clock moves BACKWARDS, because that can only be a new
    battle. With ``reset`` disabled the episodes must still replay identically --
    that is the backwards-tick guard doing the work, and the plant below removes it
    too to show the guards are what is holding this up.
    """
    monkeypatch.setattr(obs_mod.ObsBuilder, "reset", lambda self, state: None)
    env = _counting_env()

    def run(seed):
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        frames = [obs["blue"]["vector"].copy()]
        for _ in range(40):
            acts = {}
            for a in env.agents:
                legal = np.flatnonzero(obs[a]["action_mask"])[1:]
                acts[a] = int(rng.choice(legal)) if legal.size and rng.random() < 0.5 else 0
            obs, *_ = env.step(acts)
            frames.append(obs["blue"]["vector"].copy())
        return frames

    first = run(11)
    _rollout(env, 60, seed=99)
    second = run(11)
    assert all(np.array_equal(f, s) for f, s in zip(first, second, strict=True))


def test_plant_a_builder_with_neither_guard_leaks_across_episodes(monkeypatch):
    """Remove BOTH guards and the second episode is no longer the first."""
    monkeypatch.setattr(obs_mod.ObsBuilder, "reset", lambda self, state: None)
    original = obs_mod.MatchMemory.observe

    def no_reseed(self, state, team):
        if 0 <= state.tick < self.tick:  # the backwards-tick guard, removed
            self.tick = state.tick
            return
        original(self, state, team)

    monkeypatch.setattr(obs_mod.MatchMemory, "observe", no_reseed)
    env = _counting_env()

    def run(seed):
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        frames = [obs["blue"]["vector"].copy()]
        for _ in range(40):
            acts = {}
            for a in env.agents:
                legal = np.flatnonzero(obs[a]["action_mask"])[1:]
                acts[a] = int(rng.choice(legal)) if legal.size and rng.random() < 0.5 else 0
            obs, *_ = env.step(acts)
            frames.append(obs["blue"]["vector"].copy())
        return frames

    first = run(11)
    _rollout(env, 60, seed=99)
    second = run(11)
    assert any(
        not np.array_equal(f, s) for f, s in zip(first, second, strict=True)
    ), "PLANT DID NOT LAND: a builder that never forgets replayed the episode identically"


def test_leak_and_last_play_features_move_with_the_match():
    """The three scalar memory features are not constants."""
    env = _counting_env()
    off = vector_offsets(len(env.engine.cards()))
    seen = {k: set() for k in ("own_elixir_leaked", "own_ticks_since_play", "enemy_plays")}

    def note(e, obs):
        for k in seen:
            seen[k].add(float(obs["blue"]["vector"][off[k]][0]))

    # Quiet play, so the bar fills and the leak feature has something to report.
    _rollout(env, 190, seed=7, deploy_prob=0.05, on_step=note)
    for k, values in seen.items():
        assert len(values) > 1, f"{k} never changed: {values}"
        assert max(values) > 0.0, f"{k} was always zero"


def test_reveal_reports_whether_anything_is_open_and_refuses_a_bare_bool():
    assert not Reveal().any_enabled
    assert Reveal(enemy_hand=True).any_enabled
    assert ALL_REVEALED.any_enabled
    assert Reveal().as_dict() == dict.fromkeys(Reveal().as_dict(), False)
    # The constructors used to take ``reveal_enemy_elixir: bool`` in this position.
    with pytest.raises(TypeError, match="Reveal"):
        SpatialObsBuilder(True)
    with pytest.raises(TypeError, match="Reveal"):
        EntityListObsBuilder(96, True)


def test_the_memory_says_when_its_count_can_no_longer_be_exact():
    """The count is checked against the ONE bar the memory is allowed to look at.

    A deck with a repeated card is the case this catches: a play that swaps a card
    for itself changes no hand slot, so it is unseen, and the count is wrong from
    then on. It must not be wrong quietly.
    """
    env = _counting_env()
    _rollout(env, 60, seed=3, deploy_prob=0.4)
    assert all(m.exact for m in env.obs_builder.memory.values()), "a normal deck stays exact"

    # Scripted, not sampled: the repeated card is played from a known slot, so the
    # hidden play is a fact of the fixture rather than something a seed has to
    # stumble into.
    # The card at cycle position 5 is the one at hand slot 0, so playing slot 0
    # draws the same card straight back into it and NOTHING in the hand changes.
    doubled = card_ids(
        ("Knight", "Giant", "Cannon", "Log", "Knight", "Fireball", "Zap", "Minions"), _CARDS
    )
    eng = MockEngine()
    parser = TileActionParser()
    parser.bind(eng)
    b = SpatialObsBuilder()
    b.bind(eng, parser)
    eng.reset(
        0,
        MatchSetup(
            decks=[doubled, doubled], shuffle=ShuffleMode.NONE, elixir_milli=[10000, 10000]
        ),
    )
    b.reset(eng.state())
    per = parser.nx * parser.ny
    before = eng.state().players[RED].hand[0]
    st = eng.state()
    slot0 = [a for a in np.flatnonzero(parser.action_mask(st, RED))[1:] if (a - 1) // per == 0]
    assert slot0
    eng.step([parser.parse(int(slot0[0]), st, RED)], 5)
    assert eng.state().players[RED].hand[0] == before, (
        "fixture needs the repeated card to be drawn back into the same slot"
    )
    b.build(eng.state(), BLUE, parser.action_mask(eng.state(), BLUE))
    blue = b.memory[BLUE]
    assert blue.foe_plays == 0, "the play really is invisible in the hand"
    assert blue.enemy_elixir_milli() != eng.state().players[RED].elixir_milli
    assert not blue.exact, (
        "a deck that repeats a card hides plays, and the memory has to notice"
    )


@pytest.mark.parametrize("reveal", [None, ALL_REVEALED])
def test_spatial_layout_names_every_plane_and_exactly_the_static_ones(reveal):
    """A consumer that stores observations holds the static planes once, not per row.

    The guard is the one that matters: a plane called static must really be the same
    numbers in every state, and a plane NOT called static must vary in at least one
    of them, or the label is doing nothing.
    """
    b = SpatialObsBuilder(reveal=reveal)
    b.bind(ENG, PARSER)
    layout = b.spatial_layout()
    assert [name for name, _ in layout] == b.channel_names()
    declared = {name for name, static in layout if static}
    assert declared == {"water", "no_deploy"}
    constant = set()
    for seat in (BLUE, RED):
        first = b.build(STATES[0], seat, PARSER.action_mask(STATES[0], seat))["spatial"]
        same = np.ones(len(layout), dtype=bool)
        for s in STATES:
            sp = b.build(s, seat, PARSER.action_mask(s, seat))["spatial"]
            same &= np.array([np.array_equal(sp[i], first[i]) for i in range(len(layout))])
        constant |= {b.channel_names()[i] for i in np.flatnonzero(same)}
    assert declared <= constant, f"declared static but varies: {declared - constant}"

    # AND WHY THE DECLARATION IS NEEDED RATHER THAN SAMPLING. The tower planes look
    # static on any sample in which no tower falls -- a crown tower never moves --
    # so a consumer that decides staticness by sampling would hold them once and
    # then never see a tower destroyed. They are not declared static, and here is
    # the state that proves they must not be.
    assert {"own_towers", "enemy_towers"} <= constant - declared
    fallen = msgspec.structs.replace(
        STATES[0],
        entities=[e for e in STATES[0].entities if e.tower_slot != int(TowerSlot.LEFT)],
    )
    assert len(fallen.entities) < len(STATES[0].entities), "fixture must have a princess to drop"
    for seat in (BLUE, RED):
        before = b.build(STATES[0], seat, PARSER.action_mask(STATES[0], seat))["spatial"]
        after = b.build(fallen, seat, PARSER.action_mask(fallen, seat))["spatial"]
        idx = b.channel_names().index("own_towers")
        jdx = b.channel_names().index("enemy_towers")
        assert not np.array_equal(before[idx], after[idx]) or not np.array_equal(
            before[jdx], after[jdx]
        )

    # EntityListObsBuilder has no spatial planes and says so rather than guessing.
    e = EntityListObsBuilder()
    e.bind(ENG, PARSER)
    assert e.spatial_layout() == ()


def test_the_count_survives_a_snapshot_resume():
    """A running bar is not on the lattice ``from_milli`` floors onto.

    ``elixir_milli`` is a floor of an internal value, so seeding a mid-battle resume
    from it starts the count up to ``scale/1000 - 1`` fine units low and it never
    recovers. ``ElixirLaw.seed_fine`` snaps onto the spacing the engine can actually
    reach, which is wider than the milli window, so the snap is unambiguous.
    """
    src = MockEngine()
    env = ClashParallelEnv(engine=src, state_mutator=DefaultStateMutator(decks=[DECK, DECK[::-1]]))
    _rollout(env, 37, seed=1, deploy_prob=0.5)
    blob = src.save_state()
    mid = src.state()
    assert mid.tick > 0
    assert any(p.elixir_milli % 1000 for p in mid.players), "vacuous: both bars on a round value"

    resumed = ClashParallelEnv(
        engine=MockEngine(),
        state_mutator=SnapshotStateMutator([blob]),
        truncation_cond=StepLimitCondition(200),
    )
    bad = []

    def check(e, obs):
        st = e.battle_state
        for team in (BLUE, RED):
            mem = e.obs_builder.memory[team]
            if mem.enemy_elixir_milli() != st.players[1 - team].elixir_milli:
                bad.append(st.tick)

    _rollout(resumed, 60, seed=2, deploy_prob=0.4, on_step=check)
    assert bad == []
    assert all(m.exact for m in resumed.obs_builder.memory.values())


def test_plant_seeding_a_resume_by_flooring_is_caught(monkeypatch):
    """Put the floor back and the resumed count drifts."""
    src = MockEngine()
    env = ClashParallelEnv(engine=src, state_mutator=DefaultStateMutator(decks=[DECK, DECK[::-1]]))
    _rollout(env, 37, seed=1, deploy_prob=0.5)
    blob = src.save_state()
    law = obs_mod.ElixirLaw.load()
    monkeypatch.setattr(type(law), "seed_fine", type(law).from_milli, raising=False)
    resumed = ClashParallelEnv(
        engine=MockEngine(),
        state_mutator=SnapshotStateMutator([blob]),
        truncation_cond=StepLimitCondition(200),
    )
    bad = []

    def check(e, obs):
        st = e.battle_state
        for team in (BLUE, RED):
            if e.obs_builder.memory[team].enemy_elixir_milli() != st.players[1 - team].elixir_milli:
                bad.append(st.tick)

    _rollout(resumed, 60, seed=2, deploy_prob=0.4, on_step=check)
    assert bad, "PLANT DID NOT LAND: a floored resume seed still matched the engine"


def test_exactness_is_checked_on_both_bars_not_just_the_visible_one():
    """Only the OPPONENT's deck repeats a card, so the own bar stays perfect.

    The check that matters is the one on the bar the memory is modelling rather than
    reading. With the own-bar check alone, ``exact`` stayed True on this setup while
    the number it certified was wrong.
    """
    doubled = card_ids(
        ("Knight", "Knight", "Giant", "Giant", "Cannon", "Cannon", "Log", "Log"), _CARDS
    )
    env = ClashParallelEnv(
        engine=MockEngine(),
        state_mutator=DefaultStateMutator(decks=[DECK, doubled]),
        truncation_cond=StepLimitCondition(400),
    )
    wrong = {BLUE: 0, RED: 0}

    def check(e, obs):
        st = e.battle_state
        for team in (BLUE, RED):
            mem = e.obs_builder.memory[team]
            if mem.enemy_elixir_milli() != st.players[1 - team].elixir_milli:
                wrong[team] += 1

    _rollout(env, 200, seed=3, deploy_prob=0.6, on_step=check)
    blue = env.obs_builder.memory[BLUE]
    assert wrong[BLUE] > 0, "vacuous: Blue's count of Red's repeating deck never went wrong"
    assert not blue.exact, "the count was wrong and the flag did not say so"


# --- every channel and every field, by CONTENT ---------------------------------
#
# The coverage guard above only says a channel is not all-zero SOMEWHERE, and the
# flip and order gates only say two builds agree with each other. Neither can see
# two channels swapped, or a field written into its neighbour's slots -- and the
# rewrite that split crown towers out of the buildings planes moved every index
# after own_buildings. These two tests check the contents against the state.


def entity_channel_errors(builder, states=STATES) -> list[str]:
    """Every entity channel, cell by cell, against the entity list, for both seats."""
    builder.bind(ENG, PARSER)
    a = ENG.arena()
    names = builder.channel_names()
    errors = []
    for i, s in enumerate(states):
        for seat in (BLUE, RED):
            sp = builder.build(s, seat, PARSER.action_mask(s, seat))["spatial"]
            want = {n: np.zeros((32, 18)) for n in names[:obs_mod.ENTITY_CHANNELS]}
            for e in s.entities:
                ox, oy = to_own(a, seat, e.x, e.y)
                ty, tx = min(max(oy // a.subtile, 0), 31), min(max(ox // a.subtile, 0), 17)
                side = "own" if e.team == seat else "enemy"
                if e.kind == EntityKind.TROOP:
                    want[f"{side}_{'air' if e.flying else 'ground'}_troops"][ty, tx] += 1
                elif e.kind == EntityKind.BUILDING:
                    want[f"{side}_buildings"][ty, tx] += 1
                else:
                    want[f"{side}_towers"][ty, tx] += 1
                want[f"{side}_hp"][ty, tx] += e.hp / obs_mod.HP_SCALE
                if e.deploy_ticks > 0:
                    want[f"{side}_deploying"][ty, tx] += 1
            for name, grid in want.items():
                got = sp[names.index(name)]
                if not np.allclose(got, grid, rtol=0, atol=1e-3):
                    errors.append(f"state {i} seat {seat}: {name}")
    return errors


def test_every_entity_channel_holds_what_the_entity_list_says():
    assert entity_channel_errors(SpatialObsBuilder()) == []


@pytest.mark.parametrize(
    "swap",
    [
        ("own_ground_troops", "own_air_troops"),
        ("own_buildings", "own_towers"),
        ("own_hp", "enemy_hp"),
        ("own_deploying", "enemy_deploying"),
        ("enemy_ground_troops", "enemy_air_troops"),
    ],
)
def test_plant_two_swapped_entity_channels_are_caught(swap):
    """The gate the coverage guard could not be: both planes stay non-zero."""
    lo, hi = (CHANNEL[swap[0]], CHANNEL[swap[1]])

    class Swapped(SpatialObsBuilder):
        def build(self, state, team, action_mask):
            o = super().build(state, team, action_mask)
            o["spatial"][[lo, hi]] = o["spatial"][[hi, lo]]
            return o

    assert unseen_channels(STATES) == [], "baseline coverage must be green"
    bad = entity_channel_errors(Swapped())
    assert bad, f"PLANT DID NOT LAND: swapping {swap} went unseen"


def _vector_content_errors(builder, eng, parser, plays, seat=BLUE) -> list[str]:
    """Named vector fields against ground truth computed from the engine, not the
    builder. ``plays`` is the true list of cards each team has played, in order."""
    st = eng.state()
    n = len(eng.cards())
    off = vector_offsets(n, builder.reveal)
    v = builder.build(st, seat, parser.action_mask(st, seat))["vector"]
    me, foe = st.players[seat], st.players[1 - seat]
    full = 1000 * builder.max_mana
    errors = []

    def check(key, want):
        got = v[off[key]]
        if not np.allclose(got, np.asarray(want, dtype=np.float32), rtol=0, atol=1e-6):
            errors.append(f"{key}: got {np.round(got, 4)} want {np.round(want, 4)}")

    check("own_elixir", [me.elixir_milli / full])
    check("enemy_elixir", [foe.elixir_milli / full])
    hand = np.zeros((HAND_SIZE, n + 1), dtype=np.float32)
    for i, c in enumerate(me.hand):
        hand[i, n if c == -1 else c] = 1
    check("own_hand_cards", hand.reshape(-1))
    nxt = np.zeros(n + 1, dtype=np.float32)
    nxt[n if me.next_card == -1 else me.next_card] = 1
    check("own_next_card", nxt)
    seen = np.zeros(n, dtype=np.float32)
    for c in plays[1 - seat]:
        seen[c] = 1
    check("enemy_cards_seen", seen)
    possible = np.ones(n, dtype=np.float32)
    for c in plays[1 - seat][-(DECK_SIZE - HAND_SIZE) :]:
        possible[c] = 0
    check("enemy_possible_hand", possible)
    check("enemy_plays", [min(1.0, len(plays[1 - seat]) / obs_mod.PLAYS_SCALE)])
    last = np.zeros(n + 1, dtype=np.float32)
    last[plays[seat][-1] if plays[seat] else n] = 1
    check("own_last_card", last)
    check("crowns", [me.crowns / 3.0, foe.crowns / 3.0])
    check("own_tower_hp", [me.tower_hp[s] / max(1, me.tower_max_hp[s]) for s in TowerSlot])
    check("enemy_tower_hp", [foe.tower_hp[s] / max(1, foe.tower_max_hp[s]) for s in TowerSlot])
    check("elixir_rate", [float(st.elixir_rate == 1), float(st.elixir_rate == 2)])
    return errors


def _scripted_plays(n_plays: int = 5):
    """A battle where the test knows exactly which cards each side played."""
    eng = MockEngine()
    parser = TileActionParser()
    parser.bind(eng)
    eng.reset(
        6,
        MatchSetup(decks=[DECK, DECK], shuffle=ShuffleMode.NONE, elixir_milli=[10000, 10000]),
    )
    b = SpatialObsBuilder()
    b.bind(eng, parser)
    b.reset(eng.state())
    plays = {BLUE: [], RED: []}
    per = parser.nx * parser.ny
    for k in range(n_plays):
        team = BLUE if k % 2 else RED
        st = eng.state()
        legal = np.flatnonzero(parser.action_mask(st, team))[1:]
        slot = k % HAND_SIZE
        options = [a for a in legal if (a - 1) // per == slot]
        if not options:
            continue
        plays[team].append(st.players[team].hand[slot])
        eng.step([parser.parse(int(options[0]), st, team)], TICKS_PER_PLAY)
        b.build(eng.state(), BLUE, parser.action_mask(eng.state(), BLUE))
        b.build(eng.state(), RED, parser.action_mask(eng.state(), RED))
    return eng, parser, b, plays


def test_every_named_vector_field_holds_what_the_state_says():
    eng, parser, b, plays = _scripted_plays()
    assert plays[RED], "vacuous: Red played nothing"
    assert plays[BLUE], "vacuous: Blue played nothing"
    for seat in (BLUE, RED):
        assert _vector_content_errors(b, eng, parser, plays, seat) == []


@pytest.mark.parametrize(
    "swap",
    [
        ("enemy_cards_seen", "enemy_possible_hand"),
        ("own_tower_hp", "enemy_tower_hp"),
        ("own_elixir", "enemy_elixir"),
        ("own_next_card", "own_last_card"),
    ],
)
def test_plant_two_swapped_vector_fields_are_caught(swap):
    eng, parser, b, plays = _scripted_plays()
    off = vector_offsets(len(eng.cards()), b.reveal)
    lo, hi = off[swap[0]], off[swap[1]]
    assert lo.stop - lo.start == hi.stop - hi.start, "fixture needs two runs of equal width"

    class Swapped(SpatialObsBuilder):
        def build(self, state, team, action_mask):
            o = super().build(state, team, action_mask)
            v = o["vector"]
            v[lo], v[hi] = v[hi].copy(), v[lo].copy()
            return o

    planted = Swapped()
    planted.bind(eng, parser)
    planted.reset(eng.state())
    bad = _vector_content_errors(planted, eng, parser, plays)
    assert bad, f"PLANT DID NOT LAND: swapping {swap} went unseen"


def test_a_play_missed_because_two_came_from_one_slot_is_flagged():
    """Two plays from the SAME hand slot inside one observed interval read as one.

    The hand is compared slot by slot, so slot0 going X -> Y -> Z between two
    observations is one change. The count is then wrong, and the guard on the bar
    is what has to notice, because the hand cannot.
    """
    eng = MockEngine()
    parser = TileActionParser()
    parser.bind(eng)
    eng.reset(
        0, MatchSetup(decks=[DECK, DECK], shuffle=ShuffleMode.NONE, elixir_milli=[10000, 10000])
    )
    b = SpatialObsBuilder()
    b.bind(eng, parser)
    b.reset(eng.state())
    per = parser.nx * parser.ny
    for _ in range(2):
        st = eng.state()
        legal = np.flatnonzero(parser.action_mask(st, RED))[1:]
        slot0 = [a for a in legal if (a - 1) // per == 0]
        assert slot0, "hand slot 0 must be playable"
        eng.step([parser.parse(int(slot0[0]), st, RED)], 5)
    b.build(eng.state(), BLUE, parser.action_mask(eng.state(), BLUE))
    mem = b.memory[BLUE]
    st = eng.state()
    assert mem.foe_plays == 1, "the fixture must actually hide a play"
    assert mem.enemy_elixir_milli() != st.players[RED].elixir_milli
    assert not mem.exact, "a wrong count must never be a quiet one"


def test_the_state_corpus_exercises_every_entity_channel_non_trivially():
    """`entity_channel_errors` compares each channel against the entity list. A channel
    that is ZERO in every state compares zeros with zeros and passes whatever the builder
    does with it, so the test above would keep its universal name and lose the channel.

    This is the I10 shape from the integration harness, checked here rather than assumed:
    that check asserted determinism over 92 hash observations in which 0 contained a unit,
    so a planted defect landed and had nothing to act on. The population was empty of the
    thing the check existed to catch, and nothing said so.

    Measured when written, over 32 states x 2 seats: the thinnest channel is
    `own_deploying` at 20 of 64 and none is ever absent. The floor is low on purpose --
    this exists to catch a channel dropping to ZERO, not to pin a distribution.
    """
    builder = SpatialObsBuilder()
    builder.bind(ENG, PARSER)
    arena = ENG.arena()
    names = builder.channel_names()[: obs_mod.ENTITY_CHANNELS]
    seen = dict.fromkeys(names, 0)
    for state in STATES:
        for seat in (BLUE, RED):
            want = {n: np.zeros((32, 18)) for n in names}
            for e in state.entities:
                ox, oy = to_own(arena, seat, e.x, e.y)
                ty = min(max(oy // arena.subtile, 0), 31)
                tx = min(max(ox // arena.subtile, 0), 17)
                side = "own" if e.team == seat else "enemy"
                if e.kind == EntityKind.TROOP:
                    want[f"{side}_{'air' if e.flying else 'ground'}_troops"][ty, tx] += 1
                elif e.kind == EntityKind.BUILDING:
                    want[f"{side}_buildings"][ty, tx] += 1
                else:
                    want[f"{side}_towers"][ty, tx] += 1
                want[f"{side}_hp"][ty, tx] += e.hp / obs_mod.HP_SCALE
                if e.deploy_ticks > 0:
                    want[f"{side}_deploying"][ty, tx] += 1
            for name, grid in want.items():
                if grid.any():
                    seen[name] += 1
    never = sorted(n for n, count in seen.items() if count == 0)
    assert not never, (
        f"these entity channels are zero in every one of {len(STATES)} states on both "
        f"seats, so comparing them proves nothing: {never}. Add a state that exercises "
        "them, or the universal name on the check above is no longer true."
    )
