"""Every field that should move, moves. Every field that should not, does not.

WHY THIS IS SEPARATE FROM THE OTHER GATES
    A neighbouring project spent sixty experiment loops on reinforcement learning
    that could not work, and the post-mortem found the trunk had collapsed: cosine
    0.991 between completely different boards. There was nothing to learn from. The
    observation was fine there, so the diagnostic belongs with the encoder -- but the
    same failure can happen one layer earlier, where it would be THIS repo's bug, and
    every downstream failure would be blamed on the learner instead.

    Nothing else here would see it. The seat-flip gate compares two builds with each
    other, so an observation that ignored the board entirely would pass it. The
    coverage guard asks whether a plane is non-zero somewhere, which a constant plane
    satisfies. The ground-truth files check that what IS written is right, not that
    anything changes when the battle does.

    So these tests change ONE thing about a battle and require the observation to
    notice -- and require the fields that have nothing to do with that change to stay
    exactly where they were. The second half is the one that catches more: a field
    wired to the wrong source moves when it should not, and that is invisible to any
    test that only asks "did something change".

MOST OF THE OBSERVATION IS CONSTANT, AND THAT IS WORTH KNOWING BEFORE YOU MEASURE
    Across eight boards that differ only in where one unit stands, which tower is
    down and how much elixir each side holds, 51 of the observation's 11 749 numbers
    differ -- 0.4%. The rest is the arena's static planes, the standing towers and an
    unchanged hand. So a cosine between two raw observations is about 0.9998 no
    matter how well the builder is working, and a naive "cosine must be below X"
    check measures how much of the tensor is constant rather than whether it
    discriminates. The checks below compare the cells that CAN move; on that subspace
    the same eight boards sit at 0.984.

    The same fact is why a consumer storing observations holds the static planes once
    rather than per transition (``ObsBuilder.spatial_layout``), and it is worth
    knowing for anything downstream that normalises.
"""

from __future__ import annotations

import numpy as np
import pytest

from royalegym.action import TileActionParser
from royalegym.mock_engine import MockEngine
from royalegym.obs import (
    EntityListObsBuilder,
    SpatialObsBuilder,
    vector_offsets,
)
from royalegym.protocol import (
    BLUE,
    RED,
    EntityKind,
    MatchSetup,
    ShuffleMode,
    SpawnSpec,
)

DECK = [0, 3, 10, 14, 11, 13, 7, 9]


@pytest.fixture(scope="module")
def engine():
    eng = MockEngine()
    parser = TileActionParser()
    parser.bind(eng)
    return eng, parser


def built(engine, state, team=BLUE, builder=None):
    eng, parser = engine
    b = builder or SpatialObsBuilder()
    if not hasattr(b, "arena"):
        b.bind(eng, parser)
    b.reset(state)
    return b.build(state, team, parser.action_mask(state, team)), b


def base_state(engine, **setup):
    eng, _ = engine
    kwargs = {"decks": [DECK, DECK], "shuffle": ShuffleMode.NONE}
    kwargs.update(setup)
    eng.reset(4, MatchSetup(**kwargs))
    return eng.state()


def moved_fields(engine, before, after, builder_cls=SpatialObsBuilder):
    """Which named vector fields differ between two states, for a fresh builder each.

    A fresh builder per state on purpose: this asks what the STATE implies, not what
    a builder that watched a transition remembers.
    """
    eng, _ = engine
    off = vector_offsets(len(eng.cards()))
    out = set()
    a, _ = built(engine, before, builder=builder_cls())
    b, _ = built(engine, after, builder=builder_cls())
    for name, sl in off.items():
        if not np.array_equal(a["vector"][sl], b["vector"][sl]):
            out.add(name)
    return out, a, b


# --- the board ---------------------------------------------------------------


def test_moving_one_unit_one_tile_moves_the_observation(engine):
    """The cheapest possible collapse: an observation that ignores the board."""
    eng, _ = engine
    sub = eng.arena().subtile
    before = base_state(engine, spawns=[SpawnSpec(BLUE, 0, 6 * sub, 9 * sub)])
    after = base_state(engine, spawns=[SpawnSpec(BLUE, 0, 7 * sub, 9 * sub)])
    a, _ = built(engine, before)
    b, _ = built(engine, after)
    assert not np.array_equal(a["spatial"], b["spatial"]), (
        "a unit moved a whole tile and the spatial tensor did not change"
    )
    # and it moved in the channel it should have, by exactly one tile
    from royalegym.obs import spatial_channels

    ch = {n: i for i, (n, _) in enumerate(spatial_channels())}
    ga, gb = a["spatial"][ch["own_ground_troops"]], b["spatial"][ch["own_ground_troops"]]
    assert np.argwhere(ga > 0).size
    assert np.argwhere(gb > 0).size
    (ay, ax), (by, bx) = np.argwhere(ga > 0)[0], np.argwhere(gb > 0)[0]
    assert (ay, abs(int(bx) - int(ax))) == (by, 1), f"{(ay, ax)} -> {(by, bx)}"
    # the entity-list builder sees it too
    ea, _ = built(engine, before, builder=EntityListObsBuilder())
    eb, _ = built(engine, after, builder=EntityListObsBuilder())
    assert not np.array_equal(ea["entities"], eb["entities"])


def test_two_different_boards_are_two_different_observations(engine):
    """Pairwise distinctness over a spread of boards, not just two.

    This is the shape the collapse post-mortem measured: many states, all alike.
    """
    eng, _ = engine
    sub = eng.arena().subtile
    states = [
        base_state(engine, spawns=[SpawnSpec(BLUE, 0, x * sub, y * sub)])
        for x, y in ((4, 8), (6, 9), (9, 11), (13, 8), (6, 14))
    ]
    states.append(base_state(engine))
    states.append(base_state(engine, tower_hp=[[2400, 0, 1400], [2400, 1400, 1400]]))
    states.append(base_state(engine, elixir_milli=[9000, 2000]))
    # spatial AND vector: elixir is a vector field and does not touch the planes, so
    # comparing only the tensor would call two boards identical that a policy can
    # tell apart. What a policy sees is both.
    flat = []
    for s in states:
        o = built(engine, s)[0]
        flat.append(np.concatenate([o["spatial"].ravel(), o["vector"]]))
    clashes = [
        (i, j)
        for i in range(len(flat))
        for j in range(i + 1, len(flat))
        if np.array_equal(flat[i], flat[j])
    ]
    assert clashes == [], f"distinct boards gave identical observations: {clashes}"

    # AND NOT MERELY DISTINCT -- but measured on the right thing, which took one
    # wrong attempt to find. Raw cosine between these eight is 0.9998, and that is
    # CORRECT rather than collapse: the observation is 11 749 numbers of which 51
    # (0.4%) differ across boards this similar. The rest is the arena, the standing
    # towers and an unchanged hand. A cosine check on the raw vector therefore
    # measures how much of the observation is constant, not whether it discriminates.
    #
    # So compare the cells that CAN move. On that subspace the same eight boards sit
    # at 0.984, and a real collapse -- a builder that stopped reading the board --
    # would send it to 1.0 while the raw number barely moved.
    block = np.array(flat)
    varying = block.max(axis=0) != block.min(axis=0)
    assert varying.sum() > 20, f"only {varying.sum()} cells move across eight boards"
    live = block[:, varying]
    norm = live / np.maximum(np.linalg.norm(live, axis=1, keepdims=True), 1e-9)
    cos = norm @ norm.T
    off_diagonal = cos[~np.eye(len(flat), dtype=bool)]
    worst = float(off_diagonal.max())
    assert worst < 0.999, f"boards are collinear where they differ: {worst}"


# --- one thing at a time, and nothing else -----------------------------------


def test_a_tower_falling_moves_the_tower_fields_and_only_those(engine):
    before = base_state(engine)
    after = base_state(engine, tower_hp=[[2400, 0, 1400], [2400, 1400, 1400]])
    moved, _, _ = moved_fields(engine, before, after)
    assert "own_tower_hp" in moved, "a destroyed tower did not move own_tower_hp"
    # Two others move WITH it and both are the game rather than a crossed wire: the
    # engine awards a crown for the fallen tower, and losing a princess activates the
    # king. If either ever stops moving here, that is the finding.
    assert "crowns" in moved
    assert "king_active" in moved
    assert moved <= {"own_tower_hp", "crowns", "king_active"}, f"also moved {moved}"


def test_spending_elixir_moves_the_elixir_fields_and_only_those(engine):
    before = base_state(engine, elixir_milli=[9000, 5000])
    after = base_state(engine, elixir_milli=[3000, 5000])
    moved, _, _ = moved_fields(engine, before, after)
    assert "own_elixir" in moved
    assert moved <= {"own_elixir", "own_hand_affordable"}, f"also moved {moved}"


def test_the_clock_moves_the_clock_fields_and_only_those(engine):
    before = base_state(engine, start_tick=100)
    after = base_state(engine, start_tick=2600)
    moved, _, _ = moved_fields(engine, before, after)
    assert "clock" in moved, "the match clock did not move the clock field"
    assert "elixir_rate" in moved, "tick 2600 is past the speed-up and the rate did not move"
    # own_ticks_since_play is the time since the last play, and with no play yet that
    # IS the clock, so it moves too and should.
    assert moved <= {
        "clock",
        "elixir_rate",
        "own_ticks_since_play",
        "own_elixir",
        "enemy_elixir",
    }, f"also moved {moved}"


def test_damaging_one_unit_moves_the_hp_plane_and_only_that(engine):
    """Found by a plant: nothing here had two boards differing only in hit points.

    The hp planes could have been written as a unit COUNT -- collapsing away the one
    thing that says whether a push is winning -- and every other check in this file
    stayed green, because every other fixture changes position or ownership too.
    """
    eng, _ = engine
    sub = eng.arena().subtile
    from royalegym.obs import spatial_channels

    names = [n for n, _ in spatial_channels()]
    full = base_state(engine, spawns=[SpawnSpec(BLUE, 0, 6 * sub, 9 * sub)])
    hurt = base_state(engine, spawns=[SpawnSpec(BLUE, 0, 6 * sub, 9 * sub, hp=40)])
    a, _ = built(engine, full)
    b, _ = built(engine, hurt)
    changed = {
        names[i]
        for i in range(len(names))
        if not np.array_equal(a["spatial"][i], b["spatial"][i])
    }
    assert changed == {"own_hp"}, f"one unit lost hit points and this moved: {changed}"
    # and it moved DOWN, by the amount the unit lost
    # entity order is engine-private, so find the troop rather than indexing into it
    troop = next(e for e in full.entities if e.kind == EntityKind.TROOP and e.team == BLUE)
    plane = names.index("own_hp")
    at = tuple(np.argwhere(a["spatial"][plane] != b["spatial"][plane])[0])
    lost = float(a["spatial"][plane][at] - b["spatial"][plane][at])
    assert lost == pytest.approx((troop.hp - 40) / 1000.0, abs=1e-6)


def test_the_enemys_board_moves_the_enemy_fields_and_not_the_owns(engine):
    """The own/enemy split, tested by changing exactly one side."""
    eng, _ = engine
    sub = eng.arena().subtile
    before = base_state(engine, spawns=[SpawnSpec(BLUE, 0, 6 * sub, 9 * sub)])
    after = base_state(
        engine,
        spawns=[SpawnSpec(BLUE, 0, 6 * sub, 9 * sub), SpawnSpec(RED, 0, 6 * sub, 22 * sub)],
    )
    from royalegym.obs import spatial_channels

    names = [n for n, _ in spatial_channels()]
    a, _ = built(engine, before)
    b, _ = built(engine, after)
    changed = {
        names[i]
        for i in range(len(names))
        if not np.array_equal(a["spatial"][i], b["spatial"][i])
    }
    assert changed, "adding an enemy unit changed nothing"
    assert not any(n.startswith("own_") for n in changed), f"an ENEMY unit moved {changed}"
    assert {"enemy_ground_troops", "enemy_hp"} <= changed, changed


# --- the plants: a field wired to nothing, and a field wired to the wrong thing


def test_plant_a_vector_field_wired_to_a_constant_is_caught(engine, monkeypatch):
    """Collapse in its smallest form: one field stops reading the state."""
    import royalegym.obs as obs_mod

    real = obs_mod.build_vector
    off = vector_offsets(len(engine[0].cards()))

    def flat(state, team, cards, max_mana, reveal, memory):
        v = real(state, team, cards, max_mana, reveal, memory)
        v[off["own_tower_hp"]] = 1.0
        return v

    before = base_state(engine)
    after = base_state(engine, tower_hp=[[2400, 0, 1400], [2400, 1400, 1400]])
    assert "own_tower_hp" in moved_fields(engine, before, after)[0], "baseline must be green"
    monkeypatch.setattr(obs_mod, "build_vector", flat)
    moved, _, _ = moved_fields(engine, before, after)
    assert "own_tower_hp" not in moved
    # which is exactly what the assertion in the real test forbids
    with pytest.raises(AssertionError):
        assert "own_tower_hp" in moved, "a destroyed tower did not move own_tower_hp"


def test_plant_a_field_reading_the_wrong_source_is_caught(engine, monkeypatch):
    """The half that catches more: a field that moves when it should not.

    Wire the clock block to the elixir bar. Nothing about the CLOCK breaks -- it still
    varies with the state -- but changing elixir now moves a field that has no business
    moving, which only a test that pins what must NOT change can see.
    """
    import royalegym.obs as obs_mod

    real = obs_mod.build_vector
    off = vector_offsets(len(engine[0].cards()))

    def crossed(state, team, cards, max_mana, reveal, memory):
        v = real(state, team, cards, max_mana, reveal, memory)
        v[off["clock"]] = v[off["own_elixir"]][0]
        return v

    before = base_state(engine, elixir_milli=[9000, 5000])
    after = base_state(engine, elixir_milli=[3000, 5000])
    clean, _, _ = moved_fields(engine, before, after)
    assert clean <= {"own_elixir", "own_hand_affordable"}, "baseline must be green"
    monkeypatch.setattr(obs_mod, "build_vector", crossed)
    moved, _, _ = moved_fields(engine, before, after)
    assert "clock" in moved, "PLANT DID NOT LAND"
    assert not moved <= {"own_elixir", "own_hand_affordable"}, (
        "the crossed wire went unseen: only a test that pins what must NOT move sees this"
    )


# --- the entity rows, same question ------------------------------------------


def test_every_entity_row_field_responds_to_the_entity_it_describes(engine):
    """Each per-entity column varies across a board built to vary it."""
    eng, parser = engine
    sub = eng.arena().subtile
    state = base_state(
        engine,
        tower_hp=[[2400, 0, 1400], [2400, 1400, 1400]],
        spawns=[
            SpawnSpec(BLUE, 0, 6 * sub, 9 * sub),
            SpawnSpec(BLUE, 1, 9 * sub, 11 * sub),
            SpawnSpec(RED, 2, 9 * sub, 22 * sub, hp=50),
        ],
    )
    b = EntityListObsBuilder()
    b.bind(eng, parser)
    rows = b.build(state, BLUE, parser.action_mask(state, BLUE))["entities"]
    present = rows[rows[:, 0] > 0]
    assert len(present) == len(state.entities)
    constant = [i for i in range(present.shape[1]) if len(np.unique(present[:, i])) == 1]
    names = b.channel_names()
    # 'present' is 1 for every present row by definition; the card one-hot columns are
    # mostly zero. Everything else describing a unit should differ somewhere.
    named = len([n for n in names if n.startswith("entity.")]) - 1
    varying = [i for i in range(named) if i not in constant]
    assert len(varying) >= 10, (
        f"only {len(varying)} of {named} entity columns vary on a deliberately varied "
        f"board; constant: {[names[i] for i in constant if i < named]}"
    )


def test_damaging_one_unit_moves_only_that_row(engine):
    eng, parser = engine
    sub = eng.arena().subtile
    def board(hp):
        return base_state(engine, spawns=[
            SpawnSpec(BLUE, 0, 6 * sub, 9 * sub),
            SpawnSpec(BLUE, 0, 9 * sub, 11 * sub, hp=hp),
        ])
    b = EntityListObsBuilder()
    b.bind(eng, parser)
    a_rows = b.build(board(-1), BLUE, parser.action_mask(board(-1), BLUE))["entities"]
    s2 = board(40)
    b_rows = b.build(s2, BLUE, parser.action_mask(s2, BLUE))["entities"]
    differing = [i for i in range(a_rows.shape[0]) if not np.array_equal(a_rows[i], b_rows[i])]
    assert differing, "one unit lost most of its hit points and no row changed"
    assert len(differing) == 1, f"one unit changed but {len(differing)} rows moved: {differing}"


def test_two_different_boards_are_two_different_entity_arrays(engine):
    eng, parser = engine
    sub = eng.arena().subtile
    b = EntityListObsBuilder()
    b.bind(eng, parser)
    flat = []
    for x, y in ((4, 8), (6, 9), (9, 11), (13, 8), (6, 14)):
        s = base_state(engine, spawns=[SpawnSpec(BLUE, 0, x * sub, y * sub)])
        flat.append(b.build(s, BLUE, parser.action_mask(s, BLUE))["entities"].ravel())
    clashes = [
        (i, j)
        for i in range(len(flat))
        for j in range(i + 1, len(flat))
        if np.array_equal(flat[i], flat[j])
    ]
    assert clashes == [], f"distinct boards gave identical entity arrays: {clashes}"


# --- and the thing the post-mortem actually measured -------------------------


def test_a_played_out_battle_does_not_collapse(engine):
    """Observations from one real battle must not all point the same way.

    Not a synthetic board: the states a policy would actually meet. Cosine near 1
    across a whole match is the signature that there is nothing to learn from.
    """
    eng, parser = engine
    eng.reset(9, MatchSetup(decks=[DECK, DECK[::-1]], elixir_milli=[10000, 10000]))
    b = SpatialObsBuilder()
    b.bind(eng, parser)
    b.reset(eng.state())
    rng = np.random.default_rng(9)
    vectors, spatials = [], []
    for step in range(120):
        st = eng.state()
        if st.game_over:
            break
        cmds = []
        for team in (BLUE, RED):
            legal = np.flatnonzero(parser.action_mask(st, team))[1:]
            if legal.size and rng.random() < 0.4:
                cmds.append(parser.parse(int(rng.choice(legal)), st, team))
        eng.step(cmds, 10)
        o = b.build(eng.state(), BLUE, parser.action_mask(eng.state(), BLUE))
        if step % 6 == 0:
            vectors.append(o["vector"].copy())
            spatials.append(o["spatial"].ravel().copy())
    assert len(vectors) >= 12, f"vacuous: only {len(vectors)} samples"
    for name, block in (("vector", vectors), ("spatial", spatials)):
        m = np.array([v / max(float(np.linalg.norm(v)), 1e-9) for v in block])
        cos = m @ m.T
        off = cos[~np.eye(len(block), dtype=bool)]
        assert off.mean() < 0.995, f"{name}: mean pairwise cosine {off.mean():.4f}"
        assert len({tuple(v) for v in block}) == len(block), f"{name}: duplicate observations"


# --- the number a consumer needs before choosing a collapse threshold ---------


def test_measure_variability_reports_both_cosines_and_the_static_fraction(engine):
    """A cosine threshold is meaningless without the input's own cosine.

    The detector for a collapsed representation is a cosine between encoded states
    with a threshold under it. Pick that threshold without knowing what the INPUT's
    cosine already is and it fires on a healthy encoder or stays quiet on a dead
    one, depending only on how much of the observation happens to be static. So the
    builder can report it.
    """
    from royalegym.obs import measure_variability

    eng, parser = engine
    sub = eng.arena().subtile
    states = [
        base_state(engine, spawns=[SpawnSpec(BLUE, 0, x * sub, y * sub)])
        for x, y in ((4, 8), (6, 9), (9, 11), (13, 8), (6, 14))
    ]
    states += [
        base_state(engine),
        base_state(engine, tower_hp=[[2400, 0, 1400], [2400, 1400, 1400]]),
        base_state(engine, elixir_milli=[9000, 2000]),
    ]
    masks = [parser.action_mask(s, BLUE) for s in states]
    b = SpatialObsBuilder()
    b.bind(eng, parser)
    b.reset(states[0])
    v = measure_variability(b, states, masks)

    assert v.states == len(states)
    assert v.cells > 10000
    assert 0 < v.varying < v.cells
    assert v.fraction == pytest.approx(v.varying / v.cells)
    # the whole point: raw cosine is near 1 and that is NOT a defect
    assert v.cosine_raw > 0.99
    assert v.cosine_varying < v.cosine_raw
    assert "cells move" in str(v)

    # the masks are excluded: they are legality, not representation
    assert v.cells == sum(
        int(np.asarray(x).size)
        for k, x in b.build(states[0], BLUE, masks[0]).items()
        if k not in ("action_mask", "mask_planes")
    )


def test_measure_variability_sees_a_collapse(engine, monkeypatch):
    """A builder that stops reading the board sends the varying cosine to 1.0 while
    the raw number barely twitches -- which is exactly why both are reported."""
    import royalegym.obs as obs_mod
    from royalegym.obs import measure_variability

    eng, parser = engine
    sub = eng.arena().subtile
    states = [
        base_state(engine, spawns=[SpawnSpec(BLUE, 0, x * sub, y * sub)])
        for x, y in ((4, 8), (6, 9), (9, 11), (13, 8))
    ]
    masks = [parser.action_mask(s, BLUE) for s in states]

    def fresh():
        b = SpatialObsBuilder()
        b.bind(eng, parser)
        b.reset(states[0])
        return b

    healthy = measure_variability(fresh(), states, masks)
    real = obs_mod.entity_channels
    monkeypatch.setattr(
        obs_mod,
        "entity_channels",
        lambda entities, team, arena: real([], team, arena),  # the board vanishes
    )
    collapsed = measure_variability(fresh(), states, masks)
    assert collapsed.varying < healthy.varying
    assert collapsed.cosine_raw > healthy.cosine_raw
    assert healthy.cosine_raw - collapsed.cosine_raw < 0.01, (
        "the RAW cosine barely moves under total collapse, which is the point"
    )


def test_measure_variability_refuses_what_it_cannot_measure(engine):
    from royalegym.obs import measure_variability

    eng, parser = engine
    s = base_state(engine)
    b = SpatialObsBuilder()
    b.bind(eng, parser)
    with pytest.raises(ValueError, match="at least two"):
        measure_variability(b, [s], [parser.action_mask(s, BLUE)])
    with pytest.raises(ValueError, match="one action mask per state"):
        measure_variability(b, [s, s], [parser.action_mask(s, BLUE)])


def test_the_two_builders_variability_numbers_are_not_comparable(engine):
    """Recorded as a caution, not as a result.

    Both builders are measured here and their numbers DIFFER substantially. That is
    deliberately not asserted as one being worse: the metric compares a
    representation against itself, and across two representations with different
    sparsity and magnitude distributions it is not measuring the same property
    twice. The test pins that both are measurable and that the caution is written
    down, so nobody later reads the gap as a ranking.
    """
    from royalegym.obs import measure_variability

    eng, parser = engine
    sub = eng.arena().subtile
    states = [
        base_state(engine, spawns=[SpawnSpec(BLUE, 0, x * sub, y * sub)])
        for x, y in ((4, 8), (6, 9), (9, 11), (13, 8))
    ]
    masks = [parser.action_mask(s, BLUE) for s in states]
    numbers = {}
    for cls in (SpatialObsBuilder, EntityListObsBuilder):
        b = cls()
        b.bind(eng, parser)
        b.reset(states[0])
        numbers[cls.__name__] = measure_variability(b, states, masks)
    assert all(v.varying > 0 for v in numbers.values())
    # the two representations really are different sizes, which is the point
    sizes = {v.cells for v in numbers.values()}
    assert len(sizes) == 2, f"both builders reported the same width: {sizes}"
    assert "comparing two DIFFERENT builders" in measure_variability.__doc__
