"""EntityListObsBuilder graded against the ENGINE STATE, as tests/test_obs_ground_truth.py
grades the spatial planes.

Same premise, and see that file for it: an agreement test between the seats cannot see a
defect that is symmetric in the seats, and a coverage guard that asks "non-zero
somewhere" is satisfied by the swap it exists to catch. So nothing here compares blue to
red, and every fixture is built so the quantities being compared are MUTUALLY
DISTINGUISHABLE before anything is graded -- four entity kinds with four different
counts, five distinct hit-point values, the two sides deliberately unequal. That last
rule is not decoration: the first version of the spatial file had the air-troop count and
the tower count both at 3, so swapping exactly those two channels passed the check whose
whole purpose was to catch it.

Column meanings come from the builder's own ``channel_names()``: the ``entity.*`` names in
order, then ``entity.card_one_hot`` spreading over the remaining columns, and likewise for
``spell.*``. Everything they are compared against comes from ``engine.state()``,
``protocol.to_own`` and the arena's integers. Written this way by someone who had not
written obs.py, which is the property to keep if these are ever rewritten.

THE SPELL ROWS NEED THE COMPILED ENGINE. MockEngine resolves a spell inside the tick and
reports none, so ``state.spells`` is empty on it and there is nothing to grade the rows
against; a synthetic ``SpellState`` would make the check depend on the same fixtures it is
supposed to be independent of. The spell check therefore runs on RustEngine and skips
without it -- a skip is not a pass.

TWO THINGS THIS SHAPE OF CHECK CANNOT SEE, so nobody reads it as covering them:
  * ``Reveal.enemy_spell_aim`` APPENDS columns rather than filling 9 and 10, so the fair
    and revealed spaces differ in width, and columns 9 and 10 are the VIEWER's own aim,
    zero on an enemy row. Graded in tests/test_env_obs.py.
  * the spell sort key puts the aim LAST on purpose, because ranking it early let the fair
    observation change when only a hidden aim changed. Catching a regression there needs
    two states differing ONLY in a hidden aim with the fair rows required identical, which
    is a different check and lives in tests/test_env_obs.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from royalegym.action import TileActionParser
from royalegym.mock_engine import MockEngine
from royalegym.obs import EntityListObsBuilder
from royalegym.protocol import (
    BLUE,
    RED,
    DeployCommand,
    EntityKind,
    MatchSetup,
    ShuffleMode,
    to_own,
)
from royalegym.rust_engine import RustEngine, core_available

DECK_NAMES = (
    "Knight", "Minions", "Cannon", "Giant", "Archer", "Fireball", "Skeletons", "Musketeer",
)
# Fireball in the opening HAND (the first four), for the spell check.
SPELL_DECK_NAMES = (
    "Fireball", "Knight", "Minions", "Cannon", "Giant", "Archer", "Skeletons", "Musketeer",
)


def fresh(engine, seed=5, tower_hp=None, names=DECK_NAMES):
    cards = {c.name: c for c in engine.cards()}
    deck = [cards[n].card_id for n in names]
    engine.reset(
        seed,
        MatchSetup(
            decks=[deck, deck],
            shuffle=ShuffleMode.NONE,
            elixir_milli=[100000, 100000],
            tower_hp=tower_hp,
            # Past the opening deploy lockout, or every play() below is refused and the
            # rows this file grades are never produced.
            start_tick=engine.rules().deploy_lockout_ticks,
        ),
    )
    return cards


def play(engine, team, card_id, tx, ty, ticks=1):
    """Play a card at the centre of engine tile (tx, ty)."""
    hand = list(engine.state().players[team].hand)
    if card_id not in hand:
        return None
    sub = engine.arena().subtile
    results = engine.step(
        [
            DeployCommand(
                team=team,
                hand_slot=hand.index(card_id),
                x=tx * sub + sub // 2,
                y=ty * sub + sub // 2,
            )
        ],
        ticks,
    )
    return results[0].status


def observe(engine, team):
    parser = TileActionParser()
    parser.bind(engine)
    builder = EntityListObsBuilder()
    builder.bind(engine, parser)
    state = engine.state()
    return builder.build(state, team, parser.action_mask(state, team)), builder


def columns(builder, entity_width, spell_width):
    """Column index per published name, and how wide each trailing one-hot block is."""
    names = builder.channel_names()
    ent = [n.split(".", 1)[1] for n in names if n.startswith("entity.")]
    spl = [n.split(".", 1)[1] for n in names if n.startswith("spell.")]
    e_cols = {n: i for i, n in enumerate(ent) if n != "card_one_hot"}
    s_cols = {n: i for i, n in enumerate(spl) if n != "card_one_hot"}
    return e_cols, len(e_cols), entity_width - len(e_cols), s_cols


def present(rows, cols):
    return [i for i in range(rows.shape[0]) if rows[i, cols["present"]] > 0]


@pytest.fixture(scope="module")
def board():
    """A board whose four entity KINDS have four DIFFERENT counts.

    One Blue princess starts destroyed, so the tower count cannot collide with any
    other, and the rows carry several distinct hit-point values and card ids.
    """
    engine = MockEngine()
    cards = fresh(engine, tower_hp=[[2400, 0, 1400], [2400, 1400, 1400]])
    play(engine, BLUE, cards["Knight"].card_id, 6, 5)
    play(engine, BLUE, cards["Cannon"].card_id, 11, 5)
    play(engine, BLUE, cards["Minions"].card_id, 8, 7)
    state = engine.state()
    obs, builder = observe(engine, BLUE)
    e_cols, named, slots, _ = columns(builder, obs["entities"].shape[1], obs["spells"].shape[1])
    return engine, state, obs["entities"], e_cols, named, slots


def matched_rows(board):
    """Pair each present row with an engine entity BY POSITION.

    Position is the only field both sides agree on before anything else is graded, so
    the pairing cannot inherit the answer to any of the checks below.
    """
    engine, state, ents, e_cols, _, _ = board
    arena = engine.arena()

    def own_xy(entity):
        ox, oy = to_own(arena, BLUE, entity.x, entity.y)
        return round(ox / arena.width, 4), round(oy / arena.height, 4)

    pools = {}
    for entity in state.entities:
        pools.setdefault(own_xy(entity), []).append(entity)
    matched, unmatched = [], []
    for i in present(ents, e_cols):
        key = (round(float(ents[i, e_cols["x_own"]]), 4), round(float(ents[i, e_cols["y_own"]]), 4))
        if pools.get(key):
            matched.append((i, pools[key].pop(0)))
        else:
            unmatched.append((i, key))
    return matched, unmatched


def test_one_row_per_entity(board):
    engine, state, ents, e_cols, named, slots = board
    assert len(present(ents, e_cols)) == len(state.entities)
    assert len(state.entities) > 5, "vacuous: too few entities to grade"
    assert slots == len(engine.cards()) + 1, "one slot per card, plus one for no card"
    assert named == len(e_cols)


def test_x_own_and_y_own_are_to_own_over_the_arena_size(board):
    _, _, ents, e_cols, _, _ = board
    matched, unmatched = matched_rows(board)
    assert unmatched == [], f"rows matched no engine entity by position: {unmatched[:3]}"
    rows = present(ents, e_cols)
    xs = {round(float(ents[i, e_cols["x_own"]]), 4) for i in rows}
    ys = {round(float(ents[i, e_cols["y_own"]]), 4) for i in rows}
    # Several distinct values on each axis, or an x/y swap -- or normalising x by the
    # arena's height -- lands on the same numbers and passes.
    assert len(xs) > 2
    assert len(ys) > 2
    assert len(matched) == len(rows)


def test_hp_frac_and_hp_scaled(board):
    _, _, ents, e_cols, _, _ = board
    matched, _ = matched_rows(board)
    hps = {entity.hp for _, entity in matched}
    assert len(hps) > 2, f"vacuous: only {len(hps)} distinct hp values"
    bad = []
    for i, entity in matched:
        want_frac = entity.hp / entity.max_hp if entity.max_hp else 0.0
        want_scaled = min(entity.hp / 1000.0, 1.0)
        if abs(float(ents[i, e_cols["hp_frac"]]) - want_frac) > 2e-3:
            bad.append((i, "hp_frac", entity.hp, entity.max_hp))
        if abs(float(ents[i, e_cols["hp_scaled"]]) - want_scaled) > 2e-3:
            bad.append((i, "hp_scaled", entity.hp, float(ents[i, e_cols["hp_scaled"]])))
    assert bad == [], f"{len(bad)} mismatches, first {bad[0]}"


def test_kind_one_hots_with_four_distinct_counts(board):
    _, state, ents, e_cols, _, _ = board
    kind_col = {
        EntityKind.TROOP: "kind_troop",
        EntityKind.BUILDING: "kind_building",
        EntityKind.KING_TOWER: "kind_king",
        EntityKind.PRINCESS_TOWER: "kind_princess",
    }
    engine_counts = {k: sum(1 for e in state.entities if e.kind == k) for k in kind_col}
    assert len(set(engine_counts.values())) == len(engine_counts), (
        f"fixture counts collide, so a swap between two kinds passes: {engine_counts}"
    )
    rows = present(ents, e_cols)
    for i in rows:
        assert sum(float(ents[i, e_cols[c]]) for c in kind_col.values()) == 1.0, (
            f"row {i} is not exactly one kind"
        )
    got = {k: float(ents[rows][:, e_cols[c]].sum()) for k, c in kind_col.items()}
    assert got == {k: float(v) for k, v in engine_counts.items()}


def test_own_and_enemy_with_unequal_sides():
    """Each seat is checked against ITS OWN engine counts, never against the other seat."""
    engine = MockEngine()
    cards = fresh(engine)
    play(engine, BLUE, cards["Knight"].card_id, 6, 5)
    play(engine, RED, cards["Knight"].card_id, 6, 26)
    play(engine, RED, cards["Archer"].card_id, 11, 26)
    state = engine.state()
    per_team = {t: sum(1 for e in state.entities if e.team == t) for t in (BLUE, RED)}
    assert per_team[BLUE] != per_team[RED], f"fixture is symmetric: {per_team}"
    for team in (BLUE, RED):
        obs, builder = observe(engine, team)
        ents = obs["entities"]
        e_cols, _, _, _ = columns(builder, ents.shape[1], obs["spells"].shape[1])
        rows = present(ents, e_cols)
        for i in rows:
            assert not (ents[i, e_cols["own"]] and ents[i, e_cols["enemy"]])
        assert float(ents[rows][:, e_cols["own"]].sum()) == per_team[team]
        assert float(ents[rows][:, e_cols["enemy"]].sum()) == per_team[1 - team]


def test_card_one_hot_picks_the_card(board):
    """Measured, not assumed: a Cannon is a BUILDING and carries its own card id, so the
    last slot is for entities with NO card at all -- the crown towers."""
    _, _, ents, _e_cols, named, slots = board
    matched, _ = matched_rows(board)
    seen, bad = set(), []
    for i, entity in matched:
        block = ents[i, named:]
        hot = list(np.flatnonzero(block > 0))
        if entity.card_id >= 0:
            want = entity.card_id
            seen.add(want)
        else:
            want = slots - 1
        if hot != [want] or float(block[want]) != 1.0:
            bad.append((i, entity.kind, entity.card_id, hot, want))
    assert len(seen) > 1, f"vacuous: only {len(seen)} distinct card ids on the board"
    assert bad == [], f"{len(bad)} wrong, first {bad[0]}"


def test_flying_and_deploying_flags(board):
    _, _, ents, e_cols, _, _ = board
    matched, _ = matched_rows(board)
    flying = sum(1 for _, e in matched if e.flying)
    deploying = sum(1 for _, e in matched if e.deploy_ticks > 0)
    assert flying > 0, "vacuous: nothing on the board flies"
    assert deploying > 0, "vacuous: nothing on the board is deploying"
    bad = []
    for i, entity in matched:
        if float(ents[i, e_cols["flying"]]) != float(bool(entity.flying)):
            bad.append((i, "flying", entity.flying))
        if float(ents[i, e_cols["deploying"]]) != float(entity.deploy_ticks > 0):
            bad.append((i, "deploying", entity.deploy_ticks))
    assert bad == [], f"{len(bad)} mismatches, first {bad[0]}"


@pytest.mark.skipif(not core_available(), reason="spell rows need an engine that keeps them live")
def test_spell_rows_follow_state_spells():
    """The compiled engine, because MockEngine reports no live spell to grade against."""
    engine = RustEngine()
    cards = fresh(engine, names=SPELL_DECK_NAMES)
    arena = engine.arena()
    assert play(engine, BLUE, cards["Fireball"].card_id, 9, 20, ticks=1) == 0
    state = engine.state()
    assert state.spells, "the fixture must leave a spell in flight"
    obs, builder = observe(engine, BLUE)
    _, _, _, s_cols = columns(builder, obs["entities"].shape[1], obs["spells"].shape[1])
    rows = present(obs["spells"], s_cols)
    assert len(rows) == len(state.spells)
    spell, row = state.spells[0], obs["spells"][present(obs["spells"], s_cols)[0]]
    x, y = to_own(arena, BLUE, spell.x, spell.y)
    ax, ay = to_own(arena, BLUE, spell.aim_x, spell.aim_y)
    want = {
        "own": 1.0,
        "enemy": 0.0,
        "x_own": x / arena.width,
        "y_own": y / arena.height,
        "own_aim_x_own": ax / arena.width,
        "own_aim_y_own": ay / arena.height,
    }
    # The four positions must be mutually distinguishable, or swapping the spell's
    # centre with its aim point passes.
    quartet = [want["x_own"], want["y_own"], want["own_aim_x_own"], want["own_aim_y_own"]]
    assert len({round(v, 4) for v in quartet}) == 4, f"fixture positions collide: {quartet}"
    bad = {
        k: (float(row[s_cols[k]]), v)
        for k, v in want.items()
        if abs(float(row[s_cols[k]]) - v) > 2e-3
    }
    assert bad == {}
