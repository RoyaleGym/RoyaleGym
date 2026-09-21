"""Action masking: the mask must agree with the engine EXACTLY, in both directions.

A mask that allows an illegal move trains an agent to want it; a mask that forbids
a legal move hides part of the game. Both are silent in training, so they are
checked here exhaustively against ``Engine.check_deploy``.
"""

from __future__ import annotations

import msgspec
import numpy as np
import pytest

from royalegym.action import (
    NOOP,
    HalfTileActionParser,
    PlacementOracle,
    TileActionParser,
    mask_disagreements,
)
from royalegym.env import ClashParallelEnv
from royalegym.mock_engine import MockEngine
from royalegym.protocol import (
    BLUE,
    RED,
    DeployStatus,
    MatchSetup,
    Placement,
    ShuffleMode,
    SpawnSpec,
)
from royalegym.selfplay import RandomLegalOpponent
from royalegym.state_setter import DefaultStateSetter
from royalegym.terminal import GameOverCondition, StepLimitCondition

# Knight Giant Cannon Log | Fireball Zap Minions Valkyrie -- all four placement types
CANNON = 10
MIXED = [0, 3, 10, 14, 11, 13, 7, 9]
# Zap swapped for Goblin Barrel (mock id 15, 2026-09-13): every Placement class,
# SPELL_NOT_ON_WATER included, is in MIXED or MIXED_BARREL.
MIXED_BARREL = [0, 3, 10, 14, 11, 15, 7, 9]


def test_action_space_is_documented_size():
    eng = MockEngine()
    p = TileActionParser()
    p.bind(eng)
    assert p.space.n == 1 + 4 * 18 * 32 == 2305
    h = HalfTileActionParser()
    h.bind(eng)
    assert h.space.n == 1 + 4 * 36 * 64


@pytest.mark.parametrize("parser_cls", [TileActionParser, HalfTileActionParser])
def test_masked_legal_actions_are_never_rejected(parser_cls):
    env = ClashParallelEnv(
        action_parser=parser_cls(),
        state_setter=DefaultStateSetter(decks=[MIXED, MIXED_BARREL]),
        terminal_conditions=[GameOverCondition(), StepLimitCondition(600)],
        decision_ms=250,
    )
    rejected = []
    deploys = {p: 0 for p in Placement}
    for seed in range(2):
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        opp = RandomLegalOpponent(noop_prob=0.4)
        while env.agents:
            acts = {a: opp.act(obs[a], obs[a]["action_mask"], rng) for a in env.agents}
            prev = env.battle_state
            obs, _, _, _, info = env.step(acts)
            for a, act in acts.items():
                st = info[a]["deploy_status"]
                if act != NOOP:
                    if st != DeployStatus.OK:
                        rejected.append((seed, prev.tick, a, act, DeployStatus(st).name))
                    else:
                        team = 0 if a == "blue" else 1
                        slot = env.action_parser.decode(act)[0]
                        card = prev.players[team].hand[slot]
                        deploys[Placement(env.engine.cards()[card].placement)] += 1
    assert rejected == []
    # Not vacuous: every placement type was actually exercised.
    assert all(n >= 5 for n in deploys.values()), deploys


def _probe_states():
    """States that exercise water, no-deploy, pocket, footprints and elixir limits."""
    eng = MockEngine()
    a = eng.arena()
    s = a.subtile
    out = []
    # 1. Opening board, full elixir for Blue, low for Red.
    eng.reset(
        1, MatchSetup(decks=[MIXED, MIXED], shuffle=ShuffleMode.NONE, elixir_milli=[10000, 2500])
    )
    out.append(eng.save_state())
    # 2. Red's own-LEFT princess down (Blue pocket on Blue's right), Blue's own-RIGHT
    #    down (Red pocket), buildings on both sides to exercise footprints.
    eng.reset(
        2,
        MatchSetup(
            decks=[MIXED, list(reversed(MIXED_BARREL))],  # Red hand holds the barrel
            shuffle=ShuffleMode.NONE,
            elixir_milli=[10000, 10000],
            tower_hp=[[2400, 1400, 0], [2400, 0, 1400]],
            spawns=[
                SpawnSpec(BLUE, CANNON, s * 9, s * 10),  # on a tile corner
                SpawnSpec(RED, CANNON, a.width - s * 5 - s // 2, a.height - s * 12 - s // 2),
                SpawnSpec(BLUE, 0, s * 4, s * 20),  # troops do NOT block placement
            ],
        ),
    )
    out.append(eng.save_state())
    # 3. Both of Red's princesses down.
    eng.reset(
        3,
        MatchSetup(
            decks=[MIXED, MIXED],
            shuffle=ShuffleMode.NONE,
            elixir_milli=[7000, 10000],
            tower_hp=[[2400, 1400, 1400], [2400, 0, 0]],
        ),
    )
    out.append(eng.save_state())
    # 4-6. Mid-battle states from a random rollout.
    parser = TileActionParser()
    parser.bind(eng)
    eng.reset(4, MatchSetup(decks=[MIXED, MIXED]))
    rng = np.random.default_rng(4)
    for i in range(300):
        s_ = eng.state()
        cmds = []
        for team in (BLUE, RED):
            legal = np.flatnonzero(parser.action_mask(s_, team))[1:]
            if legal.size and rng.random() < 0.3:
                cmds.append(parser.parse(int(rng.choice(legal)), s_, team))
        eng.step(cmds, 5)
        if i % 100 == 99:
            out.append(eng.save_state())
    return out


PROBES = _probe_states()


def _all_disagreements(parser_cls, probes=PROBES):
    eng = MockEngine()
    parser = parser_cls()
    parser.bind(eng)
    found = []
    legal_total = 0
    for blob in probes:
        eng.load_state(blob)
        st = eng.state()
        for team in (BLUE, RED):
            legal_total += int(parser.action_mask(st, team).sum()) - 1
            found += mask_disagreements(eng, parser, st, team)
    return found, legal_total


@pytest.mark.parametrize("parser_cls", [TileActionParser, HalfTileActionParser])
def test_mask_equals_engine_legality_exhaustively(parser_cls):
    found, legal_total = _all_disagreements(parser_cls)
    assert found == []
    assert legal_total > 1000  # the probes are not all "nothing is legal"


# Plants: each breaks ONE rule in the mask only. The agreement check must see it.


def test_plant_mask_lets_troops_into_the_river(monkeypatch):
    # History: the first version of this plant only dropped the water term, and it
    # did NOT land -- the river rows are outside troop territory, so with the
    # current rules the water term is shadowed and cannot fail on its own. The
    # defect class worth guarding is "troops can be dropped in the river", which
    # needs territory to reach into the band as well.
    orig = PlacementOracle.cell_grid

    def into_river(self, state, team, placement):
        g = orig(self, state, team, placement)
        if placement != Placement.SPELL:
            a = self.arena
            lo = a.water_half_rows[0]
            band = np.zeros_like(g)
            if team == BLUE:
                band[lo : lo + 2, :] = True
            else:
                band[a.hy - lo - 2 : a.hy - lo, :] = True
            g = g | (band & ~self.nodeploy)
        return g

    monkeypatch.setattr(PlacementOracle, "cell_grid", into_river)
    found, _ = _all_disagreements(TileActionParser, PROBES[:1])
    statuses = {DeployStatus(st) for _, _, st in found}
    assert DeployStatus.WATER in statuses, statuses


def shrink_king_rects_toward_the_river(oracle: PlacementOracle, half_cells: int) -> None:
    """PLANT, mask only: move each king rect's river-facing edge ``half_cells`` half-
    cells back toward its king, so the mask opens ground the engine still refuses."""
    d = half_cells * oracle.arena.half_size
    before = [list(r) for r in oracle.tower_rects]
    for owner in (BLUE, RED):
        x0, y0, x1, y1 = oracle.tower_rects[owner][0]
        oracle.tower_rects[owner][0] = (
            (x0, y0 + d, x1, y1) if owner == RED else (x0, y0, x1, y1 - d)
        )
    assert oracle.tower_rects != before, "plant did not land: rects unchanged"


@pytest.mark.parametrize(
    ("parser_cls", "half_cells", "should_land"),
    [
        # A half-tile error cannot show at tile resolution: the shrunk edge lands
        # exactly ON the next tile centre, and the rect is closed. Measured, not
        # assumed: this row is the check that the plant's reach is understood.
        (TileActionParser, 1, False),
        (HalfTileActionParser, 1, True),
        (TileActionParser, 2, True),
    ],
)
def test_plant_mask_king_rect_shrunk(parser_cls, half_cells, should_land):
    """A king rect a half-tile too small in the mask must be caught: that rect is
    what bounds the ground opened by a fallen princess."""
    eng = MockEngine()  # the engine keeps the true rule
    parser = parser_cls()
    parser.bind(eng)
    # Baseline green on exactly these probes first.
    for blob in PROBES[1:3]:
        eng.load_state(blob)
        for team in (BLUE, RED):
            assert mask_disagreements(eng, parser, eng.state(), team) == []
    shrink_king_rects_toward_the_river(parser.oracle, half_cells)
    found = []
    for blob in PROBES[1:3]:
        eng.load_state(blob)
        for team in (BLUE, RED):
            found += mask_disagreements(eng, parser, eng.state(), team)
    assert bool(found) == should_land
    assert all(st == DeployStatus.OUT_OF_TERRITORY for _, _, st in found)


def test_goblin_barrel_is_refused_only_on_water():
    """SPELL_NOT_ON_WATER on the opening board: legal on the enemy king block, on
    no-deploy cells and on a bridge; refused on every tile that touches water."""
    eng = MockEngine()
    p = TileActionParser()
    p.bind(eng)
    assert eng.cards()[15].name == "GoblinBarrel"
    assert eng.cards()[15].placement == Placement.SPELL_NOT_ON_WATER
    barrel_first = [15, 0, 3, 10, 14, 11, 7, 9]  # ShuffleMode.NONE: slot 0 is the barrel
    eng.reset(
        0,
        MatchSetup(decks=[barrel_first, MIXED], elixir_milli=[10000, 0], shuffle=ShuffleMode.NONE),
    )
    st = eng.state()
    assert st.players[BLUE].hand[0] == 15
    per = 18 * 32
    grid = p.action_mask(st, BLUE)[1 : 1 + per].reshape(32, 18)
    water = p.oracle.water.reshape(32, 2, 18, 2).any(axis=(1, 3))
    assert water.sum() > 0
    assert np.array_equal(grid.astype(bool), ~water)
    assert grid[29, 9] == 1  # the enemy king block (no-deploy for troops)


def test_plant_mask_offers_the_river_to_a_goblin_barrel(monkeypatch):
    """Plant: the mask treats SPELL_NOT_ON_WATER as SPELL (anywhere) -- the shipped
    RustEngine defect before 2026-09-13 in its mildest form. The exhaustive test
    must go red on WATER verdicts only."""
    assert _all_disagreements(TileActionParser)[0] == [], "baseline must be green"
    orig = PlacementOracle.cell_grid

    def anywhere(self, state, team, placement):
        if placement == Placement.SPELL_NOT_ON_WATER:
            placement = Placement.SPELL
        return orig(self, state, team, placement)

    monkeypatch.setattr(PlacementOracle, "cell_grid", anywhere)
    assert PlacementOracle.cell_grid is anywhere, "plant did not land"
    found, _ = _all_disagreements(TileActionParser)
    assert found, "PLANT DID NOT LAND: a barrel mask open on the river passed"
    assert {(m, DeployStatus(s).name) for _, m, s in found} == {(1, "WATER")}


def test_plant_mask_ignores_building_footprints(monkeypatch):
    orig = PlacementOracle.point_grid

    def no_footprints(self, state, team, card, pitch_div):
        stripped = msgspec.structs.replace(state, entities=[])
        return orig(self, stripped, team, card, pitch_div)

    monkeypatch.setattr(PlacementOracle, "point_grid", no_footprints)
    found, _ = _all_disagreements(TileActionParser, PROBES[1:2])
    assert found
    assert all(st == DeployStatus.OCCUPIED for _, _, st in found)


def test_unmasked_illegal_action_is_rejected_and_costs_nothing():
    env = ClashParallelEnv(
        state_setter=DefaultStateSetter(decks=[MIXED, MIXED], shuffle=ShuffleMode.NONE)
    )
    obs, _ = env.reset(seed=0)
    parser = env.action_parser
    a = env.engine.arena()
    # Knight (slot 0) on a water tile, own frame tile (8, 15): mask says no.
    act = parser.encode(0, 8, 15)
    assert obs["blue"]["action_mask"][act] == 0
    # ...and on the enemy half.
    far = parser.encode(0, 8, 25)
    assert obs["blue"]["action_mask"][far] == 0
    before = env.battle_state.players[BLUE]
    _, _, _, _, info = env.step({"blue": act, "red": NOOP})
    assert info["blue"]["deploy_status"] == DeployStatus.WATER
    after = env.battle_state.players[BLUE]
    assert after.hand == before.hand
    assert after.elixir_milli >= before.elixir_milli
    _, _, _, _, info = env.step({"blue": far, "red": NOOP})
    assert info["blue"]["deploy_status"] == DeployStatus.OUT_OF_TERRITORY
    del a


def test_mask_marks_unaffordable_cards_and_keeps_noop():
    eng = MockEngine()
    p = TileActionParser()
    p.bind(eng)
    eng.reset(0, MatchSetup(decks=[MIXED, MIXED], shuffle=ShuffleMode.NONE, elixir_milli=[2999, 0]))
    st = eng.state()
    m = p.action_mask(st, BLUE)
    per = 18 * 32
    # hand = Knight(3) Giant(5) Cannon(3) Log(2): only Log is affordable at 2.999
    assert m[NOOP] == 1
    assert m[1 : 1 + 3 * per].sum() == 0
    assert m[1 + 3 * per :].sum() > 0
    assert p.action_mask(st, RED).sum() == 1  # zero elixir: only no-op


def test_spells_are_legal_everywhere_troops_only_in_territory():
    eng = MockEngine()
    p = TileActionParser()
    p.bind(eng)
    eng.reset(
        0,
        MatchSetup(
            decks=[[11, 0, 1, 2, 3, 4, 5, 6], MIXED],
            shuffle=ShuffleMode.NONE,
            elixir_milli=[10000, 0],
        ),
    )
    st = eng.state()
    m = p.action_mask(st, BLUE).reshape(-1)
    per = 18 * 32
    fireball = m[1 : 1 + per].reshape(32, 18)
    knight = m[1 + per : 1 + 2 * per].reshape(32, 18)
    assert fireball.all()
    assert knight[16:].sum() == 0  # nothing past the river (no pocket yet)
    assert knight[:15].sum() > 0


def test_pocket_opens_on_the_correct_side_after_a_princess_falls():
    eng = MockEngine()
    p = TileActionParser()
    p.bind(eng)
    # Red's own-LEFT princess stands on Blue's RIGHT (engine x=14.5).
    eng.reset(
        0,
        MatchSetup(
            decks=[MIXED, MIXED],
            shuffle=ShuffleMode.NONE,
            elixir_milli=[10000, 10000],
            tower_hp=[[2400, 1400, 1400], [2400, 0, 1400]],
        ),
    )
    st = eng.state()
    knight = p.action_mask(st, BLUE)[1 : 1 + 18 * 32].reshape(32, 18)
    enemy_half = knight[17:]
    assert enemy_half[:, 9:].sum() > 0, "pocket should open on Blue's right"
    assert enemy_half[:, :9].sum() == 0, "and not on Blue's left"
    # The opened ground is the far bank (y = 17) up to the enemy king's NoDeploySize
    # rect (y = 21): tile rows 17..20, 4 tiles = 8 half-rows.
    open_rows = [ty for ty in range(32) if knight[ty, 9:].any() and ty >= 17]
    assert open_rows == [17, 18, 19, 20], open_rows
    half = HalfTileActionParser()
    half.bind(eng)
    hk = half.action_mask(st, BLUE)[1 : 1 + 36 * 64].reshape(64, 36)
    assert [hy for hy in range(34, 64) if hk[hy, 18:].any()] == list(range(34, 42))
    # Buildings never get the pocket.
    cannon = p.action_mask(st, BLUE)[1 + 2 * 18 * 32 : 1 + 3 * 18 * 32].reshape(32, 18)
    assert cannon[17:].sum() == 0


def test_action_masks_method_matches_obs_for_maskable_ppo():
    env = ClashParallelEnv()
    obs, info = env.reset(seed=0)
    for agent in ("blue", "red"):
        m = env.action_masks(agent)
        assert m.dtype == np.bool_
        assert m.shape == (env.action_space(agent).n,)
        assert np.array_equal(m, obs[agent]["action_mask"].astype(bool))
        assert np.array_equal(info[agent]["action_mask"], obs[agent]["action_mask"])
