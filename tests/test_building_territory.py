"""placement.BUILDING_TERRITORY (royalesim 0.1.54): where a building's tap may go.

``own_half``: on its own half only. ``enemy_tower_no_deploy_rects`` (clients 16.402 and 16.635):
the troops' rule with the river band closed under either territory model, so once an enemy
princess falls her pocket past the far bank takes a building. The engine judges the building's
BOX against the alive enemy rects as well (arena.rs ``box_zone_rects``); the mask asks the engine
where a building lands for every box rule (``GridActionParser.buildable``).

The mask's own part is the TAP's rule (``PlacementOracle.cell_grid`` and the rect point rule). On
a board of towers alone the engine relocates any box that does not fit, so the tap's rule is the
engine's verdict there, and it is held to it point by point under each arm, without the engine's
landing query in between. MockEngine has no box, keeps its buildings on their own half and says
so; tests/_arms.py pins that arm wherever the two engines are compared.

SKIPS
    The engine tests skip without the engine, and on an engine whose ledger has no
    placement.BUILDING_TERRITORY (before royalesim 0.1.54): SKIPPED, NOT PASSED.
"""

from __future__ import annotations

import json

import msgspec
import numpy as np
import pytest

from royalegym.action import PlacementOracle
from royalegym.mock_engine import MockEngine
from royalegym.protocol import (
    BLUE,
    BUILDING_TERRITORY,
    BUILDINGS_IN_TROOP_RECTS,
    CLOSED_RIVER_BAND,
    OPEN_BRIDGE,
    OWN_HALF_BUILDINGS,
    RED,
    Calibration,
    DeployCommand,
    DeployRules,
    DeployStatus,
    MatchSetup,
    ShuffleMode,
    TowerSlot,
    default_calibration,
    to_own,
)
from royalegym.rust_engine import CORE_IMPORT_ERROR, core_available

KEY = "placement.BUILDING_TERRITORY"
DECK_NAMES = ["Cannon", "Knight", "Minions", "Giant", "Archer", "Musketeer", "Fireball", "Arrows"]
FULL = [2400, 1400, 1400]
BOARDS = {
    "all_up": [FULL, FULL],
    "red_left_down": [FULL, [2400, 0, 1400]],
    "blue_right_down": [[2400, 1400, 0], FULL],
    "both_sides_one_down": [[2400, 0, 1400], [2400, 1400, 0]],
}


def _ledger(arm: str | None = None) -> Calibration:
    """The default ledger with placement.BUILDING_TERRITORY at ``arm``, or without the key."""
    raw = json.loads(json.dumps(default_calibration().raw))
    placement = raw.setdefault("placement", {})
    placement.pop("BUILDING_TERRITORY", None)
    if arm is not None:
        placement["BUILDING_TERRITORY"] = {"value": arm}
    return Calibration(raw)


def test_the_arm_is_read_from_the_ledger_and_an_unknown_one_refused():
    assert DeployRules.load(_ledger()).building_territory == OWN_HALF_BUILDINGS, (
        "an engine without the key ran the own half"
    )
    for arm in BUILDING_TERRITORY:
        rules = DeployRules.load(_ledger(arm))
        assert rules.building_territory == arm
        assert rules.buildings_take_troop_rects == (arm == BUILDINGS_IN_TROOP_RECTS)
    with pytest.raises(NotImplementedError, match="BUILDING_TERRITORY"):
        DeployRules.load(_ledger("anywhere"))
    assert MockEngine().rules().building_territory == OWN_HALF_BUILDINGS, (
        "MockEngine has no box to judge against the rects, and says it runs the own half"
    )


# --- the mask's tap rule, on MockEngine's arena and towers --------------------------------------


def _mock_state(hp):
    eng = MockEngine()
    ids = {c.name: i for i, c in enumerate(eng.cards())}
    deck = [ids[n] for n in ("Cannon", "Knight", "Giant", "Archer", "Musketeer", "Minions",
                             "Fireball", "Arrows")]
    eng.reset(4, MatchSetup(decks=[deck, deck], shuffle=ShuffleMode.NONE,
                            elixir_milli=[10000, 10000], tower_hp=hp))
    return eng, eng.state()


def _oracle(
    eng, building: str, territory: str | None = None, relocate: bool = False
) -> PlacementOracle:
    over = {"building_territory": building}
    if territory is not None:
        over["territory_model"] = territory
    if relocate:
        over["illegal_building_tap"] = "relocate_first_fitting_ring"
    return PlacementOracle(eng.arena(), msgspec.structs.replace(eng.rules(), **over), eng.cards())


def _card(eng, name):
    return next(c for c in eng.cards() if c.name == name)


def _own_rows(eng, team, grid, pitch_div):
    """Own-frame (row, col) of each legal point of an engine-frame ``grid``."""
    oracle = PlacementOracle(eng.arena(), eng.rules(), eng.cards())
    px, py = oracle.points(pitch_div)
    a = eng.arena()
    out = set()
    for y, x in zip(*np.nonzero(grid), strict=True):
        ox, oy = to_own(a, team, int(px[y, x]), int(py[y, x]))
        out.add((oy, ox))
    return out


@pytest.mark.parametrize("territory", [CLOSED_RIVER_BAND, OPEN_BRIDGE])
def test_the_two_arms_agree_while_every_enemy_tower_stands(territory):
    """With every enemy tower up, the enemy rects cover the whole enemy side, so the rects arm
    offers exactly the own half: the arms part only once a princess falls."""
    eng, st = _mock_state(BOARDS["all_up"])
    cannon = _card(eng, "Cannon")
    own = _oracle(eng, OWN_HALF_BUILDINGS, territory)
    rects = _oracle(eng, BUILDINGS_IN_TROOP_RECTS, territory)
    for team in (BLUE, RED):
        for pitch in (1, 2):
            want = own.point_grid(st, team, cannon, pitch)
            assert want.any()
            assert np.array_equal(rects.point_grid(st, team, cannon, pitch), want), (team, pitch)


@pytest.mark.parametrize("board", ["red_left_down", "blue_right_down"])
def test_a_fallen_princess_opens_her_pocket_to_a_building_under_the_rects_arm(board):
    """The rects arm adds exactly ground past the far bank, in the fallen lane, where a Knight
    may go as well (the troops' rule, river band closed); the own-half arm adds nothing."""
    eng, st = _mock_state(BOARDS[board])
    a = eng.arena()
    cannon, knight = _card(eng, "Cannon"), _card(eng, "Knight")
    team = BLUE if board == "red_left_down" else RED
    fallen = TowerSlot.LEFT if board == "red_left_down" else TowerSlot.RIGHT
    far_bank = (a.water_half_rows[1] + 1) * a.half_size
    troops = _oracle(eng, OWN_HALF_BUILDINGS, CLOSED_RIVER_BAND)
    for pitch in (1, 2):
        own = _own_rows(eng, team, _oracle(eng, OWN_HALF_BUILDINGS).point_grid(
            st, team, cannon, pitch), pitch)
        rects = _own_rows(eng, team, _oracle(eng, BUILDINGS_IN_TROOP_RECTS).point_grid(
            st, team, cannon, pitch), pitch)
        knight_cells = _own_rows(eng, team, troops.point_grid(st, team, knight, pitch), pitch)
        assert own <= rects, "the rects arm took ground the own half gives"
        pocket = rects - own
        assert len(pocket) >= 20 * pitch * pitch, f"vacuous: {len(pocket)} pocket points"
        assert all(oy > far_bank for oy, _ in pocket), "a building offered short of the far bank"
        # The enemy's own-LEFT princess stands on my right half of the arena, and so on.
        mine = {TowerSlot.LEFT: lambda ox: ox > a.width // 2,
                TowerSlot.RIGHT: lambda ox: ox < a.width // 2}[fallen]
        assert all(mine(ox) for _, ox in pocket), "a pocket opened in the standing lane"
        assert pocket <= knight_cells, "a building offered where the troops' rule refuses"


def test_the_river_band_stays_closed_to_a_building_under_the_open_bridge_model():
    """Under the open-bridge territory model a fallen lane's bridge takes a troop; the engine's
    building rule keeps the band closed (state.rs deploy_rule: open_bridge false)."""
    eng, st = _mock_state(BOARDS["red_left_down"])
    a = eng.arena()
    lo, hi = a.water_half_rows
    band = range(lo * a.half_size, (hi + 1) * a.half_size + 1)
    knight, cannon = _card(eng, "Knight"), _card(eng, "Cannon")
    open_rects = _oracle(eng, BUILDINGS_IN_TROOP_RECTS, OPEN_BRIDGE)
    bridge = [oy for oy, _ in _own_rows(eng, BLUE, open_rects.point_grid(st, BLUE, knight, 2), 2)
              if oy in band]
    assert bridge, "vacuous: the open bridge took no troop"
    cells = _own_rows(eng, BLUE, open_rects.point_grid(st, BLUE, cannon, 2), 2)
    assert not [oy for oy, _ in cells if oy in band], "a building offered on the river band"


def test_one_oracle_across_tower_states_serves_each_its_own_grid():
    """The memo keys a building's grid on the alive enemy rects under the rects arm: the grid
    of a board with every tower up must not answer for one with a princess down. Under the
    relocating arm, the engine's: there the bodies are no rule for a building, so a fallen
    tower does not change the memo key through them."""
    eng, up = _mock_state(BOARDS["all_up"])
    _, down = _mock_state(BOARDS["red_left_down"])
    cannon = _card(eng, "Cannon")
    shared = _oracle(eng, BUILDINGS_IN_TROOP_RECTS, relocate=True)
    assert not shared._bodies_block(cannon.placement)
    shared.point_grid(up, BLUE, cannon, 1)
    fresh = _oracle(eng, BUILDINGS_IN_TROOP_RECTS, relocate=True).point_grid(down, BLUE, cannon, 1)
    assert not np.array_equal(fresh, shared.point_grid(up, BLUE, cannon, 1)), "vacuous"
    assert np.array_equal(shared.point_grid(down, BLUE, cannon, 1), fresh)


# --- the engine ---------------------------------------------------------------------------------


def _ledger_has_key() -> bool:
    if not core_available():
        return False
    from _arms import shipped_placement

    return "BUILDING_TERRITORY" in shipped_placement()


needs_key = pytest.mark.skipif(
    not _ledger_has_key(),
    reason=str(CORE_IMPORT_ERROR) if not core_available() else
    "SKIPPED, NOT PASSED: this engine's ledger has no placement.BUILDING_TERRITORY "
    "(royalesim 0.1.54 on)",
)


def _engine(arm: str):
    from _arms import pinned
    from royalegym.rust_engine import RustEngine

    engine = RustEngine(calibration_overrides=pinned({KEY: arm}))
    assert engine.rules().building_territory == arm, "the rules did not follow the arm"
    return engine


@needs_key
@pytest.mark.parametrize("board", list(BOARDS))
@pytest.mark.parametrize("arm", BUILDING_TERRITORY)
def test_the_taps_rule_is_the_engines_verdict_on_a_board_of_towers(arm, board):
    rust = _engine(arm)
    ids = {c.name: i for i, c in enumerate(rust.cards())}
    deck = [ids[n] for n in DECK_NAMES]
    rust.reset(4, MatchSetup(decks=[deck, deck], shuffle=ShuffleMode.NONE,
                             elixir_milli=[10000, 10000], tower_hp=BOARDS[board],
                             start_tick=rust.rules().deploy_lockout_ticks))
    st = rust.state()
    oracle = PlacementOracle(rust.arena(), rust.rules(), rust.cards())
    assert oracle.buildings_relocate, "a refusing ledger: bodies would be a rule here too"
    a = rust.arena()
    far_bank = (a.water_half_rows[1] + 1) * a.half_size
    wrong, past_river = [], 0
    for team in (BLUE, RED):
        card = rust.cards()[st.players[team].hand[0]]
        assert card.name == "Cannon"
        for pitch in (1, 2):
            grid = oracle.point_grid(st, team, card, pitch)
            px, py = oracle.points(pitch)
            for (y, x), m in np.ndenumerate(grid):
                cx, cy = int(px[y, x]), int(py[y, x])
                status = rust.check_deploy(DeployCommand(team, 0, cx, cy))
                if bool(m) != (status == DeployStatus.OK):
                    wrong.append((team, pitch, cx, cy, int(m), DeployStatus(status).name))
                past_river += int(bool(m) and to_own(a, team, cx, cy)[1] > far_bank)
    assert not wrong, f"{len(wrong)} disagreements, first {wrong[:6]}"
    opened = arm == BUILDINGS_IN_TROOP_RECTS and board != "all_up"
    assert (past_river > 0) == opened, (arm, board, past_river)
