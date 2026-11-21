"""placement.SPELL_AS_DEPLOY_NO_DEPLOY_CELLS (royalesim 0.1.55): may a rolling spell's tap touch a
NO_DEPLOY cell?

``refuse``: a tap touching a NO_DEPLOY cell is refused for every card. ``client16635_can_place_on_
buildings`` (client 16.635): a ROLLING card, placement code 3 (a spell placed as a deploy that may
stand on buildings: the Log, the Barbarian Barrel, the Royal Delivery and their hero forms), skips
that test, while out of the arena, water and its territory (the river band, the alive enemy crown
towers' rects) still refuse it. A spell with a troop's placement (the Heal) and every troop keep it.

The mask follows the arm in its cell rule (``PlacementOracle.cell_grid``) and in the zone verdict
the spell clamp reads (``zone_verdict``, ``clamped_tap``); RustEngine follows an override of it
(``OVERRIDE_FOLLOWED``). MockEngine keeps the test and says so; tests/_arms.py pins that arm
wherever the two engines are compared.

SKIPS
    The engine tests skip without the engine, and on an engine whose ledger has no
    placement.SPELL_AS_DEPLOY_NO_DEPLOY_CELLS (before royalesim 0.1.55): SKIPPED, NOT PASSED.
"""

from __future__ import annotations

import json

import msgspec
import numpy as np
import pytest

from royalegym.action import PlacementOracle, TileActionParser, mask_disagreements
from royalegym.mock_engine import MockEngine
from royalegym.protocol import (
    BLUE,
    NO_DEPLOY_REFUSED,
    RED,
    ROLLING_SKIPS_NO_DEPLOY,
    SPELL_AS_DEPLOY_NO_DEPLOY_CELLS,
    Calibration,
    DeployCommand,
    DeployRules,
    DeployStatus,
    MatchSetup,
    Placement,
    ShuffleMode,
    TowerSlot,
    default_calibration,
)
from royalegym.rust_engine import CORE_IMPORT_ERROR, core_available

KEY = "placement.SPELL_AS_DEPLOY_NO_DEPLOY_CELLS"
FULL = [2400, 1400, 1400]
BOARDS = {
    "all_up": [FULL, FULL],
    "red_left_down": [FULL, [2400, 0, 1400]],
    "both_sides_one_down": [[2400, 0, 1400], [2400, 1400, 0]],
}


def _ledger(arm: str | None = None) -> Calibration:
    """The default ledger with placement.SPELL_AS_DEPLOY_NO_DEPLOY_CELLS at ``arm``, or without
    it."""
    raw = json.loads(json.dumps(default_calibration().raw))
    placement = raw.setdefault("placement", {})
    placement.pop("SPELL_AS_DEPLOY_NO_DEPLOY_CELLS", None)
    if arm is not None:
        placement["SPELL_AS_DEPLOY_NO_DEPLOY_CELLS"] = {"value": arm}
    return Calibration(raw)


def test_the_arm_is_read_from_the_ledger_and_an_unknown_one_refused():
    assert DeployRules.load(_ledger()).spell_as_deploy_no_deploy_cells == NO_DEPLOY_REFUSED, (
        "an engine without the key refused no-deploy ground to every card"
    )
    for arm in SPELL_AS_DEPLOY_NO_DEPLOY_CELLS:
        rules = DeployRules.load(_ledger(arm))
        assert rules.spell_as_deploy_no_deploy_cells == arm
        assert rules.rolling_skips_no_deploy == (arm == ROLLING_SKIPS_NO_DEPLOY)
    with pytest.raises(NotImplementedError, match="SPELL_AS_DEPLOY_NO_DEPLOY_CELLS"):
        DeployRules.load(_ledger("everywhere"))
    assert MockEngine().rules().spell_as_deploy_no_deploy_cells == NO_DEPLOY_REFUSED, (
        "MockEngine's zone refuses no-deploy ground to every card, and says so"
    )


# --- the mask's rule, on MockEngine's arena and towers ------------------------------------------


def _mock_state(hp):
    eng = MockEngine()
    ids = {c.name: i for i, c in enumerate(eng.cards())}
    deck = [ids[n] for n in ("Log", "Knight", "Cannon", "Archer", "Giant", "Minions", "Fireball",
                             "Arrows")]
    eng.reset(4, MatchSetup(decks=[deck, deck], shuffle=ShuffleMode.NONE,
                            elixir_milli=[10000, 10000], tower_hp=hp))
    return eng, eng.state()


def _oracle(eng, arm: str) -> PlacementOracle:
    rules = msgspec.structs.replace(eng.rules(), spell_as_deploy_no_deploy_cells=arm)
    return PlacementOracle(eng.arena(), rules, eng.cards())


def _card(eng, name):
    return next(c for c in eng.cards() if c.name == name)


def _touch(oracle, cells, px, py):
    """Whether each point touches a half-cell set in ``cells``."""
    return ~oracle._touches_only(~cells, px, py)


@pytest.mark.parametrize("board", list(BOARDS))
def test_the_open_arm_adds_exactly_no_deploy_ground_to_a_rolling_spell(board):
    eng, st = _mock_state(BOARDS[board])
    log = _card(eng, "Log")
    assert log.placement == Placement.ROLLING
    shut, open_ = _oracle(eng, NO_DEPLOY_REFUSED), _oracle(eng, ROLLING_SKIPS_NO_DEPLOY)
    added_total = 0
    for team in (BLUE, RED):
        for pitch in (1, 2):
            px, py = shut.points(pitch)
            before = shut.point_grid(st, team, log, pitch)
            after = open_.point_grid(st, team, log, pitch)
            assert not (before & ~after).any(), "the open arm took ground the refusing arm gives"
            added = after & ~before
            added_total += int(added.sum())
            assert _touch(shut, shut.nodeploy, px, py)[added].all(), (
                "a point the open arm adds that touches no NO_DEPLOY cell"
            )
            assert not _touch(shut, shut.water, px, py)[added].any(), "a point on water"
            assert not _touch(shut, shut.river_band, px, py)[added].any() or (
                not shut.rules.river_band_closed_to_troops
            ), "a point on the closed river band"
            for x0, y0, x1, y1 in shut.enemy_rects(st, team):
                inside = (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
                assert not (added & inside).any(), "a point in an alive enemy tower's rect"
            for name in ("Knight", "Cannon", "Fireball"):
                card = _card(eng, name)
                assert np.array_equal(open_.point_grid(st, team, card, pitch),
                                      shut.point_grid(st, team, card, pitch)), (name, team, pitch)
    assert added_total >= 40, f"vacuous: the open arm added {added_total} points"


def test_one_oracle_keys_its_memo_on_the_arm():
    """Tests swap an oracle's rules in place (tests/test_clamped_taps.py ``_with_arms``): the memo
    must not serve the refusing arm's Log grid once the open arm is in."""
    eng, st = _mock_state(BOARDS["all_up"])
    log = _card(eng, "Log")
    oracle = _oracle(eng, NO_DEPLOY_REFUSED)
    shut = oracle.point_grid(st, BLUE, log, 2).copy()
    oracle.rules = msgspec.structs.replace(oracle.rules,
                                           spell_as_deploy_no_deploy_cells=ROLLING_SKIPS_NO_DEPLOY)
    fresh = _oracle(eng, ROLLING_SKIPS_NO_DEPLOY).point_grid(st, BLUE, log, 2)
    assert not np.array_equal(fresh, shut), "vacuous"
    assert np.array_equal(oracle.point_grid(st, BLUE, log, 2), fresh)


def test_the_zone_verdict_reads_the_arm_for_a_rolling_spell_only():
    """The spell clamp (``clamped_tap``) moves a tap refused OUT_OF_TERRITORY or on WATER and keeps
    a NO_DEPLOY refusal. Inside an alive enemy king's block (NO_DEPLOY and its rect) a Log tap is
    therefore kept under the refusing arm and moved under the open one; on the own king's block it
    is refused, then legal."""
    eng, st = _mock_state(BOARDS["all_up"])
    a = eng.arena()
    shut, open_ = _oracle(eng, NO_DEPLOY_REFUSED), _oracle(eng, ROLLING_SKIPS_NO_DEPLOY)
    king = {team: shut.tower_rects[team][TowerSlot.KING] for team in (BLUE, RED)}
    own_x, own_y = (king[BLUE][0] + king[BLUE][2]) // 2, (king[BLUE][1] + king[BLUE][3]) // 2
    enemy_x, enemy_y = (king[RED][0] + king[RED][2]) // 2, (king[RED][1] + king[RED][3]) // 2
    nd = _touch(shut, shut.nodeploy, np.asarray([own_x, enemy_x]), np.asarray([own_y, enemy_y]))
    assert nd.all(), "vacuous: a king's centre touches no NO_DEPLOY cell on this arena"
    assert 0 < own_x < a.width
    assert 0 < own_y < a.height
    verdict = {}
    for name, oracle in (("shut", shut), ("open", open_)):
        no_deploy = oracle._no_deploy_rule(Placement.ROLLING)
        got = oracle.zone_verdict(st, BLUE, np.asarray([own_x, enemy_x]),
                                  np.asarray([own_y, enemy_y]), troop_laws=False,
                                  no_deploy=no_deploy)
        verdict[name] = [DeployStatus(int(v)) for v in got]
    assert verdict["shut"] == [DeployStatus.NO_DEPLOY, DeployStatus.NO_DEPLOY]
    assert verdict["open"] == [DeployStatus.OK, DeployStatus.OUT_OF_TERRITORY]
    for placement in (Placement.TROOP, Placement.BUILDING, Placement.SPELL):
        assert open_._no_deploy_rule(placement), placement


# --- the engine ---------------------------------------------------------------------------------


def _engine_reads_the_key() -> bool:
    """royalesim 0.1.55 on: the first core whose ledger has
    placement.SPELL_AS_DEPLOY_NO_DEPLOY_CELLS."""
    if not core_available():
        return False
    import royalesim
    from _arms import shipped_placement

    have = tuple(int(p) for p in royalesim.__version__.split(".")[:3])
    return have >= (0, 1, 55) and "SPELL_AS_DEPLOY_NO_DEPLOY_CELLS" in shipped_placement()


needs_key = pytest.mark.skipif(
    not _engine_reads_the_key(),
    reason=str(CORE_IMPORT_ERROR) if not core_available() else
    "SKIPPED, NOT PASSED: this royalesim does not read placement.SPELL_AS_DEPLOY_NO_DEPLOY_CELLS "
    "(royalesim 0.1.55 on)",
)

#: Dealt in order: the hand is the three rolling spells and the Heal.
SPELL_DECK = ["Log", "BarbLog", "RoyalDelivery", "Heal", "Knight", "Cannon", "Fireball", "Arrows"]


def _engine(arm: str, clamps: bool):
    from _arms import CLAMPS_OFF, pinned
    from royalegym.rust_engine import RustEngine

    over = {KEY: arm} if clamps else {KEY: arm, **CLAMPS_OFF}
    engine = RustEngine(calibration_overrides=pinned(over))
    assert engine.rules().spell_as_deploy_no_deploy_cells == arm, "the rules did not follow the arm"
    return engine


def _reset(rust, board):
    ids = {c.name: i for i, c in enumerate(rust.cards())}
    deck = [ids[n] for n in SPELL_DECK]
    rust.reset(4, MatchSetup(decks=[deck, deck], shuffle=ShuffleMode.NONE,
                             elixir_milli=[10000, 10000], tower_hp=BOARDS[board],
                             start_tick=rust.rules().deploy_lockout_ticks))
    return rust.state()


@needs_key
@pytest.mark.parametrize("board", list(BOARDS))
@pytest.mark.parametrize("arm", SPELL_AS_DEPLOY_NO_DEPLOY_CELLS)
def test_the_mask_is_the_engines_verdict_point_by_point(arm, board):
    """Clamps off, so ``check_deploy`` is the zone's own verdict at every tile and half-tile centre,
    for each rolling spell and the Heal, both seats."""
    rust = _engine(arm, clamps=False)
    st = _reset(rust, board)
    oracle = PlacementOracle(rust.arena(), rust.rules(), rust.cards())
    wrong, opened = [], {}
    for team in (BLUE, RED):
        for slot, card_id in enumerate(st.players[team].hand):
            card = rust.cards()[card_id]
            assert card.name == SPELL_DECK[slot], (team, slot, card.name)
            for pitch in (1, 2):
                grid = oracle.point_grid(st, team, card, pitch)
                px, py = oracle.points(pitch)
                on_nd = _touch(oracle, oracle.nodeploy, px, py)
                opened[card.name] = opened.get(card.name, 0) + int((grid & on_nd).sum())
                for (y, x), m in np.ndenumerate(grid):
                    cx, cy = int(px[y, x]), int(py[y, x])
                    status = rust.check_deploy(DeployCommand(team, slot, cx, cy))
                    if bool(m) != (status == DeployStatus.OK):
                        wrong.append((team, card.name, pitch, cx, cy, int(m),
                                      DeployStatus(status).name))
    assert not wrong, f"{len(wrong)} disagreements, first {wrong[:6]}"
    rolling = ("Log", "BarbLog", "RoyalDelivery")
    if arm == ROLLING_SKIPS_NO_DEPLOY:
        assert all(opened[n] > 0 for n in rolling), opened
    else:
        assert not any(opened[n] for n in rolling), opened
    assert opened["Heal"] == 0, "the Heal stood on no-deploy ground"


@needs_key
@pytest.mark.parametrize("team", [BLUE, RED])
@pytest.mark.parametrize("board", ["all_up", "red_left_down"])
@pytest.mark.parametrize("arm", SPELL_AS_DEPLOY_NO_DEPLOY_CELLS)
def test_the_whole_mask_agrees_with_the_engine_under_its_clamps(arm, board, team):
    """With the clamps the plain engine runs: an accepted tap the mask refuses must be one the
    spell clamp moves (``clamped_tap``), which under the open arm includes a Log tap inside an
    alive enemy king's block."""
    rust = _engine(arm, clamps=True)
    st = _reset(rust, board)
    parser = TileActionParser()
    parser.bind(rust)
    assert parser.oracle.rules.spell_as_deploy_no_deploy_cells == arm
    assert mask_disagreements(rust, parser, st, team) == []
