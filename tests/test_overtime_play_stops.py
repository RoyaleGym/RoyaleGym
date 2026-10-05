"""When a level overtime ends, play stops, the same on both engines (RoyaleSim 0.1.4's rule).

Measured on the client (6 of 6 live level overtimes, 16.402) and written into RoyaleSim and
MockEngine together: no play is accepted from t6000; t6000 and t6001 run as normal; at the
head of t6002 every unit and building leaves the board, with no death effects, and only the
crown towers stand, unchanged; the drain runs from t6067; full, equal towers drain once and the
match is a Draw at t6147. Each assertion below runs on both engines, so the two cannot drift.

SKIPS
    The Rust arm without the engine, or on an engine whose ledger does not select
    client_hp_drain (before RoyaleSim 0.1.4). Not a pass.
"""

from __future__ import annotations

import copy

import pytest

from royalegym.mock_engine import MockEngine
from royalegym.obs import TOWER_KINDS
from royalegym.protocol import (
    BLUE,
    RED,
    Calibration,
    DeployCommand,
    DeployStatus,
    MatchSetup,
    Winner,
    default_calibration,
)
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available

DRAIN = "client_hp_drain"
#: Cards both catalogues hold (the mock's 2018 table and the engine's), so one deck by name.
NAMES = ["Knight", "Archer", "Giant", "Minions", "Fireball", "Arrows", "Musketeer", "Valkyrie"]


def _mock():
    raw = copy.deepcopy(default_calibration().raw)
    raw["match"]["OVERTIME_TIEBREAK"]["value"] = DRAIN
    return MockEngine(Calibration(raw))


def _rust():
    if not core_available():
        pytest.skip(str(CORE_IMPORT_ERROR))
    rule = str(default_calibration().value("match.OVERTIME_TIEBREAK"))
    if rule != DRAIN:
        pytest.skip(f"SKIPPED, NOT PASSED: this engine's ledger selects {rule!r}, not {DRAIN!r}")
    return RustEngine()


@pytest.fixture(params=["mock", "rust"])
def make(request):
    return _mock if request.param == "mock" else _rust


def _board(eng, start_tick: int, elixir: int = 10000):
    ids = {c.name: c.card_id for c in eng.cards()}
    deck = [ids[n] for n in NAMES]
    eng.reset(5, MatchSetup(decks=[deck, deck], start_tick=start_tick, elixir_milli=[elixir] * 2))
    return eng


def _troop_on_own_side(eng, team: int):
    a = eng.arena()
    kinds = {c.card_id: c.card_kind for c in eng.cards()}
    hand = eng.state().players[team].hand
    ys = [a.height // 5, a.height // 4] if team == BLUE else [a.height - a.height // 5]
    for slot, card in enumerate(hand):
        if kinds.get(card) != "TROOP":
            continue
        for y in ys:
            for x in (a.width // 2, a.width // 3, 2 * a.width // 3):
                cmd = DeployCommand(team, slot, x, y)
                if eng.check_deploy(cmd) == DeployStatus.OK:
                    return cmd
    return None


def _units(s):
    return [e for e in s.entities if e.kind not in TOWER_KINDS]


def _towers(s):
    return sorted((e.uid, e.hp) for e in s.entities if e.kind in TOWER_KINDS)


def test_a_play_is_open_at_t5999_and_refused_from_t6000(make):
    assert _troop_on_own_side(_board(make(), 5999), BLUE) is not None
    eng = _board(make(), 6000)
    a = eng.arena()
    statuses = {
        eng.check_deploy(DeployCommand(BLUE, slot, a.width // 2, a.height // 5))
        for slot in range(4)
    }
    assert statuses == {DeployStatus.GAME_OVER}, statuses


def test_the_mask_offers_no_play_from_t6000(make):
    """The mask knows the freeze: at t6000 it agrees with the engine on every action of both
    seats (it offers only the no-op), and at t5999 it still offers plays. Before, it read only
    ``game_over``, which the drain sets later, so a policy obeying its mask had every play of
    the tiebreak refused GAME_OVER: in a training run with a command delay, about one battle in
    fifty ended in a tied level overtime, and each such play stopped the run."""
    from royalegym.action import TileActionParser, mask_disagreements

    eng = _board(make(), 5999)
    parser = TileActionParser()
    parser.bind(eng)
    assert parser.action_mask(eng.state(), BLUE)[1:].any(), "t5999: no play offered"
    eng = _board(make(), 6000)
    parser.bind(eng)
    s = eng.state()
    for team in (BLUE, RED):
        assert mask_disagreements(eng, parser, s, team) == [], team
        assert parser.action_mask(s, team).sum() == 1, "only the no-op"


def test_the_board_clears_at_the_head_of_t6002_and_the_towers_stand_unchanged(make):
    eng = _board(make(), 5990)
    for team in (BLUE, RED):
        cmd = _troop_on_own_side(eng, team)
        assert cmd is not None, team
        assert eng.step([cmd], 1)[0].status == DeployStatus.OK
    eng.step([], 6002 - eng.state().tick)
    s = eng.state()
    assert (s.tick, s.game_over) == (6002, False)
    assert _units(s), "the units placed before t6000 are gone before t6002"
    towers = _towers(s)
    eng.step([], 1)
    s = eng.state()
    assert _units(s) == []
    assert _towers(s) == towers
    assert s.spells == []


def test_full_equal_towers_drain_once_and_draw_at_t6147(make):
    eng = _board(make(), 6000)
    start = _towers(eng.state())
    while not eng.state().game_over and eng.state().tick < 6300:
        eng.step([], 1)
    s = eng.state()
    assert (s.game_over, s.winner, s.tick) == (True, Winner.DRAW, 6148)
    # One drain tick, the same step on every tower (each started full).
    lost = {hp - after for (_, hp), (_, after) in zip(start, _towers(s), strict=True)}
    assert len(lost) == 1, lost
