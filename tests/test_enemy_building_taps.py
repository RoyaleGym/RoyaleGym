"""placement.ENEMY_BUILDING_TAPS: a troop tapped on an ENEMY building's box is moved off it.

Under ``client15535_tile_box_ring`` (and the half-tile arm before it) the core moves a tap whose
one-tile box shares area with an alive ENEMY building's placement box, by the ring search, as
it moves one off an own building (state.rs relocate_off_own_crown_tower, which reads the key
only when a tower or building relocation arm is on). Measured on client 15.535.29 with an enemy
Hero Musketeer's turret. Before the key, such a tap was refused OCCUPIED.

The mask follows it as it follows the own-building move: the enemy boxes join the move zone
(``PlacementOracle.own_tower_zone``), and the engine is asked whether each moved tap lands
(``GridActionParser.moved_taps_that_land``). The arm is read from the ledger
(``DeployRules.enemy_building_taps``) and an override of it is followed.

The board puts each side's buildings in the OTHER side's half (a state mutator may), on the
tile grid, where a troop can tap them: a building the play path lays stands there.

SKIPS
    The battle tests need the engine and a ledger listing a moving arm. Not a pass without them.
"""

from __future__ import annotations

import copy

import msgspec
import pytest

from royalegym.action import (
    HalfTileActionParser,
    PlacementOracle,
    TileActionParser,
    mask_disagreements,
)
from royalegym.mock_engine import MockEngine
from royalegym.protocol import (
    BLUE,
    ENEMY_BUILDING_TAPS,
    RED,
    Calibration,
    DeployRules,
    DeployStatus,
    MatchSetup,
    ShuffleMode,
    SpawnSpec,
)
from royalegym.rust_engine import (
    CORE_IMPORT_ERROR,
    OVERRIDE_FOLLOWED,
    RustEngine,
    core_available,
)

KEY = "placement.ENEMY_BUILDING_TAPS"
MOVING = ("client15535_tile_box_ring", "client15535_half_tile_box_ring")
needs_engine = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
#: Dealt in order: a hand of a troop, the Miner, a spell placed as a troop and a pair.
DECK = ["Knight", "Miner", "Heal", "Archer", "Giant", "Minions", "Goblins", "Fireball"]
#: Own-frame tiles of the buildings each side gets in the OTHER side's half.
SITES = (("Cannon", 6, 10), ("Tesla", 12, 11), ("InfernoTower", 9, 7))


def _ledger_with(value):
    raw = copy.deepcopy(Calibration.load().raw)
    raw.setdefault("placement", {})
    if value is None:
        raw["placement"].pop("ENEMY_BUILDING_TAPS", None)
    else:
        raw["placement"]["ENEMY_BUILDING_TAPS"] = {
            "value": value, "candidates": list(ENEMY_BUILDING_TAPS)}
    return Calibration(raw)


def test_the_arm_is_read_from_the_ledger_and_an_unknown_one_refused():
    assert ("ignored", *reversed(MOVING)) == ENEMY_BUILDING_TAPS
    for value, want in ((None, "ignored"), *((a, a) for a in ENEMY_BUILDING_TAPS)):
        assert DeployRules.load(_ledger_with(value)).enemy_building_taps == want
    with pytest.raises(NotImplementedError, match="ENEMY_BUILDING_TAPS"):
        DeployRules.load(_ledger_with("stand_on_the_turret"))


def test_an_override_of_the_arm_is_followed():
    assert KEY in OVERRIDE_FOLLOWED


def test_mockengine_states_the_arm_it_runs():
    """MockEngine moves nothing off an enemy building, whatever its ledger ships; its rules,
    which the mask reads, say so."""
    for arm in ENEMY_BUILDING_TAPS:
        assert MockEngine(calibration=_ledger_with(arm)).rules().enemy_building_taps == "ignored"


def _arms():
    """(a moving arm the ledger lists, the arm it ships), or a skip."""
    entry = Calibration.load().raw.get("placement", {}).get("ENEMY_BUILDING_TAPS")
    if entry is None:
        pytest.skip(f"SKIPPED, NOT PASSED: this engine's ledger has no {KEY}")
    listed = [*entry.get("candidates", []), entry["value"]]
    moving = next((a for a in MOVING if a in listed), None)
    if moving is None:
        pytest.skip(f"SKIPPED, NOT PASSED: this engine's ledger lists no moving arm of {KEY}")
    return moving, entry["value"]


def _site(engine, team, name, tx, ty):
    """``team``'s ``name`` on own-frame tile (tx, ty) of the OTHER side's half: centred on the
    tile for an odd box, on its low corner for an even one, as the play path lays it."""
    card = next(c for c in engine.cards() if c.name == name)
    a, t = engine.arena(), engine.arena().subtile
    n = card.footprint_tiles
    assert n, f"{name} states no footprint"
    x, y = tx * t + (t // 2 if n % 2 else 0), ty * t + (t // 2 if n % 2 else 0)
    if team == BLUE:  # Blue's building in Red's half: Red's own frame, rotated
        x, y = a.width - x, a.height - y
    return SpawnSpec(team, card.card_id, x, y)


def _board(arm, parser_cls, down):
    shipped = _arms()[1]
    engine = RustEngine(calibration_overrides={KEY: arm} if arm != shipped else None)
    assert engine.rules().enemy_building_taps == arm
    ids = {c.name: c.card_id for c in engine.cards()}
    deck = [ids[n] for n in DECK]
    hp = [[2400, 0, 1400], [2400, 1400, 0]] if down else None
    spawns = [_site(engine, team, *s) for s in SITES for team in (BLUE, RED)]
    engine.reset(5, MatchSetup(
        decks=[deck, deck], shuffle=ShuffleMode.NONE, elixir_milli=[100000, 100000],
        start_tick=engine.rules().deploy_lockout_ticks, tower_hp=hp, spawns=spawns))
    parser = parser_cls()
    parser.bind(engine)
    return engine, parser


def _as_if(parser, arm):
    """``parser`` judging by ``arm`` while the engine runs its own."""
    parser.oracle = PlacementOracle(
        parser.arena,
        msgspec.structs.replace(parser.oracle.rules, enemy_building_taps=arm),
        parser.cards,
    )
    return parser


@needs_engine
@pytest.mark.parametrize("parser_cls", [TileActionParser, HalfTileActionParser])
@pytest.mark.parametrize("down", [False, True], ids=["princesses-up", "princesses-down"])
@pytest.mark.parametrize("team", [BLUE, RED])
def test_the_mask_follows_taps_moved_off_enemy_buildings(parser_cls, down, team):
    """Under a moving arm the mask and the engine agree on every action. Judged as if the
    arm were ignored, the same battle disagrees wherever a tap on an enemy box is moved and
    lands, for the troop, the Miner, the pair and the Heal alike, which is what makes the
    first assertion say something."""
    arm = _arms()[0]
    engine, parser = _board(arm, parser_cls, down)
    state = engine.state()
    assert mask_disagreements(engine, parser, state, team) == []
    found = mask_disagreements(engine, _as_if(parser, "ignored"), state, team)
    assert {(m, s) for _, m, s in found} == {(0, DeployStatus.OK)}
    hand = [engine.cards()[c].name for c in state.players[team].hand]
    moved = {hand[parser.parse(a, state, team).hand_slot] for a, _, _ in found}
    assert moved == {"Knight", "Miner", "Heal", "Archer"}, moved


@needs_engine
@pytest.mark.parametrize("team", [BLUE, RED])
def test_under_ignored_a_tap_on_an_enemy_box_is_refused(team):
    """The old arm: nothing is moved off an enemy box, and the mask refuses those taps as the
    engine does. The same board under the moving arm takes some of them, so the board puts
    taps on enemy boxes at all."""
    moving = _arms()[0]
    engine, parser = _board("ignored", TileActionParser, False)
    state = engine.state()
    assert parser.oracle.rules.enemy_building_taps == "ignored"
    assert mask_disagreements(engine, parser, state, team) == []
    mover, _ = _board(moving, TileActionParser, False)
    taken = 0
    for action in range(1, int(parser.space.n)):
        cmd = parser.parse(action, state, team)
        if engine.check_deploy(cmd) == DeployStatus.OCCUPIED and mover.check_deploy(cmd) == 0:
            taken += 1
    assert taken > 0, "no tap on an enemy box: the board exercises nothing"
