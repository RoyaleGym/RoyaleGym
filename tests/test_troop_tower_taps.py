"""The action mask under placement.TROOP_TOWER_TAPS = client16402_half_open_relocate.

The engine judges a troop tap in two parts under that arm (see ``PlacementOracle``):
- the zone at the tap, with each king block HALF-OPEN in the arena's frame, so the own
  block's max edges open;
- the bodies where the troop will stand. A tap whose tile overlaps an alive OWN crown
  tower's box is moved off it, so no body refuses it.
The mask reads the arm from ``DeployRules`` and must equal the engine's verdict everywhere.
These tests prove it on an engine selecting the arm through ``calibration_overrides``, with
its rules following the override, so they run the same before and after the ledger ships
the arm. The key's flip waited on this half landing first.
"""

from __future__ import annotations

import collections
import json

import msgspec
import numpy as np
import pytest

from royalegym.action import (
    HalfTileActionParser,
    PlacementOracle,
    TileActionParser,
    mask_disagreements,
)
from royalegym.protocol import (
    BLUE,
    HALF_OPEN_RELOCATE,
    RED,
    SNAP_EVEN_CORNER,
    TAP_SNAP,
    TILE_CENTRE_SNAP,
    DeployCommand,
    DeployRules,
    DeployStatus,
    MatchSetup,
    ShuffleMode,
    SpawnSpec,
    default_calibration,
)
from royalegym.rust_engine import (
    CORE_IMPORT_ERROR,
    OVERRIDE_FOLLOWED,
    RustEngine,
    SymmetricRustEngine,
    battle_takes,
    core_available,
    symmetric_overrides,
)

KEY = "placement.TROOP_TOWER_TAPS"
needs_core = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
FULL = [2400, 1400, 1400]
#: Hands (shuffle NONE deals the first four): three troops, a rolling spell, a building.
DECKS = {
    "troops": ["Knight", "Minions", "Archer", "Log", "Cannon", "Fireball", "Giant", "Musketeer"],
    "building": ["Cannon", "Giant", "Knight", "Zap", "Minions", "Archer", "Log", "Fireball"],
}
BOARDS = {
    "all_up": ([FULL, FULL], False),
    "own_princesses_down": ([[2400, 0, 1400], [2400, 1400, 0]], False),
    "enemy_princess_down": ([FULL, [2400, 0, 1400]], False),
    "buildings_by_the_towers": ([FULL, FULL], "beside"),
    # A Cannon straddling the left princess box's outer edge, so its body covers points ON
    # the box edge: the only points whose verdict depends on the frame the tap is snapped in.
    "building_on_the_box_edge": ([FULL, FULL], "edge"),
}


def shipped_arm() -> str | None:
    """The arm the compiled ledger ships, or None on a build older than the key."""
    import royalesim

    entry = json.loads(royalesim.EMBEDDED_CALIBRATION_JSON)["placement"].get("TROOP_TOWER_TAPS")
    return entry["value"] if entry else None


def ledger_has_key() -> bool:
    return shipped_arm() is not None


def engine_on_arm(arm: str, **kwargs) -> RustEngine:
    """A RustEngine running ``arm``, overriding the key only when the ledger ships another,
    so nothing is overridden for nothing (``symmetric_overrides`` does the same)."""
    if shipped_arm() != arm:
        kwargs["calibration_overrides"] = {**kwargs.get("calibration_overrides", {}), KEY: arm}
    engine = RustEngine(**kwargs)
    assert engine.rules().troop_tower_taps == arm, "the rules did not follow the arm"
    return engine


def flipped_engine() -> RustEngine:
    return engine_on_arm(HALF_OPEN_RELOCATE)


def setup(engine: RustEngine, deck: str, board: str) -> MatchSetup:
    ids = {c.name: c.card_id for c in engine.cards()}
    hp, buildings = BOARDS[board]
    t = engine.arena().subtile
    spawns = []
    if buildings:
        # A Cannon by each own left princess box, rotated for Red: the relocated troop must
        # avoid it, and its body still refuses taps of its own.
        x, y = (6 * t, 7 * t) if buildings == "beside" else (5 * t + t // 2, 6 * t + t // 2)
        w, h = engine.arena().width, engine.arena().height
        spawns = [SpawnSpec(BLUE, ids["Cannon"], x, y), SpawnSpec(RED, ids["Cannon"], w - x, h - y)]
    return MatchSetup(
        decks=[[ids[n] for n in DECKS[deck]]] * 2,
        shuffle=ShuffleMode.NONE,
        elixir_milli=[10000, 10000],
        tower_hp=hp,
        spawns=spawns,
        start_tick=engine.rules().deploy_lockout_ticks,
    )


def gate(engine: RustEngine, oracle_patch=None) -> tuple[list[str], dict]:
    """Every (board, deck, parser, seat) disagreement between mask and engine, plus the
    half-cell CORNERS through ``legal_points`` for the Knight (the only points where the
    tap's tile depends on the placer's frame)."""
    from test_rust_engine import on_own_tower_tile

    problems: list[str] = []
    opened: dict[int, collections.Counter] = {team: collections.Counter() for team in (BLUE, RED)}
    # The closed block BY NAME: once the ledger ships the half-open arm, RustEngine() runs
    # it too, and the vacuity count below would compare the arm with itself.
    closed = engine_on_arm("closed_block")
    for board in BOARDS:
        for deck in DECKS:
            engine.reset(1, setup(engine, deck, board))
            closed.reset(1, setup(closed, deck, board))
            state = engine.state()
            for parser_cls in (TileActionParser, HalfTileActionParser):
                parser = parser_cls()
                parser.bind(engine)
                if oracle_patch is not None:
                    oracle_patch(parser.oracle)
                for team in (BLUE, RED):
                    for action, m, status in mask_disagreements(engine, parser, state, team):
                        problems.append(
                            f"{board} {deck} {parser_cls.__name__} seat {team} action {action}: "
                            f"mask {m}, engine {DeployStatus(status).name}"
                        )
                    if parser_cls is TileActionParser and deck == "troops":
                        # Every tap the arm opens, by the law that opens it, placed from the
                        # arena's own king block and the engine's own tower boxes.
                        old = TileActionParser()
                        old.bind(closed)
                        new = parser.action_mask(state, team) & ~old.action_mask(state, team)
                        x0, y0, x1, y1 = engine.arena().king_blocks[team]
                        for action in np.flatnonzero(new).tolist():
                            c = parser.parse(action, state, team)
                            if x0 <= c.x <= x1 and y0 <= c.y <= y1 and (c.x == x1 or c.y == y1):
                                opened[team]["own king block edge"] += 1
                            elif on_own_tower_tile(engine.arena(), state, team, c.x, c.y):
                                opened[team]["own crown tower tile"] += 1
                            else:
                                opened[team][f"neither: ({c.x}, {c.y})"] += 1
            if deck == "troops":
                problems += corner_problems(engine, state, oracle_patch, board)
    return problems, opened


def corner_problems(engine, state, oracle_patch, board) -> list[str]:
    a = engine.arena()
    oracle = PlacementOracle(a, engine.rules(), engine.cards())
    if oracle_patch is not None:
        oracle_patch(oracle)
    h = a.half_size
    kx, ky = np.meshgrid(np.arange(1, a.hx) * h, np.arange(1, a.hy) * h)
    out = []
    for team in (BLUE, RED):
        slot = 0  # the Knight
        card = engine.cards()[state.players[team].hand[slot]]
        legal = oracle.legal_points(state, team, card, kx, ky)
        for x, y, m in zip(kx.ravel().tolist(), ky.ravel().tolist(), legal.ravel().tolist(),
                           strict=True):
            status = engine.check_deploy(DeployCommand(team, slot, x, y))
            if bool(m) != (status == DeployStatus.OK):
                out.append(f"{board} corner seat {team} ({x}, {y}): mask {int(m)}, engine "
                           f"{DeployStatus(status).name}")
    return out


def test_rules_read_the_arm_and_refuse_one_the_mask_lacks():
    cal = default_calibration()
    if "TROOP_TOWER_TAPS" not in cal.raw.get("placement", {}):
        pytest.skip("SKIPPED, NOT PASSED: this ledger has no placement.TROOP_TOWER_TAPS")
    assert DeployRules.load(cal.with_override(KEY, HALF_OPEN_RELOCATE)).troop_tower_taps == (
        HALF_OPEN_RELOCATE
    )
    with pytest.raises(NotImplementedError, match="TROOP_TOWER_TAPS"):
        DeployRules.load(cal.with_override(KEY, "some_future_arm"))
    # Two keys the half-open arm makes the mask depend on: every arm the mask implements is
    # read into the rules, and an arm it does not know is refused.
    for other, field, arms in (
        ("SNAP_EVEN_CORNER", "snap_even_corner", SNAP_EVEN_CORNER),
        ("TAP_SNAP", "tap_snap", TAP_SNAP),
    ):
        if other not in cal.raw["placement"]:
            continue
        half_open = cal.with_override(KEY, HALF_OPEN_RELOCATE)
        for arm in arms:
            rules = DeployRules.load(half_open.with_override(f"placement.{other}", arm))
            assert getattr(rules, field) == arm, (other, arm)
        with pytest.raises(NotImplementedError, match=other):
            DeployRules.load(half_open.with_override(f"placement.{other}", "some_future_arm"))
        closed = cal.with_override(KEY, "closed_block").with_override(
            f"placement.{other}", arms[-1]
        )
        assert DeployRules.load(closed).troop_tower_taps == "closed_block", (
            f"{other} changes no troop verdict under the closed block and must not be refused there"
        )


def test_the_symmetric_vehicle_keeps_the_closed_block_once_the_ledger_ships_the_open_one():
    flipped_ledger = {"placement": {"TROOP_TOWER_TAPS": {"value": HALF_OPEN_RELOCATE}}}
    assert symmetric_overrides(flipped_ledger) == {KEY: "closed_block"}


@needs_core
def test_the_flipped_mask_equals_the_flipped_engine_everywhere():
    if not ledger_has_key():
        pytest.skip(f"SKIPPED, NOT PASSED: this engine's ledger has no {KEY}")
    engine = flipped_engine()
    problems, opened = gate(engine)
    assert not problems, f"{len(problems)} disagreements; first {problems[:6]}"
    # Vacuity: EACH law opens taps the closed block refuses, on both seats, and nothing else
    # opens. Measured 2026-09-27 on 126992a, tile parser, 3 troop slots x 5 boards: 105 king
    # edge taps per seat (7 per slot and board), 135 tower tile taps for Blue and 120 for Red.
    for team in (BLUE, RED):
        assert set(opened[team]) == {"own king block edge", "own crown tower tile"}, opened
        assert opened[team]["own king block edge"] >= 90, opened
        assert opened[team]["own crown tower tile"] >= 100, opened


@needs_core
@pytest.mark.parametrize(
    ("name", "patch"),
    [
        ("law 1 closed again", lambda o: setattr(o, "king_half_open", False)),
        ("law 2 off", lambda o: setattr(o, "troop_taps_relocate", False)),
    ],
)
def test_plant_a_mask_missing_one_law_is_caught(name, patch):
    if not ledger_has_key():
        pytest.skip(f"SKIPPED, NOT PASSED: this engine's ledger has no {KEY}")
    problems, _ = gate(flipped_engine(), oracle_patch=patch)
    assert problems, f"PLANT DID NOT LAND: {name}"


@needs_core
def test_plant_the_tower_tile_snapped_in_the_arenas_frame_is_caught(monkeypatch):
    """Red's tap on a tile boundary floors to the tile on the other side in the arena's
    frame. Only the corners show it; the plant snaps Red like Blue."""
    if not ledger_has_key():
        pytest.skip(f"SKIPPED, NOT PASSED: this engine's ledger has no {KEY}")
    real = PlacementOracle.own_tower_zone

    def arena_frame(self, state, team, px, py):
        if team != RED:
            return real(self, state, team, px, py)
        a, t = self.arena, self.arena.subtile
        cx, cy = (px // t) * t + t // 2, (py // t) * t + t // 2
        zone = np.zeros(np.broadcast(px, py).shape, dtype=bool)
        lo_x, hi_x, lo_y, hi_y = cx - t // 2, cx + t // 2, cy - t // 2, cy + t // 2
        for x0, y0, x1, y1 in self.own_tower_boxes(state, team):
            zone |= (lo_x < x1) & (x0 < hi_x) & (lo_y < y1) & (y0 < hi_y)
        del a
        return zone

    monkeypatch.setattr(PlacementOracle, "own_tower_zone", arena_frame)
    problems, _ = gate(flipped_engine())
    assert any("corner seat 1" in p for p in problems), "PLANT DID NOT LAND: the frame of the snap"


#: The arms of the two snap keys sim's round 9 flips (SNAP_EVEN_CORNER = absolute, TAP_SNAP =
#: client16402_tile_centre), alone and together, run under the half-open arm.
SNAP_ARMS = {
    "absolute": {"placement.SNAP_EVEN_CORNER": "absolute"},
    "tile_centre": {"placement.TAP_SNAP": TILE_CENTRE_SNAP},
    "both": {"placement.SNAP_EVEN_CORNER": "absolute", "placement.TAP_SNAP": TILE_CENTRE_SNAP},
}


def snap_engine(arms: dict[str, str]) -> RustEngine:
    ledger = default_calibration().raw["placement"]
    missing = [k for k in arms if k.split(".", 1)[1] not in ledger]
    if missing:
        pytest.skip(f"SKIPPED, NOT PASSED: this ledger has no {missing}")
    engine = engine_on_arm(HALF_OPEN_RELOCATE, calibration_overrides=dict(arms))
    rules = engine.rules()
    assert rules.snap_even_corner == arms.get("placement.SNAP_EVEN_CORNER", rules.snap_even_corner)
    assert rules.tap_snap == arms.get("placement.TAP_SNAP", rules.tap_snap), "not followed"
    return engine


def test_the_snap_keys_are_followed():
    assert {"placement.SNAP_EVEN_CORNER", "placement.TAP_SNAP"} <= OVERRIDE_FOLLOWED


@needs_core
@pytest.mark.parametrize("arms", sorted(SNAP_ARMS))
def test_the_mask_equals_the_engine_under_each_snap_arm(arms):
    """Both parsers, both seats, every board, and the half-cell corners, where the frame a
    tap is floored in picks the tile. Parity's king-box tie taps (side 1 at (7500, 30500) and
    (10500, 30500), side 0 at (10500, 1500)) are tile centres, so the tile parser asks them."""
    if not ledger_has_key():
        pytest.skip(f"SKIPPED, NOT PASSED: this engine's ledger has no {KEY}")
    problems, _ = gate(snap_engine(SNAP_ARMS[arms]))
    assert not problems, f"{arms}: {len(problems)} disagreements; first {problems[:6]}"


@needs_core
def test_plant_bodies_judged_at_the_raw_tap_under_the_tile_centre_snap_is_caught(monkeypatch):
    """Under client16402_tile_centre the core judges a troop's body at its tap's tile centre;
    a mask that judges it at the tap disagrees where a half-tile tap and its tile centre fall
    on different sides of a body's edge, or of an own tower's box at a corner."""
    if not ledger_has_key():
        pytest.skip(f"SKIPPED, NOT PASSED: this engine's ledger has no {KEY}")
    monkeypatch.setattr(PlacementOracle, "_bodies_at_tile_centre", lambda self, card: False)
    problems, _ = gate(snap_engine(SNAP_ARMS["tile_centre"]))
    assert problems, "PLANT DID NOT LAND: bodies at the raw tap"


@needs_core
def test_plant_the_placer_frame_snap_under_the_absolute_arm_is_caught(monkeypatch):
    """The absolute arm floors Red's tap in the arena's frame; a mask still flooring it in
    Red's own frame disagrees at the half-cell corners (the mirror of the plant above)."""
    if not ledger_has_key():
        pytest.skip(f"SKIPPED, NOT PASSED: this engine's ledger has no {KEY}")
    real = PlacementOracle.own_tower_zone

    def placer_frame(self, state, team, px, py):
        self.rules = msgspec.structs.replace(self.rules, snap_even_corner="placer_frame")
        try:
            return real(self, state, team, px, py)
        finally:
            self.rules = msgspec.structs.replace(self.rules, snap_even_corner="absolute")

    monkeypatch.setattr(PlacementOracle, "own_tower_zone", placer_frame)
    problems, _ = gate(snap_engine(SNAP_ARMS["absolute"]))
    assert any("corner seat 1" in p for p in problems), "PLANT DID NOT LAND: the frame of the snap"


@needs_core
def test_the_symmetric_engine_asks_for_no_tap_snap_where_the_core_takes_the_keyword():
    """Sim's round 9 adds Battle(tap_snap=); the tile-centre arm snaps in the arena's frame,
    so the rotation gates run under "none". An engine without the keyword refuses the
    argument, and the symmetric engine then passes nothing."""
    engine = SymmetricRustEngine()
    if battle_takes("tap_snap"):
        assert engine.tap_snap == "none"
        assert engine.rules().tap_snap == "none" or engine.rules().troop_tower_taps != (
            HALF_OPEN_RELOCATE
        )
        assert engine.config()["tap_snap"] == "none"
    else:
        assert engine.tap_snap is None
        assert "tap_snap" not in engine.config()
        with pytest.raises(ValueError, match="tap_snap"):
            RustEngine(tap_snap="none")


@needs_core
def test_a_spell_with_a_troop_placement_keeps_the_closed_block():
    """The core applies the half-open laws to a card of KIND troop only (state.rs
    check_position), and gives a spell whose deploy rule is a troop's (Heal, from RoyaleSim's
    next build) a troop's placement code. The mask must keep such a spell on the closed
    block, and the ``point_grid`` memo must not hand it a troop's grid. Checked on a Knight
    re-kinded as a spell: its grid is the Knight's under the closed block, not the open one,
    in either call order. The engine's own verdict for Heal is held by the every-card gate
    in tests/test_building_footprint.py."""
    import msgspec

    if not ledger_has_key():
        pytest.skip(f"SKIPPED, NOT PASSED: this engine's ledger has no {KEY}")
    engine = flipped_engine()
    engine.reset(1, setup(engine, "troops", "all_up"))
    state = engine.state()
    knight = next(c for c in engine.cards() if c.name == "Knight")
    spell = msgspec.structs.replace(knight, card_kind="SPELL")
    closed_rules = msgspec.structs.replace(engine.rules(), troop_tower_taps="closed_block")
    closed = PlacementOracle(engine.arena(), closed_rules, engine.cards())
    for first, second in ((spell, knight), (knight, spell)):
        oracle = PlacementOracle(engine.arena(), engine.rules(), engine.cards())
        for team in (BLUE, RED):
            for pitch in (1, 2):
                got = {c.card_kind: oracle.point_grid(state, team, c, pitch)
                       for c in (first, second)}
                want = closed.point_grid(state, team, knight, pitch)
                assert np.array_equal(got["SPELL"], want), (team, pitch, "not the closed block")
                assert not np.array_equal(got["SPELL"], got["TROOP"]), (team, pitch, "shared")
                px, py = oracle.points(pitch)
                assert np.array_equal(
                    got["SPELL"], oracle.legal_points(state, team, spell, px, py)
                ), (team, pitch, "point_grid != legal_points")
