"""Building footprints: where a building may stand, and the box it covers once it does.

WHY IT EXISTS
    A Cannon covers a 3x3 block of tiles in the game. The engine has placed a building
    by its CollisionRadius circle and tested only the tapped point, so a Cannon could
    stand flush against the back wall, and the viewer drew it about one tile wide. The
    engine is moving to tile boxes. This file holds the gym to what the engine reports,
    and says plainly which checks wait for the engine.

WHAT IT CHECKS
    a. THE GATE (passes today, permanent): the action mask equals ``check_deploy`` for
       every building card in the catalogue, at every tile and every half-tile, both
       seats, on four boards: the opening, buildings of both teams placed, a princess
       tower down on each side, and both at once.
    b. Where a legal Cannon stands: for every tile the mask offers, the Cannon's tile box
       lies inside the arena, in its own half, off water and no-deploy cells and off every
       tower. And the engine refuses a Cannon tapped on the back row. Both are EXPECTED TO
       FAIL (strict) while the engine reports no footprint, and must pass once it does.
    c. Frames: once the engine reports footprints, every building and tower in its state
       has one and no troop does, and a viewer frame and a saved trace carry exactly the
       state's box. Until then the same transport is held to a state that carries boxes.
    d. Old rows: an entity row without the footprint column decodes with None.
    e. The seat rotation turns a box corner to corner.
    f. MockEngine's statement (``mock_engine.FOOTPRINT_MODEL``): it models the circle,
       reports no box, and its deploy check has the circle's shape. Compared with the
       model RustEngine runs; while the two differ the comparison SKIPS, naming both.

PLANTS (each seen red on a green baseline)
    the mask blind to placed buildings (the gate; kept below as a test); the frame
    dropping the footprint; a box rotated without swapping its corners, and one not
    rotated at all; an old row decoding with a zero box; the limit check reading Red's
    half in the engine frame; MockEngine reporting a box under a circle statement (kept
    below as a test).

SKIPS
    Engine tests skip when royalesim is not built. The engine frame test and the
    engine-graded rotation skip, saying why, until the engine reports footprints. A skip
    is not a pass.
"""

from __future__ import annotations

import json
from typing import Any

import msgspec
import numpy as np
import pytest

from royalegym import mock_engine as mock_engine_module
from royalegym.action import (
    GridActionParser,
    HalfTileActionParser,
    PlacementOracle,
    TileActionParser,
    mask_disagreements,
)
from royalegym.mock_engine import MockEngine
from royalegym.protocol import (
    BIT_NO_DEPLOY,
    BIT_WATER,
    BLUE,
    DECK_SIZE,
    HAND_SIZE,
    RED,
    TEAMS,
    Arena,
    BattleState,
    DeployCommand,
    DeployStatus,
    Engine,
    EntityKind,
    EntityState,
    MatchSetup,
    Placement,
    ShuffleMode,
    SpawnSpec,
    TowerSlot,
    mirror_state,
    to_engine,
    to_own,
)
from royalegym.replay import ReplayRecorder, Trace, TraceFrame, load_trace, save_trace
from royalegym.rust_engine import (
    CORE_IMPORT_ERROR,
    RustEngine,
    SymmetricRustEngine,
    core_available,
)
from royalegym.viser import frame_dict, names_of

needs_core = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))

SEAT = ("Blue", "Red")
TOWER_KINDS = (EntityKind.KING_TOWER, EntityKind.PRINCESS_TOWER)
NOT_A_PASS = "SKIPPED, NOT PASSED"
# What a Cannon covers in the game, in tiles. Used until the engine's catalogue says.
CANNON_TILES = (3, 3)
# Enough for any card, and a different bar per seat so a swapped seat cannot hide.
FULL_ELIXIR = [10000, 9000]
# Own tile both seats play a Cannon on: clear of every tower and placed building
# below, under a circle and under a 3x3 box alike.
PLAY_TILE = (4, 10)
NOT_REPORTED = (
    "the engine reports no building footprint yet: its catalogue has no footprint_tiles "
    "and no entity in its state has a footprint"
)
BOARDS = ("opening", "buildings", "princess_down", "buildings_and_princess_down")


@pytest.fixture(scope="module")
def rust() -> RustEngine:
    """The whole catalogue, so every building card the engine has is in the gate."""
    return RustEngine()


# ---------------------------------------------------------------------------
# What the engine reports


def catalogue_footprint_tiles(engine: RustEngine) -> dict[str, tuple[int, int]]:
    """Card name -> (w, h) tiles from the engine's own catalogue. Empty while it has none.

    Read from ``CardInfo.footprint_tiles`` when the adapter carries the column, and
    from the catalogue rows otherwise, where a building row gains a trailing [w, h].
    """
    out: dict[str, tuple[int, int]] = {}
    for c in engine.cards():
        tiles = getattr(c, "footprint_tiles", None)
        if tiles is not None:
            # The catalogue states the side N of an N x N box as one integer. A pair is
            # accepted too, so this reads a catalogue that ever states w and h apart.
            side = (int(tiles), int(tiles)) if isinstance(tiles, int) else tuple(map(int, tiles))
            n = (side[0], side[1])
            out[c.name] = n
    if out:
        return out
    for row in json.loads(engine._battle.catalogue_json()):
        if isinstance(row, dict):
            name, tiles = row.get("name"), row.get("footprint_tiles")
        else:
            name = row[0]
            tiles = next((v for v in row[7:] if isinstance(v, list) and len(v) == 2), None)
        if tiles is not None:
            out[str(name)] = (int(tiles[0]), int(tiles[1]))
    return out


def footprints_reported(engine: RustEngine) -> bool:
    """Whether the engine reports footprints at all: in its catalogue or its state."""
    if catalogue_footprint_tiles(engine):
        return True
    return any(e.footprint is not None for e in engine.state().entities)


def fail_until_reported(request: pytest.FixtureRequest, engine: RustEngine) -> None:
    """Mark the running test as a strict expected failure while no footprint is reported.

    Call it just before the one assert it excuses. A control that fails before it, or a
    crash anywhere, is still a plain failure.
    """
    if not footprints_reported(engine):
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason=f"{NOT_REPORTED}; this check must fail until it does, then pass",
            )
        )


# ---------------------------------------------------------------------------
# Boards


def card_ids(engine: Engine) -> dict[str, int]:
    return {c.name: c.card_id for c in engine.cards()}


def at_tile(engine: Engine, x_tiles2: int, y_tiles2: int) -> tuple[int, int]:
    """Engine-frame subtiles of a point given in HALF tiles (7 is 3.5 tiles)."""
    t = engine.arena().subtile
    return x_tiles2 * t // 2, y_tiles2 * t // 2


def board_setup(engine: Engine, board: str, decks: list[list[int]]) -> MatchSetup:
    """``board``'s setup with ``decks`` in hand order (no shuffle) and a full bar."""
    ids = card_ids(engine)
    spawns: list[SpawnSpec] = []
    if "buildings" in board:
        # Blue: two buildings and a troop. Red: three buildings. The counts differ per
        # seat and no Red spot is the rotation of a Blue one.
        spawns = [
            SpawnSpec(BLUE, ids["Cannon"], *at_tile(engine, 29, 23)),
            SpawnSpec(BLUE, ids["Tesla"], *at_tile(engine, 20, 24)),
            SpawnSpec(BLUE, ids["Knight"], *at_tile(engine, 17, 19)),
            SpawnSpec(RED, ids["InfernoTower"], *at_tile(engine, 9, 45)),
            SpawnSpec(RED, ids["Tombstone"], *at_tile(engine, 19, 39)),
            SpawnSpec(RED, ids["Mortar"], *at_tile(engine, 11, 57)),
        ]
    tower_hp = None
    if "princess_down" in board:
        # Blue's own-left princess and Red's own-right princess start destroyed.
        tower_hp = [[3000, 0, 2000], [3100, 2100, 0]]
    return MatchSetup(
        decks=decks,
        shuffle=ShuffleMode.NONE,
        elixir_milli=FULL_ELIXIR,
        tower_hp=tower_hp,
        spawns=spawns,
        # Past the opening deploy lockout, or every tap below answers TOO_EARLY and these
        # tests grade a timing rule instead of a footprint one. Read from the engine: 0 is
        # a real calibration arm, so a literal would be wrong on a build without a lockout.
        start_tick=engine.rules().deploy_lockout_ticks,
    )


def building_hands(engine: Engine) -> list[list[list[int]]]:
    """Deck pairs that between them put every building card in both seats' hands.

    Red holds each group in reverse order, so a hand slot names a different card per seat.
    """
    buildings = [c.card_id for c in engine.cards() if c.placement == Placement.BUILDING]
    pad = [card_ids(engine)["Knight"]] * DECK_SIZE
    out = []
    for i in range(0, len(buildings), HAND_SIZE):
        group = buildings[i : i + HAND_SIZE]
        out.append([(group + pad)[:DECK_SIZE], (group[::-1] + pad)[:DECK_SIZE]])
    return out


def own_tile_centre(engine: Engine, team: int, tx: int, ty: int) -> tuple[int, int]:
    t = engine.arena().subtile
    return to_engine(engine.arena(), team, tx * t + t // 2, ty * t + t // 2)


def building_gate(engine: Engine, board: str, parser: GridActionParser) -> list[str]:
    """Every (seat, card, point) where the mask and ``check_deploy`` disagree on ``board``.

    Also refuses a vacuous board: each building card must be offered somewhere and
    refused somewhere, for both seats.
    """
    problems: list[str] = []
    for decks in building_hands(engine):
        engine.reset(1, board_setup(engine, board, decks))
        parser.bind(engine)
        state = engine.state()
        for team in TEAMS:
            hand = state.players[team].hand
            planes = parser.action_mask(state, team)[1:].reshape(HAND_SIZE, -1)
            for slot, card_id in enumerate(hand):
                card = engine.cards()[card_id]
                offered = int(planes[slot].sum())
                if card.placement == Placement.BUILDING and not 0 < offered < planes[slot].size:
                    problems.append(f"{SEAT[team]} {card.name}: the mask offers {offered} points")
            for action, m, status in mask_disagreements(engine, parser, state, team):
                slot, xi, yi = parser.decode(action)
                name = engine.cards()[hand[slot]].name
                problems.append(
                    f"{SEAT[team]} {name} at own point ({xi}, {yi}): mask {m}, "
                    f"engine {DeployStatus(status).name}"
                )
    return problems


@needs_core
@pytest.mark.parametrize("parser_cls", [TileActionParser, HalfTileActionParser])
@pytest.mark.parametrize("board", BOARDS)
def test_the_mask_equals_the_engine_for_every_building_card(rust, board, parser_cls):
    problems = building_gate(rust, board, parser_cls())
    assert not problems, f"{board}: {len(problems)} disagreements, first {problems[:8]}"


@needs_core
def test_plant_mask_blind_to_placed_buildings_is_caught(rust, monkeypatch):
    """The buildings board has to reach the gate for BOTH seats: a mask that forgets the
    placed buildings must disagree with the engine on each side."""
    real = PlacementOracle.point_grid

    def blind(self, state, team, card, pitch_div):
        bare = [e for e in state.entities if e.kind != EntityKind.BUILDING]
        return real(self, msgspec.structs.replace(state, entities=bare), team, card, pitch_div)

    monkeypatch.setattr(PlacementOracle, "point_grid", blind)
    problems = building_gate(rust, "buildings", TileActionParser())
    seats = {p.split()[0] for p in problems}
    assert seats == {"Blue", "Red"}, f"PLANT DID NOT LAND on both seats: {problems[:4]}"


@needs_core
def test_each_board_is_what_it_says(rust):
    """Graded against the engine's own state and verdicts, per seat."""
    cannon = card_ids(rust)["Cannon"]
    decks = [[cannon] * DECK_SIZE, [cannon] * DECK_SIZE]

    rust.reset(1, board_setup(rust, "buildings", decks))
    s = rust.state()
    buildings = [
        sum(e.team == team and e.kind == EntityKind.BUILDING for e in s.entities) for team in TEAMS
    ]
    troops = [
        sum(e.team == team and e.kind == EntityKind.TROOP for e in s.entities) for team in TEAMS
    ]
    assert (buildings, troops) == ([2, 3], [1, 0])
    # A building card tapped on the seat's own placed building is ACCEPTED: a tap whose
    # box does not fit is relocated, not refused, so nothing already on the board makes a
    # building tap illegal. Where it then lands is what
    # test_every_cannon_the_engine_builds_stands_on_ground_it_may_stand_on grades.
    for team, spot in ((BLUE, at_tile(rust, 29, 23)), (RED, at_tile(rust, 9, 45))):
        assert rust.check_deploy(DeployCommand(team, 0, *spot)) == DeployStatus.OK

    rust.reset(1, board_setup(rust, "princess_down", decks))
    s = rust.state()
    assert s.players[BLUE].tower_hp[TowerSlot.LEFT] == 0 < s.players[BLUE].tower_hp[TowerSlot.RIGHT]
    assert s.players[RED].tower_hp[TowerSlot.RIGHT] == 0 < s.players[RED].tower_hp[TowerSlot.LEFT]
    # A Cannon on the own tile of a princess tower is accepted whether or not that tower
    # still stands, because a tower in the way relocates the building rather than
    # refusing the tap. The board is still what it says: the tower states differ, which
    # is what the two asserts above check.
    for team, tx in ((BLUE, 3), (RED, 3), (BLUE, 14), (RED, 14)):
        cmd = DeployCommand(team, 0, *own_tile_centre(rust, team, tx, 6))
        assert rust.check_deploy(cmd) == DeployStatus.OK, (SEAT[team], tx)


# ---------------------------------------------------------------------------
# Where a legal Cannon stands


def box_overlaps_tower(box: tuple[int, int, int, int], e: EntityState) -> bool:
    """Whether ``box`` covers part of tower ``e``: its reported box, else its circle.

    Boxes that only share an edge do not overlap, and a circle only touched is not covered.
    """
    x0, y0, x1, y1 = box
    if e.footprint is not None:
        a0, b0, a1, b1 = e.footprint
        return x0 < a1 and a0 < x1 and y0 < b1 and b0 < y1
    dx = max(x0 - e.x, 0, e.x - x1)
    dy = max(y0 - e.y, 0, e.y - y1)
    return dx * dx + dy * dy < e.radius * e.radius


def box_limit_failures(
    arena: Arena,
    state: BattleState,
    team: int,
    own_tile: tuple[int, int],
    size: tuple[int, int],
) -> list[str]:
    """Which limits a building of ``size`` tiles breaks when tapped on ``own_tile``.

    An odd side is centred on the tapped tile. Checked on the tile box itself: inside
    the arena, every half-cell under it in the seat's own half and neither water nor
    no-deploy, and no crown tower under it.
    """
    w, h = size
    if w % 2 == 0 or h % 2 == 0:
        raise ValueError(f"size {size}: an even side snaps to a tile corner not modelled here")
    tx, ty = own_tile
    ex, ey = (tx, ty) if team == BLUE else (arena.tiles_x - 1 - tx, arena.tiles_y - 1 - ty)
    x0, x1, y0, y1 = ex - w // 2, ex + w // 2, ey - h // 2, ey + h // 2  # tiles, inclusive
    if x0 < 0 or y0 < 0 or x1 >= arena.tiles_x or y1 >= arena.tiles_y:
        return ["outside the arena"]
    cells = [
        (hx, hy)
        for hy in range(y0 * arena.half, (y1 + 1) * arena.half)
        for hx in range(x0 * arena.half, (x1 + 1) * arena.half)
    ]
    out = []
    own_rows = [hy if team == BLUE else arena.hy - 1 - hy for _, hy in cells]
    if max(own_rows) >= arena.water_half_rows[0]:
        out.append("outside its own half")
    if any(arena.grid[hy][hx] & BIT_WATER for hx, hy in cells):
        out.append("on water")
    if any(arena.grid[hy][hx] & BIT_NO_DEPLOY for hx, hy in cells):
        out.append("on a no-deploy cell")
    t = arena.subtile
    box = (x0 * t, y0 * t, (x1 + 1) * t, (y1 + 1) * t)
    if any(e.kind in TOWER_KINDS and box_overlaps_tower(box, e) for e in state.entities):
        out.append("on a tower")
    return out


def test_the_limit_check_passes_a_clear_spot_and_names_each_limit():
    """The instrument of the next test, on MockEngine's opening board (the same arena and
    tower centres), both seats on the same own tiles: a spot clear of everything passes,
    and each limit is named where it is at stake."""
    eng = MockEngine()
    eng.reset(1, MatchSetup(decks=[list(range(DECK_SIZE))] * 2))
    arena, state = eng.arena(), eng.state()
    cases = {
        (PLAY_TILE, CANNON_TILES): [],
        ((8, 0), CANNON_TILES): ["outside the arena"],
        ((0, 10), CANNON_TILES): ["outside the arena"],
        ((6, 14), CANNON_TILES): ["outside its own half", "on water"],
        ((14, 15), (1, 1)): ["outside its own half"],  # a bridge tile: dry, not home
        ((9, 5), CANNON_TILES): ["on a no-deploy cell", "on a tower"],  # the king
        ((3, 8), CANNON_TILES): ["on a tower"],  # the own-left princess
    }
    for team in TEAMS:
        for (tile, size), want in cases.items():
            got = box_limit_failures(arena, state, team, tile, size)
            assert got == want, (SEAT[team], tile)


@needs_core
def test_every_cannon_the_engine_builds_stands_on_ground_it_may_stand_on(rust):
    """THE OWNER'S DEFECT, graded on what the engine BUILT rather than on what was tapped.

    The owner saw a Cannon sitting against the arena wall. A tap is not the place to
    catch that any more: a tap whose box does not fit is relocated rather than refused
    (``placement.ILLEGAL_TAP``), so almost every tap on a player's own half is legal and
    the question is only where the building ends up. So this taps EVERY tile the mask
    offers, lets the engine place the Cannon, and reads the box back out of
    ``engine.state()``: it must lie inside the arena, inside its own half, off water and
    off no-deploy cells, and never overlap a crown tower.

    Both seats, and the two seats must also agree: a rule that holds for Blue and not
    for Red is a rule about the colour, which this game does not have.
    """
    cannon = card_ids(rust)["Cannon"]
    setup = board_setup(rust, "opening", [[cannon] * DECK_SIZE] * 2)
    arena = rust.arena()
    built = [0, 0]
    failures: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for team in TEAMS:
        rust.reset(1, setup)
        state = rust.state()
        parser = TileActionParser()
        parser.bind(rust)
        mask = parser.action_mask(state, team)[1:].reshape(HAND_SIZE, parser.ny, parser.nx)
        offers = [(int(tx), int(ty)) for ty, tx in zip(*np.nonzero(mask[0]), strict=True)]
        for tile in offers:
            rust.reset(1, setup)
            before = {e.uid for e in rust.state().entities}
            cmd = DeployCommand(team, 0, *own_tile_centre(rust, team, *tile))
            if rust.check_deploy(cmd) != DeployStatus.OK:
                continue
            rust.step([cmd], 1)
            after = rust.state()
            new_ents = [e for e in after.entities if e.uid not in before]
            if not new_ents:
                failures.setdefault((SEAT[team], "accepted but nothing was built"), []).append(tile)
                continue
            built[team] += 1
            for e in new_ents:
                for why in box_failures(arena, after, team, e):
                    failures.setdefault((SEAT[team], why), []).append(tile)
    assert min(built) > 0, f"no Cannon was built on some seat: {built}"
    assert built[0] == built[1], f"the seats were offered different numbers of taps: {built}"
    detail = "\n".join(
        f"  {seat} {why}: {len(tiles)} own tiles {sorted(tiles)[:12]}"
        for (seat, why), tiles in sorted(failures.items())
    )
    assert not failures, f"a Cannon the engine built stands where no building may:\n{detail}"


@needs_core
def test_a_cannon_tapped_at_the_back_wall_is_moved_inside_it(rust):
    """The owner's case exactly: tap the back row and see where the Cannon ends up.

    It is no longer refused, so the test is not "was it refused" but "did it end up
    somewhere a building may stand". A Cannon whose box would hang off the back wall
    must come back inside it.
    """
    cannon = card_ids(rust)["Cannon"]
    setup = board_setup(rust, "opening", [[cannon] * DECK_SIZE] * 2)
    arena = rust.arena()
    moved = [0, 0]
    problems: list[str] = []
    for team in TEAMS:
        for tx in range(arena.tiles_x):
            rust.reset(1, setup)
            before = {e.uid for e in rust.state().entities}
            cmd = DeployCommand(team, 0, *own_tile_centre(rust, team, tx, 0))
            if rust.check_deploy(cmd) != DeployStatus.OK:
                continue
            rust.step([cmd], 1)
            after = rust.state()
            for e in (x for x in after.entities if x.uid not in before):
                why = box_failures(arena, after, team, e)
                if why:
                    problems.append(f"{SEAT[team]} own column {tx}: {', '.join(why)}")
                elif e.footprint is not None:
                    moved[team] += 1
    detail = "\n".join(problems[:12])
    assert not problems, f"a Cannon tapped on the back row stands where no building may:\n{detail}"
    assert min(moved) > 0, f"no back-row tap built a Cannon with a box on some seat: {moved}"


def box_failures(arena: Arena, state: BattleState, team: int, e) -> list[str]:
    """Which limits the box an entity actually carries breaks. Empty when it breaks none."""
    if e.footprint is None:
        return []
    x0, y0, x1, y1 = e.footprint
    hy0 = y0 // arena.half_size
    hx0 = x0 // arena.half_size
    out = []
    if x0 < 0 or y0 < 0 or x1 > arena.width or y1 > arena.height:
        out.append("outside the arena")
        return out
    cells = [
        (hx, hy)
        for hy in range(hy0, max(hy0 + 1, y1 // arena.half_size))
        for hx in range(hx0, max(hx0 + 1, x1 // arena.half_size))
        if hy < arena.hy and hx < arena.hx
    ]
    own_rows = [hy if team == BLUE else arena.hy - 1 - hy for _, hy in cells]
    if own_rows and max(own_rows) >= arena.water_half_rows[0]:
        out.append("outside its own half")
    if any(arena.grid[hy][hx] & BIT_WATER for hx, hy in cells):
        out.append("on water")
    if any(arena.grid[hy][hx] & BIT_NO_DEPLOY for hx, hy in cells):
        out.append("on a no-deploy cell")
    for other in state.entities:
        if other.uid == e.uid or other.kind not in TOWER_KINDS:
            continue
        if box_overlaps_tower((x0, y0, x1, y1), other):
            out.append("on a tower")
            break
    return out


# ---------------------------------------------------------------------------
# Frames and traces


def record(engine: Engine, setup: MatchSetup, cannon: int) -> tuple[list[BattleState], Trace]:
    """A short battle: both seats play a Cannon on ``PLAY_TILE``, then a few ticks pass.
    Returns the states in the order the trace recorded its frames."""
    rec = ReplayRecorder(frame_every_tick=False)
    engine.reset(1, setup)
    rec.begin(engine, 1, setup)
    states = [engine.state()]
    hands = [p.hand for p in states[0].players]
    plays = [
        DeployCommand(team, hands[team].index(cannon), *own_tile_centre(engine, team, *PLAY_TILE))
        for team in TEAMS
    ]
    for cmds in (plays, [], []):
        tick = engine.state().tick
        results = engine.step(cmds, 5)
        assert [r.status for r in results] == [DeployStatus.OK] * len(cmds)
        rec.record_step(tick, 5, cmds, results)
        rec.record_frame(engine)
        states.append(engine.state())
    trace = rec.end(engine)
    assert trace is not None
    return states, trace


def transport_problems(
    engine: Engine, states: list[BattleState], trace: Trace, tmp_path: Any
) -> list[str]:
    """Where a viewer frame or a saved and reloaded trace carries a box other than the
    state's own."""
    out = []
    name_of = names_of(engine.cards())
    for i, s in enumerate(states):
        d = frame_dict(s, name_of, engine.arena().subtile)
        want = [list(e.footprint) if e.footprint is not None else None for e in s.entities]
        if [u.get("footprint") for u in d["units"]] != want:
            out.append(f"frame {i}: the viewer frame's footprints are not the state's")
    if trace.header.entity_fields[-1:] != ["footprint"]:
        out.append(f"trace header entity_fields {trace.header.entity_fields}")
    for suffix in (".json", ".msgpack"):
        loaded = load_trace(save_trace(trace, tmp_path / f"trace{suffix}"))
        for i, (fr, s) in enumerate(zip(loaded.frames, states, strict=True)):
            if [e.footprint for e in fr.entities] != [e.footprint for e in s.entities]:
                out.append(f"{suffix} frame {i}: the trace's footprints are not the state's")
    return out


def with_boxes(s: BattleState) -> BattleState:
    """``s`` with a box on every building and tower. Each box is its own, per entity and
    per seat, so a shared or swapped box shows; each holds its entity's centre."""
    ents = [
        e
        if e.kind == EntityKind.TROOP
        else msgspec.structs.replace(
            e,
            footprint=(e.x - 900 - e.uid, e.y - 1800 - 3 * e.uid, e.x + 2700 + e.team, e.y + 3600),
        )
        for e in s.entities
    ]
    return msgspec.structs.replace(s, entities=ents)


class BoxedMock(MockEngine):
    """MockEngine whose state carries boxes (``with_boxes``): a stand-in for an engine that
    reports them, so the transport can be held to a state with boxes today."""

    def state(self) -> BattleState:
        return with_boxes(super().state())


def mock_cannon_setup(engine: Engine) -> MatchSetup:
    """Cannons in hand at a different slot per seat; a Cannon of each seat and a Blue
    Knight already on the board, clear of ``PLAY_TILE``."""
    ids = card_ids(engine)
    cannon, knight = ids["Cannon"], ids["Knight"]
    return MatchSetup(
        decks=[[cannon] + [knight] * 7, [knight] * 3 + [cannon] * 5],
        shuffle=ShuffleMode.NONE,
        elixir_milli=FULL_ELIXIR,
        spawns=[
            SpawnSpec(BLUE, cannon, *at_tile(engine, 29, 23)),
            SpawnSpec(BLUE, knight, *at_tile(engine, 17, 19)),
            SpawnSpec(RED, cannon, *at_tile(engine, 9, 45)),
        ],
    )


def boxed_battle() -> tuple[BoxedMock, list[BattleState], Trace]:
    eng = BoxedMock()
    states, trace = record(eng, mock_cannon_setup(eng), card_ids(eng)["Cannon"])
    return eng, states, trace


def test_a_frame_and_a_trace_carry_exactly_the_state_footprint(tmp_path):
    eng, states, trace = boxed_battle()
    last = states[-1]
    assert sum(e.footprint is not None for e in last.entities) == 6 + 2 + 2, (
        "six towers, two placed and two played Cannons"
    )
    assert any(e.footprint is None for e in last.entities), "a troop, with no box"
    assert transport_problems(eng, states, trace, tmp_path) == []


def test_the_viewer_decodes_the_frame_footprint():
    model = pytest.importorskip("royaleviser.model")
    eng, states, _ = boxed_battle()
    s = states[-1]
    d = frame_dict(s, names_of(eng.cards()), eng.arena().subtile)
    frame = model.decode_frame(msgspec.msgpack.encode(d))
    assert model.problems(frame) == []
    assert [u.footprint for u in frame.units] == [e.footprint for e in s.entities]


@needs_core
def test_the_engine_state_frames_and_traces_carry_the_engine_footprint(rust, tmp_path):
    cannon = card_ids(rust)["Cannon"]
    setup = board_setup(rust, "buildings", [[cannon] * DECK_SIZE] * 2)
    rust.reset(1, setup)
    if not footprints_reported(rust):
        pytest.skip(
            f"{NOT_A_PASS}: {NOT_REPORTED}. This test holds the state, the viewer frame and "
            "the trace to the engine's boxes, and runs once the engine reports them."
        )
    states, trace = record(rust, setup, cannon)
    problems = []
    for i, s in enumerate(states):
        for e in s.entities:
            label = f"state {i} {SEAT[e.team]} {EntityKind(e.kind).name} uid {e.uid}"
            box = e.footprint
            if e.kind == EntityKind.TROOP:
                if box is not None:
                    problems.append(f"{label}: a troop with a footprint {box}")
            elif box is None:
                problems.append(f"{label}: no footprint")
            elif not (box[0] <= e.x <= box[2] and box[1] <= e.y <= box[3]):
                problems.append(f"{label}: footprint {box} does not hold its centre")
    kinds = {EntityKind(e.kind).name for s in states for e in s.entities}
    assert kinds == {"TROOP", "BUILDING", "KING_TOWER", "PRINCESS_TOWER"}
    assert not problems, problems[:8]
    assert transport_problems(rust, states, trace, tmp_path) == []


# ---------------------------------------------------------------------------
# Old rows, and the seat rotation


ROW = [7, 1, EntityKind.BUILDING, 3, -1, 99000, 405000, 500, 824, 10800, False, 2, 0, 0]


@pytest.mark.parametrize("codec", [msgspec.json, msgspec.msgpack])
def test_a_row_without_the_footprint_column_decodes_with_none(codec):
    """Rows as an engine or a trace wrote them before the column: 14 elements, and 12
    from before the status timers. A trailing box decodes as the box."""
    old = codec.decode(codec.encode(ROW), type=EntityState)
    assert (old.uid, old.x, old.y, old.max_hp, old.footprint) == (7, 99000, 405000, 824, None)
    older = codec.decode(codec.encode(ROW[:12]), type=EntityState)
    assert (older.stun_ticks, older.footprint) == (0, None)
    boxed = codec.decode(codec.encode([*ROW, [72000, 378000, 126000, 432000]]), type=EntityState)
    assert boxed.footprint == (72000, 378000, 126000, 432000)
    assert codec.decode(codec.encode([*ROW, None]), type=EntityState).footprint is None
    frame = codec.decode(
        codec.encode([12, [ROW, ROW[:12]], [5000, 4000], [0, 1], [[0, 1, 2, 3]] * 2, "00ab"]),
        type=TraceFrame,
    )
    assert [e.footprint for e in frame.entities] == [None, None]


def box_corners(b: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    return {(b[0], b[1]), (b[0], b[3]), (b[2], b[1]), (b[2], b[3])}


def rotation_problems(arena: Arena, s: BattleState) -> list[str]:
    """Where ``mirror_state`` does not turn each box into the rotation of its corners."""
    out = []
    m = mirror_state(arena, s)
    for e, r in zip(s.entities, m.entities, strict=True):
        if e.footprint is None or r.footprint is None:
            if e.footprint != r.footprint:
                out.append(f"uid {e.uid}: {e.footprint} became {r.footprint}")
            continue
        turned = {to_own(arena, RED, x, y) for x, y in box_corners(e.footprint)}
        well_formed = r.footprint[0] <= r.footprint[2] and r.footprint[1] <= r.footprint[3]
        if box_corners(r.footprint) != turned or not well_formed:
            out.append(f"uid {e.uid}: {e.footprint} became {r.footprint}")
    if mirror_state(arena, m) != s:
        out.append("rotating twice is not the identity")
    return out


def test_the_seat_rotation_turns_a_box_corner_to_corner():
    eng, states, _ = boxed_battle()
    s = states[-1]
    assert sum(e.footprint is not None for e in s.entities) == 10
    assert rotation_problems(eng.arena(), s) == []


@needs_core
def test_mirrored_cannons_report_rotated_boxes(rust):
    """Graded against the engine: both seats play a Cannon on the same own tile, and the
    state must equal its own rotation, boxes included."""
    cannon = card_ids(rust)["Cannon"]
    rust.reset(1, board_setup(rust, "opening", [[cannon] * DECK_SIZE] * 2))
    if not footprints_reported(rust):
        pytest.skip(f"{NOT_A_PASS}: {NOT_REPORTED}. This test runs once the engine reports them.")
    eng = SymmetricRustEngine()
    eng.reset(1, board_setup(eng, "opening", [[cannon] * DECK_SIZE] * 2))
    eng.step([DeployCommand(team, 0, *own_tile_centre(eng, team, *PLAY_TILE)) for team in TEAMS], 3)
    s = eng.state()

    def key(st: BattleState) -> list:
        return sorted(
            (e.team, e.kind, e.card_id, e.tower_slot, e.x, e.y, e.footprint) for e in st.entities
        )

    assert sum(e.kind == EntityKind.BUILDING for e in s.entities) == 2
    assert key(s) == key(mirror_state(eng.arena(), s))
    assert rotation_problems(eng.arena(), s) == []


# ---------------------------------------------------------------------------
# MockEngine's statement


def circle_edge_verdicts(engine: Engine, team: int) -> dict[str, int]:
    """A TROOP tapped around the seat's own-left princess, where a circle and a box differ.

    The radius comes from the tower in the engine's own state. Exactly on it is
    OCCUPIED under the circle and one subtile further is not; a
    point on the diagonal, inside the square of that half-width but outside the circle,
    is accepted by the circle and refused by any box that large.

    A TROOP and not a building: a building tap is not refused for a body in the way any
    more, so a building can no longer tell a circle from a box here. What the circle
    still governs is where a troop may be put down.
    """
    a = engine.arena()
    s = engine.state()
    tower = next(e for e in s.entities if e.team == team and e.tower_slot == TowerSlot.LEFT)
    reach = tower.radius  # a troop is refused within the BODY's own radius, not a sum
    diagonal = reach * 3 // 4  # 3/4 < 1 < 3/4 * sqrt(2)
    ox, oy = to_own(a, team, tower.x, tower.y)
    points = {
        "on the sum": (ox + reach, oy),
        "one past it": (ox + reach + 1, oy),
        "on the diagonal": (ox + diagonal, oy + diagonal),
    }
    return {
        what: engine.check_deploy(DeployCommand(team, 0, *to_engine(a, team, x, y)))
        for what, (x, y) in points.items()
    }


def mock_on_cannon_board() -> MockEngine:
    """A board with a Cannon standing, and KNIGHT in every hand slot.

    The hand is a troop because ``circle_edge_verdicts`` asks what the circle still
    decides, and since a building tap stopped being refused for a body in the way, that
    is troop placement.
    """
    eng = MockEngine()
    setup = mock_cannon_setup(eng)
    knight = card_ids(eng)["Knight"]
    eng.reset(1, msgspec.structs.replace(setup, decks=[[knight] * DECK_SIZE] * 2))
    return eng


def mock_statement_problems(eng: MockEngine) -> list[str]:
    """Where MockEngine does not do what ``FOOTPRINT_MODEL`` says, graded on its own state
    and its own verdicts."""
    stated = mock_engine_module.FOOTPRINT_MODEL
    if stated != "collision_radius_circle":
        return [f"FOOTPRINT_MODEL {stated!r} is not the circle this engine runs"]
    out = []
    in_config = eng.config().get("footprint_model")
    if in_config != stated or eng.rules().footprint_model != stated:
        out.append(f"config {in_config!r}, rules {eng.rules().footprint_model!r}")
    s = eng.state()
    if not any(e.kind == EntityKind.BUILDING for e in s.entities):
        out.append("the board has no building")
    out += [f"uid {e.uid} reports a box {e.footprint}" for e in s.entities if e.footprint]
    want = {
        "on the sum": DeployStatus.OCCUPIED,
        "one past it": DeployStatus.OK,
        "on the diagonal": DeployStatus.OK,
    }
    if eng.rules().illegal_building_tap != "refuse" and mock_engine_module.__dict__.get(
        "RELOCATES_A_BUILDING_THAT_DOES_NOT_FIT", False
    ):
        out.append("it claims to relocate a building that does not fit, and it does not")
    for team in TEAMS:
        got = circle_edge_verdicts(eng, team)
        if got != want:
            out.append(f"{SEAT[team]}: {got}")
    return out


def test_mock_engine_does_what_its_footprint_statement_says():
    assert mock_statement_problems(mock_on_cannon_board()) == []


def test_plant_mock_engine_reporting_a_box_under_the_circle_statement_is_caught(monkeypatch):
    real = MockEngine.state
    monkeypatch.setattr(MockEngine, "state", lambda self: with_boxes(real(self)))
    assert any("reports a box" in p for p in mock_statement_problems(mock_on_cannon_board()))


def engine_footprint_model(engine: RustEngine) -> tuple[str, str]:
    """(model, evidence) for what the engine runs, read from what it reports."""
    tiles = catalogue_footprint_tiles(engine)
    boxes = sum(e.footprint is not None for e in engine.state().entities)
    if tiles or boxes:
        return "tile_box", f"footprint_tiles for {len(tiles)} cards, {boxes} boxes in its state"
    return engine.rules().footprint_model, "it reports no footprint; the ledger's model"


@needs_core
def test_mock_engine_states_the_footprint_model_the_engine_runs(rust):
    cannon = card_ids(rust)["Cannon"]
    rust.reset(1, board_setup(rust, "buildings", [[cannon] * DECK_SIZE] * 2))
    engine_model, evidence = engine_footprint_model(rust)
    stated = mock_engine_module.FOOTPRINT_MODEL
    if engine_model != stated:
        pytest.skip(
            f"{NOT_A_PASS}: MockEngine states {stated!r} (mock_engine.FOOTPRINT_MODEL) and "
            f"RustEngine runs {engine_model!r} ({evidence}). MockEngine's building verdicts "
            "and its building state are not the engine's while these differ."
        )
    assert [e.uid for e in rust.state().entities if e.footprint is not None] == []
    assert mock_statement_problems(mock_on_cannon_board()) == []
