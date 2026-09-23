"""Action parsing and action masking.

THE ACTION SPACE: ``Discrete(1 + 4 * 18 * 32) = 2305``
    Index 0 is NO-OP. Index ``1 + slot * 576 + ty * 18 + tx`` plays hand slot
    ``slot`` at the centre of tile ``(tx, ty)``, in the ACTING player's own frame
    (own king at the bottom), so Blue and Red share one policy head.

WHY THIS AND NOT THE ALTERNATIVES
    * Joint Discrete over (slot x tile) is the only shape in which a mask can say
      "Giant is legal here but Fireball is not" exactly. Legality is a property of
      the PAIR: elixir is per card, territory depends on placement type (spells go
      anywhere, a Goblin Barrel anywhere but water, a Log only where a troop may go
      but over buildings, buildings never into the pocket), footprints depend on
      card radius.
      sb3-contrib MaskablePPO applies a MultiDiscrete mask per dimension
      independently, so a MultiDiscrete([5, 18, 32]) head could only mask the
      marginals and would happily sample "Knight on the enemy king". An
      autoregressive factorised head (card, then position conditioned on card)
      CAN mask exactly, and is the better choice at scale, but it needs a custom
      policy; the joint Discrete works with stock MaskablePPO today and 2305
      logits is small.
    * Tile resolution, not half-tile. The tilemap is half-tile (36 x 64), and
      some real placements differ by half a tile (the bridge span is half-tile
      offset). Half-tile would be Discrete(9217): 4x the logits for placements
      that mostly differ by a quarter of a unit hitbox. ``HalfTileActionParser``
      is provided so that trade can be measured instead of argued.
    * No "hold" or timing sub-action: timing is expressed by choosing NO-OP on
      a decision step, at ``decision_ms`` granularity (see env.py).

HOW THE MASK IS COMPUTED, AND WHY INDEPENDENTLY OF THE ENGINE
    ``PlacementOracle`` recomputes legality from the state snapshot, arena and
    ``DeployRules`` with numpy grids. The engine enforces the same rules on its
    own code path (``Engine.check_deploy``). The two are kept deliberately
    separate: tests compare them exhaustively, so a silently wrong mask -- which
    trains an agent to want illegal moves, or never to find legal ones -- fails
    a test instead of a training run.

    A point on a half-cell boundary belongs to every cell it touches. A tile
    centre touches four half-cells, so a tile is legal iff its whole 2x2 block
    is. This closed-cell rule is what makes placement exactly invariant under
    the 180-degree seat rotation.

    TWO KINDS OF RULE, AS IN THE RUST ENGINE (arena.rs ``deploy_zone``)
    CELL rules (water, no-deploy, the river band closed to troops, buildings'
    own half) are grids over half-cells, and a point needs every cell it
    touches. POINT rules are tested on the point itself: the bodies already on
    the board, and the closed NoDeploySize rect of every alive enemy crown tower
    (``DeployRules``). The rect rule equals a cell rule only while every rect
    edge lies on a half-cell boundary (the shipped sizes do); testing the point
    keeps the mask right if a regenerated cards.json ever breaks that.

    A BUILDING IS ASKED A DIFFERENT QUESTION, AND IT IS NOT "DOES IT FIT HERE"
    A building stands on a square of TILES, and a tap that does not fit is not
    refused: the game moves the building to the nearest place it does fit. So
    for a building card the mask answers "will a tap here build anything",
    which stops depending on what is already on the board -- nothing can be in
    the way, because being in the way relocates rather than refuses. Only the
    cell rules remain. What a tap actually BUILDS, and where, is the engine's
    ``building_placement``, and the mask is not the place to ask it: an action
    space over 2 304 tiles cannot say "here, but two tiles left" anyway.

    Measured against the engine on 2026-09-22, tile centres, both seats, a board
    with buildings and towers standing: the cell rules alone reproduce
    ``check_deploy`` for a Cannon on all 576 cells, where the old
    body-overlap rule missed 18 of them. Before relocation shipped, those 18 were
    right; a building really was refused for touching another body.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
from gymnasium import spaces

from .protocol import (
    BIT_NO_DEPLOY,
    BIT_WATER,
    BLUE,
    EMPTY_CARD,
    HAND_SIZE,
    RED,
    Arena,
    BattleState,
    CardInfo,
    DeployCommand,
    DeployRules,
    Engine,
    EntityKind,
    Placement,
    TowerSlot,
    to_engine,
    to_own,
)

NOOP = 0
# How many ``point_grid`` results one oracle keeps. A battle reaches at most a few
# distinct keys per tick -- one per placement class per seat -- and the key changes
# whenever a building appears or a tower falls, so the useful window is the current
# tick and its neighbours. The cache is cleared wholesale rather than evicted one at
# a time: it is a within-step memo, not a long-lived store, and a clear costs less
# than tracking ages.
GRID_CACHE_SIZE = 64


class PlacementOracle:
    """Placement legality grids from a state snapshot. Engine frame internally."""

    def __init__(self, arena: Arena, rules: DeployRules, cards: Sequence[CardInfo]) -> None:
        if rules.footprint_model != "collision_radius_circle":
            raise NotImplementedError(rules.footprint_model)
        self.arena = arena
        self.rules = rules
        self.cards = list(cards)
        # True when the engine relocates a building whose box does not fit rather than
        # refusing it. Then nothing on the board can make a building tap illegal, so the
        # bodies already standing are not a rule for buildings at all.
        self.buildings_relocate = rules.illegal_building_tap == "relocate_first_fitting_ring"
        grid = np.asarray(arena.grid, dtype=np.int64)  # [hy, hx]
        self.water = (grid & BIT_WATER) != 0
        self.nodeploy = (grid & BIT_NO_DEPLOY) != 0
        hy_idx = np.arange(arena.hy)[:, None]
        lo, hi = arena.water_half_rows
        # The river band is the same half-rows in both teams' frames (the tilemap
        # is rotation-invariant), so one grid serves both.
        self.river_band = np.array(np.broadcast_to((hy_idx >= lo) & (hy_idx <= hi), grid.shape))
        self.own_half: list[np.ndarray] = []
        for team in (BLUE, RED):
            oy = hy_idx if team == BLUE else arena.hy - 1 - hy_idx
            self.own_half.append(np.array(np.broadcast_to(oy < lo, grid.shape)))
        # [owner team][TowerSlot] closed NoDeploySize rects at the arena's tower centres.
        self.tower_rects = [rules.tower_rects(arena, owner) for owner in (BLUE, RED)]
        self._points: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # ``point_grid`` memo. See ``grid_key`` for why it is safe and
        # ``GRID_CACHE_SIZE`` for why it is small.
        self._grids: dict[tuple[Any, ...], np.ndarray] = {}
        self.grid_hits = 0
        self.grid_misses = 0

    def _bodies_block(self, placement: int) -> bool:
        """Whether the bodies already on the board are a rule for this placement.

        Always for a troop. For a BUILDING only while the engine refuses a tap whose
        box does not fit; once it relocates instead, no board state can make a building
        tap illegal and including the bodies would mask legal cells away.
        """
        if placement == Placement.TROOP:
            return True
        return placement == Placement.BUILDING and not self.buildings_relocate

    # -- static + tower-dependent part, at half-cell resolution --------------

    def cell_grid(self, state: BattleState, team: int, placement: int) -> np.ndarray:
        """The CELL rules. Troop territory here is only 'not the river band'; the
        enemy tower rects are a point rule, applied in ``legal_points``.

        Per placement (the Rust core's ``Arena::deploy_zone`` by ``state.rs
        deploy_rule``): SPELL no cell rule; SPELL_NOT_ON_WATER water only (no
        no-deploy, no territory: the king block is a legal Goblin Barrel target);
        TROOP and ROLLING water, no-deploy and the river band; BUILDING water,
        no-deploy and own half.
        """
        a = self.arena
        del state  # nothing tower-dependent is a cell rule any more
        if placement == Placement.SPELL:
            return np.ones((a.hy, a.hx), dtype=bool)
        if placement == Placement.SPELL_NOT_ON_WATER:
            not_water: np.ndarray = ~self.water
            return not_water
        terr = self.own_half[team] if placement == Placement.BUILDING else ~self.river_band
        legal: np.ndarray = terr & ~self.water & ~self.nodeploy
        return legal

    def enemy_rects(self, state: BattleState, team: int) -> list[tuple[int, int, int, int]]:
        """The rects a ``team`` troop may not be placed in: every ALIVE enemy crown tower's."""
        enemy = 1 - team
        hp = state.players[enemy].tower_hp
        return [self.tower_rects[enemy][slot] for slot in TowerSlot if hp[slot] > 0]

    def legal_points(
        self, state: BattleState, team: int, card: CardInfo, px: np.ndarray, py: np.ndarray
    ) -> np.ndarray:
        """Legality of placing ``card`` at engine-frame points (px, py), any shape.

        Ignores elixir and hand. A point must lie strictly inside the arena, every
        half-cell it touches must pass ``cell_grid``, and it must pass the point
        rules (enemy tower rects for troops and rolling spells, footprints).
        """
        a = self.arena
        h = a.half_size
        px = np.asarray(px, dtype=np.int64)
        py = np.asarray(py, dtype=np.int64)
        inside = (px > 0) & (px < a.width) & (py > 0) & (py < a.height)
        cells = self.cell_grid(state, team, card.placement)
        ix1 = np.clip(px // h, 0, a.hx - 1)
        iy1 = np.clip(py // h, 0, a.hy - 1)
        ix0 = np.clip(np.where(px % h == 0, px // h - 1, px // h), 0, a.hx - 1)
        iy0 = np.clip(np.where(py % h == 0, py // h - 1, py // h), 0, a.hy - 1)
        ok: np.ndarray = (
            inside & cells[iy0, ix0] & cells[iy0, ix1] & cells[iy1, ix0] & cells[iy1, ix1]
        )
        if card.placement in (Placement.TROOP, Placement.ROLLING):
            for x0, y0, x1, y1 in self.enemy_rects(state, team):
                ok &= ~((px >= x0) & (px <= x1) & (py >= y0) & (py <= y1))
        if self._bodies_block(card.placement):
            extra = card.radius if card.placement == Placement.BUILDING else 0
            for e in state.entities:
                if e.kind == EntityKind.TROOP:
                    continue
                r = e.radius + extra
                ok &= (px - e.x) ** 2 + (py - e.y) ** 2 > r * r
        return ok

    def points(self, pitch_div: int) -> tuple[np.ndarray, np.ndarray]:
        """Engine-frame subtile coordinates of the candidate points, [ny, nx]."""
        if pitch_div not in self._points:
            a = self.arena
            pitch = a.subtile // pitch_div
            xs = np.arange(a.tiles_x * pitch_div, dtype=np.int64) * pitch + pitch // 2
            ys = np.arange(a.tiles_y * pitch_div, dtype=np.int64) * pitch + pitch // 2
            self._points[pitch_div] = (
                np.broadcast_to(xs[None, :], (ys.size, xs.size)),
                np.broadcast_to(ys[:, None], (ys.size, xs.size)),
            )
        return self._points[pitch_div]

    def grid_key(
        self, state: BattleState, team: int, card: CardInfo, pitch_div: int
    ) -> tuple[Any, ...] | None:
        """Everything ``point_grid`` reads, as a hashable key -- or None, do not cache.

        The grid is NOT a function of the card, and that is the whole point. It reads
        the placement class, the acting team, the pitch, which enemy crown towers are
        still standing, and the position and radius of every NON-troop entity (troops
        do not block a deploy). A building adds its own radius to each footprint, so
        that enters the key too; for a troop the term is zero, which is why one grid
        serves every troop card in a hand.

        That collapses the calls that actually happen. Per env step both seats build a
        mask over up to four hand slots and an observation over the OPPONENT's troop
        zone -- and Blue's ``enemy_troop_zone`` is the identical grid Red's mask needs.
        Measured on MockEngine: 2.66 ``point_grid`` calls per ``env.step`` before this,
        and the observation's own call was 45% of the time spent building it.

        Returns None for a placement whose grid this cannot key safely, so a future
        rule that reads something else is a cache MISS rather than a stale hit.
        """
        placement = card.placement
        if placement not in (
            Placement.TROOP,
            Placement.BUILDING,
            Placement.ROLLING,
            Placement.SPELL,
            Placement.SPELL_NOT_ON_WATER,
        ):
            return None
        rects = (
            tuple(self.enemy_rects(state, team))
            if placement in (Placement.TROOP, Placement.ROLLING)
            else ()
        )
        if self._bodies_block(placement):
            extra = card.radius if placement == Placement.BUILDING else 0
            blockers = tuple(
                sorted((e.x, e.y, e.radius) for e in state.entities if e.kind != EntityKind.TROOP)
            )
        else:
            extra, blockers = 0, ()
        return (int(placement), team, pitch_div, extra, rects, blockers)

    def point_grid(
        self, state: BattleState, team: int, card: CardInfo, pitch_div: int
    ) -> np.ndarray:
        """Engine-frame legality of placing ``card`` at each candidate point.

        pitch_div=1: tile centres [32, 18]. pitch_div=2: half-cell centres [64, 36].
        Ignores elixir and hand; those are applied per slot by the parser.

        The same rules as ``legal_points``, specialised to the two regular grids
        because this is the hot path (4 mask slots + 3 obs zones per seat per env
        step): a tile centre touches exactly its tile's 2x2 half-cells and a
        half-cell centre exactly one, and every point rule separates into 1-D x and
        y tests that broadcast. Measured 2026-09-13: routing this through the
        general ``legal_points`` cost 170/251 us per call (tile/half, Knight, all
        towers up, MockEngine opening board) and dropped env.step through the full
        Python stack from about 1 012/s to 753/s on the Rust engine; this
        specialisation measures 82/91 us on the same board.
        tests/test_rust_engine.py holds this path equal to ``legal_points``.

        MEMOISED on everything it reads (``grid_key``). The returned array is marked
        READ-ONLY, because callers share it: writing to it would change what another
        seat's mask sees. Both shipped callers already copy on their way out -- the
        mask reshapes into its own buffer, the observation casts to float32 -- so the
        flag is a guard against a future one, not a change to either.
        """
        key = self.grid_key(state, team, card, pitch_div)
        if key is not None:
            cached = self._grids.get(key)
            if cached is not None:
                self.grid_hits += 1
                return cached
            self.grid_misses += 1
        a = self.arena
        cells = self.cell_grid(state, team, card.placement)
        if pitch_div == 1:
            ok: np.ndarray = cells.reshape(a.tiles_y, a.half, a.tiles_x, a.half).all(axis=(1, 3))
        elif pitch_div == 2:
            ok = cells.copy()
        else:
            raise ValueError("pitch_div must be 1 or 2")
        if card.placement in (Placement.SPELL, Placement.SPELL_NOT_ON_WATER):
            return ok  # no point rule applies to either (cell_grid)
        px, py = self.points(pitch_div)
        xs, ys = px[0], py[:, 0]
        if card.placement in (Placement.TROOP, Placement.ROLLING):
            for x0, y0, x1, y1 in self.enemy_rects(state, team):
                iny = (ys >= y0) & (ys <= y1)
                inx = (xs >= x0) & (xs <= x1)
                if iny.any() and inx.any():
                    ok &= ~(iny[:, None] & inx[None, :])
        if self._bodies_block(card.placement):
            extra = card.radius if card.placement == Placement.BUILDING else 0
            for e in state.entities:
                if e.kind == EntityKind.TROOP:
                    continue
                r = e.radius + extra
                ok &= ((ys - e.y) ** 2)[:, None] + ((xs - e.x) ** 2)[None, :] > r * r
        if key is not None:
            ok.flags.writeable = False
            if len(self._grids) >= GRID_CACHE_SIZE:
                self._grids.clear()
            self._grids[key] = ok
        return ok


class ActionParser(ABC):
    """Agent action -> engine command, plus the legality mask for that action space."""

    def bind(self, engine: Engine) -> None:
        self.arena = engine.arena()
        self.cards = list(engine.cards())
        self.oracle = PlacementOracle(self.arena, engine.rules(), self.cards)

    @property
    @abstractmethod
    def space(self) -> spaces.Discrete: ...

    @abstractmethod
    def action_mask(self, state: BattleState, team: int) -> np.ndarray:
        """int8 array of shape (space.n,): 1 = the engine will accept this action now."""

    def mask_plane_shape(self) -> tuple[int, int, int] | None:
        """Shape the mask MINUS the no-op reshapes to, or None if it does not.

        ``obs.py`` hands the policy the mask twice: flat for the head, and as
        ``mask_planes`` for a convolutional trunk, which is only meaningful when
        the action space is a grid. A parser whose space is not one returns None
        and no ``mask_planes`` key appears in the observation.
        """
        return None

    def config(self) -> dict[str, object]:
        """Constructor state, JSON-able, for ``ClashParallelEnv.config()``."""
        return {}

    @abstractmethod
    def parse(self, action: int, state: BattleState, team: int) -> DeployCommand | None:
        """None means no-op."""

    def noop(self) -> int:
        return NOOP


#: What a BUILDING tap is allowed to do, as an action-space choice rather than a rule.
#:
#: ``any_tap`` is the shipped default and the engine's own answer: every tap the engine
#: accepts is offered, and the engine moves the building when its box does not fit where
#: you tapped.
#:
#: ``taps_where_the_building_stays`` offers only the taps that put the building on the
#: tile that was tapped. Measured 2026-09-22 on the compiled engine, both seats, a board
#: with buildings and towers standing: of the 240 tiles the default offers a Cannon, 124
#: put it on a tile the agent did not choose, and 73 of 240 for a Tesla. So under the
#: default a policy asks for one cell and gets another more often than not, and nothing
#: in its observation says which taps are which.
#:
#: This arm is lossless over the boards it was checked on: every landing tile reachable
#: by any tap is also reachable by a tap that stays put, so it removes aliases and no
#: placement. ``tests/test_building_tap_arms.py`` checks that per board rather than
#: trusting the sweep, because it is a property of the board and not of the engine.
BUILDING_TAP_ARMS = ("any_tap", "taps_where_the_building_stays")


class GridActionParser(ActionParser):
    """Discrete(1 + HAND_SIZE * ny * nx) over a regular grid of placement points.

    ``buildings`` picks what a building tap means; see ``BUILDING_TAP_ARMS``. It changes
    the ACTION SPACE and not the rules, so two parsers with different arms describe the
    same battles and a policy trained under one is not comparable to a policy trained
    under the other without saying so.
    """

    pitch_div = 1

    def __init__(self, buildings: str = "any_tap") -> None:
        if buildings not in BUILDING_TAP_ARMS:
            raise ValueError(f"buildings must be one of {BUILDING_TAP_ARMS}, not {buildings!r}")
        self.buildings = buildings
        self._engine: Engine | None = None

    def bind(self, engine: Engine) -> None:
        super().bind(engine)
        # Where a building LANDS is the engine's rule, and asking the engine is the
        # point: working it out here would be a second copy of that rule, drifting from
        # the first. BOTH arms need it. The default arm needs it because relocation does
        # not always rescue a tap: when nothing fits within the engine's search, the tap
        # is refused, and a mask built from the cell rules alone offers it anyway.
        if hasattr(engine, "building_placement") and self.oracle.buildings_relocate:
            self._engine = engine
        elif self.buildings == "taps_where_the_building_stays":
            raise NotImplementedError(
                f"{type(engine).__name__} cannot say where a building would land, so "
                f"the {self.buildings!r} arm cannot be built on it. It needs a "
                "building_placement(team, card_name, x, y) method returning the "
                "landing, or None when the tap is refused."
            )
        self.nx = self.arena.tiles_x * self.pitch_div
        self.ny = self.arena.tiles_y * self.pitch_div
        self.pitch = self.arena.subtile // self.pitch_div
        self.n_actions = 1 + HAND_SIZE * self.nx * self.ny
        self._space: spaces.Discrete = spaces.Discrete(self.n_actions)

    @property
    def space(self) -> spaces.Discrete:
        return self._space

    def mask_plane_shape(self) -> tuple[int, int, int] | None:
        """[hand slot, y, x]: the mask without index 0, in the acting seat's own frame.

        ``encode`` lays the space out as ``1 + slot * ny * nx + y * nx + x``, so
        ``mask[1:].reshape(HAND_SIZE, ny, nx)`` is that same mask with no
        arithmetic -- a view, not a copy.
        """
        return (HAND_SIZE, self.ny, self.nx)

    def config(self) -> dict[str, object]:
        return {
            "pitch_div": self.pitch_div,
            "n_actions": self.n_actions,
            "buildings": self.buildings,
        }

    def encode(self, slot: int, x_idx: int, y_idx: int) -> int:
        return 1 + slot * self.nx * self.ny + y_idx * self.nx + x_idx

    def decode(self, action: int) -> tuple[int, int, int]:
        a = int(action) - 1
        per = self.nx * self.ny
        slot, rest = divmod(a, per)
        y_idx, x_idx = divmod(rest, self.nx)
        return slot, x_idx, y_idx

    def action_mask(self, state: BattleState, team: int) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=np.int8)
        mask[NOOP] = 1
        if state.game_over:
            return mask
        player = state.players[team]
        per = self.nx * self.ny
        for slot, card_id in enumerate(player.hand):
            if card_id == EMPTY_CARD:
                continue
            card = self.cards[card_id]
            if player.elixir_milli < card.elixir * 1000:
                continue
            grid = self.oracle.point_grid(state, team, card, self.pitch_div)
            if team == RED:
                grid = grid[::-1, ::-1]  # engine frame -> Red's own frame
            if self._engine is not None and card.placement == Placement.BUILDING:
                grid = self.buildable(team, card, grid)
            mask[1 + slot * per : 1 + (slot + 1) * per] = grid.reshape(-1)
        return mask

    def buildable(self, team: int, card: CardInfo, legal: np.ndarray) -> np.ndarray:
        """Own-frame grid of the taps this arm offers for ``card``, asked of the engine.

        Both arms need the engine, for different reasons. ``any_tap`` needs it because
        relocation does NOT always rescue a tap: when nothing fits within the engine's
        search the tap is refused, and the cell rules alone cannot tell, so a mask built
        from them offers actions the engine turns down. Measured on a Cannon lattice
        filling one half, 15 buildings was enough: the engine refused every tap and the
        cell-rule mask offered all 960. ``taps_where_the_building_stays`` needs it to
        know where the building would land.

        Asks the engine, once per cell, where the building would land. It reads the
        engine's CURRENT battle, which is the state the caller is masking for; a parser
        handed a snapshot of some other battle would get a mask for the live one. The
        env masks the battle it just stepped, so this holds there, and
        ``tests/test_building_tap_arms.py`` grades the mask against the engine rather
        than assuming it.

        Not memoised. Measured 2026-09-22: 0.7 ms for a whole 576-tile grid against
        0.05 ms for the memoised legality mask, on an iteration whose collection is a
        fifth of its time. A memo here would have to key on every building and tower on
        the board, which the legality memo no longer does, and a memo keyed on too
        little is worse than none.
        """
        assert self._engine is not None
        stays = self.buildings == "taps_where_the_building_stays"
        out = np.zeros((self.ny, self.nx), dtype=bool)
        pitch, half = self.pitch, self.pitch // 2
        # Only cells the cell rules already allow are worth asking about: the engine
        # would refuse the rest for a reason this side already knows. On a Cannon that
        # is about 240 of 576 cells.
        for yi, xi in zip(*np.nonzero(legal), strict=True):
            if True:
                x, y = to_engine(self.arena, team, xi * pitch + half, yi * pitch + half)
                landed = self._engine.building_placement(team, card.name, x, y)
                if landed is None:
                    continue  # nothing fits within the engine's search: refused
                if not stays:
                    out[yi, xi] = True
                    continue
                ox, oy = to_own(self.arena, team, landed[0], landed[1])
                out[yi, xi] = (ox // self.arena.subtile, oy // self.arena.subtile) == (xi, yi)
        return out

    def parse(self, action: int, state: BattleState, team: int) -> DeployCommand | None:
        action = int(action)
        if action == NOOP:
            return None
        if not 0 < action < self.n_actions:
            raise ValueError(f"action {action} outside {self._space}")
        slot, xi, yi = self.decode(action)
        x_own = xi * self.pitch + self.pitch // 2
        y_own = yi * self.pitch + self.pitch // 2
        x, y = to_engine(self.arena, team, x_own, y_own)
        return DeployCommand(team=team, hand_slot=slot, x=x, y=y)


class TileActionParser(GridActionParser):
    """The default: Discrete(2305), tile centres."""

    pitch_div = 1


class HalfTileActionParser(GridActionParser):
    """Discrete(9217), half-cell centres. For measuring whether resolution matters."""

    pitch_div = 2


def mask_disagreements(
    engine: Engine, parser: GridActionParser, state: BattleState, team: int
) -> list[tuple[int, int, int]]:
    """Every action where the mask and ``engine.check_deploy`` disagree, as
    (action, mask value, engine status). ``state`` must be ``engine.state()``.

    Exhaustive over the whole action space. This is the check to run against any
    new engine (the Rust core included) before training on it.
    """
    mask = parser.action_mask(state, team)
    out = []
    for action in range(1, int(parser.space.n)):
        cmd = parser.parse(action, state, team)
        assert cmd is not None
        status = engine.check_deploy(cmd)
        if bool(mask[action]) != (status == 0):
            out.append((action, int(mask[action]), int(status)))
    return out
