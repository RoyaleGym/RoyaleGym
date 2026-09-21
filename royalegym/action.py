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
    touches. POINT rules are tested on the point itself: building footprints,
    and the closed NoDeploySize rect of every alive enemy crown tower
    (``DeployRules``). The rect rule equals a cell rule only while every rect
    edge lies on a half-cell boundary (the shipped sizes do); testing the point
    keeps the mask right if a regenerated cards.json ever breaks that.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

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
)

NOOP = 0


class PlacementOracle:
    """Placement legality grids from a state snapshot. Engine frame internally."""

    def __init__(self, arena: Arena, rules: DeployRules, cards: Sequence[CardInfo]) -> None:
        if rules.footprint_model != "collision_radius_circle":
            raise NotImplementedError(rules.footprint_model)
        self.arena = arena
        self.rules = rules
        self.cards = list(cards)
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
        if card.placement in (Placement.TROOP, Placement.BUILDING):
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
        """
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
        if card.placement in (Placement.TROOP, Placement.BUILDING):
            extra = card.radius if card.placement == Placement.BUILDING else 0
            for e in state.entities:
                if e.kind == EntityKind.TROOP:
                    continue
                r = e.radius + extra
                ok &= ((ys - e.y) ** 2)[:, None] + ((xs - e.x) ** 2)[None, :] > r * r
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

    @abstractmethod
    def parse(self, action: int, state: BattleState, team: int) -> DeployCommand | None:
        """None means no-op."""

    def noop(self) -> int:
        return NOOP


class GridActionParser(ActionParser):
    """Discrete(1 + HAND_SIZE * ny * nx) over a regular grid of placement points."""

    pitch_div = 1

    def bind(self, engine: Engine) -> None:
        super().bind(engine)
        self.nx = self.arena.tiles_x * self.pitch_div
        self.ny = self.arena.tiles_y * self.pitch_div
        self.pitch = self.arena.subtile // self.pitch_div
        self.n_actions = 1 + HAND_SIZE * self.nx * self.ny
        self._space: spaces.Discrete = spaces.Discrete(self.n_actions)

    @property
    def space(self) -> spaces.Discrete:
        return self._space

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
            mask[1 + slot * per : 1 + (slot + 1) * per] = grid.reshape(-1)
        return mask

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
