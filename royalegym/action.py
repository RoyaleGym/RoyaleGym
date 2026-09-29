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
      but over buildings, buildings never into the pocket, a Miner anywhere on land
      but not on a building, a Mirror wherever the card it copies may go), footprints
      depend on card radius.
      sb3-contrib MaskablePPO applies a MultiDiscrete mask per dimension
      independently, so a MultiDiscrete([5, 18, 32]) head could only mask the
      marginals and would happily sample "Knight on the enemy king". An
      autoregressive factorised head (card, then position conditioned on card)
      CAN mask exactly, and is the better choice at scale, but it needs a custom
      policy; the joint Discrete works with stock MaskablePPO today and 2305
      logits is small.
    * Tile resolution, not half-tile. The tilemap is half-tile (36 x 64).
      Half-tile would be Discrete(9217): 4x the logits, and under placement.TAP_SNAP =
      client16402_tile_centre the engine snaps every tap to its tile centre, spells
      too, so half-tile taps land where tile taps do. Measured by the docs session on
      RoyaleSim 0d0ccd6: a Knight's 992 legal half-tile moves put it on 215 points,
      against the tile grid's 213; a Fireball's 576 either way. ``HalfTileActionParser``
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
    (``DeployRules``). A troop's body is judged where the tap resolves -- its tile
    centre under placement.TAP_SNAP -- and an own building's, tower's or live
    bottle's tile does not refuse a troop tap, which the engine moves off it
    (``PlacementOracle.own_tower_zone``), unless the move finds nowhere to go
    (``GridActionParser.moved_taps_that_land``). The rect rule equals a cell rule
    only while every rect edge lies on a half-cell boundary (the shipped sizes do);
    testing the point keeps the mask right if a regenerated cards.json ever breaks
    that.

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
    ABILITY_BUTTONS,
    AS_TOWER_TAP,
    BIT_NO_DEPLOY,
    BIT_WATER,
    BLUE,
    BOTTLE_RELOCATE,
    EMPTY_CARD,
    HALF_OPEN_RELOCATE,
    HAND_SIZE,
    RED,
    TILE_CENTRE_SNAP,
    TROOP_RELOCATION,
    Arena,
    BattleState,
    CardInfo,
    DeployCommand,
    DeployRules,
    DeployStatus,
    Engine,
    EntityKind,
    Placement,
    SpellMotion,
    TowerSlot,
    ability_row,
    slot_cost,
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
        # A TUNNEL card is judged by its KIND's footprint rule, a troop's or a building's,
        # so a catalogue that does not say which cannot be masked.
        unkinded = [
            c.name
            for c in self.cards
            if c.placement == Placement.TUNNEL and c.card_kind not in ("TROOP", "BUILDING")
        ]
        if unkinded:
            raise ValueError(
                f"TUNNEL cards {unkinded} state no troop or building card_kind, and the mask "
                "judges a tunnelling card by its kind's footprint rule"
            )
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
        # placement.TROOP_TOWER_TAPS (DeployRules.troop_tower_taps). Under the half-open arm
        # the core judges a TROOP (not a rolling spell) in two parts:
        #   the zone, at the tap: a point touching a no-deploy cell is refused only if it lies
        #     in a king block HALF-OPEN, [x0, x1) x [y0, y1), in the arena's frame, or touches
        #     a no-deploy cell whose centre is outside both king blocks (arena.rs
        #     deploy_zone_king_half_open). So the king-block cells stop refusing at the cell
        #     level and the half-open rect refuses at the point level instead;
        #   the bodies, where the troop will stand: a tap whose tile overlaps an alive OWN
        #     crown tower's placement box is moved off it (state.rs
        #     relocate_off_own_crown_tower), unless placement.ILLEGAL_TAP is "refuse", so no
        #     body blocks it.
        self.king_half_open = rules.troop_tower_taps == HALF_OPEN_RELOCATE
        # THE RELOCATIONS (state.rs resolve_point), for a card placed as a troop
        # (``_places_as_troop``). A tap whose snapped one-tile box shares area with an alive
        # OWN crown tower's box (the half-open arm) or an alive OWN building's box
        # (placement.TROOP_BUILDING_TAPS = as_tower_tap), or whose tile is the tile of a live
        # bottle of its own side (placement.LIVE_BOTTLE_TAPS, ``own_live_bottles``), is moved off
        # it, unless placement.ILLEGAL_TAP is "refuse", so no body there blocks it.
        self.buildings_move_taps = rules.troop_building_taps == AS_TOWER_TAP
        self.bottles_move_taps = rules.live_bottle_taps == BOTTLE_RELOCATE
        self.troop_taps_relocate = rules.illegal_building_tap != "refuse" and (
            self.king_half_open or self.buildings_move_taps or self.bottles_move_taps
        )
        # With ANY relocation on, the core judges the body where the tap resolves: under
        # placement.TAP_SNAP = client16402_tile_centre at the centre of the tile the tap is in,
        # in the ARENA's frame. The zone is still judged at the tap itself.
        any_relocation = self.king_half_open or self.buildings_move_taps or self.bottles_move_taps
        self.bodies_at_tile_centre = any_relocation and rules.tap_snap == TILE_CENTRE_SNAP
        if self.king_half_open and len(arena.king_blocks) != 2:
            raise ValueError(
                "placement.TROOP_TOWER_TAPS is half-open but the arena states no king blocks"
            )
        king_cells = np.zeros_like(self.nodeploy)
        if self.king_half_open:
            h = arena.half_size
            cx = np.arange(arena.hx) * h + h // 2
            cy = np.arange(arena.hy) * h + h // 2
            for x0, y0, x1, y1 in arena.king_blocks:
                rows, cols = (cy >= y0) & (cy <= y1), (cx >= x0) & (cx <= x1)
                king_cells |= rows[:, None] & cols[None, :]
        # The no-deploy cells a troop's CELL rule still refuses: all of them, or under the
        # half-open arm all but the king blocks' own (the point rule takes those).
        self.troop_nodeploy = self.nodeploy & ~king_cells
        self._points: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # ``point_grid`` memo. See ``grid_key`` for why it is safe and
        # ``GRID_CACHE_SIZE`` for why it is small.
        self._grids: dict[tuple[Any, ...], np.ndarray] = {}
        self.grid_hits = 0
        self.grid_misses = 0
        # The blockers part of ``grid_key`` for the last state seen: the state OBJECT and
        # its sorted blocker tuple. See ``_blockers``.
        self._blocker_memo: tuple[BattleState | None, tuple[tuple[int, int, int], ...]] = (
            None,
            (),
        )

    def _blockers(self, state: BattleState) -> tuple[tuple[int, int, int], ...]:
        """Every non-troop entity's (x, y, radius), sorted: what ``grid_key`` keys bodies on.

        Worked out once per state object instead of once per call. Masks and observations
        ask for it up to five times per seat per step, all for one state. Keyed on the
        OBJECT, which is safe because ``BattleState`` is frozen and every engine's
        ``state()`` builds a new one. It holds that object, so its id cannot be reused. A
        caller that edits a state's entity LIST in place and asks again would get the old
        answer. Nothing here does that; build a new state with ``msgspec.structs.replace``.
        """
        seen, key = self._blocker_memo
        if seen is not state:
            # Owner, kind, life and box too: under the half-open arm an ALIVE OWN crown
            # tower's box is a rule of its own (``own_tower_zone``), not only a body.
            key = tuple(
                sorted(
                    (e.x, e.y, e.radius, int(e.kind), e.team, e.hp > 0, e.footprint or ())
                    for e in state.entities
                    if e.kind != EntityKind.TROOP
                )
            )
            self._blocker_memo = (state, key)
        return key

    def own_tower_boxes(self, state: BattleState, team: int) -> list[tuple[int, int, int, int]]:
        """The placement box (engine frame, closed) of every ALIVE crown tower ``team`` owns."""
        out = []
        for e in state.entities:
            if e.team != team or e.hp <= 0:
                continue
            if e.kind not in (EntityKind.KING_TOWER, EntityKind.PRINCESS_TOWER):
                continue
            if e.footprint is None:
                raise ValueError(
                    "placement.TROOP_TOWER_TAPS is half-open, which needs each crown tower's "
                    "placement box, and this engine reports none"
                )
            out.append(e.footprint)
        return out

    def own_building_boxes(self, state: BattleState, team: int) -> list[tuple[int, int, int, int]]:
        """The placement box (engine frame, closed) of every ALIVE building ``team`` owns."""
        out = []
        for e in state.entities:
            if e.team != team or e.hp <= 0 or e.kind != EntityKind.BUILDING:
                continue
            if e.footprint is None:
                raise ValueError(
                    "placement.TROOP_BUILDING_TAPS is as_tower_tap, which needs each building's "
                    "placement box, and this engine reports none"
                )
            out.append(e.footprint)
        return out

    @staticmethod
    def own_live_bottles(state: BattleState, team: int) -> list[tuple[int, int]]:
        """The points of the live bottles ``team`` owns: spell objects standing out a positive
        fuse (a Rage's bottle, a Lumberjack's death bottle), on the board from the cast to the
        release. A zero fuse (the Goblin Curse's area that makes an area) is no bottle; the
        engine reports it with no fuse left, so it is left out by its ``delay_ticks``. Under
        spells.SUMMON_FUSE_START = death_bomb_flight, which does not ship, a bottle can stand one
        tick at zero, and that tick is missed."""
        return [
            (s.x, s.y)
            for s in state.spells
            if s.team == team and s.motion == SpellMotion.FUSE and s.delay_ticks > 0
        ]

    def own_tower_zone(
        self, state: BattleState, team: int, px: np.ndarray, py: np.ndarray
    ) -> np.ndarray:
        """Where a ``team`` troop tap is moved off an own crown tower (the half-open arm), an
        own building (placement.TROOP_BUILDING_TAPS = as_tower_tap) or an own live bottle's tile
        (placement.LIVE_BOTTLE_TAPS = client16402_relocate, state.rs
        relocate_off_own_live_bottle), so no body blocks it.

        The core snaps the tap to a one-tile box, floored in the PLACER's frame
        (placement.SNAP_EVEN_CORNER = placer_frame: in the own frame, then back) or in the
        arena's (absolute), and moves the troop when that box shares positive area with the
        placement box of an alive own crown tower or building (state.rs
        relocate_off_own_crown_tower). The two frames pick different tiles only for a point on
        a tile edge, which no point the mask asks about is. Broadcasts over px, py.
        """
        a = self.arena
        t = a.subtile
        own = self.rules.snap_even_corner == "placer_frame" and team != BLUE
        fx = a.width - px if own else px
        fy = a.height - py if own else py
        cx = (fx // t) * t + t // 2
        cy = (fy // t) * t + t // 2
        if own:
            cx, cy = a.width - cx, a.height - cy
        lo_x, hi_x, lo_y, hi_y = cx - t // 2, cx + t // 2, cy - t // 2, cy + t // 2
        zone: np.ndarray = np.zeros(np.broadcast(px, py).shape, dtype=bool)
        boxes = self.own_tower_boxes(state, team) if self.king_half_open else []
        if self.buildings_move_taps:
            boxes += self.own_building_boxes(state, team)
        for x0, y0, x1, y1 in boxes:
            zone |= (lo_x < x1) & (x0 < hi_x) & (lo_y < y1) & (y0 < hi_y)
        if self.bottles_move_taps:
            # A bottle's tile is the tile its point is on, snapped as a tap is; two one-tile
            # boxes on that grid share area only when they are the same tile.
            for bx, by in self.own_live_bottles(state, team):
                fbx = a.width - bx if own else bx
                fby = a.height - by if own else by
                bcx, bcy = (fbx // t) * t + t // 2, (fby // t) * t + t // 2
                if own:
                    bcx, bcy = a.width - bcx, a.height - bcy
                zone |= (cx == bcx) & (cy == bcy)
        return zone

    @staticmethod
    def _troop_laws(card: CardInfo) -> bool:
        """Whether placement.TROOP_TOWER_TAPS's laws can judge this card: a TROOP placement
        of a card whose KIND is troop. The core applies them to CardKind::Troop only
        (state.rs check_position), and gives a spell with a troop's deploy rule (Heal) a
        troop's placement code, so the placement alone would give that spell the laws."""
        return card.placement == Placement.TROOP and card.card_kind in (None, "TROOP")

    @staticmethod
    def _building_like(card: CardInfo) -> bool:
        """Whether ``card`` is judged by a building's footprint: a BUILDING, or a TUNNEL
        card whose kind is building (the Goblin Drill, placed on its building's box by
        ``building_placement``, state.rs check_position)."""
        return card.placement == Placement.BUILDING or (
            card.placement == Placement.TUNNEL and card.card_kind == "BUILDING"
        )

    def _places_as_troop(self, card: CardInfo) -> bool:
        """Whether a tap of ``card`` takes the relocations a troop tap takes (state.rs
        places_as_troop): a card whose KIND is troop, the Miner as much as a Knight, and under
        placement.SPELL_AS_DEPLOY_TAPS = troop_relocation a spell placed as a troop that may
        not stand on a building (a SPELL kind with a troop's placement: the Heal)."""
        troop = card.card_kind in (None, "TROOP") and card.placement in (
            Placement.TROOP,
            Placement.TUNNEL,
        )
        spell = (
            self.rules.spell_as_deploy_taps == TROOP_RELOCATION
            and card.card_kind == "SPELL"
            and card.placement == Placement.TROOP
        )
        return troop or spell

    def _moved_off_own_tower(self, card: CardInfo) -> bool:
        """Whether a tap of ``card`` on an own crown tower or building is MOVED off it, so no
        body there blocks it (``own_tower_zone``)."""
        return self.troop_taps_relocate and self._places_as_troop(card)

    def _bodies_at_tile_centre(self, card: CardInfo) -> bool:
        """Whether ``card``'s body is judged at its tap's tile centre rather than the tap:
        placement.TAP_SNAP = client16402_tile_centre with any relocation on, for a card placed
        as a troop (state.rs check_position resolves the point only for those)."""
        return self.bodies_at_tile_centre and self._places_as_troop(card)

    def _card_bodies_block(self, card: CardInfo) -> bool:
        """``_bodies_block`` for a card: a TUNNEL card takes its kind's answer."""
        if card.placement == Placement.TUNNEL:
            like = Placement.BUILDING if self._building_like(card) else Placement.TROOP
            return self._bodies_block(like)
        return self._bodies_block(card.placement)

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

    def cell_grid(
        self, state: BattleState, team: int, placement: int, troop_laws: bool = True
    ) -> np.ndarray:
        """The CELL rules. Troop territory here is only 'not the river band'; the
        enemy tower rects are a point rule, applied in ``legal_points``.

        Per placement (the Rust core's ``Arena::deploy_zone`` by ``state.rs
        deploy_rule``): SPELL no cell rule; SPELL_NOT_ON_WATER water only (no
        no-deploy, no territory: the king block is a legal Goblin Barrel target);
        TROOP and ROLLING water, no-deploy and the river band; BUILDING water,
        no-deploy and own half; TUNNEL water only, as SPELL_NOT_ON_WATER (the
        no-deploy strips and the enemy half are its ground). A MIRROR has no grid of its
        own: it is placed as the card it copies, which is what to ask with.
        """
        a = self.arena
        del state  # nothing tower-dependent is a cell rule any more
        if placement == Placement.MIRROR:
            raise ValueError(
                "a Mirror is placed as the card it copies (match.MIRROR_PLACEMENT): ask "
                "with the player's mirror_target card"
            )
        if placement == Placement.SPELL:
            return np.ones((a.hy, a.hx), dtype=bool)
        if placement in (Placement.SPELL_NOT_ON_WATER, Placement.TUNNEL):
            not_water: np.ndarray = ~self.water
            return not_water
        terr = self.own_half[team] if placement == Placement.BUILDING else ~self.river_band
        troop = placement == Placement.TROOP and troop_laws
        nodeploy = self.troop_nodeploy if troop else self.nodeploy
        legal: np.ndarray = terr & ~self.water & ~nodeploy
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
        cells = self.cell_grid(state, team, card.placement, self._troop_laws(card))
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
        if self.king_half_open and self._troop_laws(card):
            for x0, y0, x1, y1 in self.arena.king_blocks:
                ok &= ~((px >= x0) & (px < x1) & (py >= y0) & (py < y1))
        if self._card_bodies_block(card):
            extra = card.radius if self._building_like(card) else 0
            bx, by = px, py
            if self._bodies_at_tile_centre(card):
                t = a.subtile
                bx, by = (px // t) * t + t // 2, (py // t) * t + t // 2
            clear = np.ones(ok.shape, dtype=bool)
            for e in state.entities:
                if e.kind == EntityKind.TROOP:
                    continue
                r = e.radius + extra
                clear &= (bx - e.x) ** 2 + (by - e.y) ** 2 > r * r
            if self._moved_off_own_tower(card):
                # From the snapped point: the core moves the tap it has already snapped.
                clear |= self.own_tower_zone(state, team, bx, by)
            ok &= clear
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
            blockers = self._blockers(state)
        else:
            extra, blockers = 0, ()
        # The own live bottles, which are spells and not entities, so not in ``blockers``.
        bottles = (
            tuple(sorted((s.team, s.x, s.y, s.delay_ticks > 0) for s in state.spells
                         if s.motion == SpellMotion.FUSE))
            if self.bottles_move_taps
            else ()
        )
        return (
            int(placement), team, pitch_div, extra, rects, blockers, self._troop_laws(card), bottles
        )

    def point_grid(
        self, state: BattleState, team: int, card: CardInfo, pitch_div: int, moves: bool = True
    ) -> np.ndarray:
        """Engine-frame legality of placing ``card`` at each candidate point.

        ``moves=False``: as if no tap were moved off an own tower, building or bottle, so a
        body under the tap refuses it. The parser compares the two to find the taps that are
        legal only because they are moved (``GridActionParser.moved_taps_that_land``).

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
        if key is not None and not moves:
            key = (*key, "no moves")
        if key is not None:
            cached = self._grids.get(key)
            if cached is not None:
                self.grid_hits += 1
                return cached
            self.grid_misses += 1
        a = self.arena
        cells = self.cell_grid(state, team, card.placement, self._troop_laws(card))
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
        if self.king_half_open and self._troop_laws(card):
            for x0, y0, x1, y1 in self.arena.king_blocks:
                iny = (ys >= y0) & (ys < y1)
                inx = (xs >= x0) & (xs < x1)
                if iny.any() and inx.any():
                    ok &= ~(iny[:, None] & inx[None, :])
        if self._card_bodies_block(card):
            extra = card.radius if self._building_like(card) else 0
            bxs, bys = xs, ys
            if self._bodies_at_tile_centre(card):
                t = a.subtile
                bxs, bys = (xs // t) * t + t // 2, (ys // t) * t + t // 2
            clear = np.ones(ok.shape, dtype=bool)
            for e in state.entities:
                if e.kind == EntityKind.TROOP:
                    continue
                r = e.radius + extra
                clear &= ((bys - e.y) ** 2)[:, None] + ((bxs - e.x) ** 2)[None, :] > r * r
            if moves and self._moved_off_own_tower(card):
                # From the snapped point: the core moves the tap it has already snapped.
                clear |= self.own_tower_zone(state, team, bxs[None, :], bys[:, None])
            ok &= clear
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
#: put it on a tile the agent did not choose, and 73 of 240 for a Tesla.
#:
#: HOW MUCH THAT COSTS THE AGENT DEPENDS ON WHICH QUESTION YOU ASK, and until 2026-09-23
#: the sentence here answered a different one from the figures above it. It read "a policy
#: asks for one cell and gets another more often than not" -- true of the 124/240, and a
#: reader takes it to mean the placement was LOST. Three criteria, over all 11 building
#: cards, every tile of the board, build d872d792711934c2
#: (``examples/measure_building_relocation.py`` re-derives this):
#:
#:     criterion                              3x3 (10 cards)   2x2 (Tesla)
#:     centre != tapped point                        51.7%        100.0%
#:     centre's tile != tapped tile                  51.7%         30.4%
#:     footprint does NOT cover chosen tile          15.0%         10.0%
#:
#: The top row is worthless and is recorded so nobody measures it again: an EVEN footprint
#: snaps to a tile CORNER, so its centre can never sit on a tile centre, and 100% is
#: geometry rather than relocation. The middle row is what this arm filters on and what the
#: 240-tile sweep counted. The bottom row is the one that bears on credit assignment,
#: because a 3x3 shifted by one tile still STANDS ON the tile the agent chose. So the agent
#: loses its chosen tile about 15% of the time, not "more often than not"; the aliasing is
#: real and under a third the size the old sentence implied. What is unqualified is the
#: last clause: nothing in the observation says which taps are which.
#:
#: RELOCATION DEPENDS ONLY ON THE FOOTPRINT, not on the card. All ten 3x3 buildings agree
#: to the decimal on all three criteria; Tesla differing is the control that shows the card
#: reaches the engine at all. Treat this as a 3x3-vs-2x2 property and pool cards freely.
#:
#: MEASURE IT ON EVERY TILE OR NOT AT ALL. A coarse lattice over the agent's own half gave
#: 47.5% and 39.0% where the full board gives 51.7% and 30.4%, and the narrowed gap was
#: briefly read as evidence that board CROWDING drives relocation. It does not: the full
#: scan above, on a near-empty board, reproduces the busy-board sweep exactly. The lattice
#: was the entire difference, and the hypothesis it suggested was about the sampling.
#:
#: This arm is lossless over the boards it was checked on: every landing tile reachable
#: by any tap is also reachable by a tap that stays put, so it removes aliases and no
#: placement. ``tests/test_building_tap_arms.py`` checks that per board rather than
#: trusting the sweep, because it is a property of the board and not of the engine. That
#: file was checked on 2026-09-23 for sensitivity rather than assumed: a plant making the
#: arm lossy failed 8 of its 36 tests, and a blind control perturbing only a diagnostic
#: counter left all 36 green.
#:
#: LOSSLESS IS ABOUT LANDING TILES, NOT ABOUT TAPS, and the difference is large enough to
#: matter when choosing an arm. On the board above the honest arm offers a Cannon 116 of
#: 240 taps; of the 124 it drops, 88 would still have left the building standing on the
#: tile the agent chose. Tesla drops only 73 of 240. Every landing tile stays reachable,
#: but a 3x3 loses over half its ways to ask for one and a 2x2 loses under a third, so the
#: arms differ in mask density asymmetrically BETWEEN FOOTPRINTS. A behaviour change under
#: this arm therefore has two candidate causes, not one.
BUILDING_TAP_ARMS = ("any_tap", "taps_where_the_building_stays")


class GridActionParser(ActionParser):
    """Discrete(1 + HAND_SIZE * ny * nx) over a regular grid of placement points.

    ``buildings`` picks what a building tap means; see ``BUILDING_TAP_ARMS``. It changes
    the ACTION SPACE and not the rules, so two parsers with different arms describe the
    same battles and a policy trained under one is not comparable to a policy trained
    under the other without saying so.
    """

    pitch_div = 1

    def __init__(self, buildings: str = "any_tap", ability_buttons: bool = False) -> None:
        if buildings not in BUILDING_TAP_ARMS:
            raise ValueError(f"buildings must be one of {BUILDING_TAP_ARMS}, not {buildings!r}")
        self.buildings = buildings
        # OPT-IN: one more action per ability button after the tile ones, action n_tile + k
        # pressing button k. Off, the space is what every policy so far was trained on.
        self.ability_buttons = bool(ability_buttons)
        # How many: the bound engine's count (``bind``), heroes' and champions' buttons.
        self.n_buttons = ABILITY_BUTTONS if self.ability_buttons else 0
        self._engine: Engine | None = None
        self._judge: Engine | None = None

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
        self.n_tile_actions = 1 + HAND_SIZE * self.nx * self.ny
        if self.ability_buttons:
            self.n_buttons = int(getattr(engine, "ability_button_count", ABILITY_BUTTONS))
        self.n_actions = self.n_tile_actions + self.n_buttons
        self._space: spaces.Discrete = spaces.Discrete(self.n_actions)
        # ``buildable`` memo. Keyed on everything relocation reads; see that method.
        self._buildable: dict[tuple[Any, ...], np.ndarray] = {}
        self.buildable_hits = 0
        self.buildable_misses = 0
        # Whether a MOVED troop tap lands is the engine's too (``moved_taps_that_land``).
        self._judge = engine if self.oracle.troop_taps_relocate else None
        self._landed: dict[tuple[Any, ...], np.ndarray] = {}
        self.landed_hits = 0
        self.landed_misses = 0

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
            # Only when on, so a parser without buttons reports what it always did.
            **({"ability_buttons": True} if self.ability_buttons else {}),
        }

    def button_of(self, action: int) -> int | None:
        """The ability button ``action`` presses, or None for the no-op or a tile action."""
        k = int(action) - self.n_tile_actions
        return k if 0 <= k < self.n_buttons else None

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
        # THE OPENING LOCKOUT, asked of the engine's rules rather than assumed. A match
        # refuses every deploy for its first `deploy_lockout_ticks` ticks, and a mask that
        # does not know that offers all four cards while the engine refuses all four --
        # which is not a cosmetic disagreement. `IllegalActionPenalty` fires when a seat
        # commands and nothing of its appears, so a policy would be punished for obeying
        # its own mask on the opening steps of EVERY battle, and would have to learn from
        # that penalty a rule the mask could simply have told it.
        #
        # 0 disables it, which is both the default for an engine that states nothing and a
        # real calibration arm, so this needs no special case for MockEngine.
        if state.tick < self.oracle.rules.deploy_lockout_ticks:
            return mask
        player = state.players[team]
        per = self.nx * self.ny
        # A button is pressable when the engine calls it available (a living hero or
        # champion behind it, off cooldown, not mid-ability), a hero's charge is unspent and
        # the bar can pay: the engine's own check (state.rs check_ability_button), read off
        # the state by column, whichever card the button is.
        for k, row in enumerate(player.abilities[: self.n_buttons]):
            b = ability_row(row)
            usable = b.available and not b.spent and player.elixir_milli >= b.cost * 1000
            mask[self.n_tile_actions + k] = int(usable)
        for slot, card_id in enumerate(player.hand):
            if card_id == EMPTY_CARD:
                continue
            card = self.cards[card_id]
            # The engine's price for the slot where it states one (a Mirror costs the card
            # it copies plus its own), -1 when no play of it resolves.
            cost = slot_cost(player, slot, card)
            if cost < 0 or player.elixir_milli < cost * 1000:
                continue
            if card.placement == Placement.MIRROR:
                # Placed exactly as the card it copies (match.MIRROR_PLACEMENT); with
                # nothing to copy the engine refuses it everywhere (NOTHING_TO_MIRROR).
                if player.mirror_target < 0:
                    continue
                card = self.cards[player.mirror_target]
            grid = self.oracle.point_grid(state, team, card, self.pitch_div)
            if team == RED:
                grid = grid[::-1, ::-1]  # engine frame -> Red's own frame
            if self._engine is not None and self.oracle._building_like(card):
                grid = self.buildable(state, team, card, grid)
            if self._judge is not None and self.oracle._moved_off_own_tower(card):
                grid = self.moved_taps_that_land(state, team, slot, card, grid)
            mask[1 + slot * per : 1 + (slot + 1) * per] = grid.reshape(-1)
        return mask

    def moved_taps_that_land(
        self, state: BattleState, team: int, slot: int, card: CardInfo, legal: np.ndarray
    ) -> np.ndarray:
        """Own-frame ``legal`` without the MOVED troop taps the engine would still refuse.

        A troop tap on an own crown tower, an own building or an own live bottle's tile is moved
        off it (``PlacementOracle.own_tower_zone``), so the mask offers it where a body would
        otherwise refuse it. But the move searches a bounded ring of tiles for one that fits,
        and when none does the tap stays where it was and the body under it refuses it (state.rs
        ring_nearest_fit). Measured on a board tiled with own Cannons (2026-09-28): 156 taps a
        seat offered and refused OCCUPIED, each a DeployRefused in a training run. So the engine
        is asked about exactly those cells, the ones legal only because the tap is moved, as
        ``buildable`` asks it where a building lands: the move is its rule, and a copy here
        would drift from it.

        MEMOISED on what the move reads: the team, the card's placement and kind (every troop
        of one placement takes the same territory and the same footprint test, so one answer
        serves a whole hand of them; the Miner's and the Heal's classes are their own), the
        pitch, every non-troop body (the ring's fit is off every building's box, and off the
        troop's territory, which no enemy tower changes) and the own live bottles, and the
        cells asked about. Troops do not enter it. It reads the engine's CURRENT battle, as
        ``buildable`` does.
        """
        assert self._judge is not None
        stay = self.oracle.point_grid(state, team, card, self.pitch_div, moves=False)
        if team == RED:
            stay = stay[::-1, ::-1]
        moved = legal & ~stay
        if not moved.any():
            return legal
        bottles = tuple(sorted(self.oracle.own_live_bottles(state, team)))
        key = (
            team,
            card.placement,
            card.card_kind,
            self.pitch_div,
            self.oracle._blockers(state),
            bottles,
            moved.tobytes(),
        )
        hit = self._landed.get(key)
        if hit is None:
            self.landed_misses += 1
            hit = np.zeros_like(moved)
            pitch, half = self.pitch, self.pitch // 2
            for yi, xi in zip(*np.nonzero(moved), strict=True):
                x, y = to_engine(self.arena, team, xi * pitch + half, yi * pitch + half)
                status = self._judge.check_deploy(DeployCommand(team, slot, x, y))
                hit[yi, xi] = status == DeployStatus.OK
            if len(self._landed) >= GRID_CACHE_SIZE:
                self._landed.clear()
            hit.flags.writeable = False
            self._landed[key] = hit
        else:
            self.landed_hits += 1
        out: np.ndarray = legal & (~moved | hit)
        return out

    def buildable(
        self, state: BattleState, team: int, card: CardInfo, legal: np.ndarray
    ) -> np.ndarray:
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

        MEMOISED, and it has to be. Without a memo this was 58% of ``env.step`` in a
        profile of 400 real steps: one engine call per offered cell, 96 000 of them,
        because a hand holding a building pays it every step for both seats. The first
        measurement missed that entirely, because the hand it sampled held no building.

        The key is everything relocation reads and nothing else: the acting team, the
        card's own size, and every body that can block a box, which is the buildings and
        the ALIVE towers. Not the tick, not elixir, not troops -- a troop cannot block a
        building (``DeployRules``), so including it would miss the memo on every step
        for nothing. Buildings and towers change rarely, so this hits across steps rather
        than only across the two seats of one step.
        """
        assert self._engine is not None
        blockers = tuple(
            sorted(
                (e.x, e.y, e.radius, int(e.kind))
                for e in state.entities
                if e.kind != EntityKind.TROOP
            )
        )
        # ``legal`` is in the key, not assumed constant. The grid below is filled only
        # at cells ``legal`` allows, so a hit against a DIFFERENT legal set would return
        # False for cells never asked about: an under-offering mask, silently. It is
        # constant for a building today, since the cell rules are static, but that is an
        # invariant of another function and not one to bet a wrong mask on.
        key = (
            team,
            card.name,
            self.pitch_div,
            self.buildings,
            blockers,
            legal.tobytes(),
        )
        hit = self._buildable.get(key)
        if hit is not None:
            self.buildable_hits += 1
            return legal & hit
        self.buildable_misses += 1
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
                snap = self.oracle.rules.snap_even_corner
                out[yi, xi] = landing_tile(self.arena, snap, team, *landed[:2]) == (xi, yi)
        if len(self._buildable) >= GRID_CACHE_SIZE:
            self._buildable.clear()
        out.flags.writeable = False
        self._buildable[key] = out
        return legal & out

    def parse(self, action: int, state: BattleState, team: int) -> DeployCommand | None:
        action = int(action)
        if action == NOOP:
            return None
        if not 0 < action < self.n_actions:
            raise ValueError(f"action {action} outside {self._space}")
        button = self.button_of(action)
        if button is not None:
            return DeployCommand(team=team, hand_slot=HAND_SIZE + button, x=0, y=0)
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


def landing_tile(
    arena: Arena, snap_even_corner: str, team: int, x: int, y: int
) -> tuple[int, int]:
    """The own-frame TILE a building whose centre the engine put at engine-frame (x, y)
    stands on, for ``team``.

    An odd box's centre (a 3x3 Cannon's) is a tile centre, the same tile in any frame. An
    even box's (a 2x2 Tesla's) is a tile CORNER, and of the four tiles meeting there it
    stands on the one its snap floored the tap to: in the placer's own frame under
    placement.SNAP_EVEN_CORNER = placer_frame, in the arena's under absolute. Flooring the
    corner in the own frame under absolute put every Red Tesla one tile up and right of the
    tile it was tapped on, and the ``taps_where_the_building_stays`` arm offered Red 50 taps
    to Blue's 142 (measured on the placement batch's build, 2026-09-28).
    """
    t = arena.subtile
    if snap_even_corner == "absolute":
        ax, ay = x // t, y // t
        if team == BLUE:
            return int(ax), int(ay)
        return int(arena.tiles_x - 1 - ax), int(arena.tiles_y - 1 - ay)
    ox, oy = to_own(arena, team, x, y)
    return int(ox // t), int(oy // t)


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
