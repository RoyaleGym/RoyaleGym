"""Observation builders: BattleState -> what the policy sees.

PERSPECTIVE
    Every observation is in the ACTING player's own frame (see protocol.py):
    Red's board is rotated 180 degrees and its towers/crowns/elixir are "own",
    Blue's are "enemy". A mirrored battle therefore yields a bit-identical
    observation for the other seat, which tests assert, so one policy plays both.

IMPERFECT INFORMATION
    The live game hides the opponent's elixir and hand. Both are hidden by
    default; ``reveal_enemy_elixir`` exists for curriculum/debugging, not as a
    faithful setting.

Floats appear here and only here-onwards (policy input). They are computed from
integer state by the same operations for both seats, so the flip is exact.

NO FLOAT MAY DEPEND ON ENTITY LIST ORDER
    ``state.entities`` order is engine-private and is NOT seat-canonical: the Rust
    engine lists entities by storage slot, and slots are reused, so an exactly
    rotation-mirrored battle can list Blue's twins in a different order from Red's.
    Float addition is not associative, so any per-entity float accumulation (or any
    sort whose key omits a feature it then writes) makes obs[blue] != obs[red]. The
    rule: accumulate INTEGERS, convert once; sort rows by every input they are
    built from. Accumulating ``sp[hp channel] += e.hp / HP_SCALE`` in list order
    instead costs a 1 ulp seat mismatch on mirrored states (channels 3 and 7),
    measured on both hand-built boards and boards the Rust engine played out.

SPELLS AND STATUS EFFECTS
    What the engine exposes, and nothing it does not: live spell objects
    (``BattleState.spells``: team, card, motion, centre, aim point, flight delay, roll
    progress, hits) and per-entity ``stun_ticks`` / ``knockback_ticks``. Spell
    objects are visible information in the live game (a Fireball in the air, a Log
    rolling), so both seats see both teams' spells. The spatial builder rasterises
    them in channels 15..20 (``SPATIAL_CHANNELS``); the entity-list builder adds
    status features to each entity row and a separate ``spells`` array. Not exposed,
    because the engine does not export it: a spell's hit radius or damage (use the
    card one-hot), buffs, a unit's current target. MockEngine resolves spells within
    a tick and has no status effects, so on it channels 15..20 and the ``spells``
    array are always zero (mock_engine.py WHAT IT IS NOT); tests/test_rust_engine.py
    checks they carry information on RustEngine battles.
    KNOCKBACK: ``knockback_ticks`` is a per-entity feature only, never a spatial
    channel. Under the shipped calibration knockback.DURATION_MS = 0 the push is
    instant and the timer is 0 between ticks, so a channel for it would be constant
    zero and unverifiable by any coverage guard.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
from gymnasium import spaces

from .action import ActionParser, PlacementOracle
from .protocol import (
    EMPTY_CARD,
    HAND_SIZE,
    RED,
    Arena,
    BattleState,
    Calibration,
    Engine,
    EntityKind,
    EntityState,
    Placement,
    SpellMotion,
    TowerSlot,
    default_calibration,
    to_own,
)

# Observation scaling. Presentation constants for the network, not physics.
HP_SCALE = 1000.0
SPATIAL_CLIP = 64.0


class ObsBuilder(ABC):
    """state -> observation dict. Always includes ``action_mask``."""

    calibration: Calibration

    def bind(self, engine: Engine, action_parser: ActionParser) -> None:
        if not hasattr(self, "calibration"):
            self.calibration = default_calibration()
        self.arena = engine.arena()
        self.cards = list(engine.cards())
        self.num_cards = len(self.cards)
        self.oracle = PlacementOracle(self.arena, engine.rules(), self.cards)
        self.mask_space = spaces.Box(0, 1, shape=(int(action_parser.space.n),), dtype=np.int8)
        self._probe_card = {
            p: next((c for c in self.cards if c.placement == p), None) for p in Placement
        }
        self.max_mana = self.calibration.int("match.MAX_MANA")

    def reset(self, state: BattleState) -> None:
        """Called at the start of every episode. Stateless builders ignore it."""
        del state

    @abstractmethod
    def observation_space(self) -> spaces.Dict: ...

    @abstractmethod
    def build(self, state: BattleState, team: int, action_mask: np.ndarray) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# The flat vector, shared by both builders
# ---------------------------------------------------------------------------


def vector_fields(num_cards: int) -> list[tuple[str, int]]:
    """Layout of the flat vector, in order. The test suite checks the sizes add up."""
    onehot = num_cards + 1  # last index = empty slot
    return [
        ("own_elixir / MAX_MANA", 1),
        ("enemy_elixir / MAX_MANA (0 unless reveal_enemy_elixir)", 1),
        ("hand slot card one-hot [4 x (num_cards+1)]", HAND_SIZE * onehot),
        ("hand slot elixir cost / MAX_MANA [4]", HAND_SIZE),
        ("hand slot affordable now [4]", HAND_SIZE),
        ("next card one-hot [num_cards+1]", onehot),
        ("own tower hp / max [king, left, right]", 3),
        ("enemy tower hp / max [king, left, right] (enemy's own-frame slots)", 3),
        ("own crowns / 3, enemy crowns / 3", 2),
        ("own king active, enemy king active", 2),
        ("regulation time remaining / regulation length", 1),
        ("in overtime", 1),
        ("overtime remaining / overtime length (0 before overtime)", 1),
        ("elixir rate one-hot [1x, 2x]", 2),
    ]


def build_vector(
    state: BattleState,
    team: int,
    cards: list[Any],
    max_mana: int,
    reveal_enemy_elixir: bool,
) -> np.ndarray:
    me, foe = state.players[team], state.players[1 - team]
    num_cards = len(cards)
    onehot = num_cards + 1
    full = 1000 * max_mana
    out: list[np.ndarray] = []
    out.append(np.array([me.elixir_milli / full], dtype=np.float32))
    out.append(
        np.array([foe.elixir_milli / full if reveal_enemy_elixir else 0.0], dtype=np.float32)
    )
    hand = np.zeros((HAND_SIZE, onehot), dtype=np.float32)
    cost = np.zeros(HAND_SIZE, dtype=np.float32)
    afford = np.zeros(HAND_SIZE, dtype=np.float32)
    for i, c in enumerate(me.hand):
        if c == EMPTY_CARD:
            hand[i, num_cards] = 1
            continue
        hand[i, c] = 1
        cost[i] = cards[c].elixir / max_mana
        afford[i] = 1.0 if me.elixir_milli >= cards[c].elixir * 1000 else 0.0
    out += [hand.reshape(-1), cost, afford]
    nxt = np.zeros(onehot, dtype=np.float32)
    nxt[num_cards if me.next_card == EMPTY_CARD else me.next_card] = 1
    out.append(nxt)
    for p in (me, foe):
        out.append(
            np.array(
                [p.tower_hp[s] / max(1, p.tower_max_hp[s]) for s in TowerSlot], dtype=np.float32
            )
        )
    out.append(np.array([me.crowns / 3.0, foe.crowns / 3.0], dtype=np.float32))
    out.append(np.array([float(me.king_active), float(foe.king_active)], dtype=np.float32))
    reg_left = max(0, state.regular_ticks - state.tick) / max(1, state.regular_ticks)
    ot_left = 0.0
    if state.overtime:
        ot_end = state.regular_ticks + state.overtime_ticks
        ot_left = max(0, ot_end - state.tick) / max(1, state.overtime_ticks)
    out.append(np.array([reg_left, float(state.overtime), ot_left], dtype=np.float32))
    out.append(
        np.array([float(state.elixir_rate == 1), float(state.elixir_rate == 2)], dtype=np.float32)
    )
    vec: np.ndarray = np.clip(np.concatenate(out), 0.0, 1.0).astype(np.float32)
    return vec


# ---------------------------------------------------------------------------
# Spatial builder (the default)
# ---------------------------------------------------------------------------

SPATIAL_CHANNELS: list[tuple[str, str]] = [
    ("own_ground_troops", "count of own ground troops whose centre is in the tile"),
    ("own_air_troops", "count of own flying troops"),
    ("own_buildings", "count of own buildings and crown towers (by centre)"),
    ("own_hp", "sum of own entity hp / 1000 in the tile"),
    ("enemy_ground_troops", "count of enemy ground troops"),
    ("enemy_air_troops", "count of enemy flying troops"),
    ("enemy_buildings", "count of enemy buildings and crown towers"),
    ("enemy_hp", "sum of enemy entity hp / 1000"),
    ("own_deploying", "count of own entities still in their deploy timer"),
    ("enemy_deploying", "count of enemy entities still in their deploy timer"),
    ("water", "fraction of the tile's 4 half-cells that are water (static)"),
    ("no_deploy", "fraction of the tile's 4 half-cells flagged no-deploy (static)"),
    ("own_troop_zone", "1 where a TROOP could be placed by me now (territory, pocket, footprints)"),
    ("enemy_troop_zone", "1 where the enemy could place a TROOP now"),
    ("own_building_zone", "1 where a BUILDING (radius of my first building card) could go"),
    ("own_spells", "count of my live spell objects whose current centre is in the tile"),
    ("enemy_spells", "count of enemy live spell objects whose current centre is in the tile"),
    (
        "own_spell_aim",
        "count of my live spells whose aim point is in the tile (landing point; roll end)",
    ),
    ("enemy_spell_aim", "count of enemy live spells whose aim point is in the tile"),
    ("own_stunned", "count of my entities with stun_ticks > 0"),
    ("enemy_stunned", "count of enemy entities with stun_ticks > 0"),
]


ENTITY_CHANNELS = 10  # SPATIAL_CHANNELS[0:10] are rasterised from state.entities
HP_CHANNELS = (3, 7)  # own_hp, enemy_hp
SPELL_CHANNELS = (15, 21)  # SPATIAL_CHANNELS[15:21], ``spell_channels``


def entity_channels(entities: Sequence[EntityState], team: int, arena: Arena) -> np.ndarray:
    """SPATIAL_CHANNELS[0:10] as float32 [10, tiles_y, tiles_x], seen by ``team``.

    Every cell is an INTEGER sum (counts, raw hp) converted to float32 exactly once,
    so the result is a function of the entity SET and cannot depend on list order
    (module doc). hp: int64 / HP_SCALE in float64 (exact for any sum below 2**53),
    then one rounding to float32. Module-level so a test can plant the old
    order-dependent float accumulation back in.
    """
    acc = np.zeros((ENTITY_CHANNELS, arena.tiles_y, arena.tiles_x), dtype=np.int64)
    for e in entities:
        ox, oy = to_own(arena, team, e.x, e.y)
        tx = min(max(ox // arena.subtile, 0), arena.tiles_x - 1)
        ty = min(max(oy // arena.subtile, 0), arena.tiles_y - 1)
        base = 0 if e.team == team else 4
        if e.kind == EntityKind.TROOP:
            acc[base + (1 if e.flying else 0), ty, tx] += 1
        else:
            acc[base + 2, ty, tx] += 1
        acc[base + 3, ty, tx] += e.hp
        if e.deploy_ticks > 0:
            acc[8 if e.team == team else 9, ty, tx] += 1
    out = acc.astype(np.float32)
    for c in HP_CHANNELS:
        out[c] = (acc[c] / HP_SCALE).astype(np.float32)
    return out


def _tile(arena: Arena, team: int, x: int, y: int) -> tuple[int, int]:
    """Own-frame tile of an engine-frame point, clamped onto the board (a Log's roll
    end point can lie past the arena edge)."""
    ox, oy = to_own(arena, team, x, y)
    return (
        min(max(oy // arena.subtile, 0), arena.tiles_y - 1),
        min(max(ox // arena.subtile, 0), arena.tiles_x - 1),
    )


def spell_channels(state: BattleState, team: int, arena: Arena) -> np.ndarray:
    """SPATIAL_CHANNELS[15:21] as float32 [6, tiles_y, tiles_x], seen by ``team``.

    Integer counts converted once, like ``entity_channels``, so the result is a
    function of the spell and entity SETS. Module-level so a test can plant a
    defect into it.
    """
    acc = np.zeros((6, arena.tiles_y, arena.tiles_x), dtype=np.int64)
    for sp in state.spells:
        side = 0 if sp.team == team else 1
        ty, tx = _tile(arena, team, sp.x, sp.y)
        acc[side, ty, tx] += 1
        ty, tx = _tile(arena, team, sp.aim_x, sp.aim_y)
        acc[2 + side, ty, tx] += 1
    for e in state.entities:
        if e.stun_ticks > 0:
            ty, tx = _tile(arena, team, e.x, e.y)
            acc[4 if e.team == team else 5, ty, tx] += 1
    out: np.ndarray = acc.astype(np.float32)
    return out


class SpatialObsBuilder(ObsBuilder):
    """Dict(spatial [C, 32, 18], vector [V], action_mask [A]).

    Entities are rasterised by the tile containing their centre in the own frame
    (``x_own // SUBTILE``, clamped). Channels are listed in ``SPATIAL_CHANNELS``.
    """

    def __init__(
        self, reveal_enemy_elixir: bool = False, calibration: Calibration | None = None
    ) -> None:
        self.reveal_enemy_elixir = reveal_enemy_elixir
        if calibration is not None:
            self.calibration = calibration

    def bind(self, engine: Engine, action_parser: ActionParser) -> None:
        super().bind(engine, action_parser)
        a = self.arena
        self.shape = (len(SPATIAL_CHANNELS), a.tiles_y, a.tiles_x)
        self.vec_size = sum(n for _, n in vector_fields(self.num_cards))
        h = a.half
        water = self.oracle.water.reshape(a.tiles_y, h, a.tiles_x, h).mean(axis=(1, 3))
        nodep = self.oracle.nodeploy.reshape(a.tiles_y, h, a.tiles_x, h).mean(axis=(1, 3))
        # Static channels in each team's own frame.
        self._static = [
            np.stack([water, nodep]).astype(np.float32),
            np.stack([water[::-1, ::-1], nodep[::-1, ::-1]]).astype(np.float32),
        ]
        self._space = spaces.Dict(
            {
                "spatial": spaces.Box(0.0, SPATIAL_CLIP, shape=self.shape, dtype=np.float32),
                "vector": spaces.Box(0.0, 1.0, shape=(self.vec_size,), dtype=np.float32),
                "action_mask": self.mask_space,
            }
        )

    def observation_space(self) -> spaces.Dict:
        return self._space

    def _zone(self, state: BattleState, team: int, viewer: int, placement: Placement) -> np.ndarray:
        card = self._probe_card.get(placement)
        a = self.arena
        if card is None:
            return np.zeros((a.tiles_y, a.tiles_x), dtype=np.float32)
        g = self.oracle.point_grid(state, team, card, 1)
        if viewer == RED:
            g = g[::-1, ::-1]
        return g.astype(np.float32)

    def build(self, state: BattleState, team: int, action_mask: np.ndarray) -> dict[str, Any]:
        sp = np.zeros(self.shape, dtype=np.float32)
        sp[:ENTITY_CHANNELS] = entity_channels(state.entities, team, self.arena)
        sp[10:12] = self._static[team]
        sp[12] = self._zone(state, team, team, Placement.TROOP)
        sp[13] = self._zone(state, 1 - team, team, Placement.TROOP)
        sp[14] = self._zone(state, team, team, Placement.BUILDING)
        sp[SPELL_CHANNELS[0] : SPELL_CHANNELS[1]] = spell_channels(state, team, self.arena)
        np.clip(sp, 0.0, SPATIAL_CLIP, out=sp)
        return {
            "spatial": sp,
            "vector": build_vector(
                state, team, self.cards, self.max_mana, self.reveal_enemy_elixir
            ),
            "action_mask": action_mask.astype(np.int8, copy=False),
        }


# ---------------------------------------------------------------------------
# Entity-list builder (for attention / transformer policies)
# ---------------------------------------------------------------------------


def entity_row_key(row: tuple[Any, ...]) -> tuple[Any, ...]:
    """Sort key of an EntityListObsBuilder row: every field but the entity itself.
    Module-level so a test can plant a shorter six-field key back in."""
    return row[:-1]


def spell_row_key(row: tuple[Any, ...]) -> tuple[Any, ...]:
    """Sort key of a ``spells`` row: every field but the spell itself (as entity rows).
    ``state.spells`` is in engine cast order, which is not seat-canonical."""
    return row[:-1]


class EntityListObsBuilder(ObsBuilder):
    """Dict(entities [N, F], vector [V], action_mask [A]).

    Per-entity features (F = 18 + num_cards + 1):
        0 present, 1 own, 2 enemy, 3..6 kind one-hot (troop, building, king, princess),
        7 x_own / width, 8 y_own / height, 9 hp / max_hp, 10 hp / 1000 (clipped to 1),
        11 radius / tile, 12 flying, 13 deploying, 14 deploy_ticks / 100 (clipped),
        15 stunned, 16 stun_ticks / 100 (clipped), 17 knockback slide in progress,
        18.. card one-hot (last index = crown tower / no card). A unit a spell
        released (Goblin Barrel's Goblins) carries that spell's card.
    Rows are sorted canonically by ``entity_row_key``: (enemy, y_own, x_own, kind,
    card, hp, max_hp, radius, flying, deploy_ticks, stun_ticks, knockback_ticks) --
    EVERY entity field a row is built from, so two rows that tie are identical and
    the order is seat-invariant whatever order the engine listed them in. A shorter
    key -- (enemy, y_own, x_own, kind, card, hp) -- leaves stacked twins that differ
    only in deploy_ticks in engine list order. Entities beyond
    ``max_entities`` are dropped in that order; the drop is silent, so size
    ``max_entities`` generously.

    Live spell objects, ``spells`` [max_spells, S] (S = 14 + num_cards):
        0 present, 1 own, 2 enemy, 3..6 motion one-hot (flight, airborne, rolling,
        area; a pulsing area effect sets the area bit), 7 x_own / width, 8 y_own /
        height, 9 aim_x_own / width, 10 aim_y_own / height, 11 delay_ticks / 100
        (clipped; a pulsing area's life left), 12 travelled / length (0 when length
        is 0), 13 hits / 16 (clipped), 14.. card one-hot. Sorted by ``spell_row_key``
        (enemy, y_own, x_own, motion, card, aim y_own, aim x_own, delay, travelled,
        length, hits); beyond ``max_spells`` dropped in that order. Positions are
        clipped to [0, 1] (a roll end point can lie past the arena edge).
    """

    BASE_FEATURES = 18
    SPELL_BASE_FEATURES = 14

    def __init__(
        self,
        max_entities: int = 96,
        reveal_enemy_elixir: bool = False,
        calibration: Calibration | None = None,
        max_spells: int = 16,
    ) -> None:
        self.max_entities = max_entities
        self.max_spells = max_spells
        self.reveal_enemy_elixir = reveal_enemy_elixir
        if calibration is not None:
            self.calibration = calibration

    def bind(self, engine: Engine, action_parser: ActionParser) -> None:
        super().bind(engine, action_parser)
        self.features = self.BASE_FEATURES + self.num_cards + 1
        self.spell_features = self.SPELL_BASE_FEATURES + self.num_cards
        self.vec_size = sum(n for _, n in vector_fields(self.num_cards))
        self._space = spaces.Dict(
            {
                "entities": spaces.Box(
                    0.0, 1.0, shape=(self.max_entities, self.features), dtype=np.float32
                ),
                "spells": spaces.Box(
                    0.0, 1.0, shape=(self.max_spells, self.spell_features), dtype=np.float32
                ),
                "vector": spaces.Box(0.0, 1.0, shape=(self.vec_size,), dtype=np.float32),
                "action_mask": self.mask_space,
            }
        )

    def observation_space(self) -> spaces.Dict:
        return self._space

    def build(self, state: BattleState, team: int, action_mask: np.ndarray) -> dict[str, Any]:
        a = self.arena
        rows = []
        for e in state.entities:
            ox, oy = to_own(a, team, e.x, e.y)
            enemy = int(e.team != team)
            card = self.num_cards if e.card_id == EMPTY_CARD else e.card_id
            rows.append(
                (
                    enemy,
                    oy,
                    ox,
                    int(e.kind),
                    card,
                    e.hp,
                    e.max_hp,
                    e.radius,
                    int(e.flying),
                    e.deploy_ticks,
                    e.stun_ticks,
                    e.knockback_ticks,
                    e,
                )
            )
        rows.sort(key=entity_row_key)
        out = np.zeros((self.max_entities, self.features), dtype=np.float32)
        for i, (enemy, oy, ox, kind, card, hp, *_, e) in enumerate(rows[: self.max_entities]):
            f = out[i]
            f[0] = 1
            f[1 + enemy] = 1
            f[3 + kind] = 1
            f[7] = ox / a.width
            f[8] = oy / a.height
            f[9] = hp / max(1, e.max_hp)
            f[10] = min(1.0, hp / HP_SCALE)
            f[11] = e.radius / a.subtile
            f[12] = float(e.flying)
            f[13] = float(e.deploy_ticks > 0)
            f[14] = min(1.0, e.deploy_ticks / 100.0)
            f[15] = float(e.stun_ticks > 0)
            f[16] = min(1.0, e.stun_ticks / 100.0)
            f[17] = float(e.knockback_ticks > 0)
            f[self.BASE_FEATURES + card] = 1
        np.clip(out, 0.0, 1.0, out=out)
        return {
            "entities": out,
            "spells": self._spell_rows(state, team),
            "vector": build_vector(
                state, team, self.cards, self.max_mana, self.reveal_enemy_elixir
            ),
            "action_mask": action_mask.astype(np.int8, copy=False),
        }

    def _spell_rows(self, state: BattleState, team: int) -> np.ndarray:
        a = self.arena
        rows = []
        for sp in state.spells:
            ox, oy = to_own(a, team, sp.x, sp.y)
            ax, ay = to_own(a, team, sp.aim_x, sp.aim_y)
            rows.append(
                (
                    int(sp.team != team),
                    oy,
                    ox,
                    sp.motion,
                    sp.card_id,
                    ay,
                    ax,
                    sp.delay_ticks,
                    sp.travelled,
                    sp.length,
                    sp.hits,
                    sp,
                )
            )
        rows.sort(key=spell_row_key)
        out = np.zeros((self.max_spells, self.spell_features), dtype=np.float32)
        for i, (enemy, oy, ox, motion, card, ay, ax, delay, trav, length, hits, _) in enumerate(
            rows[: self.max_spells]
        ):
            f = out[i]
            f[0] = 1
            f[1 + enemy] = 1
            if 0 <= motion < 4:
                f[3 + motion] = 1
            elif motion == SpellMotion.PULSING:
                f[3 + SpellMotion.AREA] = 1
            f[7] = ox / a.width
            f[8] = oy / a.height
            f[9] = ax / a.width
            f[10] = ay / a.height
            f[11] = min(1.0, delay / 100.0)
            f[12] = trav / length if length > 0 else 0.0
            f[13] = min(1.0, hits / 16.0)
            if 0 <= card < self.num_cards:
                f[self.SPELL_BASE_FEATURES + card] = 1
        np.clip(out, 0.0, 1.0, out=out)
        return out
