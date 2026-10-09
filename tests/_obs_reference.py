"""The observation builder's per-entity plane functions as they were before the vectorized
rewrite (RoyaleGym 27b2a64, 2026-10-09), copied verbatim: the frozen oracle that
tests/test_obs_fast_equality.py holds the fast versions to, byte for byte, on real battles.

Do not edit these to follow a change in royalegym.obs: a plane whose meaning changes on purpose
gets a new reference, written beside the change. Every name they use is bound here from a fixed
source (protocol, or royalegym.obs constants), never from a module a test may plant into.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from royalegym.obs import (
    ABILITY_TICKS_SCALE,
    ACTION_SPATIAL_CHANNELS,
    CARD_ID_EMPTY,
    CARD_ID_OFFSET,
    CARD_ID_TOWER,
    ENTITY_CHANNELS,
    HP_CHANNELS,
    HP_SCALE,
    RAGE_BUFF,
    SLOW_BUFF,
    SPELL_ROWS,
    STATUS_SPATIAL_CHANNELS,
    TEAM_STRIDE,
    TOWER_KINDS,
    UNIT_ID_EMPTY,
    UNIT_ID_OFFSET,
)
from royalegym.protocol import (
    EMPTY_CARD,
    STATUS_ABILITY_ACTIVE,
    STATUS_CHARGED,
    STATUS_CLONE,
    STATUS_EVOLVED,
    STATUS_HERO,
    STATUS_HIDDEN,
    STATUS_INVISIBLE,
    STATUS_UNDERGROUND,
    STATUS_WINDUP,
    Arena,
    BattleState,
    EntityKind,
    EntityState,
    ability_row,
    in_the_air,
    status_of,
    to_own,
)


def _tile(arena: Arena, team: int, x: int, y: int) -> tuple[int, int]:
    """Own-frame tile of an engine-frame point, clamped onto the board (a Log's roll
    end point can lie past the arena edge)."""
    ox, oy = to_own(arena, team, x, y)
    return (
        min(max(oy // arena.subtile, 0), arena.tiles_y - 1),
        min(max(ox // arena.subtile, 0), arena.tiles_x - 1),
    )


def entity_channels(
    entities: Sequence[EntityState], team: int, arena: Arena, grounded_said: bool = False
) -> np.ndarray:
    """The first ``ENTITY_CHANNELS`` channels, float32 [12, tiles_y, tiles_x], seen by ``team``.

    A troop goes in the air channel while it is in the air (``protocol.in_the_air``): a flier
    a Vines catch holds on the ground counts as a ground troop for the hold, where an engine
    says it (``grounded_said``, ``STATUS_GROUNDED``).

    Every cell is an INTEGER sum (counts, raw hp) converted to float32 exactly once,
    so the result is a function of the entity SET and cannot depend on list order
    (module doc). hp: int64 / HP_SCALE in float64 (exact for any sum below 2**53),
    then one rounding to float32. Crown towers are counted in their OWN channel and
    not in ``own_buildings`` / ``enemy_buildings``: a Cannon and a princess tower
    pose different problems, and one channel holding both could not say which it
    was. Module-level so a test can plant the old order-dependent float
    accumulation back in.
    """
    acc = np.zeros((ENTITY_CHANNELS, arena.tiles_y, arena.tiles_x), dtype=np.int64)
    for e in entities:
        ox, oy = to_own(arena, team, e.x, e.y)
        tx = min(max(ox // arena.subtile, 0), arena.tiles_x - 1)
        ty = min(max(oy // arena.subtile, 0), arena.tiles_y - 1)
        base = 0 if e.team == team else TEAM_STRIDE
        if e.kind == EntityKind.TROOP:
            acc[base + (1 if in_the_air(e, grounded_said) else 0), ty, tx] += 1
        elif e.kind == EntityKind.BUILDING:
            acc[base + 2, ty, tx] += 1
        else:
            acc[base + 3, ty, tx] += 1
        acc[base + 4, ty, tx] += e.hp
        if e.deploy_ticks > 0:
            acc[10 if e.team == team else 11, ty, tx] += 1
    out = acc.astype(np.float32)
    for c in HP_CHANNELS:
        out[c] = (acc[c] / HP_SCALE).astype(np.float32)
    return out


def spell_channels(state: BattleState, team: int, arena: Arena) -> np.ndarray:
    """The ``SPELL_ROWS``, float32 [6, tiles_y, tiles_x], seen by ``team``.

    Always all six rows, whatever the Reveal: the builder decides which of them
    reach the observation, so this stays one function with one layout that a test
    can plant a defect into. Integer counts converted once, like
    ``entity_channels``, so the result is a function of the spell and entity SETS.
    """
    acc = np.zeros((len(SPELL_ROWS), arena.tiles_y, arena.tiles_x), dtype=np.int64)
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


def evolved_channels(entities: Sequence[EntityState], team: int, arena: Arena) -> np.ndarray:
    """float32 [2, tiles_y, tiles_x], seen by ``team``: own, then enemy, evolved units.

    Counted on the centre tile, as ``entity_channels`` counts troops, from each unit's
    ``STATUS_EVOLVED`` bit. A hero's unit carries ``STATUS_HERO`` instead and is not
    counted. An engine that does not report the bits (``status_flags`` -1) is refused:
    reading "not reported" as "not evolved" would hand a network zeros that look like an
    answer. Module-level so a test can plant a defect in it.
    """
    acc = np.zeros((2, arena.tiles_y, arena.tiles_x), dtype=np.int64)
    for e in entities:
        status = status_of(e)  # towers too: an engine that reports the bits reports theirs
        if status is None:
            raise ValueError(
                f"evolutions=True, but this engine does not report which units are evolved "
                f"(entity uid {e.uid} has no status_flags). Use an engine that reports them, "
                "such as RustEngine, or leave evolutions off."
            )
        if status & STATUS_EVOLVED and e.kind not in TOWER_KINDS:
            ty, tx = _tile(arena, team, e.x, e.y)
            acc[0 if e.team == team else 1, ty, tx] += 1
    out: np.ndarray = acc.astype(np.float32)
    return out


def hero_channels(
    state: BattleState, team: int, arena: Arena, champions: frozenset[int] = frozenset()
) -> np.ndarray:
    """float32 [3, tiles_y, tiles_x], seen by ``team``: own heroes, enemy heroes, and the enemy
    heroes whose one ability charge is not used yet (``HERO_SPATIAL_CHANNELS``).

    Counted on the centre tile, from each unit's ``STATUS_HERO`` bit. An engine that does not
    report the bits is refused, as ``evolved_channels`` refuses it. An enemy hero's charge is
    read from the enemy's button row that names its card, and from that row's ``spent`` column
    alone: ``available`` and ``cost`` are not on a player's screen for the enemy's buttons. A
    hero no row names is refused rather than guessed. Module-level so a test can plant a defect.

    ``champions``: the card ids the catalogue calls champions (``CardInfo.champion``). Their
    units are skipped: royalesim up to 0.1.19 sets ``STATUS_HERO`` on a champion's unit too,
    and a champion is not a hero form (its row is never spent, so an enemy champion would sit
    in ``enemy_hero_unspent`` for its whole life).
    """
    spent: dict[int, int] = {}
    for r in map(ability_row, state.players[1 - team].abilities):
        if r.card_id != EMPTY_CARD:
            spent[r.card_id] = r.spent
    acc = np.zeros((3, arena.tiles_y, arena.tiles_x), dtype=np.int64)
    for e in state.entities:
        status = status_of(e)
        if status is None:
            raise ValueError(
                f"heroes=True, but this engine does not report which units are heroes "
                f"(entity uid {e.uid} has no status_flags). Use an engine that reports them, "
                "such as RustEngine, or leave heroes off."
            )
        if not status & STATUS_HERO or e.kind in TOWER_KINDS or e.card_id in champions:
            continue
        ty, tx = _tile(arena, team, e.x, e.y)
        if e.team == team:
            acc[0, ty, tx] += 1
            continue
        acc[1, ty, tx] += 1
        if e.card_id not in spent:
            raise ValueError(
                f"enemy hero uid {e.uid} (card {e.card_id}) is on the board, but no ability "
                f"row names its card (the enemy's rows: {state.players[1 - team].abilities}), "
                "so whether its charge is used cannot be read"
            )
        if spent[e.card_id] == 0:
            acc[2, ty, tx] += 1
    out: np.ndarray = acc.astype(np.float32)
    return out


def status_channels(entities: Sequence[EntityState], team: int, arena: Arena) -> np.ndarray:
    """float32 [18, tiles_y, tiles_x], seen by ``team``: ``STATUS_SPATIAL_CHANNELS``.

    Counted on the centre tile like ``entity_channels``: shield hp summed as an integer and
    scaled once by ``HP_SCALE``, raged and slowed units counted (``RAGE_BUFF``, ``SLOW_BUFF``),
    units whose ``target_uid`` is a crown tower's uid (a locked unit ignores a building put
    down to pull it) or another building's, each side's strongest unit's hp fraction (the
    largest max hp, ties to the lowest uid, so a function of the entity set), and units with
    the invisible, underground or hidden bit. Crown towers are in none of them.
    An engine that does not report status bits is refused, as ``evolved_channels`` refuses it:
    such an engine predates the shield and buff fields too, and their defaults (no shield, no
    buffs) would read as an answer. Module-level so a test can plant a defect in it.
    """
    acc = np.zeros((len(STATUS_SPATIAL_CHANNELS), arena.tiles_y, arena.tiles_x), dtype=np.int64)
    towers = {e.uid for e in entities if e.kind in TOWER_KINDS}
    buildings = {e.uid for e in entities if e.kind == EntityKind.BUILDING}
    strongest: dict[tuple[int, int, int], tuple[int, int, int, int]] = {}
    bits = ((STATUS_INVISIBLE, 12), (STATUS_UNDERGROUND, 14), (STATUS_HIDDEN, 16))
    for e in entities:
        status = status_of(e)
        if status is None:
            raise ValueError(
                f"unit_status=True, but this engine does not report unit status (entity uid "
                f"{e.uid} has no status_flags). Use an engine that reports it, such as "
                "RustEngine, or leave unit_status off."
            )
        if e.kind in TOWER_KINDS:
            continue
        ty, tx = _tile(arena, team, e.x, e.y)
        side = 0 if e.team == team else 1
        acc[side, ty, tx] += e.shield
        members = {m for name, _ in e.buffs for m in name.split("|")}
        if RAGE_BUFF in members:
            acc[2 + side, ty, tx] += 1
        if SLOW_BUFF in members:
            acc[4 + side, ty, tx] += 1
        if e.target_uid in towers:
            acc[6 + side, ty, tx] += 1
        elif e.target_uid in buildings:
            acc[8 + side, ty, tx] += 1
        for bit, plane in bits:
            if status & bit:
                acc[plane + side, ty, tx] += 1
        key = (side, ty, tx)
        held = strongest.get(key)
        if held is None or (e.max_hp, -e.uid) > (held[0], -held[1]):
            strongest[key] = (e.max_hp, e.uid, e.hp, max(1, e.max_hp))
    out = acc.astype(np.float32)
    out[:2] = (acc[:2] / HP_SCALE).astype(np.float32)
    for (side, ty, tx), (_, _, hp, max_hp) in strongest.items():
        # In permille, rounded in integers and written as float32(q) / 1000: a plane that
        # stores as uint16 x 1000 exactly (RoyaleImitate's shards refuse anything else), so a
        # live bot reads what the clone trained on.
        q = (2000 * min(max(hp, 0), max_hp) + max_hp) // (2 * max_hp)
        out[10 + side, ty, tx] = np.float32(q) / np.float32(1000)
    return out


def action_channels(state: BattleState, team: int, arena: Arena) -> np.ndarray:
    """float32 [14, tiles_y, tiles_x], seen by ``team``: ``ACTION_SPATIAL_CHANNELS``.

    Counted on the centre tile like ``entity_channels``; charge and ability ticks are the most
    on the tile, a tunneller is counted at its landing tile. Crown towers are in none of them.
    An engine before royalesim 0.1.8 sends -1, "not said", for every unit's charge, and is
    refused rather than read as a board where nothing charges. Module-level for plants.
    """
    acc = np.zeros((len(ACTION_SPATIAL_CHANNELS), arena.tiles_y, arena.tiles_x), dtype=np.int64)
    most = np.zeros((4, arena.tiles_y, arena.tiles_x), dtype=np.int64)  # charge, ticks x side
    bits = ((STATUS_CHARGED, 2), (STATUS_WINDUP, 4), (STATUS_ABILITY_ACTIVE, 6), (STATUS_CLONE, 10))
    for e in state.entities:
        status = status_of(e)
        if status is None or e.charge < 0:
            raise ValueError(
                "unit_actions=True, but this engine does not report what units are doing "
                f"(entity uid {e.uid}: status_flags {e.status_flags}, charge {e.charge}). It "
                "needs royalesim 0.1.8's charge, dest and ability_ticks columns; use an engine "
                "that sends them, or leave unit_actions off."
            )
        if e.kind in TOWER_KINDS:
            continue
        side = 0 if e.team == team else 1
        ty, tx = _tile(arena, team, e.x, e.y)
        most[side, ty, tx] = max(most[side, ty, tx], min(e.charge, 1000))
        most[2 + side, ty, tx] = max(most[2 + side, ty, tx], max(e.ability_ticks, 0))
        for bit, plane in bits:
            if status & bit:
                acc[plane + side, ty, tx] += 1
        if e.dest_x >= 0 and e.dest_y >= 0:
            dy, dx = _tile(arena, team, e.dest_x, e.dest_y)
            acc[12 + side, dy, dx] += 1
    out: np.ndarray = acc.astype(np.float32)
    out[0:2] = (most[0:2] / 1000.0).astype(np.float32)
    out[8:10] = np.minimum(most[2:4] / ABILITY_TICKS_SCALE, 1.0).astype(np.float32)
    return out


def card_id_planes(
    entities: Sequence[EntityState], team: int, arena: Arena, num_cards: int
) -> np.ndarray:
    """uint8 [2, tiles_y, tiles_x], seen by ``team``: plane 0 own, plane 1 enemy.

    Which CARD occupies each tile, which the float ``spatial`` planes cannot say: they
    count troops and sum hp, so a Giant and a Knight on one tile look alike
    (docs/observation-spec.md 3b; decision D2). Tiles are the same own-frame centre tiles
    ``entity_channels`` uses, so the two line up cell for cell.

    TIES GO TO THE LOWEST UID, and that is not cosmetic. ``BattleState.entities`` is not in
    uid order -- a live battle gives [0, 2, 4, 1, 3, 5] -- so "whichever comes first" would
    make a plane a function of iteration order, and two runs of one seed could differ. A
    uid is unique for a whole battle and never reused.

    Spells in flight never appear: ``BattleState.spells`` is a separate list, so a live
    spell has no entity and no tile. Units a spell releases do appear, under the releasing
    spell's catalogue id, because that is what the engine reports for them.

    Module-level so a test can plant a defect in it.
    """
    out = np.zeros((2, arena.tiles_y, arena.tiles_x), dtype=np.uint8)
    for e in sorted(entities, key=lambda ent: ent.uid):
        ox, oy = to_own(arena, team, e.x, e.y)
        tx = min(max(ox // arena.subtile, 0), arena.tiles_x - 1)
        ty = min(max(oy // arena.subtile, 0), arena.tiles_y - 1)
        plane = 0 if e.team == team else 1
        if out[plane, ty, tx] != CARD_ID_EMPTY:
            continue  # a lower uid already holds this tile
        if e.kind in TOWER_KINDS:
            out[plane, ty, tx] = CARD_ID_TOWER
        elif 0 <= e.card_id < num_cards:
            out[plane, ty, tx] = CARD_ID_OFFSET + e.card_id
        else:
            # Writing it anyway would put an id outside the declared vocabulary into an
            # embedding lookup, or fold an unknown entity into "tower". Neither is a value.
            raise ValueError(
                f"entity uid {e.uid} (kind {e.kind}) has card_id {e.card_id}, outside the "
                f"{num_cards}-card catalogue, and is not a crown tower, so card_ids has no "
                "index for it"
            )
    return out


def unit_id_planes(
    entities: Sequence[EntityState],
    team: int,
    arena: Arena,
    num_units: int,
    lookup: np.ndarray | None = None,
) -> np.ndarray:
    """uint8 [2, tiles_y, tiles_x], seen by ``team``: plane 0 own, plane 1 enemy.

    Which UNIT TYPE stands on each tile (``EntityState.unit_type``, the game's characters row),
    where ``card_ids`` says which CARD produced it. A summon carries its producer's card id, so
    the Witch's skeletons read "Witch" there and "Skeleton" here, as Tombstone's do: one type
    under every producer, and the producer kept beside it. A summon the game's data gives a row
    of its own (the Graveyard's skeleton) keeps that row unless ``lookup`` maps it to its base
    (``SAME_UNIT_ALIASES``, ``SpatialObsBuilder(unit_aliases=...)``).

    Ties go to the lowest uid, as in ``card_id_planes``, so the two planes describe the same
    entity cell for cell. An entity whose type is not said (-1) or lies outside the vocabulary
    is refused: either would be drawn as some other type. Module-level so a test can plant a
    defect in it.
    """
    out = np.zeros((2, arena.tiles_y, arena.tiles_x), dtype=np.uint8)
    for e in sorted(entities, key=lambda ent: ent.uid):
        ox, oy = to_own(arena, team, e.x, e.y)
        tx = min(max(ox // arena.subtile, 0), arena.tiles_x - 1)
        ty = min(max(oy // arena.subtile, 0), arena.tiles_y - 1)
        plane = 0 if e.team == team else 1
        if out[plane, ty, tx] != UNIT_ID_EMPTY:
            continue  # a lower uid already holds this tile
        if not 0 <= e.unit_type < num_units:
            raise ValueError(
                f"entity uid {e.uid} (kind {e.kind}, card {e.card_id}) has unit_type "
                f"{e.unit_type}, outside the {num_units}-type vocabulary"
                + (" (-1: the engine does not say it)" if e.unit_type == -1 else "")
                + ", so unit_ids has no index for it"
            )
        u = int(lookup[e.unit_type]) if lookup is not None else e.unit_type
        out[plane, ty, tx] = UNIT_ID_OFFSET + u
    return out
