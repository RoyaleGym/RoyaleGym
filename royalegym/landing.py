"""Where a deploy actually put something, as opposed to where it was tapped.

WHY THIS EXISTS
    ``DeployResult.x`` and ``.y`` are the command, not the outcome. That was harmless
    while every accepted tap placed a unit exactly where it was asked to, and it stopped
    being harmless when the engine started RELOCATING a building whose footprint does not
    fit: it now searches outward for a cell the box fits in and puts the building there.

    The size of the gap is the reason this module exists rather than a footnote. Measured
    over 234 accepted Cannon taps at tile centres on the blue seat, one fresh battle per
    tap so no earlier building could block the search:

        relocated          118 of 234 (50%)
        displacement       median 1.00 tile, mean 0.67, max 3.00
        moved >= 1 tile    118 of 118

    Every single relocation moved a full tile or more. There is no small-error tail, so
    anything reading the command as a position is not slightly wrong on half its building
    sample, it is a tile or more wrong. That is a defect, not a precision note. It already
    retired a published placement-entropy figure, which had been computed over a
    distribution with half the building mass in the wrong bin.

HOW IT WORKS, AND WHY IT NEEDS NO ENGINE CHANGE
    ``EntityState.uid`` is unique for a whole battle and never reused, and a building
    appears in ``engine.state()`` the moment it is placed, with ``deploy_ticks > 0``. So
    the uid set before a step and after it differ by exactly the entities that step
    created, and their positions are where the engine actually put them. Nothing is
    re-derived in Python: the number comes from the engine.

WHAT THIS IS NOT
    It is not a replacement for ``DeployResult`` carrying the resolved position, which is
    the right fix and belongs in the engine. Three things this cannot see, and each one is
    reported rather than guessed at:

    - **A spell that resolves inside a tick never enters the entity list.** There is no
      entity to find, so there is no landing, and ``reason`` says so.
    - **A refused command leaves no trace at all**, which is why refusals are reported
      with the status rather than with a position.
    - **A multi-unit card spawns several entities.** "The position" is then a group, and
      this refuses to collapse it to one number. ``entities`` carries them all and ``x``
      and ``y`` stay ``None``: a centroid would be a derivation this module has no right
      to make, and it would be indistinguishable from a real single position downstream.

    Attribution is by ``(team, card_id)``. Two accepted commands in ONE step for the same
    team and the same card are genuinely ambiguous -- nothing in the entity list says
    which tap made which unit -- so both are marked ambiguous rather than paired off in
    arrival order, which would look right and be arbitrary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import msgspec
import numpy as np

from royalegym.protocol import DeployStatus

if TYPE_CHECKING:
    from collections.abc import Sequence

    from royalegym.protocol import BattleState, DeployCommand, DeployResult, EntityState


class _Engine(Protocol):
    """The two methods this needs. Duck-typed so it works on both engines."""

    def state(self) -> BattleState: ...

    def step(self, commands: Sequence[DeployCommand], ticks: int) -> list[DeployResult]: ...


class Landing(msgspec.Struct, frozen=True):
    """One command, and where the engine actually put what it created.

    ``x`` and ``y`` are filled ONLY when the command created exactly one entity, which is
    the case the relocation defect is about. Everywhere else they are ``None`` and
    ``reason`` says which case it was, so a caller cannot silently average a group or
    treat a refusal as an origin-placed unit.
    """

    result: DeployResult
    #: Where the command ASKED for, kept here because DeployResult no longer holds it:
    #: since the engine started returning the resolved position, `result.x` IS the
    #: landing. Comparing the landing against it asked whether the engine agreed with
    #: itself, which is a gate comparing a thing to itself and is always True.
    commanded: tuple[int, int] = (0, 0)
    #: Entities this command created, in engine order. Empty for a refusal or a spell.
    entities: tuple[EntityState, ...] = ()
    #: The landing, in the ENGINE frame, when there is exactly one entity.
    x: int | None = None
    y: int | None = None
    #: Empty when x and y are filled; otherwise why they are not.
    reason: str = ""

    @property
    def relocated(self) -> bool:
        """True when the engine put it somewhere other than the tap.

        False for every case where the landing is unknown, so this is safe to count but
        never treat a False as "the engine honoured the tap" without checking ``x``.
        """
        return self.x is not None and (self.x, self.y) != self.commanded

    @property
    def displacement(self) -> int | None:
        """Chebyshev distance from tap to landing in subtiles, or None if unknown.

        Chebyshev because the relocation search is a square ring search, so that is the
        metric the engine's own rule moves in.
        """
        if self.x is None or self.y is None:
            return None
        return max(abs(self.x - self.commanded[0]), abs(self.y - self.commanded[1]))


def landings(
    engine: _Engine, commands: Sequence[DeployCommand], ticks: int
) -> tuple[list[DeployResult], list[Landing]]:
    """Step the engine and report where each command's units actually landed.

    Returns the engine's own results unchanged alongside one :class:`Landing` per
    command, in the same order, so this is a drop-in for a ``step`` whose positions you
    intend to read. The results are returned rather than only the landings because a
    caller still needs the statuses, and re-deriving them from the landings would lose
    the refusal reasons.

    POSITIONS ARE READ AT THE DEPLOY TICK, not after ``ticks`` of them. The first version
    read them after the whole step, which is correct for a building and quietly wrong for
    anything that walks: at a decision's ten ticks a troop has half a second of movement
    in it, and this would have reported where it WENT as where it landed. Nothing about
    the number would have looked wrong.

    Caught from the outside. The integrator measured a group card on a pre-change core and
    got three Minions up to 2.5 tiles from the tap, all displaced toward the enemy, where
    reading at the deploy tick gives a ring of 0.58 tile centred on it. Units on one side
    of a tap is movement; a formation is not lopsided. The same reading mistake was sitting
    in this function, one caller away.

    So the step is split: one tick with the commands, snapshot, then the rest. The protocol
    validates commands up front and tick-by-tick is the same battle as one multi-tick call,
    which ``env.py`` already relies on and its tests check by state hash. The split is only
    paid when there are commands.
    """
    before = {e.uid for e in engine.state().entities}
    if commands and ticks > 1:
        results = engine.step(commands, 1)
        created = [e for e in engine.state().entities if e.uid not in before]
        engine.step([], ticks - 1)
    else:
        results = engine.step(commands, ticks)
        created = [e for e in engine.state().entities if e.uid not in before]

    # Which (team, card_id) pairs are claimed by more than one ACCEPTED result in this
    # same step. Those cannot be attributed, and saying so is the whole point.
    accepted = [r for r in results if r.status == DeployStatus.OK]
    claims: dict[tuple[int, int], int] = {}
    for r in accepted:
        claims[(r.team, r.card_id)] = claims.get((r.team, r.card_id), 0) + 1

    out: list[Landing] = []
    for cmd, r in zip(commands, results, strict=True):
        if r.status != DeployStatus.OK:
            out.append(
                Landing(
                    result=r,
                    commanded=(cmd.x, cmd.y),
                    reason=f"the command was refused ({DeployStatus(r.status).name}), so "
                    "nothing was created and there is no landing",
                )
            )
            continue
        key = (r.team, r.card_id)
        if claims[key] > 1:
            out.append(
                Landing(
                    result=r,
                    commanded=(cmd.x, cmd.y),
                    reason=f"{claims[key]} accepted commands in this step are team "
                    f"{r.team} card {r.card_id}, and the entity list does not say which "
                    "tap made which unit, so no landing can be attributed",
                )
            )
            continue
        mine = tuple(e for e in created if e.team == r.team and e.card_id == r.card_id)
        if not mine:
            out.append(
                Landing(
                    result=r,
                    commanded=(cmd.x, cmd.y),
                    reason="the command was accepted but created no entity, which is what "
                    "a spell that resolves inside the step does; it never enters the "
                    "entity list and has no landing to read",
                )
            )
            continue
        if len(mine) > 1:
            out.append(
                Landing(
                    result=r,
                    commanded=(cmd.x, cmd.y),
                    entities=mine,
                    reason=f"this card created {len(mine)} entities, so its position is a "
                    "group; a single x and y would be a derivation this cannot make "
                    "honestly. Read `entities`.",
                )
            )
            continue
        only = mine[0]
        out.append(
            Landing(result=r, commanded=(cmd.x, cmd.y), entities=mine, x=only.x, y=only.y)
        )
    return results, out


# ---------------------------------------------------------------------------
# The same question asked of a RECORDED trace rather than a live engine.
#
# WHY THIS IS NOT "landed != commanded". That comparison is the criterion that means the
# least, and it is the one a reader reaches for first. A building with an EVEN footprint
# snaps to a tile CORNER, so its centre can never sit on the tile centre that was tapped and
# the comparison reports 100% of Tesla taps as relocated however well the engine behaved. It
# is a fact about the definition, not about the game. Three criteria, and they disagree:
#
#     moved_point   the centre is not the point tapped   -- geometry for even footprints
#     moved_tile    the centre's tile is not the tile tapped
#     lost_tile     the FOOTPRINT does not cover the tile tapped
#
# `lost_tile` is the one to use, and filter buildings with `r.has_footprint` rather than
# `r.footprint >= 2`: footprint is None for troops and spells, and `None >= 2` raises.
#
# `lost_tile` is the one that bears on credit assignment, because the action space chooses a
# TILE: a 3x3 shifted by one tile is displaced but still stands on the tile the agent asked
# for. Measured over every legal tile on a near-empty board: 51.7% / 51.7% / 15.0% for a 3x3
# and 100.0% / 30.4% / 10.0% for a 2x2. Quote one of these only with its name attached.
#
# THE RATE A RUN PAYS IS NOT THAT SWEEP. Those figures weight every legal tile equally and a
# trained policy does not: it concentrates. Whether relocation costs a given agent more or
# less than the sweep says depends on whether its favoured tiles are ones its buildings fit
# on, which is exactly what this function exists to answer from its own recorded taps.


class Relocation(msgspec.Struct, frozen=True):
    """One recorded command, with where it was aimed and where the engine put it."""

    tick: int
    team: int
    card_id: int
    card_name: str
    footprint: int | None  # None for troops, spells, spawners
    commanded: tuple[int, int]
    landed: tuple[int, int]
    subtile: int

    @property
    def commanded_tile(self) -> tuple[int, int]:
        return (self.commanded[0] // self.subtile, self.commanded[1] // self.subtile)

    @property
    def landed_tile(self) -> tuple[int, int]:
        return (self.landed[0] // self.subtile, self.landed[1] // self.subtile)

    @property
    def moved_point(self) -> bool:
        """Least useful of the three; see the note above before quoting it."""
        return self.landed != self.commanded

    @property
    def moved_tile(self) -> bool:
        return self.landed_tile != self.commanded_tile

    @property
    def has_footprint(self) -> bool:
        """True for buildings. Use THIS to filter, never ``footprint >= 2``.

        ``footprint`` is None for troops, spells and spawners, and ``None >= 2`` is a
        TypeError rather than False -- so the obvious filter crashes on the first troop
        row, which in a real trace is most of them.
        """
        return bool(self.footprint)

    @property
    def lost_tile(self) -> bool:
        """The thing does not stand on the tile that was tapped.

        A card with NO footprint is placed at the point tapped, so it occupies the tile it
        landed on and nothing else: for those this is exactly ``moved_tile``. Returning
        False outright would also have been defensible, but it would make the property mean
        two different things depending on the row, and this way it answers the same question
        for every card.

        This used to raise a TypeError on any troop row -- the two criteria documented as
        the ones NOT to use returned cleanly while the recommended one crashed, so a caller
        who followed the advice hit it and a caller who ignored it did not. It survived
        because every test here used a deck of buildings, so the tests could not see the
        part of the space where most cards live.
        """
        if not self.footprint:
            return self.moved_tile
        half = self.footprint * self.subtile // 2
        tx, ty = self.commanded_tile
        xs = range((self.landed[0] - half) // self.subtile, (self.landed[0] + half) // self.subtile)
        ys = range((self.landed[1] - half) // self.subtile, (self.landed[1] + half) // self.subtile)
        return not (tx in xs and ty in ys)


def relocations(trace, accepted_only: bool = True) -> list[Relocation]:
    """Every recorded command with its aim and its outcome, for per-card analysis.

    Reads the card table from the TRACE'S OWN header rather than from a live engine, which
    is not a convenience: catalogue ids are POSITIONAL, so scoring a trace against a
    different card table renames every card silently and produces a per-card table that
    looks entirely reasonable.

    Raises on a trace recorded before steps carried these fields, rather than returning an
    empty list. An empty list reads as "nothing relocated", which is the one answer that is
    both wrong and plausible.
    """
    header = trace.header
    cards = header.cards
    out: list[Relocation] = []
    for step in trace.steps:
        if not step.commands:
            continue
        if not step.card_ids or not step.landed:
            raise ValueError(
                f"the step at tick {step.tick} carries commands but no card_ids/landed, so "
                "this trace predates them. Re-record it; a rate computed from the commands "
                "alone would be 0% and look like a result."
            )
        for cmd, card_id, pos, status in zip(
            step.commands, step.card_ids, step.landed, step.statuses, strict=True
        ):
            if accepted_only and status != DeployStatus.OK:
                continue
            info = cards[card_id] if 0 <= card_id < len(cards) else None
            out.append(
                Relocation(
                    tick=step.tick,
                    team=cmd.team,
                    card_id=card_id,
                    card_name=info.name if info else f"card:{card_id}",
                    footprint=info.footprint_tiles if info else None,
                    commanded=(cmd.x, cmd.y),
                    landed=(pos[0], pos[1]),
                    subtile=header.subtile,
                )
            )
    return out


# ---------------------------------------------------------------------------
# The same question asked of the BOARD rather than of a tap: for every tile, would a
# building tapped there keep the tile?
#
# WHY THIS EXISTS SEPARATELY FROM `relocations`. A relocation rate measured from a trace is
# a rate over taps the policy ACTUALLY MADE, and that is useless for a card the policy has
# learned not to play: train's agent plays Cannon on 0.6% of its opportunities, so more
# traces only scale a near-zero numerator. The rate they want decomposes instead:
#
#     rate = sum over tiles of  P(policy taps this tile) x (a footprint loses this tile)
#
# The first factor is well sampled from ANY trace, because a policy makes hundreds of taps a
# battle across all cards. The second is this map, and it needs no plays of the card at all.
#
# WHAT IT IS A FACT ABOUT, precisely, because this is the part that is easy to over-claim: it
# is computed against the engine's CURRENT state. On a fresh board that is the terrain-only
# answer. In a live battle other buildings block the search and change it, so a map is a
# snapshot and not a constant of the arena. It is stable across CARDS of one footprint --
# that is measured, and `tests/` asserts it -- and not across boards.


class TileLossMap(msgspec.Struct, frozen=True):
    """Per-tile outcome of tapping a building at each tile centre, in the seat's own frame.

    Three arrays rather than one, because "the tap did not give me the tile" and "the tap was
    not offered at all" are different facts and collapsing them makes an illegal tile look
    like a relocating one.
    """

    card: str
    footprint: int
    team: int
    offered: np.ndarray  # [ny, nx] bool: the engine found somewhere for the box
    moved: np.ndarray  # [ny, nx] bool: the centre's tile is not the tapped tile
    lost: np.ndarray  # [ny, nx] bool: the footprint does not cover the tapped tile

    def rate(self) -> float:
        """Fraction of OFFERED tiles that lose the tile. The figure quoted for a card."""
        n = int(self.offered.sum())
        return float(self.lost.sum()) / n if n else 0.0

    def would_lose_tile(self, tx: int, ty: int) -> bool | None:
        """None when the tile is not offered at all, which is not the same as 'keeps it'."""
        if not (0 <= ty < self.offered.shape[0] and 0 <= tx < self.offered.shape[1]):
            return None
        if not self.offered[ty, tx]:
            return None
        return bool(self.lost[ty, tx])


def tile_loss_map(engine, card: str, team: int = 0) -> TileLossMap:
    """Ask the engine, tile by tile, where it would actually put ``card``.

    Nothing is re-derived in Python: every answer comes from ``building_placement``, the
    same resolver the action mask uses, so this cannot drift from the engine's own rule.
    """
    arena = engine.arena()
    sub = arena.subtile
    info = next((c for c in engine.cards() if c.name == card), None)
    if info is None:
        raise ValueError(f"no card named {card!r} in this engine's table")
    footprint = info.footprint_tiles
    if not footprint:
        raise ValueError(
            f"{card!r} has no footprint, so it is never relocated and a loss map for it "
            "would be all False -- which would read as a measurement rather than a category "
            "error. Pass a building."
        )
    ny, nx = arena.tiles_y, arena.tiles_x
    offered = np.zeros((ny, nx), dtype=bool)
    moved = np.zeros((ny, nx), dtype=bool)
    lost = np.zeros((ny, nx), dtype=bool)
    half = footprint * sub // 2
    for ty in range(ny):
        for tx in range(nx):
            x, y = tx * sub + sub // 2, ty * sub + sub // 2
            landed = engine.building_placement(team, card, x, y)
            if landed is None:
                continue
            lx, ly = landed[0], landed[1]
            offered[ty, tx] = True
            moved[ty, tx] = (lx // sub, ly // sub) != (tx, ty)
            covers_x = tx in range((lx - half) // sub, (lx + half) // sub)
            covers_y = ty in range((ly - half) // sub, (ly + half) // sub)
            lost[ty, tx] = not (covers_x and covers_y)
    return TileLossMap(
        card=card, footprint=footprint, team=team, offered=offered, moved=moved, lost=lost
    )
