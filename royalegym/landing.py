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
