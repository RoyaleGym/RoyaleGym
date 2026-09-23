"""``royalegym.landing`` reports where a deploy actually put something.

WHAT THESE GRADE AGAINST
    ``engine.state()``, never a second Python implementation of the relocation rule. A
    test that recomputed the ring search here would agree with itself and would have been
    green throughout the period the defect existed.

THE CASES THAT MATTER MOST ARE THE ONES WHERE IT MUST REFUSE
    A helper that always returns a number is worse than no helper, because a caller
    cannot tell a real landing from a guess. So the refusal paths are tested as hard as
    the success path: a refused command, a card that creates several entities, and two
    identical accepted commands in one step, which are genuinely unattributable.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from royalegym.landing import landings
from royalegym.protocol import DeployCommand, DeployResult, DeployStatus, MatchSetup
from royalegym.rust_engine import core_available


def battle_setup(engine, idx: int) -> MatchSetup:
    """One card in every slot, starting PAST the opening deploy lockout.

    A match refuses every command for its first `deploy_lockout_ticks` ticks, so a battle
    beginning at 0 answers TOO_EARLY to every tap here and these tests would grade a timing
    rule rather than the one they are about. Read from the engine: 0 is a real calibration
    arm, so a literal 90 would be wrong on a build without a lockout.
    """
    return MatchSetup(
        decks=[[idx] * 8] * 2,
        elixir_milli=[10000] * 2,
        start_tick=engine.rules().deploy_lockout_ticks,
    )

pytestmark = pytest.mark.skipif(
    not core_available(), reason="the compiled engine is not built; a skip here is not a pass"
)

TILE = 18000


def engine_with(card_name: str):
    from royalegym.rust_engine import RustEngine

    eng = RustEngine()
    cards = eng.cards()
    idx = next((i for i, c in enumerate(cards) if c.name == card_name), None)
    if idx is None:
        pytest.skip(
            f"this catalogue has no {card_name!r}; a skip here is not a pass, it means the "
            "card table changed and this test needs a different card"
        )
    eng.reset(seed=0, setup=battle_setup(eng, idx))
    eng.step([], 10)
    return eng, idx


def tap(tx: int, ty: int) -> tuple[int, int]:
    return tx * TILE + TILE // 2, ty * TILE + TILE // 2


def test_a_relocated_building_reports_where_it_stands_not_where_it_was_tapped() -> None:
    """The defect itself. Graded against the entity in ``engine.state()``."""
    found = None
    # The battle is re-made per tap so no earlier Cannon can block the ring search. That
    # is what made my first measurement of this report displacements of up to five tiles:
    # it was my own leftovers, not the law.
    eng, idx = engine_with("Cannon")
    for tx in range(18):
        for ty in range(1, 15):
            eng.reset(seed=0, setup=battle_setup(eng, idx))
            eng.step([], 10)
            x, y = tap(tx, ty)
            _, land = landings(eng, [DeployCommand(team=0, hand_slot=0, x=x, y=y)], 2)
            if land[0].x is not None and land[0].relocated:
                found = land[0]
                break
        if found:
            break
    assert found is not None, (
        "no tile-centre Cannon tap on the blue seat relocated. Half of them did when this "
        "was written, so either the engine stopped relocating or this stopped looking."
    )
    # Against the COMMAND, not against result.x/y. Since the engine began returning the
    # resolved position, result.x IS the landing, so the old form of this line asked
    # whether the engine agreed with itself and was true by construction.
    assert (found.x, found.y) != found.commanded
    assert found.displacement is not None
    assert found.displacement >= TILE, (
        f"the landing moved {found.displacement} subtiles, less than one tile. Every "
        "relocation moved a full tile or more when this was measured; a sub-tile move "
        "means the rule changed and the claim in landing.py's docstring needs re-taking."
    )
    # The claim that matters: this is the engine's own number, not a recomputation.
    entity = next(e for e in eng.state().entities if e.uid == found.entities[0].uid)
    assert (found.x, found.y) == (entity.x, entity.y)


def test_a_tap_that_fits_is_reported_unmoved() -> None:
    """The other half. Without this, a helper that always claimed relocation would pass."""
    eng, idx = engine_with("Cannon")
    unmoved = []
    for tx in range(18):
        for ty in range(1, 15):
            eng.reset(seed=0, setup=battle_setup(eng, idx))
            eng.step([], 10)
            x, y = tap(tx, ty)
            _, land = landings(eng, [DeployCommand(team=0, hand_slot=0, x=x, y=y)], 2)
            if land[0].x is not None and not land[0].relocated:
                unmoved.append(land[0])
    assert unmoved, (
        "every accepted Cannon tap relocated, so this test cannot tell a correct helper "
        "from one that always reports a move"
    )
    for land in unmoved:
        assert (land.x, land.y) == (land.result.x, land.result.y)
        assert land.displacement == 0


def test_a_refused_command_gets_no_landing_and_says_why() -> None:
    eng, _ = engine_with("Cannon")
    # Deep in the enemy half: out of territory, so refused.
    x, y = tap(9, 28)
    results, land = landings(eng, [DeployCommand(team=0, hand_slot=0, x=x, y=y)], 2)
    assert results[0].status != DeployStatus.OK, (
        "this tap was meant to be refused and was accepted, so the refusal path is untested"
    )
    assert land[0].x is None
    assert land[0].y is None
    assert land[0].entities == ()
    assert "refused" in land[0].reason


def test_two_identical_commands_in_one_step_are_ambiguous_rather_than_paired_off() -> None:
    """Nothing in the entity list says which tap made which unit, so neither gets a landing.

    THIS USES A FAKE ENGINE ON PURPOSE, and that is the finding rather than a shortcut.
    The compiled engine refuses a second command from the same team in one step with
    DUPLICATE_TEAM, so this branch cannot be reached through it at all. Driving it through
    the real engine gave a test that SKIPPED every run: a guard whose test can never fire,
    which is the same shape as a control that is secretly illegal. Either the guard is
    dead code or it is reachable by something; it is reachable by any engine that does not
    refuse duplicates, so it is tested against one.

    Pairing the two in arrival order would look right and be arbitrary, and it would be
    invisible, because both landings would carry plausible numbers.
    """

    class TwoPerTeamEngine:
        """Accepts both commands, which the compiled engine will not do."""

        def __init__(self) -> None:
            self._entities: list[object] = []

        def state(self):
            return SimpleNamespace(entities=list(self._entities))

        def step(self, commands, ticks):
            out = []
            for i, c in enumerate(commands):
                uid = 100 + len(self._entities)
                self._entities.append(
                    SimpleNamespace(uid=uid, team=c.team, card_id=7, x=c.x, y=c.y)
                )
                out.append(
                    DeployResult(
                        team=c.team, hand_slot=c.hand_slot, card_id=7,
                        status=int(DeployStatus.OK), tick=i, x=c.x, y=c.y,
                    )
                )
            return out

    eng = TwoPerTeamEngine()
    results, land = landings(
        eng,
        [
            DeployCommand(team=0, hand_slot=0, x=1000, y=2000),
            DeployCommand(team=0, hand_slot=1, x=9000, y=2000),
        ],
        2,
    )
    assert all(r.status == DeployStatus.OK for r in results), (
        "the fake engine must accept BOTH commands or this tests nothing"
    )
    assert len(land) == 2
    for one in land:
        assert one.x is None, (
            "a landing was attributed although two accepted commands in this step are the "
            "same team and card. That attribution cannot be right except by luck."
        )
        assert "which tap made which unit" in one.reason


def test_the_real_engine_refuses_a_second_command_from_one_team() -> None:
    """Pins WHY the test above needs a fake, so nobody deletes it as over-engineering.

    If the engine ever starts accepting two commands per team per step, this goes red and
    the ambiguity branch becomes reachable for real.
    """
    eng, _ = engine_with("Cannon")
    results = eng.step(
        [
            DeployCommand(team=0, hand_slot=0, x=6 * TILE + TILE // 2, y=5 * TILE + TILE // 2),
            DeployCommand(team=0, hand_slot=1, x=11 * TILE + TILE // 2, y=5 * TILE + TILE // 2),
        ],
        2,
    )
    assert [r.status for r in results] == [DeployStatus.OK, DeployStatus.DUPLICATE_TEAM], (
        f"expected the second same-team command to be refused as DUPLICATE_TEAM, got "
        f"{[DeployStatus(r.status).name for r in results]}. If it is now accepted, the "
        "ambiguity branch in landing.py is reachable through the real engine and its test "
        "should be driven through it rather than through a fake."
    )


def test_a_multi_unit_card_keeps_its_entities_and_refuses_a_single_position() -> None:
    """A centroid would be a derivation, and downstream it would look like a real position."""
    eng = None
    for name in ("Skeletons", "SkeletonArmy", "Barbarians", "Minions"):
        from royalegym.rust_engine import RustEngine

        probe = RustEngine()
        hit = next((c for c in probe.cards() if c.name == name and c.count > 1), None)
        if hit:
            eng, _ = engine_with(name)
            break
    if eng is None:
        pytest.skip("this catalogue has no multi-unit card to test with; a skip is not a pass")
    x, y = tap(9, 8)
    results, land = landings(eng, [DeployCommand(team=0, hand_slot=0, x=x, y=y)], 2)
    if results[0].status != DeployStatus.OK:
        pytest.skip("the multi-unit tap was refused, so the group case is untested here")
    assert len(land[0].entities) > 1, (
        f"expected more than one entity from a multi-unit card, got {len(land[0].entities)}"
    )
    assert land[0].x is None
    assert land[0].y is None
    assert land[0].displacement is None
    assert "group" in land[0].reason


def test_a_walking_unit_is_reported_where_it_LANDED_not_where_it_went() -> None:
    """The reading mistake this function was one caller away from making.

    Positions read after a whole decision's ticks are where a troop WALKED to, not where
    it landed, and nothing about the number looks wrong. A building does not move, so the
    defect is invisible in exactly the case the module was written for.

    Graded by comparing a long step against a one-tick step for the same deploy, rather
    than against a pinned coordinate: a literal would pin this engine's formation and go
    red on any formation change, which is a different claim than the one being made here.
    """
    eng, idx = engine_with("Minions")
    x, y = tap(9, 8)

    eng.reset(seed=0, setup=battle_setup(eng, idx))
    eng.step([], 10)
    _, short = landings(eng, [DeployCommand(team=0, hand_slot=0, x=x, y=y)], 1)

    eng.reset(seed=0, setup=battle_setup(eng, idx))
    eng.step([], 10)
    _, long_ = landings(eng, [DeployCommand(team=0, hand_slot=0, x=x, y=y)], 20)

    if not short[0].entities or not long_[0].entities:
        pytest.skip("the Minions tap was refused, so this measured nothing; a skip is not a pass")
    at_landing = sorted((e.x, e.y) for e in short[0].entities)
    reported = sorted((e.x, e.y) for e in long_[0].entities)
    assert reported == at_landing, (
        f"a 20-tick step reported {reported} and the deploy tick has them at {at_landing}. "
        "The longer step is reading where the units walked to. That is the defect: it is "
        "invisible for a building, which does not move, and a building is the case this "
        "module was written for."
    )
    # Non-vacuity: if these units never move, the test above passes for the wrong reason.
    eng.reset(seed=0, setup=battle_setup(eng, idx))
    eng.step([], 10)
    before = {e.uid for e in eng.state().entities}
    eng.step([DeployCommand(team=0, hand_slot=0, x=x, y=y)], 1)
    mine = [e.uid for e in eng.state().entities if e.uid not in before]
    eng.step([], 60)
    after = sorted((e.x, e.y) for e in eng.state().entities if e.uid in mine)
    assert after != at_landing, (
        "these units did not move at all over 60 further ticks, so the comparison above "
        "could not have told a landing from a walk. Use a card that walks."
    )
