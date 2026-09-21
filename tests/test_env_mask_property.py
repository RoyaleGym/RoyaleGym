"""Property test: the action mask and ``Engine.check_deploy`` agree on GENERATED battles.

WHY IT EXISTS
    A silently wrong mask is the most expensive bug in the RL layer: a mask that
    allows an illegal move trains the agent to want it, and a mask that forbids a
    legal move hides part of the game. Neither shows up as an error -- only as a
    worse agent, many hours later. tests/test_env_action_mask.py checks agreement
    exhaustively on six hand-built probe states; this file checks it on battles
    hypothesis GENERATES (random decks and shuffle modes, starting elixir, tower
    HP with princesses already down so pockets open, buildings spawned at arbitrary
    subtile positions so footprints cut through tile centres, start ticks up to the
    end of overtime, then a random rollout), so the board is not the one the mask's
    author had in mind.

WHAT IT CHECKS, per generated state, for both seats and both parsers
    * shape (space.n,), dtype int8, values in {0, 1}, no-op always legal;
    * EVERY action the mask marks legal gets DeployStatus.OK from check_deploy;
    * a sample of actions the mask marks illegal (up to ``ILLEGAL_SAMPLE``) is
      rejected;
    * for the first and last legal action, ``step`` itself accepts the command
      (check_deploy is the protocol's promise about step; this holds it to it).

WHY BOTH DIRECTIONS
    An all-zero mask passes "legal => accepted" vacuously. The plants below show
    each direction catching a defect the other cannot see.

REPRODUCIBILITY
    ``derandomize=True`` and no example database: the same examples every run, so
    a red run is reproducible. (Hypothesis still writes a source-constants cache
    to .hypothesis/constants, which its own .hypothesis/.gitignore keeps out of
    git.) To search wider for a new bug, set CLASHENV_MASK_EXAMPLES (e.g. 2000)
    and CLASHENV_MASK_RANDOM=1 for one run; a failure prints the falsifying
    example, which should then become a fixed probe in
    tests/test_env_action_mask.py. The vacuity floors are calibrated for the
    derandomized default only.

WHAT IT CANNOT CATCH
    A rule the mask and the engine BOTH get wrong the same way (e.g. both keep
    the river band closed to troops, a guess -- DeployRules.territory_status).
    Agreement is not fidelity; only the oracle can say what
    the real game allows.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass

import msgspec
import numpy as np
import pytest
from hypothesis import HealthCheck, Phase, find, given, settings
from hypothesis import strategies as st
from hypothesis.errors import NoSuchExample

from royalegym import action as action_mod
from royalegym.action import (
    NOOP,
    GridActionParser,
    HalfTileActionParser,
    PlacementOracle,
    TileActionParser,
)
from royalegym.mock_engine import MockEngine
from royalegym.protocol import (
    DECK_SIZE,
    RED,
    TEAMS,
    BattleState,
    DeployStatus,
    MatchSetup,
    Placement,
    ShuffleMode,
    SpawnSpec,
    TowerSlot,
    spawn_violation,
)

# Everything below reaches the engine ONLY through protocol.Engine, so pointing this
# file at another engine (the Rust core) is a one-line change here. The
# rejection-reason coverage test and the named-status plants additionally assume
# the engine reports the fine-grained DeployStatus codes.
ENGINE = MockEngine()
PARSERS: dict[str, GridActionParser] = {"tile": TileActionParser(), "half": HalfTileActionParser()}
for _p in PARSERS.values():
    _p.bind(ENGINE)

ARENA = ENGINE.arena()
CARDS = list(ENGINE.cards())
UNIT_CARDS = [c.card_id for c in CARDS if c.placement in (Placement.TROOP, Placement.BUILDING)]
BUILDING_CARDS = [c.card_id for c in CARDS if c.placement == Placement.BUILDING]
ENGINE.reset(0, MatchSetup(decks=[list(range(DECK_SIZE))] * 2))
_S0 = ENGINE.state()
KING_HP = _S0.players[0].tower_max_hp[TowerSlot.KING]
PRINCESS_HP = _S0.players[0].tower_max_hp[TowerSlot.LEFT]
REGULAR_TICKS = _S0.regular_ticks
END_TICK = _S0.regular_ticks + _S0.overtime_ticks
ILLEGAL_SAMPLE = 256

EXAMPLES = int(os.environ.get("CLASHENV_MASK_EXAMPLES", "0"))
BASE = settings(
    derandomize=os.environ.get("CLASHENV_MASK_RANDOM") != "1",
    database=None,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
# Plants and coverage searches want the FIRST hit, not a shrunk one.
SEARCH = settings(BASE, phases=[Phase.generate], max_examples=300)


@dataclass(frozen=True)
class Battle:
    seed: int
    setup: MatchSetup
    steps: int
    chunk: int
    play_prob_pct: int
    policy_seed: int
    settle_ticks: int


@st.composite
def battles(draw: st.DrawFn) -> Battle:
    """Generated battles, biased toward boards where much is legal.

    An earlier generator drew elixir uniformly and let the
    rollout spend it, and 49 of 60 derandomized examples had NO legal deploy for
    either team -- the "legal => accepted" direction was checked on nothing. The
    elixir bias, the no-command ``settle_ticks`` after the rollout (elixir
    regenerates), and the vacuity floor in the property tests are the fix.
    """
    decks = [list(draw(st.permutations(range(len(CARDS))))[:DECK_SIZE]) for _ in TEAMS]
    princess = st.one_of(st.just(PRINCESS_HP), st.just(0), st.integers(1, PRINCESS_HP))
    tower_hp = [
        [
            draw(st.one_of(st.just(KING_HP), st.integers(1, KING_HP))),
            draw(princess),
            draw(princess),
        ]
        for _ in TEAMS
    ]
    spawn = st.builds(
        SpawnSpec,
        team=st.sampled_from(TEAMS),
        card_id=st.one_of(st.sampled_from(BUILDING_CARDS), st.sampled_from(UNIT_CARDS)),
        x=st.integers(1, ARENA.width - 1),
        y=st.integers(1, ARENA.height - 1),
        # Every engine refuses a ground unit or building spawned
        # touching water (protocol.spawn_violation); such a setup is not a battle.
    ).filter(lambda sp: spawn_violation(ARENA, CARDS, sp) is None)
    elixir = st.one_of(st.just(10_000), st.integers(4_000, 10_000), st.integers(0, 10_000))
    setup = MatchSetup(
        decks=decks,
        shuffle=int(draw(st.sampled_from(list(ShuffleMode)))),
        start_tick=draw(
            st.one_of(
                st.just(0),
                st.integers(0, REGULAR_TICKS - 1),
                # Overtime is sudden death: with a princess already down it ends on
                # the first tick, so it is drawn rarely -- but it is drawn.
                st.integers(0, END_TICK),
            )
        ),
        elixir_milli=[draw(elixir) for _ in TEAMS],
        tower_hp=tower_hp,
        spawns=draw(st.lists(spawn, max_size=6)),
    )
    return Battle(
        seed=draw(st.integers(0, 2**32 - 1)),
        setup=setup,
        steps=draw(st.integers(0, 25)),
        chunk=draw(st.integers(1, 12)),
        play_prob_pct=draw(st.integers(0, 100)),
        policy_seed=draw(st.integers(0, 2**32 - 1)),
        settle_ticks=draw(st.one_of(st.just(0), st.integers(0, 300))),
    )


def build_state(b: Battle, parser: GridActionParser) -> BattleState:
    ENGINE.reset(b.seed, b.setup)
    rng = np.random.default_rng(b.policy_seed)
    for _ in range(b.steps):
        s = ENGINE.state()
        if s.game_over:
            break
        cmds = []
        for team in TEAMS:
            legal = np.flatnonzero(parser.action_mask(s, team))[1:]
            if legal.size and rng.integers(100) < b.play_prob_pct:
                cmds.append(parser.parse(int(rng.choice(legal)), s, team))
        ENGINE.step(cmds, b.chunk)
    if b.settle_ticks and not ENGINE.state().game_over:
        ENGINE.step([], b.settle_ticks)
    return ENGINE.state()


@dataclass
class Report:
    problems: list[str]
    legal_checked: int
    illegal_statuses: Counter


def check_battle(b: Battle, parser: GridActionParser) -> Report:
    state = build_state(b, parser)
    problems: list[str] = []
    legal_checked = 0
    illegal_statuses: Counter = Counter()
    blob = ENGINE.save_state()
    for team in TEAMS:
        mask = parser.action_mask(state, team)
        if mask.shape != (parser.space.n,) or mask.dtype != np.int8:
            problems.append(f"team {team}: mask shape/dtype {mask.shape} {mask.dtype}")
            continue
        if not np.isin(mask, (0, 1)).all():
            problems.append(f"team {team}: mask values outside {{0, 1}}")
        if mask[NOOP] != 1:
            problems.append(f"team {team}: no-op masked out")
        legal = np.flatnonzero(mask)
        legal = legal[legal != NOOP]
        illegal = np.flatnonzero(mask == 0)
        for a in legal:
            status = ENGINE.check_deploy(parser.parse(int(a), state, team))
            legal_checked += 1
            if status != DeployStatus.OK:
                problems.append(
                    f"team {team}: legal action {a} rejected {DeployStatus(status).name}"
                )
        rng = np.random.default_rng((b.policy_seed, team))
        if illegal.size > ILLEGAL_SAMPLE:
            illegal = rng.choice(illegal, ILLEGAL_SAMPLE, replace=False)
        for a in illegal:
            status = ENGINE.check_deploy(parser.parse(int(a), state, team))
            illegal_statuses[DeployStatus(status)] += 1
            if status == DeployStatus.OK:
                problems.append(f"team {team}: illegal action {a} accepted")
        for a in sorted({int(legal[0]), int(legal[-1])}) if legal.size else ():
            (result,) = ENGINE.step([parser.parse(a, state, team)], 1)
            ENGINE.load_state(blob)
            if result.status != DeployStatus.OK:
                problems.append(
                    f"team {team}: step rejected legal {a} {DeployStatus(result.status).name}"
                )
    assert ENGINE.save_state() == blob, "the check itself mutated the battle"
    return Report(problems, legal_checked, illegal_statuses)


# --------------------------------------------------------------------------
# the property
# --------------------------------------------------------------------------


def run_property(parser_key: str, max_examples: int) -> list[Report]:
    """Run the property over derandomized examples; return every example's report."""
    reports: list[Report] = []

    @settings(BASE, max_examples=EXAMPLES or max_examples)
    @given(battles())
    def prop(b: Battle) -> None:
        r = check_battle(b, PARSERS[parser_key])
        reports.append(r)
        assert r.problems == [], r.problems[:10]

    prop()
    return reports


def vacuity_problems(reports: list[Report], min_informative_pct: int, min_legal: int) -> list[str]:
    """The property must have checked a lot of legal actions, on most examples."""
    informative = sum(1 for r in reports if r.legal_checked > 0)
    total = sum(r.legal_checked for r in reports)
    out = []
    if informative * 100 < min_informative_pct * len(reports):
        out.append(f"only {informative}/{len(reports)} examples had any legal deploy")
    if total < min_legal:
        out.append(f"only {total} legal actions checked (floor {min_legal})")
    return out


# Floors sit below what the derandomized examples measure. The
# example set is NOT frozen by derandomize alone: editing code in this file moved it
# (most likely hypothesis mining source literals, cf. .hypothesis/constants) -- 35/60 tile
# examples informative (78 412 legal actions) before one edit, 45/60 (89 166) after;
# half-tile 18/25 (110 696) then 23/25 (181 427). Re-measure after an edit that
# touches literals. A one-off randomized run (1500 tile + 300 half-tile examples,
# 2 141 592 and 1 648 810 legal actions) was also clean. Re-measured after the spawn
# generator gained its water filter and territory became the NoDeploySize rects: tile
# 41/60 informative (86 579 legal), half-tile 15/25 (128 183). Re-measured again after
# MockEngine's catalogue gained GoblinBarrel (16 cards, so the deck permutations moved
# every example): tile 43/60 (84 840), half-tile 11/25 (84 415) -- under the floor, so
# the half-tile run uses 40 examples: 24/40 (221 479 legal), 1.2 s.
MIN_INFORMATIVE_PCT = 50
MIN_LEGAL_CHECKED = 50_000


def test_tile_mask_agrees_with_check_deploy_on_generated_battles():
    reports = run_property("tile", 60)
    assert vacuity_problems(reports, MIN_INFORMATIVE_PCT, MIN_LEGAL_CHECKED) == []


def test_half_tile_mask_agrees_with_check_deploy_on_generated_battles():
    reports = run_property("half", 40)
    assert vacuity_problems(reports, MIN_INFORMATIVE_PCT, MIN_LEGAL_CHECKED) == []


def test_plant_vacuity_floor_rejects_the_first_generator():
    """Regression plant: the first generator's measured shape (49 of 60 examples
    with nothing legal) must trip the floor."""
    measured = [920, 920, 236, 238, 240, 231, 1738, 1738, 1738, 940, 1842] + [0] * 49
    starved = [Report([], n, Counter()) for n in measured]
    assert vacuity_problems(starved, MIN_INFORMATIVE_PCT, MIN_LEGAL_CHECKED) == [
        "only 11/60 examples had any legal deploy",
        "only 10781 legal actions checked (floor 50000)",
    ]


# --------------------------------------------------------------------------
# vacuity: the generator reaches every branch the mask has to model
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [
        DeployStatus.NOT_ENOUGH_ELIXIR,
        DeployStatus.WATER,
        DeployStatus.NO_DEPLOY,
        DeployStatus.OUT_OF_TERRITORY,
        DeployStatus.OCCUPIED,
        DeployStatus.GAME_OVER,
    ],
    ids=lambda s: s.name,
)
def test_generator_reaches_every_rejection_reason(status):
    find(
        battles(),
        lambda b: check_battle(b, PARSERS["tile"]).illegal_statuses[status] > 0,
        settings=SEARCH,
    )


def _pocket_troop_is_legal(b: Battle) -> bool:
    """A troop is legal past the river: a princess is down AND the mask opened a pocket."""
    parser = PARSERS["tile"]
    state = build_state(b, parser)
    for team in TEAMS:
        mask = parser.action_mask(state, team)
        per = parser.nx * parser.ny
        for slot, card_id in enumerate(state.players[team].hand):
            if CARDS[card_id].placement != Placement.TROOP:
                continue
            grid = mask[1 + slot * per : 1 + (slot + 1) * per].reshape(parser.ny, parser.nx)
            if grid[ARENA.tiles_y // 2 + 1 :].any():  # own frame: rows past the river
                return True
    return False


def test_generator_reaches_open_pockets():
    find(battles(), _pocket_troop_is_legal, settings=SEARCH)


def test_generator_reaches_many_legal_actions_and_mid_battle_boards():
    find(
        battles(),
        lambda b: b.steps >= 10 and check_battle(b, PARSERS["tile"]).legal_checked > 2000,
        settings=SEARCH,
    )


# --------------------------------------------------------------------------
# plants: each breaks the MASK only; the aimed direction must catch it, and the
# unplanted mask must be clean on the very example that caught it
# --------------------------------------------------------------------------


def _landed(monkeypatch, parser_key: str, aimed: str) -> list[str]:
    """Find an example the planted mask fails on; return the planted problems.

    Then undo the plant and require the SAME example to be clean, so the red is
    the plant's and not a latent defect the search happened to trip over.
    """
    parser = PARSERS[parser_key]

    def hit(b: Battle) -> bool:
        return any(aimed in p for p in check_battle(b, parser).problems)

    try:
        example = find(battles(), hit, settings=SEARCH)
    except NoSuchExample as exc:
        pytest.fail(f"PLANT DID NOT LAND ({aimed!r}): {exc}")
    planted = check_battle(example, parser).problems
    monkeypatch.undo()
    assert check_battle(example, parser).problems == [], "baseline not green on the caught example"
    return planted


def test_plant_mask_ignores_elixir_is_caught_by_the_legal_direction(monkeypatch):
    orig = GridActionParser.action_mask

    def rich(self, state, team):
        players = [msgspec.structs.replace(p, elixir_milli=10**6) for p in state.players]
        return orig(self, msgspec.structs.replace(state, players=players), team)

    monkeypatch.setattr(GridActionParser, "action_mask", rich)
    assert GridActionParser.action_mask is rich
    _landed(monkeypatch, "tile", "rejected NOT_ENOUGH_ELIXIR")


def test_plant_all_zero_mask_is_caught_only_by_the_illegal_direction(monkeypatch):
    def only_noop(self, state, team):
        m = np.zeros(self.space.n, dtype=np.int8)
        m[NOOP] = 1
        return m

    monkeypatch.setattr(GridActionParser, "action_mask", only_noop)
    assert GridActionParser.action_mask is only_noop
    planted = _landed(monkeypatch, "tile", "illegal action")
    # ...and the legal direction, alone, would have passed it.
    assert not any("legal action" in p and "illegal" not in p for p in planted), planted


def test_plant_mask_ignores_building_footprints(monkeypatch):
    orig = PlacementOracle.point_grid

    def no_footprints(self, state, team, card, pitch_div):
        return orig(self, msgspec.structs.replace(state, entities=[]), team, card, pitch_div)

    monkeypatch.setattr(PlacementOracle, "point_grid", no_footprints)
    assert PlacementOracle.point_grid is no_footprints
    _landed(monkeypatch, "half", "rejected OCCUPIED")


def test_plant_red_mask_left_in_the_engine_frame(monkeypatch):
    monkeypatch.setattr(action_mod, "RED", -1)  # the parser never rotates Red's grid
    assert action_mod.RED != RED
    _landed(monkeypatch, "tile", "team 1: legal action")
