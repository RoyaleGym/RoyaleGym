"""placement.ILLEGAL_TROOP_TAP (RoyaleSim 0.1.29): a troop tapped out of its territory is clamped.

Under ``client15535_clamp_to_legal_edge`` the engine accepts a TROOP card (by kind) tapped out of
its territory, inside an alive enemy tower's no-deploy rect or (closed-band model) on the river
band, moving the tap back along its tile column to the first legal tile. Water, off the arena,
occupied ground and the rest stay refused, and so does a column with no legal tile.

The action mask keeps REFUSING those taps: offering them would give a policy a whole column of
actions that all land on one edge tile, which the edge tile already offers. So the mask is
deliberately stricter than the engine there, and ``mask_disagreements`` counts such an
acceptance as intended under that arm, and only that one: a disagreement anywhere else still
counts. The arm is read from the ledger (``DeployRules.illegal_troop_tap``), and an override of
it is followed; before 0.1.29 the ledger has no key and the engine refuses, as "refuse" says.

Two engines grade it. The core's own clamp, through ``calibration_overrides`` (skipped, not
passed, on a ledger without the key). And a simulation, on any engine, that turns its own
OUT_OF_TERRITORY refusals of troop-kind cards into acceptances: exactly the set the clamp can
accept, whose real refinements (a column with no legal tile, a line card) only shrink it, and
which plants acceptances the exemption must NOT cover (water, spells).

The clamp also hid which cards travel under ground: ``rust_engine.tunnelling_cards`` asks whether
the enemy half takes a card, and under the clamp it takes every troop. Its probe refuses.

SKIPS
    The battle tests need the engine. Not a pass without it.
"""

from __future__ import annotations

import copy

import msgspec
import numpy as np
import pytest

from royalegym.action import HalfTileActionParser, TileActionParser, mask_disagreements
from royalegym.mock_engine import MockEngine
from royalegym.protocol import (
    BLUE,
    CLOSED_RIVER_BAND,
    ILLEGAL_TROOP_TAP,
    RED,
    TERRITORY_MODELS,
    Calibration,
    DeployRules,
    DeployStatus,
    EntityKind,
    MatchSetup,
    Placement,
    ShuffleMode,
)
from royalegym.rust_engine import (
    CORE_IMPORT_ERROR,
    OVERRIDE_FOLLOWED,
    RustEngine,
    core_available,
)
from royalegym.state_mutator import DefaultStateMutator

CLAMP = "client15535_clamp_to_legal_edge"
needs_engine = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))


def _ledger_with(value):
    raw = copy.deepcopy(Calibration.load().raw)
    raw.setdefault("placement", {})
    if value is None:
        raw["placement"].pop("ILLEGAL_TROOP_TAP", None)
    else:
        raw["placement"]["ILLEGAL_TROOP_TAP"] = {
            "value": value, "candidates": list(ILLEGAL_TROOP_TAP)}
    return Calibration(raw)


def test_the_arm_is_read_from_the_ledger_and_an_unknown_one_refused():
    assert ILLEGAL_TROOP_TAP == ("refuse", CLAMP)
    assert DeployRules.load(_ledger_with(None)).illegal_troop_tap == "refuse"
    assert DeployRules.load(_ledger_with("refuse")).illegal_troop_tap == "refuse"
    assert DeployRules.load(_ledger_with(CLAMP)).illegal_troop_tap == CLAMP
    with pytest.raises(NotImplementedError, match="ILLEGAL_TROOP_TAP"):
        DeployRules.load(_ledger_with("teleport_to_the_king"))


class _Clamping:
    """The bound engine, with its OUT_OF_TERRITORY refusals of a troop-kind card accepted, as
    the clamp arm accepts them. Planted disagreements the exemption must NOT cover: with
    ``water`` its WATER refusals are accepted too, with ``every_card`` a spell's
    OUT_OF_TERRITORY refusals as well."""

    def __init__(self, engine, water=False, every_card=False):
        self._engine = engine
        self._water = water
        self._every_card = every_card

    def __getattr__(self, name):
        return getattr(self._engine, name)

    def check_deploy(self, cmd):
        status = self._engine.check_deploy(cmd)
        player = self._engine.state().players[cmd.team]
        if cmd.hand_slot >= len(player.hand):
            return status
        card = self._engine.cards()[player.hand[cmd.hand_slot]]
        if card.name == "Mirror" and player.mirror_target >= 0:
            card = self._engine.cards()[player.mirror_target]
        troop = self._every_card or card.card_kind in (None, "TROOP")
        refused = {DeployStatus.OUT_OF_TERRITORY} | ({DeployStatus.WATER} if self._water else set())
        return DeployStatus.OK if troop and status in refused else status


DECK = ["Knight", "Archer", "Goblins", "Giant", "Minions", "Musketeer", "Valkyrie", "Fireball"]


def _board(parser_cls=TileActionParser, tower_hp=None, deck=DECK, shuffle=ShuffleMode.INDEPENDENT):
    """A RustEngine past the opening lockout, both hands holding troops, nothing placed."""
    engine = RustEngine()
    mutator = DefaultStateMutator(decks=[deck, deck], shuffle=shuffle)
    setup = mutator.build(np.random.default_rng(0), engine.cards())
    setup = msgspec.structs.replace(setup, elixir_milli=[100000, 100000], tower_hp=tower_hp)
    engine.reset(3, setup)
    engine.step([], engine.rules().deploy_lockout_ticks + 20)
    parser = parser_cls()
    parser.bind(engine)
    return engine, parser


def _with_arm(parser, arm):
    parser.oracle.rules = msgspec.structs.replace(parser.oracle.rules, illegal_troop_tap=arm)
    return parser


@needs_engine
@pytest.mark.parametrize("team", [BLUE, RED])
def test_under_the_clamp_an_out_of_territory_troop_acceptance_is_intended(team):
    engine, parser = _board()
    state = engine.state()
    clamping = _Clamping(engine)
    # Vacuity: under "refuse" the simulated clamp's acceptances ARE disagreements, many of them.
    found = mask_disagreements(clamping, _with_arm(parser, "refuse"), state, team)
    assert len(found) > 50, len(found)
    assert all(m == 0 and s == DeployStatus.OK for _, m, s in found)
    # Under the clamp arm every one of them is intended.
    assert mask_disagreements(clamping, _with_arm(parser, CLAMP), state, team) == []
    # And the real engine (which refuses) still agrees with the mask under either arm.
    assert mask_disagreements(engine, _with_arm(parser, CLAMP), state, team) == []


@needs_engine
def test_the_exemption_covers_out_of_territory_only():
    """A troop accepted on WATER is still a disagreement under the clamp arm."""
    engine, parser = _board()
    state = engine.state()
    found = mask_disagreements(_Clamping(engine, water=True), _with_arm(parser, CLAMP), state, BLUE)
    assert found, "accepting troops on water went unreported"
    assert all(s == DeployStatus.OK and m == 0 for _, m, s in found)


@needs_engine
def test_the_exemption_covers_troops_only():
    """A Log (rolling) or a Heal (a spell with a troop's placement) refused for territory and
    accepted anyway is still a disagreement under the clamp arm: the clamp is for troops."""
    deck = ["Knight", "Log", "Heal", "Archer", "Giant", "Minions", "Musketeer", "Fireball"]
    engine, parser = _board(deck=deck, shuffle=ShuffleMode.NONE)
    state = engine.state()
    hand = [engine.cards()[c].name for c in state.players[BLUE].hand]
    assert {"Log", "Heal", "Knight"} <= set(hand), hand
    clamping = _Clamping(engine, every_card=True)
    found = mask_disagreements(clamping, _with_arm(parser, CLAMP), state, BLUE)
    names = {hand[parser.parse(a, state, BLUE).hand_slot] for a, _, _ in found}
    assert names == {"Log", "Heal"}, names


def _verdicts_agree(engine, parser, team):
    """Every troop tap of ``team``: ``out_of_territory`` against the engine's own status.
    Returns (disagreements, taps it calls out of territory, of those on an enemy tower's
    body, troop slots seen)."""
    state = engine.state()
    oracle = parser.oracle
    player = state.players[team]
    wrong, hits, on_towers, slots = [], 0, 0, set()
    for action in range(1, int(parser.space.n)):
        cmd = parser.parse(action, state, team)
        if cmd is None or cmd.hand_slot >= len(player.hand):
            continue
        card = oracle.cards[player.hand[cmd.hand_slot]]
        if not oracle._troop_laws(card):
            continue
        slots.add(cmd.hand_slot)
        point = np.asarray([cmd.x]), np.asarray([cmd.y])
        said = bool(oracle.out_of_territory(state, team, *point)[0])
        status = engine.check_deploy(cmd)
        if said != (status == DeployStatus.OUT_OF_TERRITORY):
            wrong.append((action, card.name, cmd.x, cmd.y, int(status), said))
        hits += said
        on_towers += said and any(
            (e.x - cmd.x) ** 2 + (e.y - cmd.y) ** 2 <= e.radius**2
            for e in state.entities
            if e.team != team and e.kind != EntityKind.TROOP
        )
    return wrong, hits, on_towers, slots


@needs_engine
@pytest.mark.parametrize("parser_cls", [TileActionParser, HalfTileActionParser])
@pytest.mark.parametrize("team", [BLUE, RED])
def test_out_of_territory_is_the_engines_own_verdict(parser_cls, team):
    """``PlacementOracle.out_of_territory`` is exactly the set the engine refuses with
    OUT_OF_TERRITORY, both ways, for every troop tap of both seats: with one Blue princess
    down (the pocket in front of it is Red's ground) and a tap on an enemy tower's body,
    which the engine refuses for territory before any footprint is looked at."""
    engine, parser = _board(parser_cls, tower_hp=[[2400, 0, 1400], [2400, 1400, 1400]])
    wrong, hits, on_towers, slots = _verdicts_agree(engine, parser, team)
    assert wrong == [], (len(wrong), wrong[:5])
    assert slots, "no troop in the hand: nothing was checked"
    assert hits > 100 * len(slots), hits
    assert on_towers > 0, "no tap on an enemy tower's body was checked"


@pytest.mark.parametrize("model", TERRITORY_MODELS)
@pytest.mark.parametrize("team", [BLUE, RED])
def test_out_of_territory_under_both_territory_models(model, team):
    """The same on MockEngine (Arena::deploy_zone's order), under the closed river band as
    well as the open bridge the ledger ships, with one Blue princess down. Under the closed
    band the river's half-rows are out of territory even on the fallen lane's bridge."""
    engine = MockEngine()
    engine._rules = msgspec.structs.replace(engine._rules, territory_model=model)
    ids = {c.name: c.card_id for c in engine.cards()}
    names = ["Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Cannon", "Musketeer"]
    deck = [ids[n] for n in names]
    engine.reset(3, MatchSetup(
        decks=[deck, deck], shuffle=ShuffleMode.NONE, elixir_milli=[100000, 100000],
        start_tick=engine.rules().deploy_lockout_ticks,
        tower_hp=[[2400, 0, 1400], [2400, 1400, 1400]]))
    parser = TileActionParser()
    parser.bind(engine)
    wrong, hits, _, slots = _verdicts_agree(engine, parser, team)
    assert wrong == [], (len(wrong), wrong[:5])
    assert slots, "no troop in the hand: nothing was checked"
    assert hits > 100 * len(slots), (slots, hits)
    if model == CLOSED_RIVER_BAND:
        a = engine.arena()
        lo, hi = a.water_half_rows
        river = [(x, hy * a.half_size + a.half_size // 2)
                 for x in a.bridge_centers_x() for hy in range(lo, hi + 1)]
        px, py = np.asarray([p[0] for p in river]), np.asarray([p[1] for p in river])
        assert parser.oracle.out_of_territory(engine.state(), team, px, py).all()


def test_an_override_of_the_arm_is_followed():
    """An engine built with placement.ILLEGAL_TROOP_TAP overridden states that arm in its
    rules, so the mask reads the arm the core runs (``OVERRIDE_FOLLOWED``)."""
    assert "placement.ILLEGAL_TROOP_TAP" in OVERRIDE_FOLLOWED


def _needs_the_key():
    ledger = Calibration.load().raw.get("placement", {})
    if "ILLEGAL_TROOP_TAP" not in ledger:
        pytest.skip("SKIPPED, NOT PASSED: this engine's ledger has no placement.ILLEGAL_TROOP_TAP")
    return ledger["ILLEGAL_TROOP_TAP"]["value"]


def _real(arm, parser_cls=TileActionParser, tower_hp=None):
    """``_board`` on a RustEngine running ``arm`` (the core's own clamp, not a simulation)."""
    engine = RustEngine(calibration_overrides={"placement.ILLEGAL_TROOP_TAP": arm})
    setup = DefaultStateMutator(decks=[DECK, DECK]).build(np.random.default_rng(0), engine.cards())
    setup = msgspec.structs.replace(setup, elixir_milli=[100000, 100000], tower_hp=tower_hp)
    engine.reset(3, setup)
    engine.step([], engine.rules().deploy_lockout_ticks + 20)
    parser = parser_cls()
    parser.bind(engine)
    return engine, parser


@needs_engine
def test_on_rustengine_the_clamp_leaves_every_cards_placement_alone():
    """The tunnel probe asks whether the enemy half is a card's territory, and under the clamp
    the core accepts every troop tapped there: the probe battle refuses such taps, so a
    Knight stays a TROOP and only the cards that travel under ground are TUNNEL."""
    _needs_the_key()
    placements = {
        arm: {c.name: c.placement for c in RustEngine(
            calibration_overrides={"placement.ILLEGAL_TROOP_TAP": arm}).cards()}
        for arm in ILLEGAL_TROOP_TAP
    }
    assert placements["refuse"] == placements[CLAMP]
    assert placements[CLAMP]["Knight"] == Placement.TROOP
    assert placements[CLAMP]["Miner"] == Placement.TUNNEL


@needs_engine
@pytest.mark.parametrize("parser_cls", [TileActionParser, HalfTileActionParser])
@pytest.mark.parametrize("down", [False, True], ids=["princesses-up", "blue-princess-down"])
@pytest.mark.parametrize("team", [BLUE, RED])
def test_on_rustengine_the_real_clamp_agrees_with_the_mask(parser_cls, down, team):
    """The core's own clamp: every acceptance the mask refuses is one the clamp moved, and
    nothing else disagrees. Graded as if the arm were refuse, the same battle disagrees
    wherever the clamp accepts, which is what makes the first assertion say something."""
    _needs_the_key()
    hp = [[2400, 0, 1400], [2400, 1400, 1400]] if down else None
    engine, parser = _real(CLAMP, parser_cls, hp)
    state = engine.state()
    assert parser.oracle.rules.illegal_troop_tap == CLAMP
    assert mask_disagreements(engine, parser, state, team) == []
    found = mask_disagreements(engine, _with_arm(parser, "refuse"), state, team)
    assert len(found) > 50, len(found)
    assert {(m, s) for _, m, s in found} == {(0, DeployStatus.OK)}


@needs_engine
def test_on_rustengine_an_override_of_the_arm_reaches_the_rules():
    shipped = _needs_the_key()
    other = next(a for a in ILLEGAL_TROOP_TAP if a != shipped)
    engine = RustEngine(calibration_overrides={"placement.ILLEGAL_TROOP_TAP": other})
    assert engine.rules().illegal_troop_tap == other
    assert RustEngine().rules().illegal_troop_tap == shipped
