"""The clamps: a tap out of its territory moved back to the first legal tile of its column.

placement.ILLEGAL_TROOP_TAP (RoyaleSim 0.1.29). Under ``client15535_clamp_to_legal_edge`` the
engine accepts a TROOP card (by kind) tapped out of its territory, inside an alive enemy tower's
no-deploy rect or (closed-band model) on the river band.

spells.ILLEGAL_SPELL_TAP. Under ``client16402_clamp_to_legal_edge`` it accepts a spell judged in
troop territory (the Log, the Barbarian Barrel, the Royal Delivery, the Heal) tapped out of that
territory or on water, which the zone check reports before the territory.

Each moves the tap back along its tile column to the first legal tile. Off the arena, no-deploy
ground, occupied ground and the rest stay refused, and so does a column with no legal tile.

The action mask keeps REFUSING those taps: offering them would give a policy a whole column of
actions that all land on one edge tile, which the edge tile already offers. So the mask is
deliberately stricter than the engine there, and ``mask_disagreements`` counts such an
acceptance as intended under its arm, and only that: a disagreement anywhere else still
counts. The arms are read from the ledger (``DeployRules.illegal_troop_tap``,
``illegal_spell_tap``), and an override of either is followed.

Two engines grade it. The core's own clamps, through ``calibration_overrides`` (skipped, not
passed, on a ledger without the key). And a simulation, on any engine, that turns its own
OUT_OF_TERRITORY refusals of troop-kind cards into acceptances: exactly the set the troop clamp
can accept, whose real refinements (a column with no legal tile, a line card) only shrink it,
and which plants acceptances the exemption must NOT cover (a troop on water, a spell under the
troop arm).

The troop clamp also hid which cards travel under ground: ``rust_engine.tunnelling_cards`` asks
whether the enemy half takes a card, and under the clamp it takes every troop. Its probe refuses.

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
    ILLEGAL_SPELL_TAP,
    ILLEGAL_TROOP_TAP,
    RED,
    TERRITORY_MODELS,
    Calibration,
    DeployCommand,
    DeployRules,
    DeployStatus,
    EntityKind,
    MatchSetup,
    Placement,
    ShuffleMode,
    card_is_spell,
)
from royalegym.rust_engine import (
    CORE_IMPORT_ERROR,
    OVERRIDE_FOLLOWED,
    RustEngine,
    core_available,
)
from royalegym.state_mutator import DefaultStateMutator

CLAMP = "client15535_clamp_to_legal_edge"
SPELL_CLAMP = "client16402_clamp_to_legal_edge"
TROOP_KEY = "placement.ILLEGAL_TROOP_TAP"
SPELL_KEY = "spells.ILLEGAL_SPELL_TAP"
#: The zone check's refusals, in its order (Arena::deploy_zone).
ZONE = (
    DeployStatus.OUT_OF_ARENA,
    DeployStatus.WATER,
    DeployStatus.NO_DEPLOY,
    DeployStatus.OUT_OF_TERRITORY,
)
needs_engine = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))


def _ledger_with(key, value, arms):
    section, name = key.split(".")
    raw = copy.deepcopy(Calibration.load().raw)
    raw.setdefault(section, {})
    if value is None:
        raw[section].pop(name, None)
    else:
        raw[section][name] = {"value": value, "candidates": list(arms)}
    return Calibration(raw)


@pytest.mark.parametrize(
    ("key", "arms", "clamp", "field"),
    [
        (TROOP_KEY, ILLEGAL_TROOP_TAP, CLAMP, "illegal_troop_tap"),
        (SPELL_KEY, ILLEGAL_SPELL_TAP, SPELL_CLAMP, "illegal_spell_tap"),
    ],
)
def test_the_arm_is_read_from_the_ledger_and_an_unknown_one_refused(key, arms, clamp, field):
    assert arms == ("refuse", clamp)
    for value, want in ((None, "refuse"), ("refuse", "refuse"), (clamp, clamp)):
        assert getattr(DeployRules.load(_ledger_with(key, value, arms)), field) == want
    with pytest.raises(NotImplementedError, match=key.split(".")[1]):
        DeployRules.load(_ledger_with(key, "teleport_to_the_king", arms))


class _Clamping:
    """The bound engine, with its OUT_OF_TERRITORY refusals of a troop-kind card accepted, as
    the troop clamp accepts them. Planted disagreements the exemption must NOT cover: with
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
#: Dealt in order (ShuffleMode.NONE): a hand of the four spells judged in troop territory.
SPELL_DECK = ["Log", "Heal", "BarbLog", "RoyalDelivery", "Knight", "Archer", "Giant", "Fireball"]
#: A hand of two troops and two such spells.
MIXED_DECK = ["Knight", "Log", "Heal", "Archer", "Giant", "Minions", "Musketeer", "Fireball"]
DOWN = [[2400, 0, 1400], [2400, 1400, 1400]]  # one Blue princess down


def _board(parser_cls=TileActionParser, tower_hp=None, deck=DECK, overrides=None):
    """A RustEngine past the opening lockout, nothing placed. ``DECK`` is shuffled, the others
    are dealt in order."""
    engine = RustEngine(calibration_overrides=overrides)
    shuffle = ShuffleMode.INDEPENDENT if deck is DECK else ShuffleMode.NONE
    mutator = DefaultStateMutator(decks=[deck, deck], shuffle=shuffle)
    setup = mutator.build(np.random.default_rng(0), engine.cards())
    setup = msgspec.structs.replace(setup, elixir_milli=[100000, 100000], tower_hp=tower_hp)
    engine.reset(3, setup)
    engine.step([], engine.rules().deploy_lockout_ticks + 20)
    parser = parser_cls()
    parser.bind(engine)
    return engine, parser


def _with_arms(parser, troop=None, spell=None):
    rules = parser.oracle.rules
    if troop is not None:
        rules = msgspec.structs.replace(rules, illegal_troop_tap=troop)
    if spell is not None:
        rules = msgspec.structs.replace(rules, illegal_spell_tap=spell)
    parser.oracle.rules = rules
    return parser


def _hand(engine, team=BLUE):
    return [engine.cards()[c].name for c in engine.state().players[team].hand]


@needs_engine
@pytest.mark.parametrize("team", [BLUE, RED])
def test_under_the_clamp_an_out_of_territory_troop_acceptance_is_intended(team):
    engine, parser = _board()
    state = engine.state()
    clamping = _Clamping(engine)
    # Vacuity: under "refuse" the simulated clamp's acceptances ARE disagreements, many of them.
    found = mask_disagreements(clamping, _with_arms(parser, "refuse", "refuse"), state, team)
    assert len(found) > 50, len(found)
    assert all(m == 0 and s == DeployStatus.OK for _, m, s in found)
    # Under the troop clamp's arm every one of them is intended; the spell clamp's is no help.
    assert mask_disagreements(clamping, _with_arms(parser, CLAMP), state, team) == []
    assert mask_disagreements(clamping, _with_arms(parser, "refuse", SPELL_CLAMP), state, team)
    # And the real engine (which refuses) still agrees with the mask under either arm.
    assert mask_disagreements(engine, _with_arms(parser, CLAMP, SPELL_CLAMP), state, team) == []


@needs_engine
def test_the_exemption_covers_no_troop_on_water():
    """A troop accepted on WATER is still a disagreement under both clamps' arms: the spell
    clamp's water is for spells."""
    engine, parser = _board()
    state = engine.state()
    clamping = _Clamping(engine, water=True)
    found = mask_disagreements(clamping, _with_arms(parser, CLAMP, SPELL_CLAMP), state, BLUE)
    assert found, "accepting troops on water went unreported"
    assert all(s == DeployStatus.OK and m == 0 for _, m, s in found)


@needs_engine
def test_the_troop_clamps_exemption_covers_troops_only():
    """A Log (rolling) or a Heal (a spell with a troop's placement) refused for territory and
    accepted anyway is still a disagreement under the troop clamp's arm alone; under the spell
    clamp's too, it is intended."""
    engine, parser = _board(deck=MIXED_DECK)
    state = engine.state()
    hand = _hand(engine)
    assert {"Log", "Heal", "Knight"} <= set(hand), hand
    clamping = _Clamping(engine, every_card=True)
    found = mask_disagreements(clamping, _with_arms(parser, CLAMP, "refuse"), state, BLUE)
    names = {hand[parser.parse(a, state, BLUE).hand_slot] for a, _, _ in found}
    assert names == {"Log", "Heal"}, names
    assert mask_disagreements(clamping, _with_arms(parser, CLAMP, SPELL_CLAMP), state, BLUE) == []


def _laws(oracle, card):
    """The troop laws for a troop-kind card, False for a spell judged in troop territory, None
    for a card neither clamp moves."""
    if oracle._troop_laws(card):
        return True
    if card_is_spell(card) and card.placement in (Placement.TROOP, Placement.ROLLING):
        return False
    return None


def _verdicts_agree(engine, parser, team):
    """Every tap of ``team`` a clamp could move: ``zone_verdict`` against the engine's own
    status, which is a zone refusal exactly where the verdict names it, and otherwise is not.
    Returns (disagreements, taps called out of territory, of those on an enemy tower's body,
    slots seen)."""
    state = engine.state()
    oracle = parser.oracle
    player = state.players[team]
    wrong, hits, on_towers, slots = [], 0, 0, set()
    for action in range(1, int(parser.space.n)):
        cmd = parser.parse(action, state, team)
        if cmd is None or cmd.hand_slot >= len(player.hand):
            continue
        card = oracle.cards[player.hand[cmd.hand_slot]]
        laws = _laws(oracle, card)
        if laws is None:
            continue
        slots.add(cmd.hand_slot)
        point = np.asarray([cmd.x]), np.asarray([cmd.y])
        said = int(oracle.zone_verdict(state, team, *point, troop_laws=laws)[0])
        status = int(engine.check_deploy(cmd))
        if said != (status if status in ZONE else DeployStatus.OK):
            wrong.append((action, card.name, cmd.x, cmd.y, status, said))
        out = said == DeployStatus.OUT_OF_TERRITORY
        hits += out
        on_towers += out and any(
            (e.x - cmd.x) ** 2 + (e.y - cmd.y) ** 2 <= e.radius**2
            for e in state.entities
            if e.team != team and e.kind != EntityKind.TROOP
        )
    return wrong, hits, on_towers, slots


@needs_engine
@pytest.mark.parametrize("deck", [DECK, SPELL_DECK], ids=["troops", "spells"])
@pytest.mark.parametrize("parser_cls", [TileActionParser, HalfTileActionParser])
@pytest.mark.parametrize("team", [BLUE, RED])
def test_the_zone_verdict_is_the_engines_own(deck, parser_cls, team):
    """``PlacementOracle.zone_verdict`` names exactly the zone refusal the engine gives, for
    every tap of both seats of a troop or of a spell judged in troop territory: with one Blue
    princess down (the pocket in front of it is Red's ground) and a tap on an enemy tower's
    body, which the engine refuses for territory before any footprint is looked at."""
    engine, parser = _board(parser_cls, tower_hp=DOWN, deck=deck)
    wrong, hits, on_towers, slots = _verdicts_agree(engine, parser, team)
    assert wrong == [], (len(wrong), wrong[:5])
    assert slots, "no card a clamp moves in the hand: nothing was checked"
    assert hits > 100 * len(slots), hits
    assert on_towers > 0, "no tap on an enemy tower's body was checked"


@needs_engine
def test_water_is_judged_before_no_deploy():
    """Off the action lattice, a point can touch a water cell and a no-deploy cell at once (on
    the river's edge row). The zone check reports WATER there, which the spell clamp moves and
    NO_DEPLOY would not; the verdict says the same, for a troop and for the Log."""
    engine, parser = _board(deck=MIXED_DECK)
    state = engine.state()
    oracle, a = parser.oracle, engine.arena()
    step = a.half_size // 2
    px, py = np.meshgrid(np.arange(1, a.hx * 2) * step, np.arange(1, a.hy * 2) * step)
    on_water = ~oracle._touches_only(~oracle.water, px, py)
    both = on_water & ~oracle._touches_only(~oracle.nodeploy, px, py)
    points = list(zip(px[both].tolist(), py[both].tolist(), strict=True))
    assert points, "no point touches both: nothing was checked"
    hand = _hand(engine)
    for name, laws in (("Knight", True), ("Log", False)):
        slot = hand.index(name)
        for x, y in points:
            status = engine.check_deploy(DeployCommand(BLUE, slot, x, y))
            verdict = oracle.zone_verdict(state, BLUE, np.asarray([x]), np.asarray([y]), laws)
            assert status == int(verdict[0]) == DeployStatus.WATER, (name, x, y, status)


@pytest.mark.parametrize("model", TERRITORY_MODELS)
@pytest.mark.parametrize("team", [BLUE, RED])
def test_the_zone_verdict_under_both_territory_models(model, team):
    """The same on MockEngine (Arena::deploy_zone's order), troops and the Log, under the
    closed river band as well as the open bridge the ledger ships, with one Blue princess
    down. Under the closed band the river's half-rows are out of territory even on the fallen
    lane's bridge."""
    engine = MockEngine()
    engine._rules = msgspec.structs.replace(engine._rules, territory_model=model)
    ids = {c.name: c.card_id for c in engine.cards()}
    names = ["Knight", "Archer", "Giant", "Log", "Fireball", "Zap", "Cannon", "Musketeer"]
    deck = [ids[n] for n in names]
    engine.reset(3, MatchSetup(
        decks=[deck, deck], shuffle=ShuffleMode.NONE, elixir_milli=[100000, 100000],
        start_tick=engine.rules().deploy_lockout_ticks, tower_hp=DOWN))
    parser = TileActionParser()
    parser.bind(engine)
    wrong, hits, _, slots = _verdicts_agree(engine, parser, team)
    assert wrong == [], (len(wrong), wrong[:5])
    assert len(slots) == 4, "the Log or a troop went unchecked"
    assert hits > 100 * len(slots), (slots, hits)
    if model == CLOSED_RIVER_BAND:
        a = engine.arena()
        lo, hi = a.water_half_rows
        river = [(x, hy * a.half_size + a.half_size // 2)
                 for x in a.bridge_centers_x() for hy in range(lo, hi + 1)]
        px, py = np.asarray([p[0] for p in river]), np.asarray([p[1] for p in river])
        verdict = parser.oracle.zone_verdict(engine.state(), team, px, py)
        assert (verdict == DeployStatus.OUT_OF_TERRITORY).all()


def test_an_override_of_either_arm_is_followed():
    """An engine built with either key overridden states that arm in its rules, so the mask
    reads the arm the core runs (``OVERRIDE_FOLLOWED``)."""
    assert {TROOP_KEY, SPELL_KEY} <= OVERRIDE_FOLLOWED


def _needs(key):
    section, name = key.split(".")
    ledger = Calibration.load().raw.get(section, {})
    if name not in ledger:
        pytest.skip(f"SKIPPED, NOT PASSED: this engine's ledger has no {key}")
    return ledger[name]["value"]


@needs_engine
def test_on_rustengine_the_clamps_leave_every_cards_placement_alone():
    """The tunnel probe asks whether the enemy half is a card's territory, and under the troop
    clamp the core accepts every troop tapped there: the probe battle refuses such taps, so a
    Knight stays a TROOP and only the cards that travel under ground are TUNNEL."""
    _needs(TROOP_KEY)
    _needs(SPELL_KEY)
    shipped = {c.name: c.placement for c in RustEngine(
        calibration_overrides={TROOP_KEY: "refuse", SPELL_KEY: "refuse"}).cards()}
    clamped = {c.name: c.placement for c in RustEngine(
        calibration_overrides={TROOP_KEY: CLAMP, SPELL_KEY: SPELL_CLAMP}).cards()}
    assert shipped == clamped
    assert clamped["Knight"] == Placement.TROOP
    assert clamped["Miner"] == Placement.TUNNEL
    assert clamped["Log"] == Placement.ROLLING


@needs_engine
@pytest.mark.parametrize(
    ("deck", "overrides", "refused"),
    [
        (DECK, {TROOP_KEY: CLAMP}, {"troop": "refuse"}),
        (SPELL_DECK, {SPELL_KEY: SPELL_CLAMP}, {"spell": "refuse"}),
        (MIXED_DECK, {TROOP_KEY: CLAMP, SPELL_KEY: SPELL_CLAMP},
         {"troop": "refuse", "spell": "refuse"}),
    ],
    ids=["troop-clamp", "spell-clamp", "both"],
)
@pytest.mark.parametrize("parser_cls", [TileActionParser, HalfTileActionParser])
@pytest.mark.parametrize("down", [False, True], ids=["princesses-up", "blue-princess-down"])
@pytest.mark.parametrize("team", [BLUE, RED])
def test_on_rustengine_the_real_clamps_agree_with_the_mask(
    deck, overrides, refused, parser_cls, down, team
):
    """The core's own clamps: every acceptance the mask refuses is one a clamp moved, and
    nothing else disagrees. Graded as if the arms were refuse, the same battle disagrees
    wherever a clamp accepts, for every card in the hand a clamp moves, which is what makes
    the first assertion say something."""
    for key in overrides:
        _needs(key)
    engine, parser = _board(parser_cls, DOWN if down else None, deck, overrides)
    state = engine.state()
    rules = parser.oracle.rules
    assert (rules.illegal_troop_tap, rules.illegal_spell_tap) == (
        overrides.get(TROOP_KEY, "refuse"), overrides.get(SPELL_KEY, "refuse"))
    assert mask_disagreements(engine, parser, state, team) == []
    found = mask_disagreements(engine, _with_arms(parser, **refused), state, team)
    assert {(m, s) for _, m, s in found} == {(0, DeployStatus.OK)}
    hand = _hand(engine, team)
    moved = {hand[parser.parse(a, state, team).hand_slot] for a, _, _ in found}
    cards = {c.name: c for c in parser.oracle.cards}
    clamped_laws = {True} if TROOP_KEY in overrides else set()
    clamped_laws |= {False} if SPELL_KEY in overrides else set()
    want = {n for n in hand if _laws(parser.oracle, cards[n]) in clamped_laws}
    assert moved == want, (moved, want)
    assert len(found) > 20 * len(want), len(found)


@needs_engine
@pytest.mark.parametrize(
    ("key", "arms", "field"),
    [(TROOP_KEY, ILLEGAL_TROOP_TAP, "illegal_troop_tap"),
     (SPELL_KEY, ILLEGAL_SPELL_TAP, "illegal_spell_tap")],
)
def test_on_rustengine_an_override_of_the_arm_reaches_the_rules(key, arms, field):
    shipped = _needs(key)
    other = next(a for a in arms if a != shipped)
    engine = RustEngine(calibration_overrides={key: other})
    assert getattr(engine.rules(), field) == other
    assert getattr(RustEngine().rules(), field) == shipped
