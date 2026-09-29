"""ElixirTradeReward graded against the engine's own states: what spending elixir costs.

WHAT IS CHECKED, AND WHY IT IS NOT OBVIOUS
    The term pays a seat for elixir the enemy spent and lost, and charges it for its
    own. Two kinds of spending are invisible on the board, and the term got both wrong
    until 2026-09-22:

    A SPELL IS NOT AN ENTITY. Nothing appears on the board for a Fireball, so a term
    that diffs the entity lists collected its kills and charged nothing for the card.
    The trade was free, and a policy graded that way learns that spells are free. Here
    a Fireball that hits nothing at all still costs its four, and a Fireball spent on
    three elixir of units is a losing trade rather than a winning one.

    A CARD CAN PUT UNITS ON THE BOARD THAT ARE NOT ITS OWN. The catalogue has one row
    per card: its elixir, how many units it summons, and what one of them looks like.
    A Tombstone's skeletons are none of those, and the engine reports them under the
    Witch -- a five-elixir card their owner need not even hold. A term that looks the
    reported card up and charges its price bills a skeleton as a Witch. Here a unit is
    paid for only when it is the unit its own card's row describes.

    That rule is measured rather than assumed, and what is measured is the TOTAL, since
    the per-unit rule is not the whole story: a Goblin Gang puts three spear goblins down
    beside its three goblins and the row describes only the goblins.
    ``test_one_tap_of_any_card_is_priced_at_exactly_that_cards_elixir`` taps every card
    either engine will place, on both seats, and checks the play comes to the card's
    price counting the tap charge and the units together -- so a card that was charged
    twice, or not at all, shows up there.
    ``test_a_card_can_put_down_a_unit_its_own_row_does_not_describe`` keeps that total
    from being a restatement of the per-unit rule, and
    ``test_no_unit_a_card_produced_looks_like_the_card_itself`` runs the producers.

HOW IT IS GRADED
    Straight off ``engine.state()``. Each check names what left the board by comparing
    the two snapshots the term was handed, says which of those the catalogue prices and
    which it does not, and only then asserts the number the term returned against a
    total written out in elixir. The two seats are always given different spells or
    different losses on the same step, so the seat holding the bill is identifiable and
    a term that swapped the seats returns the other sign.

    ``SCALE`` is not the default, so a term that ignored its own scale cannot pass.
"""

from __future__ import annotations

from fractions import Fraction

import msgspec
import pytest

from _lockout import lockout_ticks
from royalegym.mock_engine import MockEngine
from royalegym.protocol import (
    BLUE,
    RED,
    BattleState,
    DeployCommand,
    DeployStatus,
    EntityKind,
    EntityState,
    MatchSetup,
    Placement,
    ShuffleMode,
    SpawnSpec,
    card_is_spell,
    slot_cost,
    to_engine,
)
from royalegym.reward import ElixirTradeReward
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available

needs_rust = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))

#: Ticks a match refuses every deploy for. These battles start past it: a reward that
#: scores a PLAY needs the play to land, and at tick 0 every command is TOO_EARLY, so
#: the terms below would be graded over an empty set of deploys. Read from an engine
#: because 0 is a real calibration arm.
LOCKOUT = lockout_ticks()

TOWERS = (EntityKind.KING_TOWER, EntityKind.PRINCESS_TOWER)
SCALE = 4.0
# Both engines hold these. Fireball and Arrows cost a different number of elixir, so a
# step in which each seat casts one of them is not symmetric between the seats.
SPELL_DECK = (
    "Fireball", "Arrows", "Knight", "Cannon", "Archer", "Giant", "MiniPekka", "Musketeer",
)
FIREBALL, ARROWS, CANNON = 4, 3, 3  # elixir; checked against the catalogue in the tests
# A deck with two cards that keep producing units, for the compiled engine.
PRODUCER_DECK = (
    "Fireball", "Arrows", "Witch", "Tombstone", "Knight", "Cannon", "Giant", "Musketeer",
)
ENGINES = ["mock", pytest.param("rust", marks=needs_rust)]

#: Cards whose price the catalogue test does not grade, each with a strict xfail of its own.
#: The Tri Wizards: from RoyaleSim round 9 the card puts its Electro Wizard and Ice Wizard down
#: under THEIR OWN card ids (42, 23), so the term prices a play of it at 7 + 4 + 3. Every other
#: multi-unit card stamps the played card's id on what it puts down (a Goblin Gang's Spear
#: Goblins are 41, not 19). It is an event-only card, and event-only cards are deferred to
#: the end of the engine's queue, so the stamp waits with them.
PRICED_ELSEWHERE = frozenset({"TriWizards"})
# The three ways the catalogue describes a card that puts nothing of its own on the board.
# Spelled out from the engine's own placement classes rather than taken from the term, so a
# term that dropped one of them cannot quietly narrow this file's idea of what a spell is.
SPELL_PLACEMENTS = (Placement.SPELL, Placement.ROLLING, Placement.SPELL_NOT_ON_WATER)


def make_engine(kind: str, names: tuple[str, ...] | None = None):
    """``names`` None means the engine's whole catalogue."""
    if kind == "rust":
        return RustEngine(card_names=list(names)) if names else RustEngine()
    return MockEngine(card_names=list(names)) if names else MockEngine()


def term(engine) -> ElixirTradeReward:
    t = ElixirTradeReward(scale=SCALE)
    t.bind(engine)
    return t


def gone(prev: BattleState, cur: BattleState) -> list[EntityState]:
    """The non-tower entities that left the board over the step, from the two snapshots."""
    alive = {e.uid for e in cur.entities}
    return [e for e in prev.entities if e.uid not in alive and e.kind not in TOWERS]


def born(prev: BattleState, cur: BattleState) -> list[EntityState]:
    was = {e.uid for e in prev.entities}
    return [e for e in cur.entities if e.uid not in was and e.kind not in TOWERS]


def is_own_unit(card, e: EntityState) -> bool:
    """Is this entity the unit ``card``'s catalogue row describes?"""
    return (e.max_hp, e.radius, e.flying) == (card.hitpoints, card.radius, card.flying)


def slot_of(state: BattleState, team: int, card_id: int) -> int:
    hand = state.players[team].hand
    assert card_id in hand, f"team {team} does not hold card {card_id}: {hand}"
    return hand.index(card_id)


class Table:
    """The catalogue by name and by id, plus own-frame to engine-frame placement."""

    def __init__(self, engine) -> None:
        self.engine = engine
        self.by_name = {c.name: c for c in engine.cards()}
        self.by_id = {c.card_id: c for c in engine.cards()}
        self.arena = engine.arena()

    def deck(self, names) -> list[int]:
        return [self.by_name[n].card_id for n in names]

    def at(self, team: int, tx: float, ty: float) -> tuple[int, int]:
        """A tile in ``team``'s own frame, as the engine frame."""
        sub = self.arena.subtile
        return to_engine(self.arena, team, int(tx * sub), int(ty * sub))


def cast(tbl: Table, state: BattleState, team: int, name: str, aim) -> DeployCommand:
    slot = slot_of(state, team, tbl.by_name[name].card_id)
    return DeployCommand(team=team, hand_slot=slot, x=aim[0], y=aim[1])


# --------------------------------------------------------------- a spell costs elixir


@pytest.mark.parametrize("kind", ENGINES)
@pytest.mark.parametrize("burner", [BLUE, RED])
def test_a_spell_that_kills_nothing_still_costs_the_seat_that_cast_it(kind, burner):
    """Nothing on the board, a spell cast into it by each seat: each is billed its own."""
    engine = make_engine(kind, SPELL_DECK)
    tbl = Table(engine)
    assert (tbl.by_name["Fireball"].elixir, tbl.by_name["Arrows"].elixir) == (FIREBALL, ARROWS)
    archer = 1 - burner  # the other seat casts the cheaper spell
    deck = tbl.deck(SPELL_DECK)
    engine.reset(
        1, MatchSetup(
            decks=[deck, deck], shuffle=ShuffleMode.NONE,
            elixir_milli=[10**7] * 2, start_tick=LOCKOUT,
        )
    )

    prev = engine.state()
    assert [e for e in prev.entities if e.kind not in TOWERS] == [], "the board is meant to be bare"
    results = engine.step(
        [
            cast(tbl, prev, burner, "Fireball", tbl.at(burner, 9, 8)),
            cast(tbl, prev, archer, "Arrows", tbl.at(archer, 9, 8)),
        ],
        30,
    )
    cur = engine.state()

    assert [r.status for r in results] == [0, 0], "both casts were meant to be accepted"
    assert [e for e in cur.entities if e.kind not in TOWERS] == [], "nothing was meant to appear"
    assert gone(prev, cur) == [], "nothing was meant to die"

    t = term(engine)
    # The burner spent four and the other seat three, and neither got anything for it.
    assert t.get_reward(burner, prev, cur, results) == pytest.approx((ARROWS - FIREBALL) / SCALE)
    assert t.get_reward(archer, prev, cur, results) == pytest.approx((FIREBALL - ARROWS) / SCALE)


@pytest.mark.parametrize("kind", ENGINES)
@pytest.mark.parametrize("burner", [BLUE, RED])
def test_a_spell_that_kills_is_billed_for_the_cast_and_paid_for_the_kill(kind, burner):
    """Each seat spends a spell on the other's Cannon. The dearer spell is the worse trade."""
    engine = make_engine(kind, SPELL_DECK)
    tbl = Table(engine)
    assert tbl.by_name["Cannon"].elixir == CANNON
    assert tbl.by_name["Cannon"].count == 1
    archer = 1 - burner
    deck = tbl.deck(SPELL_DECK)
    cannon = {team: tbl.at(team, 5, 9) for team in (BLUE, RED)}
    engine.reset(
        1,
        MatchSetup(
            decks=[deck, deck],
            start_tick=LOCKOUT,
            shuffle=ShuffleMode.NONE,
            elixir_milli=[10**7] * 2,
            # One hit point each, so whichever spell reaches one kills it on both engines.
            spawns=[
                SpawnSpec(team=team, card_id=tbl.by_name["Cannon"].card_id, x=x, y=y, hp=1)
                for team, (x, y) in cannon.items()
            ],
        ),
    )

    prev = engine.state()
    results = engine.step(
        [
            cast(tbl, prev, burner, "Fireball", cannon[archer]),
            cast(tbl, prev, archer, "Arrows", cannon[burner]),
        ],
        12,
    )
    cur = engine.state()

    assert [r.status for r in results] == [0, 0]
    dead = gone(prev, cur)
    assert sorted(e.team for e in dead) == [BLUE, RED], f"one Cannon each was meant to die: {dead}"
    for e in dead:
        card = tbl.by_id[e.card_id]
        assert card.name == "Cannon"
        assert is_own_unit(card, e)

    t = term(engine)
    # Each seat lost three elixir of Cannon; the burner paid four for the spell that did
    # it and the other seat three, so the burner is one elixir down on the exchange.
    assert t.get_reward(burner, prev, cur, results) == pytest.approx(
        ((CANNON + ARROWS) - (CANNON + FIREBALL)) / SCALE
    )
    assert t.get_reward(archer, prev, cur, results) == pytest.approx(
        ((CANNON + FIREBALL) - (CANNON + ARROWS)) / SCALE
    )


@pytest.mark.parametrize("kind", ENGINES)
@pytest.mark.parametrize("caster", [BLUE, RED])
def test_every_card_the_catalogue_calls_a_spell_is_charged_once_at_the_tap(kind, caster):
    """Every spell the engine holds, not just the two the checks above cast.

    A card is charged at the tap when the catalogue puts it in one of three placement
    classes, and the two spells above are both the plainest of the three. A Log rolls and
    a Goblin Barrel lands somewhere a plain spell may not, and if either fell out of that
    set it would be free. So each is cast here in turn, on a bare board, and billed.

    The other seat taps the SAME card with an empty bar, so the engine refuses it. That
    refusal is the discriminator: a term that charged for a tap the engine turned down
    would cancel the two against each other and return zero for both seats.
    """
    catalogue = make_engine(kind)
    # By KIND: a spell may carry a troop's placement code (Heal, from the next RoyaleSim
    # build), so the spells are at least one of each spell placement class, not exactly.
    # The Mirror is a spell by kind, but it copies its side's last play and costs that play
    # plus its own one: on this bare board there is nothing to copy. It has its own test.
    spells = [
        c for c in catalogue.cards() if card_is_spell(c) and c.placement != Placement.MIRROR
    ]
    troops = [
        c.name
        for c in catalogue.cards()
        if c.placement == Placement.TROOP and not card_is_spell(c)
    ]
    assert {Placement(c.placement) for c in spells} >= set(SPELL_PLACEMENTS), (
        f"{kind} does not hold one of each placement class: {[c.name for c in spells]}"
    )
    broke = 1 - caster
    for spell in spells:
        names = (spell.name, *[n for n in troops if n != spell.name][:7])
        engine = make_engine(kind, names)
        tbl = Table(engine)
        deck = tbl.deck(names)
        elixir = [0, 0]
        elixir[caster] = 10**7
        engine.reset(
            1, MatchSetup(
                decks=[deck, deck], shuffle=ShuffleMode.NONE,
                elixir_milli=elixir, start_tick=LOCKOUT,
            )
        )

        prev = engine.state()
        assert [e for e in prev.entities if e.kind not in TOWERS] == [], "the board is bare"
        results = engine.step(
            [
                cast(tbl, prev, caster, spell.name, tbl.at(caster, 9, 8)),
                cast(tbl, prev, broke, spell.name, tbl.at(broke, 9, 8)),
            ],
            1,
        )
        cur = engine.state()

        accepted = [r for r in results if r.status == 0]
        assert [r.team for r in accepted] == [caster], (
            f"{spell.name}: the funded seat's tap was meant to be the only one taken: "
            f"{[(r.team, r.status) for r in results]}"
        )
        assert gone(prev, cur) == [], f"{spell.name} was meant to kill nothing"

        t = term(engine)
        # Whatever the spell left behind was born this step, so it is in neither seat's
        # losses; the whole step is the one cast.
        assert t.get_reward(caster, prev, cur, results) == pytest.approx(-spell.elixir / SCALE), (
            f"{spell.name} ({Placement(spell.placement).name}) was not charged to its caster"
        )
        assert t.get_reward(broke, prev, cur, results) == pytest.approx(spell.elixir / SCALE)


# ------------------------------------------------- a card's own unit, and what it makes


@needs_rust
@pytest.mark.parametrize("owner", [BLUE, RED])
def test_a_unit_a_card_produced_is_not_billed_at_that_cards_price(owner):
    """A Tombstone dies, leaves skeletons behind, and the skeletons are swept up.

    The engine reports every one of those skeletons under the Witch, so a term that
    trusts the reported card pays five elixir a head for what a three-elixir card left
    behind. Two steps: the Tombstone itself, which the catalogue does price, and then
    only skeletons, which it does not.
    """
    engine = make_engine("rust", PRODUCER_DECK)
    tbl = Table(engine)
    tomb = tbl.by_name["Tombstone"]
    caster = 1 - owner
    deck = tbl.deck(PRODUCER_DECK)
    x, y = tbl.at(owner, 9, 6)
    engine.reset(
        1,
        MatchSetup(
            decks=[deck, deck],
            start_tick=LOCKOUT,
            shuffle=ShuffleMode.NONE,
            elixir_milli=[10**7] * 2,
            spawns=[SpawnSpec(team=owner, card_id=tomb.card_id, x=x, y=y, hp=1)],
        ),
    )
    t = term(engine)

    # STEP ONE: the Tombstone. It is the unit its own row describes, so it is priced.
    prev = engine.state()
    results = engine.step([cast(tbl, prev, caster, "Fireball", (x, y))], 12)
    cur = engine.state()
    assert [r.status for r in results] == [0]
    dead = gone(prev, cur)
    assert [(e.team, tbl.by_id[e.card_id].name) for e in dead] == [(owner, "Tombstone")]
    assert is_own_unit(tomb, dead[0])
    left = born(prev, cur)
    assert len(left) >= 4, f"the Tombstone was meant to leave skeletons behind: {left}"
    for e in left:
        under = tbl.by_id[e.card_id]
        assert not is_own_unit(under, e), "a skeleton is not the unit its reported card describes"
        assert under.name != "Tombstone", "the engine reports it under some other card entirely"
    assert t.get_reward(owner, prev, cur, results) == pytest.approx(
        (FIREBALL - tomb.elixir) / SCALE
    )
    assert t.get_reward(caster, prev, cur, results) == pytest.approx(
        (tomb.elixir - FIREBALL) / SCALE
    )

    # STEP TWO: only the skeletons. The catalogue prices none of them, so the whole step
    # is the caster's Arrows and nothing else.
    prev = engine.state()
    crowd = [e for e in prev.entities if e.team == owner and e.kind not in TOWERS]
    assert crowd, "the skeletons were meant to still be standing"
    aim = (sum(e.x for e in crowd) // len(crowd), sum(e.y for e in crowd) // len(crowd))
    results = engine.step([cast(tbl, prev, caster, "Arrows", aim)], 40)
    cur = engine.state()
    assert [r.status for r in results] == [0]
    dead = gone(prev, cur)
    assert dead, "the Arrows were meant to kill the skeletons"
    at_the_reported_price = Fraction(0)
    for e in dead:
        under = tbl.by_id[e.card_id]
        assert e.team == owner
        assert not is_own_unit(under, e)
        at_the_reported_price += Fraction(under.elixir, max(1, under.count))
    # Not a vacuous check: reading those deaths off the reported card is worth elixir.
    assert at_the_reported_price >= 5
    assert t.get_reward(owner, prev, cur, results) == pytest.approx(ARROWS / SCALE)
    assert t.get_reward(caster, prev, cur, results) == pytest.approx(-ARROWS / SCALE)


def cycle_in(
    engine, tbl: Table, setup: MatchSetup, seat: int, card_id: int, singles
) -> BattleState:
    """The state once ``card_id`` is in ``seat``'s hand, for a card the deal keeps OUT of the
    starting hand (economy.OMIT_FROM_STARTING_HAND, the Elixir Collector since RoyaleSim
    1d661b0). The deal swaps it with the first eligible card in the queue, so it is the next
    card drawn: one play of slot 0 brings it in. The deck is refilled with single-unit troops
    so that play puts down one unit and nothing more, and the battle runs on until that unit
    has landed, so nothing it leaves can be counted as the probed card's, and until the
    elixir the play spent is back (the seeded 10**7 is capped, so the tap could not pay)."""
    others = [i for i in singles if i != card_id][:7]
    assert len(others) == 7, f"only {len(others)} single-unit troops to cycle with"
    deck = [card_id, *others]
    engine.reset(1, msgspec.structs.replace(setup, decks=[deck, deck]))
    state = engine.state()
    assert card_id not in state.players[seat].hand, "the deal no longer omits it"
    x, y = tbl.at(seat, 4, 10)
    played = engine.step([DeployCommand(team=seat, hand_slot=0, x=x, y=y)], 1)[0]
    assert played.status == 0, f"the filler play was refused: {played}"
    for _ in range(100):
        engine.step([], 1)
        state = engine.state()
        if card_id in state.players[seat].hand:
            engine.step([], 40)  # the filler's unit lands before the probe's snapshot
            card = engine.cards()[card_id]
            for _ in range(100):
                state = engine.state()
                p = state.players[seat]
                # The slot's price, not the card's elixir: a Mirror costs its copy plus one.
                cost = slot_cost(p, p.hand.index(card_id), card)
                if cost >= 0 and p.elixir_milli >= 1000 * cost:
                    return state
                engine.step([], 10)
            raise AssertionError(f"seat {seat} never afforded card {card_id} again")
    raise AssertionError(f"card {card_id} never reached seat {seat}'s hand")


#: How long a tap's own spell and area rows may last before ``tap_everything`` reads what it
#: put down: the longest spell in the catalogue (a Graveyard's 180-odd ticks) with room.
SETTLE_TICKS = 600


def tap_everything(engine, tbl: Table, only: frozenset[str] | None = None):
    """Tap every card the engine will accept (or those named in ``only``), each seat, and
    yield (card, seat, the state the tap was made from, its result, the units it put down).

    A TAP and not a seeded spawn, because they do not put the same thing down: a seeded
    Goblin Gang is one goblin, a tapped one is six. The term scores what a tap left, so
    that is what has to be measured.

    AND WHAT IT LEFT ONCE IT HAS FINISHED LEAVING IT: the units are read when the card's
    own spell and area rows are gone, not two ticks after the tap. A card that puts its
    units down through a deploy spawn area (the Tri Wizards: the TriWizard on C+5, the
    other two on C+6, RoyaleSim round 9) had put nothing down at two ticks and read as a
    play worth 0. So had a Goblin Barrel (its goblins land around +8) and a Graveyard (its
    skeletons over some 180 ticks), which is why "anything a spell leaves scores zero" was
    never graded until this read. A unit is counted from the tick it is first seen.
    """
    ids = [c.card_id for c in engine.cards()]
    # Single-unit troops: a filler that puts down one unit and nothing after it.
    singles = [
        c.card_id for c in engine.cards() if c.placement == Placement.TROOP and c.count == 1
    ]
    refused: list[tuple[str, int, str]] = []
    for card in engine.cards():
        if only is not None and card.name not in only:
            continue
        deck = [card.card_id, *[i for i in ids if i != card.card_id][:7]]
        for seat in (BLUE, RED):
            setup = MatchSetup(
                decks=[deck, deck], shuffle=ShuffleMode.NONE,
                elixir_milli=[10**7] * 2, start_tick=LOCKOUT,
            )
            engine.reset(1, setup)
            prev = engine.state()
            if card.card_id not in prev.players[seat].hand:
                prev = cycle_in(engine, tbl, setup, seat, card.card_id, singles)
            before = {e.uid for e in prev.entities}
            x, y = tbl.at(seat, 9, 6)
            slot = slot_of(prev, seat, card.card_id)
            result = engine.step([DeployCommand(team=seat, hand_slot=slot, x=x, y=y)], 2)[0]
            if result.status != 0:
                # Once a silent skip: the Elixir Collector, never dealt and then unaffordable
                # (2026-09-27), dropped out of both tests without a word.
                refused.append((card.name, seat, DeployStatus(result.status).name))
                continue
            put_down: dict[int, EntityState] = {}
            for waited in range(SETTLE_TICKS + 1):
                state = engine.state()
                for e in state.entities:
                    if e.uid not in before and e.kind not in TOWERS:
                        put_down.setdefault(e.uid, e)
                if not any(s.team == seat and s.card_id == card.card_id for s in state.spells):
                    break
                assert waited < SETTLE_TICKS, (
                    f"{card.name}'s spell or area rows outlived {SETTLE_TICKS} ticks, so what "
                    "it puts down was never read whole"
                )
                engine.step([], 1)
            yield card, seat, prev, result, list(put_down.values())
    assert not refused, f"taps the engine refused, so these cards went unmeasured: {refused}"


@pytest.mark.parametrize("kind", ENGINES)
def test_one_tap_of_any_card_is_priced_at_exactly_that_cards_elixir(kind):
    """The whole catalogue, both seats: one play of a card is worth what the card cost.

    This is the property the term rests on, and it is not the per-unit rule it is built
    from. A card is paid for ONCE, either at the tap or through the units it left, never
    both and never neither:

      * most cards put down only the unit their row describes, and are priced one by one;
      * a few put a second KIND down as well, which the row does not describe and the
        term scores zero -- the card's summon count covers exactly the kind it does
        describe, so the play still totals the card's price;
      * a spell is charged at the tap, and anything it leaves on the board scores zero,
        so a Goblin Barrel costs three rather than three plus its goblins.

    If any of that slipped, the term would quietly under- or over-pay every play of that
    card for the rest of training, so it is measured rather than believed.

    The cards in ``PRICED_ELSEWHERE`` are graded by their own strict xfails, not here.
    """
    engine = make_engine(kind)
    tbl = Table(engine)
    t = term(engine)
    taps, off = 0, []
    for card, seat, prev, result, put_down in tap_everything(engine, tbl):
        if card.name in PRICED_ELSEWHERE:
            continue
        taps += 1
        # What the play cost: the slot's stated price. That is the card's own elixir for
        # every card but a Mirror, which costs the card it copies plus its own one.
        price = Fraction(slot_cost(prev.players[seat], result.hand_slot, card))
        if card.placement != Placement.MIRROR and price != card.elixir:
            off.append((card.name, seat, "stated", str(price), "listed", card.elixir))
        listed = t.cast.get(card.card_id)
        at_the_tap = Fraction(0) if listed is None else t._paid(prev, result, listed)
        on_the_board = sum((t.unit_value(e) for e in put_down), Fraction(0))
        if at_the_tap + on_the_board != price:
            off.append(
                (card.name, seat, len(put_down), str(at_the_tap), str(on_the_board), str(price))
            )
    assert off == [], f"plays priced at something other than what they cost: {off}"
    assert taps >= 20, f"only {taps} taps landed; the check would be vacuous"


@needs_rust
@pytest.mark.xfail(
    reason=(
        "RoyaleSim reports the Tri Wizards' Electro and Ice Wizards under their own card ids "
        "(42, 23), so the term prices the play 14; stamping the played card's id is deferred "
        "with the other event-only cards"
    ),
    strict=True,
)
def test_a_play_of_the_tri_wizards_is_priced_at_the_card():
    """A play of the Tri Wizards is worth the card: 7, not its wizards' own cards' 7 + 4 + 3.

    From round 9 the card puts three different wizards down through a deploy spawn area;
    before it, only the TriWizard. The price is the property either way, so it is what is
    asserted, with the units read listed beside it."""
    engine = make_engine("rust")
    if "TriWizards" not in {c.name for c in engine.cards()}:
        pytest.skip("this engine's catalogue has no Tri Wizards")
    t = term(engine)
    priced, read = [], []
    for _card, seat, _prev, _result, put_down in tap_everything(
        engine, Table(engine), only=frozenset({"TriWizards"})
    ):
        priced.append((seat, str(sum((t.unit_value(e) for e in put_down), Fraction(0)))))
        read.append(sorted((e.card_id, e.max_hp) for e in put_down))
    assert priced == [(BLUE, "7"), (RED, "7")], f"{priced}; units read: {read}"


@needs_rust
def test_a_card_can_put_down_a_unit_its_own_row_does_not_describe():
    """Why the check above is a total and not a per-unit rule.

    A Goblin Gang's three spear goblins and the two Rascal girls are not the unit their
    card's row describes, so the term scores them zero and the rest of the card carries
    its whole price. Without this, a per-unit rule would look like the whole truth and
    the total above would look like a restatement of it.
    """
    engine = make_engine("rust")
    tbl = Table(engine)
    mixed = []
    for card, seat, _prev, _result, put_down in tap_everything(engine, tbl):
        if len({(e.max_hp, e.radius, e.flying) for e in put_down}) < 2:
            continue
        own = [e for e in put_down if is_own_unit(card, e)]
        mixed.append((card.name, seat, len(put_down), len(own)))
        # The card's summon count covers exactly the kind its row describes. That is the
        # whole reason the total comes out right while the per-unit rule does not hold.
        assert len(own) == card.count, (
            f"{card.name} put down {len(put_down)} units, {len(own)} of them the unit its "
            f"row describes, but the catalogue counts {card.count}"
        )
        assert len(own) < len(put_down), f"{card.name} is not mixed after all"
    assert mixed, "no card put down a second kind of unit; the total check adds nothing"


MIRROR_DECK = (
    "Mirror", "Knight", "Musketeer", "Valkyrie", "MiniPekka", "Giant", "Prince", "Wizard",
)


@needs_rust
@pytest.mark.parametrize("caster", [BLUE, RED])
def test_a_mirror_is_charged_at_the_tap_what_the_engine_took(caster):
    """A Mirror costs the card it copies plus its own one (match.MIRROR_COST_RULE), and puts
    a copy down one level up, which matches no row and scores nothing. So the play is worth
    its whole price at the tap. Charged its own one elixir, as it was until 2026-09-28, a
    Mirror of a Knight cost the term 1 where the engine took 4."""
    engine = make_engine("rust", MIRROR_DECK)
    tbl = Table(engine)
    mirror, knight = tbl.by_name["Mirror"], tbl.by_name["Knight"]
    deck = tbl.deck(MIRROR_DECK)
    engine.reset(
        1,
        MatchSetup(
            decks=[deck, deck], shuffle=ShuffleMode.NONE,
            elixir_milli=[10**7] * 2, start_tick=LOCKOUT,
        ),
    )
    # The deal keeps the Mirror out of the opening hand; the Knight's play brings it in.
    prev = engine.state()
    assert mirror.card_id not in prev.players[caster].hand
    played = engine.step([cast(tbl, prev, caster, "Knight", tbl.at(caster, 9, 6))], 1)
    assert [r.status for r in played] == [0]
    prev = engine.state()
    p = prev.players[caster]
    slot = slot_of(prev, caster, mirror.card_id)
    assert p.mirror_target == knight.card_id
    price = p.hand_costs[slot]
    assert price == knight.elixir + mirror.elixir, p.hand_costs
    assert p.elixir_milli >= 1000 * price

    results = engine.step([cast(tbl, prev, caster, "Mirror", tbl.at(caster, 5, 6))], 2)
    cur = engine.state()
    assert [r.status for r in results] == [0]
    took = prev.players[caster].elixir_milli - cur.players[caster].elixir_milli
    assert 1000 * price - 100 <= took <= 1000 * price, f"the engine took {took}, not {price}"
    copies = born(prev, cur)
    assert [tbl.by_id[e.card_id].name for e in copies] == ["Knight"], copies
    assert not is_own_unit(knight, copies[0]), "the copy was meant to be one level up"

    t = term(engine)
    assert t.get_reward(caster, prev, cur, results) == pytest.approx(-price / SCALE)
    assert t.get_reward(1 - caster, prev, cur, results) == pytest.approx(price / SCALE)


@needs_rust
def test_no_unit_a_card_produced_looks_like_the_card_itself():
    """A Witch on one side and a Tombstone on the other, left alone to produce.

    The other half of the rule: nothing either card puts on the board can be mistaken
    for the card's own unit, so scoring produced units zero costs the term nothing it
    could otherwise have had right.
    """
    engine = make_engine("rust", PRODUCER_DECK)
    tbl = Table(engine)
    deck = tbl.deck(PRODUCER_DECK)
    witch, tomb = tbl.at(BLUE, 9, 6), tbl.at(RED, 9, 6)
    engine.reset(
        1,
        MatchSetup(
            decks=[deck, deck],
            start_tick=LOCKOUT,
            shuffle=ShuffleMode.NONE,
            spawns=[
                SpawnSpec(
                    team=BLUE, card_id=tbl.by_name["Witch"].card_id, x=witch[0], y=witch[1]
                ),
                SpawnSpec(
                    team=RED, card_id=tbl.by_name["Tombstone"].card_id, x=tomb[0], y=tomb[1]
                ),
            ],
        ),
    )
    producers = {e.uid for e in engine.state().entities}
    seen = {BLUE: 0, RED: 0}
    for _ in range(40):
        prev = engine.state()
        engine.step([], 8)
        for e in born(prev, engine.state()):
            if e.uid in producers:
                continue
            seen[e.team] += 1
            under = tbl.by_id[e.card_id]
            assert not is_own_unit(under, e), f"{under.name} produced a unit that looks like itself"
    assert seen[BLUE] > 0, f"the Witch produced nothing: {seen}"
    assert seen[RED] > 0, f"the Tombstone produced nothing: {seen}"
