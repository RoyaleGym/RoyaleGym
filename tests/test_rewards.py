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

import pytest

from royalegym.mock_engine import MockEngine
from royalegym.protocol import (
    BLUE,
    RED,
    BattleState,
    DeployCommand,
    EntityKind,
    EntityState,
    MatchSetup,
    Placement,
    ShuffleMode,
    SpawnSpec,
    to_engine,
)
from royalegym.reward import ElixirTradeReward
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available

needs_rust = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))

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
        1, MatchSetup(decks=[deck, deck], shuffle=ShuffleMode.NONE, elixir_milli=[10**7] * 2)
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
    spells = [c for c in catalogue.cards() if c.placement in SPELL_PLACEMENTS]
    troops = [c.name for c in catalogue.cards() if c.placement == Placement.TROOP]
    assert {Placement(c.placement) for c in spells} == set(SPELL_PLACEMENTS), (
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
            1, MatchSetup(decks=[deck, deck], shuffle=ShuffleMode.NONE, elixir_milli=elixir)
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


def tap_everything(engine, tbl: Table):
    """Tap every card the engine will accept, each seat, and yield (card, units it put down).

    A TAP and not a seeded spawn, because they do not put the same thing down: a seeded
    Goblin Gang is one goblin, a tapped one is six. The term scores what a tap left, so
    that is what has to be measured.
    """
    ids = [c.card_id for c in engine.cards()]
    for card in engine.cards():
        deck = [card.card_id, *[i for i in ids if i != card.card_id][:7]]
        for seat in (BLUE, RED):
            engine.reset(
                1,
                MatchSetup(
                    decks=[deck, deck], shuffle=ShuffleMode.NONE, elixir_milli=[10**7] * 2
                ),
            )
            prev = engine.state()
            before = {e.uid for e in prev.entities}
            x, y = tbl.at(seat, 9, 6)
            slot = slot_of(prev, seat, card.card_id)
            if engine.step([DeployCommand(team=seat, hand_slot=slot, x=x, y=y)], 2)[0].status != 0:
                continue
            yield card, seat, [
                e for e in engine.state().entities if e.uid not in before and e.kind not in TOWERS
            ]


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
    """
    engine = make_engine(kind)
    tbl = Table(engine)
    t = term(engine)
    taps, off = 0, []
    for card, seat, put_down in tap_everything(engine, tbl):
        taps += 1
        at_the_tap = t.cast.get(card.card_id, Fraction(0))
        on_the_board = sum((t.unit_value(e) for e in put_down), Fraction(0))
        if at_the_tap + on_the_board != Fraction(card.elixir):
            off.append(
                (card.name, seat, len(put_down), str(at_the_tap), str(on_the_board), card.elixir)
            )
    assert off == [], f"plays priced at something other than the card's elixir: {off}"
    assert taps >= 20, f"only {taps} taps landed; the check would be vacuous"


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
    for card, seat, put_down in tap_everything(engine, tbl):
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
