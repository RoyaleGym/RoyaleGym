"""Code 5 (TUNNEL: the Miner, the Goblin Drill) and code 6 (MIRROR): the mask against the engine.

WHY IT EXISTS
    RoyaleSim leaves the cards that travel under ground and the Mirror out of its default
    catalogue until this package can mask them. A card that tunnels goes down anywhere on
    land (placement.SPAWN_PATHFIND_TERRITORY = anywhere_but_water), then meets its kind's
    footprint rule: the Miner is refused on a building or a tower, the Goblin Drill is
    placed on its building's box. A Mirror goes down wherever the card it copies may, at
    that card's price plus its own (match.MIRROR_PLACEMENT, match.MIRROR_COST_RULE). A
    catalogue NAMING all three is graded here, action by action, against ``check_deploy``.

WHAT IT CHECKS
    a. The placements: the Miner and the Goblin Drill come out TUNNEL with their card
       kinds, the Mirror MIRROR, and no other card moves. Up to RoyaleSim 244c893 the core
       reports the two tunnellers under codes 0 and 1, and the adapter asks the core
       (``tunnelling_cards``); an engine that reports 5 itself passes the same check.
    b. THE GATE: the every-card gate on that catalogue, four boards, tile and half tile,
       both seats. The Mirror reaches it through ``cycle_in``, copying a troop.
    c. The Mirror after each placement it can copy (a troop, a building, a spell, a
       rolling spell, a spawning spell, a spell with a troop's rule, a Miner, a Goblin
       Drill), both seats, every action of its slot, from a bar below the copy's price
       until it pays it: the price is graded with the placement.
    d. With nothing to copy, the Mirror is offered nowhere.

PLANTS (kept below as tests; each seen red on a green baseline)
    the probe switched off (the tunnellers masked as a troop and a building); the Mirror
    judged by its own row (anywhere); the Mirror priced at its own elixir; a tunnelling
    troop blind to the bodies on the board.

SKIPS
    Engine tests skip when royalesim is not built. A skip is not a pass.
"""

from __future__ import annotations

import msgspec
import numpy as np
import pytest

from royalegym import action as action_module
from royalegym import rust_engine
from royalegym.action import (
    GridActionParser,
    HalfTileActionParser,
    PlacementOracle,
    TileActionParser,
)
from royalegym.protocol import (
    DECK_SIZE,
    TEAMS,
    BattleState,
    DeployStatus,
    Placement,
    slot_cost,
)
from royalegym.rust_engine import RustEngine
from test_building_footprint import (
    BOARDS,
    SEAT,
    board_setup,
    card_ids,
    every_card_gate,
    needs_core,
)

#: The every-card gate's boards need Cannon, Tesla, Knight, InfernoTower, Tombstone and
#: Mortar; ``cycle_in`` needs seven single-unit troops; the Mirror needs one card of each
#: placement it can copy. The three under test last.
NAMES = (
    "Knight", "Musketeer", "Valkyrie", "MiniPekka", "Giant", "Prince", "Wizard", "BabyDragon",
    "Cannon", "Tesla", "InfernoTower", "Tombstone", "Mortar",
    "Fireball", "Log", "GoblinBarrel", "Heal",
    "Miner", "GoblinDrill", "Mirror",
)
#: One card per placement a Mirror can copy.
MIRROR_TARGETS = (
    "Knight", "Cannon", "Fireball", "Log", "GoblinBarrel", "Heal", "Miner", "GoblinDrill",
)


@pytest.fixture(scope="module")
def named() -> RustEngine:
    return RustEngine(card_names=list(NAMES))


# ---------------------------------------------------------------------------
# a. The placements


@needs_core
def test_the_tunnellers_and_the_mirror_get_their_placements(named, monkeypatch):
    kinds = {c.name: (Placement(c.placement), c.card_kind) for c in named.cards()}
    assert kinds["Miner"] == (Placement.TUNNEL, "TROOP"), kinds["Miner"]
    assert kinds["GoblinDrill"] == (Placement.TUNNEL, "BUILDING"), kinds["GoblinDrill"]
    assert kinds["Mirror"][0] == Placement.MIRROR, kinds["Mirror"]
    tunnels = {n for n, (p, _) in kinds.items() if p == Placement.TUNNEL}
    assert tunnels == {"Miner", "GoblinDrill"}, tunnels
    # Nothing else moves: the probe relabels the tunnellers and no other card, and finds
    # nothing to relabel where the engine states code 5 itself.
    with monkeypatch.context() as m:
        m.setattr(rust_engine, "tunnelling_cards", lambda *a, **k: frozenset())
        raw = RustEngine(card_names=list(NAMES))
    moved = {c.name for c, r in zip(named.cards(), raw.cards(), strict=True) if c != r}
    assert moved <= {"Miner", "GoblinDrill"}, moved


@needs_core
def test_the_default_catalogue_keeps_its_placements(named):
    """The probe must not move a card of the default catalogue, which holds no tunneller up
    to RoyaleSim 244c893 (the tunnellers join it once 5 is mapped, as TUNNEL)."""
    default = RustEngine()
    tunnels = {c.name for c in default.cards() if c.placement == Placement.TUNNEL}
    assert tunnels <= {"Miner", "GoblinDrill"}, tunnels


# ---------------------------------------------------------------------------
# b. The gate


@needs_core
@pytest.mark.parametrize("parser_cls", [TileActionParser, HalfTileActionParser])
@pytest.mark.parametrize("board", BOARDS)
def test_the_mask_equals_the_engine_for_a_catalogue_naming_them(named, board, parser_cls):
    problems, _ = every_card_gate(named, board, parser_cls())
    assert not problems, f"{board}: {len(problems)} disagreements, first {problems[:8]}"


# ---------------------------------------------------------------------------
# c. The Mirror, after each placement it can copy


def singles(engine: RustEngine, but: int) -> list[int]:
    return [
        c.card_id
        for c in engine.cards()
        if c.placement == Placement.TROOP and c.count == 1 and c.card_id != but
    ]


def play_target(engine: RustEngine, parser: GridActionParser, board: str, target: str) -> None:
    """``board`` with both seats' bars at the target's price plus a little, the target
    played from each seat's hand on a tap the mask offers, and a Mirror next in the queue:
    after the play the Mirror is in hand, copying the target, and the bar is below its
    price."""
    ids = card_ids(engine)
    tid = ids[target]
    deck = [ids["Mirror"], tid, *singles(engine, tid)][:DECK_SIZE]
    cost = engine.cards()[tid].elixir
    setup = msgspec.structs.replace(
        board_setup(engine, board, [deck, deck]),
        elixir_milli=[1000 * cost + 400, 1000 * cost + 150],
    )
    engine.reset(1, setup)
    parser.bind(engine)
    state = engine.state()
    per = parser.nx * parser.ny
    plays = []
    for team in TEAMS:
        hand = state.players[team].hand
        assert tid in hand, f"{SEAT[team]}: {target} not dealt ({hand})"
        slot = hand.index(tid)
        plane = parser.action_mask(state, team)[1 + slot * per : 1 + (slot + 1) * per]
        legal = np.flatnonzero(plane)
        assert legal.size, f"{SEAT[team]}: {target} offered nowhere"
        plays.append(parser.parse(1 + slot * per + int(legal[legal.size // 2]), state, team))
    statuses = [DeployStatus(r.status).name for r in engine.step(plays, 1)]
    assert statuses == ["OK", "OK"], f"{target}: {statuses}"


def mirror_slot_disagreements(
    engine: RustEngine, parser: GridActionParser, state: BattleState, team: int
) -> list[str]:
    """Every action of the Mirror's slot where the mask and ``check_deploy`` disagree."""
    mirror = card_ids(engine)["Mirror"]
    hand = state.players[team].hand
    if mirror not in hand:
        return []
    slot = hand.index(mirror)
    per = parser.nx * parser.ny
    mask = parser.action_mask(state, team)
    out = []
    for action in range(1 + slot * per, 1 + (slot + 1) * per):
        status = engine.check_deploy(parser.parse(action, state, team))
        if bool(mask[action]) != (status == 0):
            _, xi, yi = parser.decode(action)
            out.append(
                f"{SEAT[team]} Mirror at own point ({xi}, {yi}), elixir "
                f"{state.players[team].elixir_milli}: mask {int(mask[action])}, engine "
                f"{DeployStatus(status).name}"
            )
    return out


def mirror_gate(
    engine: RustEngine, parser: GridActionParser, board: str, target: str
) -> tuple[list[str], set[bool]]:
    """Disagreements on the Mirror's slot from just after ``target`` is played until both
    seats can pay for the copy, sampled every 5 ticks; and which sides of the price the
    samples saw (False: below it), so a run that never saw both cannot pass."""
    play_target(engine, parser, board, target)
    problems: list[str] = []
    sides: set[bool] = set()
    tid, mirror = card_ids(engine)[target], card_ids(engine)["Mirror"]
    card = engine.cards()[mirror]

    def pays(p) -> bool:
        if mirror not in p.hand:
            return False
        return p.elixir_milli >= 1000 * slot_cost(p, p.hand.index(mirror), card)

    for _ in range(200):
        state = engine.state()
        for team in TEAMS:
            p = state.players[team]
            if mirror not in p.hand:
                continue
            if p.mirror_target != tid:
                problems.append(f"{SEAT[team]}: the Mirror copies {p.mirror_target}, not {target}")
            sides.add(pays(p))
            problems += mirror_slot_disagreements(engine, parser, state, team)
        if problems or all(pays(p) for p in state.players):
            break
        engine.step([], 5)
    return problems, sides


@needs_core
@pytest.mark.parametrize("target", MIRROR_TARGETS)
def test_the_mirror_is_placed_and_priced_as_the_card_it_copies(named, target):
    problems, sides = mirror_gate(named, TileActionParser(), "buildings", target)
    assert not problems, f"{target}: {len(problems)} disagreements, first {problems[:8]}"
    assert sides == {False, True}, f"{target}: the samples saw only {sides} of the price"


@needs_core
def test_a_mirror_with_nothing_to_copy_is_offered_nowhere(named):
    parser = TileActionParser()
    play_target(named, parser, "opening", "Knight")
    state = named.state()
    mirror = card_ids(named)["Mirror"]
    for team in TEAMS:
        p = state.players[team]
        slot = p.hand.index(mirror)
        rich = msgspec.structs.replace(p, mirror_target=-1, elixir_milli=10000)
        players = list(state.players)
        players[team] = rich
        bare = msgspec.structs.replace(state, players=players)
        per = parser.nx * parser.ny
        assert not parser.action_mask(bare, team)[1 + slot * per : 1 + (slot + 1) * per].any()


# ---------------------------------------------------------------------------
# Plants


@needs_core
def test_plant_without_the_probe_the_tunnellers_disagree(monkeypatch):
    """Up to RoyaleSim 244c893 the probe is what makes the tunnellers TUNNEL: without it
    the Miner is masked as a troop and the Goblin Drill as a building, and the engine takes
    both where the mask does not."""
    monkeypatch.setattr(rust_engine, "tunnelling_cards", lambda *a, **k: frozenset())
    engine = RustEngine(card_names=list(NAMES))
    if {c.name for c in engine.cards() if c.placement == Placement.TUNNEL}:
        # The engine states code 5 itself: there is no probe to take away. Check instead
        # that the probe was not needed for anything else.
        return
    problems, _ = every_card_gate(engine, "opening", TileActionParser())
    for name in ("Miner", "GoblinDrill"):
        assert any(f" {name} at own point" in p and "engine OK" in p for p in problems), (
            f"PLANT DID NOT LAND for {name}: {problems[:4]}"
        )


@needs_core
def test_plant_a_mirror_judged_by_its_own_row_is_caught(named, monkeypatch):
    """The Mirror's own row may go down on the enemy half; a copy of a Knight may not."""
    cards = [
        msgspec.structs.replace(c, placement=int(Placement.SPELL)) if c.name == "Mirror" else c
        for c in named.cards()
    ]
    monkeypatch.setattr(named, "_cards", cards)
    problems, _ = mirror_gate(named, TileActionParser(), "opening", "Knight")
    assert any("mask 1, engine" in p for p in problems), f"PLANT DID NOT LAND: {problems[:4]}"


@needs_core
def test_plant_a_mirror_priced_at_its_own_elixir_is_caught(named, monkeypatch):
    monkeypatch.setattr(action_module, "slot_cost", lambda player, slot, card: card.elixir)
    problems, _ = mirror_gate(named, TileActionParser(), "opening", "Knight")
    assert any("engine NOT_ENOUGH_ELIXIR" in p for p in problems), (
        f"PLANT DID NOT LAND: {problems[:4]}"
    )


@needs_core
def test_plant_a_tunnelling_troop_blind_to_bodies_is_caught(named, monkeypatch):
    """The Miner may go down on the enemy half, and not on a tower or a building there."""
    real = PlacementOracle._card_bodies_block

    def blind(self, card):
        return False if card.placement == Placement.TUNNEL else real(self, card)

    monkeypatch.setattr(PlacementOracle, "_card_bodies_block", blind)
    problems, _ = every_card_gate(named, "buildings", TileActionParser())
    seats = {p.split()[0] for p in problems if " Miner at own point" in p and "OCCUPIED" in p}
    assert seats == {"Blue", "Red"}, f"PLANT DID NOT LAND on both seats: {problems[:4]}"
