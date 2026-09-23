"""The two answers to "what does tapping a tile with a building card mean".

WHY THERE ARE TWO
    The engine does not refuse a building tap whose tile box does not fit. It moves the
    building to the nearest place it does fit. So the default action space offers taps
    that put the building somewhere the agent did not choose: measured on the compiled
    engine, 124 of the 240 tiles offered to a Cannon and 73 of 240 offered to a Tesla.
    A policy under that arm asks for one cell and gets another more often than not, and
    nothing in its observation says which taps are which.

    ``taps_where_the_building_stays`` offers only the taps that keep their word.

WHAT THIS FILE CHECKS, AND WHY IT CHECKS IT PER BOARD
    The arm is only worth having if it is LOSSLESS: every landing tile reachable under
    the default arm has to stay reachable, or the agent has lost a placement in exchange
    for a tidier action space. That was measured over 14 boards before the arm was
    written, and a sweep is not a proof: relocation targets depend on what is occupied,
    so losslessness is a property of the BOARD. These tests re-check it on every board
    they use, which is the part that would catch a board where it fails.

    Everything is graded against the engine: where a building lands comes from
    ``engine.building_placement`` and never from a rule restated here.
"""

from __future__ import annotations

import numpy as np
import pytest

from royalegym.action import BUILDING_TAP_ARMS, TileActionParser
from royalegym.protocol import (
    BLUE,
    HAND_SIZE,
    RED,
    MatchSetup,
    Placement,
    SpawnSpec,
    to_engine,
    to_own,
)
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available

pytestmark = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))

HONEST = "taps_where_the_building_stays"
SEAT = {BLUE: "Blue", RED: "Red"}
#: An odd building snaps to the tapped tile's centre, an even one to a corner of it, so
#: both are covered: the even case is the one that can never land ON the tapped point.
BUILDINGS = ("Cannon", "Tesla")


def ids_of(engine: RustEngine) -> dict[str, int]:
    return {c.name: c.card_id for c in engine.cards()}


def board(engine: RustEngine, card: str, spawns=(), tower_hp=None) -> MatchSetup:
    """A battle holding ``card`` in every hand slot, so both seats always hold it."""
    deck = [ids_of(engine)[card]] * 8
    kw: dict = {"decks": [deck, deck], "elixir_milli": [10000, 10000]}
    if spawns:
        kw["spawns"] = list(spawns)
    if tower_hp is not None:
        kw["tower_hp"] = tower_hp
    return MatchSetup(**kw)


def at(engine: RustEngine, team: int, card: str, tx: int, ty: int) -> SpawnSpec:
    a = engine.arena()
    x, y = to_engine(a, team, tx * a.subtile + a.subtile // 2, ty * a.subtile + a.subtile // 2)
    return SpawnSpec(team=team, card_id=ids_of(engine)[card], x=x, y=y)


def landing_tile(engine: RustEngine, team: int, card: str, tx: int, ty: int):
    """The own-frame tile a tap on ``(tx, ty)`` actually builds on, or None if refused."""
    a = engine.arena()
    x, y = to_engine(a, team, tx * a.subtile + a.subtile // 2, ty * a.subtile + a.subtile // 2)
    got = engine.building_placement(team, card, x, y)
    if got is None:
        return None
    ox, oy = to_own(a, team, got[0], got[1])
    return ox // a.subtile, oy // a.subtile


def offered(parser: TileActionParser, state, team: int, slot: int = 0) -> set[tuple[int, int]]:
    planes = parser.action_mask(state, team)[1:].reshape(HAND_SIZE, parser.ny, parser.nx)
    return {(int(tx), int(ty)) for ty, tx in zip(*np.nonzero(planes[slot]), strict=True)}


def parsers(engine: RustEngine) -> tuple[TileActionParser, TileActionParser]:
    default, honest = TileActionParser(), TileActionParser(buildings=HONEST)
    default.bind(engine)
    honest.bind(engine)
    return default, honest


#: Boards whose occupancy differs, because relocation targets depend on what is there.
BOARDS = {
    "empty": ({}, None),
    "a princess down on each side": ({}, [[2400, 0, 1400], [2400, 1400, 0]]),
    "buildings of both teams": (
        {"spawns": [("Cannon", BLUE, 4, 7), ("Cannon", RED, 9, 4), ("Tesla", BLUE, 12, 9)]},
        None,
    ),
    "a crowded own half": (
        {
            "spawns": [
                ("Cannon", BLUE, tx, ty) for tx, ty in ((3, 3), (6, 6), (9, 3), (12, 6), (15, 9))
            ]
        },
        None,
    ),
}


def build(engine: RustEngine, card: str, name: str) -> None:
    spawns_spec, tower_hp = BOARDS[name]
    spawns = [at(engine, t, c, x, y) for c, t, x, y in spawns_spec.get("spawns", [])]
    engine.reset(5, board(engine, card, spawns, tower_hp))


@pytest.fixture(scope="module")
def engine() -> RustEngine:
    return RustEngine()


@pytest.mark.parametrize("name", sorted(BOARDS))
@pytest.mark.parametrize("card", BUILDINGS)
def test_every_tap_the_arm_offers_puts_the_building_on_the_tile_that_was_tapped(
    engine, card, name
) -> None:
    """The guarantee the arm's name makes, graded against the engine, both seats."""
    build(engine, card, name)
    _, honest = parsers(engine)
    state = engine.state()
    for team in (BLUE, RED):
        taps = offered(honest, state, team)
        assert taps, f"{SEAT[team]} was offered no {card} tap at all on {name!r}"
        broken = {t: landing_tile(engine, team, card, *t) for t in taps}
        moved = {t: got for t, got in broken.items() if got != t}
        assert not moved, (
            f"{SEAT[team]} {card} on {name!r}: {len(moved)} taps the arm offered do not "
            f"keep their word, first few {sorted(moved.items())[:5]}"
        )


@pytest.mark.parametrize("name", sorted(BOARDS))
@pytest.mark.parametrize("card", BUILDINGS)
def test_the_arm_is_lossless_on_this_board(engine, card, name) -> None:
    """No placement is given up: every tile reachable under the default arm is still
    reachable under this one.

    This is the claim the arm rests on and it is a property of the BOARD, since where a
    relocated building goes depends on what is occupied. So it is re-checked per board
    rather than taken from the sweep that was run before the arm was written.
    """
    build(engine, card, name)
    default, honest = parsers(engine)
    state = engine.state()
    for team in (BLUE, RED):
        reachable = {
            got
            for t in offered(default, state, team)
            if (got := landing_tile(engine, team, card, *t)) is not None
        }
        kept = offered(honest, state, team)
        lost = reachable - kept
        assert not lost, (
            f"{SEAT[team]} {card} on {name!r}: {len(lost)} landing tiles are reachable "
            f"only by a tap that moves the building, {sorted(lost)[:5]}"
        )


@pytest.mark.parametrize("card", BUILDINGS)
def test_the_arm_only_ever_narrows_the_default(engine, card) -> None:
    """It must not invent a tap. Anything it offers, the engine already accepted."""
    build(engine, card, "buildings of both teams")
    default, honest = parsers(engine)
    state = engine.state()
    for team in (BLUE, RED):
        extra = offered(honest, state, team) - offered(default, state, team)
        assert not extra, (
            f"{SEAT[team]} {card}: the arm offered {sorted(extra)[:5]} the default did not"
        )


def test_the_arm_leaves_troops_and_spells_alone(engine) -> None:
    """It is a rule about buildings. A troop or a spell mask must be bit-identical."""
    for card in ("Knight", "Fireball"):
        build(engine, card, "buildings of both teams")
        default, honest = parsers(engine)
        state = engine.state()
        for team in (BLUE, RED):
            a = default.action_mask(state, team)
            b = honest.action_mask(state, team)
            assert np.array_equal(a, b), f"{SEAT[team]} {card}: the arm changed a non-building mask"


@pytest.mark.parametrize("card", BUILDINGS)
def test_the_two_seats_are_offered_the_same_taps_on_a_mirrored_board(engine, card) -> None:
    """Own-frame, so a rule that held for one colour and not the other would show here.

    The board has to be a mirror of itself for this to mean anything: each team gets the
    same buildings on the same OWN-frame tiles. On a board whose occupancy differs by
    seat the two masks differ for a good reason and the comparison says nothing.
    """
    tiles = ((4, 7), (9, 4), (12, 9))
    spawns = [at(engine, t, card, tx, ty) for tx, ty in tiles for t in (BLUE, RED)]
    engine.reset(5, board(engine, card, spawns))
    _, honest = parsers(engine)
    state = engine.state()
    blue, red = (offered(honest, state, t) for t in (BLUE, RED))
    assert blue == red, (
        f"{card}: Blue was offered {len(blue)} taps and Red {len(red)}; "
        f"only Blue {sorted(blue - red)[:5]}, only Red {sorted(red - blue)[:5]}"
    )


def test_the_default_arm_is_the_one_that_ships(engine) -> None:
    """A change of default is a change of action space for everyone, so it is a test."""
    build(engine, "Cannon", "empty")
    default, _ = parsers(engine)
    assert default.buildings == "any_tap"
    assert default.config()["buildings"] == "any_tap"
    assert set(BUILDING_TAP_ARMS) == {"any_tap", HONEST}


def test_an_engine_that_cannot_say_where_a_building_lands_refuses_the_arm() -> None:
    """MockEngine has no tile box to move, so it cannot answer. Refusing beats guessing,
    and beats silently behaving like the default arm."""
    from royalegym.mock_engine import MockEngine

    with pytest.raises(NotImplementedError, match="building_placement"):
        TileActionParser(buildings=HONEST).bind(MockEngine())


def test_an_unknown_arm_is_refused_when_it_is_named() -> None:
    with pytest.raises(ValueError, match="buildings must be one of"):
        TileActionParser(buildings="wherever_it_likes")


def test_the_arm_records_itself_in_config(engine) -> None:
    """A checkpoint has to say which action space produced it."""
    build(engine, "Cannon", "empty")
    _, honest = parsers(engine)
    assert honest.config()["buildings"] == HONEST


@pytest.mark.parametrize("card", BUILDINGS)
def test_the_arm_actually_removes_something(engine, card) -> None:
    """Vacuity guard. If the arm offered everything the default does, every test above
    would pass while the arm did nothing at all."""
    build(engine, card, "buildings of both teams")
    default, honest = parsers(engine)
    state = engine.state()
    for team in (BLUE, RED):
        removed = offered(default, state, team) - offered(honest, state, team)
        assert removed, (
            f"{SEAT[team]} {card}: the arm removed no tap, so either every tap already "
            "kept its word on this board or the arm is not doing anything"
        )


def test_a_building_card_is_what_these_boards_are_made_of(engine) -> None:
    """The boards are only about buildings if the cards in them are buildings."""
    cards = {c.name: c for c in engine.cards()}
    for name in BUILDINGS:
        assert cards[name].placement == Placement.BUILDING, f"{name} is not a building card"
