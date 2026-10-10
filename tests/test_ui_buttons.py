"""``ui_buttons=True``: the mask follows the game's ability buttons (2026-10-09).

In the game a hero's or champion's ability button sits over two back corners of its owner's
side while that unit lives, and a card placed there by tapping lands on the button: nothing is
placed (measured for taps; a drag is not). The engine has no screen, so it accepts those
deploys; a policy that taps, trained on its mask, learns corners the game will not give it.
With the flag the mask refuses every card's deploy on the tiles each SHOWN button covers
(``action.UI_BUTTON_TILES``). Buttons fill from the right: one sits on the right, two sit left
then right in their ``PlayerState.abilities`` order (the deck's). A third's place is not
measured, so with three or more nothing is covered. Off by default: no existing mask moves.

SKIPS
    Everything here needs the engine. Not a pass without it.
"""

from __future__ import annotations

import msgspec
import numpy as np
import pytest

from royalegym.action import (
    UI_BUTTON_TILES,
    HalfTileActionParser,
    TileActionParser,
)
from royalegym.env import ClashParallelEnv
from royalegym.obs import SpatialObsBuilder
from royalegym.protocol import BLUE, HAND_SIZE, RED, EntityKind, ShuffleMode, ability_row
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available
from royalegym.state_mutator import DefaultStateMutator

pytestmark = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))

#: A champion first in hand, unshuffled: the Golden Knight.
ONE = ["GoldenKnight", "Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Cannon"]
#: A hero (the Knight, form 2) then a champion: two buttons, left then right.
TWO = ["Knight", "GoldenKnight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Cannon"]
RIGHT, LEFT = set(UI_BUTTON_TILES["right"]), set(UI_BUTTON_TILES["left"])


def _env(names, forms=None, parser=TileActionParser, **flags):
    eng = RustEngine()
    return ClashParallelEnv(
        eng,
        action_parser=parser(ability_buttons=True, **flags),
        obs_builder=SpatialObsBuilder(),
        state_mutator=DefaultStateMutator(
            decks=[names, names], forms=forms, shuffle=ShuffleMode.NONE
        ),
    )


def _offered(env, agent: str) -> set[tuple[int, int, int]]:
    """(slot, x, y) of every tile action the mask offers ``agent``."""
    p = env.action_parser
    mask = env.action_masks(agent)[1 : p.n_tile_actions].reshape(HAND_SIZE, p.ny, p.nx)
    return {(int(s), int(x), int(y)) for s, y, x in zip(*np.nonzero(mask), strict=True)}


def _step_both(envs, acts):
    for env in envs:
        env.step(acts)


def _play_on_both(envs, agent: str, name: str, x: int = 9, y: int = 8) -> None:
    """Wait until ``agent`` can play ``name`` at own (x, y) on every env, then play it there."""
    for _ in range(300):
        env = envs[0]
        team = BLUE if agent == "blue" else RED
        hand = env.battle_state.players[team].hand
        ids = {c.name: c.card_id for c in env.engine.cards()}
        if ids[name] in hand:
            p = env.action_parser
            action = 1 + hand.index(ids[name]) * p.nx * p.ny + y * p.nx + x
            if all(e.action_masks(agent)[action] for e in envs):
                _step_both(envs, {agent: action, "blue" if agent == "red" else "red": 0})
                return
        _step_both(envs, {"blue": 0, "red": 0})
    raise AssertionError(f"{agent} never could play {name}")


@pytest.mark.parametrize("agent", ["blue", "red"])
def test_one_shown_button_covers_the_right_corner_while_its_unit_lives(agent):
    on, off = _env(ONE, ui_buttons=True), _env(ONE)
    for env in (on, off):
        env.reset(seed=4)
    # Before the champion is on the board there is no button on screen: the masks agree.
    for _ in range(12):
        _step_both((on, off), {"blue": 0, "red": 0})
    assert _offered(on, agent) == _offered(off, agent)
    _play_on_both((on, off), agent, "GoldenKnight")
    team = BLUE if agent == "blue" else RED
    ids = {c.name: c.card_id for c in on.engine.cards()}
    shown = 0
    for _ in range(40):
        lives = any(e.team == team and e.card_id == ids["GoldenKnight"]
                    for e in on.battle_state.entities)
        a, b = _offered(on, agent), _offered(off, agent)
        assert a <= b
        dropped = {(x, y) for _, x, y in b - a}
        if lives:
            assert dropped <= RIGHT, dropped
            assert not {(x, y) for _, x, y in a} & RIGHT
            shown += bool(dropped)
        else:
            assert not dropped
        other = "red" if agent == "blue" else "blue"
        assert _offered(on, other) == _offered(off, other)  # the other seat's own screen
        _step_both((on, off), {"blue": 0, "red": 0})
    assert shown > 5  # vacuity: the corner was offered without the flag and refused with it


def test_two_buttons_sit_left_then_right_in_deck_order():
    forms = [[2] + [0] * 7, [0] * 8]
    on, off = _env(TWO, forms=forms, ui_buttons=True), _env(TWO, forms=forms)
    for env in (on, off):
        env.reset(seed=6)
    ids = {c.name: c.card_id for c in on.engine.cards()}
    rows = [ability_row(r).card_id for r in on.battle_state.players[BLUE].abilities]
    assert rows == [ids["Knight"], ids["GoldenKnight"]], rows  # deck order
    _play_on_both((on, off), "blue", "Knight")  # the hero: button 0, on the LEFT
    dropped: set[tuple[int, int]] = set()
    for _ in range(10):
        a, b = _offered(on, "blue"), _offered(off, "blue")
        dropped |= {(x, y) for _, x, y in b - a}
        _step_both((on, off), {"blue": 0, "red": 0})
    assert dropped, "the hero's corner was never offered without the flag"
    assert dropped <= LEFT, dropped


def test_the_covered_tiles_by_row_count_and_three_buttons_cover_nothing():
    """Forged rows on a real state: the place follows the row's position and the count."""
    env = _env(ONE, ui_buttons=True)
    env.reset(seed=4)
    _play_on_both((env,), "blue", "GoldenKnight")
    state = env.battle_state
    parser = env.action_parser
    gk = state.players[BLUE].abilities[0]
    other = [1, 0, 1, 9999, -1]  # a button whose unit is not on the board
    for rows, want in (
        ([gk], RIGHT),
        ([gk, other], LEFT),
        ([other, gk], RIGHT),
        ([gk, other, other], set()),
        ([other, gk, other], set()),
    ):
        player = msgspec.structs.replace(state.players[BLUE], abilities=rows)
        forged = msgspec.structs.replace(state, players=[player, state.players[RED]])
        assert set(parser.covered_tiles(forged, BLUE)) == want, rows


def test_the_half_tile_parser_covers_every_half_cell_of_a_covered_tile():
    on, off = _env(ONE, parser=HalfTileActionParser, ui_buttons=True), _env(
        ONE, parser=HalfTileActionParser
    )
    for env in (on, off):
        env.reset(seed=4)
    _play_on_both((on, off), "blue", "GoldenKnight", x=18, y=16)
    a, b = _offered(on, "blue"), _offered(off, "blue")
    dropped = {(x, y) for _, x, y in b - a}
    halves = {(2 * x + dx, 2 * y + dy) for x, y in RIGHT for dx in (0, 1) for dy in (0, 1)}
    assert dropped, "no covered half-cell was offered without the flag"
    assert dropped <= halves, dropped
    assert not {(x, y) for _, x, y in a} & halves  # every half of a covered tile, not some


def test_off_by_default_and_named_in_config():
    # Without a hero or champion there is no button: the flag changes no mask.
    names = ["Knight", "Archer", "Goblins", "Giant", "Minions", "Fireball", "Zap", "Cannon"]
    on, off = _env(names, ui_buttons=True), _env(names)
    assert "ui_buttons" not in off.action_parser.config()
    assert on.action_parser.config()["ui_buttons"] is True
    rng = np.random.default_rng(2)
    on.reset(seed=2)
    off.reset(seed=2)
    for _ in range(120):
        if not on.agents:
            break
        for agent in ("blue", "red"):
            assert on.action_masks(agent).tobytes() == off.action_masks(agent).tobytes()
        legal = {a: np.flatnonzero(on.action_masks(a)) for a in on.agents}
        acts = {a: int(rng.choice(v[v != 0])) if (v != 0).any() and rng.random() < 0.3 else 0
                for a, v in legal.items()}
        _step_both((on, off), acts)
    assert EntityKind.TROOP in {e.kind for e in on.battle_state.entities}
