"""The action space graded against the arena, not against itself.

WHAT THE EXISTING GATE DOES NOT SAY. ``mask_disagreements`` is exhaustive and it
compares two COMPUTATIONS: the parser's mask and the engine's ``check_deploy``. It says
they agree on which actions are legal. It says nothing about whether action k lands
where the policy believes it lands. A parser whose index arithmetic is transposed, or
whose Red mapping is a y-mirror instead of the 180-degree rotation, still produces a
mask the engine accepts -- every action is legal somewhere -- and a policy trained on it
would place cards in the wrong half of the board with nothing reporting an error.

So each check here derives the answer from the ARENA and the parser's published layout
(``space.n``, ``mask_plane_shape()``, ``HAND_SIZE``) and compares it with what ``parse``
actually returns. Written by someone who had not written action.py, which is the
property to keep if these are ever rewritten.

``test_the_engine_accepts_the_legal_and_refuses_the_illegal`` is the one to keep if only
one survives: it is the only check in the suite that exercises the ILLEGAL direction.
Everything else establishes that two descriptions of legality agree; that one makes the
engine actually refuse.
"""

from __future__ import annotations

import numpy as np
import pytest

from royalegym.action import (
    HalfTileActionParser,
    TileActionParser,
    mask_disagreements,
)
from royalegym.env import ClashParallelEnv
from royalegym.mock_engine import MockEngine
from royalegym.obs import SpatialObsBuilder
from royalegym.protocol import BLUE, HAND_SIZE, RED, MatchSetup, ShuffleMode, to_own

PARSERS = [TileActionParser, HalfTileActionParser]


@pytest.fixture(scope="module", params=PARSERS, ids=lambda c: c.__name__)
def bound(request):
    """A parser bound to a mid-battle state, with elixir enough that the mask is wide.

    ASYMMETRIC on purpose: one Blue princess starts destroyed, which opens ground on
    that side for Red. With a symmetric board the two seats' masks are identical --
    correctly, since the arena is rotation-invariant and both hands match -- and a
    test cannot then tell a per-seat mask from one that ignores the seat.
    """
    engine = MockEngine()
    deck = list(range(8))
    engine.reset(
        3,
        MatchSetup(
            decks=[deck, deck],
            shuffle=ShuffleMode.NONE,
            start_tick=900,
            elixir_milli=[100000, 100000],
            tower_hp=[[2400, 0, 1400], [2400, 1400, 1400]],
        ),
    )
    engine.step([], 30)
    parser = request.param()
    parser.bind(engine)
    return engine, parser, engine.state()


def test_the_published_layout_is_the_space_minus_the_no_op(bound):
    engine, parser, _ = bound
    planes = parser.mask_plane_shape()
    assert planes is not None
    assert planes[0] == HAND_SIZE
    assert 1 + planes[0] * planes[1] * planes[2] == int(parser.space.n)
    arena = engine.arena()
    assert planes[1] == arena.tiles_y * parser.pitch_div
    assert planes[2] == arena.tiles_x * parser.pitch_div


def test_every_index_means_the_grid_point_the_arena_puts_at_that_index(bound):
    """Distinct (slot, own point) per action, and the point derived from the arena."""
    engine, parser, state = bound
    arena = engine.arena()
    planes = parser.mask_plane_shape()
    pitch = arena.subtile // parser.pitch_div
    per = planes[1] * planes[2]
    seen: dict[tuple[int, int, int], int] = {}
    wrong, missing = [], 0
    for action in range(1, int(parser.space.n)):
        cmd = parser.parse(action, state, BLUE)
        if cmd is None:
            missing += 1
            continue
        slot, rest = divmod(action - 1, per)
        yi, xi = divmod(rest, planes[2])
        want = (xi * pitch + pitch // 2, yi * pitch + pitch // 2)
        got = to_own(arena, BLUE, cmd.x, cmd.y)
        key = (slot, *got)
        if key in seen:
            wrong.append(f"actions {seen[key]} and {action} both mean slot {slot} at {got}")
        seen[key] = action
        if got != want and len(wrong) < 3:
            wrong.append(f"action {action}: slot {slot} ({xi},{yi}) -> {got}, arena says {want}")
        if cmd.hand_slot != slot:
            wrong.append(f"action {action}: hand slot {cmd.hand_slot}, index says {slot}")
    assert missing == 0
    assert wrong == [], wrong[0]
    assert len(seen) == int(parser.space.n) - 1, "two actions collapsed onto one placement"


def test_one_index_means_one_own_frame_point_for_both_seats(bound):
    """The whole basis for one policy playing both seats.

    Checked through ``parse``, in engine coordinates, so a Red mapping that is a
    y-mirror rather than a rotation fails here even though its mask would be accepted.
    """
    engine, parser, state = bound
    arena = engine.arena()
    n = int(parser.space.n)
    step = max(1, n // 400)
    sampled = 0
    for action in range(1, n, step):
        blue, red = parser.parse(action, state, BLUE), parser.parse(action, state, RED)
        assert blue is not None
        assert red is not None
        assert blue.team == BLUE
        assert red.team == RED
        assert to_own(arena, BLUE, blue.x, blue.y) == to_own(arena, RED, red.x, red.y), (
            f"action {action} means different own-frame points to the two seats"
        )
        sampled += 1
    assert sampled > 100, f"vacuous: only {sampled} actions sampled"


def test_mask_planes_in_the_observation_are_the_flat_mask(bound):
    """The policy is handed the mask twice; the two copies cannot disagree."""
    engine, parser, state = bound
    planes = parser.mask_plane_shape()
    env = ClashParallelEnv(engine, obs_builder=SpatialObsBuilder(), action_parser=parser)
    for team in (BLUE, RED):
        mask = parser.action_mask(state, team)
        obs = env.obs_builder.build(state, team, mask)
        assert "mask_planes" in obs
        got = np.asarray(obs["mask_planes"], dtype=np.int8)
        assert got.shape == tuple(planes)
        assert np.array_equal(got, np.asarray(mask[1:], dtype=np.int8).reshape(planes))
    assert int(parser.action_mask(state, BLUE).sum()) > 1, "vacuous: only the no-op is legal"


def test_the_engine_accepts_the_legal_and_refuses_the_illegal(bound):
    """The only check that exercises the ILLEGAL direction.

    Everything else establishes that two descriptions of legality agree with each other.
    This makes the engine actually refuse. Run against a saved state and reloaded after,
    so reading the battle cannot change it.
    """
    engine, parser, state = bound
    mask = parser.action_mask(state, BLUE)
    blob = engine.save_state()
    rng = np.random.default_rng(1)
    legal = np.flatnonzero(mask[1:]) + 1
    illegal = np.flatnonzero(mask[1:] == 0) + 1
    assert legal.size, "vacuous: the state has no legal action"
    assert illegal.size, "vacuous: the state has no illegal action"
    picks = list(rng.choice(legal, min(60, legal.size), replace=False)) + list(
        rng.choice(illegal, min(60, illegal.size), replace=False)
    )
    accepted = refused = 0
    disagreed = []
    for action in picks:
        cmd = parser.parse(int(action), state, BLUE)
        assert cmd is not None
        engine.load_state(blob)
        status = engine.check_deploy(cmd)
        if mask[int(action)] and status == 0:
            accepted += 1
        elif not mask[int(action)] and status != 0:
            refused += 1
        else:
            disagreed.append((int(action), int(mask[int(action)]), int(status)))
    engine.load_state(blob)
    assert disagreed == [], f"{len(disagreed)} disagreed, first {disagreed[0]}"
    assert accepted > 0
    assert refused > 0, "vacuous: the engine refused nothing, so the illegal half graded nothing"


def test_encode_and_decode_are_inverses_over_the_whole_space(bound):
    """``parse`` uses ``decode``; nothing else did, so ``encode`` could drift alone.

    Found by a plant: transposing x and y inside ``encode`` left every check above
    green, because they all derive the grid point from the index the same way
    ``decode`` does. ``encode`` is what a caller uses to ask "which action plays slot
    s on tile (x, y)" -- tests/test_env_obs.py asks exactly that of ``mask_planes`` --
    so the two have to be inverses, and now something says so.
    """
    _, parser, _ = bound
    for slot in range(HAND_SIZE):
        for xi in (0, 1, parser.nx // 2, parser.nx - 1):
            for yi in (0, 1, parser.ny // 2, parser.ny - 1):
                action = parser.encode(slot, xi, yi)
                assert 0 < action < int(parser.space.n)
                assert parser.decode(action) == (slot, xi, yi)
    seen = {parser.encode(s, x, y) for s in range(HAND_SIZE)
            for x in range(parser.nx) for y in range(parser.ny)}
    assert len(seen) == int(parser.space.n) - 1, "encode is not onto the non-no-op space"


def test_the_mask_and_parse_agree_about_red(bound):
    """The mask has its OWN seat mapping, and it can drift from ``parse``'s.

    Found by a plant: dropping the ``grid[::-1, ::-1]`` in ``action_mask``'s Red branch
    left every check above green, because they all go through ``parse``. This is
    ``mask_disagreements`` -- the mask against what the engine says about the command
    ``parse`` builds -- run here for both seats so this file cannot pass while the two
    halves of the action space disagree about which end of the board Red is at.
    """
    engine, parser, state = bound
    for team in (BLUE, RED):
        assert mask_disagreements(engine, parser, state, team) == []
    # Vacuity: Red's legal set is not the same as Blue's on this state, so a mapping
    # that ignored the seat could not pass by coincidence.
    blue_mask = parser.action_mask(state, BLUE)
    red_mask = parser.action_mask(state, RED)
    assert blue_mask.sum() > 1
    assert not np.array_equal(blue_mask, red_mask)
