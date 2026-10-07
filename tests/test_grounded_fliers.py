"""A flier held on the ground: the Vines.

A Vines catch drags a flier to the ground for a while: a player sees it there, and ground
attacks reach it. ``EntityState.flying`` still says what the unit IS (a Minion), so up to
royalesim 0.1.19 the observation drew it in the air channel for the whole hold, while the
engine targeted it as a ground unit. From 0.1.20 the engine sets ``STATUS_GROUNDED`` (512) for
the hold and lists its bits in ``royalesim.STATUS_BITS``. The spatial troop channels and the
entity list's ``flying`` feature then follow ``protocol.in_the_air``: a held flier counts as a
ground troop. An engine that does not list the bit is read as before.

SKIPS
    The engine-backed test skips on an engine that does not set the bit (royalesim before
    0.1.20). Not a pass.
"""

from __future__ import annotations

import msgspec
import pytest

from royalegym.action import TileActionParser
from royalegym.mock_engine import MockEngine
from royalegym.obs import EntityListObsBuilder, SpatialObsBuilder
from royalegym.protocol import (
    BLUE,
    RED,
    STATUS_BIT_NAMES,
    STATUS_GROUNDED,
    DeployCommand,
    DeployStatus,
    EntityKind,
    EntityState,
    MatchSetup,
    ShuffleMode,
    in_the_air,
    to_engine,
)
from royalegym.rust_engine import CORE_IMPORT_ERROR, core_available

needs_engine = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))


class SaysBits(MockEngine):
    """MockEngine that lists its status bits as royalesim 0.1.20 does."""

    def status_bits(self) -> list[str]:
        return list(STATUS_BIT_NAMES)


def _flier(arena, status: int) -> EntityState:
    t = arena.subtile
    x, y = to_engine(arena, RED, 9 * t + t // 2, 20 * t + t // 2)
    return EntityState(uid=900, team=RED, kind=int(EntityKind.TROOP), card_id=7, tower_slot=-1,
                       x=x, y=y, hp=100, max_hp=100, radius=500, flying=True, deploy_ticks=0,
                       status_flags=status)


def _seen(engine, builder, status: int):
    parser = TileActionParser()
    engine.reset(1, MatchSetup(decks=[list(range(8)), list(range(8))]))
    parser.bind(engine)
    builder.bind(engine, parser)
    state = engine.state()
    builder.reset(state)
    towers = [msgspec.structs.replace(e, status_flags=0) for e in state.entities]
    state = msgspec.structs.replace(state, entities=[*towers, _flier(engine.arena(), status)])
    return builder.build(state, BLUE, parser.action_mask(state, BLUE))


def _air_ground(obs, builder) -> tuple[float, float]:
    names = builder.channel_names()
    sp = obs["spatial"]
    return (float(sp[names.index("enemy_air_troops")].sum()),
            float(sp[names.index("enemy_ground_troops")].sum()))


def test_the_predicate():
    flier = EntityState(1, RED, int(EntityKind.TROOP), 7, -1, 0, 0, 1, 1, 1, True, 0)
    walker = msgspec.structs.replace(flier, flying=False, status_flags=0)
    held = msgspec.structs.replace(flier, status_flags=STATUS_GROUNDED)
    assert in_the_air(flier, True), "status not reported: in the air, as before"
    assert in_the_air(msgspec.structs.replace(flier, status_flags=0), True)
    assert not in_the_air(held, True)
    assert in_the_air(held, False), "an engine that does not say the bit: not read"
    assert not in_the_air(walker, True)
    assert STATUS_BIT_NAMES.index("grounded") == 9
    assert STATUS_GROUNDED == 1 << 9


def test_a_held_flier_counts_as_a_ground_troop_where_the_engine_says_it():
    builder = SpatialObsBuilder()
    assert _air_ground(_seen(SaysBits(), builder, 0), builder) == (1.0, 0.0)
    assert _air_ground(_seen(SaysBits(), builder, STATUS_GROUNDED), builder) == (0.0, 1.0)


def test_an_engine_that_does_not_list_the_bit_is_read_as_before():
    builder = SpatialObsBuilder()
    assert _air_ground(_seen(MockEngine(), builder, STATUS_GROUNDED), builder) == (1.0, 0.0)


def test_the_entity_lists_flying_feature_follows_the_same_rule():
    builder = EntityListObsBuilder()
    for status, want in ((0, 1.0), (STATUS_GROUNDED, 0.0)):
        rows = _seen(SaysBits(), builder, status)["entities"]
        flier = rows[(rows[:, 0] == 1) & (rows[:, 3 + int(EntityKind.TROOP)] == 1)]
        assert flier.shape[0] == 1
        assert flier[0][12] == want, status


def test_an_engine_naming_its_bits_otherwise_is_refused():
    class Shuffled(MockEngine):
        def status_bits(self) -> list[str]:
            names = list(STATUS_BIT_NAMES)
            names[5], names[9] = names[9], names[5]
            return names

    with pytest.raises(ValueError, match="another name"):
        _seen(Shuffled(), SpatialObsBuilder(), 0)


@needs_engine
def test_vines_on_minions_moves_them_to_the_ground_channel_for_the_hold():
    """On RustEngine: Red's Minions, then Blue's Vines on them. While the engine holds them
    (STATUS_GROUNDED), Blue sees them in enemy_ground_troops and not in enemy_air_troops; when
    the hold ends the survivors are back in the air channel."""
    from royalegym.rust_engine import RustEngine
    from royalegym.state_mutator import deck_ids

    eng = RustEngine()
    bits = eng.status_bits() or []
    if "grounded" not in bits:
        pytest.skip(f"SKIPPED, NOT PASSED: this engine sets no grounded bit ({bits or 'none'})")
    blue = deck_ids(["Vines", "Knight", "Archer", "Giant", "Minions", "Fireball", "Zap",
                     "Cannon"], eng.cards())
    red = deck_ids(["Minions", "Knight", "Archer", "Giant", "Vines", "Fireball", "Zap",
                    "Cannon"], eng.cards())
    eng.reset(3, MatchSetup(decks=[blue, red], shuffle=ShuffleMode.NONE,
                            elixir_milli=[10000, 10000],
                            start_tick=eng.rules().deploy_lockout_ticks))
    a = eng.arena()
    t = a.subtile
    rx, ry = to_engine(a, RED, 9 * t + t // 2, 14 * t + t // 2)
    assert eng.step([DeployCommand(RED, 0, rx, ry)], 1)[0].status == DeployStatus.OK
    eng.step([], 30)
    minion = next(e for e in eng.state().entities if e.team == RED and e.flying)
    assert eng.step([DeployCommand(BLUE, 0, minion.x, minion.y)], 1)[0].status == DeployStatus.OK
    parser = TileActionParser()
    parser.bind(eng)
    builder = SpatialObsBuilder()
    builder.bind(eng, parser)
    builder.reset(eng.state())
    held_ticks = 0
    for _ in range(120):
        s = eng.state()
        red_fliers = [e for e in s.entities if e.team == RED and e.flying]
        held = [e for e in red_fliers if not in_the_air(e, True)]
        air, ground = _air_ground(builder.build(s, BLUE, parser.action_mask(s, BLUE)), builder)
        assert air == len(red_fliers) - len(held), (s.tick, air, len(red_fliers), len(held))
        assert ground >= len(held), (s.tick, ground, len(held))
        held_ticks += bool(held)
        eng.step([], 1)
    assert held_ticks > 20, "the Vines never held a minion: this test would show nothing"
