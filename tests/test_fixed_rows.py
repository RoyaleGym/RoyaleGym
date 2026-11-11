"""Own commands and the enemy's cycle as three vector options, each off by default (11-11).

``own_button_pending`` [K]: which own ability buttons have a press waiting to run.
``own_waiting_plays`` [2 x 5]: each own play waiting to run, soonest first: present, the tap in the
own frame, ticks left / 32, cost / MAX_MANA. ``enemy_card_cycle`` [2 x n]: per card, the plays
until it is back in the enemy's hand from the deduced cycle (the queue ``enemy_queue_5_8`` shows),
and the plays since the enemy last played it. All from what a player knows: its own commands and
the enemy's plays it watched. Both builders take them, and ``EntityListObsBuilder`` takes the
vector options ``SpatialObsBuilder`` always had.

SKIPS
    The engine test skips only without the engine.
"""

from __future__ import annotations

import msgspec
import numpy as np
import pytest

from royalegym.action import TileActionParser
from royalegym.env import ClashParallelEnv
from royalegym.mock_engine import MockEngine
from royalegym.obs import EntityListObsBuilder, SpatialObsBuilder, vector_layout
from royalegym.protocol import BLUE, EMPTY_CARD, HAND_SIZE, RED, MatchSetup
from royalegym.rust_engine import CORE_IMPORT_ERROR, core_available

FLAGS = dict(own_button_pending=True, own_waiting_plays=True, enemy_card_cycle=True)


def test_off_by_default_and_appended_after_the_enemy_queue():
    plain = [f.key for f in vector_layout(20, enemy_queue=True)]
    more = [f.key for f in vector_layout(20, enemy_queue=True, button_pending=3,
                                         waiting_plays=True, enemy_card_cycle=True)]
    assert more[: len(plain)] == plain, "an existing field moved"
    assert more[len(plain):] == ["own_button_pending", "own_waiting_plays", "enemy_card_cycle"]
    sizes = {f.key: f.size for f in vector_layout(20, button_pending=3, waiting_plays=True,
                                                  enemy_card_cycle=True)}
    assert (sizes["own_button_pending"], sizes["own_waiting_plays"], sizes["enemy_card_cycle"]) == (
        3, 10, 40
    )
    for builder in (SpatialObsBuilder, EntityListObsBuilder):
        assert not set(FLAGS) & set(builder().config())
        cfg = builder(**FLAGS).config()
        assert all(cfg[k] is True for k in FLAGS)


def _bound(builder):
    eng = MockEngine()
    parser = TileActionParser(ability_buttons=True)
    parser.bind(eng)
    builder.bind(eng, parser)
    eng.reset(1, MatchSetup(decks=[list(range(8)), list(range(8))]))
    s = eng.state()
    builder.reset(s)
    return eng, parser, s


@pytest.mark.parametrize("builder_cls", [SpatialObsBuilder, EntityListObsBuilder])
def test_own_pending_rows_fill_the_button_and_waiting_play_fields(builder_cls):
    b = builder_cls(own_button_pending=True, own_waiting_plays=True)
    eng, parser, s = _bound(b)
    a = eng.arena()
    t = a.subtile
    later = [0, 5, 6 * t, 9 * t, 12, 3]
    sooner = [0, 2, 3 * t, 20 * t, 5, 4]
    press = [1, HAND_SIZE + 1, 0, 0, 7, 2]
    players = [msgspec.structs.replace(s.players[BLUE], pending=[later, press, sooner]),
               msgspec.structs.replace(s.players[RED], pending=[later])]
    st = msgspec.structs.replace(s, players=players)
    off = b.vector_offsets()
    vec = b.build(st, BLUE, parser.action_mask(st, BLUE))["vector"]
    assert vec[off["own_button_pending"]].tolist() == [0.0, 1.0, *[0.0] * (b.ability_buttons - 2)]
    plays = vec[off["own_waiting_plays"]].reshape(2, 5)
    assert plays[0].tolist() == pytest.approx([1, 3 * t / a.width, 20 * t / a.height, 5 / 32, 0.4])
    assert plays[1].tolist() == pytest.approx([1, 6 * t / a.width, 9 * t / a.height, 12 / 32, 0.3])
    red = b.build(st, RED, parser.action_mask(st, RED))["vector"]
    assert not red[off["own_button_pending"]].any()
    (one, empty) = red[off["own_waiting_plays"]].reshape(2, 5)
    assert one.tolist() == pytest.approx(
        [1, (a.width - 6 * t) / a.width, (a.height - 9 * t) / a.height, 12 / 32, 0.3]
    ), "Red's tap is in Red's frame"
    assert not empty.any()


def test_a_parser_without_buttons_is_refused_for_the_button_field():
    eng = MockEngine()
    parser = TileActionParser()
    parser.bind(eng)
    with pytest.raises(ValueError, match="own_button_pending"):
        SpatialObsBuilder(own_button_pending=True).bind(eng, parser)


@pytest.mark.parametrize("builder_cls", [SpatialObsBuilder, EntityListObsBuilder])
def test_the_enemy_cycle_follows_the_deduced_queue_and_the_plays_since(builder_cls):
    env = ClashParallelEnv(MockEngine(), obs_builder=builder_cls(enemy_card_cycle=True,
                                                                  enemy_queue=True))
    obs, _ = env.reset(seed=6)
    rng = np.random.default_rng(6)
    off = env.obs_builder.vector_offsets()
    n = env.obs_builder.num_cards
    checked = {"queue": 0, "recent": 0}
    for _ in range(200):
        if not env.agents:
            break
        acts = {}
        for agent in env.agents:
            legal = np.flatnonzero(obs[agent]["action_mask"])[1:]
            acts[agent] = int(rng.choice(legal)) if legal.size and rng.random() < 0.5 else 0
        obs, *_ = env.step(acts)
        if "blue" not in obs:
            break
        memory = env.obs_builder.memory[BLUE]
        cycle = obs["blue"]["vector"][off["enemy_card_cycle"]].reshape(2, n)
        queue = memory.enemy_queue()
        for q, card in enumerate(queue):
            if card != EMPTY_CARD:
                assert cycle[0, card] == pytest.approx((q + 1) / 4)
                checked["queue"] += 1
        assert (cycle[0] > 0).sum() == sum(c != EMPTY_CARD for c in queue)
        if memory.foe_recent:
            assert cycle[1, memory.foe_recent[-1]] == pytest.approx(1 / 8), "just played"
            checked["recent"] += 1
        assert ((cycle[1] > 0) == memory.foe_seen).all(), "recency only for cards seen"
    assert checked["queue"] > 20, checked
    assert checked["recent"] > 20, checked


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_the_entity_builder_writes_the_spatial_builders_vector_for_the_same_options():
    from royalegym import make_env

    opts = dict(evolutions=True, evolution_progress=True, card_status=True, button_index=True,
                enemy_queue=True, elixir_income=True, **FLAGS)
    deck = ["Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Cannon", "Musketeer"]
    envs = [make_env(deck=deck, evolved=["Archer"], heroes=["Giant"], command_delay_ticks=21,
                     obs_builder=builder(**opts),
                     action_parser=TileActionParser(ability_buttons=True))
            for builder in (SpatialObsBuilder, EntityListObsBuilder)]
    obs = [e.reset(seed=8)[0] for e in envs]
    rng = np.random.default_rng(8)
    waited = 0
    for _ in range(150):
        if not envs[0].agents:
            break
        acts = {}
        for agent in envs[0].agents:
            legal = np.flatnonzero(obs[0][agent]["action_mask"])[1:]
            acts[agent] = int(rng.choice(legal)) if legal.size and rng.random() < 0.5 else 0
        obs = [e.step(acts)[0] for e in envs]
        for agent in obs[0]:
            assert np.array_equal(obs[0][agent]["vector"], obs[1][agent]["vector"]), agent
        off = envs[0].obs_builder.vector_offsets()
        waited += int(obs[0]["blue"]["vector"][off["own_waiting_plays"]][0] > 0)
    assert waited > 5, "vacuous: no play ever waited under the delay"
