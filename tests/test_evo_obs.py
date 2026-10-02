"""Evolution inputs in the observation: ``SpatialObsBuilder(evolutions=True)``.

What a player sees of evolutions, and nothing more:
- per own hand slot, whether playing it now puts down the evolved form, and the same for
  the next card (the client shows the charge on your own cards);
- with ``evolution_progress=True`` as well, how far each hand card's counter has got;
- two board planes, own and enemy evolved units, counted per tile like the troop planes.
The enemy's counters are never read: the client does not show them.

Off by default, so every existing observation is byte-for-byte what it was.

SKIPS
    The engine-backed tests skip without the engine. Not a pass.
"""

from __future__ import annotations

import msgspec
import numpy as np
import pytest

from royalegym import make_env
from royalegym.mock_engine import MockEngine
from royalegym.obs import (
    HAND_SIZE,
    MatchClock,
    MatchMemory,
    Reveal,
    SpatialObsBuilder,
    fair_fields,
    spatial_channels,
    vector_offsets,
)
from royalegym.rust_engine import CORE_IMPORT_ERROR, core_available
from royalegym.protocol import ElixirLaw, default_calibration

needs_engine = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))

DECK = ["Skeletons", "Musketeer", "Knight", "Archer", "Fireball", "Zap", "Cannon", "Giant"]


# --- layout -------------------------------------------------------------------------------


def test_off_by_default_the_layout_is_unchanged():
    assert vector_offsets(136) == vector_offsets(136, evolutions=False)
    assert "own_hand_evolved" not in vector_offsets(136)
    assert [n for n, _ in spatial_channels()] == [n for n, _ in spatial_channels(None, False)]


def test_the_evolution_fields_come_after_every_existing_fair_field():
    """Turning the flag on moves no existing offset, and the fair block stays contiguous."""
    for last in (False, True):
        before = vector_offsets(136, Reveal(enemy_hand=True), enemy_last_card=last)
        after = vector_offsets(
            136, Reveal(enemy_hand=True), enemy_last_card=last, evolutions=True
        )
        fair_end = max(s.stop for k, s in before.items() if k != "enemy_hand_cards")
        assert after["own_hand_evolved"] == slice(fair_end, fair_end + HAND_SIZE)
        assert after["own_next_evolved"] == slice(fair_end + HAND_SIZE, fair_end + HAND_SIZE + 1)
        for key, s in before.items():
            if key != "enemy_hand_cards":
                assert after[key] == s, key
        assert after["enemy_hand_cards"].start == fair_end + HAND_SIZE + 1


def test_progress_comes_after_the_flags_and_needs_them():
    off = vector_offsets(136, evolutions=True, evolution_progress=True)
    assert off["own_hand_evo_progress"].start == off["own_next_evolved"].stop
    assert off["own_hand_evo_progress"].stop - off["own_hand_evo_progress"].start == HAND_SIZE
    with pytest.raises(ValueError, match="evolution_progress needs evolutions"):
        SpatialObsBuilder(evolution_progress=True)


def test_the_two_board_planes_follow_the_fair_planes():
    names = [n for n, _ in spatial_channels(Reveal(enemy_spell_aim=True), True)]
    fair = [n for n, _ in spatial_channels()]
    assert names[: len(fair)] == fair
    assert names[len(fair) : len(fair) + 2] == ["own_evolved", "enemy_evolved"]
    assert names[-1] == "enemy_spell_aim"


# --- the fair contract, without an engine -----------------------------------------------


def _fair(hand, next_card, own_evo, progress=False):
    cards = list(MockEngine().cards())
    memory = MatchMemory(len(cards), ElixirLaw.load(default_calibration()))
    memory.bind(cards)
    return fair_fields(
        memory, MatchClock.at(0), hand, next_card, 5000, cards, 10,
        evolutions=True, evolution_progress=progress, own_evo=own_evo,
    )


def test_fair_fields_flag_the_slot_whose_next_play_is_evolved():
    out = _fair([3, 5, 7, 9], 2, [[5, 2, 1], [9, 1, 0]])
    assert out["own_hand_evolved"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert out["own_next_evolved"].tolist() == [0.0]
    out = _fair([3, 5, 7, 9], 9, [[9, 2, 1]])
    assert out["own_hand_evolved"].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert out["own_next_evolved"].tolist() == [1.0]


def test_progress_is_plays_over_the_engines_cycle_length():
    out = _fair([3, 5, 7, 9], 2, [[5, 1, 0, 2], [9, 0, 0, 1], [7, 1, 1, 1]], progress=True)
    assert out["own_hand_evo_progress"].tolist() == [0.0, 0.5, 1.0, 0.0]


def test_progress_refuses_rows_without_a_cycle_length():
    with pytest.raises(ValueError, match="cycle length"):
        _fair([3, 5, 7, 9], 2, [[5, 1, 0]], progress=True)


# --- on the engine ------------------------------------------------------------------------


def _env(**forms):
    return make_env(
        deck=DECK, obs_builder=SpatialObsBuilder(evolutions=True), **forms
    )


def _cycle(env, plays_wanted, watch, prefer=None):
    """Blue plays at (9, 10): the card named ``prefer`` whenever it can, else the first legal
    hand slot. Red waits. Calls ``watch(obs, slot, card, None)`` just before each Blue play and
    ``watch(obs, -1, -1, state)`` after every step."""
    obs, _ = env.reset(seed=0)
    p = env.action_parser
    names = {c.card_id: c.name for c in env.engine.cards()}
    played = 0
    while env.agents and played < plays_wanted:
        me = env.battle_state.players[0]
        act, slot = p.noop(), -1
        legal = [s for s in range(HAND_SIZE) if obs["blue"]["action_mask"][p.encode(s, 9, 10)]]
        liked = [s for s in legal if names.get(me.hand[s]) == prefer]
        if not liked and any(names.get(c) == prefer for c in me.hand):
            legal = []  # save up for the preferred card rather than spend on another
        if liked or legal:
            slot = (liked or legal)[0]
            act = p.encode(slot, 9, 10)
        if slot >= 0:
            watch(obs, slot, me.hand[slot], None)
            played += 1
        obs, *_ = env.step({"blue": act, "red": p.noop()})
        watch(obs, -1, -1, env.battle_state)


def _cycles_to_evolved(card_name, evolved):
    env = _env(evolved=evolved)
    names = {c.card_id: c.name for c in env.engine.cards()}
    off = env.obs_builder.vector_offsets()
    sp = env.obs_builder.channel_names()
    own_ev, foe_ev = sp.index("own_evolved"), sp.index("enemy_evolved")
    seen = {"plays": [], "evolved_on_board": [], "red_sees": []}

    def watch(obs, slot, card, state):
        if slot >= 0 and names[card] == card_name:
            seen["plays"].append(float(obs["blue"]["vector"][off["own_hand_evolved"]][slot]))
        if state is not None:
            seen["evolved_on_board"].append(float(obs["blue"]["spatial"][own_ev].sum()))
            seen["red_sees"].append(float(obs["red"]["spatial"][foe_ev].sum()))

    _cycle(env, 60, watch, prefer=card_name)
    return seen


@needs_engine
def test_evo_skeletons_play_evolved_on_the_third_play_and_light_both_planes():
    seen = _cycles_to_evolved("Skeletons", ["Skeletons"])
    assert seen["plays"][:6] == [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    # The board planes: nothing evolved until the third Skeletons play lands, then the
    # three evolved Skeletons, seen by Blue as its own and by Red as the enemy's.
    assert max(seen["evolved_on_board"]) >= 3
    assert seen["evolved_on_board"] == seen["red_sees"]
    first = next(i for i, v in enumerate(seen["evolved_on_board"]) if v)
    assert all(v == 0 for v in seen["evolved_on_board"][:first])


@needs_engine
def test_evo_musketeer_plays_evolved_on_its_cycle_too():
    seen = _cycles_to_evolved("Musketeer", ["Musketeer"])
    assert 1.0 in seen["plays"]
    k = seen["plays"].index(1.0)
    assert seen["plays"][:k] == [0.0] * k and k >= 1
    assert max(seen["evolved_on_board"]) >= 1


@needs_engine
def test_a_hero_is_not_an_evolved_unit():
    env = _env(heroes=["Musketeer"])
    sp = env.obs_builder.channel_names()
    own_ev = sp.index("own_evolved")
    lit = []
    _cycle(env, 16, lambda obs, slot, card, state: lit.append(obs["blue"]["spatial"][own_ev].sum()))
    assert max(lit) == 0


@needs_engine
def test_the_enemys_counters_are_never_read():
    """Fair: changing the enemy's evo rows changes nothing in Blue's observation."""
    env = _env(evolved=["Skeletons"])
    obs, _ = env.reset(seed=0)
    state = env.battle_state
    mask = np.ones(env.action_parser.n_actions, dtype=np.int8)
    b = env.obs_builder
    b.reset(state)
    plain = b.build(state, 0, mask)
    b.reset(state)
    red = state.players[1]
    rows = [[c, 2, 1, 2] for c, *_ in red.evo] or [[0, 2, 1, 2]]
    forged = msgspec.structs.replace(
        state, players=[state.players[0], msgspec.structs.replace(red, evo=rows)]
    )
    seen = b.build(forged, 0, mask)
    for key in plain:
        assert np.array_equal(plain[key], seen[key]), key


def test_the_flag_refuses_an_engine_that_does_not_report_evolved_units():
    from royalegym import ClashParallelEnv

    env = ClashParallelEnv(engine=MockEngine(), obs_builder=SpatialObsBuilder(evolutions=True))
    with pytest.raises(ValueError, match="does not report which units are evolved"):
        env.reset(seed=0)


def test_config_records_the_flags():
    b = SpatialObsBuilder(evolutions=True, evolution_progress=True)
    cfg = b.config()
    assert cfg["evolutions"] is True and cfg["evolution_progress"] is True
    assert "evolutions" not in SpatialObsBuilder().config()
