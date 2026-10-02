"""The flat vector written into one buffer is byte for byte the vector built field by field.

``build_vector`` and ``fair_fields`` used to make one small float32 array per field, clip
each, concatenate them and clip again. Since 2026-09-26 both write every field into one
buffer through ``obs._write_fair`` and clip once. That is a speed change and nothing else
(train's ask: it was 27% of a training worker's step), so this file holds the new path to
a frozen copy of the old one, below, byte for byte. It checks every vector and every fair
field of played battles, on both engines, both seats, with the card-identity field and
every reveal.

The copy is the code as it stood at RoyaleGym 603b188, with only its names changed. It
reads the same module constants, so a change of scale moves both sides alike. A change to
a formula has to be made in both places, and this test is where that shows.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from royalegym import obs
from royalegym.action import TileActionParser
from royalegym.mock_engine import MockEngine
from royalegym.obs import (
    LEAK_SCALE,
    PLAY_GAP_TICKS,
    PLAYS_SCALE,
    MatchClock,
    MatchMemory,
    Reveal,
    build_vector,
    fair_fields,
    vector_layout,
)
from royalegym.protocol import (
    DECK_SIZE,
    EMPTY_CARD,
    HAND_SIZE,
    TEAMS,
    BattleState,
    CardInfo,
    ElixirLaw,
    MatchSetup,
    ShuffleMode,
    TowerSlot,
    default_calibration,
)
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available

# -- the old path, frozen ---------------------------------------------------------------


def _old_one_hot(card: int, onehot: int, empty_index: int) -> np.ndarray:
    v = np.zeros(onehot, dtype=np.float32)
    v[empty_index if card == EMPTY_CARD else card] = 1
    return v


def _old_hand_block(
    hand: Sequence[int], cards: list[CardInfo], elixir_milli: int, num_cards: int, max_mana: int
) -> list[np.ndarray]:
    onehot = num_cards + 1
    one = np.zeros((HAND_SIZE, onehot), dtype=np.float32)
    cost = np.zeros(HAND_SIZE, dtype=np.float32)
    afford = np.zeros(HAND_SIZE, dtype=np.float32)
    for i, c in enumerate(hand):
        if c == EMPTY_CARD:
            one[i, num_cards] = 1
            continue
        one[i, c] = 1
        cost[i] = cards[c].elixir / max_mana
        afford[i] = 1.0 if elixir_milli >= cards[c].elixir * 1000 else 0.0
    return [one.reshape(-1), cost, afford]


def _old_fair_fields(
    memory, clock, hand, next_card, own_elixir_milli, cards, max_mana, *,
    enemy_elixir_milli=None, enemy_last_card=False, own_pending=(),
) -> dict[str, np.ndarray]:
    num_cards = len(cards)
    onehot = num_cards + 1
    full = 1000 * max_mana
    foe_milli = memory.enemy_elixir_milli() if enemy_elixir_milli is None else enemy_elixir_milli
    held = sum(int(row[5]) for row in own_pending)
    waiting = {int(row[1]) for row in own_pending if int(row[0]) == 0}
    pending = np.array(
        [1.0 if c != EMPTY_CARD and c in waiting else 0.0 for c in hand], dtype=np.float32
    )
    one, cost, afford = _old_hand_block(
        hand, cards, own_elixir_milli - 1000 * held, num_cards, max_mana
    )
    afford = afford * (1.0 - pending)
    cycle = np.zeros((DECK_SIZE - HAND_SIZE - 1, onehot), dtype=np.float32)
    # Positions 6-8. The old path's queue always held four cards; under the refill timer it holds
    # one more while a played card's slot waits, so read the same three slots the writer reads.
    for i, card in enumerate(memory.own_cycle[1 : DECK_SIZE - HAND_SIZE]):
        cycle[i, num_cards if card == EMPTY_CARD else card] = 1
    reg_left = max(0, clock.regular_ticks - clock.tick) / max(1, clock.regular_ticks)
    ot_left = 0.0
    if clock.overtime:
        ot_end = clock.regular_ticks + clock.overtime_ticks
        ot_left = max(0, ot_end - clock.tick) / max(1, clock.overtime_ticks)
    out = {
        "own_elixir": np.array([own_elixir_milli / full], dtype=np.float32),
        "enemy_elixir": np.array([foe_milli / full], dtype=np.float32),
        "own_hand_cards": one,
        "own_hand_cost": cost,
        "own_hand_affordable": afford,
        "own_hand_pending": pending,
        "own_pending_cost": np.array([held / max_mana], dtype=np.float32),
        "own_next_card": _old_one_hot(next_card, onehot, num_cards),
        "own_cycle_6_8": cycle.reshape(-1),
        "own_deck": memory.own_deck.astype(np.float32),
        "own_last_card": _old_one_hot(memory.own_last_card, onehot, num_cards),
        "own_ticks_since_play": np.array(
            [min(1.0, memory.ticks_since_own_play(clock.tick) / PLAY_GAP_TICKS)], dtype=np.float32
        ),
        "own_elixir_leaked": np.array(
            [min(1.0, memory.leaked_elixir() / LEAK_SCALE)], dtype=np.float32
        ),
        "enemy_cards_seen": memory.foe_seen.astype(np.float32),
        "enemy_possible_hand": memory.enemy_possible_hand().astype(np.float32),
        "enemy_plays": np.array([min(1.0, memory.foe_plays / PLAYS_SCALE)], dtype=np.float32),
        "clock": np.array([reg_left, float(clock.overtime), ot_left], dtype=np.float32),
        "elixir_rate": np.array(
            [float(clock.elixir_rate == r) for r in (1, 2, 3)], dtype=np.float32
        ),
    }
    if enemy_last_card:
        last = memory.foe_recent[-1] if memory.foe_recent else EMPTY_CARD
        out["enemy_last_card"] = _old_one_hot(last, onehot, num_cards)
    return {k: np.clip(v, 0.0, 1.0).astype(np.float32, copy=False) for k, v in out.items()}


def _old_build_vector(
    state: BattleState, team: int, cards: list[CardInfo], max_mana: int, reveal: Reveal,
    memory: MatchMemory, enemy_last_card: bool = False,
) -> np.ndarray:
    me, foe = state.players[team], state.players[1 - team]
    num_cards = len(cards)
    onehot = num_cards + 1
    parts = _old_fair_fields(
        memory, MatchClock.of(state), me.hand, me.next_card, me.elixir_milli, cards, max_mana,
        enemy_elixir_milli=foe.elixir_milli if reveal.enemy_elixir else None,
        enemy_last_card=enemy_last_card, own_pending=me.pending,
    )
    own_hp, foe_hp = (
        np.array([p.tower_hp[s] / max(1, p.tower_max_hp[s]) for s in TowerSlot], dtype=np.float32)
        for p in (me, foe)
    )
    parts["own_tower_hp"] = own_hp
    parts["enemy_tower_hp"] = foe_hp
    parts["crowns"] = np.array([me.crowns / 3.0, foe.crowns / 3.0], dtype=np.float32)
    parts["king_active"] = np.array(
        [float(me.king_active), float(foe.king_active)], dtype=np.float32
    )
    if reveal.enemy_hand:
        one, _cost, _afford = _old_hand_block(
            foe.hand, cards, foe.elixir_milli, num_cards, max_mana
        )
        parts["enemy_hand_cards"] = one
    if reveal.enemy_next_card:
        parts["enemy_next_card"] = _old_one_hot(foe.next_card, onehot, num_cards)
    if reveal.enemy_deck:
        deck = np.zeros(num_cards, dtype=np.float32)
        for c in (*foe.hand, foe.next_card):
            if c != EMPTY_CARD:
                deck[c] = 1
        deck[memory.foe_seen] = 1
        parts["enemy_deck"] = deck
    keys = [f.key for f in vector_layout(1, reveal, enemy_last_card)]
    return np.clip(np.concatenate([parts[k] for k in keys]), 0.0, 1.0).astype(np.float32)


# -- the comparison -----------------------------------------------------------------------

DECK = ["Fireball", "Zap", "Arrows", "Knight", "Archer", "Giant", "Musketeer", "Minions"]
CADENCES = (10, 7, 1, 13, 4)
#: (reveal, enemy_last_card) pairs: the shipped vector, card identity, the one reveal that
#: swaps a slot's source, and every reveal together.
VARIANTS = (
    (Reveal(), False),
    (Reveal(), True),
    (Reveal(enemy_elixir=True), False),
    (Reveal(enemy_elixir=True, enemy_hand=True, enemy_next_card=True, enemy_deck=True), True),
)
ENGINES = {
    "mock": MockEngine,
    "rust": pytest.param(
        RustEngine, marks=pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
    ),
}


def compare_battle(engine, seed: int, start_tick: int, ticks: int) -> tuple[list[str], dict]:
    """Play a random-legal battle; at every step compare new with old for every variant."""
    card = {c.name: i for i, c in enumerate(engine.cards())}
    deck = [card[n] for n in DECK]
    cards = list(engine.cards())
    cal = default_calibration()
    law = ElixirLaw.load(cal)
    max_mana = cal.int("match.MAX_MANA")
    rng = np.random.default_rng(seed)
    parser = TileActionParser()
    parser.bind(engine)
    setup = MatchSetup(decks=[deck, deck], shuffle=ShuffleMode.NONE, start_tick=start_tick)
    engine.reset(seed, setup)
    state = engine.state()
    memories = {}
    for team in TEAMS:
        memories[team] = MatchMemory(len(cards), law)
        memories[team].bind(cards)
        memories[team].seed(state, team)
    bad: list[str] = []
    seen = {"vectors": 0, "fields": 0, "overtime": False, "rate2": False, "plays": 0,
            "tower_down": False, "leaked": False}
    step, end = 0, state.tick + ticks
    while not state.game_over and state.tick < end:
        seen["overtime"] |= state.overtime
        seen["rate2"] |= state.elixir_rate == 2
        seen["tower_down"] |= any(0 in p.tower_hp for p in state.players)
        for team in TEAMS:
            m = memories[team]
            m.observe(state, team)
            me, foe = state.players[team], state.players[1 - team]
            for reveal, elc in VARIANTS:
                args = (state, team, cards, max_mana, reveal, m)
                new = build_vector(*args, enemy_last_card=elc)
                old = _old_build_vector(*args, enemy_last_card=elc)
                seen["vectors"] += 1
                same = new.dtype == old.dtype and new.shape == old.shape
                if not same or new.tobytes() != old.tobytes():
                    diff = np.flatnonzero(new.view(np.uint32) != old.view(np.uint32)) if (
                        new.shape == old.shape) else "shape"
                    bad.append(f"tick {state.tick} seat {team} {reveal} elc={elc}: slots {diff}")
            for elc in (False, True):
                clock = MatchClock.of(state)
                args = (m, clock, me.hand, me.next_card, me.elixir_milli, cards, max_mana)
                for foe_milli in (None, foe.elixir_milli):
                    kw = {"enemy_elixir_milli": foe_milli, "enemy_last_card": elc}
                    new_f = fair_fields(*args, **kw)
                    old_f = _old_fair_fields(*args, **kw)
                    if list(new_f) != list(old_f):
                        bad.append(f"tick {state.tick} fair_fields keys {list(new_f)} != "
                                   f"{list(old_f)}")
                    for k, want in old_f.items():
                        got = new_f.get(k)
                        seen["fields"] += 1
                        if (got is None or got.dtype != want.dtype or got.shape != want.shape
                                or got.tobytes() != want.tobytes()):
                            bad.append(
                                f"tick {state.tick} seat {team} fair_fields {k}: {got} != {want}"
                            )
            seen["leaked"] |= m.leaked_elixir() > 0
        commands = []
        for team in TEAMS:
            # Blue waits for a full bar, so it leaks and own_elixir_leaked moves.
            if team == 0 and state.players[0].elixir_milli < 1000 * max_mana:
                continue
            if rng.random() >= 0.35:
                continue
            by_slot = parser.action_mask(state, team)[1:].reshape(4, -1)
            slots = np.flatnonzero(by_slot.any(axis=1))
            if len(slots):
                slot = int(rng.choice(slots))
                tile = int(rng.choice(np.flatnonzero(by_slot[slot])))
                cmd = parser.parse(1 + slot * by_slot.shape[1] + tile, state, team)
                if cmd is not None:
                    commands.append(cmd)
        seen["plays"] += sum(r.status == 0 for r in engine.step(commands, CADENCES[step % 5]))
        step += 1
        state = engine.state()
    return bad, seen


@pytest.fixture(scope="module", params=list(ENGINES.values()), ids=list(ENGINES))
def engine(request):
    return request.param()


@pytest.mark.parametrize("start_tick", [0, 3500])
def test_the_one_buffer_vector_is_the_per_field_vector_byte_for_byte(engine, start_tick):
    """Every vector and every fair field of a played battle, new path against old.

    Two battles per engine: one from the start (the opening, the 1x rate, the first plays)
    and one from late regulation into overtime (2x, the overtime clock, towers falling).
    """
    bad, seen = compare_battle(engine, seed=7 + start_tick, start_tick=start_tick, ticks=2400)
    assert not bad, f"{len(bad)} differences; first: {bad[:3]}"
    # The population, asserted: a comparison over states that never moved a field
    # would pass for nothing.
    assert seen["vectors"] >= 400, seen
    assert seen["fields"] >= 4000, seen
    assert seen["plays"] >= 10, seen
    if start_tick:
        assert seen["overtime"], seen
        assert seen["rate2"], seen
    else:
        assert seen["leaked"], seen


def test_plant_a_one_hot_written_one_slot_off_is_caught(monkeypatch):
    """The comparison above, fed a writer that puts the next card one slot along."""
    real = obs._put_one

    def shifted(v, card, empty_index):
        real(v, (card + 1) % (len(v) - 1) if card >= 0 else card, empty_index)

    monkeypatch.setattr(obs, "_put_one", shifted)
    bad, _seen = compare_battle(MockEngine(), seed=3, start_tick=0, ticks=200)
    assert bad, "a one-hot written one slot off passed the byte comparison"
