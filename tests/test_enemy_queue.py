"""The enemy's cycle order, as a player who watched its plays deduces it (opt-in).

A played card goes to the back of an 8-card cycle, so the cards behind the enemy's hand are its
last plays, oldest first: after four plays a player knows the enemy's next card is the fourth
most recent, and the three after it. ``enemy_possible_hand`` already rules those four out of
the hand, as a set; ``SpatialObsBuilder(enemy_queue=True)`` gives their ORDER as
``enemy_queue_5_8``, four one-hots over [n+1], position 5 (the next card) first, index n where
no play has reached the position yet. While played slots wait for their refill (one per
refill period, a rule of the clock), the cards behind the hand reach that much further back.
Nothing is deduced once the memory is not exact. Off by default, appended at the end.

SKIPS
    The RustEngine run skips without the engine. Not a pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from royalegym.action import TileActionParser
from royalegym.env import ClashParallelEnv
from royalegym.mock_engine import MockEngine
from royalegym.obs import MatchMemory, SpatialObsBuilder
from royalegym.protocol import BLUE, EMPTY_CARD, RED, default_elixir_law
from royalegym.rust_engine import CORE_IMPORT_ERROR, core_available
from royalegym.state_mutator import DefaultStateMutator, random_deck

needs_engine = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
E = EMPTY_CARD
#: Cards whose units pay elixir outside the regeneration law: the Elixir Collector (its pump,
#: and one elixir to its owner when it dies) and the Elixir Golem (one elixir to the OPPONENT
#: when it dies, half for each golemite and blob). MatchMemory does not count those payments yet
#: (test_the_memory_counts_an_elixir_golem_death), and a deck with one voids the queue check, so
#: its random decks are drawn without them.
PAYS_ELIXIR = frozenset({"Elixir Collector", "ElixirGolem"})


def _decks_that_pay_no_elixir(engine, seed: int) -> list[list[int]]:
    """Two random decks (``random_deck``) from the catalogue without ``PAYS_ELIXIR``."""
    pool = [c for c in engine.cards() if c.name not in PAYS_ELIXIR]
    rng = np.random.default_rng(seed)
    return [[pool[i].card_id for i in random_deck(rng, pool)] for _ in (BLUE, RED)]


def _memory(enemy_hand=(20, 21, 22, 23)):
    m = MatchMemory(30, default_elixir_law())
    m.cost = [1] * 30
    m.start(0, 5000, 10000, [0, 1, 2, 3], 4, list(enemy_hand) if enemy_hand else None)
    return m


def _play(m, *cards):
    for card in cards:
        m.advance(m.tick + 10, 3600, False, [], [(m.tick, card)])


def test_the_queue_fills_from_the_back_one_play_at_a_time():
    m = _memory()
    assert m.enemy_queue() == [E, E, E, E]
    _play(m, 20)
    assert m.enemy_queue() == [E, E, E, 20]
    _play(m, 21, 22)
    assert m.enemy_queue() == [E, 20, 21, 22]
    _play(m, 23)
    assert m.enemy_queue() == [20, 21, 22, 23], "after four plays the next card is the first"
    _play(m, 24, 25)
    assert m.enemy_queue() == [22, 23, 24, 25]


def test_a_slot_waiting_for_its_refill_reaches_one_play_further_back():
    m = _memory()
    _play(m, 20, 21, 22, 23, 24)
    m.foe_hand = [25, E, 26, 27]  # 24's slot not refilled yet: 20 is still on its way in
    assert m.enemy_queue() == [20, 21, 22, 23]
    m.foe_hand = [25, 20, 26, 27]
    assert m.enemy_queue() == [21, 22, 23, 24]
    m.foe_hand = [E, E, 26, 27]
    assert m.enemy_queue() == [E, 20, 21, 22], "a sixth card behind the hand is not known"


def test_nothing_is_deduced_once_the_memory_is_not_exact():
    m = _memory()
    _play(m, 20, 21, 22, 23)
    m.exact = False
    assert m.enemy_queue() == [E, E, E, E]


def test_a_caller_that_never_showed_the_enemy_hand_counts_no_refill():
    m = _memory(enemy_hand=None)
    assert m.foe_hand == []
    _play(m, 20, 21, 22, 23)
    assert m.enemy_queue() == [20, 21, 22, 23]


def test_off_by_default_and_appended_after_every_other_field():
    eng = MockEngine()
    plain = ClashParallelEnv(eng, obs_builder=SpatialObsBuilder())
    assert "enemy_queue_5_8" not in plain.obs_builder.vector_offsets()
    assert "enemy_queue" not in plain.obs_builder.config()
    env = ClashParallelEnv(MockEngine(), obs_builder=SpatialObsBuilder(enemy_queue=True))
    off, plain_off = env.obs_builder.vector_offsets(), plain.obs_builder.vector_offsets()
    assert all(off[k] == v for k, v in plain_off.items()), "no existing field moved"
    n = len(eng.cards())
    queue = off["enemy_queue_5_8"]
    assert queue.stop - queue.start == 4 * (n + 1)
    assert queue.start == max(s.stop for s in plain_off.values())
    assert env.obs_builder.config()["enemy_queue"] is True


def _check_against_the_engine(engine, steps: int, seed: int) -> dict[str, int]:
    """Blue's enemy_queue_5_8 at each step, checked against the cards that then ARRIVE in Red's
    hand, in order: a deduced position k must be the k-th card to arrive after that step."""
    decks = _decks_that_pay_no_elixir(engine, seed)
    env = ClashParallelEnv(engine, action_parser=TileActionParser(),
                           obs_builder=SpatialObsBuilder(enemy_queue=True), decision_ms=50,
                           state_mutator=DefaultStateMutator(decks=decks))
    obs, _ = env.reset(seed=seed)
    n = len(engine.cards())
    dealt = [engine.cards()[c].name for p in env.battle_state.players for c in p.deck]
    assert len(dealt) == 16, dealt
    assert not PAYS_ELIXIR & set(dealt), dealt
    sl = env.obs_builder.vector_offsets()["enemy_queue_5_8"]
    rng = np.random.default_rng(seed)
    records: list[tuple[int, list[int], bool]] = []
    arrivals: list[int] = []
    hand = list(env.battle_state.players[RED].hand)
    for _ in range(steps):
        if not env.agents:
            break
        rows = obs["blue"]["vector"][sl].reshape(4, n + 1)
        queue = [int(r.argmax()) if r[:n].any() else E for r in rows]
        waiting = EMPTY_CARD in env.battle_state.players[RED].hand
        records.append((len(arrivals), queue, waiting))
        acts = {}
        for agent in env.agents:
            legal = np.flatnonzero(env.action_masks(agent))
            legal = legal[legal != 0]
            acts[agent] = int(rng.choice(legal)) if len(legal) and rng.random() < 0.3 else 0
        obs, *_ = env.step(acts)
        now = list(env.battle_state.players[RED].hand)
        arrivals += [c for c in now if c != EMPTY_CARD and c not in hand]
        hand = now
    assert env.obs_builder.memory[BLUE].exact, "the count went inexact: the check is void"
    seen = {"positions": 0, "during_refill": 0, "full_queue": 0}
    for at, queue, waiting in records:
        for k, card in enumerate(queue):
            if card == E or at + k >= len(arrivals):
                continue
            assert arrivals[at + k] == card, (at, k, queue, arrivals[at : at + 4])
            seen["positions"] += 1
            seen["during_refill"] += waiting
        seen["full_queue"] += E not in queue
    return seen


def test_on_mockengine_every_deduced_position_is_the_card_that_arrives():
    """MockEngine refills a played slot at once, so it never shows a refill window; the
    RustEngine run below checks those."""
    seen = _check_against_the_engine(MockEngine(), 1200, seed=3)
    assert seen["positions"] > 200, seen
    assert seen["during_refill"] == 0, seen
    assert seen["full_queue"] > 50, seen


@needs_engine
def test_on_rustengine_every_deduced_position_is_the_card_that_arrives():
    from royalegym.rust_engine import RustEngine

    seen = _check_against_the_engine(RustEngine(), 1200, seed=5)
    assert seen["positions"] > 200, seen
    assert seen["during_refill"] > 10, f"no refill window was checked: {seen}"
    assert seen["full_queue"] > 50, seen


class _WentInexact(AssertionError):
    """The memory's count stopped matching the engine's at an Elixir Golem piece's death."""


@needs_engine
@pytest.mark.xfail(
    strict=True,
    raises=_WentInexact,
    reason="A KNOWN GAP: MatchMemory does not count elixir a unit pays (the Elixir Golem's to "
    "the opponent when it dies, the Elixir Collector's pump and death payment), because the "
    "engine does not say what each unit pays. When the memory counts it, this passes and the "
    "mark must go.",
)
def test_the_memory_counts_an_elixir_golem_death():
    """Blue plays an Elixir Golem; when a piece of it dies, Red gains elixir a player can count
    (the card's rule is public). Blue's memory of Red's bar must stay exact through it."""
    from royalegym.rust_engine import RustEngine

    engine = RustEngine()
    names = [c.name for c in engine.cards()]
    golem = names.index("ElixirGolem")
    blue = ["ElixirGolem", "Knight", "Archer", "Goblins", "Giant", "Musketeer", "Arrows", "Zap"]
    red = ["Knight", "Archer", "Goblins", "Valkyrie", "Musketeer", "MiniPekka", "Arrows", "Zap"]
    env = ClashParallelEnv(engine, action_parser=TileActionParser(),
                           obs_builder=SpatialObsBuilder(), decision_ms=50,
                           state_mutator=DefaultStateMutator(decks=[blue, red]))
    env.reset(seed=4)
    memory = env.obs_builder.memory[BLUE]
    rng = np.random.default_rng(4)
    alive: set[int] = set()
    deaths = 0
    for _ in range(3000):
        if not env.agents:
            break
        acts = {}
        for agent in env.agents:
            legal = np.flatnonzero(env.action_masks(agent))
            legal = legal[legal != 0]
            acts[agent] = int(rng.choice(legal)) if len(legal) and rng.random() < 0.3 else 0
        env.step(acts)
        pieces = {e.uid for e in env.battle_state.entities
                  if e.team == BLUE and e.card_id == golem}
        died = len(alive - pieces)
        deaths += died
        alive = pieces
        if not memory.exact:
            # Vacuity: the count must break where a piece died, not for some other reason.
            assert died, "the memory went inexact on a step no golem piece died"
            raise _WentInexact(f"after {deaths} golem pieces died")
    assert deaths, "no Elixir Golem piece died, so nothing was checked"
