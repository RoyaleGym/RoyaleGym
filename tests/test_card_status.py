"""Every card's evolution and hero status, own and enemy: ``SpatialObsBuilder(card_status=True)``.

Owner, 2026-10-03: "the bot should also know all evolution and hero status's of the cards in its
and the enemies deck and hand". So, per card id (order-free, like ``own_deck``) and per hand slot:

OWN, which a player knows from the first frame:
- ``own_deck_all``: the whole deck (``PlayerState.deck``), not the cycle's deduction;
- ``own_deck_evo`` / ``own_deck_hero``, ``own_hand_evo`` / ``own_hand_hero``, ``own_next_form``:
  each card's form;
- ``own_evo_progress`` / ``own_evo_next``: every evolved deck card's charge, wherever it sits;
- ``own_button*``: each ability button by its card: there, available, spent, cooldown.

ENEMY, deduced as a good player does from what it sees played:
- ``enemy_seen_evolved`` / ``enemy_seen_hero``: the card has been played in that form;
- ``enemy_evo_progress`` / ``enemy_evo_next``: basic plays since its last evolved play, over the
  card's ``evo_cycle``, for every card that has an evolution, from its first play;
- ``enemy_button_spent`` / ``enemy_button_cooldown``: a hero's charge used, a champion's cooldown.
  Never whether the enemy HAS a button: that would name a card before it is played.

Appended after every other fair field; off by default.

SKIPS
    The engine test skips only without the engine.
"""

from __future__ import annotations

import msgspec
import numpy as np
import pytest

from royalegym.action import TileActionParser
from royalegym.mock_engine import MockEngine
from royalegym.obs import (
    COOLDOWN_SCALE,
    Reveal,
    SpatialObsBuilder,
    vector_offsets,
)
from royalegym.protocol import (
    BLUE,
    RED,
    STATUS_EVOLVED,
    STATUS_HERO,
    EntityKind,
    EntityState,
    MatchSetup,
)
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, _core, core_available

FIELDS = [
    "own_deck_all", "own_deck_evo", "own_deck_hero", "own_hand_evo", "own_hand_hero",
    "own_next_form", "own_evo_progress", "own_evo_next", "own_button", "own_button_available",
    "own_button_spent", "own_button_cooldown", "enemy_seen_evolved", "enemy_seen_hero",
    "enemy_evo_progress", "enemy_evo_next", "enemy_button_spent", "enemy_button_cooldown",
]


# --- layout -------------------------------------------------------------------------------


def test_off_by_default_and_after_every_other_fair_field():
    assert not set(FIELDS) & set(vector_offsets(100))
    for rev in (None, Reveal(enemy_hand=True, enemy_deck=True)):
        for flags in ((False, False, False), (True, True, True)):
            before = vector_offsets(100, rev, *flags)
            after = vector_offsets(100, rev, *flags, card_status=True)
            fair_end = max(s.stop for k, s in before.items() if not k.startswith("enemy_hand")
                           and k not in ("enemy_next_card", "enemy_deck"))
            assert [k for k in after if k in FIELDS] == FIELDS
            assert after[FIELDS[0]].start == fair_end, "the fair block stays contiguous"
            for key, s in before.items():
                if key in ("enemy_hand_cards", "enemy_next_card", "enemy_deck"):
                    continue
                assert after[key] == s, f"{key} moved"


def test_config_records_the_flag():
    assert SpatialObsBuilder(card_status=True).config()["card_status"] is True
    assert "card_status" not in SpatialObsBuilder().config()


# --- on forged states -----------------------------------------------------------------------


class _Cycles(MockEngine):
    """MockEngine whose catalogue states an evo_cycle for every card (2 for card 3, else 0),
    as an engine with the column does."""

    def cards(self):
        return [msgspec.structs.replace(c, evo_cycle=2 if c.card_id == 3 else 0)
                for c in super().cards()]


DECK = [3, 5, 7, 9, 11, 12, 13, 14]


def _builder(engine=None):
    eng = engine or _Cycles()
    eng.reset(1, MatchSetup(decks=[DECK, DECK]))
    parser = TileActionParser(ability_buttons=True)
    parser.bind(eng)
    b = SpatialObsBuilder(card_status=True)
    b.bind(eng, parser)
    state = eng.state()
    b.reset(_reporting(state))
    return eng, parser, b, state


def _reporting(state, **players):
    """``state`` with every entity's status reported (0) and players replaced where given."""
    ents = [msgspec.structs.replace(e, status_flags=max(e.status_flags, 0)) for e in state.entities]
    ps = list(state.players)
    for team, fields in players.items():
        ps[int(team[1:])] = msgspec.structs.replace(ps[int(team[1:])], **fields)
    return msgspec.structs.replace(state, entities=ents, players=ps)


def _vec(b, parser, state, team=BLUE):
    return b.build(state, team, parser.action_mask(state, team))["vector"]


def test_the_own_deck_is_whole_from_the_first_frame():
    _, parser, b, state = _builder()
    off = b.vector_offsets()
    v = _vec(b, parser, _reporting(state))
    assert sorted(np.flatnonzero(v[off["own_deck_all"]])) == sorted(DECK)
    assert v[off["own_deck"]].sum() < len(DECK), "the deduced deck is still partial"


def test_forms_are_written_per_card_and_per_slot():
    _, parser, b, state = _builder()
    off = b.vector_offsets()
    me = state.players[BLUE]
    forms = [1 if c == me.hand[0] else 2 if c == me.hand[1] else 0 for c in me.deck]
    nxt = me.next_card
    forms = [1 if c == nxt else f for c, f in zip(me.deck, forms, strict=True)]
    v = _vec(b, parser, _reporting(state, p0={"forms": forms}))
    assert np.flatnonzero(v[off["own_deck_evo"]]).tolist() == sorted([me.hand[0], nxt])
    assert np.flatnonzero(v[off["own_deck_hero"]]).tolist() == [me.hand[1]]
    assert v[off["own_hand_evo"]].tolist() == [1, 0, 0, 0]
    assert v[off["own_hand_hero"]].tolist() == [0, 1, 0, 0]
    assert v[off["own_next_form"]].tolist() == [1, 0]


def test_every_evolved_deck_cards_charge_wherever_it_sits():
    _, parser, b, state = _builder()
    off = b.vector_offsets()
    me = state.players[BLUE]
    cycling = next(c for c in me.deck if c not in me.hand and c != me.next_card)
    rows = [[cycling, 1, 0, 2], [me.hand[2], 2, 1, 2]]
    v = _vec(b, parser, _reporting(state, p0={"evo": rows}))
    assert v[off["own_evo_progress"]][cycling] == pytest.approx(0.5)
    assert v[off["own_evo_progress"]][me.hand[2]] == pytest.approx(1.0)
    assert np.flatnonzero(v[off["own_evo_next"]]).tolist() == [me.hand[2]]


def test_own_buttons_by_card_and_the_enemys_without_naming_its_cards():
    _, parser, b, state = _builder()
    off = b.vector_offsets()
    own = [[1, 0, 2, 5, 0], [0, 1, 3, 7, 0], [0, 0, 1, 9, 120]]
    foe = [[1, 0, 2, 11, 0], [0, 1, 2, 12, 0], [0, 0, 1, 13, 300]]
    v = _vec(b, parser, _reporting(state, p0={"abilities": own}, p1={"abilities": foe}))
    assert np.flatnonzero(v[off["own_button"]]).tolist() == [5, 7, 9]
    assert np.flatnonzero(v[off["own_button_available"]]).tolist() == [5]
    assert np.flatnonzero(v[off["own_button_spent"]]).tolist() == [7]
    assert v[off["own_button_cooldown"]][9] == pytest.approx(120 / COOLDOWN_SCALE)
    assert np.flatnonzero(v[off["enemy_button_spent"]]).tolist() == [12]
    assert np.flatnonzero(v[off["enemy_button_cooldown"]]).tolist() == [13]
    assert v[off["enemy_button_cooldown"]][13] == pytest.approx(300 / COOLDOWN_SCALE)
    ready = _vec(b, parser, _reporting(state, p0={"abilities": own},
                                       p1={"abilities": [[1, 0, 2, 11, 0]]}))
    for key in ("enemy_button_spent", "enemy_button_cooldown"):
        assert ready[off[key]].sum() == 0, f"{key}: an unused enemy button is invisible"


def _enemy_unit(uid, card, status, x=9000, y=27000):
    return EntityState(uid, RED, int(EntityKind.TROOP), card, -1, x, y, 100, 100, 500, False,
                       0, status_flags=status)


def test_enemy_plays_are_counted_by_their_form():
    """Card 3 (evo_cycle 2) played basic, basic, evolved, basic by Red: Blue's count goes
    1, 2 (next is evolved), 0 (seen evolved), 1. A hero play marks its card."""
    _, parser, b, state = _builder()
    off = b.vector_offsets()
    foe = state.players[RED]
    hand = list(foe.hand)
    slot = hand.index(3) if 3 in hand else 0
    hand[slot] = 3
    tick, uid = state.tick, 900
    s = _reporting(state, p1={"hand": list(hand)})
    b.reset(s)
    seen = []
    for status in (0, 0, STATUS_EVOLVED, 0):
        tick += 5
        hand[slot] = 14  # the card leaves the hand: a play
        s = msgspec.structs.replace(
            _reporting(state, p1={"hand": list(hand)}), tick=tick,
            entities=[*s.entities, _enemy_unit(uid, 3, status)],
        )
        v = _vec(b, parser, s)
        seen.append((v[off["enemy_evo_progress"]][3], v[off["enemy_evo_next"]][3],
                     v[off["enemy_seen_evolved"]][3]))
        uid += 1
        tick += 5
        hand[slot] = 3  # it cycles back
        s = msgspec.structs.replace(_reporting(state, p1={"hand": list(hand)}), tick=tick,
                                    entities=s.entities)
        _vec(b, parser, s)
    assert seen == [(0.5, 0, 0), (1.0, 1, 0), (0.0, 0, 1), (0.5, 0, 1)], seen
    assert _vec(b, parser, s)[off["enemy_seen_hero"]].sum() == 0
    hand[slot] = 14
    s = msgspec.structs.replace(_reporting(state, p1={"hand": list(hand)}), tick=tick + 5,
                                entities=[*s.entities, _enemy_unit(uid, 3, STATUS_HERO)])
    assert _vec(b, parser, s)[off["enemy_seen_hero"]][3] == 1


def test_an_engine_without_the_deck_or_the_cycles_is_refused():
    with pytest.raises(ValueError, match="evo_cycle"):
        _bind_plain()
    _, parser, b, state = _builder()
    with pytest.raises(ValueError, match="does not report the deck"):
        _vec(b, parser, _reporting(state, p0={"deck": [], "forms": []}))


def _bind_plain():
    eng = MockEngine()  # an engine whose catalogue does not state evo_cycle
    eng.cards = lambda: [msgspec.structs.replace(c, evo_cycle=None)
                         for c in MockEngine.cards(eng)]
    parser = TileActionParser()
    parser.bind(eng)
    SpatialObsBuilder(card_status=True).bind(eng, parser)


# --- on the engine: deduced equals the engine's own counters ---------------------------------


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_the_enemy_count_equals_the_enemys_own_counters():
    """Blue's deduction of Red's evolution counters, from Red's plays alone, against Red's own
    counter rows (the engine's), at every step of played battles with evolved cards."""
    from royalegym import ClashParallelEnv
    from royalegym.state_mutator import DefaultStateMutator

    has_column = "evo_cycle" in getattr(_core, "CATALOGUE_FIELDS", ())
    names = ("Skeletons", "Barbarians", "Knight", "Archer", "Giant", "Minions", "Fireball", "Zap")
    probe = RustEngine()
    ids = {c.name: c.card_id for c in probe.cards()}
    deck = [ids[n] for n in names]
    forms = [1, 1, 0, 0, 0, 0, 0, 0]
    probe.reset(1, MatchSetup(decks=[deck, deck], forms=[forms, forms]))
    cycles = {int(r[0]): int(r[3]) for r in probe.state().players[RED].evo}
    assert len(cycles) == 2, cycles

    class Cycled(RustEngine):
        def cards(self):
            got = super().cards()
            if has_column:
                return got
            return [msgspec.structs.replace(c, evo_cycle=cycles.get(c.card_id, 0)) for c in got]

    env = ClashParallelEnv(
        Cycled(), obs_builder=SpatialObsBuilder(card_status=True),
        state_mutator=DefaultStateMutator(decks=[deck, deck], forms=[forms, forms]),
    )
    off = env.obs_builder.vector_offsets()
    rng = np.random.default_rng(4)
    checked, evolved_plays = 0, 0
    for _ in range(3):
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        while env.agents:
            v = obs["blue"]["vector"]
            for card, plays, nxt, cycle in (r[:4] for r in env.battle_state.players[RED].evo):
                assert v[off["enemy_evo_progress"]][card] == pytest.approx(min(1, plays / cycle))
                assert v[off["enemy_evo_next"]][card] == nxt, (env.battle_state.tick, card)
                checked += 1
            evolved_plays = int(v[off["enemy_seen_evolved"]].sum())
            acts = {}
            for a in env.agents:
                legal = np.flatnonzero(obs[a]["action_mask"])[1:]
                acts[a] = int(rng.choice(legal)) if legal.size and rng.random() < 0.5 else 0
            obs, *_ = env.step(acts)
    assert checked > 500, checked
    assert evolved_plays >= 1, "no evolved enemy play was seen, so the reset was never tested"


def test_units_on_the_board_at_the_start_are_not_plays():
    """A battle that starts mid-match (a mutator's spawns, a loaded state) can already hold an
    evolved enemy unit. A basic play of that card on the first step is still basic."""
    _, parser, b, state = _builder()
    off = b.vector_offsets()
    hand = list(state.players[RED].hand)
    slot = hand.index(3) if 3 in hand else 0
    hand[slot] = 3
    start = msgspec.structs.replace(
        _reporting(state, p1={"hand": list(hand)}),
        entities=[*_reporting(state).entities, _enemy_unit(800, 3, STATUS_EVOLVED)],
    )
    b.reset(start)
    hand[slot] = 14
    s = msgspec.structs.replace(
        _reporting(state, p1={"hand": list(hand)}), tick=state.tick + 5,
        entities=[*start.entities, _enemy_unit(801, 3, 0)],
    )
    v = _vec(b, parser, s)
    assert v[off["enemy_evo_progress"]][3] == pytest.approx(0.5), "the play counted as evolved"
    assert v[off["enemy_seen_evolved"]][3] == 0


def test_a_unit_of_an_earlier_evolved_play_does_not_make_a_basic_play_evolved():
    """Once card 3 (evo_cycle 2) has shown its evolution, the form of each later play follows
    from the count: the next play is evolved exactly when two basic plays have passed since
    the last evolved one (the engine's own rule, PlayerState.evo, checked on 4340 rows). A
    unit of the EARLIER evolved play that appears on a basic play's step -- an Evo Wall
    Breaker's mini, an Evo Royal Ghost's summon, which royalesim 0.1.20 marks evolved -- must
    not make that basic play read as evolved and reset the count."""
    _, parser, b, state = _builder()
    off = b.vector_offsets()
    foe = state.players[RED]
    hand = list(foe.hand)
    slot = hand.index(3) if 3 in hand else 0
    hand[slot] = 3
    tick, uid = state.tick, 900
    s = _reporting(state, p1={"hand": list(hand)})
    b.reset(s)

    def play(*statuses):
        nonlocal s, tick, uid
        tick += 5
        hand[slot] = 14
        units = []
        for status in statuses:
            units.append(_enemy_unit(uid, 3, status))
            uid += 1
        s = msgspec.structs.replace(
            _reporting(state, p1={"hand": list(hand)}), tick=tick, entities=[*s.entities, *units]
        )
        v = _vec(b, parser, s)
        tick += 5
        hand[slot] = 3
        s = msgspec.structs.replace(_reporting(state, p1={"hand": list(hand)}), tick=tick,
                                    entities=s.entities)
        _vec(b, parser, s)
        return (v[off["enemy_evo_progress"]][3], v[off["enemy_evo_next"]][3],
                v[off["enemy_seen_evolved"]][3])

    assert [play(0), play(0), play(STATUS_EVOLVED)] == [(0.5, 0, 0), (1.0, 1, 0), (0.0, 0, 1)]
    # A basic play whose step also shows a unit of the evolved play before it (status 8).
    assert play(0, STATUS_EVOLVED) == (0.5, 0, 1), "the summon made a basic play evolved"
    assert play(0) == (1.0, 1, 1)
    assert play(STATUS_EVOLVED) == (0.0, 0, 1), "the count's evolved play"
