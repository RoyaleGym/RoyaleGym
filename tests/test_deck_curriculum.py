"""DeckCurriculumStateMutator: which seat gets the deck, what the other seat plays, and
whether a seed brings the same episodes back.

WHAT IT CHECKS
    Over 10 000 builds, the deck lands on each seat as often as ``p``, ``seat`` and
    ``mirror_p`` say, within 4 standard deviations of the binomial count. A fixed seat
    never deals it to the other seat. Every other deck comes from the pool, drawn
    uniformly, and each seat draws its own. ``config()`` survives JSON and rebuilds a
    mutator that deals the same episodes from the same seed, through ``from_config`` and
    through the kwargs route a config file takes. A name the catalogue lacks fails the
    first reset.

    The episode tests are graded against ``engine.state()``, the hands the engine dealt,
    on both seats: a mirror deals both seats the same hand, a one-seat episode deals the
    deck to the seat it names, one mutator deals the same cards on engines that number
    them differently, a deck with no shuffle arrives in the order written, and a seeded
    reset comes back identical.

    Every deck here is a different SET of cards, and the deck and the pool deck used
    against the engine share no card, so a deck dealt to the wrong seat cannot pass as
    the right one.

WHAT IT CANNOT CATCH
    Whether a curriculum helps a policy learn. That is a training question.
"""

from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np
import pytest

from royalegym import DeckCurriculumStateMutator
from royalegym.env import ClashParallelEnv, EnvFactory
from royalegym.mock_engine import MOCK_CARD_NAMES, MockEngine
from royalegym.protocol import BLUE, DECK_SIZE, RED, ShuffleMode
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available

# Two decks with no card in common. Both engines load all sixteen cards.
MAIN = ("HogRider", "Musketeer", "Cannon", "Skeletons", "Fireball", "Log", "Knight", "Zap")
REST = ("Archer", "Goblins", "Giant", "MiniPekka", "Minions", "Valkyrie", "Arrows", "GoblinBarrel")
# Three pool decks, each a different set, none of them MAIN.
POOL = (
    REST,
    ("Archer", "Goblins", "Giant", "MiniPekka", "Minions", "Valkyrie", "Arrows", "Knight"),
    ("Archer", "Goblins", "Giant", "MiniPekka", "Minions", "Valkyrie", "Zap", "GoblinBarrel"),
)
CARDS = MockEngine().cards()
N = 10_000
SIGMAS = 4.0

Deal = tuple[frozenset[str], frozenset[str], int]


def deal(mutator: DeckCurriculumStateMutator, n: int = N, seed: int = 0) -> list[Deal]:
    """(blue deck, red deck, shuffle) for ``n`` builds, each deck as its set of card names."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        setup = mutator.build(rng, CARDS)
        blue, red = (frozenset(CARDS[c].name for c in d) for d in setup.decks)
        assert len(blue) == len(red) == DECK_SIZE
        out.append((blue, red, setup.shuffle))
    return out


def close_to(count: int, n: int, q: float) -> bool:
    """``count`` of ``n`` is within SIGMAS binomial standard deviations of ``n * q``."""
    return abs(count - n * q) <= SIGMAS * math.sqrt(n * q * (1 - q))


def visible(engine, team: int) -> set[str]:
    """Names of the cards the engine dealt ``team``: the hand and the next card."""
    player = engine.state().players[team]
    return {engine.cards()[c].name for c in [*player.hand, player.next_card]}


# --------------------------------------------------------------------------
# where the deck lands, over many builds
# --------------------------------------------------------------------------


def test_either_seat_deals_the_deck_to_each_seat_in_p_over_2():
    p = 0.7
    dealt = deal(DeckCurriculumStateMutator(MAIN, p=p, seat="either", pool=POOL))
    main = frozenset(MAIN)
    on_blue = sum(b == main and r != main for b, r, _ in dealt)
    on_red = sum(r == main and b != main for b, r, _ in dealt)
    neither = sum(b != main and r != main for b, r, _ in dealt)
    assert on_blue + on_red + neither == N, "no mirror was asked for"
    assert close_to(on_blue, N, p / 2), f"deck on blue {on_blue} times, expected ~{N * p / 2}"
    assert close_to(on_red, N, p / 2), f"deck on red {on_red} times, expected ~{N * p / 2}"
    assert close_to(neither, N, 1 - p), f"pool against pool {neither}, expected ~{N * (1 - p)}"


@pytest.mark.parametrize(("seat", "team"), [("blue", BLUE), ("red", RED)])
def test_a_fixed_seat_never_deals_the_deck_to_the_other(seat, team):
    p = 0.6
    dealt = deal(DeckCurriculumStateMutator(MAIN, p=p, seat=seat, pool=POOL))
    main = frozenset(MAIN)
    here = sum(d[team] == main for d in dealt)
    there = sum(d[1 - team] == main for d in dealt)
    assert there == 0, f"seat={seat!r} dealt the deck to the other seat {there} times"
    assert close_to(here, N, p), f"deck on {seat} {here} times, expected ~{N * p}"


def test_every_other_deck_comes_from_the_pool_uniformly():
    dealt = deal(DeckCurriculumStateMutator(MAIN, p=0.5, pool=POOL))
    main = frozenset(MAIN)
    others = Counter(deck for b, r, _ in dealt for deck in (b, r) if deck != main)
    pool = [frozenset(d) for d in POOL]
    assert set(others) <= set(pool), f"decks from outside the pool: {set(others) - set(pool)}"
    total = sum(others.values())
    for i, deck in enumerate(pool):
        assert close_to(others[deck], total, 1 / len(pool)), (
            f"pool[{i}] drawn {others[deck]} of {total} times"
        )


def test_pool_against_pool_draws_each_seat_on_its_own():
    # p=0: every episode is pool against pool. With three pool decks drawn apart, the
    # two seats hold the same deck in a third of episodes, not in all of them.
    dealt = deal(DeckCurriculumStateMutator(MAIN, p=0.0, pool=POOL))
    same = sum(b == r for b, r, _ in dealt)
    q = 1 / len(POOL)
    assert close_to(same, N, q), f"both seats drew the same pool deck {same} times"


def test_no_pool_deals_random_decks_of_eight_different_cards():
    dealt = deal(DeckCurriculumStateMutator(MAIN, p=0.5, pool=None), n=2_000)
    main = frozenset(MAIN)
    others = [deck for b, r, _ in dealt for deck in (b, r) if deck != main]
    names = {c.name for c in CARDS}
    assert all(len(deck) == DECK_SIZE and deck <= names for deck in others)
    # C(16, 8) = 12 870 decks, so 3 000 draws that repeat often are not random.
    assert len(set(others)) > len(others) // 2, "random decks repeat far too often"


def test_the_mirror_fraction_is_p_times_mirror_p():
    p, mirror_p = 0.8, 0.25
    dealt = deal(DeckCurriculumStateMutator(MAIN, p=p, mirror_p=mirror_p, pool=POOL))
    main = frozenset(MAIN)
    mirrors = [s for b, r, s in dealt if b == main and r == main]
    rest = [s for b, r, s in dealt if not (b == main and r == main)]
    on_blue = sum(b == main and r != main for b, r, _ in dealt)
    assert close_to(len(mirrors), N, p * mirror_p), f"{len(mirrors)} mirrors"
    assert close_to(on_blue, N, p * (1 - mirror_p) / 2), f"deck on blue alone {on_blue} times"
    assert set(mirrors) == {ShuffleMode.MIRRORED}, "a mirror must shuffle both decks alike"
    assert set(rest) == {ShuffleMode.INDEPENDENT}


# --------------------------------------------------------------------------
# config and names
# --------------------------------------------------------------------------


def setups(mutator: DeckCurriculumStateMutator, n: int = 2_000, seed: int = 11) -> list:
    rng = np.random.default_rng(seed)
    return [mutator.build(rng, CARDS) for _ in range(n)]


def test_config_round_trips_through_json_and_deals_the_same_episodes():
    # Every argument away from its default, so a field config() drops changes the deals.
    m = DeckCurriculumStateMutator(
        MAIN, p=0.55, seat="red", mirror_p=0.3, pool=POOL, shuffle=ShuffleMode.NONE
    )
    cfg = json.loads(json.dumps(m.config()))
    rebuilt = DeckCurriculumStateMutator.from_config(cfg)
    assert rebuilt.config() == m.config()
    assert setups(rebuilt) == setups(m)

    # env.config() keeps the same dict, and a config file's route is cls(**params).
    env = ClashParallelEnv(MockEngine(), state_mutator=m)
    record = json.loads(json.dumps(env.config()["state_mutator"]))
    assert record["class"] == "royalegym.state_mutator.DeckCurriculumStateMutator"
    assert setups(DeckCurriculumStateMutator.from_config(record)) == setups(m)
    factory = EnvFactory(
        engine=MockEngine, state_mutator=(DeckCurriculumStateMutator, record["params"])
    )
    assert factory().config()["state_mutator"] == record

    # A record for another class is refused, even when its params would fit.
    wrong = {**record, "class": "royalegym.state_mutator.WeightedStateMutator"}
    with pytest.raises(ValueError, match="config is for"):
        DeckCurriculumStateMutator.from_config(wrong)


def test_a_name_the_catalogue_lacks_fails_the_first_reset():
    typo = ["HogRidr", *MAIN[1:]]
    with pytest.raises(ValueError, match=r"'HogRidr'.*close: HogRider"):
        DeckCurriculumStateMutator(typo).build(np.random.default_rng(0), CARDS)

    # A card this engine does not load, in a pool deck this curriculum never draws:
    # p=1 and mirror_p=1 deal only MAIN, and the reset still refuses.
    engine = MockEngine(card_names=[n for n in MOCK_CARD_NAMES if n != "Log"])
    m = DeckCurriculumStateMutator(REST, p=1.0, mirror_p=1.0, pool=[REST, MAIN])
    env = ClashParallelEnv(engine, state_mutator=m)
    with pytest.raises(ValueError, match=r"pool\[1\] names 'Log'"):
        env.reset(seed=0)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"deck": MAIN[:7]}, "must have 8 cards"),
        ({"deck": [*MAIN[:7], "HogRider"]}, "lists a card twice"),
        ({"deck": "HogRider"}, "list of card names"),
        ({"deck": MAIN, "p": 1.5}, "p must be a probability"),
        ({"deck": MAIN, "mirror_p": -0.1}, "mirror_p must be a probability"),
        ({"deck": MAIN, "seat": "left"}, "seat must be one of"),
        ({"deck": MAIN, "pool": []}, "pool must be a list of decks"),
        ({"deck": MAIN, "pool": [[*REST, "Knight"]]}, r"pool\[0\] must have 8 cards"),
        ({"deck": MAIN, "shuffle": 7}, "7 is not a valid ShuffleMode"),
    ],
)
def test_bad_arguments_are_refused_at_construction(kwargs, error):
    with pytest.raises((ValueError, TypeError), match=error):
        DeckCurriculumStateMutator(**kwargs)


def test_a_deck_given_as_an_iterator_is_read_once():
    m = DeckCurriculumStateMutator(iter(MAIN), pool=[iter(REST)])
    assert m.config()["deck"] == list(MAIN)
    assert m.config()["pool"] == [list(REST)]


# --------------------------------------------------------------------------
# episodes, graded against the hands the engine dealt
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_a_mirror_deals_both_seats_the_same_hand(seed):
    engine = MockEngine()
    m = DeckCurriculumStateMutator(MAIN, mirror_p=1.0, pool=[REST])
    ClashParallelEnv(engine, state_mutator=m).reset(seed=seed)
    blue, red = engine.state().players
    assert blue.hand == red.hand, "a mirror dealt the two seats different hands"
    assert blue.next_card == red.next_card
    assert visible(engine, BLUE) <= set(MAIN)


@pytest.mark.parametrize(("seat", "team"), [("blue", BLUE), ("red", RED)])
def test_the_deck_is_dealt_to_the_seat_it_names(seat, team):
    engine = MockEngine()
    m = DeckCurriculumStateMutator(MAIN, seat=seat, pool=[REST])
    env = ClashParallelEnv(engine, state_mutator=m)
    for seed in range(3):
        env.reset(seed=seed)
        assert visible(engine, team) <= set(MAIN), f"{seat} was not dealt the deck"
        assert visible(engine, 1 - team) <= set(REST), "the other seat was not dealt the pool"


def test_one_mutator_deals_the_same_cards_on_engines_that_number_them_differently():
    # The same card names in the opposite order: every card id means another card.
    m = DeckCurriculumStateMutator(MAIN, seat="red", pool=[REST])
    for engine in (MockEngine(), MockEngine(card_names=MOCK_CARD_NAMES[::-1]), MockEngine()):
        ClashParallelEnv(engine, state_mutator=m).reset(seed=0)
        assert visible(engine, RED) <= set(MAIN), "red was dealt ids meant for another engine"
        assert visible(engine, BLUE) <= set(REST), "blue was dealt ids meant for another engine"


def test_set_curriculum_changes_the_next_episode():
    engine = MockEngine()
    m = DeckCurriculumStateMutator(MAIN, mirror_p=1.0, pool=[REST])
    env = ClashParallelEnv(engine, state_mutator=m)
    env.reset(seed=3)
    assert visible(engine, BLUE) | visible(engine, RED) <= set(MAIN)

    # A new deck and pool on the same env: the next reset deals them, not the old ones.
    m.set_curriculum(deck=list(REST), pool=[list(MAIN)], mirror_p=0.0, seat="red")
    env.reset(seed=3)
    assert visible(engine, RED) <= set(REST)
    assert visible(engine, BLUE) <= set(MAIN)

    # A refused change leaves every field as it was.
    before = m.config()
    with pytest.raises(ValueError, match="seat"):
        m.set_curriculum(p=0.5, seat="left")
    with pytest.raises(TypeError, match="prob"):
        m.set_curriculum(prob=0.5)
    assert m.config() == before


def rust_engine() -> RustEngine:
    return RustEngine()


ENGINES = [
    pytest.param(MockEngine, id="mock"),
    pytest.param(
        rust_engine,
        id="rust",
        marks=pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR)),
    ),
]


@pytest.mark.parametrize("make_engine", ENGINES)
def test_a_seeded_reset_is_reproducible(make_engine):
    m = DeckCurriculumStateMutator(MAIN, p=0.6, mirror_p=0.3, pool=POOL)

    def states(env: ClashParallelEnv, seeds) -> list:
        out = []
        for seed in seeds:
            env.reset(seed=seed)
            out.append(env.engine.state())
        return out

    env = ClashParallelEnv(make_engine(), state_mutator=m)
    first = states(env, range(8))
    assert states(env, reversed(range(8))) == first[::-1], "same env, same seeds"
    other = ClashParallelEnv(make_engine(), state_mutator=m.from_config(m.config()))
    assert states(other, range(8)) == first, "a rebuilt mutator on a fresh env"

    # Not vacuous: the seeds dealt the deck to different seats, so the draw is seeded.
    names = [c.name for c in env.engine.cards()]
    where = {
        tuple({names[c] for c in [*p.hand, p.next_card]} <= set(MAIN) for p in s.players)
        for s in first
    }
    assert len(where) > 1, f"all eight seeds dealt the deck the same way: {where}"


@pytest.mark.parametrize("make_engine", ENGINES)
@pytest.mark.parametrize(("seat", "team"), [("blue", BLUE), ("red", RED)])
def test_without_a_shuffle_each_seat_draws_its_deck_in_the_order_written(make_engine, seat, team):
    # ShuffleMode.NONE plays a deck in the order given: the hand is its first four
    # cards and the next card is its fifth. That holds only if the names keep their order.
    engine = make_engine()
    m = DeckCurriculumStateMutator(MAIN, seat=seat, pool=[REST], shuffle=ShuffleMode.NONE)
    ClashParallelEnv(engine, state_mutator=m).reset(seed=0)
    names = [c.name for c in engine.cards()]
    for t, deck in ((team, MAIN), (1 - team, REST)):
        player = engine.state().players[t]
        assert [names[c] for c in player.hand] == list(deck[:4]), f"team {t} hand"
        assert names[player.next_card] == deck[4], f"team {t} next card"
