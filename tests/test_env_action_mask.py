"""Action masking: the mask must agree with the engine EXACTLY, in both directions.

A mask that allows an illegal move trains an agent to want it; a mask that forbids
a legal move hides part of the game. Both are silent in training, so they are
checked here exhaustively against ``Engine.check_deploy``.
"""

from __future__ import annotations

import msgspec
import numpy as np
import pytest

from _decks import MIXED_BARREL_DECK, MIXED_DECK, SPELL_FIRST_DECK, card_id, card_ids
from royalegym.action import (
    NOOP,
    HalfTileActionParser,
    PlacementOracle,
    TileActionParser,
    mask_disagreements,
)
from royalegym.done_condition import GameOverCondition, StepLimitCondition
from royalegym.env import ClashParallelEnv
from royalegym.mock_engine import MockEngine
from royalegym.protocol import (
    BLUE,
    RED,
    DeployStatus,
    MatchSetup,
    Placement,
    ShuffleMode,
    SpawnSpec,
)
from royalegym.rust_engine import RustEngine, core_available
from royalegym.selfplay import RandomLegalOpponent
from royalegym.state_mutator import DefaultStateMutator

# Knight Giant Cannon Log | Fireball Zap Minions Valkyrie -- all four placement types.
# Written as NAMES in tests/_decks.py and turned into ids here, because an id is a
# position in a catalogue and positions move when a catalogue gains a card.
_CARDS = MockEngine().cards()
CANNON = card_id("Cannon", _CARDS)  # the deck's BUILDING
KNIGHT = card_id("Knight", _CARDS)  # an ordinary ground troop
BARREL = card_id("GoblinBarrel", _CARDS)  # the SPELL_NOT_ON_WATER card
MIXED = card_ids(MIXED_DECK, _CARDS)
# Zap swapped for the Goblin Barrel (2026-09-13): every Placement class,
# SPELL_NOT_ON_WATER included, is in MIXED or MIXED_BARREL.
MIXED_BARREL = card_ids(MIXED_BARREL_DECK, _CARDS)


@pytest.mark.parametrize(
    ("make_engine", "n_cards"),
    [
        # MockEngine's size IS this repo's to pin: MOCK_CARD_NAMES is defined here.
        # The compiled engine's is not -- it is whatever table the reader extracted --
        # so None means "not ours to assert". See the note at the end of the test.
        pytest.param(MockEngine, 16, id="mock"),
        pytest.param(
            RustEngine,
            None,
            id="rust",
            marks=pytest.mark.skipif(
                not core_available(), reason="the compiled engine is not importable"
            ),
        ),
    ],
)
def test_the_shared_deck_still_holds_one_card_of_every_kind(make_engine, n_cards):
    """The suite's deck is not eight cards: it is one card of each KIND.

    Tests in this file and in several others reach into it for the building, for a
    spell, for the rolling spell and for the flying troop, and none of them says so
    out loud. If a card table changed so that the deck was eight ordinary ground
    troops, those tests would keep passing while exercising none of what they were
    written for. That is the failure this catches, on every engine, before the
    quiet ones get the chance.
    """
    cards = make_engine().cards()
    deck = [cards[i] for i in card_ids(MIXED_DECK, cards, "the shared deck")]
    assert [c.name for c in deck] == list(MIXED_DECK)
    kinds = {Placement(c.placement) for c in deck}
    for want in (Placement.BUILDING, Placement.SPELL, Placement.ROLLING, Placement.TROOP):
        assert want in kinds, f"the shared deck has no {want.name}: {[c.name for c in deck]}"
    # A TROOP, not any flying card: what the air tests need is a unit the board can
    # hold, and a spell that travels through the air would otherwise answer for it.
    flying = [c.name for c in deck if c.flying and c.placement == Placement.TROOP]
    ground = [c.name for c in deck if c.placement == Placement.TROOP and not c.flying]
    assert flying, f"the shared deck has no flying troop: {[c.name for c in deck]}"
    assert ground, f"the shared deck is all air: {[c.name for c in deck]}"
    # The barrel deck carries the one placement class the deck above cannot also hold.
    barrel = [cards[i] for i in card_ids(MIXED_BARREL_DECK, cards, "the barrel deck")]
    assert Placement.SPELL_NOT_ON_WATER in {Placement(c.placement) for c in barrel}
    # NOT pinned to a number, and that is the point. The catalogue size is a property of
    # the DATA the reader installed, not of this repo: a clone following the README runs
    # `extract_cards.py --vintage 2018` and gets 66 loadable cards where this machine's
    # newer table gives 100. Measured on two independent clones, 2026-09-22. Pinning
    # either number asserts something true only where it was written, and re-pinning it
    # to the other just moves which environment is wrong.
    #
    # What IS pinned is above: the deck names cards that exist and they are the kinds the
    # suite needs. That holds on both catalogues, which is what makes it a property of
    # this repo rather than of a machine. The size is reported so a failure above can say
    # which catalogue produced it.
    assert len(cards) >= len(MIXED_DECK), (
        f"the catalogue has {len(cards)} cards, fewer than the shared deck needs. "
        "Run tools/extract_cards.py in the RoyaleSim checkout the engine was built in."
    )
    if n_cards is not None:
        assert len(cards) == n_cards, f"catalogue size moved from {n_cards} to {len(cards)}"


def hand_slot(state, team: int, card: int) -> int:
    """Which hand slot holds ``card``, so a test can name the card it is about.

    The alternative is an index into the deck, which says nothing about what is in
    that position and stops being true the moment the deck is written differently.
    """
    hand = list(state.players[team].hand)
    assert card in hand, f"card {card} is not in the hand {hand}"
    return hand.index(card)


def mask_plane(mask, slot: int, nx: int = 18, ny: int = 32):
    """One hand slot's grid out of a flat action mask."""
    per = nx * ny
    return mask[1 + slot * per : 1 + (slot + 1) * per].reshape(ny, nx)


def test_action_space_is_documented_size():
    eng = MockEngine()
    p = TileActionParser()
    p.bind(eng)
    assert p.space.n == 1 + 4 * 18 * 32 == 2305
    h = HalfTileActionParser()
    h.bind(eng)
    assert h.space.n == 1 + 4 * 36 * 64


@pytest.mark.parametrize("parser_cls", [TileActionParser, HalfTileActionParser])
def test_masked_legal_actions_are_never_rejected(parser_cls):
    env = ClashParallelEnv(
        action_parser=parser_cls(),
        state_mutator=DefaultStateMutator(decks=[MIXED, MIXED_BARREL]),
        termination_cond=GameOverCondition(),
        truncation_cond=StepLimitCondition(600),
        decision_ms=250,
    )
    rejected = []
    deploys = {p: 0 for p in Placement}
    for seed in range(2):
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        opp = RandomLegalOpponent(noop_prob=0.4)
        while env.agents:
            acts = {a: opp.act(obs[a], obs[a]["action_mask"], rng) for a in env.agents}
            prev = env.battle_state
            obs, _, _, _, info = env.step(acts)
            for a, act in acts.items():
                st = info[a]["deploy_status"]
                if act != NOOP:
                    if st != DeployStatus.OK:
                        rejected.append((seed, prev.tick, a, act, DeployStatus(st).name))
                    else:
                        team = 0 if a == "blue" else 1
                        slot = env.action_parser.decode(act)[0]
                        card = prev.players[team].hand[slot]
                        deploys[Placement(env.engine.cards()[card].placement)] += 1
    assert rejected == []
    # Not vacuous: every placement type was actually exercised.
    assert all(n >= 5 for n in deploys.values()), deploys


def _probe_states():
    """States that exercise water, no-deploy, pocket, footprints and elixir limits."""
    eng = MockEngine()
    a = eng.arena()
    s = a.subtile
    out = []
    # 1. Opening board, full elixir for Blue, low for Red.
    eng.reset(
        1, MatchSetup(decks=[MIXED, MIXED], shuffle=ShuffleMode.NONE, elixir_milli=[10000, 2500])
    )
    out.append(eng.save_state())
    # 2. Red's own-LEFT princess down (Blue pocket on Blue's right), Blue's own-RIGHT
    #    down (Red pocket), buildings on both sides to exercise footprints.
    eng.reset(
        2,
        MatchSetup(
            decks=[MIXED, list(reversed(MIXED_BARREL))],  # Red hand holds the barrel
            shuffle=ShuffleMode.NONE,
            elixir_milli=[10000, 10000],
            tower_hp=[[2400, 1400, 0], [2400, 0, 1400]],
            spawns=[
                SpawnSpec(BLUE, CANNON, s * 9, s * 10),  # on a tile corner
                SpawnSpec(RED, CANNON, a.width - s * 5 - s // 2, a.height - s * 12 - s // 2),
                SpawnSpec(BLUE, KNIGHT, s * 4, s * 20),  # troops do NOT block placement
            ],
        ),
    )
    out.append(eng.save_state())
    # 3. Both of Red's princesses down.
    eng.reset(
        3,
        MatchSetup(
            decks=[MIXED, MIXED],
            shuffle=ShuffleMode.NONE,
            elixir_milli=[7000, 10000],
            tower_hp=[[2400, 1400, 1400], [2400, 0, 0]],
        ),
    )
    out.append(eng.save_state())
    # 4-6. Mid-battle states from a random rollout.
    parser = TileActionParser()
    parser.bind(eng)
    eng.reset(4, MatchSetup(decks=[MIXED, MIXED]))
    rng = np.random.default_rng(4)
    for i in range(300):
        s_ = eng.state()
        cmds = []
        for team in (BLUE, RED):
            legal = np.flatnonzero(parser.action_mask(s_, team))[1:]
            if legal.size and rng.random() < 0.3:
                cmds.append(parser.parse(int(rng.choice(legal)), s_, team))
        eng.step(cmds, 5)
        if i % 100 == 99:
            out.append(eng.save_state())
    return out


PROBES = _probe_states()


def _all_disagreements(parser_cls, probes=PROBES):
    eng = MockEngine()
    parser = parser_cls()
    parser.bind(eng)
    found = []
    legal_total = 0
    for blob in probes:
        eng.load_state(blob)
        st = eng.state()
        for team in (BLUE, RED):
            legal_total += int(parser.action_mask(st, team).sum()) - 1
            found += mask_disagreements(eng, parser, st, team)
    return found, legal_total


@pytest.mark.parametrize("parser_cls", [TileActionParser, HalfTileActionParser])
def test_mask_equals_engine_legality_exhaustively(parser_cls):
    found, legal_total = _all_disagreements(parser_cls)
    assert found == []
    assert legal_total > 1000  # the probes are not all "nothing is legal"


# Plants: each breaks ONE rule in the mask only. The agreement check must see it.


def test_plant_mask_lets_troops_into_the_river(monkeypatch):
    # History: the first version of this plant only dropped the water term, and it
    # did NOT land -- the river rows are outside troop territory, so with the
    # current rules the water term is shadowed and cannot fail on its own. The
    # defect class worth guarding is "troops can be dropped in the river", which
    # needs territory to reach into the band as well.
    orig = PlacementOracle.cell_grid

    def into_river(self, state, team, placement):
        g = orig(self, state, team, placement)
        if placement != Placement.SPELL:
            a = self.arena
            lo = a.water_half_rows[0]
            band = np.zeros_like(g)
            if team == BLUE:
                band[lo : lo + 2, :] = True
            else:
                band[a.hy - lo - 2 : a.hy - lo, :] = True
            g = g | (band & ~self.nodeploy)
        return g

    monkeypatch.setattr(PlacementOracle, "cell_grid", into_river)
    found, _ = _all_disagreements(TileActionParser, PROBES[:1])
    statuses = {DeployStatus(st) for _, _, st in found}
    assert DeployStatus.WATER in statuses, statuses


def shrink_king_rects_toward_the_river(oracle: PlacementOracle, half_cells: int) -> None:
    """PLANT, mask only: move each king rect's river-facing edge ``half_cells`` half-
    cells back toward its king, so the mask opens ground the engine still refuses."""
    d = half_cells * oracle.arena.half_size
    before = [list(r) for r in oracle.tower_rects]
    for owner in (BLUE, RED):
        x0, y0, x1, y1 = oracle.tower_rects[owner][0]
        oracle.tower_rects[owner][0] = (
            (x0, y0 + d, x1, y1) if owner == RED else (x0, y0, x1, y1 - d)
        )
    assert oracle.tower_rects != before, "plant did not land: rects unchanged"


@pytest.mark.parametrize(
    ("parser_cls", "half_cells", "should_land"),
    [
        # A half-tile error cannot show at tile resolution: the shrunk edge lands
        # exactly ON the next tile centre, and the rect is closed. Measured, not
        # assumed: this row is the check that the plant's reach is understood.
        (TileActionParser, 1, False),
        (HalfTileActionParser, 1, True),
        (TileActionParser, 2, True),
    ],
)
def test_plant_mask_king_rect_shrunk(parser_cls, half_cells, should_land):
    """A king rect a half-tile too small in the mask must be caught: that rect is
    what bounds the ground opened by a fallen princess."""
    eng = MockEngine()  # the engine keeps the true rule
    parser = parser_cls()
    parser.bind(eng)
    # Baseline green on exactly these probes first.
    for blob in PROBES[1:3]:
        eng.load_state(blob)
        for team in (BLUE, RED):
            assert mask_disagreements(eng, parser, eng.state(), team) == []
    shrink_king_rects_toward_the_river(parser.oracle, half_cells)
    found = []
    for blob in PROBES[1:3]:
        eng.load_state(blob)
        for team in (BLUE, RED):
            found += mask_disagreements(eng, parser, eng.state(), team)
    assert bool(found) == should_land
    assert all(st == DeployStatus.OUT_OF_TERRITORY for _, _, st in found)


def test_goblin_barrel_is_refused_only_on_water():
    """SPELL_NOT_ON_WATER on the opening board: legal on the enemy king block, on
    no-deploy cells and on a bridge; refused on every tile that touches water."""
    eng = MockEngine()
    p = TileActionParser()
    p.bind(eng)
    assert eng.cards()[BARREL].placement == Placement.SPELL_NOT_ON_WATER
    # ShuffleMode.NONE: hand slot 0 is the first card of the deck.
    barrel_first = card_ids(("GoblinBarrel", *(n for n in MIXED_DECK if n != "Zap")), _CARDS)
    eng.reset(
        0,
        MatchSetup(decks=[barrel_first, MIXED], elixir_milli=[10000, 0], shuffle=ShuffleMode.NONE),
    )
    st = eng.state()
    assert st.players[BLUE].hand[0] == BARREL
    per = 18 * 32
    grid = p.action_mask(st, BLUE)[1 : 1 + per].reshape(32, 18)
    water = p.oracle.water.reshape(32, 2, 18, 2).any(axis=(1, 3))
    assert water.sum() > 0
    assert np.array_equal(grid.astype(bool), ~water)
    assert grid[29, 9] == 1  # the enemy king block (no-deploy for troops)


def test_plant_mask_offers_the_river_to_a_goblin_barrel(monkeypatch):
    """Plant: the mask treats SPELL_NOT_ON_WATER as SPELL (anywhere) -- the shipped
    RustEngine defect before 2026-09-13 in its mildest form. The exhaustive test
    must go red on WATER verdicts only."""
    assert _all_disagreements(TileActionParser)[0] == [], "baseline must be green"
    orig = PlacementOracle.cell_grid

    def anywhere(self, state, team, placement):
        if placement == Placement.SPELL_NOT_ON_WATER:
            placement = Placement.SPELL
        return orig(self, state, team, placement)

    monkeypatch.setattr(PlacementOracle, "cell_grid", anywhere)
    assert PlacementOracle.cell_grid is anywhere, "plant did not land"
    found, _ = _all_disagreements(TileActionParser)
    assert found, "PLANT DID NOT LAND: a barrel mask open on the river passed"
    assert {(m, DeployStatus(s).name) for _, m, s in found} == {(1, "WATER")}


def test_plant_mask_ignores_building_footprints(monkeypatch):
    orig = PlacementOracle.point_grid

    def no_footprints(self, state, team, card, pitch_div):
        stripped = msgspec.structs.replace(state, entities=[])
        return orig(self, stripped, team, card, pitch_div)

    monkeypatch.setattr(PlacementOracle, "point_grid", no_footprints)
    found, _ = _all_disagreements(TileActionParser, PROBES[1:2])
    assert found
    assert all(st == DeployStatus.OCCUPIED for _, _, st in found)


def test_unmasked_illegal_action_is_rejected_and_costs_nothing():
    env = ClashParallelEnv(
        state_mutator=DefaultStateMutator(decks=[MIXED, MIXED], shuffle=ShuffleMode.NONE)
    )
    obs, _ = env.reset(seed=0)
    parser = env.action_parser
    a = env.engine.arena()
    # A troop on a water tile, own frame tile (8, 15): mask says no.
    slot = hand_slot(env.battle_state, BLUE, KNIGHT)
    act = parser.encode(slot, 8, 15)
    assert obs["blue"]["action_mask"][act] == 0
    # ...and on the enemy half.
    far = parser.encode(slot, 8, 25)
    assert obs["blue"]["action_mask"][far] == 0
    before = env.battle_state.players[BLUE]
    _, _, _, _, info = env.step({"blue": act, "red": NOOP})
    assert info["blue"]["deploy_status"] == DeployStatus.WATER
    after = env.battle_state.players[BLUE]
    assert after.hand == before.hand
    assert after.elixir_milli >= before.elixir_milli
    _, _, _, _, info = env.step({"blue": far, "red": NOOP})
    assert info["blue"]["deploy_status"] == DeployStatus.OUT_OF_TERRITORY
    del a


def test_mask_marks_unaffordable_cards_and_keeps_noop():
    eng = MockEngine()
    p = TileActionParser()
    p.bind(eng)
    eng.reset(0, MatchSetup(decks=[MIXED, MIXED], shuffle=ShuffleMode.NONE, elixir_milli=[2999, 0]))
    st = eng.state()
    m = p.action_mask(st, BLUE)
    # Blue holds 2.999 elixir. Which cards that pays for is the catalogue's business,
    # so ask it rather than trusting a comment about a deck position.
    hand = st.players[BLUE].hand
    cheap = [i for i, c in enumerate(hand) if eng.cards()[c].elixir * 1000 <= 2999]
    dear = [i for i, c in enumerate(hand) if i not in cheap]
    held = [eng.cards()[c].name for c in hand]
    assert cheap, f"fixture needs a card 2.999 pays for, hand is {held}"
    assert dear, f"fixture needs a card 2.999 does not pay for, hand is {held}"
    assert m[NOOP] == 1
    assert all(mask_plane(m, i).sum() == 0 for i in dear)
    assert all(mask_plane(m, i).sum() > 0 for i in cheap)
    assert p.action_mask(st, RED).sum() == 1  # zero elixir: only no-op


def test_spells_are_legal_everywhere_troops_only_in_territory():
    eng = MockEngine()
    p = TileActionParser()
    p.bind(eng)
    eng.reset(
        0,
        MatchSetup(
            decks=[card_ids(SPELL_FIRST_DECK, _CARDS), MIXED],
            shuffle=ShuffleMode.NONE,
            elixir_milli=[10000, 0],
        ),
    )
    st = eng.state()
    m = p.action_mask(st, BLUE).reshape(-1)
    fireball = mask_plane(m, hand_slot(st, BLUE, card_id("Fireball", _CARDS)))
    knight = mask_plane(m, hand_slot(st, BLUE, KNIGHT))
    assert fireball.all()
    assert knight[16:].sum() == 0  # nothing past the river (no pocket yet)
    assert knight[:15].sum() > 0


def test_pocket_opens_on_the_correct_side_after_a_princess_falls():
    eng = MockEngine()
    p = TileActionParser()
    p.bind(eng)
    # Red's own-LEFT princess stands on Blue's RIGHT (engine x=14.5).
    eng.reset(
        0,
        MatchSetup(
            decks=[MIXED, MIXED],
            shuffle=ShuffleMode.NONE,
            elixir_milli=[10000, 10000],
            tower_hp=[[2400, 1400, 1400], [2400, 0, 1400]],
        ),
    )
    st = eng.state()
    m = p.action_mask(st, BLUE)
    knight = mask_plane(m, hand_slot(st, BLUE, KNIGHT))
    enemy_half = knight[17:]
    assert enemy_half[:, 9:].sum() > 0, "pocket should open on Blue's right"
    assert enemy_half[:, :9].sum() == 0, "and not on Blue's left"
    # The opened ground is the far bank (y = 17) up to the enemy king's NoDeploySize
    # rect (y = 21): tile rows 17..20, 4 tiles = 8 half-rows.
    open_rows = [ty for ty in range(32) if knight[ty, 9:].any() and ty >= 17]
    assert open_rows == [17, 18, 19, 20], open_rows
    half = HalfTileActionParser()
    half.bind(eng)
    hk = mask_plane(half.action_mask(st, BLUE), hand_slot(st, BLUE, KNIGHT), 36, 64)
    assert [hy for hy in range(34, 64) if hk[hy, 18:].any()] == list(range(34, 42))
    # Buildings never get the pocket.
    cannon = mask_plane(m, hand_slot(st, BLUE, CANNON))
    assert cannon[17:].sum() == 0


def test_action_masks_method_matches_obs_for_maskable_ppo():
    env = ClashParallelEnv()
    obs, info = env.reset(seed=0)
    for agent in ("blue", "red"):
        m = env.action_masks(agent)
        assert m.dtype == np.bool_
        assert m.shape == (env.action_space(agent).n,)
        assert np.array_equal(m, obs[agent]["action_mask"].astype(bool))
        # The mask reaches the policy through the OBSERVATION, in both shapes, and
        # the info dict no longer repeats it (env.py, ACTION MASKS).
        assert "action_mask" not in info[agent]
        planes = obs[agent]["mask_planes"]
        assert planes.shape == (4, 32, 18)
        assert np.array_equal(planes.reshape(-1), obs[agent]["action_mask"][1:])


# --- the point_grid memo ------------------------------------------------------
#
# One grid serves every troop card in a hand, and Blue's enemy_troop_zone is the
# identical grid Red's mask needs, so the same result was being computed several
# times per step. Memoising it took grids computed per env.step from 2.66 to 0.27.
# The risk a memo adds is a STALE hit, so these check the key notices everything
# point_grid reads.


def _oracle_and_states():
    eng = MockEngine()
    parser = TileActionParser()
    parser.bind(eng)
    return eng, parser


def test_the_memo_returns_exactly_what_recomputing_would():
    """Cached against a deliberately un-cacheable oracle, over a played-out battle."""
    eng, parser = _oracle_and_states()
    plain = PlacementOracle(eng.arena(), eng.rules(), eng.cards())
    plain.grid_key = lambda *a, **k: None  # every call a miss
    deck = MIXED
    rng = np.random.default_rng(3)
    checked = 0
    for trial in range(4):
        eng.reset(
            trial,
            MatchSetup(
                decks=[deck, deck[::-1]],
                elixir_milli=[10000, 10000],
                tower_hp=[[2400, 0 if trial % 2 else 900, 1400], [2400, 1400, 1400]],
            ),
        )
        for _ in range(8):
            st = eng.state()
            for team in (BLUE, RED):
                for card in eng.cards():
                    for pitch in (1, 2):
                        a = parser.oracle.point_grid(st, team, card, pitch)
                        b = plain.point_grid(st, team, card, pitch)
                        assert np.array_equal(a, b), (team, card.name, pitch)
                        checked += 1
            cmds = []
            for team in (BLUE, RED):
                legal = np.flatnonzero(parser.action_mask(st, team))[1:]
                if legal.size and rng.random() < 0.6:
                    cmds.append(parser.parse(int(rng.choice(legal)), st, team))
            eng.step(cmds, 10)
    assert checked > 2000
    assert parser.oracle.grid_hits > 0, "vacuous: the memo never served anything"


def test_a_memoised_grid_is_read_only():
    """Callers share it, so writing to one seat's grid would change the other's."""
    eng, parser = _oracle_and_states()
    eng.reset(0, MatchSetup(decks=[MIXED] * 2))
    grid = parser.oracle.point_grid(eng.state(), BLUE, eng.cards()[KNIGHT], 1)
    with pytest.raises(ValueError, match="read-only"):
        grid[0, 0] = True


@pytest.mark.parametrize("what", ["tower falls", "building appears", "seat", "pitch"])
def test_the_memo_key_notices_everything_the_grid_reads(what):
    """A stale hit is the only way a memo can be wrong. Change one input at a time."""
    eng, parser = _oracle_and_states()
    deck = MIXED
    sub = eng.arena().subtile
    troop = next(c for c in eng.cards() if c.placement == Placement.TROOP)

    base = MatchSetup(decks=[deck, deck], elixir_milli=[10000, 10000])
    eng.reset(0, base)
    before = parser.oracle.point_grid(eng.state(), BLUE, troop, 1)

    if what == "seat":
        after = parser.oracle.point_grid(eng.state(), RED, troop, 1)
    elif what == "pitch":
        after = parser.oracle.point_grid(eng.state(), BLUE, troop, 2)
        assert after.shape != before.shape
        return
    else:
        if what == "tower falls":
            fallen = [[2400, 1400, 1400], [2400, 0, 1400]]
            eng.reset(0, msgspec.structs.replace(base, tower_hp=fallen))
        else:
            # A real BUILDING, at a tile CENTRE. Both parts took a wrong test to
            # find: a troop blocks nothing (point_grid skips them), and a building
            # at a tile corner reaches no tile centre either -- the nearest is
            # sqrt(2)/2 of a tile away, which is further than a Cannon's radius.
            building = next(c for c in eng.cards() if c.placement == Placement.BUILDING)
            centre = 9 * sub + sub // 2
            eng.reset(
                0,
                msgspec.structs.replace(
                    base, spawns=[SpawnSpec(BLUE, building.card_id, centre, centre)]
                ),
            )
        after = parser.oracle.point_grid(eng.state(), BLUE, troop, 1)
    assert not np.array_equal(before, after), f"{what} did not change the grid"


def test_every_troop_card_shares_one_grid_and_a_building_does_not():
    """Why the memo pays: the grid is not a function of the card, except for size."""
    eng, parser = _oracle_and_states()
    eng.reset(0, MatchSetup(decks=[MIXED] * 2))
    st = eng.state()
    troops = [c for c in eng.cards() if c.placement == Placement.TROOP]
    assert len(troops) > 2
    keys = {parser.oracle.grid_key(st, BLUE, c, 1) for c in troops}
    assert len(keys) == 1, "troop cards should share one key; the footprint term is 0"
    buildings = [c for c in eng.cards() if c.placement == Placement.BUILDING]
    if len({c.radius for c in buildings}) > 1:
        bkeys = {parser.oracle.grid_key(st, BLUE, c, 1) for c in buildings}
        assert len(bkeys) > 1, "buildings of different radius must not share a grid"
