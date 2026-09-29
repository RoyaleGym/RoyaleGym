"""An ability button, as an action (RoyaleSim 244c893 on; HERO-SPEC 6.2).

The engine takes a command in slot HAND_SIZE + k as a press of ability button k. A side's
buttons are its hero entries (form 2) in deck order, then its champions in deck order;
how many is the engine's count. The parser offers the buttons only when asked
(``ability_buttons=True``), after every tile action, and masks each from the engine's row
for it, whichever card it is: on when the engine calls it available (a living unit behind
it, off cooldown, not mid-ability), a hero's charge is unspent and the bar can pay. Graded
here against the engine itself for a hero: the mask is on exactly when the engine accepts
the press, off before the hero exists and after its charge is spent, and a press the mask
offered makes the ability happen. The champion's columns (card_id, cooldown_ticks) and its
third button are graded on stated rows until the engine has them.
"""

from __future__ import annotations

import msgspec
import numpy as np
import pytest

from royalegym import ClashParallelEnv, DefaultStateMutator, RandomLegalOpponent
from royalegym.action import HalfTileActionParser, TileActionParser
from royalegym.mock_engine import MockEngine
from royalegym.obs import SpatialObsBuilder
from royalegym.protocol import (
    ABILITY_BUTTONS,
    BLUE,
    EMPTY_CARD,
    HAND_SIZE,
    RED,
    DeployCommand,
    DeployStatus,
    EntityKind,
    MatchSetup,
    PlayerState,
    ShuffleMode,
    ability_row,
    to_engine,
)
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, _core, core_available
from royalegym.viser import player_dict

needs_heroes = pytest.mark.skipif(
    not core_available() or not hasattr(_core, "ABILITY_BUTTONS"),
    reason=str(CORE_IMPORT_ERROR) if not core_available() else "this engine has no heroes",
)
DECK = ("Musketeer", "Knight", "Archer", "Giant", "Minions", "Fireball", "Zap", "Cannon")


def test_the_parser_adds_the_buttons_only_when_asked():
    eng = MockEngine()
    for cls in (TileActionParser, HalfTileActionParser):
        plain, with_buttons = cls(), cls(ability_buttons=True)
        plain.bind(eng)
        with_buttons.bind(eng)
        assert with_buttons.n_actions == plain.n_actions + ABILITY_BUTTONS
        assert "ability_buttons" not in plain.config(), "a parser without buttons moved"
        assert with_buttons.button_of(plain.n_actions) == 0
        assert with_buttons.button_of(plain.n_actions + 1) == 1
        assert with_buttons.button_of(plain.n_actions - 1) is None
        cmd = with_buttons.parse(plain.n_actions + 1, None, BLUE)  # a press reads no state
        assert (cmd.team, cmd.hand_slot) == (BLUE, HAND_SIZE + 1)


def test_a_row_is_read_by_column_whatever_the_engine_appends():
    """The rows grow columns (card_id and cooldown_ticks for the champions), and a reader
    that unpacked three would stop at the first engine that sends five."""
    assert ability_row([1, 0, 2]) == (1, 0, 2, EMPTY_CARD, -1)
    assert ability_row([0, 0, 3, 17, 40]) == (0, 0, 3, 17, 40)
    assert ability_row([1, 0, 3, 17, 0, 99]).card_id == 17


def _three_buttons() -> MockEngine:
    eng = MockEngine()
    eng.ability_button_count = 3  # the champion engine's count: two hero forms, a champion
    return eng


def test_the_button_count_is_the_engines():
    eng = _three_buttons()
    plain, parser = TileActionParser(), TileActionParser(ability_buttons=True)
    plain.bind(eng)
    parser.bind(eng)
    assert (parser.n_buttons, parser.n_actions) == (3, plain.n_actions + 3)
    assert parser.button_of(plain.n_actions + 2) == 2
    for team in (BLUE, RED):
        cmd = parser.parse(plain.n_actions + 2, None, team)
        assert (cmd.team, cmd.hand_slot) == (team, HAND_SIZE + 2)
    b = SpatialObsBuilder()
    b.bind(eng, parser)
    assert b.observation_space()["ability_ready"].shape == (3,)


def test_the_mask_reads_each_buttons_row_on_both_seats():
    """No card named anywhere: a hero's row of three columns, champions' rows of five (and
    one with a sixth the gym does not know), cooldown, a spent charge and the bar, per seat."""
    eng = _three_buttons()
    parser = TileActionParser(ability_buttons=True)
    parser.bind(eng)
    eng.reset(1, MatchSetup(decks=[list(range(8)), list(range(8))]))
    state = eng.state()
    rows = {
        BLUE: [
            [1, 0, 2],  # a hero, up and unspent, 2 of the 5 elixir: on
            [0, 0, 3, 17, 40],  # a champion on cooldown for 40 more ticks: off
            [1, 0, 6, 18, 0],  # a champion ready, but 6 elixir on a bar of 5: off
        ],
        RED: [
            [0, 1, 2],  # a hero whose one charge is spent: off
            [1, 0, 3, 17, 0, 7],  # a champion ready, and a column no engine sends yet: on
        ],  # no third button on this side: off
    }
    state = msgspec.structs.replace(state, players=[
        msgspec.structs.replace(p, elixir_milli=5000, abilities=rows[p.team])
        for p in state.players
    ])
    got = {t: list(parser.action_mask(state, t)[parser.n_tile_actions:]) for t in (BLUE, RED)}
    assert got == {BLUE: [1, 0, 0], RED: [0, 1, 0]}


def _champions(rows: int) -> type[MockEngine]:
    """MockEngine whose Red reports ``rows`` champion buttons: a champion deck marks no
    form, so only the engine's rows show it has a button."""
    class Champions(MockEngine):
        def state(self):
            s = super().state()
            red = msgspec.structs.replace(s.players[RED], abilities=[[0, 0, 1, 0, 0]] * rows)
            return msgspec.structs.replace(s, players=[s.players[BLUE], red])

    return Champions


def test_a_champion_on_a_parser_without_buttons_plays_as_a_troop():
    """Any deck may hold a champion (the default mutator deals the whole catalogue), so a
    parser without buttons is not refused one: its space is what it always was, and the
    champion's ability is never offered. Refused, every default env on an engine with a
    champion button stopped at the first deck that dealt one (RoyaleSim's Golden Knight
    build: three of the examples)."""
    env = ClashParallelEnv(engine=_champions(1)())
    obs, _ = env.reset(seed=1)
    assert "ability_ready" not in obs["red"]
    assert len(obs["red"]["action_mask"]) == env.action_parser.n_tile_actions
    env.step({"blue": 0, "red": 0})


def test_a_parser_with_buttons_must_reach_every_button_the_engine_reports():
    env = ClashParallelEnv(
        engine=_champions(1)(), action_parser=TileActionParser(ability_buttons=True)
    )
    obs, _ = env.reset(seed=1)
    assert list(obs["red"]["ability_ready"]) == [0, 0]
    with pytest.raises(ValueError, match=r"reports 3 ability buttons for seat 1.*can press 2"):
        ClashParallelEnv(
            engine=_champions(3)(), action_parser=TileActionParser(ability_buttons=True)
        ).reset(seed=1)


def test_mock_engine_models_no_hero():
    eng = MockEngine()
    ids = {c.name: c.card_id for c in eng.cards()}
    deck = [ids[n] for n in DECK]
    forms = [[2, 0, 0, 0, 0, 0, 0, 0], [0] * 8]
    with pytest.raises(NotImplementedError, match="forms"):
        eng.reset(1, MatchSetup(decks=[deck, deck], forms=forms))
    eng.reset(1, MatchSetup(decks=[deck, deck], forms=[[0] * 8, [0] * 8]))
    for k in range(ABILITY_BUTTONS):
        assert eng.check_deploy(DeployCommand(BLUE, HAND_SIZE + k, 0, 0)) == DeployStatus.NO_HERO


def test_a_hero_deck_on_a_parser_without_buttons_is_refused():
    env = ClashParallelEnv(
        engine=MockEngine(),
        state_mutator=DefaultStateMutator(forms=[[2] + [0] * 7, [0] * 8]),
    )
    with pytest.raises(ValueError, match="ability_buttons=True"):
        env.reset(seed=1)


def test_the_stream_names_the_heroes_and_the_evolutions():
    names = {3: "Musketeer", 5: "Cannon"}
    p = PlayerState(
        team=BLUE, elixir_milli=5000, hand=[3, 5, 1, 2], next_card=4, crowns=0,
        tower_hp=[1, 1, 1], tower_max_hp=[1, 1, 1], king_active=False,
        abilities=[[1, 0, 2]], evo=[[5, 2, 1]],
    )
    d = player_dict(p, lambda c: names.get(c, f"card{c}"), [3, 5, 1, 2, 4, 6, 7, 8],
                    [2, 1, 0, 0, 0, 0, 0, 0])
    assert d["abilities"] == [["Musketeer", 1, 0, 2]]
    assert d["evo"] == [["Cannon", 2, 1]]
    assert "ability_cooldowns" not in d, "no row says a cooldown: the frame is unchanged"
    unknown = player_dict(p, lambda c: names.get(c, "?"), None, None)
    assert unknown["abilities"] == [["", 1, 0, 2]], "no forms: the name is unknown, not guessed"


def test_the_stream_names_a_champion_button_by_its_card_in_the_viewers_four_columns():
    names = {3: "Musketeer", 9: "Golden Knight"}
    p = PlayerState(
        team=RED, elixir_milli=5000, hand=[3, 5, 1, 2], next_card=4, crowns=0,
        tower_hp=[1, 1, 1], tower_max_hp=[1, 1, 1], king_active=False,
        abilities=[[1, 0, 2], [0, 0, 1, 9, 40, 7]],
    )
    d = player_dict(p, lambda c: names.get(c, f"card{c}"), [3, 5, 1, 2, 4, 6, 7, 9],
                    [2, 0, 0, 0, 0, 0, 0, 0])
    assert d["abilities"] == [["Musketeer", 1, 0, 2], ["Golden Knight", 0, 0, 1]]
    # The cooldown on its own key, parallel to the rows: -1 where a row does not say.
    assert d["ability_cooldowns"] == [-1, 40]
    unknown = player_dict(p, lambda c: names.get(c, "?"), None, None)
    assert [r[0] for r in unknown["abilities"]] == ["", "Golden Knight"], (
        "a row that names its card needs no forms"
    )


def _hero_battle():
    eng = RustEngine()
    ids = {c.name: c.card_id for c in eng.cards()}
    deck = [ids[n] for n in DECK]
    forms = [[2, 0, 0, 0, 0, 0, 0, 0], [0] * 8]  # Blue's Musketeer is her hero form
    eng.reset(1, MatchSetup(
        decks=[deck, deck], shuffle=ShuffleMode.NONE, forms=forms,
        elixir_milli=[10000, 10000], start_tick=eng.rules().deploy_lockout_ticks,
    ))
    parser = TileActionParser(ability_buttons=True)
    parser.bind(eng)
    return eng, parser, ids


def _agree(eng, parser, state) -> list[str]:
    """Every button, both seats: the mask's bit against the engine's verdict."""
    out = []
    for team in (BLUE, RED):
        mask = parser.action_mask(state, team)
        for k in range(parser.n_buttons):
            status = eng.check_deploy(DeployCommand(team, HAND_SIZE + k, 0, 0))
            if bool(mask[parser.n_tile_actions + k]) != (status == DeployStatus.OK):
                out.append(f"tick {state.tick} seat {team} button {k}: mask "
                           f"{int(mask[parser.n_tile_actions + k])}, engine "
                           f"{DeployStatus(status).name}")
    return out


@needs_heroes
def test_the_button_is_on_exactly_when_the_engine_takes_the_press_and_the_ability_fires():
    eng, parser, ids = _hero_battle()
    state = eng.state()
    problems = _agree(eng, parser, state)
    # No hero on the board yet: nothing to press, on either seat.
    assert not parser.action_mask(state, BLUE)[parser.n_tile_actions:].any()
    assert eng.check_deploy(DeployCommand(BLUE, HAND_SIZE, 0, 0)) == DeployStatus.NO_HERO

    # Blue plays its hero Musketeer on its own side.
    slot = state.players[BLUE].hand.index(ids["Musketeer"])
    t = eng.arena().subtile
    x, y = to_engine(eng.arena(), BLUE, 9 * t + t // 2, 10 * t + t // 2)
    assert eng.step([DeployCommand(BLUE, slot, x, y)], 1)[0].status == DeployStatus.OK
    on_at = None
    for _ in range(200):
        eng.step([], 1)
        state = eng.state()
        problems += _agree(eng, parser, state)
        if parser.action_mask(state, BLUE)[parser.n_tile_actions]:
            on_at = state.tick
            break
    assert on_at is not None, f"the button never came on: {state.players[BLUE].abilities}"
    assert not parser.action_mask(state, BLUE)[parser.n_tile_actions + 1], "no second hero"
    assert not parser.action_mask(state, RED)[parser.n_tile_actions:].any(), "Red has none"

    # The press the mask offered: accepted, paid for, and the ability happens.
    before_elixir = state.players[BLUE].elixir_milli
    before = {e.uid for e in state.entities}
    cost = state.players[BLUE].abilities[0][2]
    cmd = parser.parse(parser.n_tile_actions, state, BLUE)
    result = eng.step([cmd], 1)[0]
    assert result.status == DeployStatus.OK, DeployStatus(result.status).name
    state = eng.state()
    assert before_elixir - state.players[BLUE].elixir_milli >= cost * 1000 - 100, (
        "the press was not paid for"
    )
    fired = False
    for _ in range(100):
        state = eng.state()
        problems += _agree(eng, parser, state)
        new = [e for e in state.entities if e.uid not in before and e.team == BLUE]
        if any(e.kind == EntityKind.BUILDING for e in new):
            fired = True
            break
        eng.step([], 1)
    assert fired, "the press was accepted and nothing appeared (Hero Musketeer's turret)"

    # Its one charge is spent: off, and the engine says why.
    state = eng.state()
    assert not parser.action_mask(state, BLUE)[parser.n_tile_actions]
    assert eng.check_deploy(DeployCommand(BLUE, HAND_SIZE, 0, 0)) == DeployStatus.ABILITY_SPENT
    assert problems == [], problems[:6]


@needs_heroes
def test_the_observation_carries_the_buttons_and_keeps_its_planes():
    eng, parser, _ = _hero_battle()
    b = SpatialObsBuilder()
    b.bind(eng, parser)
    state = eng.state()
    b.reset(state)
    obs = b.build(state, BLUE, parser.action_mask(state, BLUE))
    assert obs["action_mask"].shape == (parser.n_actions,)
    assert obs["mask_planes"].shape == parser.mask_plane_shape()
    assert list(obs["ability_ready"]) == list(obs["action_mask"][parser.n_tile_actions:])
    assert b.observation_space()["ability_ready"].shape == (parser.n_buttons,)
    assert parser.n_buttons == eng.ability_button_count == _core.ABILITY_BUTTONS
    # One past the last button is refused as a slot, whatever the count.
    assert eng._wire(DeployCommand(BLUE, 99, 0, 0))[1] == HAND_SIZE + parser.n_buttons
    plain = TileActionParser()
    plain.bind(eng)
    b2 = SpatialObsBuilder()
    b2.bind(eng, plain)
    assert "ability_ready" not in b2.observation_space().spaces, "off by default"


@needs_heroes
def test_an_env_plays_a_hero_deck_with_buttons():
    eng = RustEngine()
    ids = {c.name: c.card_id for c in eng.cards()}
    deck = [ids[n] for n in DECK]
    env = ClashParallelEnv(
        engine=eng,
        action_parser=TileActionParser(ability_buttons=True),
        state_mutator=DefaultStateMutator(decks=[deck, deck], forms=[[2] + [0] * 7, [0] * 8]),
    )
    obs, _ = env.reset(seed=3)
    rng = np.random.default_rng(3)
    policy = RandomLegalOpponent(noop_prob=0.5)
    for _ in range(300):
        acts = {a: policy.act(obs[a], obs[a]["action_mask"], rng) for a in env.agents}
        obs, _, term, trunc, info = env.step(acts)
        # Every action the mask offered, a press included, is one the engine takes.
        assert acts["blue"] == 0 or info["blue"]["deploy_status"] == 0, info["blue"]
        if term["blue"] or trunc["blue"]:
            break
    assert "ability_ready" in obs["blue"]


def _champion_battle(name: str):
    """Both seats deal the same deck with the champion ``name`` in it (a champion needs no
    form), past the opening lockout, with elixir to spare; None when the engine has no such
    card, or reports no button for it."""
    eng = RustEngine()
    ids = {c.name: c.card_id for c in eng.cards()}
    if name not in ids:
        return None
    deck = [ids[name], *(ids[n] for n in DECK if n != "Musketeer")]
    eng.reset(1, MatchSetup(
        decks=[deck, deck], shuffle=ShuffleMode.NONE,
        elixir_milli=[10000, 10000], start_tick=eng.rules().deploy_lockout_ticks,
    ))
    if not any(ability_row(r).card_id == ids[name] for r in eng.state().players[BLUE].abilities):
        return None
    parser = TileActionParser(ability_buttons=True)
    parser.bind(eng)
    return eng, parser, ids[name]


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_a_champions_button_agrees_with_the_engine_through_a_press_and_its_cooldown():
    """The Golden Knight (RoyaleSim's first champion) on both seats, every button, every tick:
    the mask is on exactly when the engine takes the press. Each seat presses the first time
    its mask offers it; the press is taken and paid, the button goes off while the chain runs
    and through the cooldown the row counts down, and comes back on. A champion's charge is
    never spent. Nothing here is the Golden Knight's but the name that deals it."""
    battle = _champion_battle("GoldenKnight")
    if battle is None:
        pytest.skip("this engine has no champion button (the Golden Knight's)")
    eng, parser, champion = battle
    giant = next(c.card_id for c in eng.cards() if c.name == "Giant")
    t = eng.arena().subtile
    # Each seat's champion at its own left bridge, and its Giant at its own right bridge, so
    # each champion meets the other seat's Giant: something in reach to dash at, that does
    # not fight back, so the chain and its cooldown come soon after the press. What a press
    # with nothing in reach does next (wait for a target, or run to one) is the engine's
    # rule, and it has moved; this test reads only the rows, never how long the wait is.
    for card, (tx, ty) in ((champion, (3, 13)), (giant, (14, 14))):
        state = eng.state()
        cmds = []
        for team in (BLUE, RED):
            slot = state.players[team].hand.index(card)
            x, y = to_engine(eng.arena(), team, tx * t + t // 2, ty * t + t // 2)
            cmds.append(DeployCommand(team, slot, x, y))
        assert [r.status for r in eng.step(cmds, 1)] == [DeployStatus.OK] * 2

    problems: list[str] = []
    pressed: dict[int, int] = {}
    counted: dict[int, int] = {}
    back: dict[int, int] = {}
    for _ in range(900):
        state = eng.state()
        problems += _agree(eng, parser, state)
        presses = []
        for team in (BLUE, RED):
            p = state.players[team]
            rows = [ability_row(r) for r in p.abilities]
            assert [(b.card_id, b.spent) for b in rows] == [(champion, 0)], rows
            on = bool(parser.action_mask(state, team)[parser.n_tile_actions])
            if team in pressed and rows[0].cooldown_ticks > 0:
                counted[team] = max(counted.get(team, 0), rows[0].cooldown_ticks)
            if on and team in pressed and team not in back:
                back[team] = state.tick
            if on and team not in pressed:
                presses.append(parser.parse(parser.n_tile_actions, state, team))
        if presses:
            before = {q: state.players[q].elixir_milli for q in (BLUE, RED)}
            results = eng.step(presses, 1)
            after = eng.state()
            for cmd, r in zip(presses, results, strict=True):
                assert r.status == DeployStatus.OK, (cmd, DeployStatus(r.status).name)
                cost = ability_row(state.players[cmd.team].abilities[0]).cost
                paid = before[cmd.team] - after.players[cmd.team].elixir_milli
                assert paid >= cost * 1000 - 100, f"seat {cmd.team} paid {paid} for {cost}"
                assert not parser.action_mask(after, cmd.team)[parser.n_tile_actions], (
                    f"seat {cmd.team}: the button is still on the tick after its press"
                )
                pressed[cmd.team] = state.tick
            continue
        if len(back) == 2:
            break
        eng.step([], 1)
    assert problems == [], problems[:6]
    assert set(pressed) == {BLUE, RED}, f"a seat never had its button on: pressed {pressed}"
    assert set(counted) == {BLUE, RED}, f"no cooldown was counted down: {counted}"
    assert set(back) == {BLUE, RED}, (
        f"the button did not come back after its cooldown: pressed {pressed}, counted "
        f"{counted}, back {back}"
    )


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_a_sides_buttons_are_its_heroes_then_its_champion():
    """Blue deals a hero Musketeer and a Golden Knight: its hero's button first and its
    champion's second, each row naming its card. Red deals the same cards, no hero form: its
    champion's button is its first. The champion's second button, pressed through the action
    space, is the one the engine takes, on every tick of the champion's arrival."""
    if _champion_battle("GoldenKnight") is None:
        pytest.skip("this engine has no champion button (the Golden Knight's)")
    eng = RustEngine()
    ids = {c.name: c.card_id for c in eng.cards()}
    names = ("Musketeer", "GoldenKnight", "Knight", "Archer", "Giant", "Minions", "Fireball", "Zap")
    deck = [ids[n] for n in names]
    eng.reset(1, MatchSetup(
        decks=[deck, deck], shuffle=ShuffleMode.NONE, forms=[[2] + [0] * 7, [0] * 8],
        elixir_milli=[10000, 10000], start_tick=eng.rules().deploy_lockout_ticks,
    ))
    parser = TileActionParser(ability_buttons=True)
    parser.bind(eng)
    state = eng.state()
    assert [ability_row(r).card_id for r in state.players[BLUE].abilities] == [
        ids["Musketeer"], ids["GoldenKnight"]
    ]
    assert [ability_row(r).card_id for r in state.players[RED].abilities] == [ids["GoldenKnight"]]

    t = eng.arena().subtile
    slot = state.players[BLUE].hand.index(ids["GoldenKnight"])
    x, y = to_engine(eng.arena(), BLUE, 3 * t + t // 2, 3 * t + t // 2)
    assert eng.step([DeployCommand(BLUE, slot, x, y)], 1)[0].status == DeployStatus.OK
    problems, on_at = [], None
    for _ in range(100):
        state = eng.state()
        problems += _agree(eng, parser, state)
        if parser.action_mask(state, BLUE)[parser.n_tile_actions + 1]:
            on_at = state.tick
            break
        eng.step([], 1)
    assert on_at is not None, (
        f"Blue's champion button never came on: {state.players[BLUE].abilities}"
    )
    assert not parser.action_mask(state, BLUE)[parser.n_tile_actions], "no hero on the board"
    cmd = parser.parse(parser.n_tile_actions + 1, state, BLUE)
    assert cmd.hand_slot == HAND_SIZE + 1
    assert eng.step([cmd], 1)[0].status == DeployStatus.OK
    assert problems == [], problems[:6]
