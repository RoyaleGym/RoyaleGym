"""The command delay (RoyaleSim r16): a play or press accepted on tick T runs on T + delay.

WHAT IT CHECKS
    a. The option is refused where the engine cannot run it or cannot name its refusal, and
       recorded in the engine's config where it can.
    b. The held mask (``hold_while_pending=True``): a seat with a command waiting is offered
       only the no-op.
    c. The engine's own mask (the default) agrees with ``check_deploy`` over
       the whole action space at every state where a command waits: the waiting card and
       button refused (CARD_PENDING), the bar less what the waiting commands spoke for.
    d. Each seat's MatchMemory stays exact through a battle played under a delay: its
       enemy count equals the engine's bar at every step, plays paid at the tick they run.

SKIPS
    Without royalesim built, or on an engine without a command delay. Not a pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from royalegym.action import TileActionParser, mask_disagreements
from royalegym.env import ClashParallelEnv
from royalegym.mock_engine import MockEngine
from royalegym.protocol import DeployStatus, MatchSetup
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, _core, core_available

HAS_DELAY = core_available() and "CARD_PENDING" in getattr(_core, "DEPLOY_REASONS", ())
needs_delay = pytest.mark.skipif(
    not HAS_DELAY,
    reason=str(CORE_IMPORT_ERROR) if not core_available() else "this engine has no command delay",
)
DELAY = 21  # the live client's, measured (RoyaleSim r16)


def test_a_delay_is_refused_where_it_cannot_run():
    with pytest.raises(NotImplementedError, match="no command delay"):
        ClashParallelEnv(MockEngine(), command_delay_ticks=5)
    ClashParallelEnv(MockEngine(), command_delay_ticks=0)  # nothing asked, nothing refused
    if core_available() and not HAS_DELAY:
        with pytest.raises(NotImplementedError):
            RustEngine(command_delay_ticks=DELAY)
    if core_available():
        with pytest.raises(ValueError, match="ticks >= 0"):
            RustEngine(command_delay_ticks=-1)


@needs_delay
def test_the_delay_is_set_and_recorded():
    eng = RustEngine(command_delay_ticks=(DELAY, 3))
    assert eng.command_delay_ticks == (DELAY, 3)
    assert eng.config()["command_delay_ticks"] == [DELAY, 3]
    assert "command_delay_ticks" not in RustEngine().config(), "a plain engine reads as before"
    env = ClashParallelEnv(RustEngine(), command_delay_ticks=DELAY)
    assert env.engine.command_delay_ticks == (DELAY, DELAY)


def _play(env, rng, steps, on_state):
    obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
    for _ in range(steps):
        on_state(obs)
        acts = {}
        for a in env.agents:
            legal = np.flatnonzero(obs[a]["action_mask"])[1:]
            acts[a] = int(rng.choice(legal)) if legal.size and rng.random() < 0.6 else 0
        obs, *_ = env.step(acts)
        if not env.agents:
            break


@needs_delay
def test_the_held_mask_offers_only_the_noop_while_a_command_waits():
    env = ClashParallelEnv(
        RustEngine(), action_parser=TileActionParser(hold_while_pending=True),
        command_delay_ticks=DELAY,
    )
    seen = {"waiting": 0}

    def on_state(obs):
        for agent, team in (("blue", 0), ("red", 1)):
            if env.battle_state.players[team].pending:
                seen["waiting"] += 1
                assert obs[agent]["action_mask"].sum() == 1, "offered more than the no-op"
                assert obs[agent]["action_mask"][0] == 1

    _play(env, np.random.default_rng(1), 400, on_state)
    assert seen["waiting"] > 20, f"only {seen['waiting']} states with a command waiting"


@needs_delay
def test_the_engines_own_mask_agrees_with_the_engine_while_commands_wait():
    parser = TileActionParser()
    env = ClashParallelEnv(RustEngine(), action_parser=parser, command_delay_ticks=DELAY)
    seen = {"states": 0, "refused_pending": 0}

    def on_state(obs):
        state = env.battle_state
        for team in (0, 1):
            if not state.players[team].pending or seen["states"] >= 30:
                continue
            seen["states"] += 1
            bad = mask_disagreements(env.engine, parser, state, team)
            assert bad == [], bad[:5]
            seen["refused_pending"] += sum(
                1 for a in range(1, parser.n_tile_actions, 97)
                if env.engine.check_deploy(parser.parse(a, state, team))
                == DeployStatus.CARD_PENDING
            )

    _play(env, np.random.default_rng(2), 400, on_state)
    assert seen["states"] >= 10, seen
    assert seen["refused_pending"] > 0, "no tap met a waiting card: the check saw no pending"


#: Named, not drawn: a drawn deck may hold a card that pays the OPPONENT elixir (an Elixir
#: Golem's pieces do when they die), which the count does not model and flags as not exact.
#: So "always exact" held only while the seed never drew one; RoyaleSim r17's longer card
#: list drew one. Eight plain cards across the elixir range, both seats.
MEMORY_DECK = ("Knight", "Archer", "Giant", "Minions", "Fireball", "Cannon", "Zap", "Musketeer")


@needs_delay
def test_the_memories_stay_exact_under_a_delay():
    from royalegym.state_mutator import DefaultStateMutator

    engine = RustEngine()
    ids = {c.name: c.card_id for c in engine.cards()}
    deck = [ids[n] for n in MEMORY_DECK]
    env = ClashParallelEnv(
        engine, command_delay_ticks=DELAY,
        state_mutator=DefaultStateMutator(decks=[deck, deck]),
    )
    off = []

    def on_state(obs):
        state = env.battle_state
        for team, m in env.obs_builder.memory.items():
            foe = state.players[1 - team]
            if m.enemy_elixir_milli() != foe.elixir_milli or not m.exact:
                off.append((state.tick, team, m.enemy_elixir_milli(), foe.elixir_milli, m.exact))

    plays = {"n": 0}
    rng = np.random.default_rng(3)
    for _ in range(3):
        _play(env, rng, 700, on_state)
        plays["n"] += env.obs_builder.memory[0].foe_plays
    assert plays["n"] > 30, f"only {plays['n']} enemy plays seen"
    assert off == [], off[:6]


def test_the_vector_carries_the_own_waiting_commands_and_never_the_enemys():
    """A player knows its own taps: a hand slot with a play waiting is flagged and not
    affordable, and the waiting cost is shown. The client shows an opponent's play only
    when it runs, so the enemy's waiting commands change nothing in the fair vector."""
    import msgspec

    from royalegym.obs import SpatialObsBuilder, vector_offsets

    eng = MockEngine()
    eng.reset(1, MatchSetup(decks=[list(range(8)), list(range(8))], elixir_milli=[8000, 8000]))
    parser = TileActionParser()
    parser.bind(eng)
    state = eng.state()
    card = state.players[0].hand[1]
    cost = eng.cards()[card].elixir
    row = [0, card, 0, 0, 12, cost]

    def vector(own, foe):
        b = SpatialObsBuilder()
        b.bind(eng, parser)
        s = msgspec.structs.replace(state, players=[
            msgspec.structs.replace(p, pending=rows, pending_cost=sum(r[5] for r in rows))
            for p, rows in zip(state.players, (own, foe), strict=True)
        ])
        b.reset(s)
        return b.build(s, 0, parser.action_mask(s, 0))["vector"]

    off = vector_offsets(len(eng.cards()))
    plain = vector([], [])
    assert np.array_equal(vector([], [row]), plain), "the enemy's waiting play leaked"
    own = vector([row], [])
    assert list(own[off["own_hand_pending"]]) == [0, 1, 0, 0]
    assert own[off["own_hand_affordable"]][1] == 0
    assert own[off["own_pending_cost"]][0] == pytest.approx(cost / 10)
    assert own[off["own_elixir"]][0] == plain[off["own_elixir"]][0], "the bar is spent at the tap"
