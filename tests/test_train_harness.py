"""What a training harness needs from the env, and nothing about the game itself.

Six behaviours live here because a trainer -- not a battle -- is what breaks when
they are wrong, and two of them were bugs that no game-level test could see:

* episode statistics reach ``final_info`` once per EPISODE, so a run does not keep
  a shadow copy of the state to compute its own;
* ``config()`` and ``build_digest()`` say what env and what data a checkpoint was
  produced on -- including whether hidden information was revealed to it;
* ``CombinedReward.last_terms`` is per TEAM. It was one flat dict cleared at the
  top of every call, and the env calls the reward once per seat, so the first
  seat's breakdown was destroyed by the second every single step;
* the viewer's publisher is bound ONCE by the vector env. Every ClashParallelEnv
  used to build its own from an environment variable and bind the same fixed UDP
  port, so self-play with more than one game raised OSError the moment a viewer
  was configured;
* the batched rewards are float32 and the info dict no longer repeats the mask;
* an env cannot be pickled, so ``EnvFactory`` is the recipe a subprocess worker
  gets instead.
"""

from __future__ import annotations

import json
import pickle

import msgspec
import numpy as np
import pytest

from royalegym.done_condition import GameOverCondition, StepLimitCondition
from royalegym.env import (
    AGENTS,
    EPISODE_STAT_KEYS,
    ClashParallelEnv,
    ClashSelfPlayVecEnv,
    EnvFactory,
)
from royalegym.mock_engine import MockEngine
from royalegym.obs import EntityListObsBuilder, Reveal, SpatialObsBuilder
from royalegym.protocol import MatchSetup, TowerSlot, calibration_digest
from royalegym.reward import CombinedReward, CrownReward, TowerHPReward, WinLossReward
from royalegym.rust_engine import core_available
from royalegym.state_mutator import DefaultStateMutator
from royalegym.viser import ENV_VAR, ViserPublisher

DECK = [0, 3, 10, 14, 11, 13, 7, 9]


def short_env(max_steps: int = 6, **kwargs) -> ClashParallelEnv:
    return ClashParallelEnv(
        engine=MockEngine(),
        state_mutator=DefaultStateMutator(decks=[DECK, list(reversed(DECK))]),
        termination_cond=GameOverCondition(),
        truncation_cond=StepLimitCondition(max_steps),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. terminal-only episode statistics
# ---------------------------------------------------------------------------


def test_the_last_info_of_an_episode_carries_its_statistics_and_no_other_info_does():
    env = short_env(max_steps=6)
    _, infos = env.reset(seed=1)
    for agent in AGENTS:
        assert not set(EPISODE_STAT_KEYS) & set(infos[agent]), "reset is not a terminal step"
    seen_terminal = False
    for step in range(1, 7):
        _, _, term, trunc, infos = env.step(dict.fromkeys(AGENTS, 0))
        done = term["blue"] or trunc["blue"]
        for agent in AGENTS:
            present = set(EPISODE_STAT_KEYS) & set(infos[agent])
            if done:
                assert present == set(EPISODE_STAT_KEYS), infos[agent]
                assert infos[agent]["episode_steps"] == step
                assert infos[agent]["episode_ticks"] == step * env.decision_ticks
            else:
                assert present == set(), f"step {step} is not terminal but carries {present}"
        seen_terminal |= done
    assert seen_terminal, "the episode never ended: the test graded nothing"


def test_the_statistics_describe_the_battle_from_each_seat():
    """Own and enemy are swapped between the seats, and tower hp is a real fraction."""
    env = short_env(max_steps=3)
    setup = MatchSetup(decks=[DECK, DECK], tower_hp=[[2400, 900, 1400], [2400, 0, 1400]])
    env.reset(seed=2, options={"setup": setup})
    for _ in range(3):
        _, _, _, _, infos = env.step(dict.fromkeys(AGENTS, 0))
    blue, red = infos["blue"], infos["red"]
    assert blue["own_crowns"] == red["enemy_crowns"]
    assert blue["enemy_crowns"] == red["own_crowns"]
    assert blue["own_tower_hp_frac"] == pytest.approx(red["enemy_tower_hp_frac"])
    state = env.battle_state
    p = state.players[0]
    want = sum(p.tower_hp[s] / p.tower_max_hp[s] for s in TowerSlot) / 3
    assert blue["own_tower_hp_frac"] == pytest.approx(want)
    assert 0.0 < blue["own_tower_hp_frac"] <= 1.0
    # It IS TowerHPReward's potential / 3, so a run's log and its shaping term agree.
    assert blue["own_tower_hp_frac"] == pytest.approx(TowerHPReward()._potential(state, 0) / 3)
    assert blue["enemy_tower_hp_frac"] == pytest.approx(TowerHPReward()._potential(state, 1) / 3)
    # Red started a princess down, so the two seats do NOT see the same board.
    assert blue["own_tower_hp_frac"] != pytest.approx(blue["enemy_tower_hp_frac"])


def test_elixir_leak_steps_count_the_steps_spent_at_a_full_bar():
    """Nobody plays anything, so both bars fill and then every step leaks."""
    env = short_env(max_steps=90)
    setup = MatchSetup(decks=[DECK, DECK], elixir_milli=[10000, 10000])
    env.reset(seed=3, options={"setup": setup})
    for _ in range(90):
        _, _, _, _, infos = env.step(dict.fromkeys(AGENTS, 0))
    for agent in AGENTS:
        leak = infos[agent]["elixir_leak_steps"]
        assert leak == 90, f"{agent} started full and never spent, so every step leaked: {leak}"


def test_episode_statistics_reach_final_info_of_the_self_play_vec_env():
    vec = ClashSelfPlayVecEnv(2, lambda: short_env(max_steps=4), viser=None)
    vec.reset(seed=7)
    for step in range(1, 5):
        _, rewards, _, truncs, infos = vec.step(np.zeros(vec.num_envs, dtype=np.int64))
        assert rewards.dtype == np.float32, "the batch feeds float32 buffers"
        if step < 4:
            assert "final_info" not in infos
            continue
        assert truncs.all()
        assert infos["_final_info"].all()
        # gymnasium recurses into a nested info dict, so ``final_info`` arrives as
        # a dict of BATCHED arrays (one row per agent slot), not a list of dicts.
        final = infos["final_info"]
        assert set(EPISODE_STAT_KEYS) <= set(final), sorted(final)
        for key in EPISODE_STAT_KEYS:
            assert final[f"_{key}"].all(), f"{key} missing for some slot"
            assert final[key].shape == (vec.num_envs,)
        assert (final["episode_steps"] == 4).all()
        assert (final["episode_ticks"] == 4 * vec.envs[0].decision_ticks).all()
    vec.close()


# ---------------------------------------------------------------------------
# 2. config() and build_digest()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reveal", [None, Reveal(enemy_elixir=True), Reveal(enemy_hand=True, enemy_deck=True)]
)
def test_config_is_json_able_and_records_whether_the_policy_was_cheating(reveal):
    env = short_env(obs_builder=SpatialObsBuilder(reveal=reveal))
    cfg = env.config()
    assert json.loads(json.dumps(cfg)) == cfg, "a checkpoint has to be able to write this"
    assert cfg["reveal"] == (reveal or Reveal()).as_dict()
    assert cfg["decision_ms"] == env.decision_ms
    assert cfg["decision_ticks"] == env.decision_ticks
    assert cfg["calibration_digest"] == calibration_digest()
    for key in ("engine", "obs_builder", "action_parser", "reward_fn", "state_mutator"):
        assert cfg[key]["class"].startswith("royalegym."), cfg[key]
    assert cfg["obs_builder"]["params"]["reveal"] == cfg["reveal"]
    assert cfg["action_parser"]["params"]["n_actions"] == int(env.action_parser.space.n)
    assert cfg["state_mutator"]["params"]["decks"] == [DECK, list(reversed(DECK))]
    assert [t["class"] for t in cfg["reward_fn"]["params"]["terms"]] == [
        "WinLossReward",
        "CrownReward",
        "TowerHPReward",
        "ElixirTradeReward",
    ]
    assert cfg["truncation_cond"]["params"] == {"max_steps": 6}


def test_two_configs_differ_exactly_where_the_envs_do():
    def built(**kwargs):
        env = short_env(**kwargs)
        env.reset(seed=1)  # decision_ticks is only known once the engine's clock is
        return env.config()

    a = built()
    b = built(obs_builder=EntityListObsBuilder(max_entities=32), decision_ms=250)
    differ = {k for k in a if a[k] != b[k]}
    assert differ == {"decision_ms", "decision_ticks", "obs_builder"}
    assert b["obs_builder"]["params"]["max_entities"] == 32


def test_calibration_digest_moves_with_a_value_and_not_with_prose():
    from royalegym.protocol import default_calibration

    cal = default_calibration()
    assert calibration_digest(cal) == calibration_digest(cal)
    assert len(calibration_digest(cal)) == 16
    moved = cal.with_override("match.MAX_MANA", 11)
    assert calibration_digest(moved) != calibration_digest(cal)


@pytest.mark.skipif(not core_available(), reason="the compiled engine is not importable")
def test_build_digest_reads_the_data_the_extension_was_compiled_with():
    """It is deliberately callable WITHOUT constructing an engine.

    Constructing one is exactly what refuses on a stale build, and a stale build is
    when a checkpoint most needs to record which data it actually ran on.
    """
    from royalegym.rust_engine import build_digest

    first = build_digest()
    assert first == build_digest()
    assert len(first) == 16
    assert set(first) <= set("0123456789abcdef")


# ---------------------------------------------------------------------------
# 3. the reward breakdown, per team
# ---------------------------------------------------------------------------


def _damaged_pair():
    """(prev, state) where Blue lost tower hp and Red lost more, so the two seats'
    breakdowns are DIFFERENT numbers and a lost one cannot hide as a zero."""
    eng = MockEngine()
    eng.reset(0, MatchSetup(decks=[DECK, DECK]))
    prev = eng.state()
    hit = [
        msgspec.structs.replace(prev.players[0], tower_hp=[2400, 700, 1400]),
        msgspec.structs.replace(prev.players[1], tower_hp=[2400, 100, 1400]),
    ]
    return eng, prev, msgspec.structs.replace(prev, players=hit)


def test_the_reward_breakdown_survives_being_asked_for_both_seats():
    reward = CombinedReward([(WinLossReward(), 1.0), (CrownReward(), 0.2), (TowerHPReward(), 5.0)])
    eng, prev, state = _damaged_pair()
    reward.bind(eng)
    reward.reset(prev)
    assert reward.last_terms == {}, "reset clears the breakdowns with the episode"
    scalars = {team: reward.get_reward(team, prev, state, []) for team in (0, 1)}
    assert set(reward.last_terms) == {0, 1}, "one breakdown per seat"
    for team in (0, 1):
        terms = reward.terms_for(team)
        assert set(terms) == {"WinLossReward", "CrownReward", "TowerHPReward"}
        assert sum(terms.values()) == pytest.approx(scalars[team])
    # The two seats' breakdowns are not the same numbers, so "keyed by team" is
    # doing work rather than storing the same dict twice.
    blue, red = reward.terms_for(0), reward.terms_for(1)
    assert blue["TowerHPReward"] == pytest.approx(-red["TowerHPReward"])
    assert blue["TowerHPReward"] != 0.0
    assert blue != red


def test_the_env_leaves_a_breakdown_for_each_seat_after_a_step():
    reward = CombinedReward([(WinLossReward(), 1.0), (TowerHPReward(), 1.0)])
    env = short_env(max_steps=3, reward_fn=reward)
    env.reset(seed=4)
    env.step(dict.fromkeys(AGENTS, 0))
    assert set(reward.last_terms) == {0, 1}


def test_plant_one_shared_breakdown_loses_the_first_seat(monkeypatch):
    """The shipped bug: a flat dict cleared at the top of every ``get_reward``."""

    def flat(self, team, prev, state, results):
        total = 0.0
        self.last_terms = {}
        for t, w in self.terms:
            v = w * t.get_reward(team, prev, state, results)
            self.last_terms[type(t).__name__] = v
            total += v
        return total

    monkeypatch.setattr(CombinedReward, "get_reward", flat)
    reward = CombinedReward([(TowerHPReward(), 5.0)])
    eng, prev, state = _damaged_pair()
    reward.bind(eng)
    for team in (0, 1):
        reward.get_reward(team, prev, state, [])
    assert set(reward.last_terms) != {0, 1}, "PLANT DID NOT LAND: the flat dict kept both seats"
    # and what survives is only the SECOND seat's number, which is the defect
    assert reward.last_terms["TowerHPReward"] == pytest.approx(
        5.0 * TowerHPReward()._potential(state, 1)
        - 5.0 * TowerHPReward()._potential(prev, 1)
        - (5.0 * TowerHPReward()._potential(state, 0) - 5.0 * TowerHPReward()._potential(prev, 0))
    )


# ---------------------------------------------------------------------------
# 4. one viewer, one port
# ---------------------------------------------------------------------------


def test_a_bare_parallel_env_never_reads_the_viewer_variable(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "127.0.0.1:0")
    env = short_env()
    assert env.viser is None, "the env must publish only when it is HANDED a publisher"
    env.close()


def test_many_games_bind_one_publisher_and_only_game_zero_gets_it(monkeypatch):
    """The OSError 10048 regression: N games used to bind the same fixed UDP port."""
    monkeypatch.setenv(ENV_VAR, "127.0.0.1:0")
    vec = ClashSelfPlayVecEnv(3, lambda: short_env(max_steps=3))
    try:
        assert vec.viser is not None, "the default reads the variable"
        assert vec.envs[0].viser is vec.viser
        assert [e.viser for e in vec.envs[1:]] == [None, None]
        bound = {id(e.viser) for e in vec.envs if e.viser is not None}
        assert len(bound) == 1
        vec.reset(seed=1)
        vec.step(np.zeros(vec.num_envs, dtype=np.int64))  # nobody attached: nothing sent
        assert vec.viser.sent == 0
    finally:
        vec.close()


def test_the_publisher_can_be_given_or_refused_explicitly():
    pub = ViserPublisher(port=0)
    vec = ClashSelfPlayVecEnv(2, lambda: short_env(max_steps=3), viser=pub)
    assert vec.envs[0].viser is pub
    vec.close()
    off = ClashSelfPlayVecEnv(2, lambda: short_env(max_steps=3), viser=None)
    assert all(e.viser is None for e in off.envs)
    off.close()


# ---------------------------------------------------------------------------
# 5. pickling: the env, and the recipe a worker gets instead
# ---------------------------------------------------------------------------


def test_an_env_on_the_python_engine_now_pickles():
    """The msgspec codecs were the only unpicklable part of MockEngine.

    They are a compiled schema with no state, so dropping and rebuilding them is
    exact -- checked here by round-tripping a battle and comparing state hashes.
    """
    env = short_env()
    env.reset(seed=5)
    env.step(dict.fromkeys(AGENTS, 0))
    clone = pickle.loads(pickle.dumps(env))
    assert clone.engine.state_hash() == env.engine.state_hash()
    assert clone.config() == env.config()
    acts = dict.fromkeys(AGENTS, 0)
    a = env.step(acts)
    b = clone.step(acts)
    assert env.engine.state_hash() == clone.engine.state_hash()
    assert a[1] == b[1]


def test_env_factory_pickles_builds_and_refuses_what_a_worker_could_not_receive():
    factory = EnvFactory(
        engine=MockEngine,
        obs_builder=(SpatialObsBuilder, {"reveal": Reveal(enemy_elixir=True)}),
        state_mutator=(DefaultStateMutator, {"decks": [DECK, DECK]}),
        truncation_cond=(StepLimitCondition, {"max_steps": 4}),
        decision_ms=250,
    )
    revived = pickle.loads(pickle.dumps(factory))
    env = revived()
    assert isinstance(env, ClashParallelEnv)
    assert env.decision_ms == 250
    assert env.config()["reveal"]["enemy_elixir"] is True
    assert env.config()["truncation_cond"]["params"] == {"max_steps": 4}
    obs, _ = env.reset(seed=1)
    assert env.observation_space("blue").contains(obs["blue"])
    assert json.loads(json.dumps(factory.config())) == factory.config()
    # A lambda fails HERE, naming the key, rather than inside multiprocessing.
    with pytest.raises(TypeError, match="obs_builder"):
        EnvFactory(engine=MockEngine, obs_builder=lambda: SpatialObsBuilder())


def test_env_factory_feeds_the_self_play_vec_env():
    factory = EnvFactory(
        engine=MockEngine,
        state_mutator=(DefaultStateMutator, {"decks": [DECK, DECK]}),
        truncation_cond=(StepLimitCondition, {"max_steps": 3}),
    )
    vec = ClashSelfPlayVecEnv(2, factory, viser=None)
    obs, _ = vec.reset(seed=2)
    assert vec.observation_space.contains(obs)
    vec.close()
