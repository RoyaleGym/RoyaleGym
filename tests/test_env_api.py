"""Third-party API conformance, run against the INSTALLED gymnasium and pettingzoo.

WHY THIS FILE EXISTS
    A trainer (sb3-contrib, CleanRL, RLlib, ...) meets this package only through
    the gymnasium / pettingzoo APIs. An env that violates them trains anyway and
    fails in ways that look like bad hyperparameters: a seeded reset that is not
    reproducible makes every A/B comparison noise (a delta inside the seed-noise band is
    not a result), and a space object that is copied per call silently un-seeds action
    sampling.

WHAT IT RUNS
    * ``gymnasium.utils.env_checker.check_env`` on ``ClashGymEnv`` built through
      ``gymnasium.make`` (so the spec exists and the render-mode and close checks
      run too), with a no-op and with a stochastic opponent.
    * ``pettingzoo.test.parallel_api_test`` over a WHOLE game, plus pettingzoo's
      ``parallel_seed_test`` and its parallel ``state_test`` half.
    * Our own seeding test: ``reset(seed=s)`` twice gives identical observation,
      reward, termination, deploy-status and engine-hash sequences under an
      identical action script -- in a fresh env and in a used one.
    * The sb3-contrib MaskablePPO mask convention (see the test's docstring).

    Warnings are FAILURES here. Both checkers report several defects only through
    ``warnings.warn`` (pettingzoo: "Live agent was not given info"), which pytest
    would print in a summary nobody reads. A plant below proves it.

WHAT IT CANNOT CATCH
    Whether the observation is a GOOD one, or whether the mask is right (that is
    tests/test_env_action_mask.py and tests/test_env_mask_property.py). pettingzoo's
    ``parallel_seed_test`` compares only ONE step (see its test); the multi-step
    seeding guarantee rests on this file's own replay test.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from pettingzoo.test import parallel_api_test, parallel_seed_test
from pettingzoo.test.state_test import test_parallel_env as pettingzoo_state_check

import royalegym
from royalegym.action import NOOP
from royalegym.done_condition import (
    AnyCondition,
    DoneCondition,
    GameOverCondition,
    StepLimitCondition,
    TerminationCondition,
    TickLimitCondition,
    TruncationCondition,
)
from royalegym.env import ClashGymEnv, ClashParallelEnv, ClashSelfPlayVecEnv, make_gym_vec_env
from royalegym.mock_engine import MockEngine
from royalegym.obs import vector_offsets
from royalegym.protocol import DeployStatus, MatchSetup, ShuffleMode
from royalegym.selfplay import NoopOpponent, RandomLegalOpponent
from royalegym.state_mutator import StateMutator


def with_warnings(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, list[str]]:
    """Run ``fn`` and return (result, every warning it raised, as text)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, [f"{w.category.__name__}: {w.message}" for w in caught]


# --------------------------------------------------------------------------
# gymnasium
# --------------------------------------------------------------------------


def make_gym(**kwargs: Any) -> ClashGymEnv:
    # engine NAMED, not defaulted. These tests are about API conformance, and the
    # default id warns when nobody chose an engine -- deliberately, since it decides
    # which engine a result came from. check_env re-makes the env from its spec, so the
    # warning would fire inside the no-warnings assertion and read as a conformance
    # failure. Same engine either way; the only difference is that this says so.
    kwargs.setdefault("engine", MockEngine())
    env = gym.make(royalegym.GYM_ENV_ID, **kwargs).unwrapped
    assert isinstance(env, ClashGymEnv)
    assert env.spec is not None  # so check_env also runs the render-mode and close checks
    return env


@pytest.mark.parametrize(
    "opponent", [NoopOpponent(), RandomLegalOpponent(noop_prob=0.0)], ids=["noop", "random"]
)
def test_gymnasium_check_env_passes_without_warnings(opponent):
    env = make_gym(opponent=opponent, render_mode="ansi")
    _, caught = with_warnings(check_env, env)
    assert caught == []


class DriftingDecks(StateMutator):
    """PLANT: episode setup drifts with a hidden call counter instead of the seeded rng.

    Deck k is card ids [k, k+8), so decks from different calls differ element-wise
    and stay different after the engine's (seeded, content-blind) shuffle.
    """

    def __init__(self) -> None:
        self.calls = 0

    def build(self, rng, cards):
        del rng
        self.calls += 1
        deck = [(self.calls + i) % len(cards) for i in range(8)]
        return MatchSetup(decks=[deck, deck], shuffle=int(ShuffleMode.INDEPENDENT))


def test_plant_unseeded_episode_setup_is_caught_by_check_env():
    env = make_gym(state_mutator=DriftingDecks())
    with pytest.raises(AssertionError, match=r"reset\(seed=123\)` is non-deterministic"):
        check_env(env)
    # The plant really ran: the mutator was consulted for more than one episode.
    assert env.parallel.state_mutator.calls >= 2


class CountingOpponent:
    """PLANT: an opponent whose choice depends on a hidden counter, not on env.np_random."""

    def __init__(self) -> None:
        self.calls = 0

    def act(self, obs, mask, rng):
        del obs, rng
        self.calls += 1
        legal = np.flatnonzero(mask)
        legal = legal[legal != NOOP]
        return int(legal[self.calls % legal.size]) if legal.size else NOOP


def test_plant_opponent_with_hidden_state_is_caught_by_check_env():
    env = make_gym(opponent=CountingOpponent())
    with pytest.raises(AssertionError, match="Deterministic step observations are not equivalent"):
        check_env(env)
    assert env.opponent.calls >= 2


# --------------------------------------------------------------------------
# pettingzoo
# --------------------------------------------------------------------------


def test_pettingzoo_parallel_api_test_over_whole_games_without_warnings(capsys):
    env = ClashParallelEnv()
    _, caught = with_warnings(parallel_api_test, env, num_cycles=1000)
    assert caught == []
    assert "Passed Parallel API test" in capsys.readouterr().out
    # Not vacuous about episode ends: the last game really finished inside the
    # cycle budget, so the terminate/agents-removed path was exercised.
    assert env.battle_state.game_over
    assert env.agents == []


def test_plant_space_copied_per_call_is_caught_by_parallel_api_test():
    class CopiesSpaces(ClashParallelEnv):
        def observation_space(self, agent):
            return copy.deepcopy(super().observation_space(agent))

    env = CopiesSpaces(termination_cond=GameOverCondition(), truncation_cond=StepLimitCondition(3))
    assert env.observation_space("blue") is not env.observation_space("blue")
    with pytest.raises(AssertionError, match="exact same space object"):
        parallel_api_test(env, num_cycles=5)


def test_plant_missing_info_key_is_only_a_warning_so_warnings_must_fail():
    class DropsRedInfo(ClashParallelEnv):
        def step(self, actions):
            obs, rew, term, trunc, info = super().step(actions)
            info.pop("red", None)
            return obs, rew, term, trunc, info

    env = DropsRedInfo(termination_cond=GameOverCondition(), truncation_cond=StepLimitCondition(3))
    _, caught = with_warnings(parallel_api_test, env, num_cycles=5)
    assert any("Live agent was not given info" in w for w in caught), caught


def test_done_conditions_feed_gymnasium_flags_by_role():
    """terminated comes from termination_cond, truncated from truncation_cond, exclusively."""
    env = ClashParallelEnv(
        termination_cond=GameOverCondition(), truncation_cond=StepLimitCondition(3)
    )
    env.reset(seed=0)
    flags = []
    while env.agents:
        _, _, term, trunc, _ = env.step({a: NOOP for a in env.agents})
        flags.append((term["blue"], trunc["blue"]))
    assert flags == [(False, False), (False, False), (False, True)]

    class EndsNow(DoneCondition):
        def is_done(self, state):
            return True

    # Both fire on the same step: a real ending dominates the cut.
    env = ClashParallelEnv(termination_cond=EndsNow(), truncation_cond=StepLimitCondition(1))
    env.reset(seed=0)
    _, _, term, trunc, _ = env.step({a: NOOP for a in env.agents})
    assert (term["blue"], trunc["blue"]) == (True, False)

    # No truncation condition: the flag is never set.
    env = ClashParallelEnv(termination_cond=EndsNow())
    env.reset(seed=0)
    assert env.step({a: NOOP for a in env.agents})[3] == {"blue": False, "red": False}


def test_done_condition_roles_are_declared_and_enforced():
    assert issubclass(GameOverCondition, TerminationCondition)
    assert issubclass(StepLimitCondition, TruncationCondition)
    assert issubclass(TickLimitCondition, TruncationCondition)
    with pytest.raises(TypeError, match="truncation, not a termination"):
        ClashParallelEnv(termination_cond=StepLimitCondition(3))
    with pytest.raises(TypeError, match="termination, not a truncation"):
        ClashParallelEnv(truncation_cond=GameOverCondition())
    # AnyCondition is role-neutral, so it is accepted in either slot.
    env = ClashParallelEnv(
        termination_cond=AnyCondition([GameOverCondition()]),
        truncation_cond=AnyCondition([StepLimitCondition(2), TickLimitCondition(10**9)]),
    )
    env.reset(seed=0)
    env.step({a: NOOP for a in env.agents})
    assert env.step({a: NOOP for a in env.agents})[3]["blue"] is True


def test_names_from_before_the_rename_still_work():
    assert royalegym.StateSetter is royalegym.StateMutator
    assert royalegym.DefaultStateSetter is royalegym.DefaultStateMutator
    assert royalegym.TerminalCondition is royalegym.DoneCondition
    # The old flat list is split by declared role: StepLimit truncates, GameOver terminates.
    env = ClashParallelEnv(
        terminal_conditions=[GameOverCondition(), StepLimitCondition(2)],
        state_setter=royalegym.DefaultStateSetter(),
    )
    assert isinstance(env.termination, GameOverCondition)
    assert isinstance(env.truncation, StepLimitCondition)
    env.reset(seed=0)
    env.step({a: NOOP for a in env.agents})
    _, _, term, trunc, _ = env.step({a: NOOP for a in env.agents})
    assert (term["blue"], trunc["blue"]) == (False, True)
    with pytest.raises(TypeError):
        ClashParallelEnv(
            terminal_conditions=[GameOverCondition()], termination_cond=GameOverCondition()
        )


def test_pettingzoo_parallel_seed_test_passes():
    """Installed pettingzoo 1.27.0 breaks its loop on ``any(terminations1)``, which
    is ``any`` over the dict's KEYS and so always true: it compares exactly one
    step. Measured 2026-09-13 by counting: 2 ``step()`` calls in total, one per env.
    Our replay test is what covers many steps."""
    steps: list[int] = []

    class Counting(ClashParallelEnv):
        def step(self, actions):
            steps.append(1)
            return super().step(actions)

    _, caught = with_warnings(parallel_seed_test, Counting, num_cycles=50)
    assert caught == []
    assert len(steps) >= 2  # at least one step on each of the two envs


def test_pettingzoo_state_check_and_state_space():
    env = ClashParallelEnv()
    _, caught = with_warnings(pettingzoo_state_check, env, seed=0)
    assert caught == []
    obs, _ = env.reset(seed=1)
    s = env.state()
    assert s.shape == env.state_space.shape
    spatial = obs["blue"]["spatial"].ravel()
    assert np.array_equal(s[: spatial.size], spatial)
    assert np.array_equal(s[spatial.size :], obs["blue"]["vector"])


# --------------------------------------------------------------------------
# seeding: identical sequences under identical actions
# --------------------------------------------------------------------------


def _frame(env: ClashParallelEnv, obs, rew=None, term=None, trunc=None, info=None) -> dict:
    return {
        "obs": {a: {k: np.array(v, copy=True) for k, v in o.items()} for a, o in obs.items()},
        "rew": rew,
        "term": term,
        "trunc": trunc,
        "status": None if info is None else {a: i["deploy_status"] for a, i in info.items()},
        "hash": env.engine.state_hash(),
        "tick": env.battle_state.tick,
    }


def parallel_rollout(
    env: ClashParallelEnv,
    seed: int,
    episodes: int,
    script: list[dict[str, int]] | None = None,
    policy_seed: int = 0,
) -> tuple[list[dict[str, int]], list[dict]]:
    """Episode 1 starts from ``reset(seed=seed)``, later ones from ``reset()``.

    With ``script=None`` actions are drawn from the masks by a separate rng and
    returned, so a second run can replay the SAME actions whether or not its
    observations match (the comparison must not depend on what it tests).
    """
    rng = np.random.default_rng(policy_seed)
    chosen: list[dict[str, int]] = []
    frames: list[dict] = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed if ep == 0 else None)
        frames.append(_frame(env, obs, info=info))
        while env.agents:
            if script is None:
                act = {}
                for a in env.agents:
                    legal = np.flatnonzero(obs[a]["action_mask"])
                    act[a] = int(rng.choice(legal)) if rng.random() < 0.3 else NOOP
            else:
                act = script[len(chosen)]
            chosen.append(act)
            obs, rew, term, trunc, info = env.step(act)
            frames.append(_frame(env, obs, rew, term, trunc, info))
    return chosen, frames


def first_divergence(a: list[dict], b: list[dict]) -> str | None:
    if len(a) != len(b):
        return f"lengths differ: {len(a)} != {len(b)}"
    for i, (fa, fb) in enumerate(zip(a, b, strict=True)):
        for key in ("rew", "term", "trunc", "status", "hash", "tick"):
            if fa[key] != fb[key]:
                return f"frame {i}: {key} {fa[key]} != {fb[key]}"
        if fa["obs"].keys() != fb["obs"].keys():
            return f"frame {i}: agents differ"
        for agent, oa in fa["obs"].items():
            for k, v in oa.items():
                if not np.array_equal(v, fb["obs"][agent][k]):
                    return f"frame {i}: obs[{agent}][{k}]"
    return None


def short_parallel_env() -> ClashParallelEnv:
    return ClashParallelEnv(
        termination_cond=GameOverCondition(), truncation_cond=StepLimitCondition(60)
    )


def test_parallel_env_seeded_reset_replays_identically():
    env = short_parallel_env()
    script, want = parallel_rollout(env, seed=7, episodes=2)
    accepted = sum(
        1 for f in want if f["status"] for s in f["status"].values() if s == DeployStatus.OK
    )
    assert accepted >= 10, "too few deploys for the replay to be evidence"

    # A fresh env, and the same env after an unrelated episode, must both replay it.
    _, fresh = parallel_rollout(short_parallel_env(), seed=7, episodes=2, script=script)
    assert first_divergence(want, fresh) is None
    parallel_rollout(env, seed=99, episodes=1)
    _, reused = parallel_rollout(env, seed=7, episodes=2, script=script)
    assert first_divergence(want, reused) is None

    # Not vacuous: the seed is what made them equal.
    _, other = parallel_rollout(short_parallel_env(), seed=8, episodes=2, script=script)
    assert first_divergence(want, other) is not None


def gym_rollout(env: ClashGymEnv, seed: int, script: list[int] | None = None, n: int = 80):
    rng = np.random.default_rng(1)
    obs, _ = env.reset(seed=seed)
    frames = [{k: v.copy() for k, v in obs.items()}]
    chosen: list[int] = []
    for t in range(n):
        if script is None:
            legal = np.flatnonzero(env.action_masks())
            action = int(rng.choice(legal)) if rng.random() < 0.3 else NOOP
        else:
            action = script[t]
        chosen.append(action)
        obs, rew, term, trunc, info = env.step(action)
        frames.append({k: v.copy() for k, v in obs.items()})
        frames.append(
            {
                "rew": rew,
                "term": term,
                "trunc": trunc,
                "opp": info["opponent_deploy_status"],
                "hash": env.parallel.engine.state_hash(),
            }
        )
        if term or trunc:
            break
    return chosen, frames


def gym_divergence(a: list[dict], b: list[dict]) -> str | None:
    if len(a) != len(b):
        return f"lengths differ: {len(a)} != {len(b)}"
    for i, (fa, fb) in enumerate(zip(a, b, strict=True)):
        for k, v in fa.items():
            same = np.array_equal(v, fb[k]) if isinstance(v, np.ndarray) else v == fb[k]
            if not same:
                return f"frame {i}: {k}"
    return None


def test_gym_env_with_stochastic_opponent_replays_identically():
    """The opponent draws from ``env.np_random``, so the seed must fix it too."""
    env = ClashGymEnv(opponent=RandomLegalOpponent(noop_prob=0.5))
    script, want = gym_rollout(env, seed=3)
    opp_deploys = sum(1 for f in want if f.get("opp") == DeployStatus.OK)
    assert opp_deploys >= 5, "the opponent must actually act for this to test its rng"
    _, again = gym_rollout(env, seed=3, script=script)
    assert gym_divergence(want, again) is None
    _, other = gym_rollout(env, seed=4, script=script)
    assert gym_divergence(want, other) is not None


def test_plant_opponent_with_hidden_state_breaks_the_replay():
    env = ClashGymEnv(opponent=CountingOpponent())
    script, want = gym_rollout(env, seed=3)
    _, again = gym_rollout(env, seed=3, script=script)
    assert gym_divergence(want, again) is not None


# --------------------------------------------------------------------------
# vector envs
# --------------------------------------------------------------------------


def _vector_offset(key: str) -> int:
    """Where a named field starts. The layout is self-describing, so no test counts slots."""
    n_cards = len(ClashParallelEnv().engine.cards())
    return vector_offsets(n_cards)[key].start


REGULATION_LEFT = _vector_offset("clock")  # first slot of the clock block


def test_selfplay_vec_env_is_seeded_masked_and_autoresets_same_step():
    def env_fn() -> ClashParallelEnv:
        return ClashParallelEnv(
            termination_cond=GameOverCondition(), truncation_cond=StepLimitCondition(5)
        )

    runs = []
    for _ in range(2):
        vec = ClashSelfPlayVecEnv(2, env_fn)
        rng = np.random.default_rng(0)
        obs, _ = vec.reset(seed=5)
        seq = [obs]
        for t in range(12):
            masks = vec.action_masks()
            assert masks.dtype == np.bool_
            assert masks.shape == (vec.num_envs, vec.single_action_space.n)
            assert np.array_equal(masks, obs["action_mask"].astype(bool))
            assert vec.observation_space.contains(obs)
            acts = np.array([int(rng.choice(np.flatnonzero(m))) for m in masks])
            obs, _, term, trunc, info = vec.step(acts)
            seq.append(obs)
            ended = (t + 1) % 5 == 0
            assert bool(trunc.all()) == ended
            assert not term.any()
            if ended:
                assert info["_final_obs"].all()
                for final in info["final_obs"]:
                    assert vec.single_observation_space.contains(final)
                # SAME_STEP: the returned obs is already the next game's first one.
                assert (obs["vector"][:, REGULATION_LEFT] == 1.0).all()
        runs.append(seq)
    for a, b in zip(*runs, strict=True):
        for k in a:
            assert np.array_equal(a[k], b[k])


def test_maskable_ppo_mask_convention():
    """What sb3-contrib MaskablePPO requires, and that both env shapes satisfy it.

    sb3-contrib is NOT installed here and was not installed for this test. The
    convention was read from its source on GitHub (Stable-Baselines-Team/
    stable-baselines3-contrib, master, read 2026-09-13):

    * ``common/maskable/utils.py``: the method name is ``action_masks``; for a
      VecEnv ``get_action_masks`` is ``np.stack(env.env_method("action_masks"))``,
      for a bare env ``env.get_wrapper_attr("action_masks")()``.
    * ``common/maskable/buffers.py``: for ``Discrete`` ``mask_dims = n``; masks are
      stored as ``action_masks.reshape((n_envs, mask_dims))`` in a float32 buffer.
    * ``common/maskable/distributions.py``: ``th.as_tensor(masks, dtype=th.bool)
      .reshape(logits.shape)`` then ``th.where(masks, logits, -1e8)`` -- True means
      LEGAL, and any truthy dtype works, but bool is the canonical one.

    So: one 1-D array of length ``action_space.n`` per env, True = legal, and it
    must describe the observation the policy was just given.
    """
    # Bare env through gymnasium's wrapper chain, as sb3's DummyVecEnv reaches it.
    wrapped = gym.make(
        royalegym.GYM_ENV_ID, engine=MockEngine(), opponent=RandomLegalOpponent(0.5)
    )
    obs, _ = wrapped.reset(seed=0)
    n = wrapped.action_space.n
    for _ in range(20):
        mask = wrapped.get_wrapper_attr("action_masks")()
        assert mask.dtype == np.bool_
        assert mask.shape == (n,)
        assert np.array_equal(mask, obs["action_mask"].astype(bool))
        # The buffer's float32 round trip and the distribution's bool cast are lossless.
        stored = mask.reshape((1, n)).astype(np.float32)
        assert np.array_equal(stored.astype(bool).reshape(n), mask)
        assert mask[NOOP]  # never an all-False row: -1e8 everywhere would be a uniform draw
        legal = np.flatnonzero(mask)
        obs, *_ = wrapped.step(int(legal[-1]))
    wrapped.close()

    # A stock gymnasium vector env stacked exactly the way get_action_masks stacks.
    vec = make_gym_vec_env(3, opponent=RandomLegalOpponent(0.5))
    obs, _ = vec.reset(seed=0)
    stacked = np.stack(vec.call("action_masks"))
    assert stacked.shape == (3, n)
    assert stacked.dtype == np.bool_
    assert np.array_equal(stacked, obs["action_mask"].astype(bool))
    vec.close()
