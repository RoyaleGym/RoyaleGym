"""Environments: PettingZoo ParallelEnv, Gymnasium single-agent wrapper, vectorised self-play.

DECOMPOSITION (after RLGym)
    ObsBuilder, ActionParser, RewardFunction, TerminalCondition and StateSetter
    are constructor arguments. Changing a reward or an observation is a Python
    edit and never a recompile of the simulator, because that is where most
    research iteration happens.

TIMING
    One env step = one decision = ``decision_ms`` of game time, rounded UP to a
    whole number of engine ticks (engine tick length comes from the engine's
    state, which reads calibration.json). Both players act simultaneously; the
    engine validates both commands against the same pre-step state.

ACTION MASKS
    Every observation dict carries ``action_mask`` (int8, as pettingzoo's
    ``parallel_api_test`` and gymnasium ``Discrete.sample(mask=...)`` expect) and
    every info dict carries it too. ``action_masks()`` returns the bool array
    sb3-contrib's MaskablePPO calls for. An action the engine rejects is turned
    into a no-op and reported in ``info["deploy_status"]``.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space, concatenate, create_empty_array
from pettingzoo import ParallelEnv

from .action import NOOP, ActionParser, TileActionParser
from .mock_engine import MockEngine
from .obs import ObsBuilder, SpatialObsBuilder
from .protocol import (
    BLUE,
    RED,
    BattleState,
    DeployCommand,
    DeployResult,
    DeployStatus,
    Engine,
    Winner,
)
from .replay import ReplayRecorder
from .reward import RewardFunction, default_reward
from .selfplay import NoopOpponent, Opponent
from .state_setter import DefaultStateSetter, Snapshot, StateSetter
from .terminal import AnyCondition, GameOverCondition, TerminalCondition
from .viser import ViserPublisher, play_event

AGENTS = ("blue", "red")
AGENT_TEAM = {"blue": BLUE, "red": RED}
NO_COMMAND = -1  # info["deploy_status"] when the agent chose no-op


class ClashParallelEnv(ParallelEnv[str, dict[str, np.ndarray], int]):
    metadata: ClassVar[dict[str, Any]] = {
        "name": "clash_royale_v0",
        "render_modes": ["ansi"],
        "is_parallelizable": True,
    }

    def __init__(
        self,
        engine: Engine | None = None,
        *,
        obs_builder: ObsBuilder | None = None,
        action_parser: ActionParser | None = None,
        reward_fn: RewardFunction | None = None,
        terminal_conditions: Sequence[TerminalCondition] | None = None,
        state_setter: StateSetter | None = None,
        decision_ms: int = 500,
        render_mode: str | None = None,
        recorder: ReplayRecorder | None = None,
        viser: ViserPublisher | None = None,
    ) -> None:
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode {render_mode!r} not supported")
        self.engine: Engine = engine if engine is not None else MockEngine()
        self.action_parser = action_parser or TileActionParser()
        self.action_parser.bind(self.engine)
        self.obs_builder = obs_builder or SpatialObsBuilder()
        self.obs_builder.bind(self.engine, self.action_parser)
        self.reward_fn = reward_fn or default_reward()
        self.reward_fn.bind(self.engine)
        self.terminal = AnyCondition(list(terminal_conditions or [GameOverCondition()]))
        self.state_setter = state_setter or DefaultStateSetter()
        self.decision_ms = decision_ms
        self.render_mode = render_mode
        self.recorder = recorder
        # RoyaleViser (viser.py): one publish per reset/step, and only while a viewer is
        # attached; None (the default, unless ROYALEVISER=host:port is set) costs one ``if``.
        self.viser = viser if viser is not None else ViserPublisher.from_env()
        self._decks: list[list[int]] | None = None

        self.possible_agents = list(AGENTS)
        self.agents: list[str] = []
        base_obs = self.obs_builder.observation_space()
        base_act = self.action_parser.space
        # One object per agent, reused forever: pettingzoo requires identity, and
        # separate objects keep seeding one agent's space from affecting the other.
        self._obs_spaces = {a: copy.deepcopy(base_obs) for a in AGENTS}
        self._act_spaces = {a: copy.deepcopy(base_act) for a in AGENTS}
        self._state_keys, self.state_space = _state_layout(base_obs)
        self._np_random, self._np_random_seed = seeding.np_random(None)
        self._state: BattleState | None = None
        self._masks: dict[str, np.ndarray] = {}
        self._obs: dict[str, dict[str, np.ndarray]] = {}
        self.decision_ticks = 1
        self.last_results: list[DeployResult] = []

    # -- spaces -------------------------------------------------------------

    def observation_space(self, agent: str) -> gym.spaces.Space[Any]:
        return self._obs_spaces[agent]

    def action_space(self, agent: str) -> gym.spaces.Space[Any]:
        return self._act_spaces[agent]

    @property
    def np_random(self) -> np.random.Generator:
        return self._np_random

    @property
    def battle_state(self) -> BattleState:
        if self._state is None:
            raise RuntimeError("call reset() first")
        return self._state

    # -- core loop ------------------------------------------------------------

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
        if seed is not None:
            self._np_random, self._np_random_seed = seeding.np_random(seed)
        options = options or {}
        cards = self.engine.cards()
        init: Any = options.get("setup") or options.get("snapshot")
        if init is None:
            init = self.state_setter.build(self._np_random, cards)
        elif isinstance(init, bytes):
            init = Snapshot(init)
        engine_seed = int(self._np_random.integers(0, 2**63 - 1))
        if isinstance(init, Snapshot):
            self.engine.load_state(init.blob)
        else:
            self.engine.reset(engine_seed, init)
        state = self.engine.state()
        self._state = state
        self.decision_ticks = max(1, -(-self.decision_ms // state.tick_ms))
        self.obs_builder.reset(state)
        self.reward_fn.reset(state)
        self.terminal.reset(state)
        self.agents = list(self.possible_agents)
        self.last_results = []
        if self.recorder is not None:
            self.recorder.begin(self.engine, engine_seed, init)
        if self.viser is not None:
            self._decks = None if isinstance(init, Snapshot) else [list(d) for d in init.decks]
            self.viser.publish(state, cards, self.engine.arena(), self._decks)
        self._refresh(state)
        infos = {a: self._info(a, state, NO_COMMAND) for a in self.agents}
        return dict(self._obs), infos

    def step(
        self, actions: dict[str, Any]
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        if not self.agents:
            raise RuntimeError("step() called on a finished episode; call reset()")
        prev = self.battle_state
        commands: list[DeployCommand] = []
        owner: list[str] = []
        for agent in self.agents:
            cmd = self.action_parser.parse(int(actions.get(agent, NOOP)), prev, AGENT_TEAM[agent])
            if cmd is not None:
                commands.append(cmd)
                owner.append(agent)
        results = self._advance(commands)
        state = self.engine.state()
        self._state = state
        self.last_results = results
        terminated, truncated = self.terminal.check(state)
        status = {a: NO_COMMAND for a in self.agents}
        for agent, res in zip(owner, results, strict=True):
            status[agent] = res.status
        rewards = {
            a: float(self.reward_fn.get_reward(AGENT_TEAM[a], prev, state, results))
            for a in self.agents
        }
        self._refresh(state)
        live = list(self.agents)
        obs = {a: self._obs[a] for a in live}
        infos = {a: self._info(a, state, status[a]) for a in live}
        terms = dict.fromkeys(live, terminated)
        truncs = dict.fromkeys(live, truncated)
        if terminated or truncated:
            self.agents = []
            if self.recorder is not None:
                self.recorder.end(self.engine)
        if self.viser is not None:
            self._publish(state, commands, results)
        return obs, rewards, terms, truncs, infos

    def _publish(
        self, state: BattleState, commands: list[DeployCommand], results: list[DeployResult]
    ) -> None:
        """The frame after a step to the viewer, with the accepted deploys as event lines."""
        if not self.viser.attached:
            return
        cards = self.engine.cards()
        subtile = self.engine.arena().subtile
        names = {c.card_id: c.name for c in cards}
        events = [
            play_event(
                res.tick, cmd.team, names.get(res.card_id, f"#{res.card_id}"), cmd.x, cmd.y, subtile
            )
            for cmd, res in zip(commands, results, strict=True)
            if res.status == DeployStatus.OK
        ]
        self.viser.publish(state, cards, self.engine.arena(), self._decks, events)

    def _advance(self, commands: list[DeployCommand]) -> list[DeployResult]:
        tick_before = self.battle_state.tick
        rec = self.recorder
        if rec is None or not rec.frame_every_tick:
            results = self.engine.step(commands, self.decision_ticks)
            if rec is not None:
                rec.record_frame(self.engine)
        else:
            # Tick-by-tick is the same battle as one multi-tick call (the protocol
            # validates commands up front); tests check the hashes agree.
            results = self.engine.step(commands, 1)
            rec.record_frame(self.engine)
            for _ in range(self.decision_ticks - 1):
                if self.engine.state().game_over:
                    break
                self.engine.step([], 1)
                rec.record_frame(self.engine)
        if rec is not None:
            rec.record_step(tick_before, self.decision_ticks, commands, results)
        return results

    def _refresh(self, state: BattleState) -> None:
        self._masks = {}
        self._obs = {}
        for agent in AGENTS:
            team = AGENT_TEAM[agent]
            mask = self.action_parser.action_mask(state, team)
            self._masks[agent] = mask
            self._obs[agent] = self.obs_builder.build(state, team, mask)

    def _info(self, agent: str, state: BattleState, status: int) -> dict[str, Any]:
        info: dict[str, Any] = {
            "action_mask": self._masks[agent],
            "deploy_status": status,
            "tick": state.tick,
        }
        if state.game_over:
            w = state.winner
            info["winner"] = w
            info["outcome"] = 0 if w == Winner.DRAW else (1 if w == AGENT_TEAM[agent] else -1)
        return info

    # -- extras ---------------------------------------------------------------

    def action_masks(self, agent: str = "blue") -> np.ndarray:
        """Boolean legal-action mask, the shape sb3-contrib MaskablePPO expects."""
        return self._masks[agent].astype(bool)

    def state(self) -> np.ndarray:
        """Global state for centralised critics: Blue's observation, flattened.

        Layout is ``state_space`` (every non-mask key of the observation Dict, in
        the Dict space's key order). It inherits Blue's imperfect information
        (Red's elixir and hand are hidden unless the builder reveals them).
        """
        obs = self._obs["blue"]
        return np.concatenate(
            [np.asarray(obs[k], dtype=np.float32).ravel() for k in self._state_keys]
        )

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        return render_ansi(self.engine, self.battle_state)

    def close(self) -> None:
        if self.viser is not None:
            self.viser.close()


def _state_layout(obs_space: gym.spaces.Space[Any]) -> tuple[list[str], gym.spaces.Box]:
    """Keys and Box of ``ClashParallelEnv.state()``: the observation minus its action mask.

    pettingzoo's ``state_test`` reads ``state_space``; without it a centralised
    critic has no declared shape for ``state()``.
    """
    if not isinstance(obs_space, gym.spaces.Dict):
        raise TypeError("ObsBuilder.observation_space() must be a gymnasium Dict")
    keys = [k for k in obs_space.spaces if k != "action_mask"]
    boxes = []
    for k in keys:
        box = obs_space.spaces[k]
        if not isinstance(box, gym.spaces.Box):
            raise TypeError(f"observation key {k!r} is not a Box; state() cannot flatten it")
        boxes.append(box)
    low = np.concatenate([b.low.astype(np.float32).ravel() for b in boxes])
    high = np.concatenate([b.high.astype(np.float32).ravel() for b in boxes])
    return keys, gym.spaces.Box(low, high, dtype=np.float32)


def render_ansi(engine: Engine, state: BattleState) -> str:
    """Tile-resolution ASCII board, engine frame, Blue at the bottom."""
    a = engine.arena()
    grid = np.asarray(a.grid).reshape(a.tiles_y, a.half, a.tiles_x, a.half)
    rows = []
    for ty in range(a.tiles_y):
        row = []
        for tx in range(a.tiles_x):
            cell = grid[ty, :, tx, :]
            row.append("~" if (cell & 32).all() else ("#" if (cell & 16).all() else "."))
        rows.append(row)
    for e in state.entities:
        tx = min(max(e.x // a.subtile, 0), a.tiles_x - 1)
        ty = min(max(e.y // a.subtile, 0), a.tiles_y - 1)
        glyph = "T" if e.tower_slot >= 0 else ("B" if e.kind == 1 else "u")
        rows[ty][tx] = glyph if e.team == BLUE else glyph.lower() if glyph != "u" else "r"
    p = state.players
    head = (
        f"tick {state.tick}  blue {p[0].elixir_milli / 1000:.1f}e {p[0].crowns}c  "
        f"red {p[1].elixir_milli / 1000:.1f}e {p[1].crowns}c"
    )
    return head + "\n" + "\n".join("".join(r) for r in reversed(rows))


class ClashGymEnv(gym.Env[dict[str, np.ndarray], int]):
    """Single-agent Gymnasium env: one seat is the learner, the other an ``Opponent``."""

    # gymnasium.Env declares `metadata` as an instance attribute, so ClassVar here would
    # trip mypy's override check; the dict is never mutated.
    metadata: dict[str, Any] = {"render_modes": ["ansi"], "render_fps": 2}  # noqa: RUF012

    def __init__(
        self,
        agent: str = "blue",
        opponent: Opponent | None = None,
        render_mode: str | None = None,
        **parallel_kwargs: Any,
    ) -> None:
        if agent not in AGENTS:
            raise ValueError(f"agent must be one of {AGENTS}")
        self.parallel = ClashParallelEnv(render_mode=render_mode, **parallel_kwargs)
        self.agent = agent
        self.other = AGENTS[1 - AGENTS.index(agent)]
        self.opponent: Opponent = opponent or NoopOpponent()
        self.render_mode = render_mode
        self.observation_space = self.parallel.observation_space(agent)
        self.action_space = self.parallel.action_space(agent)
        self._last: dict[str, dict[str, np.ndarray]] = {}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        sub_seed = int(self.np_random.integers(0, 2**63 - 1))
        obs, infos = self.parallel.reset(seed=sub_seed, options=options)
        self._last = obs
        return obs[self.agent], infos[self.agent]

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        opp_obs = self._last[self.other]
        opp_action = self.opponent.act(opp_obs, opp_obs["action_mask"], self.np_random)
        obs, rew, term, trunc, info = self.parallel.step(
            {self.agent: int(action), self.other: int(opp_action)}
        )
        self._last = obs
        i = dict(info[self.agent])
        i["opponent_deploy_status"] = info[self.other]["deploy_status"]
        return obs[self.agent], rew[self.agent], term[self.agent], trunc[self.agent], i

    def action_masks(self) -> np.ndarray:
        return self._last[self.agent]["action_mask"].astype(bool)

    def render(self) -> str | None:  # type: ignore[override]
        return self.parallel.render()

    def close(self) -> None:
        self.parallel.close()


class ClashSelfPlayVecEnv(VectorEnv[Any, Any, Any]):
    """N simultaneous games exposed as 2N agent slots: slot 2i = blue, 2i+1 = red, of game i.

    Both seats feed one batch, which is how a single shared policy gets both
    players' experience in self-play. Autoreset is SAME_STEP: when a game ends,
    the returned observation is already the next game's first one, and the final
    observation / info are in ``infos["final_obs"]`` / ``infos["final_info"]``
    (masked by ``infos["_final_obs"]``).
    """

    def __init__(
        self,
        num_games: int,
        env_fn: Callable[[], ClashParallelEnv] = ClashParallelEnv,
    ) -> None:
        self.envs = [env_fn() for _ in range(num_games)]
        self.num_games = num_games
        self.num_envs = 2 * num_games
        self.single_observation_space = self.envs[0].observation_space("blue")
        self.single_action_space = self.envs[0].action_space("blue")
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.metadata = {"autoreset_mode": AutoresetMode.SAME_STEP}
        self._obs_buf = create_empty_array(
            self.single_observation_space, self.num_envs, fn=np.zeros
        )
        self._last: list[dict[str, np.ndarray]] = []

    def reset(
        self, *, seed: int | list[int | None] | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        if seed is None or isinstance(seed, int):
            seeds: list[int | None] = [
                None if seed is None else seed + i for i in range(self.num_games)
            ]
        else:
            seeds = list(seed)
        flat: list[dict[str, np.ndarray]] = []
        infos: dict[str, Any] = {}
        for g, (env, s) in enumerate(zip(self.envs, seeds, strict=True)):
            obs, info = env.reset(seed=s, options=options)
            for k, agent in enumerate(AGENTS):
                flat.append(obs[agent])
                infos = self._add_info(infos, info[agent], 2 * g + k)
        self._last = flat
        return copy.deepcopy(concatenate(self.single_observation_space, flat, self._obs_buf)), infos

    def step(self, actions: Any) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        actions = np.asarray(actions).reshape(self.num_envs)
        rewards = np.zeros(self.num_envs, dtype=np.float64)
        terms = np.zeros(self.num_envs, dtype=np.bool_)
        truncs = np.zeros(self.num_envs, dtype=np.bool_)
        infos: dict[str, Any] = {}
        flat: list[dict[str, np.ndarray]] = []
        for g, env in enumerate(self.envs):
            acts = {agent: int(actions[2 * g + k]) for k, agent in enumerate(AGENTS)}
            obs, rew, term, trunc, info = env.step(acts)
            done = term["blue"] or trunc["blue"]
            if done:
                new_obs, new_info = env.reset()
            for k, agent in enumerate(AGENTS):
                i = 2 * g + k
                rewards[i], terms[i], truncs[i] = rew[agent], term[agent], trunc[agent]
                if done:
                    infos = self._add_info(
                        infos, {"final_obs": obs[agent], "final_info": info[agent]}, i
                    )
                    infos = self._add_info(infos, new_info[agent], i)
                    flat.append(new_obs[agent])
                else:
                    infos = self._add_info(infos, info[agent], i)
                    flat.append(obs[agent])
        self._last = flat
        batched = copy.deepcopy(concatenate(self.single_observation_space, flat, self._obs_buf))
        return batched, rewards, terms, truncs, infos

    def action_masks(self) -> np.ndarray:
        return np.stack([o["action_mask"].astype(bool) for o in self._last])

    def close_extras(self, **kwargs: Any) -> None:
        for env in self.envs:
            env.close()


def make_gym_vec_env(
    num_envs: int, autoreset_mode: AutoresetMode = AutoresetMode.NEXT_STEP, **env_kwargs: Any
) -> gym.vector.SyncVectorEnv:
    """Stock SyncVectorEnv of ClashGymEnv. Masks: ``np.stack(vec.call("action_masks"))``."""
    return gym.vector.SyncVectorEnv(
        [lambda: ClashGymEnv(**env_kwargs) for _ in range(num_envs)],
        autoreset_mode=autoreset_mode,
    )
