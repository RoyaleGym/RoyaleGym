"""Environments: PettingZoo ParallelEnv, Gymnasium single-agent wrapper, vectorised self-play.

DECOMPOSITION
    ObsBuilder, ActionParser, RewardFunction, StateMutator and two DoneConditions
    (one in the termination role, one in the truncation role) are constructor
    arguments. Changing a reward or an observation is a Python edit and never a
    recompile of the simulator, because that is where most research iteration
    happens.

DONE FLAGS
    Gymnasium's ``terminated`` comes from ``termination_cond`` (default
    ``GameOverCondition``) and ``truncated`` from ``truncation_cond`` (default
    none). Both are consulted every step so counters advance; a termination on
    the same step as a truncation is reported as a termination only.

TIMING
    One env step = one decision = ``decision_ms`` of game time, rounded UP to a
    whole number of engine ticks (engine tick length comes from the engine's
    state, which reads calibration.json). Both players act simultaneously; the
    engine validates both commands against the same pre-step state.

ACTION MASKS
    Every observation dict carries ``action_mask`` (int8, as pettingzoo's
    ``parallel_api_test`` and gymnasium ``Discrete.sample(mask=...)`` expect), and
    ``mask_planes`` -- the same mask without the no-op, as [4, 32, 18] -- beside
    it. ``action_masks()`` returns the bool array sb3-contrib's MaskablePPO calls
    for. An action the engine rejects is turned into a no-op and reported in
    ``info["deploy_status"]``. The info dict does NOT repeat the mask: it was the
    same array the observation already carried, batched a second time by every
    vector env for nothing.

EPISODE STATISTICS
    The info dict of the LAST step of an episode carries how that episode went --
    length, crowns, tower hp, elixir leaked -- so a training run reads it out of
    ``infos["final_info"]`` instead of keeping a shadow copy of the state.
    ``EPISODE_STAT_KEYS`` names them. ``outcome`` and ``winner`` are there too, as
    before, but only when the engine ended the battle: a truncation has no winner.

    It also carries the reward TERM BY TERM, as ``reward_sum/<TermName>`` summed
    over the episode for that seat. That is the number which says which term a
    policy is actually chasing, and without it a run logs one scalar and cannot tell
    a win from a well-farmed shaping term. Those keys follow the reward function's
    own terms rather than a fixed list, which is why they are not in
    ``EPISODE_STAT_KEYS``. ``log_reward_terms=True`` adds the same breakdown every
    STEP as ``reward/<TermName>``, for watching one battle rather than training.

THE VIEWER
    ``ClashParallelEnv`` publishes only when it is HANDED a publisher, and reads no
    environment variable of its own: N envs each binding the viewer's one fixed UDP
    port is an OSError, not a feature. ``ClashSelfPlayVecEnv`` owns that decision
    instead and hands one publisher to one game (see its ``viser`` argument).
"""

from __future__ import annotations

import copy
import pickle
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space, concatenate, create_empty_array
from pettingzoo import ParallelEnv

from .action import NOOP, ActionParser, TileActionParser
from .done_condition import (
    AnyCondition,
    DoneCondition,
    GameOverCondition,
    TerminationCondition,
    TruncationCondition,
)
from .mock_engine import MockEngine
from .obs import ObsBuilder, SpatialObsBuilder
from .protocol import (
    BLUE,
    RED,
    TEAMS,
    BattleState,
    DeployCommand,
    DeployResult,
    DeployStatus,
    Engine,
    TowerSlot,
    Winner,
    calibration_digest,
    default_calibration,
)
from .replay import ReplayRecorder
from .reward import RewardFunction, default_reward
from .selfplay import NoopOpponent, Opponent
from .state_mutator import DefaultStateMutator, Snapshot, StateMutator
from .viser import ViserPublisher, play_event

AGENTS = ("blue", "red")
AGENT_TEAM = {"blue": BLUE, "red": RED}
NO_COMMAND = -1  # info["deploy_status"] when the agent chose no-op
# Keys ``ClashParallelEnv.episode_stats`` writes into the LAST info of an episode.
EPISODE_STAT_KEYS = (
    "episode_steps",
    "episode_ticks",
    "own_crowns",
    "enemy_crowns",
    "own_tower_hp_frac",
    "enemy_tower_hp_frac",
    "elixir_leak_steps",
    "elixir_count_exact",
)
# Observation keys ``state()`` leaves out: the action mask in either of its shapes.
# It is legality, not state, it is already an input to the policy, and a centralised
# critic fed both shapes would carry 2 304 duplicated numbers per seat per step.
MASK_KEYS = ("action_mask", "mask_planes")


def class_name(obj: Any) -> str:
    return f"{type(obj).__module__}.{type(obj).__name__}"


def component_config(obj: Any) -> dict[str, Any] | None:
    """``{class, params}`` for one env component, for ``ClashParallelEnv.config()``.

    ``params`` is whatever the component's own ``config()`` returns; a component
    that has none reports its class name and an empty dict rather than guessing at
    its constructor, so the record never claims to describe something it cannot.
    """
    if obj is None:
        return None
    cfg = getattr(obj, "config", None)
    return {"class": class_name(obj), "params": dict(cfg()) if callable(cfg) else {}}


def engine_build_digest(engine: Engine) -> str | None:
    """The engine's ``build_digest()`` if it has one (rust_engine.py), else None."""
    fn = getattr(engine, "build_digest", None)
    if not callable(fn):
        return None
    try:
        return str(fn())
    except ImportError:
        return None


def _counts_are_exact(obs_builder: ObsBuilder, team: int) -> bool:
    """Whether the builder's counted features are still provably right for ``team``.

    True for a builder that keeps no counts -- nothing was counted, so nothing can
    have been counted wrong -- which is also what a builder written before
    ``ObsBuilder.counts_are_exact`` existed reports.
    """
    fn = getattr(obs_builder, "counts_are_exact", None)
    return bool(fn(team)) if callable(fn) else True


def tower_hp_frac(state: BattleState, team: int) -> float:
    """Mean of ``team``'s three crown towers' hp fractions, in [0, 1].

    The MEAN of the per-tower fractions, not total hp over total max hp, so that it
    is exactly ``TowerHPReward``'s potential (reward.py) divided by three: what a run
    logs about an episode and what its shaping term optimised are then the same
    number, rather than two plausible summaries that drift apart. The difference is
    real -- total-over-total weights the king by its much larger hp pool, the mean
    weights each tower equally, and losing a princess is worth more than the hp says.
    """
    p = state.players[team]
    return sum(
        max(0, p.tower_hp[s]) / max(1, p.tower_max_hp[s]) for s in TowerSlot
    ) / len(TowerSlot)


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
        termination_cond: DoneCondition | None = None,
        truncation_cond: DoneCondition | None = None,
        state_mutator: StateMutator | None = None,
        decision_ms: int = 500,
        log_reward_terms: bool = False,
        render_mode: str | None = None,
        recorder: ReplayRecorder | None = None,
        viser: ViserPublisher | None = None,
        # Kept for callers written before the rename.
        terminal_conditions: Sequence[DoneCondition] | None = None,
        state_setter: StateMutator | None = None,
    ) -> None:
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode {render_mode!r} not supported")
        if terminal_conditions is not None:
            if termination_cond is not None or truncation_cond is not None:
                raise TypeError("pass termination_cond/truncation_cond or terminal_conditions")
            termination_cond, truncation_cond = _split_by_role(terminal_conditions)
        if state_setter is not None:
            if state_mutator is not None:
                raise TypeError("pass state_mutator or state_setter, not both")
            state_mutator = state_setter
        if isinstance(termination_cond, TruncationCondition):
            raise TypeError(f"{type(termination_cond).__name__} is a truncation, not a termination")
        if isinstance(truncation_cond, TerminationCondition):
            raise TypeError(f"{type(truncation_cond).__name__} is a termination, not a truncation")
        self.engine: Engine = engine if engine is not None else MockEngine()
        self.action_parser = action_parser or TileActionParser()
        self.action_parser.bind(self.engine)
        self.obs_builder = obs_builder or SpatialObsBuilder()
        self.obs_builder.bind(self.engine, self.action_parser)
        self.reward_fn = reward_fn or default_reward()
        self.reward_fn.bind(self.engine)
        self.termination: DoneCondition = termination_cond or GameOverCondition()
        self.truncation: DoneCondition | None = truncation_cond
        self.state_mutator = state_mutator or DefaultStateMutator()
        self.decision_ms = decision_ms
        # Per-STEP reward terms in info. The per-episode sums are always there and
        # cost nothing; this adds one key per term per step, which a beginner
        # watching a battle wants and a training run at scale does not.
        self.log_reward_terms = log_reward_terms
        self.render_mode = render_mode
        self.recorder = recorder
        # RoyaleViser (viser.py): one publish per reset/step, and only while a viewer is
        # attached; None (the default) costs one ``if``. This env NEVER builds its own
        # publisher -- see the module doc, THE VIEWER.
        self.viser = viser
        self._decks: list[list[int]] | None = None
        # The builder's calibration when it has one (every shipped ObsBuilder does),
        # so an env and its observation cannot read two different ledgers.
        self.calibration = getattr(self.obs_builder, "calibration", None) or default_calibration()
        self.full_elixir_milli = 1000 * self.calibration.int("match.MAX_MANA")

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
        # From the calibration this env already holds, so ``config()`` is honest
        # before the first reset; ``reset`` recomputes it from the engine's own
        # state, which is the authority if an engine ever reports a different tick.
        self.decision_ticks = max(1, -(-decision_ms // self.calibration.int("time.TICK_MS")))
        self.last_results: list[DeployResult] = []
        self._episode_steps = 0
        self._start_tick = 0
        self._leak_steps = dict.fromkeys(TEAMS, 0)
        self._term_sums: dict[int, dict[str, float]] = {t: {} for t in TEAMS}

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
            init = self.state_mutator.build(self._np_random, cards)
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
        self.termination.reset(state)
        if self.truncation is not None:
            self.truncation.reset(state)
        self.agents = list(self.possible_agents)
        self.last_results = []
        self._episode_steps = 0
        self._start_tick = state.tick
        self._leak_steps = dict.fromkeys(TEAMS, 0)
        self._term_sums = {t: {} for t in TEAMS}
        if self.recorder is not None:
            self.recorder.begin(self.engine, engine_seed, init)
        if self.viser is not None:
            self._decks = None if isinstance(init, Snapshot) else [list(d) for d in init.decks]
            self.viser.publish(state, cards, self.engine.arena(), self._decks)
        self._refresh(state)
        infos = {a: self._info(a, state, NO_COMMAND, terminal=False) for a in self.agents}
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
        self._episode_steps += 1
        full = self.full_elixir_milli
        for team in TEAMS:
            if prev.players[team].elixir_milli >= full and state.players[team].elixir_milli >= full:
                self._leak_steps[team] += 1
        terminated = self.termination.is_done(state)
        truncated = self.truncation is not None and self.truncation.is_done(state)
        truncated = truncated and not terminated
        status = {a: NO_COMMAND for a in self.agents}
        for agent, res in zip(owner, results, strict=True):
            status[agent] = res.status
        rewards = {
            a: float(self.reward_fn.get_reward(AGENT_TEAM[a], prev, state, results))
            for a in self.agents
        }
        self._accumulate_terms()
        self._refresh(state)
        live = list(self.agents)
        obs = {a: self._obs[a] for a in live}
        infos = {a: self._info(a, state, status[a], terminated or truncated) for a in live}
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

    def _info(
        self, agent: str, state: BattleState, status: int, terminal: bool
    ) -> dict[str, Any]:
        info: dict[str, Any] = {"deploy_status": status, "tick": state.tick}
        team = AGENT_TEAM[agent]
        if self.log_reward_terms:
            terms_for = getattr(self.reward_fn, "terms_for", None)
            step_terms = terms_for(team) if callable(terms_for) else {}
            # Present at reset too, as zeros, so the key set does not change between
            # reset and step -- gymnasium's info batching is happier that way.
            for name in self._term_sums[team] or step_terms:
                info[f"reward/{name}"] = float(step_terms.get(name, 0.0))
        if state.game_over:
            w = state.winner
            info["winner"] = w
            info["outcome"] = 0 if w == Winner.DRAW else (1 if w == team else -1)
        if terminal:
            info.update(self.episode_stats(state, team))
        return info

    def _accumulate_terms(self) -> None:
        """Sum this step's reward breakdown into the episode's, per seat.

        A no-op for a reward that keeps no breakdown. ``CombinedReward`` keys its by
        team (reward.py), which it has to: the env asks it once per seat on the same
        transition, and a single flat dict would hold only the second seat's.
        """
        terms_for = getattr(self.reward_fn, "terms_for", None)
        if not callable(terms_for):
            return
        for team in TEAMS:
            into = self._term_sums[team]
            for name, value in terms_for(team).items():
                into[name] = into.get(name, 0.0) + float(value)

    def reward_terms(self, team: int) -> dict[str, float]:
        """This episode's reward, term by term, for one seat.

        The number a training run plots and the one that says WHICH term a policy is
        actually chasing. It reaches a trainer through the terminal info as
        ``reward_sum/<TermName>`` -- and through ``final_info`` after an autoreset,
        which is the only place it survives -- so nothing has to reach into the env
        to get it. The keys follow the reward's own terms, so they are not in
        ``EPISODE_STAT_KEYS``; the reward function is the authority on the set.
        """
        return dict(self._term_sums[team])

    def episode_stats(self, state: BattleState, team: int) -> dict[str, Any]:
        """How the episode that just ended went, from ``team``'s seat.

        Written only on the terminal step, so a rollout buffer carries one of these
        per EPISODE rather than one per step; ``EPISODE_STAT_KEYS`` is the key list.
        Tower hp is the MEAN of the three towers' hp fractions (``tower_hp_frac``),
        which is TowerHPReward's potential over three -- so what a run logs and what
        its shaping term optimised are the same number. Crowns alone cannot tell a
        tower left at 1 hp from a tower never touched, which is why it is here.
        """
        return {
            "episode_steps": self._episode_steps,
            "episode_ticks": state.tick - self._start_tick,
            "own_crowns": state.players[team].crowns,
            "enemy_crowns": state.players[1 - team].crowns,
            "own_tower_hp_frac": tower_hp_frac(state, team),
            "enemy_tower_hp_frac": tower_hp_frac(state, 1 - team),
            "elixir_leak_steps": self._leak_steps[team],
            "elixir_count_exact": _counts_are_exact(self.obs_builder, team),
            **{f"reward_sum/{k}": v for k, v in self._term_sums[team].items()},
        }

    def config(self) -> dict[str, Any]:
        """What this env IS, as a JSON-able dict, for a checkpoint to record.

        Class names and constructor kwargs of every component, the decision
        granularity, the ``Reveal`` the observation was built with, and digests of
        the data underneath: calibration.json as it is on disk, plus the copy the
        compiled engine was built with when there is one. A checkpoint that pins
        this can say whether a policy is being evaluated on the env it was trained
        on -- including whether it was trained with hidden information revealed,
        which nothing about a weights file would otherwise show.

        A description, not a constructor: ``EnvFactory`` is the picklable recipe
        that BUILDS one.
        """
        reveal = getattr(self.obs_builder, "reveal", None)
        return {
            "env": class_name(self),
            "decision_ms": self.decision_ms,
            "decision_ticks": self.decision_ticks,
            "log_reward_terms": self.log_reward_terms,
            "reveal": reveal.as_dict() if reveal is not None else None,
            "engine": component_config(self.engine),
            "obs_builder": component_config(self.obs_builder),
            "action_parser": component_config(self.action_parser),
            "reward_fn": component_config(self.reward_fn),
            "termination_cond": component_config(self.termination),
            "truncation_cond": component_config(self.truncation),
            "state_mutator": component_config(self.state_mutator),
            "calibration_digest": calibration_digest(self.calibration),
            "build_digest": engine_build_digest(self.engine),
        }

    # -- extras ---------------------------------------------------------------

    def action_masks(self, agent: str = "blue") -> np.ndarray:
        """Boolean legal-action mask, the shape sb3-contrib MaskablePPO expects."""
        return self._masks[agent].astype(bool)

    def state(self) -> np.ndarray:
        """Global state for centralised critics: Blue's observation, flattened.

        Layout is ``state_space`` (every non-mask key of the observation Dict, in
        the Dict space's key order -- ``MASK_KEYS`` are left out). It inherits
        Blue's imperfect information: Red's elixir is the count Blue's builder
        keeps and Red's hand is not there at all, unless the builder's ``Reveal``
        opens them.
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


def _split_by_role(
    conditions: Sequence[DoneCondition],
) -> tuple[DoneCondition | None, DoneCondition | None]:
    """One flat list into the two slots: declared truncations truncate, the rest terminate."""
    terms = [c for c in conditions if not isinstance(c, TruncationCondition)]
    truncs = [c for c in conditions if isinstance(c, TruncationCondition)]

    def one(cs: list[DoneCondition]) -> DoneCondition | None:
        return None if not cs else cs[0] if len(cs) == 1 else AnyCondition(cs)

    return one(terms), one(truncs)


def _state_layout(obs_space: gym.spaces.Space[Any]) -> tuple[list[str], gym.spaces.Box]:
    """Keys and Box of ``ClashParallelEnv.state()``: the observation minus its action mask.

    pettingzoo's ``state_test`` reads ``state_space``; without it a centralised
    critic has no declared shape for ``state()``.
    """
    if not isinstance(obs_space, gym.spaces.Dict):
        raise TypeError("ObsBuilder.observation_space() must be a gymnasium Dict")
    keys = [k for k in obs_space.spaces if k not in MASK_KEYS]
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
    (masked by ``infos["_final_obs"]``). ``final_info`` is where an episode's
    statistics arrive -- ``EPISODE_STAT_KEYS``.

    THE VIEWER, and why it is decided here. A viewer watches ONE battle: it has
    one fixed UDP port, and N envs each binding it is ``OSError 10048``, which is
    what ``ClashSelfPlayVecEnv(num_games>1)`` used to raise the moment ROYALEVISER
    was set. So the publisher is bound once, here, and handed to game 0 only.
    ``viser="env"`` (the default) means ``ViserPublisher.from_env()``: a publisher
    when ROYALEVISER=host:port is set, and None -- costing nothing -- when it is
    not. Pass a ``ViserPublisher`` to bind one explicitly, or None never to
    publish.

    ADDRESSABLE EPISODES, and why a run cannot resume without them. With
    ``autoreset_seed_fn`` unset, an autoreset calls ``reset()`` with no seed, and
    ``ClashParallelEnv.reset`` leaves its generator alone when the seed is None --
    so the battle a game plays depends on how many battles that game has already
    played. A fresh worker cannot arrive at episode 400 without playing 399 first,
    which is why a resumed run diverges from the one it is continuing even when
    the learner itself came back byte for byte.

    Set ``autoreset_seed_fn`` and each episode is named instead of counted: the
    seed for the nth episode of game g is ``fn(g, n)``, so any episode can be
    reached directly. ``episode_ordinals`` is the counter to checkpoint and
    ``set_episode_ordinals`` puts it back, after which the stream continues row
    for row rather than restarting::

        fn = lambda game, n: (run_seed * 1_000_003 + game * 9973 + n) % 2**31
        vec = ClashSelfPlayVecEnv(8, autoreset_seed_fn=fn)
        ...
        saved = vec.episode_ordinals          # into the checkpoint

        vec = ClashSelfPlayVecEnv(8, autoreset_seed_fn=fn)
        vec.set_episode_ordinals(saved)       # out of it, before reset()
        vec.reset()

    The default is None and changes nothing: ``reset(seed=None)`` is what the
    autoreset already did. An explicit ``reset(seed=...)`` still wins over the
    function, and still consumes an ordinal, so the numbering keeps meaning "the
    nth episode this object started in game g" either way.
    """

    def __init__(
        self,
        num_games: int,
        env_fn: Callable[[], ClashParallelEnv] = ClashParallelEnv,
        *,
        viser: str | ViserPublisher | None = "env",
        autoreset_seed_fn: Callable[[int, int], int] | None = None,
    ) -> None:
        if isinstance(viser, str) and viser != "env":
            raise ValueError(
                f'viser must be "env", a ViserPublisher or None, not {viser!r}: a string '
                "that is not the sentinel would be stored as the publisher and fail "
                "later, on the first step, as an AttributeError inside publish()"
            )
        if autoreset_seed_fn is not None and not callable(autoreset_seed_fn):
            raise TypeError(
                "autoreset_seed_fn must be callable as fn(game_index, episode_ordinal) "
                f"-> int, not {autoreset_seed_fn!r}: a non-callable would be stored and "
                "fail later, on the first episode that ends"
            )
        self.autoreset_seed_fn = autoreset_seed_fn
        # Episodes STARTED per game, not finished: the seed of the next one is
        # fn(game, ordinal) and the counter moves after it is used. Checkpoint it.
        self._ordinals = [0] * num_games
        self.envs = [env_fn() for _ in range(num_games)]
        self.viser = ViserPublisher.from_env() if viser == "env" else viser
        if self.viser is not None:
            if not self.envs:
                raise ValueError("a viser publisher needs at least one game to watch")
            self.envs[0].viser = self.viser
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

    @property
    def episode_ordinals(self) -> tuple[int, ...]:
        """Episodes started so far, per game. The number a checkpoint has to carry.

        Without it a resumed run can name its episodes and still not know WHICH one
        to name next, so it starts at 0 and replays a run it has already played.
        """
        return tuple(self._ordinals)

    def set_episode_ordinals(self, ordinals: Sequence[int]) -> None:
        """Put the counter back where a checkpoint left it, before ``reset()``.

        The other half of ``episode_ordinals``. Call it before ``reset()``: reset
        starts an episode and therefore consumes an ordinal, so setting it after
        would skip one.
        """
        values = [int(n) for n in ordinals]
        if len(values) != self.num_games:
            raise ValueError(
                f"expected {self.num_games} ordinals, one per game, got {len(values)}"
            )
        if any(n < 0 for n in values):
            raise ValueError(f"episode ordinals count episodes and cannot be negative: {values}")
        self._ordinals = values

    def _next_seed(self, game: int) -> int | None:
        """The seed for the episode this game is about to start, and move the counter.

        None when no function is set, which is exactly what the autoreset passed
        before this existed: ``reset(seed=None)`` leaves the generator running.
        """
        ordinal = self._ordinals[game]
        self._ordinals[game] = ordinal + 1
        if self.autoreset_seed_fn is None:
            return None
        seed = self.autoreset_seed_fn(game, ordinal)
        # bool is an int, and fn returning True would silently seed every episode 1.
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError(
                f"autoreset_seed_fn({game}, {ordinal}) returned {seed!r}; it has to "
                "return an int, because the seed is what makes the episode addressable"
            )
        return int(seed)

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
            # An explicit seed wins; the ordinal moves either way, so the numbering
            # keeps meaning "the nth episode this object started in this game".
            named = self._next_seed(g)
            obs, info = env.reset(seed=s if s is not None else named, options=options)
            for k, agent in enumerate(AGENTS):
                flat.append(obs[agent])
                infos = self._add_info(infos, info[agent], 2 * g + k)
        self._last = flat
        return copy.deepcopy(concatenate(self.single_observation_space, flat, self._obs_buf)), infos

    def step(self, actions: Any) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        actions = np.asarray(actions).reshape(self.num_envs)
        # float32: the batch goes straight into a policy's buffers, which are
        # float32, so float64 here was a copy per step and nothing else.
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terms = np.zeros(self.num_envs, dtype=np.bool_)
        truncs = np.zeros(self.num_envs, dtype=np.bool_)
        infos: dict[str, Any] = {}
        flat: list[dict[str, np.ndarray]] = []
        for g, env in enumerate(self.envs):
            acts = {agent: int(actions[2 * g + k]) for k, agent in enumerate(AGENTS)}
            obs, rew, term, trunc, info = env.step(acts)
            done = term["blue"] or trunc["blue"]
            if done:
                new_obs, new_info = env.reset(seed=self._next_seed(g))
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


class EnvFactory:
    """A picklable recipe for a ``ClashParallelEnv``, for subprocess rollout workers.

    WHY A RECIPE AND NOT THE ENV. A built env cannot be pickled, and the reason is
    the ENGINE: MockEngine carried two msgspec codecs, which are compiled objects
    with no pickle support (that one is fixed -- they are stateless and are now
    dropped and rebuilt, so a MockEngine env does pickle), and RustEngine holds a
    live PyO3 ``Battle``, which pickle cannot reach at all. That could be worked
    around -- the engine can already ``save_state()`` to bytes -- but it would be
    the wrong thing to ship: it cannot be verified in a tree whose extension is not
    built, and sending a whole battle down a pipe at every worker spawn is not what
    a worker wants anyway. It wants a build recipe, which is also the shape
    gymnasium's vector envs ask for.

    NOT a drop-in for ``gym.vector.AsyncVectorEnv``: that wants a single-agent
    ``gym.Env`` factory, and this builds a two-seat ``ClashParallelEnv``. Hand it to
    ``ClashSelfPlayVecEnv``, or call it in a worker you own.

    Every component is given as its CLASS (or any importable callable) and its
    kwargs, never as a built instance::

        factory = EnvFactory(
            engine=MockEngine,
            obs_builder=(SpatialObsBuilder, {"reveal": Reveal(enemy_elixir=True)}),
            decision_ms=250,
        )
        env = factory()                                  # in the worker
        ClashSelfPlayVecEnv(4, factory)                  # or hand it to the vec env

    The spec is checked with ``pickle.dumps`` at construction, so a lambda or a
    locally-defined class fails HERE, naming the key, rather than at worker spawn
    with a traceback from inside multiprocessing.
    """

    COMPONENTS: ClassVar[tuple[str, ...]] = (
        "engine",
        "obs_builder",
        "action_parser",
        "reward_fn",
        "termination_cond",
        "truncation_cond",
        "state_mutator",
        "recorder",
        "viser",
    )

    def __init__(self, **spec: Any) -> None:
        #: component name -> class, or (class, kwargs). Names are ClashParallelEnv's.
        self.components: Mapping[str, Any] = {
            k: v for k, v in spec.items() if k in self.COMPONENTS
        }
        #: plain constructor arguments, passed through (decision_ms, render_mode, ...).
        self.kwargs: Mapping[str, Any] = {
            k: v for k, v in spec.items() if k not in self.COMPONENTS
        }
        for key, value in spec.items():
            try:
                pickle.dumps(value)
            except (TypeError, AttributeError, pickle.PicklingError) as exc:
                raise TypeError(
                    f"EnvFactory({key}=...) does not pickle, so a subprocess worker could "
                    f"never be handed it: pass an importable class or function, not a lambda "
                    f"or a locally defined class ({exc})"
                ) from exc

    def __call__(self) -> ClashParallelEnv:
        built = {name: _build_component(spec) for name, spec in self.components.items()}
        return ClashParallelEnv(**built, **dict(self.kwargs))

    def __repr__(self) -> str:
        return f"EnvFactory({self.components!r}, {self.kwargs!r})"

    def config(self) -> dict[str, Any]:
        """The recipe as a JSON-able dict, beside ``ClashParallelEnv.config()``."""
        out: dict[str, Any] = {"kwargs": dict(self.kwargs), "components": {}}
        for name, spec in self.components.items():
            factory, kwargs = spec if isinstance(spec, tuple) else (spec, {})
            out["components"][name] = {
                "class": getattr(factory, "__qualname__", str(factory)),
                "module": getattr(factory, "__module__", None),
                "kwargs": {k: repr(v) for k, v in dict(kwargs).items()},
            }
        return out


def _build_component(spec: Any) -> Any:
    factory, kwargs = spec if isinstance(spec, tuple) else (spec, {})
    return factory(**dict(kwargs))


#: Constructor arguments that carry per-battle state. One INSTANCE of any of these
#: cannot be shared between environments, and ``make_gym_vec_env`` refuses to.
UNSHAREABLE = (
    "engine",
    "obs_builder",
    "action_parser",
    "reward_fn",
    "termination_cond",
    "truncation_cond",
    "recorder",
    "viser",
)


def make_gym_vec_env(
    num_envs: int,
    autoreset_mode: AutoresetMode = AutoresetMode.NEXT_STEP,
    **env_kwargs: Any,
) -> gym.vector.SyncVectorEnv:
    """Stock SyncVectorEnv of ClashGymEnv. Masks: ``np.stack(vec.call("action_masks"))``.

    REFUSES A SHARED COMPONENT, because the signature invites one. ``**env_kwargs``
    is evaluated once and handed to every environment, so
    ``make_gym_vec_env(3, engine=MockEngine())`` gives three environments one engine
    and nothing anywhere says so. Measured before this check existed: after a single
    vector step the three sat at ticks 10, 20 and 30 -- three windows onto ONE
    battle, each stepping it again, with every reward computed against another
    environment's previous state. No exception, no warning, and a training run
    collecting it would see only that learning did not work.

    The same holds for every piece that carries state. An ObsBuilder now remembers
    the match (``MatchMemory``); an ActionParser caches placement grids; a
    ``StepLimitCondition`` counts steps; ``CombinedReward`` keeps a breakdown per
    team. Sharing any of them silently crosses the wires between environments.

    So pass a CLASS or a factory instead of an instance, and each environment builds
    its own::

        make_gym_vec_env(8, engine=MockEngine)                 # the class
        make_gym_vec_env(8, engine=lambda: RustEngine(card_names=SHARED))

    ``decision_ms``, ``agent``, ``render_mode`` and an ``opponent`` hold no
    per-battle state and pass through unchanged.
    """
    shared = [
        f"{name}={type(env_kwargs[name]).__name__}(...)"
        for name in UNSHAREABLE
        if name in env_kwargs and not callable(env_kwargs[name])
    ]
    if shared:
        raise TypeError(
            "make_gym_vec_env builds every environment from the same arguments, so "
            f"these would be ONE object shared by all {num_envs}: {', '.join(shared)}. "
            "Each carries per-battle state -- an engine IS the battle, a builder "
            "remembers the match, a step limit counts steps -- so the environments "
            "would silently step each other's. Pass the class or a factory instead: "
            "make_gym_vec_env(n, engine=MockEngine) or engine=lambda: RustEngine(...)."
        )

    def build() -> ClashGymEnv:
        made = {k: (v() if k in UNSHAREABLE and callable(v) else v) for k, v in env_kwargs.items()}
        return ClashGymEnv(**made)

    return gym.vector.SyncVectorEnv(
        [build for _ in range(num_envs)], autoreset_mode=autoreset_mode
    )
