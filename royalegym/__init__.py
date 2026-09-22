"""royalegym -- RL environments over a deterministic Clash Royale engine (RoyaleSim).

Components (each an ABC with swappable implementations):
    ObsBuilder        obs.py           SpatialObsBuilder, EntityListObsBuilder
    ActionParser      action.py        TileActionParser (Discrete 2305), HalfTileActionParser
    RewardFunction    reward.py        WinLoss, Crown, TowerHP, ElixirTrade, ElixirLeak, Combined
    DoneCondition     done_condition.py  GameOver, FirstCrown (terminations); StepLimit,
                                         TickLimit (truncations); Any, All (either role)
    StateMutator      state_mutator.py   Default, MidGame, ScriptedBoard, Snapshot, Weighted

A DoneCondition is used in one of two roles, termination (the outcome is decided)
or truncation (the episode is cut); TerminationCondition and TruncationCondition
are the role-declaring subclasses, and the envs take termination_cond and
truncation_cond. The names are RLGym v2's.

Environments (env.py): ClashParallelEnv (PettingZoo), ClashGymEnv (Gymnasium),
ClashSelfPlayVecEnv (vectorised self-play). Engine contract: protocol.Engine;
MockEngine is the pure-Python reference implementation; RustEngine
(rust_engine.py) drives the compiled Rust core through the same protocol.

Replays: replay.py records and re-verifies traces; render.py turns one into a
self-contained HTML viewer (``python -m royalegym.render trace.msgpack -o out.html``).
render is deliberately NOT imported here: importing it from the package would make
``python -m royalegym.render`` load the module twice (runpy's RuntimeWarning).
"""

from .action import ActionParser, HalfTileActionParser, PlacementOracle, TileActionParser
from .done_condition import (
    AllCondition,
    AnyCondition,
    DoneCondition,
    FirstCrownCondition,
    GameOverCondition,
    StepLimitCondition,
    TerminalCondition,
    TerminationCondition,
    TickLimitCondition,
    TruncationCondition,
)
from .env import ClashGymEnv, ClashParallelEnv, ClashSelfPlayVecEnv, make_gym_vec_env
from .mock_engine import MockEngine
from .obs import (
    EntityListObsBuilder,
    ObsBuilder,
    Reveal,
    SpatialObsBuilder,
    Variability,
    measure_variability,
)
from .protocol import Engine
from .replay import ReplayRecorder, load_trace, save_trace, verify_trace
from .reward import (
    CombinedReward,
    CrownReward,
    ElixirLeakPenalty,
    ElixirTradeReward,
    IllegalActionPenalty,
    RewardFunction,
    TowerHPReward,
    WinLossReward,
    default_reward,
)

# The Rust engine is optional: rust_engine.py imports without the compiled
# extension and RustEngine() raises ImportError naming the build command.
from .rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available
from .selfplay import NoopOpponent, OpponentPool, RandomLegalOpponent
from .state_mutator import (
    DefaultStateMutator,
    DefaultStateSetter,
    MidGameStateMutator,
    MidGameStateSetter,
    ScriptedBoardStateMutator,
    ScriptedBoardStateSetter,
    SnapshotStateMutator,
    SnapshotStateSetter,
    StateMutator,
    StateSetter,
    WeightedStateMutator,
    WeightedStateSetter,
)

GYM_ENV_ID = "royalegym/ClashRoyale-v0"


def _register() -> None:
    import gymnasium

    if GYM_ENV_ID not in gymnasium.registry:
        gymnasium.register(id=GYM_ENV_ID, entry_point="royalegym.env:ClashGymEnv")


_register()

__all__ = [
    "CORE_IMPORT_ERROR",
    "GYM_ENV_ID",
    "ActionParser",
    "AllCondition",
    "AnyCondition",
    "ClashGymEnv",
    "ClashParallelEnv",
    "ClashSelfPlayVecEnv",
    "CombinedReward",
    "CrownReward",
    "DefaultStateMutator",
    "DefaultStateSetter",
    "DoneCondition",
    "ElixirLeakPenalty",
    "ElixirTradeReward",
    "Engine",
    "EntityListObsBuilder",
    "FirstCrownCondition",
    "GameOverCondition",
    "HalfTileActionParser",
    "IllegalActionPenalty",
    "MidGameStateMutator",
    "MidGameStateSetter",
    "MockEngine",
    "NoopOpponent",
    "ObsBuilder",
    "OpponentPool",
    "PlacementOracle",
    "RandomLegalOpponent",
    "ReplayRecorder",
    "Reveal",
    "RewardFunction",
    "RustEngine",
    "ScriptedBoardStateMutator",
    "ScriptedBoardStateSetter",
    "SnapshotStateMutator",
    "SnapshotStateSetter",
    "SpatialObsBuilder",
    "StateMutator",
    "StateSetter",
    "StepLimitCondition",
    "TerminalCondition",
    "TerminationCondition",
    "TickLimitCondition",
    "TileActionParser",
    "TowerHPReward",
    "TruncationCondition",
    "Variability",
    "WeightedStateMutator",
    "WeightedStateSetter",
    "WinLossReward",
    "core_available",
    "default_reward",
    "load_trace",
    "make_gym_vec_env",
    "measure_variability",
    "save_trace",
    "verify_trace",
]
