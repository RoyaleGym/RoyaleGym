"""royalegym -- RL environments over a deterministic Clash Royale engine (RoyaleSim).

Components (each an ABC with swappable implementations):
    ObsBuilder        obs.py           SpatialObsBuilder, EntityListObsBuilder
    ActionParser      action.py        TileActionParser (Discrete 2305), HalfTileActionParser
    RewardFunction    reward.py        WinLoss, Crown, TowerHP, ElixirTrade, ElixirLeak, Combined;
                                       IllegalAction and PlacementDepth ship unused, as templates
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

Opponents: selfplay.py has the trivial two (noop, random); opponents.py adds four
scripted strategies and ``ladder()``, whose ordering is measured rather than assumed.

Comparing bots: evaluate.py plays two Opponents head to head on both seats and
reports a win rate with a Wilson interval, so "A is better" is a claim with an error
bar rather than a percentage. selfplay.py's OpponentPool keeps Elo across many such
results.

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
from .evaluate import MatchResult, SeatResult, evaluate
from .mock_engine import MockEngine
from .obs import (
    EntityListObsBuilder,
    MatchClock,
    MatchMemory,
    ObsBuilder,
    Reveal,
    SpatialObsBuilder,
    Variability,
    fair_fields,
    measure_variability,
)
from .opponents import (
    DefendOpponent,
    FirstAffordableOpponent,
    PatientOpponent,
    PushOpponent,
    ladder,
)
from .protocol import Engine
from .public_log import PublicLogMemory
from .replay import ReplayRecorder, load_trace, save_trace, verify_trace
from .reward import (
    CombinedReward,
    CrownReward,
    ElixirLeakPenalty,
    ElixirTradeReward,
    IllegalActionPenalty,
    PlacementDepthReward,
    RewardFunction,
    TowerHPReward,
    WinLossReward,
    default_reward,
)

# The Rust engine is optional: rust_engine.py imports without the compiled
# extension and RustEngine() raises ImportError naming the build command.
from .rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available
from .selfplay import (
    CallableOpponent,
    NoopOpponent,
    OpponentPool,
    RandomLegalOpponent,
)
from .state_mutator import (
    DeckCurriculumStateMutator,
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

#: ``gym.make`` ids. The plain one takes whatever engine you pass and defaults to the
#: mock WITH A WARNING, because a default that decides which engine a result came from
#: and does not say so is the one thing this package should never do. The other two name
#: their engine, so a reader can tell from the id what a number means, and neither warns.
GYM_ENV_ID = "royalegym/ClashRoyale-v0"
GYM_MOCK_ENV_ID = "royalegym/ClashRoyaleMock-v0"
GYM_RUST_ENV_ID = "royalegym/ClashRoyaleRust-v0"

#: id -> entry point. Registered at import, as gymnasium expects.
GYM_ENV_IDS = {
    GYM_ENV_ID: "royalegym.env:ClashGymEnv",
    GYM_MOCK_ENV_ID: "royalegym.env:mock_gym_env",
    GYM_RUST_ENV_ID: "royalegym.env:rust_gym_env",
}


def _register() -> None:
    import gymnasium

    for env_id, entry_point in GYM_ENV_IDS.items():
        if env_id not in gymnasium.registry:
            gymnasium.register(id=env_id, entry_point=entry_point)


_register()

__all__ = [
    "CORE_IMPORT_ERROR",
    "GYM_ENV_ID",
    "GYM_ENV_IDS",
    "GYM_MOCK_ENV_ID",
    "GYM_RUST_ENV_ID",
    "ActionParser",
    "AllCondition",
    "AnyCondition",
    "CallableOpponent",
    "ClashGymEnv",
    "ClashParallelEnv",
    "ClashSelfPlayVecEnv",
    "CombinedReward",
    "CrownReward",
    "DeckCurriculumStateMutator",
    "DefaultStateMutator",
    "DefaultStateSetter",
    "DefendOpponent",
    "DoneCondition",
    "ElixirLeakPenalty",
    "ElixirTradeReward",
    "Engine",
    "EntityListObsBuilder",
    "FirstAffordableOpponent",
    "FirstCrownCondition",
    "GameOverCondition",
    "HalfTileActionParser",
    "IllegalActionPenalty",
    "MatchClock",
    "MatchMemory",
    "MatchResult",
    "MidGameStateMutator",
    "MidGameStateSetter",
    "MockEngine",
    "NoopOpponent",
    "ObsBuilder",
    "OpponentPool",
    "PatientOpponent",
    "PlacementDepthReward",
    "PlacementOracle",
    "PublicLogMemory",
    "PushOpponent",
    "RandomLegalOpponent",
    "ReplayRecorder",
    "Reveal",
    "RewardFunction",
    "RustEngine",
    "ScriptedBoardStateMutator",
    "ScriptedBoardStateSetter",
    "SeatResult",
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
    "evaluate",
    "fair_fields",
    "ladder",
    "load_trace",
    "make_gym_vec_env",
    "measure_variability",
    "save_trace",
    "verify_trace",
]
