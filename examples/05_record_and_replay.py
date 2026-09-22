"""Record a battle, prove the recording, and turn it into a page you can double-click.

A recording holds the seed, the setup, every command and a hash of the board for every
tick. Re-running it on a fresh engine has to reproduce all of them, which is what makes
it evidence rather than a video.

    python examples/05_record_and_replay.py [output-dir]
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

from royalegym import (
    ClashParallelEnv,
    DefaultStateMutator,
    RandomLegalOpponent,
    ReplayRecorder,
    RustEngine,
    load_trace,
    save_trace,
    verify_trace,
)
from royalegym.done_condition import GameOverCondition, StepLimitCondition
from royalegym.render import render_html


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    out.mkdir(parents=True, exist_ok=True)

    recorder = ReplayRecorder()
    env = ClashParallelEnv(
        engine=RustEngine(),
        state_mutator=DefaultStateMutator(),
        termination_cond=GameOverCondition(),
        truncation_cond=StepLimitCondition(120),
        recorder=recorder,
    )
    obs, _ = env.reset(seed=7)
    rng = np.random.default_rng(7)
    policy = RandomLegalOpponent(0.5)
    while env.agents:
        actions = {a: policy.act(obs[a], obs[a]["action_mask"], rng) for a in env.agents}
        obs, *_ = env.step(actions)

    trace = recorder.trace
    path = save_trace(trace, out / "battle.msgpack")
    print(f"recorded {len(trace.frames)} frames to {path} ({path.stat().st_size / 1024:.0f} KB)")

    # The proof. A FRESH engine replays the seed and the commands, and every frame hash
    # has to come back the same. An empty list is the whole claim.
    divergences = verify_trace(load_trace(path), RustEngine())
    print(f"re-run on a fresh engine: {len(divergences)} divergences")
    if divergences:
        raise SystemExit(f"the recording did not reproduce: {divergences[:3]}")

    page = out / "battle.html"
    page.write_text(render_html(load_trace(path)), encoding="utf-8")
    print(f"replay page: {page} ({page.stat().st_size / 1024:.0f} KB, no server needed)")


if __name__ == "__main__":
    main()
