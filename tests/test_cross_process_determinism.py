"""The same battle, rebuilt in a fresh process from an EnvFactory, plays out identically.

Determinism inside one process is the weaker claim. A rollout worker, an evaluation
re-scored tomorrow, someone else reproducing a bug: each of them rebuilds the env in
another process, with its own string hash seed, its own addresses and its own import
order. If anything in the env or the engine reads one of those, two processes given the
same seed and the same actions play different battles, and no test inside one process can
see it.

So the parent pickles an ``EnvFactory``, builds its env from the factory, plays a seeded
episode in which both seats deploy, and keeps every action. A child process, started
with a different random ``PYTHONHASHSEED``, unpickles the factory, rebuilds the env,
replays the same actions and reports a digest of what happened: every observation array,
reward, done flag and deploy status, and the engine's own ``state()`` and ``state_hash()``
after every step. The two digests must match step for step. The child also reports
``hash()`` of a fixed string, so the test can confirm that the processes really hashed
strings differently rather than assume it.

The factory carries a non-default ``decision_ms``, so a factory that lost part of its
recipe on the way through pickle would build a different env in the child and fail here.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import subprocess
import sys

import msgspec
import numpy as np
import pytest

from royalegym import MockEngine, SpatialObsBuilder
from royalegym.env import AGENTS, EnvFactory
from royalegym.rust_engine import CORE_IMPORT_ERROR, RustEngine, core_available

needs_rust = pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))

SEED = 21
STEPS = 150
DECISION_MS = 250  # not the default, so a dropped keyword changes the battle
PROBE = "the same text, hashed in two processes"


def trajectory(factory, seed: int, actions: list[list[int]] | None = None, steps: int = STEPS):
    """Play one episode from ``factory``; choose actions from ``seed`` unless given them."""
    env = factory()
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    played = []
    step_digests = []
    deployed = dict.fromkeys(AGENTS, 0)
    nonzero_rewards = 0
    most_entities = 0
    for t in range(steps if actions is None else len(actions)):
        if actions is None:
            chosen = []
            for agent in AGENTS:
                legal = np.flatnonzero(obs[agent]["action_mask"][1:]) + 1
                play = legal.size and rng.random() < 0.3
                chosen.append(int(rng.choice(legal)) if play else 0)
        else:
            chosen = actions[t]
        played.append(chosen)
        obs, rewards, terms, truncs, infos = env.step(dict(zip(AGENTS, chosen, strict=True)))
        state = env.engine.state()
        h = hashlib.sha256()
        for agent in AGENTS:
            for key in sorted(obs[agent]):
                arr = np.ascontiguousarray(obs[agent][key])
                h.update(f"{agent}.{key}{arr.dtype}{arr.shape}".encode())
                h.update(arr.tobytes())
            h.update(repr((rewards[agent], terms[agent], truncs[agent])).encode())
            h.update(repr(infos[agent]["deploy_status"]).encode())
            deployed[agent] += infos[agent]["deploy_status"] == 0
        h.update(msgspec.msgpack.encode(state))
        h.update(str(env.engine.state_hash()).encode())
        step_digests.append(h.hexdigest())
        nonzero_rewards += any(rewards[a] != 0 for a in AGENTS)
        most_entities = max(most_entities, len(state.entities))
        if terms["blue"] or truncs["blue"]:
            obs, _ = env.reset()
    return {
        "actions": played,
        "digests": step_digests,
        "deployed": deployed,
        "nonzero_rewards": nonzero_rewards,
        "most_entities": most_entities,
        "hash_probe": hash(PROBE),
        "hashseed": os.environ.get("PYTHONHASHSEED", "unset"),
        "pid": os.getpid(),
    }


def child_hash_seed() -> str:
    """A random hash seed for the child, never the one this process was given."""
    mine = os.environ.get("PYTHONHASHSEED")
    while True:
        seed = str(random.SystemRandom().randrange(1, 2**32 - 1))
        if seed != mine:
            return seed


def run_child(path) -> list[dict]:
    env = {**os.environ, "PYTHONHASHSEED": child_hash_seed()}
    done = subprocess.run(
        [sys.executable, os.path.abspath(__file__), str(path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


ENGINES = {"mock": MockEngine, "rust": RustEngine}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Each available engine's battle here and in ONE child, which pays one startup."""
    kinds = ["mock", "rust"] if core_available() else ["mock"]
    recipes, here = [], {}
    for kind in kinds:
        factory = EnvFactory(
            engine=ENGINES[kind], obs_builder=SpatialObsBuilder, decision_ms=DECISION_MS
        )
        here[kind] = trajectory(factory, SEED)
        recipes.append({"factory": factory, "actions": here[kind]["actions"]})
    path = tmp_path_factory.mktemp("recipes") / "recipes.pkl"
    path.write_bytes(pickle.dumps(recipes))
    there = dict(zip(kinds, run_child(path), strict=True))
    return {kind: (here[kind], there[kind]) for kind in kinds}


@pytest.mark.parametrize("kind", ["mock", pytest.param("rust", marks=needs_rust)])
def test_a_rebuilt_env_in_another_process_plays_the_same_battle(runs, kind):
    here, there = runs[kind]
    # Both seats played and the board filled, so the digest covers a battle and not towers.
    assert min(here["deployed"].values()) >= 3, here["deployed"]
    assert here["most_entities"] > 10
    assert here["nonzero_rewards"] > 0

    assert there["pid"] != here["pid"]
    assert there["hash_probe"] != here["hash_probe"], (
        f"the child hashed strings like this process (seed {there['hashseed']})"
    )
    assert there["actions"] == here["actions"]
    first = next(
        (
            i
            for i, (a, b) in enumerate(zip(here["digests"], there["digests"], strict=True))
            if a != b
        ),
        None,
    )
    assert first is None, f"the child's battle diverged at step {first} of {STEPS}"


def child(path: str) -> None:
    """The child's side: rebuild from each pickled recipe and replay the parent's actions."""
    with open(path, "rb") as fh:
        recipes = pickle.load(fh)
    out = [trajectory(r["factory"], SEED, actions=r["actions"]) for r in recipes]
    sys.stdout.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    child(sys.argv[1])
