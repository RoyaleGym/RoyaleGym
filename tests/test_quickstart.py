"""examples/quickstart.py runs as a user runs it: one file, no arguments, on the engine.

The file is copied into an empty folder and run there at a test's size (``SMALL``: the only
changes), so the check is on the file the user copies. It must train and save a checkpoint.
Watching a battle is the README's "Try it" and its own test.

SKIPS
    Without the engine. Not a pass. A missing royalelearn.Learner is a failure: the quickstart
    needs it, and the [learn] extra pins a royalelearn that has it.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from royalegym.rust_engine import CORE_IMPORT_ERROR, core_available

QUICKSTART = Path(__file__).resolve().parents[1] / "examples" / "quickstart.py"
#: The literals swapped for a run that finishes in a test: a few battles on the CPU (CI has
#: no GPU, and here the GPU belongs to training), two small updates, and no live viewer
#: (its port may be held by a real run). Each must occur exactly once, so a moved setting
#: fails here instead of silently running at full size. tools/fresh_user_test.py uses the
#: same table.
SMALL = {
    "n_envs=32,": "n_envs=4,",
    'device="auto",': 'device="cpu",',
    "steps_per_update=16_384,": "steps_per_update=512,",
    "ppo_batch_size=16_384,": "ppo_batch_size=512,",
    "ppo_minibatch_size=2_048,": "ppo_minibatch_size=256,",
    "checkpoint_every=200_000,": "checkpoint_every=512,",
    "timestep_limit=1_000_000_000,": "timestep_limit=1_024,",
    "viser=True,": "viser=False,",
}


def small(source: str) -> str:
    for big, little in SMALL.items():
        assert source.count(big) == 1, f"the quickstart's {big!r} moved or repeats"
        source = source.replace(big, little)
    return source


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_the_quickstart_trains_and_saves_a_checkpoint(tmp_path):
    script = tmp_path / "quickstart.py"
    script.write_text(small(QUICKSTART.read_text(encoding="utf-8")), encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(QUICKSTART.parents[1]), *filter(None, [env.get("PYTHONPATH")])]
    )
    done = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=900,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert re.search(r"^update +2 +steps +1,024", done.stdout, re.M), done.stdout[-2000:]
    saved = list((tmp_path / "runs" / "quickstart" / "checkpoints").iterdir())
    assert saved, "no checkpoint written"


def test_build_env_names_every_config_object():
    """The quickstart shows the pieces rather than hiding them behind a helper."""
    source = QUICKSTART.read_text(encoding="utf-8")
    for piece in (
        "RustEngine()", "state_mutator=", "obs_builder=", "action_parser=", "reward_fn=",
        "termination_cond=", "truncation_cond=",
    ):
        assert piece in source, piece
    assert "make_env" not in source


def test_the_quickstart_is_one_file_of_one_screenful_or_two():
    lines = QUICKSTART.read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 100, f"{len(lines)} lines"


def test_the_quickstart_does_not_force_the_cpu():
    """Users are assumed to have a GPU: the Learner picks it when torch can use one, and
    falls back to the CPU with a printed line otherwise (CI, the fresh-user test)."""
    source = QUICKSTART.read_text(encoding="utf-8")
    assert 'device="cpu"' not in source
    assert 'device="cuda"' not in source, "with no GPU it would fail, not fall back"


def test_the_fresh_user_test_sizes_the_quickstart_the_same_way():
    """tools/fresh_user_test.py runs the quickstart at a test's size too. Its table is this one
    without the forced CPU and the viewer switch, which it keeps as a user has them."""
    import importlib.util

    path = QUICKSTART.parents[1] / "tools" / "fresh_user_test.py"
    spec = importlib.util.spec_from_file_location("fresh_user_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    theirs = dict(module.SMALL)
    ours = {k: v for k, v in SMALL.items() if not k.startswith(("device=", "viser=", "timestep_"))}
    assert theirs == ours
