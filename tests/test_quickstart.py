"""examples/quickstart.py runs as a user runs it: one file, no arguments, on the engine.

The file is copied into an empty folder and run there with a smaller training budget, the
only change, so the check is on the file the user copies. It must train, save, play one
battle and point at the viewer with a battle that loads.

SKIPS
    Without the engine. Not a pass. A missing royalelearn.Learner is a failure: the quickstart
    needs it, and the [learn] extra pins a royalelearn that has it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from royalegym import load_trace
from royalegym.rust_engine import CORE_IMPORT_ERROR, core_available

QUICKSTART = Path(__file__).resolve().parents[1] / "examples" / "quickstart.py"
BUDGET = ("total_steps=200_000", "total_steps=4_000")


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_the_quickstart_trains_saves_and_points_at_a_battle_to_watch(tmp_path):
    source = QUICKSTART.read_text(encoding="utf-8")
    assert source.count(BUDGET[0]) == 1, "the quickstart's training budget line moved"
    script = tmp_path / "quickstart.py"
    script.write_text(source.replace(*BUDGET), encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(QUICKSTART.parents[1]), *filter(None, [env.get("PYTHONPATH")])]
    )
    done = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=900,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert "Watch it: royaleviser" in done.stdout, done.stdout[-2000:]
    battle = tmp_path / "my_bot_battle.msgpack"
    assert battle.exists()
    assert load_trace(battle).result is not None
    assert (tmp_path / "runs" / "quickstart" / "bot").exists()


def test_the_quickstart_is_one_short_file():
    lines = QUICKSTART.read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 60, f"{len(lines)} lines; the quickstart is meant to fit one screen"
