"""The README's "Try it" runs as pasted: one battle, a printed winner, a saved battle to watch.

The README is the short front page (40 lines at most). Its program is copied into an empty
folder and run there, as a reader runs it. What it prints depends on the engine, so the
check is on what it must do, not on the numbers.

SKIPS
    Without the engine. Not a pass.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from royalegym import load_trace
from royalegym.rust_engine import CORE_IMPORT_ERROR, core_available

README = Path(__file__).resolve().parents[1] / "README.md"


def _try_it() -> str:
    text = README.read_text(encoding="utf-8")
    section = text.split("## Try it", 1)[1]
    m = re.search(r"```python\n(.*?)```", section, re.S)
    assert m, 'the README\'s "Try it" has no python block'
    return m.group(1)


def test_the_readme_is_short():
    lines = README.read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 40, len(lines)


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_the_try_it_program_plays_a_battle_and_saves_it(tmp_path):
    program = _try_it()
    saved = re.search(r'save_to="([^"]+)"', program)
    assert saved, "the Try it program saves no battle to watch"
    done = subprocess.run(
        [sys.executable, "-c", program], cwd=tmp_path, capture_output=True, text=True, timeout=600
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert re.search(r"^winner \d crowns \[\d, \d\]$", done.stdout, re.M), done.stdout
    trace = load_trace(tmp_path / saved.group(1))
    assert trace.frames
    # And the README tells the reader the command that opens that file.
    assert f"royaleviser {saved.group(1)}" in README.read_text(encoding="utf-8")
