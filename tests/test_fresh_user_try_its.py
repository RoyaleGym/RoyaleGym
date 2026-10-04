"""The release gate runs each sibling README's "Try it" from the installed wheels (phase P8).

A sibling's own CI runs from an editable checkout, so what only a wheel install does stays
invisible there: on 2026-10-02 RoyaleImitate's Try it was refused for every release install
while every CI was green. These tests pin how the gate reads a README: the first python block
under "## Try it", every total_steps=N cut to 1 (one update) and nothing else changed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "fresh_user_test.py"


def _tool():
    spec = importlib.util.spec_from_file_location("fresh_user_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


README = """# Pkg

## Install

```python
print("not this block")
```

## Try it

```python
teacher.learn(total_steps=20_000)
student.learn(total_steps=500)  # counts all steps
print("done")
```

```python
print("not the second block either")
```

## Next
"""


def test_the_try_it_block_is_read_and_only_its_sizes_are_cut():
    program, cuts = _tool().try_it_program(README)
    assert cuts == 2
    assert program == (
        "teacher.learn(total_steps=1)\n"
        "student.learn(total_steps=1)  # counts all steps\n"
        'print("done")\n'
    )


def test_a_readme_without_a_try_it_program_gives_none():
    tool = _tool()
    assert tool.try_it_program("# Pkg\n\n## Install\n\n```python\nx = 1\n```\n") is None
    # A Try it of shell lines only (RoyaleViser's) has no program either.
    assert tool.try_it_program("## Try it\n\n    royaleviser runs/\n\n## Next\n") is None
    # A python block under a LATER section is not the Try it's.
    assert tool.try_it_program("## Try it\n\nrun it\n\n## Next\n\n```python\nx\n```\n") is None


def test_the_gate_covers_every_sibling_with_a_program_and_names_the_cut():
    tool = _tool()
    assert set(tool.SIBLING_TRY_ITS) == {"royalesim", "royalelearn", "royaleimitate"}
    assert "P8" in tool.__doc__
    assert "total_steps" in tool.__doc__


def test_a_failed_install_is_blocked_only_when_the_engine_is_out_of_reach():
    """Since every package is on PyPI, pip names royalesim on every normal download. Only an
    engine pip cannot find, or would build from source with Rust, is BLOCKED; a timeout or any
    other failure is a FAIL, and a timeout says so."""
    tool = _tool()
    downloading = (
        "Collecting royalesim>=0.1.10\n  Downloading royalesim-0.1.10-cp310-abi3-win_amd64.whl"
    )
    assert tool.install_verdict(downloading + "\nTIMEOUT after 1800s", False) == (
        "FAIL", ["pip was still running at the time limit: a slow download, or a hang"]
    )
    assert tool.install_verdict(downloading + "\nERROR: some other failure", False)[0] == "FAIL"
    missing = "ERROR: No matching distribution found for royalesim>=0.1.10"
    assert tool.install_verdict(missing, False)[0] == "BLOCKED"
    source = (
        "Collecting royalesim\n  Building wheel for royalesim (pyproject.toml)\n  maturin failed"
    )
    assert tool.install_verdict(source, False)[0] == "BLOCKED"
    assert tool.install_verdict(missing, True)[0] == "FAIL", "with a wheels folder it is a FAIL"


def test_an_install_of_another_royalegym_version_is_refused():
    """``--expect-version``: the PyPI leg names the release it tests, and pip giving any other
    royalegym (an older one, before the release is published) fails P3 instead of testing the
    wrong package and reporting PASS."""
    tool = _tool()
    assert tool.version_check({"royalegym": "0.1.14", "royalesim": "0.1.11"}, "0.1.14") is None
    assert tool.version_check({"royalegym": "0.1.12"}, None) is None
    why = tool.version_check({"royalegym": "0.1.12", "royalesim": "0.1.11"}, "0.1.14")
    assert "0.1.12" in why
    assert "0.1.14" in why
    assert "0.1.14" in tool.version_check({}, "0.1.14")
