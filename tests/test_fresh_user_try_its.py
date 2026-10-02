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
