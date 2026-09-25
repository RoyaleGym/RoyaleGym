"""Every example program on the documentation site prints what its page says it prints.

WHY THIS EXISTS. Each page under docs/site/pages shows programs with their output right
under them, and a reader runs them first. Until 2026-09-25 nothing ran them automatically:
on 2026-09-24 two published programs had been crashing for a day and a half while every
gate stayed green, and running every example by hand the same evening found stale outputs
on six pages. This is that hand run as a test, one per page, with the rules the hand run
learned, so the next stale page goes red here instead of waiting for someone to look.

WHAT IT RUNS. Every ```python block on a page, each in a process of its own, in a scratch
folder, so nothing one block opens or writes reaches the next block or a checkout. A block
runs on its own first, the way a reader who copies only that block would run it. If that
fails with a NameError, it runs again after the blocks above it on the same page, the way a
page that says "add this to the program above" means it.

WHAT IT COMPARES. A block's stdout against the untagged fenced block right under it, with
nothing but blank lines between, line by line, trailing spaces ignored.
  - A block that prints nothing is not compared; the untagged block under it is usually a
    shell command. It must still run.
  - An output may be split: the first lines under the code, the rest in a LATER untagged
    block on the same page. That matches only if the remainder is exactly such a block.
  - A block with no output under it that does not run is an excerpt, not a program.

BUILDS. A page that stamps the engine build it was run on ("on engine build `<16 hex>`")
is compared on any build. The same output passes. A different output on the stamped build
fails; on another build it SKIPS, naming both, because a whole simulation legitimately ends
differently on a different build (the README test's rule). A page with no stamp makes an
unconditional claim, so any difference fails.

WHAT IT DOES NOT DO. It does not know whether an output is RIGHT, only whether the page
and a run agree. It does not check prose around an output. And it cannot compare a program
whose output depends on something outside the process; those are EXEMPT by name below,
with the reason, and a test fails if an exemption stops naming exactly one block.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import NamedTuple

import pytest

from royalegym.rust_engine import CORE_IMPORT_ERROR, core_available
from royalegym.rust_engine import build_digest as engine_build_digest

REPO = Path(__file__).resolve().parents[1]
PAGES = REPO / "docs" / "site" / "pages"
FENCE = re.compile(r"^( *)```(\w*)[^\n]*\n(.*?)^\1```", re.MULTILINE | re.DOTALL)
STAMP = re.compile(r"engine build\s+`([0-9a-f]{16})`")
MARK = "@@SITE_EXAMPLE_RESULT@@"

#: (page path suffix, text the block contains) -> why its output cannot be compared here.
EXEMPT = {
    ("first-bot.md", "ViserPublisher()"): (
        "its count depends on when a viewer attaches, which no single process controls"
    ),
    ("pieces/viewer.md", '"shot.png"'): (
        "prints an image's size in bytes, which moves with the viewer's drawing and the "
        "platform's zlib; the page says yours may differ. RoyaleViser's tests cover capture"
    ),
    ("pieces/viewer.md", '"clip.gif"'): (
        "prints a gif's size in bytes, which moves with the drawing and the ffmpeg build, "
        "and needs the viewer's media extra; RoyaleViser's tests cover capture"
    ),
}

#: An address already in use is the machine's state, not the page's: a published example
#: binds the viewer's fixed port, and on a shared machine a viewer is often holding it.
PORT_BUSY = (
    "WinError 10048",
    "EADDRINUSE",
    "Address already in use",
    "Only one usage of each socket address",
)

# Receives {"prefix": [...], "code": ...}: the prefix runs first with its output discarded.
WORKER = r'''
import contextlib, io, json, sys, traceback, warnings
warnings.simplefilter("ignore")
job = json.load(sys.stdin)
ns = {"__name__": "__main__"}
try:
    with contextlib.redirect_stdout(io.StringIO()):
        for code in job["prefix"]:
            exec(compile(code, "<block above>", "exec"), ns)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(job["code"], "<block>", "exec"), ns)
    res = {"how": "ok", "out": out.getvalue()}
except NameError:
    res = {"how": "name_error", "err": traceback.format_exc()[-700:]}
except BaseException:
    res = {"how": "error", "err": traceback.format_exc()[-700:]}
print(job["mark"] + json.dumps(res))
'''


class Verdict(NamedTuple):
    """One page: what failed, what could not be checked, and a line per block."""

    failures: list[str]
    unchecked: list[str]
    lines: list[str]


def norm(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip("\n").splitlines()).strip()


def exemption(page: Path, body: str) -> str | None:
    return next(
        (why for (suffix, marker), why in EXEMPT.items()
         if page.as_posix().endswith(suffix) and marker in body),
        None,
    )


def fences(text: str) -> list[tuple[int, int, str, str]]:
    return [
        (m.start(), m.end(), m.group(2), textwrap.dedent(m.group(3)))
        for m in FENCE.finditer(text)
    ]


def run(prefix: list[str], code: str, cwd: Path) -> dict:
    job = json.dumps({"prefix": prefix, "code": code, "mark": MARK})
    done = subprocess.run(
        [sys.executable, "-c", WORKER], input=job, capture_output=True, text=True,
        timeout=600, cwd=cwd, check=False,
    )
    at = done.stdout.rfind(MARK)
    if at < 0:
        return {"how": "error", "err": "the worker died:\n" + done.stderr[-700:]}
    return json.loads(done.stdout[at + len(MARK):])


def check_page(page: Path, text: str, engine: str | None, cwd: Path) -> Verdict:
    """Run every python block on one page and hold each to the output under it."""
    fs = fences(text)
    stamps = sorted(set(STAMP.findall(text)))
    moved = bool(stamps) and engine not in stamps
    failures: list[str] = []
    unchecked: list[str] = []
    lines: list[str] = []
    ran: list[str] = []
    for i, (pos, end, lang, body) in enumerate(fs):
        if lang != "python":
            continue
        where = f"{page.name}:{text.count(chr(10), 0, pos) + 1}"
        nxt = fs[i + 1] if i + 1 < len(fs) else None
        expected = nxt[3] if nxt and nxt[2] == "" and not text[end:nxt[0]].strip() else None
        why = exemption(page, body)
        if why is not None:
            lines.append(f"exempt {where}: {why}")
            continue
        r, after = run([], body, cwd), ""
        if r["how"] == "name_error" and ran:
            r, after = run(ran, body, cwd), " (runs only after the blocks above it)"
        if r["how"] != "ok":
            busy = next((n for n in PORT_BUSY if n in r["err"]), None)
            if busy is not None:
                unchecked.append(f"{where} binds a port something here already holds ({busy})")
            elif expected is None:
                lines.append(f"excerpt {where}: does not run, no output block under it")
            else:
                failures.append(f"{where} raises{after}:\n{textwrap.indent(r['err'], '    ')}")
            continue
        ran.append(body)
        got = norm(r["out"])
        later = [norm(f[3]) for f in fs[i + 2:] if f[2] == ""]
        if not got:
            lines.append(f"prints nothing {where}{after}")
        elif expected is None:
            lines.append(f"no output block {where}{after}")
        elif got == norm(expected):
            lines.append(f"ok {where}{after}")
        elif got.startswith(norm(expected) + "\n") and norm(got[len(norm(expected)):]) in later:
            lines.append(f"ok split {where}{after}")
        else:
            shown = (
                f"{where} prints something else{after}\n  page:\n"
                f"{textwrap.indent(norm(expected), '    ')}\n  run:\n{textwrap.indent(got, '    ')}"
            )
            if moved:
                unchecked.append(
                    f"{shown}\n  The page stamps build {', '.join(stamps)} and this engine is "
                    f"{engine}: re-run it on a committed build and re-pin."
                )
            else:
                failures.append(shown)
    return Verdict(failures, unchecked, lines)


def site_pages() -> list[Path]:
    return sorted(PAGES.rglob("*.md"))


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
@pytest.mark.parametrize(
    "page", site_pages(), ids=[p.relative_to(PAGES).as_posix() for p in site_pages()]
)
def test_every_example_on_the_page_prints_what_the_page_says(page: Path, tmp_path: Path) -> None:
    verdict = check_page(page, page.read_text(encoding="utf-8"), engine_build_digest(), tmp_path)
    assert not verdict.failures, "\n\n".join(verdict.failures)
    if verdict.unchecked:
        pytest.skip("SKIPPED, NOT PASSED: " + "\n\n".join(verdict.unchecked))


def test_every_exemption_still_names_exactly_one_block() -> None:
    """An exemption that matches nothing protects nothing, and one that matches two hides one."""
    counts = {key: 0 for key in EXEMPT}
    for page in site_pages():
        for _pos, _end, lang, body in fences(page.read_text(encoding="utf-8")):
            if lang != "python":
                continue
            for suffix, marker in EXEMPT:
                if page.as_posix().endswith(suffix) and marker in body:
                    counts[(suffix, marker)] += 1
    wrong = {k: n for k, n in counts.items() if n != 1}
    assert not wrong, f"exemptions not naming exactly one block: {wrong}"


# -- the checker itself, on a page built to exercise every rule ---------------------

SELFTEST_PAGE = '''\
```python
print("same")
```

```
same
```

```python
print("new")
```

```
old
```

```python
x = 1
```

```
python -m something
```

```python
def helper():
    return 41
```

```python
print(helper() + 1)
```

```
42
```

```python
print("first")
print("second")
```

```
first
```

The second half of that program printed this:

```
second
```

```python
print("far")
```

Some prose, then a command that is not this program's output:

```
far
```

```python
import socket
a = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
a.bind(("127.0.0.1", 0))
b = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
b.bind(("127.0.0.1", a.getsockname()[1]))
print("never")
```

```
never
```

```python
class Thing(Base): ...
```
'''


def test_the_checker_sorts_every_kind_of_block(tmp_path: Path) -> None:
    """Each rule above, on one block made to hit it, including the one that must FAIL."""
    page = tmp_path / "page.md"
    v = check_page(page, SELFTEST_PAGE, None, tmp_path)
    assert [line.split(" page.md")[0] for line in v.lines] == [
        "ok",
        "prints nothing",
        "prints nothing",
        "ok",
        "ok split",
        "no output block",
        "excerpt",
    ], v.lines
    assert v.lines[3].endswith("(runs only after the blocks above it)"), v.lines[3]
    assert len(v.failures) == 1, v.failures
    assert "prints something else" in v.failures[0], v.failures
    assert len(v.unchecked) == 1, v.unchecked
    assert "binds a port" in v.unchecked[0], v.unchecked


def test_a_stamped_page_on_another_build_skips_instead_of_failing(tmp_path: Path) -> None:
    text = "Run on engine build `0123456789abcdef` here.\n\n" + SELFTEST_PAGE
    v = check_page(tmp_path / "page.md", text, "fedcba9876543210", tmp_path)
    assert not v.failures, v.failures
    assert any("re-pin" in u for u in v.unchecked), v.unchecked
    same = check_page(tmp_path / "page.md", text, "0123456789abcdef", tmp_path)
    assert len(same.failures) == 1, "on the stamped build a changed output must fail"


@pytest.mark.skipif(not core_available(), reason=str(CORE_IMPORT_ERROR))
def test_plant_a_real_page_with_one_output_line_changed_fails(tmp_path: Path) -> None:
    """The self-test page proves the rules; this proves they reach a real page.

    rewards.md stamps no build, so a changed output on it must FAIL whatever the engine.
    """
    page = PAGES / "rewards.md"
    text = page.read_text(encoding="utf-8")
    assert not STAMP.search(text), "rewards.md stamps a build now; plant on an unstamped page"
    fs = fences(text)
    at = next(
        i for i, f in enumerate(fs[:-1])
        if f[2] == "python" and fs[i + 1][2] == "" and not text[f[1]:fs[i + 1][0]].strip()
        and exemption(page, f[3]) is None
    )
    out_start = text.index(chr(10), fs[at + 1][0]) + 1  # first line inside the output fence
    planted = text[:out_start] + "PLANTED " + text[out_start:]
    v = check_page(page, planted, engine_build_digest(), tmp_path)
    assert v.failures, "PLANT DID NOT LAND: a changed output on rewards.md passed"
    assert "PLANTED" in v.failures[0], v.failures[0]
