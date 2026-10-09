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
page that says "add this to the program above" means it. A block the page tells the reader to
save as a file carries the name in its fence (```python title="my_bot.py"), and is saved under
that name in the page's folder first, so a later block that imports it runs.

ONLY ```python IS RUN. Python under any other tag (```py, ```python3, an untagged block with
import lines) would be published unchecked, so a test refuses it: on 2026-10-02 thirty ```py
blocks, a whole new page among them, had never been run.

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
FENCE = re.compile(r"^( *)```(\w*)([^\n]*)\n(.*?)^\1```", re.MULTILINE | re.DOTALL)
#: A fence's file name, as mkdocs-material shows it above the code: ```python title="x.py"
TITLE = re.compile(r'\btitle="([^"/\\]+\.py)"')
#: "engine" and "build" may sit on two lines: rewards.md wraps there, and a pattern with
#: one literal space read that page as unstamped (found by the docs session).
STAMP = re.compile(r"engine\s+build\s+`([0-9a-f]{16})`")
MARK = "@@SITE_EXAMPLE_RESULT@@"

#: (page path suffix, text the block contains, or its fence title as title="x.py") -> why it is
#: not run here. A full-size training run is a program a reader runs for hours; a block that
#: loads the bot such a run saves cannot run without it.
EXEMPT = {
    ("quickstart.md", 'title="quickstart.py"'): (
        "trains until Ctrl+C (a billion-step limit); tests/test_quickstart.py and the "
        "fresh-user test run this same file at a test's size"
    ),
    ("quickstart.md", 'title="watch.py"'): "loads runs/quickstart, which quickstart.py trains",
    ("faq.md", 'title="how_good.py"'): "loads runs/quickstart, which quickstart.py trains",
    ("clash-royale/training-an-agent.md", "learner.learn()"): "trains until Ctrl+C",
    ("clash-royale/training-an-agent.md", 'title="watch_my_bot.py"'): (
        "loads runs/my_bot, which my_bot.py trains"
    ),
    ("clash-royale/cloning-a-bot.md", 'title="clone_my_bot.py"'): (
        "records 200 battles (about 2 min) and clones them (about 30 min on a CPU)"
    ),
    ("clash-royale/cloning-a-bot.md", 'title="watch_clone.py"'): (
        "loads runs/human_clone, which clone_humans.py makes"
    ),
    ("clash-royale/cloning-a-bot.md", 'title="test_clone.py"'): (
        "loads runs/human_clone, which clone_humans.py makes, and plays 300 battles"
    ),
    ("clash-royale/cloning-a-bot.md", 'title="train_from_clone.py"'): (
        "trains from runs/human_clone until Ctrl+C"
    ),
    ("clash-royale/cloning-a-bot.md", 'title="clone_humans.py"'): (
        "downloads IL_Replay and replays 1,000 human matches (about 15 min, 1,346 read) and "
        "clones them (about 80 min on a shared GPU; the page's lines are from that run, 2026-10-09)"
    ),
    ("clash-royale/cloning-a-bot.md", 'title="clone_again.py"'): (
        "clones the matches clone_humans.py wrote in runs/human-demos (hours on a CPU); "
        "checked by hand on 100 matches, 2026-10-09"
    ),
    ("resources/royaleimitate.md", "total_steps=100_000"): "trains a student for 100,000 steps",
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

#: Pages whose MockEngine programs need a RoyaleSim checkout, and that say so to the reader:
#: MockEngine reads the 2018 tables from one, and an installed engine carries none. Run from
#: the release wheels alone, such a block cannot run; it is reported unchecked with this
#: reason, never as a pass, and the same error on any page not listed here still fails.
NEEDS_CHECKOUT = {"build-from-source.md": "a RoyaleSim checkout (the page's Step 2)"}
#: The start of MockEngine's refusal when that data is missing (mock_engine._load_cards).
MOCK_DATA_MISSING = "MockEngine reads the 2018 card tables from"

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


def names(marker: str, body: str, title: str | None) -> bool:
    return marker in body or (title is not None and marker == f'title="{title}"')


def exemption(page: Path, body: str, title: str | None = None) -> str | None:
    return next(
        (why for (suffix, marker), why in EXEMPT.items()
         if page.as_posix().endswith(suffix) and names(marker, body, title)),
        None,
    )


#: A whole line ``--8<-- "path"`` (pymdownx.snippets) includes that file, from the repo root
#: as docs/site/mkdocs.yml sets ``base_path``.
SNIPPET = re.compile(r'^[ \t]*--8<--[ \t]+"([^"]+)"[ \t]*$', re.M)


def expand_snippets(body: str) -> str:
    """The block as the site shows it: each snippet line replaced by its file. A missing file
    raises, as the site's strict build does."""
    return SNIPPET.sub(lambda m: (REPO / m.group(1)).read_text(encoding="utf-8").rstrip("\n"), body)


class Fence(NamedTuple):
    start: int
    end: int
    lang: str
    body: str
    #: The file the page tells the reader to save this block as (its ``title="x.py"``).
    title: str | None


def fences(text: str) -> list[Fence]:
    out = []
    for m in FENCE.finditer(text):
        named = TITLE.search(m.group(3))
        out.append(
            Fence(
                m.start(), m.end(), m.group(2), textwrap.dedent(m.group(4)),
                named.group(1) if named else None,
            )
        )
    return out


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
    saved: set[str] = set()
    for i, (pos, end, lang, raw, title) in enumerate(fs):
        if lang != "python":
            continue
        where = f"{page.name}:{text.count(chr(10), 0, pos) + 1}"
        body = expand_snippets(raw)
        if title is not None:
            # The page tells the reader to save this block under that name, and later blocks
            # import it: save it in the page's folder before anything runs it, exempt or not.
            # Two blocks with one name are one file, in page order.
            with (cwd / title).open("a" if title in saved else "w", encoding="utf-8") as f:
                f.write(("\n" if title in saved else "") + body)
            saved.add(title)
        nxt = fs[i + 1] if i + 1 < len(fs) else None
        expected = nxt[3] if nxt and nxt[2] == "" and not text[end:nxt[0]].strip() else None
        why = exemption(page, raw, title)
        if why is not None:
            lines.append(f"exempt {where}: {why}")
            continue
        r, after = run([], body, cwd), ""
        if r["how"] == "name_error" and ran:
            r, after = run(ran, body, cwd), " (runs only after the blocks above it)"
        if r["how"] != "ok":
            busy = next((n for n in PORT_BUSY if n in r["err"]), None)
            checkout = NEEDS_CHECKOUT.get(page.name) if MOCK_DATA_MISSING in r["err"] else None
            if busy is not None:
                unchecked.append(f"{where} binds a port something here already holds ({busy})")
            elif checkout is not None:
                unchecked.append(f"{where} needs {checkout}, and this run has none")
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
    """The pages written for the site. pages/repos/ is left out: collect.py copies it in from
    every repo's docs/ at build time, and each repo checks its own docs."""
    return sorted(p for p in PAGES.rglob("*.md") if "repos" not in p.relative_to(PAGES).parts[:1])


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
        for _pos, _end, lang, body, title in fences(page.read_text(encoding="utf-8")):
            if lang != "python":
                continue
            for suffix, marker in EXEMPT:
                if page.as_posix().endswith(suffix) and names(marker, body, title):
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

    Judged as on the build the page stamps, where a changed output must FAIL, beside a
    blind control: the same page unchanged, judged the same way, must not fail. The first
    version of this plant asserted rewards.md was unstamped. It is stamped, across a line
    break, and the plant passed only because the stamp pattern missed that stamp. rewards.md
    went with the 2026-10-01 rewrite; Game Values is the stamped page now.
    """
    page = PAGES / "cheatsheets" / "game-values.md"
    text = page.read_text(encoding="utf-8")
    stamps = STAMP.findall(text)
    assert stamps, "game-values.md no longer stamps a build; pick a stamped page for this plant"
    control = check_page(page, text, stamps[0], tmp_path)
    assert not control.failures, (
        "the control failed: game-values.md is stale on this engine, so the plant would prove "
        f"nothing\n{control.failures[0]}"
    )
    fs = fences(text)
    at = next(
        i for i, f in enumerate(fs[:-1])
        if f[2] == "python" and fs[i + 1][2] == "" and not text[f[1]:fs[i + 1][0]].strip()
        and exemption(page, f[3]) is None
    )
    out_start = text.index(chr(10), fs[at + 1][0]) + 1  # first line inside the output fence
    planted = text[:out_start] + "PLANTED " + text[out_start:]
    v = check_page(page, planted, stamps[0], tmp_path)
    assert v.failures, "PLANT DID NOT LAND: a changed output on game-values.md passed"
    assert "PLANTED" in v.failures[0], v.failures[0]


TITLED_PAGE = '''\
Make a file called `helper_mod.py`:

```python title="helper_mod.py"
VALUE = 41
```

Then, next to it:

```python
from helper_mod import VALUE
print(VALUE + 1)
```

```
42
```
'''


def test_a_titled_block_is_saved_as_its_file_for_the_blocks_after_it(tmp_path: Path) -> None:
    """A page that says "make a file called helper_mod.py" names it in the fence's title
    (``python title="helper_mod.py"``, which the site shows above the code). The block is saved
    under that name in the page's folder, so a later block that imports it runs as a reader's
    would. Control: the same page with no title must fail on the import."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    titled = check_page(tmp_path / "page.md", TITLED_PAGE, None, tmp_path / "a")
    assert not titled.failures, titled.failures
    assert [line.split(" page.md")[0] for line in titled.lines] == ["prints nothing", "ok"]
    untitled = TITLED_PAGE.replace(' title="helper_mod.py"', "")
    control = check_page(tmp_path / "page.md", untitled, None, tmp_path / "b")
    assert control.failures, control.lines
    assert "helper_mod" in control.failures[0], control.failures


def test_a_snippet_line_is_the_file_it_names() -> None:
    """quickstart.md shows examples/quickstart.py through a snippet line; the block that runs
    is the file, and a snippet of a missing file fails like the site's strict build."""
    shown = expand_snippets('--8<-- "examples/quickstart.py"\n')
    whole = (REPO / "examples" / "quickstart.py").read_text(encoding="utf-8")
    assert shown == whole.rstrip("\n") + "\n"  # the block's own line end stays
    with pytest.raises(FileNotFoundError):
        expand_snippets('--8<-- "examples/no_such_file.py"\n')


#: Fence tags a reader's Python could hide under. The checker runs ``python`` blocks only, so a
#: program under any of these would be published unchecked (cloning-a-bot.md, 2026-10-02).
OTHER_PYTHON_TAGS = {"py", "py3", "python3", "pycon", "ipython"}
IMPORT_LINE = re.compile(r"^\s*(from\s+[\w.]+\s+import\s|import\s+[\w.]+\s*$)", re.M)


def unchecked_python(page: Path) -> list[str]:
    """Where a page shows Python the checker would not run: an other-Python tag, or Python
    import lines in a block tagged anything but python."""
    text = page.read_text(encoding="utf-8")
    out = []
    for f in fences(text):
        where = f"{page.relative_to(PAGES).as_posix()}:{text.count(chr(10), 0, f[0]) + 1}"
        if f[2].lower() in OTHER_PYTHON_TAGS or (f[2] != "python" and f[2].lower() == "python"):
            out.append(f"{where}: ```{f[2]}: tag it ```python so it is run")
        elif f[2] != "python" and IMPORT_LINE.search(f[3]):
            out.append(f"{where}: Python under ```{f[2] or '(no tag)'}: tag it ```python")
    return out


def test_no_python_on_the_site_hides_under_another_tag() -> None:
    hidden = [w for page in site_pages() for w in unchecked_python(page)]
    assert not hidden, "\n".join(hidden)


def test_the_hidden_python_guard_sees_both_kinds(tmp_path: Path) -> None:
    page = PAGES / "_guard_selftest.md"
    text = "```py\nprint(1)\n```\n\n```\nfrom royalegym import make_env\n```\n\n```python\nx\n```\n"
    try:
        page.write_text(text, encoding="utf-8")
        found = unchecked_python(page)
    finally:
        page.unlink()
    assert len(found) == 2, found


def test_a_stamp_wrapped_across_a_line_is_still_a_stamp() -> None:
    assert STAMP.findall("run on engine\nbuild `0123456789abcdef` today") == ["0123456789abcdef"]
    page = PAGES / "cheatsheets" / "game-values.md"
    assert STAMP.findall(page.read_text(encoding="utf-8")), "game-values.md's stamp is not found"


MISSING_DATA_PAGE = """```python
raise FileNotFoundError(
    "MockEngine reads the 2018 card tables from /x/csv_logic, and characters.csv is not there."
)
```

```
227 cards in the stand-in
```
"""


def test_a_checkout_only_example_run_from_wheels_is_declared_not_failed(tmp_path: Path) -> None:
    """build-from-source.md's MockEngine program needs a RoyaleSim checkout (its Step 2):
    MockEngine reads the 2018 tables from one, and an installed engine carries none. Run from
    the release wheels alone it cannot run, and that is reported as unchecked with the reason,
    never as a pass. The same error on any other page is still a failure."""
    page = tmp_path / "build-from-source.md"
    v = check_page(page, MISSING_DATA_PAGE, None, tmp_path)
    assert not v.failures, v.failures
    assert len(v.unchecked) == 1, v.unchecked
    assert "needs a RoyaleSim checkout" in v.unchecked[0], v.unchecked
    other = check_page(tmp_path / "page.md", MISSING_DATA_PAGE, None, tmp_path)
    assert len(other.failures) == 1, "on a page that does not need a checkout it must fail"
