r"""No file in this repo should contain a character that renders as nothing.

WHY THIS EXISTS
    This cost an hour on 2026-09-22. A regex written as ``[0-9a-f]{7,40}\b`` went into a
    file through a shell heredoc, where ``\b`` became a real BACKSPACE, 0x08. The line
    displayed identically in the editor, in ``git diff``, in ``inspect.getsource`` and in
    the failure message; the regex simply never matched, because it required a backspace
    the subject never has. Reading the code could not find it. Dumping the bytes could.
    The project has now had that shape four times, and two of the four were in pages
    describing the trap.

WHY FOUR CLASSES AND NOT ONE COUNT
    They are different claims with different causes, so a single number would merge
    evidence about a heredoc with evidence about a paste. A C0 control almost always
    means a shell ate a backslash escape. A zero-width space or a non-breaking space
    almost always means someone copied from a rendered page. Kept apart, a failure tells
    you where to look; merged, it only tells you to go looking.

WHY THERE IS NO NUMERIC BOUND
    Because every bound this project wrote was wrong. ``< 0x20`` missed DEL; I reported
    that; Viser 1 then probed further and found it also missed the whole C1 block
    (0x80-0x9f), so the one-character fix for DEL would have left C1 open and looked
    complete. The bound was also applied to BYTES, and a C1 character is two bytes in
    UTF-8 (0xc2 0x85), neither below 0x20 nor equal to 0x7f, so it could not have been
    seen however it was read. Membership is therefore asked of ``unicodedata``, and this
    reads decoded characters rather than bytes. Docs 1 reached the same place on the
    canonical checker.

WIDENING A GUARD CAN BREAK IT IN THE OTHER DIRECTION
    A rule that fires on ordinary accented prose is a rule somebody switches off, and
    then it is protecting nothing. So the accepting set is pinned too: accents, an em
    dash, CJK, Cyrillic and a tab all have to survive every class here.

WHAT IT COVERS
    Every text file in the repo, not only ``.py``. The pages are written through heredocs
    constantly and that is where the trap lives. This does mean an edit by another
    session to a page can turn this red. That is intended and it is a different thing
    from the test-count badge that had to stop chasing the suite: this fails only on a
    character that is actually in the tree and should not be, never on a number that has
    merely moved on.
"""

from __future__ import annotations

import functools
import unicodedata
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: Suffixes that hold text a person reads or a program parses. Binary files are excluded
#: because an 0x08 in a PNG is a pixel.
TEXT_SUFFIXES = (".py", ".md", ".toml", ".cfg", ".txt", ".json", ".yml", ".yaml")

#: Generated and vendored trees. Scoped the way the rest of this project scopes scans:
#: ``target/`` and ``.venv`` hold other people's sources and would drown the signal.
SKIP_PARTS = {".venv", "__pycache__", "build", ".git", "node_modules", ".pytest_cache", "target"}

#: Tab, newline and carriage return are the only invisible characters a text file has a
#: reason to contain. Everything else here arrived by accident.
LEGITIMATE = (0x09, 0x0A, 0x0D)

#: Each class is a separate claim, and membership is asked of ``unicodedata`` rather than
#: written as a numeric range. Docs 1's point, and it is the better one: a bound encodes a
#: guess about where trouble lives, and this project got that guess wrong twice in one
#: evening. ``< 0x20`` missed DEL, then missed the whole C1 block, and a one-character fix
#: for the first would have left the second open while looking complete. The category
#: asks instead of guessing.
#:
#: NBSP is named explicitly because it is in neither category -- it is Zs, a space -- and
#: it looks exactly like the space it is not.
CLASSES = {
    "control character": lambda ch: unicodedata.category(ch) == "Cc" and ord(ch) not in LEGITIMATE,
    "invisible formatting character": lambda ch: unicodedata.category(ch) == "Cf",
    "non-breaking space": lambda ch: ord(ch) == 0x00A0,
}

#: Text that MUST survive every class above. A rule that fires on ordinary accented prose
#: is a rule somebody switches off, so the accepting set is pinned rather than assumed:
#: the danger in widening a guard is not only that it misses something.
LEGITIMATE_TEXT = (
    "café and café",  # combining accent, and the precomposed form
    "an em dash — and an ellipsis …",
    "中文 and Русский",
    "a tab\there, a newline\nhere",
)


@functools.lru_cache(maxsize=1)
def text_files() -> tuple[Path, ...]:
    found: list[Path] = []
    for path in sorted(REPO.rglob("*")):
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        found.append(path)
    return tuple(found)


@functools.lru_cache(maxsize=1)
def decoded() -> tuple[dict[str, str], list[str]]:
    """Every text file as a string, plus the ones that could not be read as UTF-8.

    The unreadable list is returned rather than swallowed because a check that silently
    skips part of its subject still reports on the whole subject and still looks healthy.
    """
    ok: dict[str, str] = {}
    unreadable: list[str] = []
    for path in text_files():
        # as_posix(), not str(): on Windows str() gives backslashes, and the first
        # version of the sweep test below compared them against forward-slash literals
        # and so could never match a nested path. It was red on a clean tree and the
        # plant run is what showed it, which is the argument for planting.
        rel = path.relative_to(REPO).as_posix()
        try:
            ok[rel] = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            unreadable.append(f"{rel}: {exc}")
    return ok, unreadable


@pytest.mark.parametrize("kind", sorted(CLASSES))
def test_no_text_file_carries(kind: str) -> None:
    readable, unreadable = decoded()
    # Asserted HERE rather than only in its own test. If a file cannot be decoded, this
    # class was not checked against it, and a pass would be a statement about a space
    # this never looked at.
    assert not unreadable, (
        f"these files are not valid UTF-8, so the {kind} check did not look at them and "
        f"cannot report on them: {unreadable}"
    )

    member = CLASSES[kind]
    found: dict[str, list[str]] = {}
    for rel, text in readable.items():
        for i, ch in enumerate(text):
            if member(ch):
                # chr(10) rather than a newline escape, so that this line cannot be
                # destroyed by the very thing it is counting.
                line = text[:i].count(chr(10)) + 1
                found.setdefault(rel, []).append(f"U+{ord(ch):04X} at line {line}")
    assert not found, (
        f"{kind} characters render as nothing and are almost never meant: {found}. A "
        "shell heredoc turns a backslash escape into the byte it names, so write the "
        "file with an editing tool or use a quoted heredoc."
    )


@pytest.mark.parametrize("kind", sorted(CLASSES))
@pytest.mark.parametrize("sample", LEGITIMATE_TEXT)
def test_legitimate_text_is_not_refused(kind: str, sample: str) -> None:
    """The accepting set, which is the half a widened guard is most likely to get wrong.

    Checking only that the rule catches bad characters cannot distinguish a working rule
    from one that refuses everything, and the second kind gets switched off within a day.
    """
    member = CLASSES[kind]
    refused = [f"U+{ord(ch):04X}" for ch in sample if member(ch)]
    assert not refused, (
        f"the {kind!r} rule refuses {refused} in ordinary text {sample!r}. A guard that "
        "fires on accented prose or a tab is one a contributor turns off."
    )


def test_the_sweep_actually_reaches_this_repos_files() -> None:
    """Four passing emptiness checks look the same whether the scan found 105 files or 0.

    So this pins that the sweep reaches real files, and specifically that it reaches the
    pages as well as the code: the widening from ``.py`` to every text file is the point
    of this file and a glob that quietly stopped matching ``.md`` would take it away
    without anything going red.
    """
    names = {p.relative_to(REPO).as_posix() for p in text_files()}
    assert len(names) > 50, f"the sweep found only {len(names)} text files; it used to find 105"
    for expected in ("README.md", "pyproject.toml", "royalegym/action.py"):
        assert expected in names, f"the sweep does not reach {expected}"
    suffixes = {Path(n).suffix for n in names}
    assert {".py", ".md", ".toml"} <= suffixes, (
        f"the sweep only reaches {sorted(suffixes)}; pages are where heredocs write, so "
        "dropping .md would remove the reason this was widened"
    )
