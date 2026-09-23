"""Would a reader's shell actually run the commands in these pages?

On 2026-09-22 the published install block ran in NO shell a newcomer would use, and it had been
that way since the repos were split. Four independent readers hit it from four entry points. The
four shapes that did it, each confirmed on the machine rather than argued:

1. ``&&`` in a block a Windows reader will paste. Windows PowerShell 5.1 is the default shell on
   Windows 10 and 11 and has no ``&&``: it is a PARSE ERROR before anything executes.
2. A trailing ``#`` comment on a shell line. ``cmd`` has no ``#`` comment, so
   ``python -m venv .venv  # Python 3.12`` passes four positional arguments to venv, which takes a
   list of them, and the reader silently gets four directories named ``.venv``, ``#``, ``Python``
   and ``3.12`` and no working environment. This is the dangerous one, because it does not fail.
3. A Windows backslash path in a bash-tagged fence. Bash eats the backslashes, so
   ``.venv\\Scripts\\python`` becomes ``.venvScriptspython: command not found``.
4. A shell block under no named shell at all, so the reader cannot tell which of the three it is
   for, and two of the three will not run it.

This refuses those four shapes. It does not understand the commands and does not try to: it is a
guard on the shapes, not a test of the instructions. The real check is a human running the recipe
verbatim in PowerShell from an empty folder.

A fence is treated as POSIX-only, and then ``&&`` and forward slashes are fine, when the nearest
heading or tab above it names macOS, Linux or bash, or when it is tagged ```bash. Everything else
is assumed to be something a Windows reader will paste, because that is the default on the
platform most hobbyists have.

Usage:  python shell_fence_check.py <file.md> [more files...]
        python shell_fence_check.py --selftest
Exit 0 if every file passes, 1 otherwise.
"""

from __future__ import annotations

import re
import sys

FENCE = re.compile(r"^( {0,3})(`{3,}|~{3,})([^\n]*)\n(.*?)^\1?\2`*[ \t]*$", re.M | re.S)
POSIX_HINT = re.compile(r"macos|linux|bash|posix|wsl", re.I)
# A line that looks like a shell command rather than prose or output.
SHELLISH = re.compile(r"^\s*(?:[A-Za-z]:)?[.\w/\\-]*(?:python|pip|git|mkdir|cd|cargo|maturin|ruff|pytest|source|export|set)\b")
TRAILING_COMMENT = re.compile(r"\S\s+#\s")
BACKSLASH_PATH = re.compile(r"[\w.]+\\[\w.]+\\")
PROSE_TAGS = {"python", "text", "json", "toml", "yaml", "rust", "console", "output", "diff"}


def fences(text: str):
    for m in FENCE.finditer(text):
        yield m.start(), (m.group(3) or "").strip().lower(), m.group(4)


def context_is_posix(text: str, upto: int) -> bool:
    """Does the nearest heading or tab line above this fence name a POSIX shell?"""
    head = text[:upto].rsplit("\n\n", 1)[0]
    for line in reversed(head.split("\n")[-12:]):
        if line.strip().startswith(("#", "===", '=== "', "**")) or "tab" in line.lower():
            if POSIX_HINT.search(line):
                return True
            if line.strip().startswith(("#", "===")):
                return False
    return bool(POSIX_HINT.search(text[max(0, upto - 400):upto]))


def check_text(text: str, name: str = "<text>") -> list[str]:
    problems = []
    for pos, tag, body in fences(text):
        if tag in PROSE_TAGS:
            continue
        line_no = text[:pos].count("\n") + 1
        posix = posix_tag = tag in {"bash", "sh", "shell", "zsh"}
        if not posix:
            posix = context_is_posix(text, pos)
        shell_lines = [ln for ln in body.split("\n") if SHELLISH.match(ln)]
        if not shell_lines:
            continue
        for ln in shell_lines:
            if "&&" in ln and not posix:
                problems.append(
                    f"{name}:{line_no}: `&&` in a block a Windows reader will paste; "
                    f"PowerShell 5.1 cannot parse it: {ln.strip()[:70]}"
                )
            if TRAILING_COMMENT.search(ln):
                problems.append(
                    f"{name}:{line_no}: trailing `#` comment on a shell line; cmd passes it as "
                    f"arguments and fails SILENTLY: {ln.strip()[:70]}"
                )
            if posix_tag and BACKSLASH_PATH.search(ln):
                problems.append(
                    f"{name}:{line_no}: Windows backslash path in a bash-tagged fence; bash eats "
                    f"the backslashes: {ln.strip()[:70]}"
                )
        if not posix and not posix_tag and BACKSLASH_PATH.search(body) and not context_names_shell(text, pos):
            problems.append(
                f"{name}:{line_no}: shell block under no named shell; say which one it is for"
            )
    return problems


def context_names_shell(text: str, upto: int) -> bool:
    """Has the page told the reader which shell these commands are for, anywhere above here?

    Deliberately the WHOLE document above the fence rather than a window. A page that says
    "the commands below are for Windows" once, at the top of its install section, has told the
    reader; demanding it again above every block would fire on pages that are already correct,
    and a check that cries wolf is one somebody switches off. The defect this rule is for is a
    page that never says at all.
    """
    return any(w in text[:upto].lower() for w in ("windows", "powershell", "cmd.exe", "macos", "linux", "bash"))


SELFTEST = [
    ("chained with &&", "## Install\n\n```\nmkdir Royale && cd Royale\n```\n", True),
    ("trailing # comment", "## Install on Windows\n\n```\npython -m venv .venv  # Python 3.12\n```\n", True),
    ("backslashes in a bash fence", "## Install\n\n```bash\n.venv\\Scripts\\python -m pip install x\n```\n", True),
    ("no named shell", "## Install\n\n```\n.venv\\Scripts\\python -m pip install x\n```\n", True),
    ("posix tab may chain", '=== "macOS and Linux"\n\n```\nmkdir Royale && cd Royale\n```\n', False),
    ("windows block, one per line", "## Install on Windows\n\n```\nmkdir Royale\ncd Royale\n```\n", False),
    ("python block untouched", "## Example\n\n```python\nx = 1  # a real comment\n```\n", False),
]


def selftest() -> int:
    bad = 0
    for name, text, must_fail in SELFTEST:
        got = bool(check_text(text, name))
        ok = got == must_fail
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {'refused' if got else 'passed'} "
              f"({'should refuse' if must_fail else 'should pass'})")
    print(f"\n{len(SELFTEST) - bad} of {len(SELFTEST)} self-tests behaved")
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--selftest":
        return selftest()
    if len(argv) < 2:
        raise SystemExit("usage: shell_fence_check.py <file.md> [more files...] | --selftest")
    bad = 0
    for path in argv[1:]:
        with open(path, encoding="utf-8") as fh:
            problems = check_text(fh.read(), path)
        for p in problems:
            print(f"REFUSE {p}")
        print(f"{'FAIL' if problems else 'ok  '} {path}: {len(problems)} problem(s)")
        bad += bool(problems)
    print(f"\n{len(argv) - 1 - bad} of {len(argv) - 1} files passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
