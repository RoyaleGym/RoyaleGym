"""Every pointer this package gives a reader has to land on something real.

WHAT IT CATCHES
    A section name quoted in the code that the README does not have. That is not
    hypothetical: ``rust_engine.CORE_IMPORT_ERROR`` ended with ``(README.md, Setup)``
    from the day it was written until 2026-09-22, and no README in this repo has ever
    had a "Setup" heading. It was the one pointer a reader gets at the moment the
    engine has not built -- the moment they most need it to land somewhere -- and
    nothing read that string, so nothing noticed.

    It also catches the same rot in the install instructions themselves: the README's
    build line and the error message's build line have to be the same command, and the
    data step has to come before it, which is what the message now promises.

WHAT IT CANNOT CATCH
    A pointer written in a form the scan below does not recognise. It knows the shapes
    this repo uses -- ``README.md, "X"``, the same without the comma, and ``the "X"
    section of the <Repo> README.md`` -- and a fourth would slip past it silently.
    That is why ``test_the_scan_finds_the_references_it_is_guarding`` pins the count: a
    refactor that hides them all turns this file red instead of green-and-vacuous.
    Writing the scan tripped on both of the holes it is guarding, which is the best
    argument for the guard: it missed test_parity_hardening.py, whose pointer has no
    comma, and it read the f-string placeholder in rust_engine.py as a heading name.

    A pointer into another repo's README. RoyaleViser's does have a "Setup", at ``###``
    -- which is how the original report of this defect overstated itself, from a sweep
    that listed ``##`` headings only. Only this repo's README is read here.

    Whether the section, once found, says the right thing. A heading that exists and is
    wrong reads as fine here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from royalegym.rust_engine import INSTALL_POINTER, INSTALL_SECTION

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"

# The shapes this repo writes a section pointer in. Deliberately strict: a loose
# pattern would match prose about READMEs and fail on a sentence, which is worse than
# missing a form, because a guard that cries wolf gets switched off.
POINTER_FORMS = (
    re.compile(r'README\.md,?\s*"([^"]{2,40})"'),
    re.compile(r'the "([^"]{2,40})" section of the \w+ README\.md'),
)

SOURCES = sorted((REPO / "royalegym").glob("*.py")) + sorted((REPO / "tests").glob("*.py"))


def readme_headings() -> set[str]:
    """Every heading in README.md, at any level.

    At ANY level on purpose: a scan that reads ``##`` only cannot see a ``###`` section
    and will report it missing.
    """
    return {
        m.group(1).strip()
        for m in re.finditer(r"^#{1,6} +(.+?)\s*$", README.read_text(encoding="utf-8"), re.M)
    }


def pointers() -> list[tuple[Path, str]]:
    """(file, section name) for every literal section pointer in the package.

    A capture holding a ``{`` is a pointer built at runtime from a constant, not a
    heading name; it is skipped here and checked resolved by the first test below.
    """
    found = []
    for path in SOURCES:
        text = path.read_text(encoding="utf-8")
        for form in POINTER_FORMS:
            found.extend(
                (path, m.group(1)) for m in form.finditer(text) if "{" not in m.group(1)
            )
    return found


def test_the_install_section_named_by_the_import_error_exists() -> None:
    """The heading the not-built message sends a reader to is in the README.

    This is the assertion that would have been red for the whole life of the defect.
    """
    headings = readme_headings()
    assert INSTALL_SECTION in headings, (
        f"rust_engine.INSTALL_SECTION is {INSTALL_SECTION!r} and README.md has no such "
        f"heading. It has: {sorted(headings)}"
    )
    assert INSTALL_SECTION in INSTALL_POINTER
    assert "README.md" in INSTALL_POINTER


@pytest.mark.parametrize(("path", "section"), pointers(), ids=lambda v: getattr(v, "name", v))
def test_every_section_pointer_in_the_source_names_a_real_heading(path: Path, section: str) -> None:
    headings = readme_headings()
    assert section in headings, (
        f"{path.relative_to(REPO)} points at README.md section {section!r}, which does "
        f"not exist. README.md has: {sorted(headings)}"
    )


def test_the_scan_finds_the_references_it_is_guarding() -> None:
    """The vacuity guard.

    Find zero pointers and every parametrised case above vanishes, leaving a file that
    passes by testing nothing. Two literal ones are there today: protocol.py's
    workspace-layout comment and test_parity_hardening.py's module docstring. The
    import error's own is built from a constant, so it carries a placeholder here and
    is checked resolved, one test up.
    """
    found = pointers()
    assert len(found) >= 2, f"expected the known literal pointers, found {found}"
    assert {p.name for p, _ in found} >= {"protocol.py", "test_parity_hardening.py"}


def test_the_build_command_in_the_error_matches_the_one_in_the_readme() -> None:
    """The message tells a reader to run what the README tells them to run.

    Not the whole line -- the README's is a full Windows-venv invocation and the
    message is the short form -- but the command and its release flag have to agree,
    because a reader who runs a debug build gets an engine that works and is slow and
    nothing tells them which one they have.
    """
    readme = README.read_text(encoding="utf-8")
    install = readme.split(f"## {INSTALL_SECTION}", 1)[1]
    assert "maturin develop --release" in install
    # And the data step really does come before the build in that section, which is
    # what the error message now promises is written down there.
    assert install.index("extract_cards.py") < install.index("maturin develop --release")
