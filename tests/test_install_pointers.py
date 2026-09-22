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

    The scan does not look for the phrasings this repo happens to use. It looks for a
    capitalised word sitting in POINTER POSITION -- straight after ``README``, with at
    most a comma, a colon, an ``'s`` or a quote in between. Any way of naming a section
    lands there, because that is what naming a section looks like in English, so a
    pointer written in a form nobody anticipated is caught by position rather than by
    wording.

    That is the second design. The first matched three spellings and reported clean on
    a tree that had a fourth: ``protocol.py`` told a reader to run "its README's Setup
    block", one function below a pointer the same file had just had corrected. Two
    other people's sweeps missed it too. An instrument whose reach is narrower than the
    question it appears to answer will report the difference as good news.

WHAT IT CANNOT CATCH
    A section named somewhere other than next to the word README -- "see the install
    section" with the file named two sentences earlier reads as prose here.

    A multi-word section name written without quotes: only the first word is captured,
    so ``README.md What you get`` is checked as "What". It fails loudly rather than
    passing, which is the right way round, but the message will be confusing.

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

# A section name in POINTER POSITION: straight after README, past at most a comma, a
# colon, an "'s" and a quote. Position rather than phrasing, so a pointer written in a
# form nobody anticipated is still caught. A capital is required because the word after
# README is otherwise ordinary prose -- "no README here ever had that heading" must not
# read as a pointer to a section called "here".
POINTER_FORMS = (
    re.compile(r"README(?:\.md)?(?:'s)?[,:]?\s+\"([A-Z][A-Za-z ]{1,30})\""),
    re.compile(r"README(?:\.md)?(?:'s)?[,:]?\s+([A-Z][A-Za-z]{1,20})\b"),
    re.compile(r'the "([^"]{2,40})" section of the \w+ README\.md'),
)

# This file is left out of its own scan. It quotes the defect it exists for, in its
# docstring and in the plant below, and a scan that reads its own explanation of a
# wrong pointer as a wrong pointer is no use to anyone.
SOURCES = [
    p
    for p in sorted((REPO / "royalegym").glob("*.py")) + sorted((REPO / "tests").glob("*.py"))
    if p.name != Path(__file__).name
]


def readme_headings(repo: str = "RoyaleGym") -> set[str] | None:
    """Every heading in a repo's README, at any level, or None if it is not checked out.

    At ANY level on purpose: a scan that reads ``##`` only cannot see a ``###`` section
    and will report it missing. Three people made that mistake on one evening.

    None rather than an empty set for a missing sibling, so "no headings" and "no
    repository" cannot be confused -- an empty set would fail every pointer into it and
    look like a swarm of findings.
    """
    path = README if repo == "RoyaleGym" else REPO.parent / repo / "README.md"
    if not path.exists():
        return None
    return {
        m.group(1).strip()
        for m in re.finditer(r"^#{1,6} +(.+?)\s*$", path.read_text(encoding="utf-8"), re.M)
    }


def pointers() -> list[tuple[Path, str, str]]:
    """(file, target repo, section name) for every literal section pointer.

    The repo is whichever ``Royale*`` is named in the same SENTENCE as the pointer,
    defaulting to this one. protocol.py's data-missing error sends a reader to
    RoyaleSim's README, not to this one, and checking it here passed only because both
    repos happen to call the section Install. A pointer checked against the wrong
    document is a pointer that is not checked.

    Sentence-scoped rather than "the nearest name within N characters", which was the
    first attempt and read the workspace-layout comment -- a list of all five repos
    ending in RoyaleLearn, then a semicolon, then a pointer at THIS repo's README -- as
    a pointer into RoyaleLearn. Proximity is not reference.

    A capture holding a ``{`` is built at runtime from a constant, not a heading name;
    it is skipped here and checked resolved by the first test below.
    """
    found = []
    for path in SOURCES:
        text = path.read_text(encoding="utf-8")
        for form in POINTER_FORMS:
            for m in form.finditer(text):
                if "{" in m.group(1):
                    continue
                sentence = re.split(r"[.;]", text[max(0, m.start() - 300) : m.start()])[-1]
                named = re.findall(r"\bRoyale[A-Z]\w+", sentence)
                found.append((path, named[-1] if named else "RoyaleGym", m.group(1)))
    return found


def test_the_install_section_named_by_the_import_error_exists() -> None:
    """The heading the not-built message sends a reader to is in the README.

    This is the assertion that would have been red for the whole life of the defect.
    """
    headings = readme_headings()
    assert headings is not None
    assert INSTALL_SECTION in headings, (
        f"rust_engine.INSTALL_SECTION is {INSTALL_SECTION!r} and README.md has no such "
        f"heading. It has: {sorted(headings)}"
    )
    assert INSTALL_SECTION in INSTALL_POINTER
    assert "README.md" in INSTALL_POINTER


@pytest.mark.parametrize(
    ("path", "repo", "section"), pointers(), ids=lambda v: getattr(v, "name", v)
)
def test_every_section_pointer_in_the_source_names_a_real_heading(
    path: Path, repo: str, section: str
) -> None:
    headings = readme_headings(repo)
    if headings is None:
        # Visible, with its reason. A silent skip here would read as a pass, which is
        # the failure this whole file is about.
        pytest.skip(f"{repo} is not checked out beside this repo, so its README cannot be read")
    assert section in headings, (
        f"{path.relative_to(REPO)} points at {repo}'s README section {section!r}, which "
        f"does not exist. That README has: {sorted(headings)}"
    )


def test_the_scan_finds_the_references_it_is_guarding() -> None:
    """The vacuity guard.

    Find zero pointers and every parametrised case above vanishes, leaving a file that
    passes by testing nothing. Three literal ones are there today: protocol.py's
    workspace-layout comment, protocol.py's data-missing error -- which points into
    RoyaleSim -- and test_parity_hardening.py's module docstring. The import error's own
    is built from a constant, so it carries a placeholder here and is checked resolved,
    one test up.

    It also pins that a pointer into ANOTHER repo is still seen. The scan was scoped to
    this repo's README once, and the cross-repo pointer passed under it by coincidence.
    """
    found = pointers()
    assert len(found) >= 3, f"expected the known literal pointers, found {found}"
    assert {p.name for p, _, _ in found} >= {"protocol.py", "test_parity_hardening.py"}
    assert {repo for _, repo, _ in found} >= {"RoyaleGym", "RoyaleSim"}


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
