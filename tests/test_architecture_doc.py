"""``docs/architecture.md`` describes this package, so hold it to the package.

WHY THIS EXISTS
    The module map went stale without anything saying so. ``evaluate.py`` and
    ``opponents.py`` shipped, and the map still listed the modules that were there
    before them, so a reader following the map would conclude the package has no way
    to compare two bots and no scripted opponents to compare them against. Nothing was
    wrong with the code and nothing failed.

    A document that lists what the code contains is a claim about the code, and a
    claim nobody checks drifts in one direction: the code grows, the list does not.
    So this compares the two and fails when they part, which is the friction that
    keeps the map true.

WHAT IT CHECKS
    Every ``royalegym/*.py`` module is named in the map, and every module the map
    names exists. Both directions matter: the first catches a module that was added
    and never written up, the second catches one that was renamed or removed while
    its row stayed.

WHAT IT DOES NOT CHECK
    Whether each row DESCRIBES its module correctly. That is a judgement, not a
    comparison, and a test that pretended to make it would be the kind of check that
    passes while saying nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARCHITECTURE = REPO / "docs" / "architecture.md"
PACKAGE = REPO / "royalegym"

#: Modules deliberately left out of the map, with the reason. ``__init__.py`` is the
#: package's front door rather than one of the pieces the map is about.
NOT_IN_THE_MAP = {"__init__.py"}


def mapped_modules() -> set[str]:
    """The ``name.py`` in the first column of every module-map row."""
    text = ARCHITECTURE.read_text(encoding="utf-8")
    heading = "## Module map"
    assert heading in text, (
        f"docs/architecture.md has no {heading!r} section. If it was renamed, rename it "
        "here too: this test cannot check a table it cannot find, and a check that "
        "cannot find its subject has to say so rather than pass."
    )
    start = text.index(heading)
    end = text.find("\n## ", start + 1)
    table = text[start : end if end != -1 else len(text)]
    return set(re.findall(r"^\|\s*`([a-z_]+\.py)`\s*\|", table, re.MULTILINE))


def package_modules() -> set[str]:
    return {p.name for p in PACKAGE.glob("*.py")} - NOT_IN_THE_MAP


def test_the_module_map_names_every_module_and_only_real_ones() -> None:
    mapped, real = mapped_modules(), package_modules()
    assert mapped, "no module-map rows were found; did the table's shape change?"
    missing = sorted(real - mapped)
    phantom = sorted(mapped - real)
    assert not missing, (
        f"docs/architecture.md's module map does not mention {missing}. A reader "
        "following it would not know these exist."
    )
    assert not phantom, (
        f"docs/architecture.md's module map names {phantom}, which are not in "
        "royalegym/. A row outlived its module."
    )


def ruff_scope(text: str, where: str) -> set[str]:
    found = re.search(r"ruff check ([a-z ]+?)\s*(?:#|\n)", text)
    assert found, f"{where} no longer shows a ruff command"
    return set(found.group(1).split())


def test_the_documented_gate_commands_name_the_scopes_that_are_gated() -> None:
    """The Tests section prints the commands a contributor runs. They are the gate, so
    a scope that drifts here is a scope nobody runs.

    Compared against the README's command rather than a list written here. A literal
    would be a third copy of the same fact, and the next person to widen the gate would
    update the two they were looking at: that is how this page came to say royalegym and
    tests while the README already gated examples/ as well. The README is the page a
    contributor actually follows, so it is the one to agree with.
    """
    doc = ruff_scope(ARCHITECTURE.read_text(encoding="utf-8"), "docs/architecture.md")
    readme = ruff_scope((REPO / "README.md").read_text(encoding="utf-8"), "README.md")
    assert doc == readme, (
        f"docs/architecture.md gates ruff on {sorted(doc)} and README.md on "
        f"{sorted(readme)}. Whichever is narrower names code that nobody lints."
    )
