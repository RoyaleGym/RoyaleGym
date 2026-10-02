"""MkDocs hooks for the site (listed under ``hooks:`` in mkdocs.yml).

The pages under ``repos/`` are copied in from each repo's ``docs/`` by collect.py, so the
site's own "edit this page" link (RoyaleGym/edit/main/docs/site/pages/...) would point at a
generated file that is in no repository. This sends it to the file in the repo it came from.
"""

from __future__ import annotations

from typing import Any

ORG = "https://github.com/RoyaleGym"
#: The folder under pages/repos/ -> the repo it was copied from.
REPOS = {
    "royalesim": "RoyaleSim",
    "royalegym": "RoyaleGym",
    "royalelearn": "RoyaleLearn",
    "royaleviser": "RoyaleViser",
    "royaleimitate": "RoyaleImitate",
}
#: Files collect.py renames on the way in, back to their names in the repo.
ROOT_FILES = {"changelog.md": "CHANGELOG.md", "contributing.md": "CONTRIBUTING.md"}


def source_of(src_uri: str) -> str | None:
    """The edit URL of a copied page, or None for a page written for the site."""
    parts = src_uri.split("/")
    if len(parts) < 3 or parts[0] != "repos" or parts[1] not in REPOS:
        return None
    repo, rest = REPOS[parts[1]], "/".join(parts[2:])
    path = ROOT_FILES.get(rest, f"docs/{rest}")
    return f"{ORG}/{repo}/edit/main/{path}"


def on_page_context(context: dict[str, Any], page: Any, config: Any, nav: Any) -> dict[str, Any]:
    url = source_of(page.file.src_uri)
    if url is not None:
        page.edit_url = url
    return context
