"""Bring every repo's docs onto the site, so nothing past a README lives anywhere else.

    python collect.py --root <folder with the five repos side by side> [--ref origin/main]

Run it from docs/site before ``mkdocs build`` or ``mkdocs gh-deploy``. It writes
``pages/repos/<package>/``, which is generated and ignored by git: the files stay in their own
repos, and this copies them at build time.

What it copies, for each repo: everything under ``docs/`` (but RoyaleGym's ``docs/site/``,
which is this site), plus ``CHANGELOG.md`` and ``CONTRIBUTING.md`` from the repo root.
RoyaleGym's files are read from this checkout; the other four are read from ``--ref`` in their
git repos, so the site shows what is published, not whatever a local checkout holds.

Links are rewritten so the site builds with ``--strict``:

* a link to another copied file becomes a relative link to its page on the site;
* a link to anything else in a repo (its README, source, tests) becomes a GitHub URL;
* a link that climbs into a sibling repo is treated the same way, in that repo.

Links inside fenced code blocks are left alone.
"""

from __future__ import annotations

import argparse
import posixpath
import re
import shutil
import subprocess
import sys
from pathlib import Path

ORG = "https://github.com/RoyaleGym"
#: Repo -> the folder its docs land in, under pages/repos/.
REPOS = {
    "RoyaleSim": "royalesim",
    "RoyaleGym": "royalegym",
    "RoyaleLearn": "royalelearn",
    "RoyaleViser": "royaleviser",
    "RoyaleImitate": "royaleimitate",
}
ROOT_FILES = {"CHANGELOG.md": "changelog.md", "CONTRIBUTING.md": "contributing.md"}
SITE = Path(__file__).resolve().parent
OUT = SITE / "pages" / "repos"
THIS_REPO = SITE.parents[1]

FENCE = re.compile(r"^\s*(```|~~~)")
MD_LINK = re.compile(r"(!?\[[^\]]*\]\()([^)\s]+)((?:\s+\"[^\"]*\")?\))")
HTML_ATTR = re.compile(r"""((?:src|href)=")([^"]+)(")""")
REF_DEF = re.compile(r"^(\s*\[[^\]]+\]:\s*)(\S+)(.*)$")


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True
    ).stdout


def files_of(repo_name: str, repo: Path, ref: str) -> dict[str, bytes]:
    """Repo-relative path -> content, for every file this site takes from the repo."""
    wanted: dict[str, bytes] = {}
    if repo_name == "RoyaleGym":
        for path in sorted((repo / "docs").rglob("*")):
            rel = path.relative_to(repo).as_posix()
            if path.is_file() and not rel.startswith("docs/site/"):
                wanted[rel] = path.read_bytes()
        for name in ROOT_FILES:
            if (repo / name).is_file():
                wanted[name] = (repo / name).read_bytes()
        return wanted
    listing = git(repo, "ls-tree", "-r", "--name-only", ref, "--", "docs", *ROOT_FILES)
    for rel in listing.decode().splitlines():
        if rel.startswith("docs/") or rel in ROOT_FILES:
            wanted[rel] = git(repo, "show", f"{ref}:{rel}")
    return wanted


def site_path(repo_name: str, rel: str) -> str | None:
    """Where a repo file lands on the site (relative to pages/), or None if it does not."""
    folder = f"repos/{REPOS[repo_name]}"
    if rel.startswith("docs/") and not rel.startswith("docs/site/"):
        return f"{folder}/{rel[len('docs/'):]}"
    if rel in ROOT_FILES:
        return f"{folder}/{ROOT_FILES[rel]}"
    return None


def page_url(site_rel: str) -> str:
    """The URL path a page or file is served at, relative to the site root. MkDocs serves
    ``x.md`` at ``x/`` and a folder's ``README.md`` or ``index.md`` at the folder itself."""
    if not site_rel.endswith(".md"):
        return site_rel
    head, name = posixpath.split(site_rel)
    if name in ("README.md", "index.md"):
        return f"{head}/" if head else ""
    return site_rel[: -len(".md")] + "/"


def rewrite(
    target: str, repo_name: str, source_rel: str, copied: dict[str, set[str]], html: bool = False
) -> str:
    """One link's new target. ``copied`` is repo -> the repo paths that are on the site.

    MkDocs rewrites relative links in Markdown itself, but not inside raw HTML (``<img src>``,
    ``<a href>``), and it serves ``x.md`` at ``x/``. So an HTML link is made relative to the URL
    the page is served at, and points at the URL of the page it links to."""
    if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith(("#", "//")):
        return target
    path, _, anchor = target.partition("#")
    if not path:
        return target
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(source_rel), path))
    repo = repo_name
    parts = resolved.split("/")
    # A link that climbs out of its repo into a sibling: ../RoyaleLearn/docs/x.md.
    while parts and parts[0] == "..":
        parts = parts[1:]
        if parts and parts[0] in REPOS:
            repo, parts = parts[0], parts[1:]
            break
    rel = "/".join(parts)
    suffix = f"#{anchor}" if anchor else ""
    landed = site_path(repo, rel) if rel in copied.get(repo, set()) else None
    if landed is not None:
        source_site = site_path(repo_name, source_rel) or ""
        if html:
            here = page_url(source_site).rstrip("/") if source_site.endswith(".md") else ""
            if posixpath.basename(source_site) in ("README.md", "index.md"):
                here = posixpath.dirname(source_site)
            goal = page_url(landed)
            out = posixpath.relpath(goal.rstrip("/") or ".", here or ".")
            return out + ("/" if goal.endswith("/") else "") + suffix
        here = posixpath.dirname(source_site)
        return posixpath.relpath(landed, here or ".") + suffix
    kind = "tree" if path.endswith("/") or not posixpath.splitext(rel)[1] else "blob"
    if rel in ("", "."):
        return f"{ORG}/{repo}" + suffix
    return f"{ORG}/{repo}/{kind}/main/{rel}" + suffix


def rewrite_markdown(
    text: str, repo_name: str, source_rel: str, copied: dict[str, set[str]]
) -> str:
    def md(m: re.Match[str]) -> str:
        return m.group(1) + rewrite(m.group(2), repo_name, source_rel, copied) + m.group(3)

    def raw(m: re.Match[str]) -> str:
        new = rewrite(m.group(2), repo_name, source_rel, copied, html=True)
        return m.group(1) + new + m.group(3)

    out, fenced = [], False
    for line in text.splitlines(keepends=True):
        if FENCE.match(line):
            fenced = not fenced
            out.append(line)
            continue
        if not fenced:
            line = MD_LINK.sub(md, line)
            line = HTML_ATTR.sub(raw, line)
            line = REF_DEF.sub(md, line)
        out.append(line)
    return "".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--root", type=Path, required=True,
        help="the folder that holds RoyaleSim, RoyaleLearn, RoyaleViser and RoyaleImitate",
    )
    parser.add_argument(
        "--ref", default="origin/main", help="the git ref to read the other repos at"
    )
    args = parser.parse_args()

    found: dict[str, dict[str, bytes]] = {}
    for name in REPOS:
        repo = THIS_REPO if name == "RoyaleGym" else args.root / name
        if not (repo / ".git").exists():
            print(f"{name}: not found at {repo}; the site needs all five repos", file=sys.stderr)
            return 1
        found[name] = files_of(name, repo, args.ref)
    copied = {name: set(files) for name, files in found.items()}

    if OUT.exists():
        shutil.rmtree(OUT)
    for name, files in found.items():
        for rel, data in files.items():
            dest = OUT.parent / site_path(name, rel)  # type: ignore[operator]
            dest.parent.mkdir(parents=True, exist_ok=True)
            if rel.endswith(".md"):
                text = rewrite_markdown(data.decode("utf-8"), name, rel, copied)
                dest.write_text(text, encoding="utf-8", newline="\n")
            else:
                dest.write_bytes(data)
        pages = sum(1 for rel in files if rel.endswith(".md"))
        others = len(files) - pages
        print(f"{name}: {pages} pages and {others} other files -> pages/repos/{REPOS[name]}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
