r"""Stage the wheels for a RoyaleGym release page: everything `pip install "royalegym[all]"` needs.

    python tools/stage_release.py OUT_DIR [--sim-tag v0.1.0]

- royalesim: the wheels attached to a RoyaleSim GitHub release (its newest v* tag, or --sim-tag),
  downloaded with `gh`, unchanged.
- royalegym, royalelearn, royaleviser, royaleimitate: each built from its repo's NEWEST v* tag,
  never from main, so a wheel's version always names the code inside it. The repos are found next
  to this checkout. A repo with no v* tag, or a tag that is not "v" + the version in that tagged
  tree, is refused.

It refuses a page whose royalegym [all] minimums are not exactly the versions it stages (so
`pip install --upgrade "royalegym[all]"` moves every package), then prints one row per package
(name, version, tag, commit) for the release notes. Then run
tools/fresh_user_test.py --by-name --wheels OUT_DIR before publishing anything.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path

ORG = "RoyaleGym"
#: The pure-Python packages, by repo, built from their newest v* tag.
BUILT = ("RoyaleGym", "RoyaleLearn", "RoyaleViser", "RoyaleImitate")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def newest_tag(repo: Path) -> str:
    """The newest v* tag, by version order. Raises if there is none."""
    tags = git(repo, "tag", "-l", "v*", "--sort=-v:refname").splitlines()
    if not tags:
        raise SystemExit(f"{repo.name} has no v* tag: tag the released commit first")
    return tags[0]


def version_at(repo: Path, tag: str) -> str:
    """[project] version in pyproject.toml as committed at ``tag``."""
    text = git(repo, "show", f"{tag}:pyproject.toml")
    return str(tomllib.loads(text)["project"]["version"])


def check_tag(repo: Path, tag: str) -> str:
    """The version, if ``tag`` is "v" + the version at that tag. Raises otherwise."""
    version = version_at(repo, tag)
    if tag != f"v{version}":
        raise SystemExit(
            f"{repo.name}: tag {tag} holds version {version}. A wheel built from it would call "
            "itself one version and hold another; fix the tag or the version first."
        )
    return version


def floor_problems(all_reqs: list[str], staged: dict[str, str]) -> list[str]:
    """What is wrong with royalegym's [all] minimums for the versions this page stages.

    Each must be exactly the staged version. One below it lets `pip install --upgrade
    "royalegym[all]"` keep the older package (pip upgrades a dependency only when a requirement
    forces it), and one above it cannot be installed from this page."""
    floors = {}
    for req in all_reqs:
        name = req.split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip()
        floors[name] = req.split(">=", 1)[1].strip() if ">=" in req else None
    problems = []
    for name, version in staged.items():
        if name not in floors:
            problems.append(f"[all] does not name {name}, which this page stages at {version}")
        elif floors[name] is None:
            problems.append(f"[all] names {name} with no minimum; this page stages {version}")
        elif floors[name] != version:
            problems.append(
                f"[all] asks for {name}>={floors[name]} and this page stages {version}: set the "
                f"minimum to {version} in royalegym's pyproject and tag again"
            )
    return problems


def build(repo: Path, tag: str, out: Path) -> None:
    """A wheel from a clean export of ``tag``: nothing from the working tree, no build folder
    left in the checkout."""
    archive = subprocess.run(
        ["git", "-C", str(repo), "archive", tag], check=True, capture_output=True
    ).stdout
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(tmp, filter="data")
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "-q", "-w", str(out), tmp],
            check=True,
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("out", help="folder to put the wheels in (created; must be empty)")
    ap.add_argument(
        "--sim-tag", help="the RoyaleSim release to take royalesim from (default: newest)"
    )
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"{out} is not empty: stage each release into a fresh folder")
    here = Path(__file__).resolve().parents[1]
    rows = []

    sim = here.parent / "RoyaleSim"
    git(sim, "fetch", "-q", "--tags", "origin")
    sim_tag = args.sim_tag or newest_tag(sim)
    subprocess.run(
        [
            "gh",
            "release",
            "download",
            sim_tag,
            "-R",
            f"{ORG}/RoyaleSim",
            "--pattern",
            "*.whl",
            "--dir",
            str(out),
        ],
        check=True,
    )
    rows.append(
        (
            "royalesim",
            sim_tag.removeprefix("v"),
            sim_tag,
            git(sim, "rev-parse", "--short", f"{sim_tag}^{{commit}}"),
        )
    )

    for name in BUILT:
        repo = here if name == "RoyaleGym" else here.parent / name
        git(repo, "fetch", "-q", "--tags", "origin")
        tag = newest_tag(repo)
        version = check_tag(repo, tag)
        build(repo, tag, out)
        rows.append(
            (name.lower(), version, tag, git(repo, "rev-parse", "--short", f"{tag}^{{commit}}"))
        )

    gym_tag = next(row[2] for row in rows if row[0] == "royalegym")
    all_reqs = tomllib.loads(git(here, "show", f"{gym_tag}:pyproject.toml"))["project"][
        "optional-dependencies"
    ]["all"]
    staged = {row[0]: row[1] for row in rows if row[0] != "royalegym"}
    problems = floor_problems(all_reqs, staged)
    if problems:
        raise SystemExit(
            "REFUSED, the page would not upgrade in one line:\n  " + "\n  ".join(problems)
        )

    print("| package | version | tag | commit |")
    print("|---|---|---|---|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print(f"\n{len(list(out.glob('*.whl')))} wheels in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
