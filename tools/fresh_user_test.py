r"""The fresh-user test: what someone new gets from one install line and one file.

In a NEW temporary folder with a NEW venv, no repo checkout on the path, no ROYALE* variable
and a temporary HOME, it:
  P1  creates the venv (the interpreter you pass, else this one's base)
  P2  installs `royalegym[all]` from THIS checkout (--gym PATH, copied by pip, not editable),
      with royalesim and any sibling not yet on PyPI from WHEELS (--wheels DIR, --find-links)
  P3  proves isolation: every royale* package imports from the new venv's site-packages,
      nothing from the checkout
  P4  copies examples/quickstart.py the way a user gets it (not imported), sets its size to a
      test's (SMALL, each literal exactly once; timestep_limit = --steps; a literal that is not
      a FAIL, not a long run) and runs it under a time limit; it must exit 0
  P5  records one battle with the INSTALLED package's public API and saves a trace
  P6  opens that trace in the viewer headless (SDL_VIDEODRIVER=dummy, --seconds, --shot)
  P7  runs the README's "Try it" as pasted, then the royaleviser command it gives, headless
Each phase reports PASS, FAIL or BLOCKED. BLOCKED means a prerequisite is missing (no prebuilt
royalesim wheel, no quickstart.py, no [all] extra): it is NOT a pass. It also lists HOLES:
things a newcomer would hit that are not failures of a phase (a missing console script, a
missing __version__). Exit 0 only when every phase passed.

  python tools/fresh_user_test.py --gym . --wheels DIR [--steps 2000] [--timeout 900]
      [--keep] [--report out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ORG = "RoyaleGym"
#: The quickstart's size literals, set to a test's (tests/test_quickstart.py has the same, plus
#: a forced CPU and no viewer). Here device stays "auto" and viser stays on, as a user runs it:
#: CI has no GPU, so this is also the check that the CPU fallback works.
SMALL = {
    "n_envs=32,": "n_envs=4,",
    "steps_per_update=16_384,": "steps_per_update=512,",
    "ppo_batch_size=16_384,": "ppo_batch_size=512,",
    "ppo_minibatch_size=2_048,": "ppo_minibatch_size=256,",
    "checkpoint_every=200_000,": "checkpoint_every=512,",
}
PACKAGES = ["royalesim", "royalegym", "royalelearn", "royaleviser", "royaleimitate"]
REPO = {
    "royalesim": "RoyaleSim",
    "royalegym": "RoyaleGym",
    "royalelearn": "RoyaleLearn",
    "royaleviser": "RoyaleViser",
    "royaleimitate": "RoyaleImitate",
}

results: list[dict] = []
holes: list[str] = []


def phase(name, status, detail, seconds=0.0):
    results.append(
        {"phase": name, "status": status, "detail": detail, "seconds": round(seconds, 1)}
    )
    print(f"[{status:7s}] {name:34s} {seconds:6.1f}s  {detail}", flush=True)


def run(cmd, env, cwd, timeout):
    t = time.time()
    try:
        p = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or ""), time.time() - t
    except subprocess.TimeoutExpired as ex:
        out = (
            (ex.stdout or b"").decode(errors="replace")
            if isinstance(ex.stdout, bytes)
            else (ex.stdout or "")
        )
        return None, out + f"\nTIMEOUT after {timeout}s", time.time() - t


def tail(text, n=12):
    return " | ".join(line.strip() for line in text.strip().splitlines()[-n:])


def clean_env(root: Path, venv: Path) -> dict:
    env = {
        k: v
        for k, v in os.environ.items()
        if not (
            k.startswith("ROYALE")
            or k in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONSTARTUP")
            or k.startswith("CONDA")
        )
    }
    home = root / "home"
    home.mkdir(exist_ok=True)
    env["HOME"] = str(home)
    env["PIP_NO_CACHE_DIR"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["SDL_VIDEODRIVER"] = "dummy"
    env["SDL_AUDIODRIVER"] = "dummy"
    env["PYTHONUTF8"] = "1"
    bindir = venv / ("Scripts" if os.name == "nt" else "bin")
    env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
    env["VIRTUAL_ENV"] = str(venv)
    return env


def base_python() -> str:
    return getattr(sys, "_base_executable", None) or sys.executable


def fetch_quickstart(gym: Path, dest: Path) -> str | None:
    """examples/quickstart.py as a user gets it: the file copied into an empty folder."""
    f = gym / "examples" / "quickstart.py"
    if not f.exists():
        return None
    dest.write_bytes(f.read_bytes())
    return str(f)


TRACE_SNIPPET = r"""
import sys, royalegym
from royalegym import ClashParallelEnv
from royalegym.replay import ReplayRecorder
from royalegym.rust_engine import RustEngine
import numpy as np
rec = ReplayRecorder()
env = ClashParallelEnv(RustEngine(), recorder=rec)
obs, _ = env.reset(seed=1)
rng = np.random.default_rng(1)
for _ in range(400):
    a = {k: int(rng.choice(np.flatnonzero(obs[k]["action_mask"]))) for k in ("blue", "red")}
    obs, _, term, trunc, _ = env.step(a)
    if term["blue"] or trunc["blue"]:
        break
royalegym.save_trace(rec.trace, "battle.msgpack")
print("trace saved", len(rec.trace.frames) if hasattr(rec.trace, "frames") else "")
"""

ISOLATION_SNIPPET = r"""
import importlib, json, sys, sysconfig
site = sysconfig.get_paths()["purelib"]
out = {}
for name in %r:
    try:
        m = importlib.import_module(name)
        out[name] = {"file": getattr(m, "__file__", None),
                     "version": getattr(m, "__version__", None)}
    except Exception as ex:
        out[name] = {"error": f"{type(ex).__name__}: {ex}"}
cts = None
try:
    import royalesim
    cts = royalesim.card_table_source() if hasattr(royalesim, "card_table_source") else "absent"
except Exception as ex:
    cts = f"error {type(ex).__name__}"
print(json.dumps({"site": site, "path": sys.path, "mods": out, "card_table_source": cts}))
"""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--gym",
        metavar="PATH",
        required=True,
        help="the RoyaleGym checkout to install royalegym[all] from",
    )
    ap.add_argument(
        "--wheels",
        metavar="DIR",
        help="a folder of wheels (pip --find-links); royalesim must come from here",
    )
    ap.add_argument("--python", default=base_python(), help="interpreter that creates the venv")
    ap.add_argument(
        "--steps",
        type=int,
        default=1024,
        help="the quickstart's timestep_limit for this run",
    )
    ap.add_argument("--timeout", type=int, default=900, help="seconds the quickstart may take")
    ap.add_argument("--keep", action="store_true", help="keep the temporary folder")
    ap.add_argument("--report", help="write the results as JSON here")
    args = ap.parse_args()
    gym = Path(args.gym).resolve()
    workspace = gym

    root = Path(tempfile.mkdtemp(prefix="royale-fresh-"))
    work, venv = root / "work", root / "venv"
    work.mkdir()
    print(f"fresh folder {root}  (the checkout {gym} is never on the path)")
    try:
        # P1
        t = time.time()
        code, out, _ = run([args.python, "-m", "venv", str(venv)], os.environ.copy(), root, 300)
        if code != 0:
            phase("P1 create a venv", "FAIL", tail(out), time.time() - t)
            return finish(args, root, 1)
        env = clean_env(root, venv)
        py = str(venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
        run([py, "-m", "pip", "install", "-q", "--upgrade", "pip"], env, work, 300)
        phase("P1 create a venv", "PASS", f"{args.python} -> {py}", time.time() - t)

        # P2
        reqs, notes = [f"{gym}[all]"], []
        if not args.wheels:
            notes.append(
                "no --wheels: royalesim has no prebuilt wheel source, and a fresh user has no Rust"
            )
        cmd = (
            [py, "-m", "pip", "install"]
            + (["--find-links", str(Path(args.wheels).resolve())] if args.wheels else [])
            + reqs
        )
        code, out, secs = run(cmd, env, work, 1800)
        if code != 0:
            status = "BLOCKED" if (not args.wheels and "royalesim" in out) else "FAIL"
            if "does not provide the extra 'all'" in out or "provides no extra" in out.lower():
                holes.append("royalegym has no [all] extra")
            phase(
                "P2 install royalegym[all]",
                status,
                tail(out, 8) + ("; " + "; ".join(notes) if notes else ""),
                secs,
            )
            return finish(args, root, 1)
        if "does not provide the extra 'all'" in out:
            holes.append(
                "royalegym has no [all] extra: pip installed it bare and skipped the siblings"
            )
        phase("P2 install royalegym[all]", "PASS", " ".join(reqs)[:300], secs)

        # P3
        code, out, secs = run([py, "-c", ISOLATION_SNIPPET % PACKAGES], env, work, 300)
        try:
            info = json.loads(out.strip().splitlines()[-1])
        except Exception:
            phase("P3 imports come from the venv", "FAIL", tail(out), secs)
            return finish(args, root, 1)
        bad, blocked = [], []
        for name, m in info["mods"].items():
            if "error" in m and name == "royalesim" and not args.wheels:
                blocked.append("royalesim not installed: no wheel source (--wheels)")
            elif "error" in m:
                bad.append(f"{name}: {m['error']}")
            elif m["file"] and str(workspace).lower() in str(Path(m["file"]).resolve()).lower():
                bad.append(f"{name} imported from the workspace {m['file']}")
            elif not m.get("version"):
                holes.append(f"{name} has no __version__")
        # A wheel must carry its card table: on a fresh machine card_table_source() is "embedded".
        cts = info.get("card_table_source")
        if "error" not in info["mods"].get("royalesim", {"error": 1}):
            if cts == "absent":
                holes.append(
                    "royalesim has no card_table_source(): the wheel's card table is unproven"
                )
            elif cts != "embedded":
                bad.append(
                    f"royalesim.card_table_source() is {cts!r}, not 'embedded': "
                    "the install reads data from outside the wheel"
                )
        leaked = [p for p in info["path"] if p and str(workspace).lower() in p.lower()]
        if leaked:
            bad.append(f"workspace on sys.path: {leaked}")
        mods = ", ".join(
            f"{k} {v.get('version')}" for k, v in info["mods"].items() if "error" not in v
        )
        phase(
            "P3 imports come from the venv",
            "FAIL" if bad else ("BLOCKED" if blocked else "PASS"),
            "; ".join(bad + blocked) + ("; others: " + mods if (bad or blocked) else mods),
            secs,
        )
        if bad:
            return finish(args, root, 1)
        sim_missing = bool(blocked)
        for script in ("royaleviser",):
            if not shutil.which(script, path=env["PATH"]):
                holes.append(
                    f"no `{script}` console command after install (only python -m {script})"
                )

        # P4
        qs = work / "quickstart.py"
        src = fetch_quickstart(gym, qs)
        if not src:
            phase("P4 quickstart.py runs", "BLOCKED", f"no examples/quickstart.py in {gym}")
        else:
            text = qs.read_text(encoding="utf-8")
            small = {**SMALL, "timestep_limit=1_000_000_000,": f"timestep_limit={args.steps},"}
            moved = [k for k in small if text.count(k) != 1]
            if moved:
                phase(
                    "P4 quickstart.py runs",
                    "FAIL",
                    f"from {src}: not there exactly once, so the size cannot be set: {moved}",
                )
            else:
                for big, little in small.items():
                    text = text.replace(big, little)
                qs.write_text(text, encoding="utf-8")
                code, out, secs = run([py, "quickstart.py"], env, work, args.timeout)
                ckpts = work / "runs" / "quickstart" / "checkpoints"
                saved = ckpts.is_dir() and any(ckpts.iterdir())
                ok = code == 0 and saved
                phase(
                    "P4 quickstart.py runs",
                    "PASS" if ok else ("BLOCKED" if sim_missing and "royalesim" in out else "FAIL"),
                    f"from {src}; timestep_limit {args.steps}; exit {code}; checkpoint "
                    + ("written" if saved else "MISSING")
                    + "; "
                    + tail(out, 6),
                    secs,
                )

        # P5
        code, out, secs = run([py, "-c", TRACE_SNIPPET], env, work, 600)
        trace = work / "battle.msgpack"
        ok = code == 0 and trace.exists() and trace.stat().st_size > 0
        phase(
            "P5 record a battle (public API)",
            "PASS" if ok else ("BLOCKED" if sim_missing and "royalesim" in out else "FAIL"),
            (f"{trace.stat().st_size} bytes; " if trace.exists() else "") + tail(out, 4),
            secs,
        )

        # P6
        if not ok:
            phase("P6 viewer opens it headless", "BLOCKED", "no trace from P5")
        else:
            shot = work / "shot.png"
            code, out, secs = run(
                [py, "-m", "royaleviser", str(trace), "--seconds", "2", "--shot", str(shot)],
                env,
                work,
                300,
            )
            good = code == 0 and shot.exists() and shot.stat().st_size > 1000
            phase(
                "P6 viewer opens it headless",
                "PASS" if good else "FAIL",
                (f"shot {shot.stat().st_size} bytes; " if shot.exists() else "no shot; ")
                + f"exit {code}; "
                + tail(out, 4),
                secs,
            )
        # P7: the README's "Try it" as pasted, then the viewer command the README gives for the
        # battle it saved, headless.
        page = gym / "README.md"
        readme = page.read_text(encoding="utf-8") if page.exists() else ""
        program = re.search(r"## Try it.*?```python\n(.*?)```", readme, re.S)
        command = re.search(r"`(royaleviser [^`]+\.msgpack)`", readme)
        if not (program and command):
            phase(
                "P7 the README's Try it and its viewer command",
                "FAIL",
                "the README has no Try it program or no royaleviser command for its battle",
            )
        else:
            code, out7, secs = run([py, "-c", program.group(1)], env, work, 600)
            words = command.group(1).split()
            exe = shutil.which(words[0], path=env["PATH"])
            shot = work / "shot7.png"
            if code != 0 or not exe:
                phase(
                    "P7 the README's Try it and its viewer command",
                    "FAIL",
                    f"Try it exit {code}; {words[0]!r} found: {bool(exe)}; " + tail(out7, 4),
                    secs,
                )
            else:
                code, out7, secs2 = run(
                    [exe, *words[1:], "--seconds", "2", "--shot", str(shot)], env, work, 300
                )
                good = code == 0 and shot.exists() and shot.stat().st_size > 1000
                phase(
                    "P7 the README's Try it and its viewer command",
                    "PASS" if good else "FAIL",
                    f"{command.group(1)} -> exit {code}; "
                    + (f"shot {shot.stat().st_size} bytes" if shot.exists() else "no shot")
                    + "; "
                    + tail(out7, 3),
                    secs + secs2,
                )
        return finish(args, root, 0 if all(r["status"] == "PASS" for r in results) else 1)
    except Exception as ex:  # a crash of this script is a FAIL of the test, said as such
        phase("test", "FAIL", f"{type(ex).__name__}: {ex}")
        return finish(args, root, 1)


def finish(args, root: Path, code: int) -> int:
    print()
    for h in holes:
        print(f"HOLE  {h}")
    counts = {s: sum(r["status"] == s for r in results) for s in ("PASS", "FAIL", "BLOCKED")}
    verdict = (
        "PASS" if code == 0 else ("BLOCKED" if counts["BLOCKED"] and not counts["FAIL"] else "FAIL")
    )
    print(
        f"FRESH-USER TEST: {verdict}  ({counts['PASS']} passed, {counts['FAIL']} failed, "
        f"{counts['BLOCKED']} blocked; "
        f"{len(holes)} holes)" + ("" if code == 0 else "  -- BLOCKED IS NOT A PASS")
    )
    if args.report:
        Path(args.report).write_text(
            json.dumps({"verdict": verdict, "phases": results, "holes": holes}, indent=1)
        )
    if args.keep:
        print(f"kept {root}")
    else:
        shutil.rmtree(root, ignore_errors=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
