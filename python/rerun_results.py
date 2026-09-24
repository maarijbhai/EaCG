#!/usr/bin/env python3
"""Regenerate committed results with the exact parameters they were made with.

Every experiment script records `vars(args)` in its JSON sidecar, so the
command that produced a CSV can be rebuilt from the sidecar alone. That
matters for a before-and-after: several committed results were made with
non-default arguments (`endtoend` at n=12, `pitchsens` at n=8), and simply
re-running a script with its defaults produces a different experiment --
different row counts, nothing comparable -- which is how this was found.

    python rerun_results.py                         # every sidecar
    python rerun_results.py results/gate            # one directory
    python rerun_results.py --repo C:/wt/before     # scripts of another
                                                    # checkout (a worktree)

`--repo` runs the scripts and writes the results of ANOTHER checkout, which
is how a before-and-after isolates one commit: add a worktree at the parent
commit, rerun there, rerun here, then `compare_results.py --against`.

Experiments run one at a time, never in parallel: several record latency,
and contention would be measured instead of the stage.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: results/<dir> -> the script that writes it.
SCRIPTS = {
    "deskew": "run_deskew.py",
    "gate": "run_gate.py",
    "grid": "run_gridchannel.py",
    "photo": "run_photo.py",
    "kitphoto": "score_kit_photo.py",
}

#: Experiments whose sidecar params are a record, not a command line.
FIXED_ARGV = {
    ("kitphoto", "selftest"): ["--simulate", "--no-diagnostics"],
    ("kitphoto", "score"): ["--no-diagnostics"],
}

#: Parameters recorded for provenance that are not command-line flags.
NOT_FLAGS = {"rois"}


def known_flags(script: Path) -> set[str]:
    """The long options a script's argparse accepts, from its --help."""
    out = subprocess.run([sys.executable, "-B", script.name, "--help"],
                         cwd=script.parent, capture_output=True, text=True).stdout
    return set(re.findall(r"(--[a-z][a-z0-9-]*)", out))


def argv_for(params: dict) -> list[str]:
    out: list[str] = []
    for k, v in params.items():
        if k in NOT_FLAGS:
            continue
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                out.append(flag)
        elif isinstance(v, list):
            out += [flag, *map(str, v)]
        elif v is None:
            continue
        else:
            out += [flag, str(v)]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*", help="results dirs or sidecars; default all")
    p.add_argument("--repo", type=Path, default=REPO,
                   help="checkout whose scripts run and whose results/ is written")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    sidecars = []
    targets = [Path(x) for x in a.paths] or [REPO / "results" / d for d in SCRIPTS]
    # Sidecars are always read from THIS checkout: they define the
    # experiment. --repo only says whose code runs it.
    for t in targets:
        t = t if t.is_absolute() else REPO / t
        sidecars += [t] if t.suffix == ".json" else sorted(t.glob("*.json"))

    failures = 0
    for sc in sidecars:
        meta = json.loads(sc.read_text(encoding="utf-8"))
        script = SCRIPTS.get(sc.parent.name)
        if script is None or "experiment" not in meta or "params" not in meta:
            print(f"skip {sc.relative_to(REPO)}: no script mapping or no params")
            continue
        fixed = FIXED_ARGV.get((sc.parent.name, meta["experiment"]))
        params = dict(meta["params"])
        # A sidecar can record flags its script has since dropped: the grid
        # suite recorded --thresholds, --synth-px-per-mm and --detrend-window,
        # which run_gridchannel.py now fixes as constants. Passing them
        # would be a usage error, so each is dropped with a notice -- the
        # script's constant is then what runs, and the notice says so.
        accepted = known_flags(a.repo / "python" / script)
        for k in list(params):
            if "--" + k.replace("_", "-") not in accepted and k not in NOT_FLAGS:
                print(f"  note: {script} no longer takes --{k.replace('_', '-')}; "
                      f"recorded value {params[k]!r} not passed")
                del params[k]
        cmd = [sys.executable, "-B", "-u", script,
               *(fixed if fixed is not None else
                 ["--experiment", meta["experiment"], *argv_for(params)])]
        print(f"{sc.parent.name}/{meta['experiment']}: {' '.join(cmd[3:])}", flush=True)
        if a.dry_run:
            continue
        t0 = time.time()
        r = subprocess.run(cmd, cwd=a.repo / "python")
        print(f"  exit {r.returncode} in {time.time() - t0:.1f} s", flush=True)
        failures += r.returncode != 0
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
