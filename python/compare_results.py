#!/usr/bin/env python3
"""Before-and-after of committed results across a code change.

The Conventions demand a before-and-after on every existing result when the
control condition changes, and a raw `git diff` of results/ is useless for
it: timing columns move on every run, so every CSV that records latency
shows as changed whether or not a single measured value did. This reads the
committed version of each CSV (`git show REF:path`) and the working copy,
drops timing columns, and reports per column how far the values moved.

    python compare_results.py                  # results/ against HEAD
    python compare_results.py --ref b4567d4 results/kitphoto/selftest.csv
    python compare_results.py --against C:/wt/before --csv out.csv

`--against DIR` compares the working copy with the same file in another
checkout instead of a git ref: the other half of `rerun_results.py --repo`.
`--csv` also writes the per-column summary, so a before-and-after can be
committed instead of living in a terminal.

A column is TIMING if its name contains `_ms`, `ms_`, `elapsed` or ends in
`ms`; those are reported as a count only. Rows are matched by position,
which holds because every experiment script iterates its conditions in a
fixed order; a changed row count is reported instead of a comparison.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TIMING = re.compile(r"(_ms|ms_|elapsed|^ms$|ms$)")


def read_ref(ref: str, path: str) -> list[dict] | None:
    try:
        text = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        return None
    return list(csv.DictReader(io.StringIO(text)))


def num(v: str) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def compare(path: str, ref: str, against: Path | None = None,
            summary: list[dict] | None = None) -> list[str]:
    if against is not None:
        f = against / path
        old = list(csv.DictReader(f.open(encoding="utf-8"))) if f.exists() else None
    else:
        old = read_ref(ref, path)
    new = list(csv.DictReader((REPO / path).open(encoding="utf-8")))
    if old is None:
        return [f"{path}: new file, {len(new)} rows"]
    if len(old) != len(new):
        return [f"{path}: row count {len(old)} -> {len(new)}; not compared"]
    cols = [c for c in new[0].keys()] if new else []
    timing = [c for c in cols if TIMING.search(c)]
    out = []
    for c in cols:
        if c in timing:
            continue
        changed, maxd, text_changes = 0, 0.0, 0
        for a, b in zip(old, new):
            va, vb = a.get(c, ""), b.get(c, "")
            if va == vb:
                continue
            na, nb = num(va), num(vb)
            if na is not None and nb is not None:
                if na != nb:
                    changed += 1
                    maxd = max(maxd, abs(nb - na))
            else:
                text_changes += 1
        if summary is not None:
            summary.append({"file": path, "column": c, "rows": len(new),
                            "numeric_changed": changed, "max_abs_delta": maxd,
                            "text_changed": text_changes})
        if changed or text_changes:
            out.append(f"    {c}: {changed} numeric rows changed (max |d| {maxd:.6g})"
                       + (f", {text_changes} text/NaN changes" if text_changes else ""))
    head = f"{path}: {len(new)} rows, {len(timing)} timing column(s) ignored"
    return [head + (" -- NO measured value changed" if not out else "")] + out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*")
    p.add_argument("--ref", default="HEAD")
    p.add_argument("--against", type=Path, default=None)
    p.add_argument("--csv", type=Path, default=None)
    a = p.parse_args(argv)
    summary: list[dict] = []
    paths = a.paths or sorted(
        str(q.relative_to(REPO)).replace("\\", "/")
        for q in (REPO / "results").rglob("*.csv"))
    for path in paths:
        for line in compare(path, a.ref, a.against, summary):
            print(line)
    if a.csv is not None:
        a.csv.parent.mkdir(parents=True, exist_ok=True)
        with a.csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["file", "column", "rows",
                                               "numeric_changed", "max_abs_delta",
                                               "text_changed"])
            w.writeheader()
            w.writerows(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
