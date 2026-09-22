"""Mutation check on the FR8 gate's test suite.

A passing test suite proves that the tests pass. It does not prove they
would notice if the code were wrong, and for a gate that distinction is
the whole value of the suite: a gate that accepted everything, or one
whose thresholds had been loosened by an order of magnitude, would sail
through tests that only ever assert "bad input is rejected".

So: break `quality.py` deliberately, one edit at a time, and require the
suite to fail. A mutation that SURVIVES is a hole in the tests, and the
holes are reported rather than quietly fixed -- the same discipline the
Kotlin port fixture used in section 12.8, which found that swapping
`Math.rint` for `Math.round` passed every test.

    python tests/mutate_quality.py

Restores the original file on every exit path, including Ctrl-C.

KNOWN SURVIVOR, and why it is not a hole worth closing. "Unavailable
checks allowed to vote" -- deleting the `c.available and` guard from the
verdict combination -- survives because every check `assess` currently
constructs with `available=False` also carries `Verdict.ACCEPT`, so the
guard has nothing to suppress and the mutant is *equivalent*: no input
can distinguish it. The guard is kept as the enforcement of an invariant
a future check could break -- one that computes a rejection and only then
finds its anchor missing -- not because today's code needs it.
Manufacturing a test for an unreachable branch would be testing the
test, so it is reported here instead.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYDIR = HERE.parent
TARGET = PYDIR / "ecgproto" / "quality.py"

#: (label, exact source to replace, replacement). Each `old` must occur
#: exactly once in the file, which is checked before anything is run --
#: a mutation that silently fails to apply would be reported as caught
#: by whatever the suite happened to do, which is worse than no check.
MUTATIONS: list[tuple[str, str, str]] = [
    ("amplitude reject ceiling loosened 10x",
     "REJECT_PP_MV = (0.05, 6.0)",
     "REJECT_PP_MV = (0.05, 60.0)"),
    ("amplitude reject floor removed",
     "FLAG_PP_MV = (0.20, 3.5)",
     "FLAG_PP_MV = (0.0, 3.5)"),
    ("band count off by one",
     "elif bands >= expected_traces + 1:",
     "elif bands >= expected_traces + 2:"),
    ("band count mean instead of median",
     'return float(np.median(counts)) if counts else float("nan")',
     'return float(np.mean(counts)) if counts else float("nan")'),
    ("band separation comparison made inclusive",
     "counts.append(1 + int(np.count_nonzero(np.diff(rows) > gap_px)))",
     "counts.append(1 + int(np.count_nonzero(np.diff(rows) >= gap_px)))"),
    ("band separation read as pixels not millimetres",
     "gap_px = max(2, int(round(band_separation_mm * px_per_mm))) \\\n        if np.isfinite(px_per_mm) and px_per_mm > 0 else 2",
     "gap_px = 8"),
    ("verdict combination takes the mildest check",
     "if c.available and c.verdict.value > verdict.value:",
     "if c.available and c.verdict.value < verdict.value:"),
    ("unavailable checks allowed to vote",
     "if c.available and c.verdict.value > verdict.value:",
     "if c.verdict.value > verdict.value:"),
    ("calibration rejection downgraded to a flag",
     "            verdict = Verdict.REJECT\n            detail = \"; \".join(cal.reasons)",
     "            verdict = Verdict.FLAG\n            detail = \"; \".join(cal.reasons)"),
    ("heart rate formula inverted",
     "return 60.0 / median_rr_s",
     "return median_rr_s / 60.0"),
    ("heart rate refractory period 10x too short",
     "distance=max(1, int(0.24 * fs))",
     "distance=max(1, int(0.024 * fs))"),
    ("coverage comparison flipped",
     "cov_verdict = (Verdict.REJECT if not np.isfinite(cov) or cov < reject_coverage",
     "cov_verdict = (Verdict.REJECT if not np.isfinite(cov) or cov > reject_coverage"),
    ("flag band collapsed into accept",
     "    if value < flag[0] or value > flag[1]:\n        return Verdict.FLAG",
     "    if value < flag[0] or value > flag[1]:\n        return Verdict.ACCEPT"),
    ("non-finite value treated as plausible",
     "    if not np.isfinite(value):\n        return Verdict.REJECT",
     "    if not np.isfinite(value):\n        return Verdict.ACCEPT"),
    ("missing trace mask reported as a passed check",
     '            detail="no trace mask supplied; stacked-layout check could not run",\n            available=False,',
     '            detail="no trace mask supplied; stacked-layout check could not run",\n            available=True,'),
]


def run_suite() -> tuple[bool, str]:
    """True if the suite passes. Bytecode caching disabled for safety."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=PYDIR, capture_output=True, text=True, env=env,
    )
    return proc.returncode == 0, (proc.stderr or proc.stdout).strip()


def main() -> int:
    original = TARGET.read_text(encoding="utf-8")

    missing = [label for label, old, _ in MUTATIONS
               if original.count(old) != 1]
    if missing:
        print("MUTATIONS THAT DO NOT APPLY CLEANLY (fix these first):")
        for label in missing:
            print(f"  {label}")
        return 2

    baseline_ok, output = run_suite()
    if not baseline_ok:
        print("BASELINE SUITE ALREADY FAILS -- mutation results would be "
              "meaningless.\n")
        print(output[-2000:])
        return 2
    print(f"baseline: suite passes\n{len(MUTATIONS)} mutations\n")

    survived: list[str] = []
    try:
        for label, old, new in MUTATIONS:
            TARGET.write_text(original.replace(old, new), encoding="utf-8")
            ok, _ = run_suite()
            if ok:
                survived.append(label)
            print(f"  {'SURVIVED' if ok else 'caught  '}  {label}")
    finally:
        TARGET.write_text(original, encoding="utf-8")

    caught = len(MUTATIONS) - len(survived)
    print(f"\n{caught}/{len(MUTATIONS)} mutations caught")
    if survived:
        print("\nSURVIVORS -- the suite does not constrain these:")
        for label in survived:
            print(f"  {label}")
    return 1 if survived else 0


if __name__ == "__main__":
    raise SystemExit(main())
