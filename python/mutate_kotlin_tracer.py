#!/usr/bin/env python3
"""Mutation check on the Kotlin tracer's port tests.

The port tests pass. That proves they pass. It does not prove they would
notice if `TraceExtractor.kt` or `QualityGate.kt` were wrong, which is the
only reason to have them -- and section 12.8 of CHANGES.txt records exactly
how that assumption fails: the skew fixture's own comment claimed it caught a
`rint`/`round` swap, and measurement showed it did not.

So: break the Kotlin deliberately, one edit at a time, and require
`testDebugUnitTest` to fail. A mutation that SURVIVES is a hole in the tests,
and holes are reported rather than quietly patched.

    python python/mutate_kotlin_tracer.py

Each mutation costs a Kotlin recompile, so the whole run is a few minutes.
The target files are restored on every exit path, including Ctrl-C.

KNOWN SURVIVOR, and why it is not a hole. "Resample interpolates on a
fractional index" -- replacing the `slope * (t - xLo) + mv[lo]` form, which
is how `np.interp` spells it, with the algebraically identical
`mv[lo] + frac * (mv[lo+1] - mv[lo])` on a fractional column index --
survives because the two forms agree to better than the tests' 1e-12
relative tolerance on this fixture. Both are correct; they differ only in
the last bits, and the tolerance does not resolve that. Worth knowing that
the NumPy-shaped spelling in `TraceExtractor.resample` is therefore
defensive rather than load-bearing. Tightening the tolerance to catch it
would make the suite fail on unrelated arithmetic noise, which is a worse
trade.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ANDROID = REPO / "android"
SRC = ANDROID / "app" / "src" / "main" / "java" / "com" / "maarijalam" / "eacg"
TRACE = SRC / "TraceExtractor.kt"
GATE = SRC / "QualityGate.kt"

#: `gradlew.bat` is broken in this project's Gradle 9.5 wrapper -- it sets an
#: empty CLASSPATH -- so builds go through the wrapper jar directly. Same
#: invocation CLAUDE.md documents.
GRADLE = [
    "java", "-Dorg.gradle.appname=gradlew",
    "-jar", "gradle/wrapper/gradle-wrapper.jar",
    "testDebugUnitTest",
]

#: (label, file, exact source, replacement). Every `old` must occur exactly
#: once, checked before anything runs: a mutation that fails to apply would
#: be scored as caught by whatever the suite happened to do anyway.
MUTATIONS: list[tuple[str, Path, str, str]] = [
    ("empty column returns 0.0 instead of NaN", TRACE,
     "out[c] = if (weight > 0.0) sum / weight else Double.NaN",
     "out[c] = if (weight > 0.0) sum / weight else 0.0"),
    ("millivolt sign flipped", TRACE,
     "return DoubleArray(yPx.size) { (baselinePx - yPx[it]) / pxPerMv }",
     "return DoubleArray(yPx.size) { (yPx[it] - baselinePx) / pxPerMv }"),
    ("median takes the lower middle at even length", TRACE,
     "else 0.5 * (finite[mid - 1] + finite[mid])",
     "else finite[mid - 1]"),
    ("coverage measured before the trim, not after", TRACE,
     "return Trace(span, finite.toDouble() / span.size, lo, hi)",
     "return Trace(span, finite.toDouble() / yPx.size, lo, hi)"),
    ("long gaps bridged instead of re-blanked", TRACE,
     "            if (end - i > maxGapPx) {\n                for (k in i until end) out[k] = Double.NaN\n            }",
     "            if (false) {\n                for (k in i until end) out[k] = Double.NaN\n            }"),
    ("resample length rounds instead of flooring", TRACE,
     "val nOut = kotlin.math.floor(durationS * fsOut).toInt()",
     "val nOut = Math.round(durationS * fsOut).toInt()"),
    ("resample interpolates on a fractional index", TRACE,
     "                val xLo = lo / pxPerS\n                val xHi = (lo + 1) / pxPerS\n                val slope = (mv[lo + 1] - mv[lo]) / (xHi - xLo)\n                out[i] = slope * (t - xLo) + mv[lo]",
     "                val frac = t * pxPerS - lo\n                out[i] = mv[lo] + frac * (mv[lo + 1] - mv[lo])"),
    ("trim accepts a single inked column", TRACE,
     "if (lo < 0 || hi - lo + 1 < 2) {",
     "if (lo < 0) {"),
    ("band count uses the mean, not the median", GATE,
     "        return if (counts.size % 2 == 1) counts[mid].toDouble()\n        else 0.5 * (counts[mid - 1] + counts[mid])",
     "        return counts.average()"),
    ("band separation read as pixels, not millimetres", GATE,
     "        val gapPx = if (pxPerMm.isFinite() && pxPerMm > 0.0) {\n            max(2, (bandSeparationMm * pxPerMm).roundToInt())\n        } else 2",
     "        val gapPx = 8"),
    ("band threshold off by one", GATE,
     "                    bands >= expectedTraces + 1 -> Verdict.REJECT",
     "                    bands >= expectedTraces + 2 -> Verdict.REJECT"),
    ("unavailable outranks accept", GATE,
     "    enum class Verdict { UNAVAILABLE, ACCEPT, FLAG, REJECT }",
     "    enum class Verdict { ACCEPT, UNAVAILABLE, FLAG, REJECT }"),
    ("amplitude reject ceiling loosened tenfold", GATE,
     "    const val REJECT_PP_MV_HI = 6.0",
     "    const val REJECT_PP_MV_HI = 60.0"),
    ("anisotropy limit loosened tenfold", GATE,
     "    const val MAX_ANISOTROPY = 0.15",
     "    const val MAX_ANISOTROPY = 1.5"),
    ("paper speed check inverted", GATE,
     "            val bad = implied < PAPER_SPEED_LO || implied > PAPER_SPEED_HI",
     "            val bad = implied > PAPER_SPEED_LO && implied < PAPER_SPEED_HI"),
    ("coverage comparison flipped", GATE,
     "                !cov.isFinite() || cov < REJECT_COVERAGE -> Verdict.REJECT",
     "                !cov.isFinite() || cov > REJECT_COVERAGE -> Verdict.REJECT"),
]


def run_tests() -> bool:
    proc = subprocess.run(GRADLE, cwd=ANDROID, capture_output=True, text=True)
    return proc.returncode == 0


def main() -> int:
    originals = {p: p.read_text(encoding="utf-8") for p in (TRACE, GATE)}

    bad = [label for label, path, old, _ in MUTATIONS
           if originals[path].count(old) != 1]
    if bad:
        print("MUTATIONS THAT DO NOT APPLY CLEANLY (fix these first):")
        for label in bad:
            print(f"  {label}")
        return 2

    print("baseline: building and running the suite ...", flush=True)
    if not run_tests():
        print("BASELINE SUITE ALREADY FAILS -- mutation results would be "
              "meaningless. Fix that first.")
        return 2
    print(f"baseline passes. {len(MUTATIONS)} mutations\n", flush=True)

    survived: list[str] = []
    try:
        for label, path, old, new in MUTATIONS:
            path.write_text(originals[path].replace(old, new), encoding="utf-8")
            ok = run_tests()
            path.write_text(originals[path], encoding="utf-8")
            if ok:
                survived.append(label)
            print(f"  {'SURVIVED' if ok else 'caught  '}  {label}", flush=True)
    finally:
        for path, text in originals.items():
            path.write_text(text, encoding="utf-8")

    caught = len(MUTATIONS) - len(survived)
    print(f"\n{caught}/{len(MUTATIONS)} mutations caught")
    if survived:
        print("\nSURVIVORS -- the suite does not constrain these:")
        for label in survived:
            print(f"  {label}")
    return 1 if survived else 0


if __name__ == "__main__":
    raise SystemExit(main())
