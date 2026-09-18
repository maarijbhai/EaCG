#!/usr/bin/env python3
"""Mutation check: does the test suite actually catch a porting bug?

A green test run means nothing on its own. It could mean the code is right,
or it could mean the tests do not look at the thing that matters. The only
way to tell is to break the code on purpose and check that the tests notice.

Each mutation below is a real mistake somebody could make porting
`ecgproto/deskew.py` to Kotlin, not an arbitrary character swap. Every one is
applied to the source, the suite is run, the source is restored, and the
result is reported. A mutation that SURVIVES -- tests still pass -- is a hole
in the suite and should be treated as a defect in the tests.

Run from `android/`::

    python check_tests_catch_bugs.py

Safety: each file's original text is held in memory and restored in a
`finally`, including on Ctrl-C, so an interrupted run does not leave a
mutated tree. Verify with `git status` afterwards regardless.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "app/src/main/java/com/maarijalam/eacg"

GRADLE = [
    "java", "-Dorg.gradle.appname=gradlew",
    "-jar", "gradle/wrapper/gradle-wrapper.jar",
    ":app:testDebugUnitTest",
]

#: (name, file, find, replace, why it is a plausible mistake)
MUTATIONS = [
    (
        "rint -> round",
        "SkewEstimator.kt",
        "val v = Math.rint(-c.toDouble() * tanA).toInt()",
        "val v = Math.round(-c.toDouble() * tanA).toInt()",
        "np.rint is round-half-to-even; Math.round is round-half-up. The "
        "obvious Kotlin spelling is the wrong one, and it only differs on "
        "exact .5 shifts.",
    ),
    (
        "binning slope sign flipped",
        "SkewEstimator.kt",
        "val t = tan(Math.toRadians(-angles[i]))",
        "val t = tan(Math.toRadians(angles[i]))",
        "The internal slope is the negation of the external angle. Dropping "
        "the minus returns the angle negated.",
    ),
    (
        "ssd divisor off by one",
        "SkewEstimator.kt",
        "return (ssd / (fixedLen - 1)) / (mean * mean)",
        "return (ssd / fixedLen) / (mean * mean)",
        "np.diff yields one fewer element than the profile; dividing by the "
        "profile length instead is a natural slip and scales every sharpness "
        "value slightly.",
    ),
    (
        "core crop not centred",
        "SkewEstimator.kt",
        "val from = span + (coreLen - fixedLen) / 2",
        "val from = span",
        "Taking the first fixedLen samples instead of the centred ones. The "
        "profile is still the right length, so nothing looks wrong.",
    ),
    (
        "arange ceil -> round",
        "SkewEstimator.kt",
        "val n = ceil((stop - start) / step).toInt()",
        "val n = Math.round((stop - start) / step).toInt()",
        "Changes sweep lengths by one at the ends, moving which sample the "
        "parabola is fitted around.",
    ),
    (
        "red and blue swapped",
        "GridResponse.kt",
        "        val r = data[i].toInt() and 0xFF\n"
        "        val g = data[i + 1].toInt() and 0xFF\n"
        "        val b = data[i + 2].toInt() and 0xFF",
        "        val b = data[i].toInt() and 0xFF\n"
        "        val g = data[i + 1].toInt() and 0xFF\n"
        "        val r = data[i + 2].toInt() and 0xFF",
        "OpenCV reads files BGR on the desktop and a Mat from a Bitmap is "
        "RGBA. This is the single easiest mistake in the whole port.",
    ),
    (
        "byte sign not masked",
        "GridResponse.kt",
        "        val r = data[i].toInt() and 0xFF",
        "        val r = data[i].toInt()",
        "Java bytes are signed, so anything above 127 goes negative and the "
        "response silently clips to zero on the brightest pixels.",
    ),
    (
        "parabola step dropped",
        "SkewEstimator.kt",
        "return angles[k] + 0.5 * step * (y0 - y2) / denom",
        "return angles[k] + 0.5 * (y0 - y2) / denom",
        "The sub-step offset is in units of the sweep step. Omitting it "
        "scales the refinement by 1/step, which at 0.05 is twentyfold.",
    ),
]


def run_tests() -> bool:
    """True if the suite passes."""
    proc = subprocess.run(GRADLE, cwd=HERE, capture_output=True, text=True)
    return proc.returncode == 0


def main() -> int:
    print("baseline: ", end="", flush=True)
    if not run_tests():
        print("FAIL -- the suite is red before any mutation. Fix that first.")
        return 1
    print("pass\n")

    caught, survived = 0, []
    for name, filename, find, replace, why in MUTATIONS:
        path = SRC / filename
        original = path.read_text(encoding="utf-8")
        if original.count(find) != 1:
            print(f"  SKIP  {name}: anchor not found exactly once in {filename}")
            print("        (the source moved; update this script)")
            survived.append(name)
            continue
        try:
            path.write_text(original.replace(find, replace), encoding="utf-8")
            passed = run_tests()
        finally:
            path.write_text(original, encoding="utf-8")

        if passed:
            print(f"  SURVIVED  {name}")
            print(f"            {why}")
            survived.append(name)
        else:
            caught += 1
            print(f"  caught    {name}")

    print(f"\n{caught}/{len(MUTATIONS)} mutations caught.")
    if survived:
        print("\nSURVIVING MUTATIONS ARE HOLES IN THE TEST SUITE:")
        for s in survived:
            print(f"  - {s}")
        print("\nEach one is a bug the tests would not have noticed.")
        return 1
    print("Every mutation was caught. Restoring and verifying the tree...")
    if not run_tests():
        print("WARNING: the suite is red after restore. Check `git status`.")
        return 1
    print("Tree restored and green.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
