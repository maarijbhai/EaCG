"""Mutation check on the time-map tests (tests/test_timemap.py).

Breaks `ecgproto/timemap.py` (and the two places the time map reaches into
`segment.py` and `sheet.py`) one edit at a time and requires
tests/test_timemap.py to fail. A survivor is a hole in the tests and is
reported, not quietly fixed. Same discipline as tests/mutate_quality.py.

    python tests/mutate_timemap.py

Restores every file on every exit path, including Ctrl-C.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYDIR = HERE.parent
TIMEMAP = PYDIR / "ecgproto" / "timemap.py"
SEGMENT = PYDIR / "ecgproto" / "segment.py"
SHEET = PYDIR / "ecgproto" / "sheet.py"

#: (label, file, exact source, replacement). Each source must occur exactly
#: once in its file; that is checked before anything runs.
MUTATIONS: list[tuple[str, Path, str, str]] = [
    ("shift sign flipped", TIMEMAP,
     "return Shift(1000.0 * lag / fs, c, status)",
     "return Shift(-1000.0 * lag / fs, c, status)"),
    ("window-edge shift reported as a value", TIMEMAP,
     'status = "FAILED" if abs(lag) == m else',
     'status = "FAILED" if abs(lag) > m else'),
    ("correlation floor ignored", TIMEMAP,
     '("ok" if c >= min_pcc else "UNRELIABLE")',
     '("ok" if c >= -1.0 else "UNRELIABLE")'),
    ("NaN samples not excluded", TIMEMAP,
     "ok = np.isfinite(a) & np.isfinite(b)",
     "ok = np.isfinite(a)"),
    ("scale error per 10 s instead of per cell", TIMEMAP,
     "return 100.0 * self.slope_ms_per_cell / (1000.0 * segment.CELL_S)",
     "return 100.0 * self.slope_ms_per_cell / (1000.0 * segment.RECORDING_S)"),
    ("fit allowed on two points", TIMEMAP,
     "MIN_FIT_POINTS = 3",
     "MIN_FIT_POINTS = 2"),
    ("r2 inverted", TIMEMAP,
     "r2 = 1.0 - float(np.sum(res ** 2)) / tot if tot > 0",
     "r2 = float(np.sum(res ** 2)) / tot if tot > 0"),
    ("row's own start ignored", TIMEMAP,
     "xs, ts = [start], [0.0]",
     "xs, ts = [xs0[0]], [0.0]"),
    ("row end not pinned", TIMEMAP,
     "            xs.append(end)\n",
     "            xs.append(xs[-1] + (segment.RECORDING_S - ts[-1]) * layout.px_per_s)\n"),
    ("missing row end falls back silently", TIMEMAP,
     "if not np.isfinite(end) or end <= xs[-1]:",
     "if False:"),
    ("missing own pulse falls back silently", TIMEMAP,
     "        if not np.isfinite(start):\n",
     "        if False:\n"),
    ("grid mode uses the tick scale", TIMEMAP,
     "pps = grid_px_per_mm * dig.MM_PER_S",
     "pps = layout.px_per_s"),
    ("grid mode without a fit falls back", TIMEMAP,
     "if grid_px_per_mm is None or not np.isfinite(grid_px_per_mm):",
     "if False:"),
    ("even-kernel opening offset not undone", TIMEMAP,
     "shift = 1 if open_kernel % 2 == 0 else 0",
     "shift = 0"),
    ("segment_leads ignores the override", SEGMENT,
     "        knots = list(knots_override)\n",
     "        knots = list(knots)\n"),
    ("unavailable row silently cut at the uniform scale", SHEET,
     "        if layout.row_knots[cell.row] is None:      # time map unavailable\n"
     "            return np.full(n, np.nan)\n"
     "        t = layout.time_at(cell.row, xs) - cell.t_start_s\n",
     "        if layout.row_knots[cell.row] is None:      # time map unavailable\n"
     "            t = (xs - cell.x0) / (px_per_mm_x * dig.MM_PER_S)\n"
     "        else:\n"
     "            t = layout.time_at(cell.row, xs) - cell.t_start_s\n"),
]


def run_suite() -> bool:
    """True if tests/test_timemap.py passes."""
    r = subprocess.run([sys.executable, "-m", "unittest", "tests.test_timemap"],
                       cwd=PYDIR, capture_output=True, text=True,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    return r.returncode == 0


def main() -> int:
    originals = {p: p.read_text(encoding="utf-8") for p in (TIMEMAP, SEGMENT, SHEET)}
    for label, path, old, _ in MUTATIONS:
        n = originals[path].count(old)
        if n != 1:
            print(f"cannot apply '{label}': source found {n} times in {path.name}")
            return 2
    if not run_suite():
        print("the unmutated suite fails; fix that first")
        return 2
    survivors = []
    try:
        for label, path, old, new in MUTATIONS:
            path.write_text(originals[path].replace(old, new), encoding="utf-8")
            caught = not run_suite()
            path.write_text(originals[path], encoding="utf-8")
            print(f"  {'caught  ' if caught else 'SURVIVED'}  {label}")
            if not caught:
                survivors.append(label)
    finally:
        for p, src in originals.items():
            p.write_text(src, encoding="utf-8")
    print(f"{len(MUTATIONS) - len(survivors)} of {len(MUTATIONS)} mutations caught")
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
