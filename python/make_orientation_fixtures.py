#!/usr/bin/env python3
"""Golden vectors for SheetOrientation.kt's orientation rule, from ecgproto/orientation.py.

Each case: whether the corner outline is sideways, the lead markers
(pulses, ticks) on the grid-fit page and on it turned half round, and what
the reference decides -- `upside_down` and the clockwise quarter turns the
app applies (a quarter if sideways, a half more if upside down).

Cases: every session photograph measured in
results/kitphoto/orientation_sessions.csv (labelled with its name), then
every combination of 0-4 pulses and 0-9 ticks on both pages, sideways or not.

    python make_orientation_fixtures.py
"""

from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ecgproto import orientation as O  # noqa: E402

REPO = HERE.parent
SESSIONS = REPO / "results" / "kitphoto" / "orientation_sessions.csv"
OUT = REPO / "android" / "app" / "src" / "test" / "resources" / "orientation_fixtures.txt"


def line(name: str, sideways: bool, up: O.Markers, turned: O.Markers) -> str:
    down = O.upside_down(up, turned)
    q = ((1 if sideways else 0) + (2 if down else 0)) % 4
    return (f"{name} {int(sideways)} {up.pulses} {up.ticks} {turned.pulses} {turned.ticks} "
            f"{int(down)} {q}")


def main() -> int:
    out = ["# name sideways up_pulses up_ticks turned_pulses turned_ticks upside_down quarter_turns_cw",
           f"# from ecgproto/orientation.py (MAX_TICKS_TURNED = {O.MAX_TICKS_TURNED}); python make_orientation_fixtures.py"]
    with SESSIONS.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["up_pulses"] == "":
                continue                                  # no corners: nothing decided
            up = O.Markers(int(float(r["up_pulses"])), int(float(r["up_ticks"])))
            tu = O.Markers(int(float(r["turned_pulses"])), int(float(r["turned_ticks"])))
            out.append(line("session:" + r["photo"], float(r["aspect"]) < O.SIDEWAYS_BELOW, up, tu))
    for s, up_p, up_t, tu_p, tu_t in itertools.product((False, True), range(5), range(10), range(5), range(10)):
        out.append(line("grid", s, O.Markers(up_p, up_t), O.Markers(tu_p, tu_t)))
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(out) - 2} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
