#!/usr/bin/env python3
"""The PTB-XL labels of the printed test sheets, for the app's Diagnosis tab.

In prototype mode, when a scan is benchmarked against a printed sheet, the
Diagnosis tab shows the record's own PTB-XL superclasses next to the
classifier's output -- a check for the person holding the phone, never an
input to anything. One line per sheet in assets/kit/targets.txt:

    ecg_id=9 superclasses=NORM codes=NORM,SR

    python make_clf_kit_labels.py        (from python/)
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ecgproto import ptbxl  # noqa: E402

ASSETS = HERE.parent / "android" / "app" / "src" / "main" / "assets"
OUT = ASSETS / "ml" / "clf_kit_labels.txt"


def main() -> int:
    ids = []
    for line in (ASSETS / "kit" / "targets.txt").read_text(encoding="utf-8").splitlines():
        kv = dict(p.split("=", 1) for p in line.split() if "=" in p)
        if "ecg_id" in kv:
            ids.append(int(kv["ecg_id"]))
    df = ptbxl.load_metadata()
    y = ptbxl.superclass_matrix(df)
    pos = {int(e): k for k, e in enumerate(df.index)}
    lines = ["# written by python/make_clf_kit_labels.py: PTB-XL superclasses of the printed test sheets"]
    for e in sorted(ids):
        sup = [c for c, v in zip(ptbxl.SUPERCLASSES, y[pos[e]]) if v]
        codes = ast.literal_eval(df.loc[e, "scp_codes"]) if isinstance(df.loc[e, "scp_codes"], str) \
            else df.loc[e, "scp_codes"]
        lines.append(f"ecg_id={e} superclasses={','.join(sup) or 'none'} codes={','.join(codes)}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
