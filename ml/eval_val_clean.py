#!/usr/bin/env python3
"""Row-head runs on CLEAN vs AUGMENTED fold-9 row bands: the scan-image check.

b2's presence head switched off on bright, pristine images (a flatbed scan
or a well-lit photo looks like that) while scoring well on augmented ones.
This scores each run twice on the same fold-9 row bands: un-augmented (the
clean render, strips.clean_canonical) and augmented (the training
distribution, one seeded draw per band). A run fit for scans keeps coverage
and PCC on the clean set without losing them on the augmented one.

    bash ml/wsl_run.sh python ml/eval_val_clean.py b2 b3

Writes results/ml/val_clean_<runs>.csv + .json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))
sys.path.insert(0, str(REPO / "python"))

import strips as S  # noqa: E402
import train_tracer as T  # noqa: E402
import score_12lead as S12  # noqa: E402

RESULTS = REPO / "results" / "ml"
N, STEP = 60, 20            # every 20th fold-9 row band, 60 of them


def main(runs: list[str]) -> int:
    root = Path.home() / "data" / "ml_rows"
    S.CANON_H = 512
    T._worker_init(str(root))
    stems = S.list_stems("val", root)[::STEP][:N]
    sets = {"clean": [T.make_example(("val", s, 0, False)) for s in stems],
            "augmented": [T.make_example(("val", s, 10_000 + i, True))
                          for i, s in enumerate(stems)]}
    import tensorflow as tf
    import keras
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    keras.mixed_precision.set_global_policy("mixed_float16")
    rows = []
    for run in runs:
        rec = json.loads((RESULTS / f"train_{run}.json").read_text())["params"]
        m = T.build_model("row", h=int(rec.get("canon_h", 384)))
        m.load_weights(REPO / "out" / "ml_runs" / run / "best.weights.h5")
        for name, ex in sets.items():
            sc = T.score_val(m, ex, "row", 1)
            rows.append({"run": run, "set": name, "n": len(ex),
                         **{k: round(v, 4) for k, v in sc.items() if isinstance(v, float)}})
            print(rows[-1], flush=True)
    S12.RESULTS = RESULTS
    print(S12.persist(rows, {"runs": runs, "n": N, "every": STEP,
                             "data": "data/ml_rows val (fold 9)"},
                      "val_clean_" + "_".join(runs)))
    return 0


if __name__ == "__main__":
    import os
    code = main(sys.argv[1:])
    sys.stdout.flush()
    os._exit(code)
