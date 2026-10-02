#!/usr/bin/env python3
"""The whole sheet with the learned tracers: classical vs model A vs model B.

The 12-lead counterpart of `eval_photos.py`. Segmentation (FR4) is the
classical `segment.locate_sheet` + `segment_leads`, identical for every
tracer; each model then runs ONCE PER ROW -- a 3x4 row is one continuous
10 s trace of four leads, the same shape of input as the rhythm strip --
and its output is cut into cells and scored by `score_12lead.score_sheet`,
the same function that scores the classical tracer.

Per row the canonical crop (strips.to_canonical) is centred on the row's
centre from `locate_sheet` and starts LEAD_IN_MM before recording time
zero, at the calibration under test. Model B's output keeps its presence
gaps (NaN); model A's mask is read by the column centroid.

    bash ml/wsl_run.sh python ml/eval_sheets.py --runs a1,b1
    bash ml/wsl_run.sh python ml/eval_sheets.py --runs a1,b1 --only ptbxl38

Writes results/ml/sheets_<runs>[_<only>].csv + .json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))
sys.path.insert(0, str(REPO / "python"))

import strips as S  # noqa: E402
import run_photo as rp  # noqa: E402
import score_kit_photo as skp  # noqa: E402
import score_12lead as S12  # noqa: E402

RESULTS = REPO / "results" / "ml"
LEAD_IN_MM = 4.0


def make_tracer(model, head: str, height: int):
    import train_tracer as T

    def tracer(sheet, layout, row):
        px_mm = layout.px_per_s / 25.0
        center = float(layout.row_y[row])
        canon, cm = S.to_canonical(sheet, px_mm, layout.t0_x - LEAD_IN_MM * px_mm,
                                   center, height=height)
        out = T.predict_tiled(model, canon[..., ::-1].copy())
        if head == "mask":
            r = T.decode_mask(out["mask"])
            c = np.arange(len(r), dtype=np.float64)
        else:
            r = T.decode_rows(out["rows"], out["presence"])
            c = (np.arange(len(r)) + 0.5) / S.OUT_UPSAMPLE - 0.5
        xs = cm.column_to_source_x(c)
        rows = (r + cm.y0 + 0.5) / cm.k - 0.5
        return xs, rows
    return tracer


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="a1,b1")
    p.add_argument("--only", default="", help="photos whose stem starts with this")
    p.add_argument("--no-clean", action="store_true")
    args = p.parse_args(argv)

    import tensorflow as tf
    import keras
    import train_tracer as T
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    keras.mixed_precision.set_global_policy("mixed_float16")
    tracers = {}
    for run in args.runs.split(","):
        rec = json.loads((RESULTS / f"train_{run}.json").read_text())["params"]
        head = rec["head"]
        h = int(rec.get("canon_h", 384))
        m = T.build_model(head, h=h)
        m.load_weights(REPO / "out" / "ml_runs" / run / "best.weights.h5")
        label = ("model_A_" if head == "mask" else "model_B_") + run
        tracers[label] = make_tracer(m, head, h)

    reg = skp.load_registry()["photos"]
    rows = []
    if not args.no_clean and not args.only:
        for name in sorted({e["target"] for e in reg.values()}):
            t = skp.Target(name)
            rows += S12.score_sheet(t.sheet, t, S12.SheetTruth(t),
                                    {"photo": f"clean:{name}", "target": name,
                                     "ecg_id": t.ecg_id}, extra_tracers=tracers)
    for stem, e in sorted(reg.items()):
        if args.only and not stem.startswith(args.only):
            continue
        t = skp.Target(e["target"])
        photo = rp.read_oriented(skp.PHOTOS / e.get("file", f"{stem}.jpg"))
        sheet = skp.warp_to_sheet(photo, np.asarray(e["corners_px"], np.float32), t)
        rows += S12.score_sheet(sheet, t, S12.SheetTruth(t),
                                {"photo": stem, "target": t.name, "ecg_id": t.ecg_id},
                                extra_tracers=tracers)
    S12.summarise(rows)
    S12.RESULTS = RESULTS
    name = "sheets_" + args.runs.replace(",", "_") + (f"_{args.only}" if args.only else "")
    print(S12.persist(rows, {"runs": args.runs, "only": args.only,
                             "lead_in_mm": LEAD_IN_MM,
                             "corners": "python/kit_photos.json as placed",
                             "tensorflow": tf.__version__}, name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
