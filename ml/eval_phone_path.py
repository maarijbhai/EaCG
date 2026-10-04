#!/usr/bin/env python3
"""The phone's full-sheet path, run on the desktop, scored lead by lead.

`ecgproto.sheet.digitise_sheet` end to end -- automatic corners, rectify
from the corners alone, locate, separator-tick calibration, segmentation,
then the row tracer -- with the learned tracer being the EXACT LiteRT file
the app bundles (`android/app/src/main/assets/ml/<model>.tflite`), run tile
by tile through `tf.lite.Interpreter`. Classical rows for comparison.
Every lead is scored against its own PTB-XL window (score_12lead.score_lead).

This is the accuracy the phone should reproduce; the Kotlin port is
golden-vector tested against the same functions.

    bash ml/wsl_run.sh python ml/eval_phone_path.py --model b3_512x1024_fp16

Writes results/ml/phone_path_<model>.csv + .json.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))
sys.path.insert(0, str(REPO / "python"))

import run_photo as rp  # noqa: E402
import score_kit_photo as skp  # noqa: E402
import score_12lead as S12  # noqa: E402
from ecgproto import sheet  # noqa: E402

ASSETS = REPO / "android" / "app" / "src" / "main" / "assets" / "ml"
RESULTS = REPO / "results" / "ml"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="b3_512x1024_fp16")
    p.add_argument("--corners", choices=("auto", "registered"), default="auto")
    p.add_argument("--geometry", choices=("corners", "grid"), default="corners")
    # The phone decodes every photograph to 3,200 px on its long side
    # (ImageSource.kt); 0 keeps the file's full resolution.
    p.add_argument("--decode-px", type=int, default=3200)
    args = p.parse_args(argv)
    from export_tflite import TfliteTracer
    tfl = TfliteTracer(ASSETS / f"{args.model}.tflite")
    height = int(tfl.run.get_input_details()[tfl.in_name]["shape"][1])
    tracers = {"classical": sheet.classical_row_tracer(),
               f"model_{args.model}": sheet.model_row_tracer(tfl, height)}

    reg = skp.load_registry()["photos"]
    rows = []
    for stem, e in sorted(reg.items()):
        t = skp.Target(e["target"])
        truth = S12.SheetTruth(t)
        photo = rp.read_oriented(skp.PHOTOS / e.get("file", f"{stem}.jpg"))
        sc = 1.0
        if args.decode_px and max(photo.shape[:2]) > args.decode_px:
            sc = args.decode_px / max(photo.shape[:2])
            photo = cv2.resize(photo, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        corners = (None if args.corners == "auto"
                   else np.asarray(e["corners_px"], np.float32) * sc)
        for name, tr in tracers.items():
            t0 = time.perf_counter()
            try:
                res = sheet.digitise_sheet(photo, tr, corners, args.geometry)
            except (RuntimeError, ValueError) as exc:
                rows.append({"photo": stem, "tracer": name, "error": str(exc)})
                continue
            el = time.perf_counter() - t0
            head = {"photo": stem, "target": t.name, "ecg_id": t.ecg_id, "tracer": name,
                    "corners_source": args.corners, "corners_refined": res.corners_refined,
                    "geometry": res.geometry, "decode_px": args.decode_px,
                    "grid_fit_rms_mm": round(res.grid_fit.rms_mm, 3) if res.grid_fit else "",
                    "rect": f"{res.rect_shape[1]}x{res.rect_shape[0]}",
                    "calibration": res.calibration, "ticks_found": res.ticks_found,
                    "px_per_mm": round(res.px_per_mm, 4), "t0_source": res.t0_source,
                    "desktop_s": round(el, 1)}
            for (lead, rhythm), sig in res.leads.items():
                tr_ = truth.leads[(lead, rhythm)]
                dur = 10.0 if rhythm else 2.5
                rows.append({**head, "lead": lead, "rhythm": rhythm,
                             **S12.score_lead(sig, tr_["mv"], dur)})
            done = [r for r in rows if r.get("photo") == stem and r.get("tracer") == name
                    and not r.get("rhythm") and r.get("pcc_asis") not in (None, "")]
            print(f"{stem} {name:28s} {res.calibration} {res.ticks_found}/9 ticks  "
                  f"lead PCC as-is median {np.median([r['pcc_asis'] for r in done]):.3f}  "
                  f"SNR {np.median([r['snr_db_asis'] for r in done]):.1f} / shifted "
                  f"{np.median([r['shifted_snr_db_asis'] for r in done]):.1f} dB",
                  flush=True)
    S12.RESULTS = RESULTS
    print(S12.persist(rows, {"model": args.model, "corners": args.corners,
                             "geometry": args.geometry, "decode_px": args.decode_px,
                             "path": "ecgproto.sheet.digitise_sheet"},
                      f"phone_path_{args.model}"
                      + ("" if args.corners == "auto" else f"_{args.corners}")
                      + ("" if args.geometry == "corners" else f"_{args.geometry}")))
    return 0


if __name__ == "__main__":
    import os
    code = main()
    sys.stdout.flush()
    os._exit(code)
