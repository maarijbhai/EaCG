#!/usr/bin/env python3
"""Bundle exported tracer models + one golden tile for the on-device benchmark.

The phone's job is to answer two questions about each exported model: how
long a tile takes on the A30s (and how much memory), and whether the phone
computes the SAME answer as the desktop. The second needs a golden pair: a
real canonical tile and what each .tflite variant outputs for it here.

The tile is cut from a real photograph (ptbxl40_20260929_05, row 1, the
first 1,024 canonical columns) through exactly the desktop path
(`eval_sheets`: warp with the registered corners, locate_sheet, tick scale,
strips.to_canonical at the run's window height), so the benchmark times a
realistic input and the fidelity check exercises real ink, not noise.

Writes into android/app/src/main/assets/ml/:
    <run>_<h>x<w>_<variant>.tflite      copied from out/ml_export/
    tile_<h>x<w>.rgb                    uint8 RGB, row-major, h*w*3 bytes
    tile_<h>x<w>.json                   provenance + expected outputs per variant

    bash ml/wsl_run.sh python ml/make_device_fixture.py --run b3 --variants int8f,fp16
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))
sys.path.insert(0, str(REPO / "python"))

import strips as S  # noqa: E402
import run_photo as rp  # noqa: E402
import score_kit_photo as skp  # noqa: E402
from ecgproto import segment  # noqa: E402

ASSETS = REPO / "android" / "app" / "src" / "main" / "assets" / "ml"
EXPORT = REPO / "out" / "ml_export"
PHOTO = "ptbxl40_20260929_05"
ROW = 1
LEAD_IN_MM = 4.0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="b3")
    p.add_argument("--variants", default="int8f,fp16")
    p.add_argument("--width", type=int, default=1024)
    args = p.parse_args(argv)
    rec = json.loads((REPO / "results" / "ml" / f"train_{args.run}.json").read_text())["params"]
    h, w = int(rec.get("canon_h", 384)), args.width

    e = skp.load_registry()["photos"][PHOTO]
    t = skp.Target(e["target"])
    sheet = skp.warp_to_sheet(rp.read_oriented(skp.PHOTOS / e["file"]),
                              np.asarray(e["corners_px"], np.float32), t)
    sl = segment.locate_sheet(sheet)
    ticks = segment.tick_scale(sheet, sl)
    px = ticks.px_per_mm
    canon, cm = S.to_canonical(sheet, px, sl.t0_x - LEAD_IN_MM * px,
                               float(sl.rows[ROW]), height=h)
    tile = np.ascontiguousarray(canon[:, :w, ::-1])          # BGR -> RGB, uint8

    import tensorflow as tf
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / f"tile_{h}x{w}.rgb").write_bytes(tile.tobytes())
    expected = {}
    for v in args.variants.split(","):
        src = EXPORT / f"{args.run}_{h}x{w}_{v}.tflite"
        shutil.copy(src, ASSETS / src.name)
        it = tf.lite.Interpreter(model_path=str(src), num_threads=4)
        run = it.get_signature_runner()
        name = list(run.get_input_details())[0]
        out = run(**{name: tile[None].astype(np.float32)})
        expected[v] = {"file": src.name,
                       "row": [round(float(x), 4) for x in np.asarray(out["row"])[0]],
                       "presence": [round(float(x), 4) for x in np.asarray(out["presence"])[0]]}
        print(f"{v}: presence>0 {float((np.asarray(out['presence'])[0] > 0).mean()):.3f}",
              flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    (ASSETS / f"tile_{h}x{w}.json").write_text(json.dumps({
        "generated_by": "ml/make_device_fixture.py",
        "git_commit": commit,
        "run": args.run, "height": h, "width": w,
        "source": {"photo": PHOTO, "row": ROW, "calibration": "ticks",
                   "px_per_mm": round(px, 4), "x0_canon": cm.x0, "y0_canon": cm.y0},
        "layout": "uint8 RGB row-major, height x width x 3",
        "expected": expected,
    }), encoding="utf-8")
    print("wrote", ASSETS)
    return 0


if __name__ == "__main__":
    import os
    code = main()
    sys.stdout.flush()
    os._exit(code)
