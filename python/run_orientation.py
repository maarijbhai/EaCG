#!/usr/bin/env python3
"""Measure `ecgproto.orientation`'s sideways rule on every sheet image, turned four ways.

Each image is decoded as the phone does (EXIF applied, 3,200 px long side),
then turned by r = 0..3 quarter turns (counter-clockwise) to stand for a
photograph whose orientation tag was wrong; the rule is right when it calls
the odd turns, and only those, sideways.

Images: the kit photographs (data/kit_photos, landscape), the in-app camera
photographs (archive/, portrait shots of a landscape sheet), and the 12
clean printed-sheet images (out/kit_targets).

    python run_orientation.py            (from python/)

Writes results/kitphoto/orientation.csv + .json.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from ecgproto import orientation as O  # noqa: E402

DECODE_PX = 3200
OUT = REPO / "results" / "kitphoto" / "orientation"


def load(path: Path) -> np.ndarray:
    """RGB, EXIF applied, long side DECODE_PX (the phone's ImageSource)."""
    im = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
    sc = DECODE_PX / max(im.shape[:2])
    if sc < 1:
        im = cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(im)


def images() -> list[tuple[str, Path]]:
    out = [("kit_photo", p) for p in sorted((REPO / "data" / "kit_photos").glob("ptbxl*.jpg"))]
    out += [("app_camera", p) for p in sorted((REPO / "archive").glob("eacg_*.jpg"))]
    out += [("clean", p) for p in sorted((REPO / "out" / "kit_targets").glob("ptbxl*/ptbxl*_kit_500hz_300dpi.png"))]
    return out


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def main() -> int:
    from ecgproto import page
    rows = []
    for kind, path in images():
        base = load(path)
        for r in range(4):
            img = np.ascontiguousarray(np.rot90(base, r))
            t0 = time.time()
            row = {"image": path.name, "kind": kind, "turned": r, "should_be_sideways": r % 2 == 1}
            try:
                c = page.find_sheet_corners(img, refine=False).corners
                asp = O.outline_aspect(c)
                row.update(aspect=round(asp, 3), sideways=O.is_sideways(c),
                           correct=O.is_sideways(c) == (r % 2 == 1), error="")
            except Exception as e:                       # noqa: BLE001 -- recorded, not hidden
                row.update(aspect="", sideways="", correct="", error=f"{type(e).__name__}: {e}")
            row["seconds"] = round(time.time() - t0, 2)
            rows.append(row)
            print(row, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    ok = [r for r in rows if not r["error"]]
    up = [r["aspect"] for r in ok if not r["should_be_sideways"]]
    sd = [r["aspect"] for r in ok if r["should_be_sideways"]]
    summary = {"correct": f"{sum(r['correct'] for r in ok)}/{len(ok)}",
               "corner_failures": sum(1 for r in rows if r["error"]),
               "aspect_upright": [min(up), max(up)], "aspect_sideways": [min(sd), max(sd)]}
    OUT.with_suffix(".json").write_text(json.dumps({
        "generated_by": "python/run_orientation.py",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--", "python")),
        "params": {"decode_px": DECODE_PX, "sideways_below": O.SIDEWAYS_BELOW,
                   "note": "clean PNGs are page-only (no margin): the corner finder needs paper around the "
                           "grid, so they fail here; on the phone the clean images had a margin"},
        "summary": summary,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__},
    }, indent=2) + "\n", encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
