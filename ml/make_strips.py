#!/usr/bin/env python3
"""Generate the learned tracer's training data: rhythm-strip bands with exact labels.

`docs/ml_plan.md` §3.2: render **clean** sheets with ecg-image-kit, keep the
per-sample truth, and do our own augmentation later with transforms whose
effect on the labels is known. This script is the rendering half. It needs
no deep-learning framework and runs on Windows alongside `python/`.

WHAT COMES OUT, per record, in `data/ml_strips/<split>/`:

    <stem>.png    the lead II rhythm band of the rendered sheet, full sheet
                  width, BAND_MARGIN_MM above the trace's highest excursion
                  and below its lowest (clipped at the sheet edge). Native
                  kit resolution, lossless. The calibration pulse, the lead
                  label and whatever of the row above falls in the margin
                  are kept on purpose: the real crop contains them too.
    <stem>.npz    rows, cols   float32, (5000,): every plotted sample's
                               position in THIS band's pixels (kit gives
                               (row, col); shifted by the frame strip and
                               the band's y0)
                  lead_ii_mv   float32, (5000,): the truth, for scoring
                  fs, px_per_mm, start_sample
    <stem>.json   one manifest row (also gathered into results/ml/)

Stems are `f<fold>_ptbxl<ecg_id>` so a sheet's fold is in its filename and a
split mistake is visible in any directory listing.

WHY THE BAND AND NOT THE SHEET. The model runs on one strip crop
(`ml_plan.md` §2.2), so the rest of the sheet is never input. A 300 dpi
sheet is ~3 MB as PNG; the band is ~10x smaller, and 2,300 of them fit
comfortably outside OneDrive.

SPLITS FOLLOW `strat_fold` AND NOTHING ELSE: folds 1-8 train, 9 val.
Fold 10 is refused outright -- the printed test sheets (PTB-XL 9, 38, 40,
57) are fold 10 and must never be seen in training.

GRID STYLE is drawn per sheet from `STYLES`. Half are the kit's red palette,
which is what the printed test sheets use (their minor rules are recoloured
at training time, `strips.recolour_minor`, since a printer drops the kit's
near-white ones). The rest are the kit's other palettes, its random dark
grids -- rules as dark as the trace, the condition that defeated every
pixel rule on the real photo -- and black and white.

LABEL CHECK. Every sheet records `label_on_ink_frac`: the fraction of the
5,000 labels whose 3x3 neighbourhood holds a pixel within INK_TOLERANCE of
the kit's recorded trace colour. The plan requires this before any label is
trusted; a sheet below MIN_LABEL_ON_INK is kept but flagged, never silently
used.

    python make_strips.py --split train --n 2000 --workers 6
    python make_strips.py --split val   --n 300  --workers 6
    python make_strips.py --split train --n 4 --workers 1     # smoke test

Resumable: a stem whose .json exists is skipped. The per-split manifest CSV
and its provenance sidecar are rebuilt from all .json rows at the end.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from ecgproto import ptbxl  # noqa: E402
import make_kit_target as mkt  # noqa: E402

DATA = REPO / "data" / "ml_strips"
DATA_ROWS = REPO / "data" / "ml_rows"
RESULTS = REPO / "results" / "ml"

SPLIT_FOLDS = {"train": (1, 2, 3, 4, 5, 6, 7, 8), "val": (9,)}
RHYTHM_LEAD = "II"
FS = 500
DPI = 300

#: Context kept above and below the trace's extremes. The canonical crop is
#: 32 mm tall centred on the baseline with a few mm of jitter
#: (`strips.CANON_H`); 16 mm either side of the extremes covers that for any
#: trace, and brings in the row above as a distractor when the sheet has one.
BAND_MARGIN_MM = 16.0

#: (name, probability, extra kit argv). `--standard_grid_color 5` is the
#: kit's red palette, the one printed for the real test sheets.
STYLES = (
    ("red", 0.50, ("--standard_grid_color", "5")),
    ("brown", 0.05, ("--standard_grid_color", "1")),
    ("pink", 0.05, ("--standard_grid_color", "2")),
    ("blue", 0.05, ("--standard_grid_color", "3")),
    ("green", 0.05, ("--standard_grid_color", "4")),
    ("random", 0.20, ("--random_grid_color",)),
    ("bw", 0.10, ("--random_bw", "1")),
)

#: The kit's frame around the sheet, (top, bottom, left, right) px at 300 dpi,
#: as `make_kit_target.fit_to_page` measured it on all four printed targets.
#: Stripped by these fixed counts rather than detected, because detection
#: keys on "dark" pixels and a green, brown, random or black-and-white grid's
#: major rules are dark by that test -- the A4 crop guard refused 3 of 4
#: smoke-test sheets for exactly that reason. No page crop is applied: the
#: band is cut from the full sheet, and the page crop is a printing concern.
KIT_FRAME_PX = (2, 1, 2, 1)

#: Fraction of sheets rendered with the kit's printed header text.
P_HEADER = 0.3

#: Max per-channel distance from the kit's trace colour for a pixel to count
#: as trace. The kit anti-aliases a 0.75 pt line (~3 px at 300 dpi), so the
#: centre pixel of the stroke is at the trace colour to within a few levels.
INK_TOLERANCE = 40.0
MIN_LABEL_ON_INK = 0.995


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--", "ml", "python"],
            cwd=REPO, text=True).strip())
    except Exception:
        return True


def pick_style(rng: np.random.Generator) -> tuple[str, tuple[str, ...]]:
    p = np.array([s[1] for s in STYLES])
    i = int(rng.choice(len(STYLES), p=p / p.sum()))
    return STYLES[i][0], STYLES[i][2]


def rhythm_lead(kit: dict, off: dict) -> tuple[int, np.ndarray, np.ndarray]:
    """(start_sample, rows, cols) of the full-length lead, in cropped-sheet px."""
    full = [l for l in kit["leads"]
            if int(l["end_sample"]) - int(l["start_sample"]) >= 10 * FS - 1]
    if not full:
        raise ValueError("no full-length lead in the kit sidecar")
    lead = full[-1]
    if lead["lead_name"].upper() != RHYTHM_LEAD:
        raise ValueError(f"rhythm lead is {lead['lead_name']}, expected II")
    p = np.asarray(lead["plotted_pixels"], dtype=np.float64)
    # (row, col), not (x, y) -- see make_kit_target.fit_to_page.
    return int(lead["start_sample"]), p[:, 0] - off["y"], p[:, 1] - off["x"]


def sheet_rows(kit: dict, off: dict) -> list[dict]:
    """The sheet's four trace rows, top to bottom, in cropped-sheet px.

    Rows 0-2 are the 3x4 block, each one continuous 10 s trace made of four
    consecutive 2.5 s leads (I aVR V1 V4 / II aVL V2 V5 / III aVF V3 V6 on
    the standard layout); row 3 is the full-length rhythm lead. Leads are
    grouped by their median plotted row, which separates rows by ~36 mm on
    a kit sheet, and each row is checked to tile the 10 s exactly once.
    """
    leads = []
    for l in kit["leads"]:
        p = np.asarray(l["plotted_pixels"], dtype=np.float64)
        leads.append({"name": l["lead_name"], "start": int(l["start_sample"]),
                      "end": int(l["end_sample"]),
                      "rows": p[:, 0] - off["y"], "cols": p[:, 1] - off["x"]})
    full = [l for l in leads if l["end"] - l["start"] >= 10 * FS - 1]
    if len(full) != 1:
        raise ValueError(f"{len(full)} full-length leads, expected 1")
    cells = sorted((l for l in leads if l is not full[0]),
                   key=lambda l: np.median(l["rows"]))
    if len(cells) != 12:
        raise ValueError(f"{len(cells)} short leads, expected 12 (3x4 layout)")
    out = []
    for k in range(3):
        row = sorted(cells[4 * k:4 * k + 4], key=lambda l: l["start"])
        starts = [l["start"] for l in row]
        if starts != [0, 1250, 2500, 3750] or any(
                l["end"] - l["start"] != 1250 for l in row):
            raise ValueError(f"row {k} does not tile 10 s: starts {starts}")
        out.append(row)
    out.append(full)
    return [{"row": k,
             "leads": [(l["name"], l["start"], l["end"]) for l in r],
             "rows": np.concatenate([l["rows"] for l in r]),
             "cols": np.concatenate([l["cols"] for l in r])}
            for k, r in enumerate(out)]


def label_on_ink(band: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                 trace_rgb) -> float:
    """Fraction of labels with trace-coloured ink in their 3x3 neighbourhood."""
    bgr = np.asarray(trace_rgb, dtype=np.float32)[::-1]
    dist = np.abs(band.astype(np.float32) - bgr).max(axis=2)
    ink = cv2.erode((dist > INK_TOLERANCE).astype(np.uint8), np.ones((3, 3), np.uint8)) == 0
    r = np.clip(np.floor(rows + 0.5).astype(int), 0, band.shape[0] - 1)
    c = np.clip(np.floor(cols + 0.5).astype(int), 0, band.shape[1] - 1)
    return float(ink[r, c].mean())


def write_band(sheet: np.ndarray, rows: np.ndarray, cols: np.ndarray,
               mv: np.ndarray, leads: list, kit: dict, px_per_mm: float,
               out_dir: Path, stem: str, base: dict) -> dict:
    """Cut, check and save one band; return its manifest row."""
    h, w = sheet.shape[:2]
    m = BAND_MARGIN_MM * px_per_mm
    want0, want1 = rows.min() - m, rows.max() + m
    y0, y1 = int(max(0, np.floor(want0))), int(min(h, np.ceil(want1)))
    band = sheet[y0:y1]
    rows_b = rows - y0
    on_ink = label_on_ink(band, rows_b, cols, kit["ecg_plot_color"])
    cv2.imwrite(str(out_dir / f"{stem}.png"), band, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    np.savez_compressed(
        out_dir / f"{stem}.npz",
        rows=rows_b.astype(np.float32), cols=cols.astype(np.float32),
        # The truth at every plotted sample, in plotting order: `mv`. The
        # rhythm-only data called it lead_ii_mv; both keys are written.
        mv=mv.astype(np.float32), lead_ii_mv=mv.astype(np.float32),
        lead_names=np.array([l[0] for l in leads]),
        lead_start=np.array([l[1] for l in leads], np.int32),
        lead_end=np.array([l[2] for l in leads], np.int32),
        fs=np.float32(FS), px_per_mm=np.float32(px_per_mm),
        start_sample=np.int32(leads[0][1]))
    row = dict(base)
    row.update({
        "stem": stem,
        "leads": " ".join(l[0] for l in leads),
        "band_y0": y0, "band_h": int(y1 - y0), "band_w": int(w),
        "band_clip_top_mm": round(max(0.0, -want0) / px_per_mm, 2),
        "band_clip_bottom_mm": round(max(0.0, want1 - h) / px_per_mm, 2),
        "t0_col": round(float(cols.min()), 2),
        "t_end_col": round(float(cols.max()), 2),
        "trace_span_mm": round(float(rows.max() - rows.min()) / px_per_mm, 2),
        "baseline_row": round(float(np.median(rows_b)), 2),
        "off_sheet_samples": int(((rows < 0) | (rows > h - 1)).sum()),
        "n_samples": int(len(rows)),
        "mv_min": round(float(mv.min()), 3),
        "mv_max": round(float(mv.max()), 3),
        "grid_major_rgb": ",".join(f"{v:.0f}" for v in kit["grid_line_color_major"]),
        "grid_minor_rgb": ",".join(f"{v:.0f}" for v in kit["grid_line_color_minor"]),
        "trace_rgb": ",".join(f"{v:.0f}" for v in kit["ecg_plot_color"]),
        "kit_bw": bool(kit.get("bw", False)),
        "label_on_ink_frac": round(on_ink, 5),
        "label_ok": bool(on_ink >= MIN_LABEL_ON_INK),
        "error": "",
    })
    return row


def one(job: dict) -> list[dict]:
    """Render one record; save its rhythm band, or (all_rows) all four rows.

    Stems: `<stem>` for the rhythm-only data; `<stem>_row0`..`_row3` with
    all_rows, row 3 being the rhythm strip.
    """
    stem, out_dir = job["stem"], Path(job["out_dir"])
    all_rows = bool(job.get("all_rows"))
    names = [f"{stem}_row{k}" for k in range(4)] if all_rows else [stem]
    done = [out_dir / f"{n}.json" for n in names]
    if all(d.exists() for d in done):
        return [json.loads(d.read_text("utf-8")) for d in done]
    t0 = time.perf_counter()
    rng = np.random.default_rng(job["seed"])
    style, style_argv = pick_style(rng)
    header = bool(rng.random() < P_HEADER)
    # The kit draws from the global `random` (and in places NumPy's global
    # state) and ignores its own --seed; seeding both here is what makes a
    # sheet reproducible from its manifest row.
    random.seed(job["seed"])
    np.random.seed(job["seed"] % (2 ** 32))

    base = {"sheet": stem, "ecg_id": job["ecg_id"], "patient_id": job["patient_id"],
            "strat_fold": job["fold"], "split": job["split"], "seed": job["seed"],
            "style": style, "print_header": header, "dpi": DPI}
    out = []
    try:
        signal, meta = ptbxl.load_record(job["ecg_id"], root=job["root"],
                                         sampling_rate=FS)
        rec_stem = Path(job["root"]) / job["rec_rel"]
        with tempfile.TemporaryDirectory() as tmp:
            png, side = mkt.run_kit(Path(job["kit"]), rec_stem, Path(tmp), DPI,
                                    RHYTHM_LEAD, header, extra_argv=style_argv)
            sheet = cv2.imread(str(png), cv2.IMREAD_COLOR)
            kit = json.loads(Path(side).read_text("utf-8"))
        px_per_mm = DPI / 25.4
        t, b, l, r = KIT_FRAME_PX
        sheet = sheet[t:sheet.shape[0] - b, l:sheet.shape[1] - r]
        rows_all = sheet_rows(kit, {"x": l, "y": t})
        wanted = rows_all if all_rows else [rows_all[3]]
        for name, rw in zip(names, wanted):
            if rw["row"] == 3 and rw["leads"][0][0].upper() != RHYTHM_LEAD:
                raise ValueError(f"rhythm lead is {rw['leads'][0][0]}, expected II")
            mv = np.concatenate([
                signal[s0:s1, ptbxl.LEAD_NAMES.index(nm.upper())]
                for nm, s0, s1 in rw["leads"]])
            if len(mv) != len(rw["rows"]):
                raise ValueError(f"{len(rw['rows'])} plotted samples, {len(mv)} truth")
            out.append(write_band(sheet, rw["rows"], rw["cols"], mv, rw["leads"],
                                  kit, px_per_mm, out_dir, name,
                                  dict(base, row=rw["row"])))
    except Exception as exc:
        out = [dict(base, stem=n, row=(k if all_rows else 3),
                    error=f"{type(exc).__name__}: {exc}")
               for k, n in enumerate(names)]
    el = round(time.perf_counter() - t0, 2)
    for rowd, d in zip(out, done):
        rowd["render_s"] = el
        d.write_text(json.dumps(rowd), encoding="utf-8")
    return out


def jobs_for(split: str, n: int, seed: int, root: Path, kit: Path,
             out_dir: Path, all_rows: bool = False) -> list[dict]:
    folds = SPLIT_FOLDS[split]
    df = ptbxl.load_metadata(root)
    pool = df[df["strat_fold"].isin(folds)].sort_index()
    if (pool["strat_fold"] == 10).any():
        raise AssertionError("fold 10 in a training pool")
    rng = np.random.default_rng(seed)
    ids = rng.choice(pool.index.to_numpy(), size=min(n, len(pool)), replace=False)
    jobs = []
    for ecg_id in sorted(int(i) for i in ids):
        r = pool.loc[ecg_id]
        jobs.append({
            "stem": f"f{int(r['strat_fold'])}_ptbxl{ecg_id}",
            "ecg_id": ecg_id, "patient_id": float(r["patient_id"]),
            "fold": int(r["strat_fold"]), "split": split,
            # One seed per record, independent of n and of worker order.
            "seed": int(seed * 1_000_003 + ecg_id),
            "rec_rel": ptbxl.record_path(r, FS),
            "root": str(root), "kit": str(kit), "out_dir": str(out_dir),
            "all_rows": all_rows,
        })
    return jobs


def write_manifest(split: str, out_dir: Path, params: dict) -> Path:
    rows = [json.loads(p.read_text("utf-8")) for p in sorted(out_dir.glob("*.json"))]
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"strips_{split}.csv"
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    ok = [r for r in rows if not r.get("error")]
    summary = {
        "sheets": len(rows),
        "errors": len(rows) - len(ok),
        "label_ok": sum(bool(r["label_ok"]) for r in ok),
        "label_on_ink_min": min((r["label_on_ink_frac"] for r in ok), default=None),
        "off_sheet_sheets": sum(r["off_sheet_samples"] > 0 for r in ok),
        "styles": {s[0]: sum(r["style"] == s[0] for r in ok) for s in STYLES},
        "folds": {str(f): sum(r["strat_fold"] == f for r in ok)
                  for f in SPLIT_FOLDS[split.removeprefix("rows_")]},
        "patients": len({r["patient_id"] for r in ok}),
        "render_s_median": float(np.median([r["render_s"] for r in rows])) if rows else None,
    }
    (RESULTS / f"strips_{split}.json").write_text(json.dumps({
        "generated_by": "ml/make_strips.py",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "params": params,
        "summary": summary,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "opencv": cv2.__version__},
    }, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=sorted(SPLIT_FOLDS), required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--root", type=Path, default=ptbxl.DEFAULT_ROOT)
    p.add_argument("--kit", type=Path, default=mkt.DEFAULT_KIT)
    p.add_argument("--out", type=Path, default=None,
                   help="default data/ml_strips (rhythm) or data/ml_rows (--all-rows)")
    p.add_argument("--all-rows", action="store_true",
                   help="save all four rows of each sheet (the 12-lead scan), "
                        "not only the rhythm strip")
    args = p.parse_args(argv)
    if args.out is None:
        args.out = DATA_ROWS if args.all_rows else DATA

    out_dir = args.out / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = jobs_for(args.split, args.n, args.seed, args.root, args.kit, out_dir,
                    args.all_rows)
    print(f"{args.split}: {len(jobs)} records from folds {SPLIT_FOLDS[args.split]}"
          f" -> {out_dir}")
    t0 = time.perf_counter()
    done = 0
    if args.workers <= 1:
        it = map(one, jobs)
    else:
        pool = Pool(args.workers, maxtasksperchild=40)
        it = pool.imap_unordered(one, jobs, chunksize=1)
    for rs in it:
        done += 1
        for r in rs:
            if r.get("error"):
                print(f"  {r['stem']} FAILED: {r['error']}", flush=True)
            elif not r["label_ok"]:
                print(f"  {r['stem']} labels on ink {r['label_on_ink_frac']:.4f}",
                      flush=True)
        if done % 50 == 0 or done == len(jobs):
            el = time.perf_counter() - t0
            print(f"  {done}/{len(jobs)}  {el/60:.1f} min", flush=True)
    if args.workers > 1:
        pool.close()
        pool.join()
    params = {"split": args.split, "n": args.n, "seed": args.seed, "dpi": DPI,
              "all_rows": args.all_rows,
              "band_margin_mm": BAND_MARGIN_MM, "styles": [s[:2] for s in STYLES],
              "p_header": P_HEADER, "ink_tolerance": INK_TOLERANCE,
              "kit": str(args.kit)}
    path = write_manifest(("rows_" if args.all_rows else "") + args.split,
                          out_dir, params)
    print(f"manifest {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
