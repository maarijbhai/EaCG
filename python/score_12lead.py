#!/usr/bin/env python3
"""Score the WHOLE sheet: all 12 leads and the rhythm strip, against the truth.

The project goal is the full 3x4 + rhythm scan. This is its desktop scorer,
the counterpart of `score_kit_photo.py` (rhythm strip only), classical path
only -- `ml/eval_sheets.py` runs the learned tracers through the same
`score_lead` so every tracer is scored by one function.

Per sheet:

    photo -> registered corners -> warp onto the printed sheet's pixel grid
    -> FR4 from the IMAGE: segment.locate_sheet (rows by layout spacing, t0
       from the four rows' pulse tops, the duration anchor) then
       segment.segment_leads (13 cells)
    -> classical FR5/FR6 per row (segment.trace_row), cut into cells
    -> each lead scored against its own PTB-XL window

Segmentation is checked against where the kit really drew each lead
(`cell_truth_inside`: fraction of that lead's plotted samples inside the
cell; `cell_t0_error_ms`: the cell's start against the lead's first
sample). Leads are scored under two calibrations: `known` (the sheet's
true px/mm, isolating segmentation and tracing) and `duration` (the 10 s
anchor, the phone's own calibration).

A 2.5 s cell cannot meet the rhythm strip's 8.5 s overlap rule, so the
same rule is applied proportionally: an aligned figure is reported only if
the alignment spans >= MIN_OVERLAP_FRACTION of the lead's window (0.85,
which is 8.5 of 10 s for the rhythm strip and 2.125 s for a cell).

    python score_12lead.py                 # every registered photo + clean sheets
    python score_12lead.py --clean-only    # the printed PNGs, no camera

Writes results/kitphoto/leads12.csv + .json.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_photo as rp  # noqa: E402
import score_kit_photo as skp  # noqa: E402
from ecgproto import digitise as dig, evaluate, ptbxl, segment, sheet  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "kitphoto"
MIN_OVERLAP_FRACTION = rp.MIN_OVERLAP_S / segment.RECORDING_S
FS = 500.0

#: Largest lag the alignment of a 2.5 s CELL may use. Unbounded, the search
#: slid cells of the CLEAN printed sheet by one heartbeat (-844 to -1012 ms,
#: about one RR interval): a periodic waveform matches its neighbouring beat,
#: and on a short window that wins. 100 ms is several times any segmentation
#: or corner error measured (cell starts within 3 ms on the clean sheet) and
#: under half a beat at any heart rate up to 300 bpm. The 10 s rhythm strip
#: keeps run_photo.best_alignment unchanged, so its figures stay comparable
#: with results/kitphoto and results/ml.
CELL_MAX_LAG_S = 0.1


def best_alignment_bounded(ref: np.ndarray, test: np.ndarray, max_lag: int):
    """run_photo.best_alignment's scale search with |lag| <= max_lag."""
    ref, test = rp.fill_nan(ref), rp.fill_nan(test)
    best = (-1.0, 1.0, 0, 0, False)
    scales = np.arange(1.0 - rp.ALIGN_HALF_RANGE,
                       1.0 + rp.ALIGN_HALF_RANGE + rp.ALIGN_STEP / 2, rp.ALIGN_STEP)
    for sc in scales:
        n = int(round(len(test) / sc))
        res = np.interp(np.arange(n) * sc, np.arange(len(test)), test)
        for lag in range(-max_lag, max_lag + 1):
            a, b = (ref[lag:], res) if lag >= 0 else (ref, res[-lag:])
            k = min(len(a), len(b))
            if k < 200:
                continue
            x, y = a[:k], b[:k]
            sd = x.std() * y.std()
            if sd <= 0:
                continue
            c = float(np.mean((x - x.mean()) * (y - y.mean())) / sd)
            if c > best[0]:
                best = (c, float(sc), lag, k, abs(lag) == max_lag)
    return best


class SheetTruth:
    """Every lead the kit drew on a target, in its sheet's pixels."""

    def __init__(self, target: skp.Target):
        d = skp.TARGETS / target.name
        kit = json.loads((d / f"{target.name}_kit.json").read_text("utf-8"))
        off = target.manifest["page"]["plotted_pixels_offset_px"]
        self.leads = {}
        for l in kit["leads"]:
            p = np.asarray(l["plotted_pixels"], np.float64)
            s0, s1 = int(l["start_sample"]), int(l["end_sample"])
            rhythm = (s1 - s0) >= int(segment.RECORDING_S * FS) - 1
            name = l["lead_name"]
            self.leads[(name, rhythm)] = {
                "rows": p[:, 0] - off["y"], "cols": p[:, 1] - off["x"],
                "mv": target.truth[s0:s1, ptbxl.LEAD_NAMES.index(name.upper())],
                "start": s0, "end": s1}


# Moved to ecgproto.sheet (the phone path's reference); re-exported.
on_sheet_columns = sheet.on_sheet_columns
cell_signal = sheet.cell_signal


def score_lead(sig: np.ndarray, truth: np.ndarray, duration_s: float) -> dict:
    """One scoring path for every tracer, every lead."""
    ok = np.isfinite(sig)
    out = {"coverage": round(float(ok.mean()), 4)}
    if ok.sum() < 50:
        return out
    sig_f = rp.fill_nan(sig)
    m = evaluate.evaluate_all(truth, sig_f, FS)
    # SNR on median-removed signals: cell_signal returns mV about the cell's
    # own median, because a printed trace carries no absolute baseline, so
    # the truth's DC offset would otherwise count as error that no
    # digitiser could remove. Shifted SNR is the Challenge 2024 convention
    # (+-50 ms, evaluate_all's default).
    t0, s0 = truth - np.median(truth), sig_f - np.median(sig_f)
    shifted, _ = evaluate.shifted_snr_db(t0, s0, int(round(0.05 * FS)))
    out.update({"pcc_asis": round(float(m.pcc), 4),
                "snr_db_asis": round(float(evaluate.snr_db(t0, s0)), 3),
                "shifted_snr_db_asis": round(float(shifted), 3),
                "rmse_mv_asis": round(float(m.rmse_mv), 4),
                "gain_ratio": round(float(m.gain_ratio), 4),
                "r_ratio": round(evaluate.r_peak_ratio(truth, sig_f, FS)[0], 4)})
    if duration_s < segment.RECORDING_S:
        pcc_a, scale, lag, n, at_bound = best_alignment_bounded(
            truth, sig_f, int(round(CELL_MAX_LAG_S * FS)))
        out["align_lag_at_bound"] = at_bound
    else:
        pcc_a, scale, lag, n, _ = rp.best_alignment(truth, sig_f)
    overlap = n / FS
    out["align_overlap_s"] = round(overlap, 3)
    out["align_scale_pct"] = round(100 * (scale - 1), 3)
    out["align_lag_ms"] = round(1000 * lag / FS, 1)
    scorable = overlap >= MIN_OVERLAP_FRACTION * duration_s
    out["pcc_aligned"] = round(float(pcc_a), 4) if scorable else ""
    if scorable:
        res = np.interp(np.arange(int(round(len(sig_f) / scale))) * scale,
                        np.arange(len(sig_f)), sig_f)
        a, b = (truth[lag:], res) if lag >= 0 else (truth, res[-lag:])
        k = min(len(a), len(b))
        out["r_ratio_aligned"] = round(evaluate.r_peak_ratio(a[:k], b[:k], FS)[0], 4)
    else:
        out["r_ratio_aligned"] = ""
    return out


def check_cell(cell: segment.LeadCell, tr: dict, px_per_s: float) -> dict:
    inside = ((tr["cols"] >= cell.x0) & (tr["cols"] < cell.x1)
              & (tr["rows"] >= cell.y0) & (tr["rows"] < cell.y1))
    return {"cell_truth_inside": round(float(inside.mean()), 4),
            "cell_t0_error_ms": round(1000 * (cell.x0 - tr["cols"][0]) / px_per_s, 1),
            "cell_box": f"{cell.x0},{cell.y0},{cell.x1},{cell.y1}"}


def score_sheet(sheet: np.ndarray, target: skp.Target, truth: SheetTruth,
                head: dict, open_kernel: int = 2, extra_tracers=None) -> list[dict]:
    """All leads of one sheet image already in the target's pixel frame.

    `extra_tracers`: optional {name: fn(sheet, layout, row) -> (xs, rows)}
    for the learned tracers, scored through the same path. `xs` are sheet
    columns, any spacing (a model's output columns do not fall on whole
    sheet pixels); `rows` are sheet rows, NaN where the tracer saw nothing.
    """
    rows = []
    try:
        sl = segment.locate_sheet(sheet)
    except RuntimeError as exc:
        return [dict(head, error=f"locate: {exc}")]
    loc = sl.as_strip_location()
    anchor = sl.anchor_px_per_mm
    head = dict(head, t0_source=sl.t0_source,
                t0_rows_found=int(np.isfinite(sl.t0_per_row).sum()),
                rows_found=" ".join(map(str, sl.rows)))
    ticks = segment.tick_scale(sheet, sl)
    head = dict(head, ticks_found=ticks.n_found,
                ticks_spread_pct=round(ticks.spread_pct, 3) if ticks.n_found else "")
    cals = {"known": target.px_per_mm, "duration": anchor}
    if np.isfinite(ticks.px_per_mm):
        cals["ticks"] = ticks.px_per_mm
    def classical(sh, lay, r):
        pos, x0 = segment.trace_row(sh, lay, r, open_kernel)
        return x0 + np.arange(len(pos), dtype=np.float64), pos
    tracers = {"classical": classical}
    tracers.update(extra_tracers or {})
    for cal, px in cals.items():
        layout = segment.segment_leads(sheet, loc, px,
                                       ticks if cal == "ticks" else None)
        for tname, fn in tracers.items():
            row_traces = {r: fn(sheet, layout, r) for r in range(4)}
            for cell in layout.cells:
                tr = truth.leads[(cell.name, cell.row == 3)]
                pos, x0 = on_sheet_columns(*row_traces[cell.row])
                sig = cell_signal(pos, x0, cell, px, px, layout)
                r = dict(head, calibration=cal, px_per_mm_used=round(px, 4),
                         px_per_mm_error_pct=round(100 * (px / target.px_per_mm - 1), 3),
                         tracer=tname, lead=cell.name,
                         rhythm=cell.row == 3, row=cell.row, col=cell.col,
                         **check_cell(cell, tr, layout.px_per_s))
                r.update(score_lead(sig, tr["mv"], cell.duration_s))
                rows.append(r)
    return rows


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True,
                          text=True).stdout.strip()


def persist(rows: list[dict], params: dict, name: str = "leads12") -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    out = RESULTS / f"{name}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    out.with_suffix(".json").write_text(json.dumps({
        "generated_by": "python/score_12lead.py",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "python", "ml")),
        "params": dict(params, min_overlap_fraction=MIN_OVERLAP_FRACTION,
                       cell_max_lag_s=CELL_MAX_LAG_S,
                       layout=[list(r) for r in segment.LAYOUT_3X4],
                       separator_guard_mm=segment.SEPARATOR_GUARD_MM),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "opencv": cv2.__version__},
    }, indent=2), encoding="utf-8")
    return out


def summarise(rows: list[dict]) -> None:
    import collections
    g = collections.defaultdict(list)
    for r in rows:
        if r.get("error"):
            print(f"  {r.get('photo')}: {r['error']}")
            continue
        g[(r["photo"], r["calibration"], r["tracer"])].append(r)
    for (ph, cal, tr), rs in g.items():
        cells = [r for r in rs if not r["rhythm"]]
        f = lambda k: [float(r[k]) for r in cells if r.get(k, "") != ""]
        pa, ra, ins = f("pcc_aligned"), f("r_ratio_aligned"), f("cell_truth_inside")
        sn, ss = f("snr_db_asis"), f("shifted_snr_db_asis")
        med = lambda v: np.median(v) if v else float("nan")
        print(f"  {ph:22s} {cal:8s} {tr:10s} cells located {sum(v >= 0.95 for v in ins)}/12"
              f"  scorable {len(pa)}/12  PCCa median {med(pa):.3f}  R {med(ra):.2f}"
              f"  SNR {med(sn):.1f} / shifted {med(ss):.1f} dB")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clean-only", action="store_true")
    # 2: the photo path's opening (PhotoPipeline.OPEN_KERNEL); 3x3 erased the
    # 2-3 px photo trace into dots (lead PCC 0.25-0.46 against 0.43-0.95).
    p.add_argument("--open-kernel", type=int, default=2)
    args = p.parse_args(argv)
    reg = skp.load_registry()["photos"]
    rows = []
    targets = sorted({e["target"] for e in reg.values()})
    for name in targets:
        t = skp.Target(name)
        head = {"photo": f"clean:{name}", "target": name, "ecg_id": t.ecg_id}
        rows += score_sheet(t.sheet, t, SheetTruth(t), head, args.open_kernel)
    if not args.clean_only:
        for stem, e in sorted(reg.items()):
            t = skp.Target(e["target"])
            photo = rp.read_oriented(skp.PHOTOS / e.get("file", f"{stem}.jpg"))
            sheet = skp.warp_to_sheet(photo, np.asarray(e["corners_px"], np.float32), t)
            head = {"photo": stem, "target": t.name, "ecg_id": t.ecg_id}
            rows += score_sheet(sheet, t, SheetTruth(t), head, args.open_kernel)
    summarise(rows)
    print(persist(rows, {"clean_only": args.clean_only, "open_kernel": args.open_kernel,
                         "corners": "python/kit_photos.json as placed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
