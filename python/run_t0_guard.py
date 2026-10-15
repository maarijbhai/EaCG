#!/usr/bin/env python3
"""The guarded time-zero fallback (segment.guard_time_zero), replayed on every phone run.

Every learned export of Set A, Set B and Set C (Tilt, Shadow, Crumple; folders
starting '_' skipped -- one export per run: the classical partner of a
both-tracers run has the same photograph and corners) is paired with its
photograph from the records (Set A by record, Set B python/test_set.csv, Set C
the folders' READMEs), decoded as the phone decodes it (EXIF
applied, 3,200 px long side), turned by its photo_rotation, and the desktop
reference is run with the export's own corners up to the tick search, so the
guard's decision is the reference's on the phone's input. The phone's own tick
and grid scales (header) are recorded beside it.

    python run_t0_guard.py                  -> results/kitphoto/t0_guard_replay.csv + .json
    python run_t0_guard.py --whatif [--tracer classical|b3]
        the runs where the guard fires, digitised before (t0_guard=False) and
        after, every lead scored against its truth (score_12lead.score_lead)
        -> results/kitphoto/t0_guard_whatif[_b3].csv + .json
        (b3 needs TensorFlow: run in WSL, ml/wsl_run.sh)

Nothing here writes results/device/sheet_scores.csv or results/report/.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import score_12lead as S12  # noqa: E402
import score_kit_photo as skp  # noqa: E402
from ecgproto import gridfit, page, segment, sheet  # noqa: E402
from run_orientation import load  # noqa: E402

EXPORTS = REPO / "data" / "device_exports"
PHOTOS = REPO / "data" / "kit_photos"
SETS = {"Set A": "SetA_Clean", "Set B": "setB", "Set C Tilt": "SetC_Tilt",
        "Set C Shadow": "SetC_Shadow", "Set C Crumple": "SetC_Crumple"}
OUT = REPO / "results" / "kitphoto"


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def header(path: Path) -> dict:
    h = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        k, _, v = line[1:].partition(":")
        h[k.strip()] = v.strip()
    return h


def runs() -> list[tuple[str, Path, dict]]:
    out = []
    for s in SETS:
        for p in sorted((EXPORTS / s).rglob("eacg_sheet_*_model_*.csv")):
            if any(part.startswith("_") for part in p.relative_to(EXPORTS / s).parts[:-1]):
                continue
            out.append((s, p, header(p)))
    return out


_photo_cache: dict = {}


def photo_bgr(path: Path, cw: int) -> np.ndarray:
    key = (path, cw)
    if key not in _photo_cache:
        _photo_cache.clear()                     # one photograph in memory at a time
        _photo_cache[key] = np.ascontiguousarray(np.rot90(load(path), -cw)[:, :, ::-1])
    return _photo_cache[key]


def photo_map() -> dict:
    """export file name or pair_id -> photograph, from the authoritative records:
    Set A by record (clean_ptbxl<id>.png), Set B from python/test_set.csv's
    photo_path (progress_log §36), Set C from each folder's README table."""
    import pandas as pd
    out = {}
    m = pd.read_csv(HERE / "test_set.csv")
    for _, r in m.iterrows():
        if isinstance(r["photo_path"], str) and r["photo_path"]:
            out[r["export"]] = REPO / r["photo_path"]
    for s in ("Set C Tilt", "Set C Shadow", "Set C Crumple"):
        for line in (EXPORTS / s / "README.txt").read_text(encoding="utf-8").splitlines():
            pid = re.match(r"\s*(\d{8}_\d{6})\s", line)
            ph = re.search(r"(eacg_\d{8}_\d{6}(?:_\d+)?\.jpg)", line)
            if pid and ph:
                found = list((PHOTOS / SETS[s]).rglob(ph.group(1)))
                if found:
                    out[pid.group(1)] = found[0]
    return out


_MAP: dict = {}


def match_photo(set_name: str, path: Path, h: dict) -> tuple[Path | None, int, float]:
    """(photograph, clockwise quarters, 2nd-smallest corner distance px -- a cross-check)."""
    if not _MAP:
        _MAP.update(photo_map())
    rot = h.get("photo_rotation", "")
    cw = int(rot.split()[0]) // 90 % 4 if rot else 0
    if set_name == "Set A":
        photo = PHOTOS / SETS[set_name] / f"clean_{h['target'].split('_')[0]}.png"
    else:
        photo = _MAP.get(path.name) or _MAP.get(h.get("pair_id", "")) or _MAP.get(h.get("local_time", ""))
    if photo is None or not photo.exists():
        return None, cw, float("nan")
    img = photo_bgr(photo, cw)
    c = np.array([float(v) for v in h["corners_px"].split()])
    try:
        pc = page.find_sheet_corners(img).corners.reshape(-1)
        d = float(np.sort(np.hypot(*(pc - c).reshape(4, 2).T))[1])
    except (RuntimeError, ValueError):
        d = float("nan")
    return photo, cw, d


def phone_scales(h: dict) -> tuple[float, float, int]:
    g = re.search(r"grid \(.*?([\d.]+) px/mm", h.get("geometry", ""))
    t = re.match(r"ticks ([\d.]+) px/mm", h.get("calibration", ""))
    n = re.search(r"(\d+)/9 ticks", h.get("calibration", ""))
    return (float(t.group(1)) if t else float("nan"), float(g.group(1)) if g else float("nan"),
            int(n.group(1)) if n else -1)


def to_ticks(img: np.ndarray, corners: np.ndarray) -> dict:
    """digitise_sheet's geometry, locate and tick search (grid), and the guard."""
    rect, _ = sheet.rectify(img, corners)
    fit = gridfit.fit_grid(rect, segment.locate_sheet(rect).anchor_px_per_mm)
    rect, _ = gridfit.regularise(img, corners, rect.shape, fit)
    sl = segment.locate_sheet(rect, fit.px_per_mm)
    tk = segment.tick_scale(rect, sl)
    g = segment.guard_time_zero(rect, sl, tk, fit.px_per_mm)
    return {"grid_px": fit.px_per_mm, "tick_px": tk.px_per_mm, "ticks": tk.n_found, "t0_source": sl.t0_source,
            "fires": g is not None, "guard": g[2] if g is not None else None}


def write(rows: list[dict], name: str, params: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with (OUT / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    (OUT / f"{name}.json").write_text(json.dumps({
        "generated_by": "python/run_t0_guard.py",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "python/ecgproto", "python/run_t0_guard.py")),
        "params": {**params, "t0_guard_pct": segment.T0_GUARD_PCT, "decode_px": 3200,
                   "sets": SETS, "photo_match": "Set A by record; Set B python/test_set.csv; Set C the folders' README tables; match_px = the 2nd-smallest corner distance to the phone's (cross-check)"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__},
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT / f"{name}.csv")


def replay() -> list[dict]:
    rows = []
    for s, p, h in runs():
        photo, cw, d = match_photo(s, p, h)
        tick_ph, grid_ph, n_ph = phone_scales(h)
        row = {"set": s, "export": p.name, "pair_id": h.get("pair_id", ""), "photo": photo.name if photo else "",
               "cw": cw, "match_px": round(d, 2), "phone_ticks": n_ph, "phone_tick_px": tick_ph,
               "phone_grid_px": grid_ph,
               "phone_disagree_pct": round(100 * (tick_ph / grid_ph - 1), 2) if np.isfinite(tick_ph) else "",
               "phone_fires": segment.t0_guard_fires(tick_ph, grid_ph)}
        if photo is None:
            row["error"] = "photograph not in the records"
        else:
            corners = np.array([float(v) for v in h["corners_px"].split()]).reshape(4, 2)
            try:
                r = to_ticks(photo_bgr(photo, cw), corners)
                row.update(desk_ticks=r["ticks"], desk_tick_px=round(r["tick_px"], 4) if np.isfinite(r["tick_px"]) else "",
                           desk_grid_px=round(r["grid_px"], 4),
                           desk_disagree_pct=round(100 * (r["tick_px"] / r["grid_px"] - 1), 2) if np.isfinite(r["tick_px"]) else "",
                           desk_t0_source=r["t0_source"], desk_fires=r["fires"],
                           desk_guard=r["guard"].line() if r["guard"] else "")
            except (RuntimeError, ValueError) as e:
                row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        print({k: row.get(k) for k in ("set", "export", "photo", "match_px", "phone_fires", "desk_fires",
                                       "phone_disagree_pct", "desk_disagree_pct", "error")}, flush=True)
    return rows


#: The knife edge (progress_log §41): the same photograph flips between 8/9
#: and 3/9 ticks with a few pixels of corner shift or a 1.5% resize, and the
#: desktop decode is not the phone's. So each run is also replayed under small
#: perturbations, fixed by seed: decode resized by -1.5 / +1.5%, and six
#: corner jitters of up to +-3 px per coordinate.
JITTER_SEED = 20261016
CONDITIONS = [("exact", 1.0, None), ("resize -1.5%", 0.985, None), ("resize +1.5%", 1.015, None)] +              [(f"corners jitter {i + 1}", 1.0, i) for i in range(6)]


def whatif_targets() -> list[tuple[str, Path, dict, str]]:
    """The runs the guard fires on: by the phone's own scales (the 7 failed
    Set C runs), and those where only the desktop replay fires (from
    t0_guard_replay.csv), labelled so."""
    desk_only = set()
    rp = OUT / "t0_guard_replay.csv"
    if rp.exists():
        for r in csv.DictReader(rp.open(encoding="utf-8")):
            if r.get("desk_fires") == "True" and r.get("phone_fires") == "False":
                desk_only.add(r["export"])
    out = []
    for s, p, h in runs():
        tick_ph, grid_ph, _ = phone_scales(h)
        if segment.t0_guard_fires(tick_ph, grid_ph):
            out.append((s, p, h, "phone fires"))
        elif p.name in desk_only:
            out.append((s, p, h, "desktop only"))
    return out


def whatif(tracer_name: str) -> tuple[list[dict], dict]:
    if tracer_name == "b3":
        sys.path.insert(0, str(REPO / "ml"))
        from export_tflite import TfliteTracer
        tfl = TfliteTracer(REPO / "android" / "app" / "src" / "main" / "assets" / "ml" / "b3_512x1024_fp16.tflite")
        height = int(tfl.run.get_input_details()[tfl.in_name]["shape"][1])
        tracer = sheet.model_row_tracer(tfl, height)
    else:
        tracer = sheet.classical_row_tracer()
    rng = np.random.default_rng(JITTER_SEED)
    jitters = [rng.uniform(-3.0, 3.0, size=(4, 2)) for _ in range(6)]
    rows, summary = [], {}
    for s, p, h, why in whatif_targets():
        photo, cw, _ = match_photo(s, p, h)
        if photo is None:
            rows.append({"set": s, "export": p.name, "error": "photograph not in the records"}); continue
        corners0 = np.array([float(v) for v in h["corners_px"].split()]).reshape(4, 2)
        truth = S12.SheetTruth(skp.Target(h["target"]))
        base = photo_bgr(photo, cw)
        per = []
        for cname, scale, j in CONDITIONS:
            img = base if scale == 1.0 else cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            corners = corners0 * scale + (jitters[j] if j is not None else 0.0)
            pair = {}
            for when, on in (("before", False), ("after", True)):
                try:
                    res = sheet.digitise_sheet(img, tracer, corners, "grid", t0_guard=on)
                except (RuntimeError, ValueError) as e:
                    rows.append({"set": s, "export": p.name, "why": why, "condition": cname, "when": when,
                                 "error": f"{type(e).__name__}: {e}"})
                    continue
                cells, rh = [], {}
                for (lead, rhythm), sig in res.leads.items():
                    sc = S12.score_lead(sig, truth.leads[(lead, rhythm)]["mv"],
                                        segment.RECORDING_S if rhythm else segment.CELL_S)
                    if rhythm:
                        rh = sc
                    else:
                        cells.append(sc)
                val = lambda c, k: float(c[k]) if c.get(k, "") != "" else np.nan
                pcc = np.array([val(c, "pcc_asis") for c in cells])
                med = lambda k: float(np.nanmedian([val(c, k) for c in cells]))
                grid_px = res.grid_fit.px_per_mm if res.grid_fit is not None else float("nan")
                tick_px = res.ticks.px_per_mm if res.ticks is not None else float("nan")
                row = {"set": s, "export": p.name, "photo": photo.name, "record": h["target"].split("_")[0],
                       "why": why, "tracer": tracer_name, "condition": cname, "when": when,
                       "geometry": res.geometry, "ticks_found": res.ticks_found, "t0_source": res.t0_source,
                       "calibration": res.calibration, "px_per_mm": round(res.px_per_mm, 4),
                       "grid_px": round(grid_px, 4),
                       "disagree_pct": round(100 * (tick_px / grid_px - 1), 2) if np.isfinite(tick_px) else "",
                       "fires": segment.t0_guard_fires(tick_px, grid_px) if when == "before" else "",
                       "guard": res.t0_guard.line() if res.t0_guard else "",
                       "median_pcc": round(float(np.nanmedian(pcc)), 4),
                       "median_pcc_aligned": round(med("pcc_aligned"), 4),
                       "median_snr_db": round(med("snr_db_asis"), 2),
                       "median_shifted_snr_db": round(med("shifted_snr_db_asis"), 2),
                       "leads_pcc_ge_0.9": int(np.sum(pcc >= 0.9)),
                       "rhythm_pcc": rh.get("pcc_asis", "")}
                rows.append(row)
                pair[when] = row
            if len(pair) == 2:
                per.append(pair)
            print(p.name[-19:-4], cname, {k: (v["ticks_found"], v["disagree_pct"], v["median_pcc"]) for k, v in pair.items()},
                  flush=True)
        fired = [x for x in per if x["before"]["fires"]]
        summary[p.name] = {
            "why": why, "photo": photo.name, "conditions": len(per), "fires_before": len(fired),
            "guard_used": sum(1 for x in fired if x["after"]["guard"].startswith("fired, used")),
            "median_pcc_before_all": round(float(np.median([x["before"]["median_pcc"] for x in per])), 4) if per else None,
            "median_pcc_after_all": round(float(np.median([x["after"]["median_pcc"] for x in per])), 4) if per else None,
            "median_pcc_before_fired": round(float(np.median([x["before"]["median_pcc"] for x in fired])), 4) if fired else None,
            "median_pcc_after_fired": round(float(np.median([x["after"]["median_pcc"] for x in fired])), 4) if fired else None,
            "leads_ge_0.9_before_fired": int(np.sum([x["before"]["leads_pcc_ge_0.9"] for x in fired])) if fired else None,
            "leads_ge_0.9_after_fired": int(np.sum([x["after"]["leads_pcc_ge_0.9"] for x in fired])) if fired else None,
            "unchanged_where_not_fired": all(x["before"]["median_pcc"] == x["after"]["median_pcc"]
                                             for x in per if not x["before"]["fires"])}
        print(p.name, summary[p.name], flush=True)
    return rows, summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--whatif", action="store_true")
    ap.add_argument("--tracer", choices=("classical", "b3"), default="classical")
    a = ap.parse_args(argv)
    if a.whatif:
        rows, summary = whatif(a.tracer)
        write(rows, "t0_guard_whatif" + ("_b3" if a.tracer == "b3" else ""),
              {"tracer": a.tracer, "conditions": [c[0] for c in CONDITIONS], "jitter_seed": JITTER_SEED,
               "in_sample": "IN-SAMPLE: the rule was chosen after seeing these failures", "summary": summary})
    else:
        write(replay(), "t0_guard_replay", {})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
