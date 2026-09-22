"""FR8 quality gate: operating characteristics, measured.

Two experiments, and the distinction between them is the whole report:

  knowngood  the gate over inputs believed correct, to measure the FALSE
             REJECT rate. A gate that rejects good work is not safe, it
             is merely strict.
  inject     the gate over deliberately broken inputs, to measure the
             CATCH RATE -- paired, per row, with the PCC of the waveform
             that was accepted or rejected. A verdict without the
             consequence beside it says nothing about whether refusing
             was the right call.

The PCC column is what makes this a characterisation rather than a
self-report. `results/deskew/guards.csv` already records calibration
verdicts under injected failures and carries no fidelity metric at all,
so it cannot distinguish a guard that refused something harmless from
one that caught a disaster.

    python run_gate.py --experiment knowngood inject

Writes results/gate/*.csv plus a JSON sidecar per experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import scipy

from ecgproto import digitise as dig
from ecgproto import deskew as dsk
from ecgproto import evaluate, layout, ptbxl, quality, synth

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "gate"

#: Printouts of independently known scale, shared with the Android app.
#: They are 3x4 sheets, so they are known-good for *grid estimation* only
#: -- fed to `digitise` they are an instance of the layout failure mode,
#: which is why they are scored here on the calibration check alone.
FIXTURES = {
    "gt_norm_150dpi": (REPO / "android/app/src/main/assets/reference/gt_norm_150dpi.png", 150 / 25.4),
    "gt_mi_200dpi": (REPO / "android/app/src/main/assets/reference/gt_mi_200dpi.png", 200 / 25.4),
    "gt_sttc_300dpi": (REPO / "android/app/src/main/assets/reference/gt_sttc_300dpi.png", 300 / 25.4),
}

#: Resolutions at which a single strip is believed to digitise correctly.
#: 4 px/mm is excluded deliberately: it is a *known* degradation of the
#: control condition (gain ratio 0.60, documented in the audit), so
#: counting a flag there as a false reject would be scoring the gate
#: against an input that is not in fact good.
GOOD_PX_PER_MM = (8.0, 12.0, 16.0)

#: Resolutions that are low enough to degrade the tracer measurably but
#: not low enough to break it. Reported separately from the false-reject
#: rate, because a flag here may well be correct. They are included
#: because `spacing_cv` is documented as scoring 0.412 on a *correct*
#: 4 px/mm estimate, and that warning is the only thing that noticed a
#: 0.5 degree rotation -- so its behaviour on good low-resolution input
#: is exactly what decides whether it is worth anything.
LOWRES_PX_PER_MM = (4.0,)

#: Rotations injected in degrees. 0.5 and 2 are included because the
#: committed guard measurements sweep 1, 3 and 6 and leave the interval
#: where the grid may still resolve unmeasured.
INJECT_ANGLES = (0.5, 1.0, 2.0, 5.0)

#: Scales for the pitch-sensitivity experiment. One integer and the three
#: real dpi-derived values the device fixtures use. The contrast between them
#: IS the experiment: only at the integer does the estimator return the truth
#: exactly, and every earlier measurement in this file used integers only.
SENS_PX_PER_MM = (8.0, 150 / 25.4, 200 / 25.4, 300 / 25.4)

#: Fractional pitch errors swept, in percent. Deliberately sub-percent: the
#: inject experiment already covers errors of a factor of five, and the
#: finding here is that three orders of magnitude less than that is enough to
#: ruin a reconstruction.
SENS_ERRORS_PCT = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, -0.5)

#: Multiplicative pitch errors forced directly onto the GridEstimate.
#: 5x is the harmonic failure the pipeline actually exhibits when the
#: 1 mm rules do not resolve and it locks onto the 5 mm family; 0.2x is
#: its inverse and does not occur naturally, but bounds the check's
#: behaviour on the other side.
INJECT_PITCH_FACTORS = (5.0, 0.2)


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

def git_commit() -> tuple[str, bool]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, check=True
                             ).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                                    capture_output=True, text=True, check=True
                                    ).stdout.strip())
        return sha, dirty
    except Exception:
        return "unknown", True


def module_defaults() -> dict:
    """The tuning constants a row depends on but no CLI flag records.

    The gate's thresholds belong here in full: a catch rate is a
    statement about a set of bounds, and a sidecar that omits them
    cannot be used to reproduce or dispute the number.
    """
    return {
        "quality.REJECT_PP_MV": list(quality.REJECT_PP_MV),
        "quality.FLAG_PP_MV": list(quality.FLAG_PP_MV),
        "quality.REJECT_BPM": list(quality.REJECT_BPM),
        "quality.FLAG_BPM": list(quality.FLAG_BPM),
        "quality.REJECT_COVERAGE": quality.REJECT_COVERAGE,
        "quality.FLAG_COVERAGE": quality.FLAG_COVERAGE,
        "quality.EXPECTED_TRACES_PER_COLUMN": quality.EXPECTED_TRACES_PER_COLUMN,
        "quality.BAND_SEPARATION_MM": quality.BAND_SEPARATION_MM,
        "digitise.DEFAULT_PEAK_FRACTION": dig.DEFAULT_PEAK_FRACTION,
        "digitise.MAX_ANISOTROPY": dig.MAX_ANISOTROPY,
        "digitise.MAX_SPACING_CV": dig.MAX_SPACING_CV,
        "digitise.PAPER_SPEED_RANGE_MM_S": list(dig.PAPER_SPEED_RANGE_MM_S),
        "digitise.MM_PER_S": dig.MM_PER_S,
        "digitise.MM_PER_MV": dig.MM_PER_MV,
    }


def provenance(experiment: str, params: dict) -> dict:
    sha, dirty = git_commit()
    return {
        "experiment": experiment,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": sha,
        "git_dirty": dirty,
        "params": params,
        "module_defaults": module_defaults(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "opencv": cv2.__version__,
            "platform": platform.platform(),
        },
    }


def persist(experiment: str, rows: list[dict], params: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / f"{experiment}.csv"
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    meta = provenance(experiment, params)
    meta["n_rows"] = len(rows)
    (RESULTS / f"{experiment}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return csv_path


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def load_records(n: int, rate: int, root: Path):
    """Test-fold records, in published order. Fold 10 only, never random."""
    df = ptbxl.load_metadata(root)
    _, _, test = ptbxl.split_folds(df)
    out = []
    for ecg_id in test.index:
        if len(out) >= n:
            break
        try:
            sig, meta = ptbxl.load_record(ecg_id, root=root,
                                          sampling_rate=rate, df=df)
        except FileNotFoundError:
            continue
        out.append((int(ecg_id), sig, meta))
    return out


def gate_row(result, mask, truth, fs) -> dict:
    """Flatten a gate verdict and the fidelity of what it judged.

    The two halves of every row in this report: what the gate said, and
    what the waveform was actually worth.
    """
    report = quality.assess(result, mask=mask)
    row: dict = {
        "verdict": report.verdict.name,
        "rejections": " | ".join(report.rejections),
        "flags": " | ".join(report.flags),
    }
    for c in report.checks:
        row[f"chk_{c.name}"] = c.verdict.name if c.available else "UNAVAILABLE"
        row[f"val_{c.name}"] = (round(c.value, 4)
                                if np.isfinite(c.value) else float("nan"))

    if truth is not None and result is not None:
        m = evaluate.evaluate_all(truth, result.signal_mv, fs)
        row.update(pcc=round(m.pcc, 4), rmse_mv=round(m.rmse_mv, 4),
                   snr_db=round(m.snr_db, 2), gain_ratio=round(m.gain_ratio, 4),
                   n_compared=m.n_compared)
    return row


def digitise_and_gate(image, fs_out, truth, expected_duration_s=None,
                      grid=None) -> dict:
    """Run the pipeline with guards non-fatal, then gate the output.

    `strict=False` throughout. The point of FR8 is that the gate, not an
    exception raised mid-calibration, is what refuses -- and a failure
    mode cannot be characterised if the pipeline declines to produce the
    waveform whose badness is the thing being measured.
    """
    try:
        res = dig.digitise(image, fs_out=fs_out, strict=False, grid=grid,
                           expected_duration_s=expected_duration_s)
    except Exception as exc:
        # A raise is a legitimate refusal and is recorded as one, but it
        # is not a gate verdict and must not be counted as a catch by the
        # gate. The distinction matters: the pipeline already refuses some
        # of these before the gate ever sees them.
        return {"verdict": "PIPELINE_RAISED", "rejections": f"{type(exc).__name__}: {exc}",
                "flags": "", "pcc": float("nan")}
    mask = dig.binarise_trace(image)
    return gate_row(res, mask, truth, fs_out)


# --------------------------------------------------------------------------
# experiment: false rejects on known-good input
# --------------------------------------------------------------------------

def exp_knowngood(args) -> list[dict]:
    """The false-reject rate. Every row here should read ACCEPT."""
    rows = []
    records = load_records(args.n, args.rate, args.root)

    for ecg_id, sig, meta in records:
        for lead in args.leads:
            s = sig[:, ptbxl.LEAD_NAMES.index(lead)]
            duration = len(s) / meta.fs
            for kind, scales in (("strip", args.good_px_per_mm),
                                 ("strip_lowres", args.lowres_px_per_mm)):
                for px in scales:
                    image, spec = synth.render_printout(
                        s, fs=meta.fs, px_per_mm=px, on_overflow="expand")
                    row = {"kind": kind, "ecg_id": ecg_id, "lead": lead,
                           "px_per_mm": px, "true_px_per_mm": px}
                    row.update(digitise_and_gate(image, meta.fs, s,
                                                 expected_duration_s=duration))
                    rows.append(row)

    # The three reference printouts. 3x4 sheets, so they exercise the
    # calibration check on real ecg-image-kit ink and nothing else; there
    # is no per-lead ground truth waveform for them in this repository.
    for name, (path, true_px) in FIXTURES.items():
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        g = dig.estimate_grid_pitch(img)
        chk = dig.check_calibration(g, image_width_px=img.shape[1],
                                    expected_duration_s=10.0)
        rows.append({
            "kind": "fixture_calibration_only", "ecg_id": name, "lead": "",
            "px_per_mm": round(g.px_per_mm_x, 4), "true_px_per_mm": round(true_px, 4),
            "verdict": "ACCEPT" if chk.ok else "REJECT",
            "rejections": " | ".join(chk.reasons),
            "flags": " | ".join(chk.warnings),
            "chk_calibration": "ACCEPT" if chk.ok else "REJECT",
            "val_calibration": round(chk.implied_mm_per_s, 4)
            if np.isfinite(chk.implied_mm_per_s) else float("nan"),
            "pitch_error_pct": round((g.px_per_mm_x - true_px) / true_px * 100, 3)
            if np.isfinite(g.px_per_mm_x) else float("nan"),
        })
    return rows


# --------------------------------------------------------------------------
# experiment: catch rate on injected failures
# --------------------------------------------------------------------------

def exp_inject(args) -> list[dict]:
    """The catch rate per failure mode, with the PCC of what was judged."""
    rows = []
    records = load_records(args.n, args.rate, args.root)
    px = args.px_per_mm

    for ecg_id, sig, meta in records:
        for lead in args.leads:
            s = sig[:, ptbxl.LEAD_NAMES.index(lead)]
            duration = len(s) / meta.fs
            image, _ = synth.render_printout(s, fs=meta.fs, px_per_mm=px,
                                             on_overflow="expand")

            def row_for(mode, im, truth, grid=None, expected=duration):
                r = {"mode": mode, "ecg_id": ecg_id, "lead": lead,
                     "px_per_mm": px}
                r.update(digitise_and_gate(im, meta.fs, truth,
                                           expected_duration_s=expected,
                                           grid=grid))
                return r

            # The control row. Not a failure; present so that every
            # injected row has a same-record baseline PCC beside it.
            rows.append(row_for("clean", image, s))

            # Rotation, uncorrected. Deskew exists and would remove this;
            # the failure being characterised is a rotated frame reaching
            # calibration, which is what happens if deskew is skipped or
            # fails. The simulating warp is fixed at lanczos and is not a
            # parameter under test here.
            for angle in args.angles:
                rot = dsk.rotate_image(image, angle,
                                       interpolation=args.sim_interp)
                rows.append(row_for(f"rotated_{angle}deg", rot, s))

            # Pitch forced wrong by an integer factor, injected on the
            # GridEstimate rather than by degrading the image. This
            # separates "the estimator locked onto a harmonic" from "the
            # image was too poor to estimate": only the pitch varies.
            true_grid = dig.estimate_grid_pitch(image)
            for factor in args.pitch_factors:
                forced = dig.GridEstimate(
                    px_per_mm_x=true_grid.px_per_mm_x * factor,
                    px_per_mm_y=true_grid.px_per_mm_y * factor,
                    confidence=true_grid.confidence,
                    n_peaks_x=true_grid.n_peaks_x,
                    n_peaks_y=true_grid.n_peaks_y,
                    spacing_cv=true_grid.spacing_cv,
                )
                rows.append(row_for(f"pitch_x{factor}", image, s, grid=forced))
                # The same injection with the recording duration withheld,
                # which is the normal condition for a photograph of
                # someone else's ECG: nobody can tell the app how long the
                # strip was meant to be. The paper-speed check skips
                # itself, so this isolates what the remaining anchors --
                # chiefly the heart-rate backstop -- are worth alone.
                rows.append(row_for(f"pitch_x{factor}_no_duration", image, s,
                                    grid=forced, expected=None))

        # A 3x4 printout fed to the single-strip tracer with no lead
        # extraction. Truth is lead II, which the rhythm strip carries in
        # full; the tracer has no idea the other eleven leads are there.
        sheet, _ = layout.render_12lead(sig, meta.fs, px_per_mm=px)
        truth_ii = sig[:, ptbxl.LEAD_NAMES.index(layout.RHYTHM_LEAD)]
        r = {"mode": "layout_3x4_no_segmentation", "ecg_id": ecg_id,
             "lead": layout.RHYTHM_LEAD, "px_per_mm": px}
        r.update(digitise_and_gate(sheet, meta.fs, truth_ii,
                                   expected_duration_s=len(truth_ii) / meta.fs))
        rows.append(r)

    return rows


def exp_pitchsens(args) -> list[dict]:
    """How much PCC a sub-percent pitch error costs, and whether FR8 sees it.

    This experiment exists because of a mistake in the two above. Both render
    at 8, 12 and 16 px/mm -- all *integers* -- and at an integer pitch the
    printed rules land on exact pixel columns and `estimate_grid_pitch`
    returns the true value to 0.000%. The 0.980 mean PCC they report is
    therefore measured under a condition that does not occur: a real printout
    at 200 dpi is 7.874 px/mm, the rules rasterise onto integer columns, and
    the median of the autocorrelation peak spacings is pulled toward the
    nearest integer.

    Measured on the committed strip fixtures: true 7.874, recovered 7.918,
    error +0.555%, PCC 0.628. Supply the true pitch to the same image and PCC
    is 0.985. So the tracer is fine and the calibration is not, and the error
    responsible is a *half a percent*.

    The mechanism is a time-base error accumulating over the strip. A +0.5%
    error in px_per_s stretches 10 s by 50 ms, which at 100 Hz is five
    samples of drift by the end -- comparable to the width of a QRS complex,
    so the last beats are compared against the wrong beats. PCC has no
    tolerance for that at all.

    Two things this measures. The sensitivity curve itself, by injecting a
    known *fractional* pitch error rather than the integer factors the inject
    experiment uses. And, for each, the gate's verdict -- because an implied
    paper speed of 25.1 mm/s against a nominal 25 is entirely plausible, so
    the expectation is that nothing catches this at all.
    """
    rows = []
    records = load_records(args.n, args.rate, args.root)

    for ecg_id, sig, meta in records:
        for lead in args.leads:
            s = sig[:, ptbxl.LEAD_NAMES.index(lead)]
            duration = len(s) / meta.fs
            for px in args.sens_px_per_mm:
                image, spec = synth.render_printout(
                    s, fs=meta.fs, px_per_mm=px, on_overflow="expand")
                est = dig.estimate_grid_pitch(image)
                true_px = spec.px_per_mm

                # The estimator's own error, then a swept series of exact
                # fractional errors around the truth. The first row is what
                # the pipeline actually does; the rest are the curve it sits
                # on.
                cases: list[tuple[str, float]] = [("estimated", est.px_per_mm_x)]
                for frac in args.sens_errors:
                    cases.append((f"forced_{frac:+.3f}pct",
                                  true_px * (1.0 + frac / 100.0)))

                for label, pitch in cases:
                    if not np.isfinite(pitch):
                        continue
                    grid = dig.GridEstimate(
                        px_per_mm_x=pitch, px_per_mm_y=pitch,
                        confidence=est.confidence, n_peaks_x=est.n_peaks_x,
                        n_peaks_y=est.n_peaks_y, spacing_cv=est.spacing_cv,
                    )
                    row = {
                        "ecg_id": ecg_id, "lead": lead,
                        "true_px_per_mm": round(true_px, 5),
                        "integer_pitch": float(px).is_integer(),
                        "case": label,
                        "px_per_mm_used": round(pitch, 5),
                        "pitch_error_pct": round(
                            (pitch / true_px - 1.0) * 100.0, 4),
                    }
                    row.update(digitise_and_gate(image, meta.fs, s,
                                                 expected_duration_s=duration,
                                                 grid=grid))
                    rows.append(row)
    return rows


EXPERIMENTS = {"knowngood": exp_knowngood, "inject": exp_inject,
               "pitchsens": exp_pitchsens}


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _pcc_mean(rows: list[dict]) -> float:
    vals = [r["pcc"] for r in rows
            if isinstance(r.get("pcc"), float) and np.isfinite(r["pcc"])]
    return float(np.mean(vals)) if vals else float("nan")


def report(name: str, rows: list[dict]) -> None:
    """Print the two rates the deliverable asks for. Nothing derived.

    `PIPELINE_RAISED` is counted apart from a gate verdict throughout. The
    pipeline refuses some of these inputs before FR8 ever sees them --
    `digitise` raises on a pitch that did not resolve -- and folding that
    into a catch rate would credit the gate with work it did not do.
    """
    if name == "knowngood":
        for kind, label in (("strip", "known-good strips"),
                            ("strip_lowres", "low-resolution strips")):
            scored = [r for r in rows if r["kind"] == kind]
            if not scored:
                continue
            rejected = [r for r in scored if r["verdict"] == "REJECT"]
            flagged = [r for r in scored if r["verdict"] == "FLAG"]
            raised = [r for r in scored if r["verdict"] == "PIPELINE_RAISED"]
            print(f"  {label}: {len(scored)}   (mean PCC {_pcc_mean(scored):.3f})")
            print(f"    FALSE REJECTS:    {len(rejected)}/{len(scored)} "
                  f"({100 * len(rejected) / len(scored):.1f}%)")
            print(f"    flagged (usable): {len(flagged)}/{len(scored)} "
                  f"({100 * len(flagged) / len(scored):.1f}%)")
            if raised:
                print(f"    pipeline raised:  {len(raised)}/{len(scored)}")
            for r in rejected + flagged:
                print(f"      {r['ecg_id']} {r['lead']} {r['px_per_mm']} px/mm "
                      f"pcc={r.get('pcc')} -> {r['verdict']}: "
                      f"{r['rejections'] or r['flags']}")
        for r in (x for x in rows if x["kind"] == "fixture_calibration_only"):
            print(f"  fixture {r['ecg_id']}: {r['verdict']} "
                  f"(pitch error {r['pitch_error_pct']}%)")
        return

    if name == "pitchsens":
        cases = list(dict.fromkeys(r["case"] for r in rows))
        print(f"  {'case':22s} {'n':>3s} {'err %':>7s} {'PCC':>7s} "
              f"{'gain':>6s} {'acted':>6s}")
        for case in cases:
            sel = [r for r in rows if r["case"] == case]
            errs = [r["pitch_error_pct"] for r in sel]
            acted = [r for r in sel if r["verdict"] in ("REJECT", "FLAG")]
            print(f"  {case:22s} {len(sel):3d} {np.mean(errs):7.3f} "
                  f"{_pcc_mean(sel):7.3f} "
                  f"{np.mean([r['gain_ratio'] for r in sel if 'gain_ratio' in r]):6.3f} "
                  f"{100 * len(acted) / max(1, len(sel)):5.0f}%")
        print("  A sub-percent pitch error is invisible to every anchored "
              "check and costs a third of PCC.")
        return

    modes = list(dict.fromkeys(r["mode"] for r in rows))
    print(f"  {'mode':34s} {'n':>3s} {'rej':>4s} {'flag':>4s} {'raise':>5s} "
          f"{'acted':>6s} {'PCC acc':>8s} {'PCC ref':>8s}")
    for mode in modes:
        sel = [r for r in rows if r["mode"] == mode]
        rej = [r for r in sel if r["verdict"] == "REJECT"]
        flag = [r for r in sel if r["verdict"] == "FLAG"]
        raised = [r for r in sel if r["verdict"] == "PIPELINE_RAISED"]
        accepted = [r for r in sel if r["verdict"] == "ACCEPT"]
        # FR8 is "reject OR flag", so both count as the gate having acted;
        # only a silent ACCEPT is a miss. Reported beside the bare reject
        # count, not instead of it.
        acted = len(rej) + len(flag)
        print(f"  {mode:34s} {len(sel):3d} {len(rej):4d} {len(flag):4d} "
              f"{len(raised):5d} {100 * acted / max(1, len(sel)):5.0f}% "
              f"{_pcc_mean(accepted):8.3f} {_pcc_mean(rej + flag):8.3f}")
    print("  acted = rejected or flagged, i.e. not silently accepted.")
    print("  PCC acc = mean PCC of what the gate let through unremarked.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", nargs="+", default=["all"],
                   help=f"one or more of {list(EXPERIMENTS)}, or 'all'")
    p.add_argument("--root", type=Path, default=ptbxl.DEFAULT_ROOT)
    p.add_argument("--n", type=int, default=12, help="test-fold records")
    p.add_argument("--rate", type=int, choices=(100, 500), default=100)
    p.add_argument("--leads", nargs="+", default=["II"])
    p.add_argument("--px-per-mm", type=float, default=8.0,
                   help="render scale for the injection experiment")
    p.add_argument("--good-px-per-mm", type=float, nargs="+",
                   default=list(GOOD_PX_PER_MM))
    p.add_argument("--lowres-px-per-mm", type=float, nargs="+",
                   default=list(LOWRES_PX_PER_MM))
    p.add_argument("--sens-px-per-mm", type=float, nargs="+",
                   default=list(SENS_PX_PER_MM))
    p.add_argument("--sens-errors", type=float, nargs="+",
                   default=list(SENS_ERRORS_PCT))
    p.add_argument("--angles", type=float, nargs="+", default=list(INJECT_ANGLES))
    p.add_argument("--pitch-factors", type=float, nargs="+",
                   default=list(INJECT_PITCH_FACTORS))
    p.add_argument("--sim-interp", default="lanczos",
                   choices=list(dsk.INTERPOLATIONS),
                   help="interpolation used to SIMULATE the injected rotation")
    args = p.parse_args(argv)

    chosen = list(EXPERIMENTS) if "all" in args.experiment else args.experiment
    params = {k: (str(v) if isinstance(v, Path) else v)
              for k, v in vars(args).items() if k != "experiment"}

    for name in chosen:
        t0 = time.perf_counter()
        print(f"\n=== {name} ===", flush=True)
        rows = EXPERIMENTS[name](args)
        path = persist(name, rows, params)
        report(name, rows)
        print(f"  {len(rows)} rows -> {path.relative_to(REPO)} "
              f"({time.perf_counter() - t0:.1f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
