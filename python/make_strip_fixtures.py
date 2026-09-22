#!/usr/bin/env python3
"""Render single-strip printouts into the Android assets, with ground truth.

**Why the app needs these.** The three printouts already bundled
(`gt_norm_150dpi`, `gt_mi_200dpi`, `gt_sttc_300dpi`) are 3x4 sheets. They
are the right fixtures for FR3, grid estimation, which is a property of the
grid and does not care how many leads are on the page. They are the *wrong*
fixtures for FR5 to FR7: `digitise` traces a single continuous strip, and
handed a sheet it runs straight across the cell rows and returns a waveform
spliced from four leads at once. Measured: PCC 0.273, with pitch correct to
0.00% and every calibration statistic clean (results/gate/inject.csv).

So porting the tracer to Kotlin with only those three assets on the device
would give a screen that displays a confident wrong waveform and no way to
tell. These are the missing half of the fixture set: one lead, rendered as
one strip, at the same three scales, with the source waveform committed
beside them so the phone can compute its own reconstruction fidelity.

That is the same trick the rotation slider plays for deskew. The estimator
is scored against truth **on the handset**, not against a number carried
over from the desktop, because a number carried over cannot detect a
porting bug — which is the whole risk the Kotlin side carries.

**What is deliberately not fixed here.** The closed loop. These images are
drawn by the same code and the same constants that later decode them, so
the pitch comes back nearly exact and the strip starts where the renderer
said it starts. A camera grants neither. Breaking that loop is what
`make_print_target.py`, `docs/photo_protocol.md` and `score_photo.py` are
for, and it needs a human with a printer. These fixtures measure the port,
not the pipeline's realism.

    python python/make_strip_fixtures.py

Writes into android/app/src/main/assets/reference/ and commits nothing by
itself.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from ecgproto import digitise as dig
from ecgproto import evaluate, ptbxl, quality, synth

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "android" / "app" / "src" / "main" / "assets" / "reference"

#: The same three scales the 3x4 fixtures use, so a reader comparing a
#: strip result against a sheet result is comparing at equal pitch. Named
#: by dpi because that is how the existing assets are named, while the
#: pipeline only ever sees px/mm -- these images carry no dpi metadata and
#: nothing is permitted to read any.
DPIS = (150, 200, 300)

#: Lead II. It is the rhythm lead on a clinical 3x4 printout and therefore
#: the one lead that genuinely exists as a single continuous 10 s strip, so
#: a strip of lead II is the fixture that most resembles something the
#: finished pipeline will really be handed.
LEAD = "II"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ptbxl.DEFAULT_ROOT)
    p.add_argument("--ecg-id", type=int, default=None,
                   help="default: the first test-fold record, so the choice "
                        "is reproducible and never a training record")
    p.add_argument("--rate", type=int, choices=(100, 500), default=100)
    p.add_argument("--lead", default=LEAD)
    args = p.parse_args(argv)

    df = ptbxl.load_metadata(args.root)
    if args.ecg_id is None:
        # Fold 10 only. A fixture drawn from the training folds would be
        # unusable the moment a classifier exists, and replacing fixtures
        # later invalidates every number measured against them.
        _, _, test = ptbxl.split_folds(df)
        ecg_id = int(test.index[0])
    else:
        ecg_id = args.ecg_id

    sig, meta = ptbxl.load_record(ecg_id, root=args.root,
                                  sampling_rate=args.rate, df=df)
    lead_idx = ptbxl.LEAD_NAMES.index(args.lead)
    s = sig[:, lead_idx]

    ASSETS.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "generated_by": "python/make_strip_fixtures.py",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "ecg_id": ecg_id,
        "lead": args.lead,
        "fs_hz": float(meta.fs),
        "n_samples": int(len(s)),
        "duration_s": float(len(s) / meta.fs),
        "mm_per_s": float(synth.MM_PER_S),
        "mm_per_mv": float(synth.MM_PER_MV),
        "truth_csv": "strip_ii_truth.csv",
        "versions": {"python": platform.python_version(),
                     "numpy": np.__version__, "opencv": cv2.__version__},
        "strips": [],
        "note": (
            "Single continuous strips, the valid input for FR5-FR7. The three "
            "gt_*.png assets in this directory are 3x4 sheets and are valid "
            "for FR3 only; fed to the tracer they produce a spliced waveform "
            "at PCC 0.273. Truth is one waveform shared by all three strips: "
            "same record, same lead, three render scales."
        ),
    }

    for dpi in DPIS:
        px_per_mm = dpi / 25.4
        image, spec = synth.render_printout(
            s, fs=meta.fs, px_per_mm=px_per_mm, on_overflow="expand")
        name = f"strip_ii_{dpi}dpi.png"
        cv2.imwrite(str(ASSETS / name), image)

        # What the DESKTOP pipeline gets from the very same committed PNG.
        #
        # Without this the handset would report a PCC with nothing to compare
        # it against, and a porting bug would look like a result. These are
        # the numbers the device should reproduce; a material gap between them
        # is a divergence in the Kotlin, not a property of the image.
        #
        # Read the image back off disk rather than scoring the in-memory
        # array, so that PNG encoding is inside the comparison exactly as it
        # is for the phone.
        reread = cv2.imread(str(ASSETS / name), cv2.IMREAD_COLOR)
        expected: dict = {}
        try:
            grid = dig.estimate_grid_pitch(reread)
            res = dig.digitise(reread, fs_out=meta.fs, strict=False,
                               expected_duration_s=len(s) / meta.fs)
            m = evaluate.evaluate_all(s, res.signal_mv, meta.fs)
            rep = quality.assess(res, mask=dig.binarise_trace(reread))
            expected = {
                "px_per_mm_estimated_x": round(float(grid.px_per_mm_x), 4),
                "px_per_mm_estimated_y": round(float(grid.px_per_mm_y), 4),
                "pitch_error_pct": round(
                    (grid.px_per_mm_x / spec.px_per_mm - 1.0) * 100.0, 3),
                "pcc": round(float(m.pcc), 4),
                "rmse_mv": round(float(m.rmse_mv), 4),
                "gain_ratio": round(float(m.gain_ratio), 4),
                "coverage": round(float(res.coverage), 4),
                "n_samples": int(len(res.signal_mv)),
                "gate_verdict": rep.verdict.name,
            }
        except Exception as exc:                        # pragma: no cover
            expected = {"error": f"{type(exc).__name__}: {exc}"}

        manifest["strips"].append({
            "desktop_expected": expected,
            "file": name,
            "dpi_nominal": dpi,
            "px_per_mm_true": float(spec.px_per_mm),
            "width_px": int(image.shape[1]),
            "height_px": int(image.shape[0]),
            "baseline_px": float(spec.baseline_px),
            # The image column of t = 0. `render_printout` insets the trace by
            # margin_mm, so a verification overlay that assumes the strip
            # starts at column 0 is shifted by ~39 px at 200 dpi -- which on a
            # QRS upstroke reads as tens of pixels of disagreement and looks
            # like the wrong waveform.
            "x0_px": float(spec.x0_px),
            "clipped": bool(spec.clipped),
            # The sampling rate one pixel column implies, which is what the
            # tracer's output rate actually is before resampling:
            # px_per_mm * mm_per_s.
            "implied_fs_hz": float(spec.px_per_mm * synth.MM_PER_S),
        })
        print(f"  {name}  {image.shape[1]}x{image.shape[0]} px, "
              f"{spec.px_per_mm:.4f} px/mm, clipped={spec.clipped}")
        print(f"      desktop: pcc {expected.get('pcc')}, "
              f"gain {expected.get('gain_ratio')}, "
              f"rmse {expected.get('rmse_mv')} mV, "
              f"pitch {expected.get('pitch_error_pct')}%, "
              f"gate {expected.get('gate_verdict')}")

    # Truth as one float per line. Not .npy: this is read by Kotlin, and a
    # hand-rolled .npy header parser on the device would be a porting
    # hazard in service of saving a few kilobytes.
    truth_path = ASSETS / manifest["truth_csv"]
    truth_path.write_text(
        "\n".join(repr(float(v)) for v in s) + "\n", encoding="utf-8")
    print(f"  {truth_path.name}  {len(s)} samples at {meta.fs} Hz")

    (ASSETS / "strip_fixtures.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  strip_fixtures.json  record {ecg_id}, lead {args.lead}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
