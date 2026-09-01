#!/usr/bin/env python3
"""Digitisation experiment on real PTB-XL waveforms.

The signals here are genuine clinical recordings; the *images* are still
rendered by `synth`, so this measures the digitiser against real waveform
morphology but not against real paper, ink or cameras. It is the second of
three rungs: synthetic signal -> real signal -> real photograph. Only the
third rung supports a claim about field performance.

Records are drawn from the **test fold only** (`strat_fold == 10`). Nothing
here trains anything, so the split is not strictly required -- but reaching
for records by index is how a project drifts into reporting numbers on data
it tuned against, and the habit is cheaper to keep than to retrofit.

Each record is scored per-lead on a single continuous strip. That is what
`digitise` supports: a 3x4 printout would need lead segmentation, which
does not exist yet. Use ``--printout`` to render one for inspection -- it
is deliberately not digitised.

Usage::

    python python/run_ptbxl.py --n 20
    python python/run_ptbxl.py --n 5 --leads II V1 --rate 500
    python python/run_ptbxl.py --n 1 --printout out/printout.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from ecgproto import digitise as dig
from ecgproto import evaluate, layout, ptbxl, synth


def score_lead(
    signal: np.ndarray,
    fs: float,
    px_per_mm: float,
) -> evaluate.Metrics:
    """Render one lead as a strip, digitise it, score against the source."""
    image, _ = synth.render_printout(signal, fs=fs, px_per_mm=px_per_mm)
    result = dig.digitise(image, fs_out=fs)
    return evaluate.evaluate_all(signal, result.signal_mv, fs=fs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=ptbxl.DEFAULT_ROOT)
    parser.add_argument("--n", type=int, default=10, help="records to score")
    parser.add_argument("--rate", type=int, choices=(100, 500), default=100)
    parser.add_argument("--px-per-mm", type=float, default=8.0)
    parser.add_argument("--leads", nargs="+", default=["II"],
                        help=f"any of {' '.join(ptbxl.LEAD_NAMES)}")
    parser.add_argument("--printout", type=Path, default=None,
                        help="render the first record as a 3x4 printout here")
    args = parser.parse_args(argv)

    unknown = set(args.leads) - set(ptbxl.LEAD_NAMES)
    if unknown:
        parser.error(f"unknown leads {sorted(unknown)}; expected {ptbxl.LEAD_NAMES}")

    if not args.root.exists():
        print(f"dataset not found at {args.root}\n"
              f"run: python python/verify_ptbxl.py --quick", file=sys.stderr)
        return 2

    df = ptbxl.load_metadata(args.root)
    _, _, test = ptbxl.split_folds(df)
    print(f"test fold: {len(test)} records; scoring the first {args.n}")

    lead_index = {name: i for i, name in enumerate(ptbxl.LEAD_NAMES)}
    rows: list[tuple[int, str, evaluate.Metrics]] = []
    skipped: list[tuple[int, str]] = []

    for ecg_id in test.index[: args.n]:
        try:
            signal, meta = ptbxl.load_record(
                ecg_id, root=args.root, sampling_rate=args.rate, df=df
            )
        except FileNotFoundError:
            # An incomplete download is a real and common state; say which
            # record was skipped rather than silently shrinking the sample.
            skipped.append((int(ecg_id), "file missing"))
            continue

        if args.printout and len(rows) == 0:
            image, spec = layout.render_12lead(signal, fs=meta.fs,
                                               px_per_mm=args.px_per_mm)
            args.printout.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.printout), image)
            print(f"wrote {args.printout} "
                  f"({image.shape[1]}x{image.shape[0]} px, "
                  f"{spec.printed_fraction:.1%} of lead-seconds printed) "
                  f"-- not digitised: needs lead segmentation")

        for lead in args.leads:
            try:
                metrics = score_lead(
                    signal[:, lead_index[lead]], meta.fs, args.px_per_mm
                )
            except ValueError as exc:
                skipped.append((int(ecg_id), f"{lead}: {exc}"))
                continue
            rows.append((int(ecg_id), lead, metrics))

    if not rows:
        print("nothing scored", file=sys.stderr)
        return 1

    header = (f"{'ecg_id':>7}  {'lead':>4}  {'SNR dB':>7}  {'shift dB':>8}  "
              f"{'RMSE mV':>8}  {'PCC':>6}  diagnosis")
    print("\n" + header)
    print("-" * len(header))
    for ecg_id, lead, m in rows:
        print(f"{ecg_id:7d}  {lead:>4}  {m.snr_db:7.2f}  {m.shifted_snr_db:8.2f}  "
              f"{m.rmse_mv:8.4f}  {m.pcc:6.3f}  {m.diagnose()}")

    snrs = np.array([m.snr_db for _, _, m in rows])
    rmses = np.array([m.rmse_mv for _, _, m in rows])
    pccs = np.array([m.pcc for _, _, m in rows])
    finite = np.isfinite(snrs)

    print(f"\nn = {len(rows)} lead-traces from {len({r[0] for r in rows})} records")
    print(f"  SNR  dB : median {np.median(snrs[finite]):6.2f}   "
          f"IQR {np.percentile(snrs[finite], 25):.2f}-"
          f"{np.percentile(snrs[finite], 75):.2f}")
    print(f"  RMSE mV : median {np.median(rmses):6.4f}   "
          f"max {rmses.max():.4f}")
    print(f"  PCC     : median {np.median(pccs):6.3f}   "
          f"min {pccs.min():.3f}")

    if skipped:
        print(f"\nskipped {len(skipped)}:")
        for ecg_id, reason in skipped[:10]:
            print(f"  {ecg_id}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
