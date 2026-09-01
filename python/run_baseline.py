#!/usr/bin/env python3
"""Baseline experiment: synthetic signal -> render -> digitise -> score.

This is the closed loop with every real-world difficulty removed. The
signal is a sum of Gaussians, the image is rendered by the same code that
defines the grid, there is no camera, no perspective, no noise and no
paper. Whatever this scores is the **ceiling** of the classical pipeline,
not an estimate of its field performance -- and if it does not score well
here, nothing downstream will.

Its real job is to isolate resolution. The sweep over `px_per_mm` answers
the question that decides whether phone capture is viable at all: how many
pixels per millimetre does `estimate_grid_pitch` need before it stops
resolving the 1 mm rules and locks onto the 5 mm ones instead?

Usage::

    python python/run_baseline.py
    python python/run_baseline.py --fs 500 --pitches 4 6 8 12 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from ecgproto import digitise as dig
from ecgproto import evaluate, synth


def run_once(
    signal: np.ndarray,
    fs: float,
    px_per_mm: float,
) -> tuple[evaluate.Metrics, dig.DigitiseResult, float]:
    """Render one strip, digitise it, and score against the source signal."""
    image, spec = synth.render_printout(signal, fs=fs, px_per_mm=px_per_mm)

    result = dig.digitise(image, fs_out=fs)
    metrics = evaluate.evaluate_all(signal, result.signal_mv, fs=fs)

    # Relative error of the recovered scale against the value we rendered
    # with. This is the only place the answer key is legitimately used --
    # to score the estimator, never to feed it.
    pitch_error = (result.grid.px_per_mm - spec.px_per_mm) / spec.px_per_mm
    return metrics, result, pitch_error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fs", type=float, default=500.0,
                        help="sampling rate of the synthetic signal (Hz)")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--heart-rate", type=float, default=60.0)
    parser.add_argument("--noise-mv", type=float, default=0.0)
    parser.add_argument("--pitches", type=float, nargs="+",
                        default=[3.0, 4.0, 6.0, 8.0, 12.0, 16.0],
                        help="px_per_mm values to sweep")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    signal = synth.synthetic_ecg(
        duration_s=args.duration, fs=args.fs,
        heart_rate_bpm=args.heart_rate, noise_mv=args.noise_mv, seed=args.seed,
    )

    print(f"synthetic ECG: {len(signal)} samples @ {args.fs:g} Hz, "
          f"{args.heart_rate:g} bpm, peak {signal.max():.2f} mV")
    print(f"paper: {synth.MM_PER_S:g} mm/s, {synth.MM_PER_MV:g} mm/mV\n")

    header = (f"{'px/mm':>6}  {'est px/mm':>9}  {'pitch err':>9}  "
              f"{'SNR dB':>7}  {'RMSE mV':>8}  {'PCC':>6}  {'gain':>6}  "
              f"{'cover':>6}  diagnosis")
    print(header)
    print("-" * len(header))

    failures = 0
    for px_per_mm in args.pitches:
        try:
            metrics, result, pitch_error = run_once(signal, args.fs, px_per_mm)
        except ValueError as exc:
            failures += 1
            print(f"{px_per_mm:6.1f}  {'--':>9}  {'--':>9}  "
                  f"{'--':>7}  {'--':>8}  {'--':>6}  {'--':>6}  {'--':>6}  "
                  f"FAILED: {exc}")
            continue

        print(f"{px_per_mm:6.1f}  {result.grid.px_per_mm:9.3f}  {pitch_error:+8.2%}  "
              f"{metrics.snr_db:7.2f}  {metrics.rmse_mv:8.4f}  "
              f"{metrics.pcc:6.3f}  {metrics.gain_ratio:6.3f}  "
              f"{result.coverage:6.1%}  {metrics.diagnose()}")

    print("\nRead the columns together, not separately:")
    print("  gain far from 1.000   -> calibration wrong, tracing fine")
    print("  pitch err near +400%  -> locked onto the 5 mm major rules")
    print("  PCC near -1           -> pixels-to-mV sign flipped")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
