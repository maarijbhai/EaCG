"""Reconstruction metrics: SNR, shifted SNR, RMSE and PCC.

**These four must be reported together**, because each is blind to a
different failure and reading any one alone will mislead you.

======  ==========================  =====================================
metric  sensitive to                blind to
======  ==========================  =====================================
SNR     shape, gain, offset, time   nothing much -- but conflates them all
RMSE    gain and offset, in mV      shape detail swamped by a DC error
PCC     shape only                  scale and offset, completely
======  ==========================  =====================================

The useful reading is the *combination*:

* **High PCC, high RMSE** -- the tracer followed the waveform correctly and
  the calibration is wrong. The fault is in `estimate_grid_pitch` or in the
  assumed mm/mV, not in the tracing. This is the single most common failure
  and the one a naive "PCC = 0.99, great" report hides completely.
* **Low PCC** -- the tracer lost the waveform. Calibration is irrelevant
  until that is fixed.
* **SNR far below shifted SNR** -- the reconstruction is right but
  time-shifted, usually because the strip's left edge was found a few
  pixels off. That is a cropping question, not a tracing one.

SNR is the field default. The PhysioNet Challenge 2024 scored a *shifted*
SNR precisely because a digitiser that recovers the waveform but starts it
two samples late is not meaningfully worse, and plain SNR punishes it
severely.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import signal as sps


@dataclass(frozen=True)
class Metrics:
    """One reconstruction scored four ways. Amplitudes in millivolts."""

    snr_db: float
    shifted_snr_db: float
    rmse_mv: float
    pcc: float
    gain_ratio: float          # least-squares amplitude scale, test/reference
    best_shift_samples: int
    n_compared: int

    def as_dict(self) -> dict:
        return asdict(self)

    def diagnose(
        self,
        rmse_tolerance_mv: float = 0.05,
        gain_tolerance: float = 0.05,
    ) -> str:
        """Plain-language reading of the metrics together.

        `gain_ratio` is what separates the two ways a high RMSE can arise.
        A miscalibrated trace is *systematically* scaled, so its best-fit
        gain sits well away from 1.0. A correctly calibrated but
        under-resolved trace has a gain near 1.0 and its error is
        broadband -- amplitude noise from column-centroid quantisation, not
        a scale factor. Reporting the second as a calibration fault sends
        you to debug `estimate_grid_pitch` when the real answer is that the
        image needs more pixels per millimetre.
        """
        if not np.isfinite(self.pcc):
            return "degenerate: a signal has zero variance"
        if self.pcc < 0.0:
            return ("inverted: shape matches but sign is flipped -- check the "
                    "pixels-to-mV sign convention (image rows grow downward)")
        if self.pcc < 0.7:
            return "tracing failure: the digitiser lost the waveform"
        if abs(self.gain_ratio - 1.0) > gain_tolerance:
            # Under-reading and over-reading have different usual causes, and
            # saying which narrows the search considerably.
            cause = ("centroid peak-clipping at low px/mm, grid pitch, or mm/mV"
                     if self.gain_ratio < 1.0 else "grid pitch or mm/mV")
            return (f"amplitude scaled by {self.gain_ratio:.3f} -- suspect {cause}")
        if self.shifted_snr_db - self.snr_db > 3.0:
            return (f"time offset: {self.best_shift_samples} samples -- "
                    "reconstruction is good but misaligned")
        if self.rmse_mv > rmse_tolerance_mv:
            return ("resolution-limited: gain is right, residual is broadband "
                    "-- more px/mm, or a sub-pixel tracer")
        return "good reconstruction"


def align_length(
    reference: np.ndarray,
    test: np.ndarray,
    method: str = "truncate",
) -> tuple[np.ndarray, np.ndarray]:
    """Bring two signals to a common length before comparison.

    **`method="resample"` biases the result, and the bias flatters the
    digitiser.** Stretching the reconstruction onto the reference's length
    is a time warp fitted from the very quantity being measured. If the
    digitiser recovered the strip 2% too long -- because
    `estimate_grid_pitch` was 2% off, so `px_per_s` was wrong -- then
    resampling by length ratio silently *undoes* exactly that error before
    any metric sees it. A systematic calibration fault on the time axis
    becomes invisible, and SNR improves for the wrong reason. The distortion
    grows with the length mismatch, so the worse the calibration, the more
    generously it is corrected.

    `method="truncate"` (the default) compares the first `min(len)` samples
    of each and leaves any time-axis error where the metrics can see it. It
    costs you the tail of the longer signal, which is the right trade.

    Use `"resample"` only when the length difference is known to be a
    cropping artefact rather than a calibration one, and say so in the
    write-up.
    """
    reference = np.asarray(reference, dtype=np.float64).ravel()
    test = np.asarray(test, dtype=np.float64).ravel()

    if len(reference) == len(test):
        return reference, test

    if method == "truncate":
        n = min(len(reference), len(test))
        return reference[:n], test[:n]
    if method == "resample":
        return reference, sps.resample(test, len(reference))
    raise ValueError(f"method must be 'truncate' or 'resample', got {method!r}")


def snr_db(reference: np.ndarray, test: np.ndarray) -> float:
    """10 log10( signal power / error power ), in dB.

    The reference supplies the signal power, so this is asymmetric: swapping
    the arguments does not give the same number.
    """
    reference, test = align_length(reference, test)
    error = reference - test
    signal_power = float(np.sum(reference ** 2))
    error_power = float(np.sum(error ** 2))
    if signal_power == 0:
        return float("nan")
    if error_power == 0:
        return float("inf")
    return 10.0 * np.log10(signal_power / error_power)


def shifted_snr_db(
    reference: np.ndarray,
    test: np.ndarray,
    max_shift_samples: int = 10,
) -> tuple[float, int]:
    """Best SNR over integer shifts in [-max_shift, +max_shift].

    Returns (snr_db, best_shift). A positive shift means `test` had to be
    delayed to match `reference`.

    Each candidate shift is scored on the overlapping region only, and the
    overlap shrinks as |shift| grows. That slightly favours large shifts --
    fewer samples, and the trimmed ones are at the edges where the trace is
    least reliable -- so keep `max_shift_samples` small relative to the
    signal (a few tens of milliseconds, not seconds).
    """
    reference, test = align_length(reference, test)
    n = len(reference)
    max_shift = int(min(max_shift_samples, n - 1))

    best_snr, best_shift = -np.inf, 0
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            ref_seg, test_seg = reference[shift:], test[: n - shift]
        else:
            ref_seg, test_seg = reference[: n + shift], test[-shift:]
        if len(ref_seg) < 2:
            continue
        candidate = snr_db(ref_seg, test_seg)
        if np.isfinite(candidate) and candidate > best_snr:
            best_snr, best_shift = candidate, shift
    return float(best_snr), int(best_shift)


def rmse_mv(reference: np.ndarray, test: np.ndarray) -> float:
    """Root-mean-square error, in millivolts.

    Amplitude-sensitive and reported in physical units, so it is only
    meaningful if both signals really are in mV. It is the metric that
    catches a gain error -- and the one that a PCC-only report misses.
    """
    reference, test = align_length(reference, test)
    return float(np.sqrt(np.mean((reference - test) ** 2)))


def pcc(reference: np.ndarray, test: np.ndarray) -> float:
    """Pearson correlation coefficient, in [-1, 1].

    **Scale- and offset-invariant**, so a trace with perfect shape and
    completely wrong calibration scores ~1.0. Never quote it alone. A value
    near -1.0 means the waveform was recovered upside down, which points
    straight at the pixels-to-mV sign convention.
    """
    reference, test = align_length(reference, test)
    if reference.std() == 0 or test.std() == 0:
        return float("nan")
    return float(np.corrcoef(reference, test)[0, 1])


def gain_ratio(reference: np.ndarray, test: np.ndarray) -> float:
    """Least-squares amplitude scale of `test` relative to `reference`.

    The slope of ``test ~ a * reference + b``, which is the gain a perfect
    calibration would have made 1.0. Fitting an intercept as well means a
    baseline (DC) error does not contaminate the gain estimate -- the two
    calibration faults are separable and worth separating.

    Read alongside PCC: gain is only interpretable once the shape is right.
    """
    reference, test = align_length(reference, test)
    variance = float(np.var(reference))
    if variance == 0:
        return float("nan")
    covariance = float(np.mean(
        (reference - reference.mean()) * (test - test.mean())
    ))
    return covariance / variance


def evaluate_all(
    reference: np.ndarray,
    test: np.ndarray,
    fs: float,
    max_shift_s: float = 0.05,
) -> Metrics:
    """Score one reconstruction on all four metrics.

    `reference` and `test` must both be in millivolts and both sampled at
    `fs`; RMSE is meaningless otherwise. `max_shift_s` defaults to 50 ms,
    which at 100 Hz is 5 samples -- enough for edge-finding slop, far short
    of a cardiac cycle.
    """
    reference, test = align_length(reference, test)
    shifted, shift = shifted_snr_db(
        reference, test, max_shift_samples=int(round(max_shift_s * fs))
    )
    return Metrics(
        snr_db=snr_db(reference, test),
        shifted_snr_db=shifted,
        rmse_mv=rmse_mv(reference, test),
        pcc=pcc(reference, test),
        gain_ratio=gain_ratio(reference, test),
        best_shift_samples=shift,
        n_compared=len(reference),
    )
