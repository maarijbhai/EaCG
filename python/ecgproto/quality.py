"""FR8: reject or flag a digitisation that fails a plausibility check.

This is the gate that sits at the *end* of the pipeline, not inside
calibration. `digitise.check_calibration` already guards the recovered
grid pitch (FR3/FR7) and is reused here as one check among several, but
it is not sufficient for FR8 and measurement says so plainly: a 3x4
printout fed to `digitise` returns pitch correct to 0.00%, an implied
paper speed of 26.6 mm/s, isotropy 0.0000 and coverage 1.000 -- every
calibration statistic correct -- and a waveform at PCC 0.334. The
calibration was not wrong. The digitisation was.

**Every check here is anchored on a quantity known from outside the
image.** That constraint is not stylistic. Audit finding B1 established,
and `digitise.GridEstimate` records at length, that the pipeline's
internal self-consistency statistics are worse than useless for this job:
`confidence` sat inside 0.45-0.64 across clean, rotated, blurred,
JPEG-crushed *and outright failed* estimates; `spacing_cv` scored 0.412
on a pitch correct to -0.0% and 0.184 on one wrong by +399%, ranking the
wrong answer higher. An unresolved 1 mm grid yields a genuinely regular
comb of 5 mm rules, so every statistic computed from that comb agrees it
is excellent. Only an external fact breaks the tie.

Each `Check` therefore carries the `anchor` it rests on, and a check
whose anchor is unavailable reports `available=False` rather than
silently passing -- so the gate's blind spots appear in the results
table instead of being mistaken for clean bills of health. Two are
unavailable today and both are blocked on unbuilt stages, not on
tuning: see `GAIN_ANCHOR_BLOCKED` and `REGION_ANCHOR_BLOCKED`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import signal as sps

from . import digitise as dig


class Verdict(Enum):
    """FR8's three outcomes, ordered by severity so `max` combines them.

    The ordering is the whole point of using an enum with a value: a
    report's verdict is the worst verdict any single check returned, and
    that has to be computable without a chain of conditionals that can be
    got subtly wrong.
    """

    ACCEPT = 0
    FLAG = 1
    REJECT = 2

    def __lt__(self, other: "Verdict") -> bool:
        return self.value < other.value


# --------------------------------------------------------------------------
# thresholds, each with the external quantity it is anchored on
# --------------------------------------------------------------------------

#: Peak-to-peak amplitude, mV, of a plausible diagnostic ECG lead.
#: Anchored on cardiac physiology, not on this pipeline's behaviour.
#: A normal QRS spans roughly 0.5-2.0 mV; the Sokolow-Lyon voltage
#: criterion calls 3.5 mV *summed across two leads* pathological, so a
#: single lead above that is already extreme and 6 mV is beyond anything
#: a heart produces. At the bottom, 0.05 mV is half a millimetre of
#: printed deflection at 10 mm/mV -- flatter than any diagnostic trace.
#: Measured on known-good single-strip renders: 0.500 to 1.376 mV.
REJECT_PP_MV = (0.05, 6.0)
FLAG_PP_MV = (0.20, 3.5)

#: Implied heart rate, bpm. Anchored on human cardiac physiology.
#: **Deliberately wide, and a backstop rather than a measurement.** The
#: detector below is a crude absolute-value peak finder and was measured
#: over-reading by up to x1.73 on a known 95.2 bpm record (it counts T
#: waves). The bounds are therefore set so that the detector's own error
#: cannot on its own cause a reject, while a factor-of-five time-axis
#: error must. Its value is that it is the *only* time-axis anchor that
#: survives when the recording duration is unknown -- which is the normal
#: case for a photograph of a stranger's ECG, where `expected_duration_s`
#: cannot be supplied and the paper-speed check is skipped entirely.
REJECT_BPM = (25.0, 300.0)
FLAG_BPM = (35.0, 220.0)

#: Fraction of columns that carried ink. Anchored on the printer: a
#: plotted trace is continuous by construction, so a column without ink
#: is a column whose ink was not *found*. Measured on known-good renders:
#: 0.996 to 1.000.
REJECT_COVERAGE = 0.50
FLAG_COVERAGE = 0.90

#: Ink bands per column expected in a single continuous strip. Anchored
#: on the printed layout convention: one lead region carries one trace.
#: A 3x4 sheet stacks three cell rows plus the rhythm strip, so a column
#: crossing it carries four. Measured: 1.0 on every single-strip render
#: at 4, 8 and 16 px/mm; 4.0 on a rendered 3x4 printout.
EXPECTED_TRACES_PER_COLUMN = 1

#: Two traces printed closer together than this could not be read apart
#: by a clinician either, so it is the smallest separation worth calling
#: two bands. Expressed in millimetres and converted with the recovered
#: pitch, so the check does not change meaning with resolution.
BAND_SEPARATION_MM = 1.0

#: Why the gain check cannot be anchored yet. The conventional 10 mm/mV
#: is *assumed* by `digitise`, never recovered, so an "implied gain"
#: computed from it would be that assumption handed back -- an internal
#: self-consistency statistic of exactly the kind this module refuses.
#: The external anchor exists on the paper: printouts carry a
#: calibration pulse of known 1 mV amplitude. Reading it is unbuilt.
GAIN_ANCHOR_BLOCKED = (
    "no external gain anchor: mm/mV is assumed, not recovered; the "
    "calibration pulse that would anchor it is not detected yet"
)

#: Why the per-region duration check cannot be anchored yet. A 3x4
#: printout gives each lead region 2.5 s, which is a strong external
#: anchor -- but computing a region's duration requires its width, and
#: cell boundaries come from lead segmentation, which does not exist.
#: The whole-sheet analogue *is* available and is what the paper-speed
#: check inside `check_calibration` already uses.
REGION_ANCHOR_BLOCKED = (
    "no external per-region duration anchor: lead segmentation is not "
    "built, so a lead region's width in pixels is unknown"
)


@dataclass(frozen=True)
class Check:
    """One plausibility test, its verdict, and what it rests on."""

    name: str
    verdict: Verdict
    anchor: str                     # the external quantity, in words
    detail: str                     # human-readable outcome
    value: float = float("nan")     # the quantity compared, for the CSV
    available: bool = True          # False => anchor missing, no verdict

    @property
    def informative(self) -> bool:
        return self.available


@dataclass(frozen=True)
class QualityReport:
    """FR8's verdict on one digitisation, with every check that fed it."""

    verdict: Verdict
    checks: tuple[Check, ...] = field(default_factory=tuple)

    def __bool__(self) -> bool:
        """True unless rejected. A flagged digitisation is usable."""
        return self.verdict is not Verdict.REJECT

    @property
    def rejections(self) -> tuple[str, ...]:
        return tuple(c.detail for c in self.checks
                     if c.verdict is Verdict.REJECT and c.available)

    @property
    def flags(self) -> tuple[str, ...]:
        return tuple(c.detail for c in self.checks
                     if c.verdict is Verdict.FLAG and c.available)

    @property
    def unavailable(self) -> tuple[str, ...]:
        """Checks that could not run. The gate's declared blind spots."""
        return tuple(c.detail for c in self.checks if not c.available)

    def check(self, name: str) -> Check:
        for c in self.checks:
            if c.name == name:
                return c
        raise KeyError(name)

    def summary(self) -> str:
        parts = [self.verdict.name]
        if self.rejections:
            parts.append("reject: " + "; ".join(self.rejections))
        if self.flags:
            parts.append("flag: " + "; ".join(self.flags))
        return " | ".join(parts)


# --------------------------------------------------------------------------
# observables
# --------------------------------------------------------------------------

def implied_bpm(signal_mv: np.ndarray, fs: float) -> float:
    """Crude heart rate from the recovered waveform, bpm.

    Not a QRS detector and not offered as one. It rectifies about the
    median and takes peaks at least 0.24 s apart -- 250 bpm, above any
    sinus rhythm -- which counts tall T waves as beats and was measured
    over-reading a 95.2 bpm record as 164.4. `REJECT_BPM` is set wide
    enough that this cannot by itself reject a good digitisation; the
    check exists to notice a time axis wrong by an integer factor, where
    the error is 400%, not 73%.
    """
    x = np.asarray(signal_mv, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2 or not np.isfinite(fs) or fs <= 0:
        return float("nan")
    x = np.abs(x - np.median(x))
    height = 0.4 * float(np.percentile(x, 98))
    if not np.isfinite(height) or height <= 0:
        return float("nan")
    peaks, _ = sps.find_peaks(x, height=height,
                              distance=max(1, int(0.24 * fs)))
    if peaks.size < 2:
        return float("nan")
    median_rr_s = float(np.median(np.diff(peaks))) / fs
    if median_rr_s <= 0:
        return float("nan")
    return 60.0 / median_rr_s


def traces_per_column(
    mask: np.ndarray,
    px_per_mm: float,
    band_separation_mm: float = BAND_SEPARATION_MM,
    column_stride: int = 7,
) -> float:
    """Median count of distinct ink bands down a column of the mask.

    The observable that distinguishes a single strip from a stacked
    printout, and the only one measured to catch a 3x4 sheet fed to
    `digitise`. A vertical run of ink is one band however tall it is, so
    a steep QRS upstroke does not inflate the count; a second *lead*
    printed above or below does, because the white paper between them
    breaks the run.

    The median rather than the mean, so that printed lead labels and
    speckle -- which add a band to the few columns they occupy -- cannot
    drag a single-strip image over the threshold.

    `column_stride` subsamples columns because the statistic is a median
    over thousands of near-identical measurements and does not need all
    of them; it makes the check cheap enough to run unconditionally.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.size == 0:
        return float("nan")
    gap_px = max(2, int(round(band_separation_mm * px_per_mm))) \
        if np.isfinite(px_per_mm) and px_per_mm > 0 else 2

    counts: list[int] = []
    for col in range(0, mask.shape[1], max(1, column_stride)):
        rows = np.flatnonzero(mask[:, col])
        if rows.size == 0:
            continue
        counts.append(1 + int(np.count_nonzero(np.diff(rows) > gap_px)))
    return float(np.median(counts)) if counts else float("nan")


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------

def _banded(value: float, reject: tuple[float, float],
            flag: tuple[float, float]) -> Verdict:
    """Verdict for a value against an outer reject and inner flag band."""
    if not np.isfinite(value):
        return Verdict.REJECT
    if value < reject[0] or value > reject[1]:
        return Verdict.REJECT
    if value < flag[0] or value > flag[1]:
        return Verdict.FLAG
    return Verdict.ACCEPT


def assess(
    result: dig.DigitiseResult,
    mask: np.ndarray | None = None,
    expected_traces: int = EXPECTED_TRACES_PER_COLUMN,
    reject_pp_mv: tuple[float, float] = REJECT_PP_MV,
    flag_pp_mv: tuple[float, float] = FLAG_PP_MV,
    reject_bpm: tuple[float, float] = REJECT_BPM,
    flag_bpm: tuple[float, float] = FLAG_BPM,
    reject_coverage: float = REJECT_COVERAGE,
    flag_coverage: float = FLAG_COVERAGE,
) -> QualityReport:
    """Gate a completed digitisation. FR8.

    `result` is what `digitise` returned -- call it with `strict=False`
    if you want the gate rather than an exception to be the thing that
    refuses, which is the point of having a gate.

    `mask` is the trace mask `binarise_trace` produced. It is optional
    only because a caller may not have kept it; without it the
    trace-multiplicity check cannot run and says so, and the 3x4 failure
    mode goes uncaught by that route. Pass it.

    The returned verdict is the worst of the individual checks. Checks
    whose anchor is unavailable contribute nothing to it.
    """
    checks: list[Check] = []

    # --- 1. calibration, delegated ------------------------------------
    # Reuses the FR3/FR7 guard rather than restating it. Its reasons are
    # rejections and its warnings are flags, which is the mapping its own
    # docstrings already describe: the warnings are things worth logging
    # and not worth refusing.
    cal = result.calibration
    if cal is None:
        checks.append(Check(
            "calibration", Verdict.ACCEPT,
            anchor="clinical paper-speed range; square geometry of printed paper",
            detail="calibration check not run by digitise()",
            available=False,
        ))
    else:
        if cal.reasons:
            verdict = Verdict.REJECT
            detail = "; ".join(cal.reasons)
        elif cal.warnings:
            verdict = Verdict.FLAG
            detail = "; ".join(cal.warnings)
        else:
            verdict = Verdict.ACCEPT
            detail = f"implied {cal.implied_mm_per_s:.1f} mm/s within range"
        checks.append(Check(
            "calibration", verdict,
            anchor="clinical paper-speed range 15-60 mm/s; printed grid is square",
            detail=detail,
            value=float(cal.implied_mm_per_s),
        ))

    # --- 2. amplitude -------------------------------------------------
    finite = result.signal_mv[np.isfinite(result.signal_mv)]
    pp = float(np.ptp(finite)) if finite.size >= 2 else float("nan")
    checks.append(Check(
        "amplitude_pp_mv", _banded(pp, reject_pp_mv, flag_pp_mv),
        anchor="cardiac physiology: QRS 0.5-2 mV, Sokolow-Lyon 3.5 mV pathological",
        detail=(f"peak-to-peak {pp:.3f} mV"
                if np.isfinite(pp) else "peak-to-peak not computable"),
        value=pp,
    ))

    # --- 3. heart rate ------------------------------------------------
    bpm = implied_bpm(result.signal_mv, result.fs)
    if np.isfinite(bpm):
        hr_verdict = _banded(bpm, reject_bpm, flag_bpm)
        hr_detail = f"implied {bpm:.1f} bpm"
        hr_available = True
    else:
        # No detectable beats is not evidence of a bad scale, so it must
        # not reject on this check's account -- the amplitude and
        # coverage checks are the ones that see a waveform this degraded.
        hr_verdict = Verdict.ACCEPT
        hr_detail = "no beats detected; rate check could not run"
        hr_available = False
    checks.append(Check(
        "heart_rate_bpm", hr_verdict,
        anchor="human cardiac physiology, 25-300 bpm",
        detail=hr_detail, value=bpm, available=hr_available,
    ))

    # --- 4. coverage --------------------------------------------------
    cov = float(result.coverage)
    cov_verdict = (Verdict.REJECT if not np.isfinite(cov) or cov < reject_coverage
                   else Verdict.FLAG if cov < flag_coverage
                   else Verdict.ACCEPT)
    checks.append(Check(
        "coverage", cov_verdict,
        anchor="a plotted trace is continuous by construction",
        detail=f"{cov:.3f} of columns carried ink",
        value=cov,
    ))

    # --- 5. trace multiplicity ----------------------------------------
    if mask is None:
        checks.append(Check(
            "traces_per_column", Verdict.ACCEPT,
            anchor="printed layout: one lead region carries one trace",
            detail="no trace mask supplied; stacked-layout check could not run",
            available=False,
        ))
    else:
        bands = traces_per_column(mask, result.grid.px_per_mm_y)
        if not np.isfinite(bands):
            tm_verdict, tm_detail = Verdict.REJECT, "no ink found in any column"
        elif bands >= expected_traces + 1:
            tm_verdict = Verdict.REJECT
            tm_detail = (f"{bands:.1f} ink bands per column, expected "
                         f"{expected_traces} -- more than one lead in frame; "
                         "this is not a single strip")
        else:
            tm_verdict = Verdict.ACCEPT
            tm_detail = f"{bands:.1f} ink bands per column"
        checks.append(Check(
            "traces_per_column", tm_verdict,
            anchor="printed layout: one lead region carries one trace",
            detail=tm_detail, value=bands,
        ))

    # --- 6, 7. the anchors that do not exist yet ----------------------
    # Reported, not omitted. A gate that silently skips a test it cannot
    # perform reads as a gate that passed it.
    checks.append(Check(
        "implied_gain", Verdict.ACCEPT,
        anchor="calibration pulse of known 1 mV amplitude, printed on the paper",
        detail=GAIN_ANCHOR_BLOCKED, available=False,
    ))
    checks.append(Check(
        "region_duration", Verdict.ACCEPT,
        anchor="3x4 layout convention: 2.5 s per lead region",
        detail=REGION_ANCHOR_BLOCKED, available=False,
    ))

    verdict = Verdict.ACCEPT
    for c in checks:
        if c.available and c.verdict.value > verdict.value:
            verdict = c.verdict
    return QualityReport(verdict=verdict, checks=tuple(checks))
