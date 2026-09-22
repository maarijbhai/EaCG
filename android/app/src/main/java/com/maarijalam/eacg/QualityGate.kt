package com.maarijalam.eacg

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * FR8: reject or flag a digitisation that fails a plausibility check.
 *
 * The port of the checks in `ecgproto/quality.py` that are computable on the
 * device. Its operating characteristics were measured on the desktop over
 * 171 rows (`results/gate/`): zero false rejects on 36 known-good strips,
 * and every injected failure either rejected or flagged.
 *
 * WHY THIS SHIPS WITH THE TRACER RATHER THAN AFTER IT
 * Three of the six bundled fixtures are 3x4 sheets, and
 * [TraceExtractor.digitise] handed a sheet returns a waveform spliced from
 * four leads that looks entirely plausible on a screen. Measured on the
 * desktop: PCC 0.273, with pitch correct to 0.00%, anisotropy 0.0000,
 * implied paper speed 26.6 mm/s and coverage 1.000. Every calibration
 * statistic clean. Without [tracesPerColumn] the app would draw that curve
 * and say nothing, which is worse than drawing nothing.
 *
 * EVERY CHECK IS ANCHORED OUTSIDE THE IMAGE
 * Not a stylistic choice. The pipeline's internal self-consistency
 * statistics were measured ranking a wrong answer above a right one:
 * `confidence` sat inside 0.45 to 0.64 across clean, rotated, blurred,
 * JPEG-crushed and outright failed estimates, and `spacing_cv` scored 0.412
 * on a pitch correct to -0.0% against 0.184 on one wrong by +399%. An
 * unresolved 1 mm grid is a *genuinely regular* comb of 5 mm rules, so
 * everything computed from that comb agrees it is excellent. Only a fact
 * from outside breaks the tie, so each check below names the external
 * quantity it rests on, and a check whose anchor is missing reports
 * [Verdict.UNAVAILABLE] rather than passing — a gate that silently skips a
 * test reads as a gate that passed it.
 */
object QualityGate {

    /**
     * Severity, ordered so the worst wins. [UNAVAILABLE] deliberately sorts
     * below [ACCEPT]: it means "not assessed" and must never contribute to
     * the overall verdict, only to the report of what was not checked.
     */
    enum class Verdict { UNAVAILABLE, ACCEPT, FLAG, REJECT }

    /** One plausibility test, its verdict, and the anchor it rests on. */
    class Check(
        val name: String,
        val verdict: Verdict,
        val anchor: String,
        val detail: String,
        val value: Double = Double.NaN
    )

    /** The gate's verdict on one digitisation, with everything that fed it. */
    class Report(val verdict: Verdict, val checks: List<Check>) {
        val usable: Boolean get() = verdict != Verdict.REJECT
        val rejections: List<String>
            get() = checks.filter { it.verdict == Verdict.REJECT }.map { it.detail }
        val flags: List<String>
            get() = checks.filter { it.verdict == Verdict.FLAG }.map { it.detail }
        val unavailable: List<String>
            get() = checks.filter { it.verdict == Verdict.UNAVAILABLE }.map { it.detail }

        fun check(name: String): Check =
            checks.firstOrNull { it.name == name } ?: error("no check '$name'")

        fun summary(): String = buildString {
            append(verdict.name)
            if (rejections.isNotEmpty()) append("  reject: ").append(rejections.joinToString("; "))
            if (flags.isNotEmpty()) append("  flag: ").append(flags.joinToString("; "))
        }
    }

    // ----------------------------------------------------------------------
    // thresholds. Values and anchors both match ecgproto/quality.py.
    // ----------------------------------------------------------------------

    /**
     * Plausible peak-to-peak amplitude, mV. Anchor: cardiac physiology.
     *
     * A normal QRS spans roughly 0.5 to 2.0 mV. The Sokolow-Lyon criterion
     * calls 3.5 mV *summed across two leads* pathological, so one lead beyond
     * that is already extreme, and 6 mV is past anything a heart produces. At
     * the floor, 0.05 mV is half a millimetre of printed deflection at
     * 10 mm/mV — flatter than any diagnostic trace. Measured on known-good
     * strip renders: 0.500 to 1.376 mV.
     */
    const val REJECT_PP_MV_LO = 0.05
    const val REJECT_PP_MV_HI = 6.0
    const val FLAG_PP_MV_LO = 0.20
    const val FLAG_PP_MV_HI = 3.5

    /**
     * Coverage. Anchor: a plotted trace is continuous by construction, so a
     * column without ink is a column whose ink was not *found*. Measured on
     * known-good renders: 0.996 to 1.000.
     */
    const val REJECT_COVERAGE = 0.50
    const val FLAG_COVERAGE = 0.90

    /**
     * Plausible paper speeds, mm/s. Anchor: what ECG machines print. 25 is
     * conventional and 50 the common alternative; the bounds admit both plus
     * honest estimation error, and are narrow enough that the 5x harmonic
     * failure (25 read as 5, or as 125) falls outside on either side.
     */
    const val PAPER_SPEED_LO = 15.0
    const val PAPER_SPEED_HI = 60.0

    /**
     * Largest accepted |x - y| / mean between the axis pitches. Anchor: ECG
     * paper is printed in square millimetres, so the two axes measure the
     * same physical unit. Mild perspective moves them a few percent apart; a
     * harmonic lock on one axis alone moves them by 60% or more.
     */
    const val MAX_ANISOTROPY = 0.15

    /**
     * Ink bands per column expected in a single strip. Anchor: the printed
     * layout convention — one lead region carries one trace. A 3x4 sheet
     * stacks three cell rows plus the rhythm strip, so a column crossing it
     * carries four. Measured 1.0 on every single-strip render at 4, 8 and
     * 16 px/mm, and 4.0 on a sheet.
     */
    const val EXPECTED_TRACES_PER_COLUMN = 1

    /**
     * Two traces printed closer together than this could not be told apart
     * by a clinician either. In millimetres, converted with the recovered
     * pitch, so the check does not change meaning with camera distance.
     *
     * Note the direction, which is the opposite of the first guess: a
     * *finer* pitch means more pixels per millimetre and therefore a *larger*
     * pixel threshold, which merges bands a coarser pitch resolves.
     */
    const val BAND_SEPARATION_MM = 1.0

    /** Columns are subsampled by this; the statistic is a median of many. */
    const val COLUMN_STRIDE = 7

    // ----------------------------------------------------------------------
    // observables
    // ----------------------------------------------------------------------

    /**
     * Median count of distinct ink bands down a column of the mask.
     *
     * The observable that separates a strip from a stacked sheet, and the
     * only one measured to catch a 3x4 printout fed to the tracer. A vertical
     * run of ink is one band however tall, so a steep QRS upstroke does not
     * inflate the count; a second *lead* above or below does, because the
     * white paper between them breaks the run.
     *
     * Median rather than mean, so that printed lead labels and speckle —
     * which add a band to the few columns they occupy — cannot drag a
     * single-strip image over the threshold.
     */
    fun tracesPerColumn(
        mask: TraceExtractor.Mask,
        pxPerMm: Double,
        bandSeparationMm: Double = BAND_SEPARATION_MM,
        columnStride: Int = COLUMN_STRIDE
    ): Double {
        if (mask.rows <= 0 || mask.cols <= 0) return Double.NaN
        val gapPx = if (pxPerMm.isFinite() && pxPerMm > 0.0) {
            max(2, (bandSeparationMm * pxPerMm).roundToInt())
        } else 2

        val counts = ArrayList<Int>()
        val stride = max(1, columnStride)
        var c = 0
        while (c < mask.cols) {
            var bands = 0
            var lastInkRow = -1
            for (r in 0 until mask.rows) {
                if (mask.data[r * mask.cols + c] == 0.toByte()) continue
                if (lastInkRow < 0 || r - lastInkRow > gapPx) bands++
                lastInkRow = r
            }
            if (bands > 0) counts.add(bands)
            c += stride
        }
        if (counts.isEmpty()) return Double.NaN
        counts.sort()
        val mid = counts.size / 2
        return if (counts.size % 2 == 1) counts[mid].toDouble()
        else 0.5 * (counts[mid - 1] + counts[mid])
    }

    /** Verdict for a value against an outer reject and inner flag band. */
    internal fun banded(
        value: Double,
        rejectLo: Double, rejectHi: Double,
        flagLo: Double, flagHi: Double
    ): Verdict = when {
        !value.isFinite() -> Verdict.REJECT
        value < rejectLo || value > rejectHi -> Verdict.REJECT
        value < flagLo || value > flagHi -> Verdict.FLAG
        else -> Verdict.ACCEPT
    }

    // ----------------------------------------------------------------------
    // the gate
    // ----------------------------------------------------------------------

    /**
     * Gate a completed digitisation.
     *
     * `expectedDurationS` is the one fact the paper-speed check needs from
     * outside the image, and it is normally **unavailable**: a photograph of
     * a stranger's ECG comes with nobody to say how long the strip was meant
     * to be. Pass null and the check reports itself unavailable rather than
     * faking a verdict. The desktop measurements record what that costs — a
     * 5x pitch error at PCC 0.005 is then only flagged, never rejected,
     * because no remaining externally anchored check can reject it.
     *
     * Two checks from the Python are absent here and both absences are
     * deliberate. **Implied gain** has no anchor in either language: mm/mV is
     * assumed, never recovered, so an implied gain is that assumption handed
     * back, and the 1 mV calibration pulse that would anchor it is not
     * detected yet. **Implied heart rate** is not ported: it depends on
     * SciPy's `find_peaks` distance filtering, whose tie-breaking by peak
     * height is fiddly to reproduce exactly, and it was measured
     * contributing one flag in twelve while false-flagging 8.3% of
     * known-good input — on a record where the *ground truth* waveform
     * produced the same value, so the flag was about the signal's morphology
     * rather than the digitisation. Not worth a porting hazard.
     */
    fun assess(
        result: TraceExtractor.Digitised,
        mask: TraceExtractor.Mask?,
        pxPerMmX: Double,
        pxPerMmY: Double,
        imageWidthPx: Int,
        expectedDurationS: Double? = null,
        mmPerS: Double = 25.0,
        expectedTraces: Int = EXPECTED_TRACES_PER_COLUMN,
        axesIndependentlyMeasured: Boolean = true
    ): Report {
        val checks = ArrayList<Check>()

        // --- 1. the axes resolved at all, and agree ----------------------
        val mean = 0.5 * (pxPerMmX + pxPerMmY)
        val anisotropy =
            if (mean > 0.0) abs(pxPerMmX - pxPerMmY) / mean else Double.POSITIVE_INFINITY
        if (!axesIndependentlyMeasured) {
            // `GridEstimator` on this device projects onto columns only and
            // returns a single pitch, so the vertical scale is *assumed*
            // equal to the horizontal rather than measured. Anisotropy is
            // then identically zero by construction and would report a
            // perfectly square grid for a photograph taken at any angle,
            // which is the most dangerous kind of clean bill of health. The
            // Python measures both axes; until the Kotlin does, this says so.
            checks.add(Check(
                "grid", Verdict.UNAVAILABLE,
                "printed grid is square by manufacture",
                "only one axis is measured, so anisotropy cannot be computed; " +
                    "perspective distortion would pass unseen",
                Double.NaN
            ))
        } else if (!pxPerMmX.isFinite() || !pxPerMmY.isFinite()) {
            checks.add(Check(
                "grid", Verdict.REJECT,
                "printed grid is square by manufacture",
                "an axis failed to resolve (x=$pxPerMmX, y=$pxPerMmY)",
                anisotropy
            ))
        } else {
            checks.add(Check(
                "grid",
                if (anisotropy > MAX_ANISOTROPY) Verdict.REJECT else Verdict.ACCEPT,
                "printed grid is square by manufacture",
                if (anisotropy > MAX_ANISOTROPY)
                    "anisotropy %.3f exceeds %.3f (x=%.3f, y=%.3f px/mm) -- perspective, or one axis locked onto a harmonic"
                        .format(anisotropy, MAX_ANISOTROPY, pxPerMmX, pxPerMmY)
                else "anisotropy %.4f".format(anisotropy),
                anisotropy
            ))
        }

        // --- 2. implied paper speed --------------------------------------
        if (expectedDurationS != null && expectedDurationS > 0.0 && pxPerMmX.isFinite()) {
            val implied = imageWidthPx / (pxPerMmX * expectedDurationS)
            val bad = implied < PAPER_SPEED_LO || implied > PAPER_SPEED_HI
            checks.add(Check(
                "paper_speed", if (bad) Verdict.REJECT else Verdict.ACCEPT,
                "clinical paper speed 15-60 mm/s",
                if (bad)
                    "implied paper speed %.1f mm/s outside [%.0f, %.0f] -- pitch is probably wrong by an integer factor (nominal %.0f mm/s)"
                        .format(implied, PAPER_SPEED_LO, PAPER_SPEED_HI, mmPerS)
                else "implied %.1f mm/s within range".format(implied),
                implied
            ))
        } else {
            checks.add(Check(
                "paper_speed", Verdict.UNAVAILABLE,
                "clinical paper speed 15-60 mm/s",
                "no expected duration supplied, so an integer-factor pitch error cannot be detected"
            ))
        }

        // --- 3. amplitude -------------------------------------------------
        var lo = Double.POSITIVE_INFINITY
        var hi = Double.NEGATIVE_INFINITY
        var finite = 0
        for (v in result.signalMv) {
            if (!v.isFinite()) continue
            finite++
            if (v < lo) lo = v
            if (v > hi) hi = v
        }
        val pp = if (finite >= 2) hi - lo else Double.NaN
        checks.add(Check(
            "amplitude_pp_mv",
            banded(pp, REJECT_PP_MV_LO, REJECT_PP_MV_HI, FLAG_PP_MV_LO, FLAG_PP_MV_HI),
            "cardiac physiology: QRS 0.5-2 mV, Sokolow-Lyon 3.5 mV pathological",
            if (pp.isFinite()) "peak-to-peak %.3f mV".format(pp)
            else "peak-to-peak not computable",
            pp
        ))

        // --- 4. coverage --------------------------------------------------
        val cov = result.coverage
        checks.add(Check(
            "coverage",
            when {
                !cov.isFinite() || cov < REJECT_COVERAGE -> Verdict.REJECT
                cov < FLAG_COVERAGE -> Verdict.FLAG
                else -> Verdict.ACCEPT
            },
            "a plotted trace is continuous by construction",
            "%.3f of columns carried ink".format(cov),
            cov
        ))

        // --- 5. trace multiplicity ---------------------------------------
        if (mask == null) {
            checks.add(Check(
                "traces_per_column", Verdict.UNAVAILABLE,
                "printed layout: one lead region carries one trace",
                "no trace mask supplied; stacked-layout check could not run"
            ))
        } else {
            val bands = tracesPerColumn(mask, pxPerMmY)
            checks.add(Check(
                "traces_per_column",
                when {
                    !bands.isFinite() -> Verdict.REJECT
                    bands >= expectedTraces + 1 -> Verdict.REJECT
                    else -> Verdict.ACCEPT
                },
                "printed layout: one lead region carries one trace",
                when {
                    !bands.isFinite() -> "no ink found in any column"
                    bands >= expectedTraces + 1 ->
                        "%.1f ink bands per column, expected %d -- more than one lead in frame; this is not a single strip"
                            .format(bands, expectedTraces)
                    else -> "%.1f ink bands per column".format(bands)
                },
                bands
            ))
        }

        // --- 6. the anchor that does not exist yet -------------------------
        checks.add(Check(
            "implied_gain", Verdict.UNAVAILABLE,
            "calibration pulse of known 1 mV amplitude, printed on the paper",
            "no external gain anchor: mm/mV is assumed, not recovered"
        ))

        var worst = Verdict.ACCEPT
        for (c in checks) if (c.verdict > worst) worst = c.verdict
        return Report(worst, checks)
    }
}
