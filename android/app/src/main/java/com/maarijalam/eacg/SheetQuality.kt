package com.maarijalam.eacg

import kotlin.math.abs

/**
 * Whole-sheet quality warnings (FR8): the port of `ecgproto/sheet_quality.py`,
 * golden-vector tested against it (SheetQualityPortTest).
 *
 * A wrong time reference silently mistimes every lead cut with it, while
 * each lead still looks like a plausible ECG, so these checks look at the
 * references the sheet was cut with -- the printed grid, the separator
 * ticks, the calibration pulses -- plus a physiological backstop per lead.
 * Each is anchored on something printed or physiological, never on the
 * pipeline agreeing with itself. The thresholds and the measurements behind
 * them are in the Python module's docstring.
 */
object SheetQuality {

    /** |tick scale / grid scale - 1| above this: the two printed rulers disagree. */
    const val SCALE_DISAGREE_FRACTION = 0.01
    /**
     * A tick further than this from the median of the same tick over the rows:
     * the threshold [SheetLayout.tickScale] rejects outliers with (reported
     * here as TICK_REPLACED); with only two rows, both are TICK_OUTLIER.
     */
    const val TICK_MATE_MM = SheetLayout.TICK_OUTLIER_MM
    /** Fewer ticks than this (of 9) found. */
    const val MIN_TICKS = 6
    /** A lead with less than this fraction of its samples traced. */
    const val MIN_LEAD_COVERAGE = 0.8
    /** quality.REJECT_PP_MV / QualityGate: the physiological amplitude band. */
    val REJECT_PP_MV = doubleArrayOf(0.05, 6.0)

    enum class Severity { WARN, SEVERE }

    class Warning(val code: String, val severity: Severity, val message: String,
                  val leads: List<String> = emptyList())

    /** "ok", "check" (any WARN) or "unreliable" (any SEVERE). */
    fun verdict(ws: List<Warning>): String = when {
        ws.any { it.severity == Severity.SEVERE } -> "unreliable"
        ws.isNotEmpty() -> "check"
        else -> "ok"
    }

    /** (fraction of samples traced, peak-to-peak mV); pp NaN if none traced. */
    fun leadStats(mv: DoubleArray): Pair<Double, Double> {
        var n = 0
        var lo = Double.POSITIVE_INFINITY
        var hi = Double.NEGATIVE_INFINITY
        for (v in mv) if (v.isFinite()) { n++; if (v < lo) lo = v; if (v > hi) hi = v }
        if (n == 0) return 0.0 to Double.NaN
        return n.toDouble() / mv.size to hi - lo
    }

    private fun tickLeads(row: Int, k: Int) =
        listOf(SheetLayout.LAYOUT_3X4[row][k - 1], SheetLayout.LAYOUT_3X4[row][k], "II_rhythm")

    private fun f1(v: Double) = String.format(java.util.Locale.US, "%.1f", v)

    /**
     * Warnings for one digitised sheet. [tickPositions]: (row, k, x) per tick
     * found; [leads]: name -> mV samples, the rhythm strip as "II_rhythm".
     */
    fun assess(gridRequested: Boolean, gridUsed: Boolean, gridPxPerMm: Double,
               calibration: String, pxPerMm: Double, ticksFound: Int,
               tickPositions: List<Triple<Int, Int, Double>>, t0Source: String,
               leads: Map<String, DoubleArray>,
               tickRejected: List<Triple<Int, Int, Double>> = emptyList(),
               rows: IntArray? = null,
               /**
                * `t0_guard_pct`: null unless the guarded time-zero fallback was USED
                * ([SheetLayout.guardTimeZero]); then the disagreement (%) that made it
                * fire, NaN if there was no tick scale. Always SEVERE: afterwards the
                * ticks agree with the grid by construction, and a time zero re-read
                * from the ticks can still be a whole cell out.
                */
               t0GuardPct: Double? = null): List<Warning> {
        val out = ArrayList<Warning>()
        if (rows != null && rows.size == 4 && SheetLayout.rowGapRatio(rows) > SheetLayout.ROW_SPACING_MAX_RATIO) {
            val s = rows.sortedArray()
            val g = DoubleArray(3) { (s[it + 1] - s[it]).toDouble() }
            val gaps = if (pxPerMm.isFinite() && pxPerMm > 0)
                g.joinToString(" / ") { String.format(java.util.Locale.US, "%.0f", it / pxPerMm) }
                else g.joinToString(" / ") { String.format(java.util.Locale.US, "%.0f px", it) }
            out += Warning("ROW_SPACING", Severity.SEVERE,
                "The four rows found are not evenly spaced (gaps $gaps mm); a shadow or a mark may have been " +
                    "taken for a row, so leads may be missing or under the wrong names.")
        }
        if (gridRequested && !gridUsed) out += Warning("GRID_REFUSED", Severity.WARN,
            "The printed grid could not be fitted; the page geometry comes from the four corners only.")
        if (calibration != "ticks") out += Warning("NO_TICK_SCALE", Severity.SEVERE,
            "Only $ticksFound of 9 lead-separator ticks found; the time scale is a fallback estimate.")
        else if (ticksFound < MIN_TICKS) out += Warning("FEW_TICKS", Severity.WARN,
            "Only $ticksFound of 9 lead-separator ticks found.")
        if (calibration == "ticks" && gridUsed && gridPxPerMm.isFinite() && gridPxPerMm > 0) {
            val d = pxPerMm / gridPxPerMm - 1.0
            if (abs(d) > SCALE_DISAGREE_FRACTION) out += Warning("SCALE_DISAGREE", Severity.SEVERE,
                "The ticks and the grid disagree on the page scale by ${if (d >= 0) "+" else ""}${f1(100 * d)}%.")
        }
        if (calibration == "ticks" && pxPerMm.isFinite() && pxPerMm > 0) {
            val pos = tickPositions.associate { (r, k, x) -> (r to k) to x }
            // sorted(pos.items()) in the Python: by row, then k.
            for ((rk, x) in pos.entries.sortedWith(compareBy({ it.key.first }, { it.key.second }))) {
                val (r, k) = rk
                // The median over the rows that found this tick, itself included;
                // two are their mean (both flagged past 2 mm apart); one is not judged.
                val same = (0 until 3).mapNotNull { pos[it to k] }
                if (same.size < 2) continue
                val offMm = (x - SheetLayout.median(same.toDoubleArray())) / pxPerMm
                if (abs(offMm) > TICK_MATE_MM) out += Warning("TICK_OUTLIER", Severity.SEVERE,
                    "Row ${r + 1}'s ${f1(2.5 * k)} s tick is ${if (offMm >= 0) "+" else ""}${f1(offMm)} mm " +
                        "from the same tick in the other rows.", tickLeads(r, k))
            }
        }
        if (calibration == "ticks" && pxPerMm.isFinite() && pxPerMm > 0) {
            val kept = tickPositions.associate { (r, k, x) -> (r to k) to x }
            for ((r, k, x) in tickRejected.sortedWith(compareBy({ it.first }, { it.second }, { it.third }))) {
                val mates = (0 until 3).mapNotNull { kept[it to k] }
                val off = if (mates.isEmpty()) "" else {
                    val mm = (x - mates.average()) / pxPerMm
                    " by ${if (mm >= 0) "+" else ""}${f1(mm)} mm"
                }
                out += Warning("TICK_REPLACED", Severity.WARN,
                    "Row ${r + 1}'s ${f1(2.5 * k)} s tick was out of line$off; " +
                        "the same tick in the other rows was used instead.", tickLeads(r, k))
            }
        }
        if (t0GuardPct != null) {
            val why = if (t0GuardPct.isFinite())
                "the ticks and the grid disagreed by ${String.format(java.util.Locale.US, "%+.1f", t0GuardPct)}%"
                else "no tick scale was found"
            out += Warning("T0_GUARD", Severity.SEVERE,
                "The calibration pulse was not trusted ($why); time zero was re-read from the " +
                    "lead-separator ticks. Check the timing on the page overlay.")
        }
        if (t0Source != "pulses" && t0Source != "ticks") out += Warning("NO_PULSE", Severity.WARN,
            "No calibration pulse found; time zero is estimated from the start of the rhythm strip.")
        for ((name, mv) in leads) {
            val (cov, pp) = leadStats(mv)
            if (cov < MIN_LEAD_COVERAGE) out += Warning("LEAD_COVERAGE", Severity.WARN,
                "$name: only ${Math.round(100 * cov)}% of the lead was traced.", listOf(name))
            else if (!(pp >= REJECT_PP_MV[0] && pp <= REJECT_PP_MV[1])) out += Warning("LEAD_AMPLITUDE",
                Severity.WARN, "$name: peak-to-peak ${String.format(java.util.Locale.US, "%.2f", pp)} mV " +
                    "is outside ${REJECT_PP_MV[0]}-${REJECT_PP_MV[1]} mV.", listOf(name))
        }
        return out
    }
}
