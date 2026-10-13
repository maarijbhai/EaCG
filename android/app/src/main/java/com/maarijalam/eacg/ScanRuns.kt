package com.maarijalam.eacg

import java.util.Locale

/**
 * What one Digitise run is given -- everything that decides its result
 * besides the photograph itself, which [ScanController] passes as the same
 * object to every run of a tap. Pure JVM code (ScanRunsTest).
 */
data class RunSpec(
    val tracer: SheetPipeline.Tracer,
    val corners: List<Double>,
    val cornersAuto: Boolean,
    val cornersRefined: Boolean?,
    val geometry: SheetPipeline.Geometry,
    /** Benchmark target name (its truth is loaded once per tap), or null. */
    val target: String?,
)

/**
 * The runs of one Digitise tap. Normally one, with the chosen tracer. In the
 * both-tracers testing mode (prototype mode only) two: the LEARNED tracer
 * first -- the run the latency results use -- then the CLASSICAL one, each
 * otherwise identical to a normal tap with that tracer: same corners, grid
 * fit and target. Nothing else differs, so each run's result is what a
 * separate tap would have produced.
 */
object ScanRuns {
    fun plan(base: RunSpec, bothTracers: Boolean): List<RunSpec> =
        if (!bothTracers) listOf(base)
        else listOf(base.copy(tracer = SheetPipeline.Tracer.MODEL), base.copy(tracer = SheetPipeline.Tracer.CLASSICAL))

    /** "Run 1 of 2: learned" above the stage list; null for a single run. */
    fun label(index: Int, count: Int, tracer: SheetPipeline.Tracer): String? =
        if (count < 2) null
        else "Run ${index + 1} of $count: " + if (tracer == SheetPipeline.Tracer.MODEL) "learned" else "classical"

    fun tracerName(t: SheetPipeline.Tracer) = t.name.lowercase(Locale.US)
}
