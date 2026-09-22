package com.maarijalam.eacg

import android.graphics.Bitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * FR3 to FR8 in one pass, timed per stage. The first point at which this app
 * produces a waveform rather than a geometric measurement.
 *
 * Deliberately shaped like [DeskewPipeline], including the split between what
 * a deployed capture would pay and what only the verification harness pays.
 * That distinction is the one CLAUDE.md limitation 9 exists for, and it bites
 * again here for a different reason: scoring the reconstruction against a
 * committed source waveform is **scaffolding**. A deployed app digitising a
 * stranger's ECG has no ground truth to compare against and pays nothing for
 * fidelity scoring, so [Timings.pipelineMs] excludes it and
 * [Timings.harnessMs] holds it.
 *
 * Deskew is NOT run here. It is a separate stage with its own screen and its
 * own ground-truth mechanism, and folding it in would make the pipeline total
 * depend on whether a rotation had been injected. Feed this an image that is
 * already square to the page — which the bundled fixtures are.
 */
object DigitisePipeline {

    /** Everything one pass produces. */
    class Run(
        val grid: GridEstimator.GridEstimate,
        /** Null when the tracer refused, with [error] saying why. */
        val digitised: TraceExtractor.Digitised?,
        /** Null only when the tracer refused before the gate could run. */
        val gate: QualityGate.Report?,
        /**
         * Reconstruction fidelity, or null when no ground truth was
         * available — which is every input except the bundled strips.
         */
        val fidelity: Fidelity.Scores?,
        /** Set when a stage threw rather than returning. */
        val error: String?,
        /** Ink bands per column, carried out so the UI can show the reason. */
        val tracesPerColumn: Double,
        val timings: Timings
    )

    /**
     * Wall clock per stage.
     *
     * Split rather than totalled for the same reason [DeskewPipeline.Timings]
     * is: the stages differ by an order of magnitude and answer different
     * questions. Binarisation is a full-resolution per-pixel OpenCV pass and
     * scales with the input; the centroid walk is one pass over the same
     * pixels in Kotlin; the gate is a strided pass over a fraction of them.
     * Which of those dominates on real hardware is exactly the sort of thing
     * the deployment characterisation is for, and a single total would hide
     * it.
     */
    class Timings(
        val toMatMs: Long,
        val gridMs: Long,
        val binariseMs: Long,
        val maskCopyMs: Long,
        val traceMs: Long,
        val gateMs: Long,
        val fidelityMs: Long,
        val totalMs: Long
    ) {
        /**
         * What a real capture would pay. Everything but the fidelity scoring,
         * which needs a ground truth no deployed capture has.
         *
         * `toMatMs` is charged here rather than to the harness: a real capture
         * still has to get its pixels into a Mat once, even though in a
         * finished app that conversion would happen upstream at capture time
         * and be shared with every later stage.
         */
        val pipelineMs: Long
            get() = toMatMs + gridMs + binariseMs + maskCopyMs + traceMs + gateMs

        /** Scaffolding: scoring the answer against truth we happen to have. */
        val harnessMs: Long get() = fidelityMs
    }

    /**
     * Run FR3 to FR8 on [source].
     *
     * [truth] is the source waveform the image was rendered from, when one is
     * committed alongside the fixture, and null otherwise. [expectedDurationS]
     * is the printed recording length, which is the one external fact the
     * gate's paper-speed check needs; pass null when it is genuinely unknown
     * — a photograph of someone else's ECG — and the gate will report that
     * check unavailable rather than inventing a verdict.
     *
     * Never throws for a bad image. A refusal is a result and is reported as
     * one, because the question "what does this pipeline do with input it
     * cannot handle?" is a measurement, not an incident.
     */
    suspend fun run(
        source: Bitmap,
        truth: DoubleArray? = null,
        truthFs: Double = 100.0,
        expectedDurationS: Double? = null,
        mmPerS: Double = 25.0,
        mmPerMv: Double = 10.0
    ): Run = withContext(Dispatchers.Default) {
        val t0 = System.nanoTime()
        val mat = ImageWarp.toMat(source)
        val tMat = System.nanoTime()
        try {
            val grid = GridEstimator.estimate(source)
            val tGrid = System.nanoTime()

            // One axis, used for both. `GridEstimator` projects onto columns
            // only and returns a single pitch, so the vertical scale is an
            // ASSUMPTION of isotropy rather than a measurement — which the
            // gate is told about below so it reports the anisotropy check
            // unavailable instead of returning a flattering zero. The Python
            // measures both axes; this is a gap in the port, not in the
            // algorithm.
            val pxPerMm = grid.pxPerMm

            val maskMat = TraceExtractor.binarise(mat)
            val tBinarise = System.nanoTime()
            try {
                val mask = TraceExtractor.maskOf(maskMat)
                val tMaskCopy = System.nanoTime()

                // The tracer refuses some inputs outright — no ink, or a
                // single inked column. That is a legitimate outcome and is
                // reported rather than raised past, so the UI can say which
                // stage declined and why.
                val digitised = try {
                    TraceExtractor.digitise(
                        mask,
                        pxPerMmX = pxPerMm,
                        pxPerMmY = pxPerMm,
                        fsOut = truthFs,
                        mmPerS = mmPerS,
                        mmPerMv = mmPerMv
                    )
                } catch (e: Exception) {
                    null
                }
                val tTrace = System.nanoTime()

                val gate = digitised?.let {
                    QualityGate.assess(
                        it, mask,
                        pxPerMmX = pxPerMm,
                        pxPerMmY = pxPerMm,
                        imageWidthPx = source.width,
                        expectedDurationS = expectedDurationS,
                        mmPerS = mmPerS,
                        axesIndependentlyMeasured = false
                    )
                }
                val tGate = System.nanoTime()

                // Scored even when the gate rejected. A rejected waveform's
                // PCC is the whole point of measuring a gate: a verdict
                // without the consequence beside it says nothing about
                // whether refusing was the right call.
                val fidelity = if (digitised != null && truth != null) {
                    try {
                        Fidelity.score(truth, digitised.signalMv)
                    } catch (e: Exception) {
                        null
                    }
                } else null
                val tFid = System.nanoTime()

                Run(
                    grid = grid,
                    digitised = digitised,
                    gate = gate,
                    fidelity = fidelity,
                    error = if (digitised == null)
                        "the tracer found no usable trace in this image" else null,
                    tracesPerColumn = QualityGate.tracesPerColumn(mask, pxPerMm),
                    timings = Timings(
                        toMatMs = (tMat - t0) / 1_000_000,
                        gridMs = (tGrid - tMat) / 1_000_000,
                        binariseMs = (tBinarise - tGrid) / 1_000_000,
                        maskCopyMs = (tMaskCopy - tBinarise) / 1_000_000,
                        traceMs = (tTrace - tMaskCopy) / 1_000_000,
                        gateMs = (tGate - tTrace) / 1_000_000,
                        fidelityMs = (tFid - tGate) / 1_000_000,
                        totalMs = (tFid - t0) / 1_000_000
                    )
                )
            } finally {
                maskMat.release()
            }
        } finally {
            mat.release()
        }
    }
}
