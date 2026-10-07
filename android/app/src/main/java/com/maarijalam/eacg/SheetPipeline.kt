package com.maarijalam.eacg

import android.content.Context
import android.graphics.Bitmap
import android.os.Debug
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.hypot
import kotlin.math.roundToInt

/**
 * The whole 3x4 + rhythm sheet, photograph to 12 leads, on the phone: the
 * port of `ecgproto.sheet.digitise_sheet`, step for step.
 *
 *   corners    [SheetCorners.find] (no clicks) unless the user moved them
 *   rectify    output size from the corners alone (as PhotoPipeline)
 *   locate     [SheetLayout.sheetRows], the rhythm locator, pulse tops
 *   calibrate  [SheetLayout.tickScale]; the duration anchor if < 3 ticks
 *   segment    [SheetLayout.segmentLeads] with the tick knots
 *   trace      per row: the learned tracer (b3, LiteRT, tiled) and/or the
 *              classical one, the control
 *   cut        [SheetLayout.onSheetColumns] + [SheetLayout.cellSignal]
 *
 * The desktop reference with the same model file scores lead PCC 0.85-0.95
 * (median of 12) on the six PTB-XL 40 photographs
 * (results/ml/phone_path_b3_512x1024_fp16.csv); the phone should match it.
 */
object SheetPipeline {

    const val MODEL_ASSET = "ml/b3_512x1024_fp16.tflite"
    const val LEAD_IN_MM = 4.0
    /**
     * segment.LOCATE_OPEN_KERNEL. Was 3: a 3x3 opening shreds a 2-3 px
     * photographed trace, and on ptbxl9_20261005_01 the phone read 6.52 px/mm
     * from 3/9 ticks and lost every lead (results/device/sheet_scores.csv).
     * Now the same 2x2 mask as the ticks, binarised once.
     */
    const val OPEN_LOCATE = 2
    const val OPEN_TRACE = 2       // the photo path's opening (PhotoPipeline.OPEN_KERNEL)

    enum class Tracer { MODEL, CLASSICAL }

    class LeadResult(val name: String, val rhythm: Boolean, val tStartS: Double,
                     val mv: DoubleArray, val pcc: Double?, val rmseMv: Double?)

    class Result(
        val tracer: Tracer,
        val corners: DoubleArray,
        val cornersAuto: Boolean,
        val cornersRefined: Boolean?,
        val rectW: Int, val rectH: Int,
        val pxPerMm: Double,
        val calibration: String,
        val ticksFound: Int,
        val t0Source: String,
        val leads: List<LeadResult>,
        val timingsMs: Map<String, Long>,
        val rectPreview: Bitmap?,
        /** The rectified page at full resolution, as JPEG (for Records). */
        val pageJpeg: ByteArray? = null,
        /** Per row 0-3: the traced row positions on the page, one per column from x0. */
        val rowTraces: List<RowTrace> = emptyList(),
        /** "grid" when the page was regularised on its printed grid, else "corners". */
        val geometry: String = "corners",
        /** The grid fit, when used. */
        val gridFit: GridFit.Fit? = null,
        /** Why the grid fit was not used, when it was asked for and refused. */
        val gridNote: String? = null,
        /** Memory before the first stage, and the peak of each stage (StageMemory). */
        val memoryBaseline: StageMemory.Reading? = null,
        val memoryPeaks: Map<String, StageMemory.Peak> = emptyMap(),
        /** The process's total proportional set size at the end of the run, MB. */
        val pssEndMb: Double? = null,
        /** Quality warnings (FR8, [SheetQuality]); empty = nothing found wrong. */
        val warnings: List<SheetQuality.Warning> = emptyList(),
        // UI data for the page overlay (docs/ui_plan.md §3.5): COPIES of what
        // the pipeline used, page pixels. Nothing is computed for them.
        /** Recording time zero, page column. */
        val t0X: Double = Double.NaN,
        /** The four row centres, page rows. */
        val rowY: List<Int> = emptyList(),
        /** Separator ticks kept, (row, k, x). */
        val tickPositions: List<Triple<Int, Int, Double>> = emptyList(),
        /** Ticks rejected as outliers and replaced by their column-mates. */
        val ticksRejected: List<Triple<Int, Int, Double>> = emptyList(),
        /** Each lead's cell on the page. */
        val cells: List<CellBox> = emptyList()
    ) {
        /** "ok", "check" or "unreliable" ([SheetQuality.verdict]). */
        val quality: String get() = SheetQuality.verdict(warnings)
        /**
         * What a deployed scan pays: every stage except the harness ones,
         * whose keys end in `*` (scoring against a stored truth; CLAUDE.md
         * limitation 9).
         */
        val pipelineMs: Long get() = timingsMs.filterKeys { !it.endsWith("*") }.values.sum()
    }

    /** A lead's cell on the page (layout.cells), for the overlay. */
    class CellBox(val name: String, val rhythm: Boolean, val x0: Int, val x1: Int, val y0: Int, val y1: Int)

    /**
     * What [run] reports as it goes, for the Digitising screen (docs/ui_plan.md
     * section 3.3). Display only: the time spent delivering these is measured
     * and left out of every stage's time, and with no listener nothing is
     * reported.
     */
    sealed class Progress {
        /** Stage [key] (the timingsMs key) finished in [ms]; [detail] for the user. */
        class Stage(val key: String, val ms: Long, val detail: String?) : Progress()
        /** The page the rows are read from, as a preview, and its full size. */
        class Page(val preview: Bitmap, val width: Int, val height: Int) : Progress()
        /** The four row centres found, page rows. */
        class Rows(val rowY: IntArray) : Progress()
        /** [done] of [total] model tiles read (learned tracer). */
        class Tile(val done: Int, val total: Int) : Progress()
        /** Row [row] traced: the row's page trace. */
        class Row(val row: Int, val trace: RowTrace) : Progress()
    }

    /** Timing key of the truth scoring: harness, not pipeline. */
    const val SCORE_KEY = "score*"

    /** Memory sampling interval, ms: well under every stage but cut+score. */
    const val MEMORY_SAMPLE_MS = 20L

    private fun readMemory(): StageMemory.Reading {
        val rt = Runtime.getRuntime()
        return StageMemory.Reading(Debug.getNativeHeapAllocatedSize(), rt.totalMemory() - rt.freeMemory())
    }

    private fun pssMb(): Double {
        val info = Debug.MemoryInfo()
        Debug.getMemoryInfo(info)
        return info.totalPss / 1024.0
    }

    enum class Geometry { CORNERS, GRID }

    /** What locate_sheet finds on a page; [searchPxPerMm] places the pulse search. */
    private class Located(val m2: TraceExtractor.Mask, val rows: IntArray, val loc: StripLocator.Location,
                          val edges: DoubleArray, val pulseT0: Double, val t0: Double, val anchor: Double)

    /** segment.locate_sheet(page, scale_hint): rows, pulses, time zero, duration anchor. */
    private fun locate(rect: Mat, scaleHint: Double?): Located {
        val m2 = mask(rect, OPEN_LOCATE)
        val rows = SheetLayout.sheetRows(m2)
        val loc = StripLocator.locate(m2)
        val approx = scaleHint ?: (loc.recordingWidthPx / (SheetLayout.RECORDING_S * SheetLayout.MM_PER_S))
        val off = SheetLayout.maskOffsetPx(OPEN_LOCATE)
        val edges = DoubleArray(4) { SheetLayout.pulseFallingEdge(m2, rows[it], approx, off) }
        val pulseT0 = SheetLayout.t0FromPulses(edges)
        val t0 = if (pulseT0.isFinite()) pulseT0 else loc.t0X - off
        val anchor = (loc.x1 - t0) / (SheetLayout.RECORDING_S * SheetLayout.MM_PER_S)
        return Located(m2, rows, loc, edges, pulseT0, t0, anchor)
    }

    /** One printed row's trace in page pixels: pos[i] is the row at column x0 + i. */
    class RowTrace(val x0: Int, val pos: DoubleArray)

    /**
     * RGBA page -> JPEG bytes, in native memory: a full-resolution page is
     * ~23 MB as a Java bitmap, against a ~200 MB heap (ImageWarp.previewBitmap).
     */
    fun encodeJpeg(rgba: Mat, quality: Int = 90): ByteArray {
        val bgr = Mat()
        val buf = MatOfByte()
        val params = MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, quality)
        try {
            Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
            check(Imgcodecs.imencode(".jpg", bgr, buf, params)) { "JPEG encoding failed" }
            return buf.toArray()
        } finally {
            bgr.release(); buf.release(); params.release()
        }
    }

    // ------------------------------------------------------------ model

    /** b3 as a [SheetCanonical.TileModel]: LiteRT, CPU, float I/O. */
    class LiteRtModel(context: Context, threads: Int = 4) : SheetCanonical.TileModel, AutoCloseable {
        private val interp: Interpreter
        val height: Int
        val tileW: Int
        private val inName: String
        private val input: ByteBuffer
        private val row: ByteBuffer
        private val pres: ByteBuffer

        init {
            val buf = context.assets.openFd(MODEL_ASSET).use { fd ->
                FileInputStream(fd.fileDescriptor).use {
                    it.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
                }
            }
            interp = Interpreter(buf, Interpreter.Options().setNumThreads(threads))
            val key = interp.signatureKeys[0]
            inName = interp.getSignatureInputs(key).first()
            val shape = interp.getInputTensorFromSignature(inName, key).shape()
            height = shape[1]; tileW = shape[2]
            input = ByteBuffer.allocateDirect(height * tileW * 3 * 4).order(ByteOrder.nativeOrder())
            row = ByteBuffer.allocateDirect(2 * tileW * 4).order(ByteOrder.nativeOrder())
            pres = ByteBuffer.allocateDirect(2 * tileW * 4).order(ByteOrder.nativeOrder())
        }

        override fun run(tileRgb: FloatArray, height: Int, tileW: Int): Pair<FloatArray, FloatArray> {
            require(height == this.height && tileW == this.tileW) {
                "model takes ${this.height} x ${this.tileW}, got $height x $tileW"
            }
            input.rewind(); input.asFloatBuffer().put(tileRgb)
            row.rewind(); pres.rewind()
            interp.runSignature(mapOf(inName to input), mapOf("row" to row, "presence" to pres))
            val r = FloatArray(2 * tileW); val p = FloatArray(2 * tileW)
            (row.rewind() as ByteBuffer).asFloatBuffer().get(r)
            (pres.rewind() as ByteBuffer).asFloatBuffer().get(p)
            return r to p
        }

        override fun close() = interp.close()
    }

    // ------------------------------------------------------------- steps

    private fun rectify(photo: Mat, c: DoubleArray): Mat {
        val top = hypot(c[2] - c[0], c[3] - c[1])
        val bottom = hypot(c[4] - c[6], c[5] - c[7])
        val left = hypot(c[6] - c[0], c[7] - c[1])
        val right = hypot(c[4] - c[2], c[5] - c[3])
        val w = StripLocator.pyRound((top + bottom) / 2.0)
        val h = StripLocator.pyRound((left + right) / 2.0)
        check(w >= 16 && h >= 16) { "the four corners enclose only ${w}x$h px" }
        val from = MatOfPoint2f(Point(c[0], c[1]), Point(c[2], c[3]), Point(c[4], c[5]), Point(c[6], c[7]))
        val to = MatOfPoint2f(Point(0.0, 0.0), Point(w - 1.0, 0.0),
            Point(w - 1.0, h - 1.0), Point(0.0, h - 1.0))
        val m = Imgproc.getPerspectiveTransform(from, to)
        val out = Mat()
        Imgproc.warpPerspective(photo, out, m, Size(w.toDouble(), h.toDouble()),
            Imgproc.INTER_CUBIC, org.opencv.core.Core.BORDER_REPLICATE)
        m.release(); from.release(); to.release()
        return out
    }

    private fun mask(rgba: Mat, open: Int): TraceExtractor.Mask {
        val b = TraceExtractor.binarise(rgba, openKernel = open)
        try { return TraceExtractor.maskOf(b) } finally { b.release() }
    }

    private fun classicalRow(rect: Mat, layout: SheetLayout.LeadLayout, row: Int): Pair<DoubleArray, DoubleArray> {
        val (x0, x1) = SheetLayout.rowBandColumns(layout, row, rect.cols()).let { it[0] to it[1] }
        val (y0, y1) = layout.rowBands[row].let { it[0] to it[1] }
        val band = Mat(rect, Rect(x0, y0, x1 - x0, y1 - y0)).clone()
        val m = try { StripLocator.isolateTrace(mask(band, OPEN_TRACE)) } finally { band.release() }
        val pos = SheetLayout.traceRowFromMask(m, layout, row, x0)
        // Mask column c holds the ink of image column c - offset (sheet.classical_row_tracer).
        val off = SheetLayout.maskOffsetPx(OPEN_TRACE)
        return DoubleArray(pos.size) { x0 + it - off } to pos
    }

    private fun modelRow(rect: Mat, layout: SheetLayout.LeadLayout, row: Int,
                         model: LiteRtModel,
                         /** The model as called per tile: [model], or it wrapped to report progress. */
                         tileModel: SheetCanonical.TileModel = model,
                         onFrame: ((SheetCanonical.Frame) -> Unit)? = null): Pair<DoubleArray, DoubleArray> {
        val pxMm = layout.pxPerS / SheetLayout.MM_PER_S
        val frame = SheetCanonical.toCanonical(rect, pxMm, layout.t0X - LEAD_IN_MM * pxMm,
            layout.rowY[row].toDouble(), model.height)
        onFrame?.invoke(frame)
        val (r, p) = SheetCanonical.predictTiled(frame, tileModel, model.tileW)
        return SheetCanonical.rowTraceToSource(r, p, frame.map)
    }

    /**
     * Pearson r and RMSE (each signal about its own median) against the truth
     * window, ignoring NaN samples. The medians are taken once: taken inside
     * the map, as before 2026-10-06, they re-sorted the window per sample --
     * O(n^2 log n), 7.3 s of the A30s's "cut+score" on a sheet.
     */
    internal fun score(sig: DoubleArray, truth: DoubleArray): Pair<Double, Double>? {
        val idx = sig.indices.filter { sig[it].isFinite() && it < truth.size }
        if (idx.size < 50) return null
        val a = DoubleArray(idx.size) { truth[idx[it]] }
        val b = DoubleArray(idx.size) { sig[idx[it]] }
        val ma = a.average(); val mb = b.average()
        var sab = 0.0; var saa = 0.0; var sbb = 0.0
        for (i in a.indices) {
            sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) * (a[i] - ma)
            sbb += (b[i] - mb) * (b[i] - mb)
        }
        val pcc = sab / kotlin.math.sqrt(saa * sbb)
        val medA = SheetLayout.median(a)
        val medB = SheetLayout.median(b)
        var sse = 0.0
        for (i in a.indices) { val d = (a[i] - medA) - (b[i] - medB); sse += d * d }
        val rmse = kotlin.math.sqrt(sse / a.size)
        return pcc to rmse
    }

    /**
     * Run the sheet. [corners] null = find them automatically. [truth12],
     * if given, is the 12-lead truth (lead name -> 5000 samples at 500 Hz).
     */
    suspend fun run(
        context: Context, photo: Bitmap, corners: DoubleArray?, tracer: Tracer,
        truth12: Map<String, DoubleArray>?, model: LiteRtModel? = null,
        /** For corners found by [SheetCorners.find] earlier and passed in. */
        cornersAuto: Boolean = corners == null,
        cornersRefined: Boolean? = null,
        /** GRID: regularise the page on its printed 5 mm grid (ecgproto geometry="grid"). */
        geometry: Geometry = Geometry.GRID,
        /**
         * Progress for the UI, called on the pipeline's thread. Null (the
         * default): nothing is reported and the run is exactly as before. With
         * a listener, the run also checks for cancellation between stages and
         * tiles.
         */
        onProgress: ((Progress) -> Unit)? = null
    ): Result = withContext(Dispatchers.Default) {
        // Peak memory per stage (NFR2): sampled in the background, filed at
        // each lap. Reads only counters, so it costs the stages nothing.
        val mem = StageMemory(::readMemory)
        val sampler = launch(Dispatchers.IO) {
            while (isActive) { mem.sample(); delay(MEMORY_SAMPLE_MS) }
        }
        val t = linkedMapOf<String, Long>()
        var mark = System.nanoTime()
        // Time spent in onProgress since the last lap: subtracted from the
        // stage it fell in (always 0 without a listener).
        var excludedNs = 0L
        fun notify(e: Progress) {
            val listener = onProgress ?: return
            val a = System.nanoTime()
            listener(e)
            excludedNs += System.nanoTime() - a
        }
        fun lap(k: String, detail: String? = null) {
            val n = System.nanoTime(); t[k] = (n - mark - excludedNs) / 1_000_000; mark = n
            excludedNs = 0L
            mem.close(k)
            if (onProgress != null) {
                ensureActive()
                notify(Progress.Stage(k, t.getValue(k), detail))
            }
        }
        try {
            var refined: Boolean? = cornersRefined
            val c = corners ?: SheetCorners.find(photo).let { refined = it.refined; it.corners }
            lap("corners")
            val src = ImageWarp.toMat(photo)
            var rect: Mat
            var fit: GridFit.Fit? = null
            var gridNote: String? = null
            try {
                rect = rectify(src, c)
                lap("rectify")
                if (geometry == Geometry.GRID) {
                    // sheet.digitise_sheet: the fit's guess is the corner page's
                    // duration anchor, and the photo is warped once onto the
                    // regular page (no second resampling of the corner page).
                    try {
                        val f = GridFit.fit(GridFit.redness(rect), locate(rect, null).anchor)
                        val page = GridFit.regularise(src, c, rect.cols(), rect.rows(), f)
                        rect.release()
                        rect = page
                        fit = f
                    } catch (e: Exception) {
                        gridNote = "grid fit refused (${e.message}); the corner page was used"
                    }
                    lap("grid")
                }
            } finally {
                src.release()
            }
            try {
                if (onProgress != null) {
                    val a = System.nanoTime()
                    val pv = ImageWarp.previewBitmap(rect)
                    excludedNs += System.nanoTime() - a
                    notify(Progress.Page(pv, rect.cols(), rect.rows()))
                }
                val located = locate(rect, fit?.pxPerMm)
                val m2 = located.m2
                val rows = located.rows
                val pulseT0 = located.pulseT0
                val t0 = located.t0
                val anchor = located.anchor
                lap("locate", "${rows.size} rows · ${located.edges.count { it.isFinite() }} pulses")
                if (onProgress != null) notify(Progress.Rows(rows.copyOf()))
                // The tick search is placed with the grid's scale when there is
                // one (SheetLocation.search_px_per_mm); the duration anchor stays
                // the calibration's fallback.
                val ticks = SheetLayout.tickScale(
                    if (OPEN_LOCATE == OPEN_TRACE) m2 else mask(rect, OPEN_TRACE), rows, t0,
                    fit?.pxPerMm ?: anchor, SheetLayout.maskOffsetPx(OPEN_TRACE))
                val useTicks = ticks.pxPerMm.isFinite()
                val px = if (useTicks) ticks.pxPerMm else anchor
                val layout = SheetLayout.segmentLeads(rect.rows(), rows, t0, px, if (useTicks) ticks else null)
                lap("calibrate+segment", "${ticks.nFound} of 9 ticks")

                val ownModel = if (tracer == Tracer.MODEL && model == null) LiteRtModel(context) else null
                val leads = ArrayList<LeadResult>()
                var rowTraces: List<RowTrace> = emptyList()
                var warnings: List<SheetQuality.Warning> = emptyList()
                try {
                    // With a listener: count tiles as the model reads them.
                    val m = if (tracer == Tracer.MODEL) (model ?: ownModel)!! else null
                    var tilesDone = 0
                    var tilesTotal = 0
                    val counting = if (m != null && onProgress != null) SheetCanonical.TileModel { rgb, h, w ->
                        val out = m.run(rgb, h, w)
                        tilesDone++
                        ensureActive()
                        notify(Progress.Tile(tilesDone, tilesTotal))
                        out
                    } else null
                    val traces = (0 until 4).map { r ->
                        val (xs, ys) = when (tracer) {
                            Tracer.MODEL -> if (counting == null) modelRow(rect, layout, r, m!!)
                                else modelRow(rect, layout, r, m!!, counting) { f ->
                                    if (tilesTotal == 0) tilesTotal = 4 * SheetCanonical.tileCount(f, m.tileW)
                                }
                            Tracer.CLASSICAL -> classicalRow(rect, layout, r)
                        }
                        val row = SheetLayout.onSheetColumns(xs, ys)
                        if (onProgress != null) {
                            ensureActive()
                            notify(Progress.Row(r, RowTrace(row.second, row.first)))
                        }
                        row
                    }
                    lap("trace")
                    val sigs = layout.cells.map { cell ->
                        val (pos, x0) = traces[cell.row]
                        SheetLayout.cellSignal(pos, x0, cell, px, layout)
                    }
                    lap("cut")
                // FR8: the time and geometry references this sheet was cut
                // with, and a physiological backstop per lead.
                warnings = SheetQuality.assess(
                    gridRequested = geometry == Geometry.GRID, gridUsed = fit != null,
                    gridPxPerMm = fit?.pxPerMm ?: Double.NaN,
                    calibration = if (useTicks) "ticks" else "duration", pxPerMm = px,
                    ticksFound = ticks.nFound, tickPositions = ticks.positions,
                    t0Source = if (pulseT0.isFinite()) "pulses" else "rhythm_locator",
                    leads = layout.cells.zip(sigs).associate { (c, s) ->
                        (if (c.rhythm) "II_rhythm" else c.name) to s },
                    tickRejected = ticks.rejected)
                lap("quality")
                    for ((cell, sig) in layout.cells.zip(sigs)) {
                        val tr = truth12?.get(cell.name)?.let {
                            val a = (cell.tStartS * SheetLayout.FS).roundToInt()
                            it.copyOfRange(a, minOf(it.size, a + sig.size))
                        }
                        val sc = tr?.let { score(sig, it) }
                        leads += LeadResult(cell.name, cell.rhythm, cell.tStartS, sig, sc?.first, sc?.second)
                    }
                    if (truth12 != null) lap(SCORE_KEY)
                    rowTraces = traces.map { (pos, x0) -> RowTrace(x0, pos) }
                } finally {
                    ownModel?.close()
                }
                // Not timed as a pipeline stage: it is for the Records tab only.
                val jpeg = runCatching { encodeJpeg(rect) }.getOrNull()
                Result(tracer, c, cornersAuto, refined, rect.cols(), rect.rows(), px,
                    if (useTicks) "ticks" else "duration", ticks.nFound,
                    if (pulseT0.isFinite()) "pulses" else "rhythm_locator", leads, t,
                    ImageWarp.previewBitmap(rect), jpeg, rowTraces,
                    if (fit != null) "grid" else "corners", fit, gridNote,
                    mem.baseline, mem.peaks(), runCatching { pssMb() }.getOrNull(), warnings,
                    t0X = t0, rowY = rows.toList(), tickPositions = ticks.positions,
                    ticksRejected = ticks.rejected,
                    cells = layout.cells.map { CellBox(it.name, it.rhythm, it.x0, it.x1, it.y0, it.y1) })
            } finally {
                rect.release()
            }
        } finally {
            sampler.cancel()
        }
    }

    /** `<name>_12lead.csv`: header of lead names, then 5000 rows. */
    suspend fun loadTruth12(context: Context, asset: String): Map<String, DoubleArray> =
        withContext(Dispatchers.IO) {
            context.assets.open(asset).bufferedReader().useLines { seq ->
                val it = seq.iterator()
                val names = it.next().trim().split(",")
                val cols = names.map { ArrayList<Double>(5000) }
                while (it.hasNext()) {
                    val line = it.next().trim()
                    if (line.isEmpty()) continue
                    val v = line.split(",")
                    for (i in names.indices) cols[i] += v[i].toDouble()
                }
                names.indices.associate { names[it] to cols[it].toDoubleArray() }
            }
        }

    /** FR10 for the whole sheet: 12 leads + rhythm, 500 Hz, mV. */
    fun csv(r: Result, photoLabel: String, targetName: String?, stamp: String): String = buildString {
        fun meta(k: String, v: Any?) = append("# $k: ${v ?: ""}\n")
        fun f(fmt: String, v: Double) = String.format(java.util.Locale.US, fmt, v)
        meta("generated_by", "EaCG SheetPipeline")
        meta("local_time", stamp)
        meta("device", "${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL} (API ${android.os.Build.VERSION.SDK_INT})")
        meta("photo", photoLabel)
        meta("target", targetName ?: "none")
        meta("tracer", if (r.tracer == Tracer.MODEL) "model $MODEL_ASSET" else "classical")
        meta("corners", if (r.cornersAuto) "automatic (refined=${r.cornersRefined})" else "user")
        meta("corners_px", r.corners.joinToString(" ") { f("%.2f", it) })
        meta("rect", "${r.rectW}x${r.rectH}")
        meta("geometry", r.gridFit?.let {
            "grid (fit rms ${f("%.3f", it.rmsMm)} mm, ${it.nVertical}x${it.nHorizontal} rules, " +
                "${f("%.4f", it.pxPerMm)} px/mm, inliers ${f("%.3f", it.inlierFraction)})"
        } ?: ("corners" + (r.gridNote?.let { " -- $it" } ?: "")))
        meta("calibration", "${r.calibration} ${f("%.4f", r.pxPerMm)} px/mm, ${r.ticksFound}/9 ticks, t0 from ${r.t0Source}")
        meta("timings_ms", r.timingsMs.entries.joinToString(" ") { "${it.key}=${it.value}" })
        // native/java MB, absolute; see StageMemory for what each one counts.
        r.memoryBaseline?.let { meta("memory_peak_mb", StageMemory.format(it, r.memoryPeaks)) }
        r.pssEndMb?.let { meta("pss_end_mb", f("%.1f", it)) }
        // FR8 (SheetQuality): CODE/severity/leads per warning, ';'-separated.
        meta("quality", r.quality)
        meta("warnings", r.warnings.joinToString("; ") {
            "${it.code}/${it.severity.name.lowercase()}" + (if (it.leads.isEmpty()) "" else "/" + it.leads.joinToString(","))
        })
        for (l in r.leads) l.pcc?.let { meta("pcc_asis_${l.name}${if (l.rhythm) "_rhythm" else ""}", f("%.4f", it)) }
        val order = listOf("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
        append("time_s," + order.joinToString(",") + ",II_rhythm\n")
        val n = (SheetLayout.RECORDING_S * SheetLayout.FS).toInt()
        val cols = order.map { name ->
            val l = r.leads.first { it.name == name && !it.rhythm }
            val a = (l.tStartS * SheetLayout.FS).roundToInt()
            DoubleArray(n) { i -> if (i in a until a + l.mv.size) l.mv[i - a] else Double.NaN }
        } + listOf(r.leads.first { it.rhythm }.mv)
        for (i in 0 until n) {
            append(f("%.3f", i / SheetLayout.FS))
            for (col in cols) {
                append(',')
                val v = if (i < col.size) col[i] else Double.NaN
                if (v.isFinite()) append(f("%.4f", v))
            }
            append('\n')
        }
    }
}
