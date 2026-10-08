package com.maarijalam.eacg

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale

/** The Scan flow's screens (docs/ui_plan.md §3): Home → Corners → Processing → Result. */
enum class ScanRoute { HOME, CORNERS, PROCESSING, RESULT }

/**
 * The Scan flow's state and logic, held once at the app shell so it outlives
 * tab switches. The pipeline calls, the CSV text, the export file name and
 * the record saved are EXACTLY those of the former SheetSection (they feed
 * python/score_device_sheets.py and the Records tab); only where they are
 * driven from moved.
 *
 * Prototype mode (docs/ui_plan.md §4) decides only what is SHOWN; with it off
 * a scan runs with the defaults -- learned tracer, grid fit, no benchmark --
 * which are also the defaults with it on.
 */
class ScanController(
    private val context: Context,
    private val scope: CoroutineScope,
    private val settings: AppSettings,
) {
    var route by mutableStateOf(ScanRoute.HOME)

    // ---- the photograph and its corners
    var bitmap by mutableStateOf<Bitmap?>(null); private set
    var preview by mutableStateOf<Bitmap?>(null); private set
    var corners by mutableStateOf<DoubleArray?>(null); private set
    /** True while the corners are exactly what [SheetCorners.find] returned. */
    var cornersAuto by mutableStateOf(false); private set
    var cornersRefined by mutableStateOf<Boolean?>(null); private set
    /** Corners were found automatically (not reset when the user drags them). */
    var cornersFound by mutableStateOf(false); private set
    var cornerFailure by mutableStateOf<String?>(null); private set
    /** "picked" (gallery) or "camera": the CSV's `photo` label and the record. */
    var photoSource by mutableStateOf("picked"); private set
    var loading by mutableStateOf(false); private set

    // ---- research options (shown in prototype mode)
    var tracer by mutableStateOf(SheetPipeline.Tracer.MODEL)
    var gridFit by mutableStateOf(true)
    var targets by mutableStateOf<List<KitTargets.Target>>(emptyList()); private set
    /** The PTB-XL sheet to benchmark against; remembered between scans and launches. */
    var target by mutableStateOf<KitTargets.Target?>(null); private set

    // ---- the run and its result
    var busy by mutableStateOf(false); private set
    var status by mutableStateOf(""); private set
    var result by mutableStateOf<SheetPipeline.Result?>(null); private set
    var truth12 by mutableStateOf<Map<String, DoubleArray>?>(null); private set
    /** The benchmark target [result] was digitised against (null: none). */
    var resultTarget by mutableStateOf<KitTargets.Target?>(null); private set
    var saved by mutableStateOf<RecordStore.Meta?>(null); private set
    var saveError by mutableStateOf<String?>(null); private set
    var exported by mutableStateOf<String?>(null); private set
    private var job: Job? = null

    // ---- progress of the run in flight (SheetPipeline.Progress), for the Digitising screen
    /** Stage key (timingsMs) -> ms, as each finishes. */
    var stageMs by mutableStateOf<Map<String, Long>>(emptyMap()); private set
    /** Stage key -> detail ("4 rows · 3 pulses", "9 of 9 ticks"). */
    var stageDetail by mutableStateOf<Map<String, String>>(emptyMap()); private set
    var pagePreview by mutableStateOf<Bitmap?>(null); private set
    var pageSize by mutableStateOf(0 to 0); private set
    var tracedRows by mutableStateOf<Map<Int, SheetPipeline.RowTrace>>(emptyMap()); private set
    var rowCentres by mutableStateOf<IntArray?>(null); private set
    var tilesDone by mutableStateOf(0); private set
    var tilesTotal by mutableStateOf(0); private set
    /** Saving the record: null not started, -1 in progress, else ms (not a pipeline stage). */
    var saveMs by mutableStateOf<Long?>(null); private set
    /** The grid fit was asked for in this run (else its stage is skipped). */
    var runGridFit by mutableStateOf(true); private set
    var runTracer by mutableStateOf(SheetPipeline.Tracer.MODEL); private set

    private fun resetProgress() {
        stageMs = emptyMap(); stageDetail = emptyMap(); pagePreview = null; pageSize = 0 to 0
        tracedRows = emptyMap(); rowCentres = null; tilesDone = 0; tilesTotal = 0; saveMs = null
    }

    /** Called on the pipeline's thread: hand the event to the UI thread. */
    private fun onProgress(e: SheetPipeline.Progress) {
        scope.launch(Dispatchers.Main) {
            when (e) {
                is SheetPipeline.Progress.Stage -> {
                    stageMs = stageMs + (e.key to e.ms)
                    e.detail?.let { stageDetail = stageDetail + (e.key to it) }
                }
                is SheetPipeline.Progress.Page -> { pagePreview = e.preview; pageSize = e.width to e.height }
                is SheetPipeline.Progress.Rows -> rowCentres = e.rowY
                is SheetPipeline.Progress.Tile -> { tilesDone = e.done; tilesTotal = e.total }
                is SheetPipeline.Progress.Row -> tracedRows = tracedRows + (e.row to e.trace)
            }
        }
    }

    init {
        scope.launch {
            targets = runCatching { KitTargets.load(context) }.getOrDefault(emptyList())
            val remembered = settings.benchmarkTarget
            target = targets.firstOrNull { it.name == remembered }
        }
    }

    /** What a run uses: the chosen options in prototype mode, else the defaults. */
    val effectiveTracer get() = if (settings.prototypeMode) tracer else SheetPipeline.Tracer.MODEL
    val effectiveGridFit get() = if (settings.prototypeMode) gridFit else true
    val effectiveTarget get() = if (settings.prototypeMode) target else null

    fun chooseTarget(t: KitTargets.Target?) {
        target = t
        settings.updateBenchmarkTarget(t?.name)
    }

    /** Load a photograph (3,200 px decode, as every desktop replay assumes) and find its corners. */
    fun load(uri: Uri, source: String) {
        job?.cancel()
        scope.launch {
            loading = true
            status = "Loading and finding the sheet"
            result = null; saved = null; exported = null; saveError = null
            runCatching {
                val bmp = UriSource(context, uri, maxDimension = SHEET_PHOTO_MAX_DIM).acquire()
                withContext(Dispatchers.Default) {
                    val m = ImageWarp.toMat(bmp)
                    val small = try { ImageWarp.previewBitmap(m, 1080) } finally { m.release() }
                    val found = runCatching { SheetCorners.find(bmp) }
                    Triple(bmp, small, found)
                }
            }.onSuccess { (bmp, small, found) ->
                val f = found.getOrNull()
                bitmap = bmp; preview = small
                corners = f?.corners ?: CornerGeometry.initialCorners(bmp.width, bmp.height)
                cornersAuto = f != null; cornersFound = f != null; cornersRefined = f?.refined
                cornerFailure = if (f == null) found.exceptionOrNull()?.message else null
                photoSource = source
                status = ""
                loading = false
                route = ScanRoute.CORNERS
            }.onFailure { e ->
                loading = false
                status = "Could not load the photograph: ${e.message}"
            }
        }
    }

    fun moveCorners(c: DoubleArray) {
        corners = c
        cornersAuto = false
        result = null; saved = null; exported = null
    }

    /** The CSV's `photo` header value, as SheetSection wrote it ("picked WxH"). */
    private fun photoLabel(bmp: Bitmap) = "$photoSource ${bmp.width}x${bmp.height}"

    /** Digitise with the effective options; save to Records; show the result. */
    fun digitise() {
        val bmp = bitmap ?: return
        val c = corners?.copyOf() ?: return
        val tr = effectiveTracer
        val tg = effectiveTarget
        job?.cancel()
        job = scope.launch {
            busy = true
            result = null; saved = null; exported = null; saveError = null
            resetProgress()
            runGridFit = effectiveGridFit
            runTracer = tr
            status = "Digitising 12 leads (${tr.name.lowercase(Locale.US)})..."
            route = ScanRoute.PROCESSING
            try {
                val truth = tg?.truth12Asset?.let { SheetPipeline.loadTruth12(context, it) }
                val r = SheetPipeline.run(context, bmp, c, tr, truth,
                    cornersAuto = cornersAuto, cornersRefined = cornersRefined,
                    geometry = if (effectiveGridFit) SheetPipeline.Geometry.GRID else SheetPipeline.Geometry.CORNERS,
                    onProgress = ::onProgress)
                truth12 = truth
                resultTarget = tg
                // Every digitisation is kept (RecordStore); a failed save is
                // reported but never discards the result on screen.
                saveMs = -1L
                val saveStart = System.nanoTime()
                val meta = runCatching {
                    val csv = SheetPipeline.csv(r, photoLabel(bmp), tg?.name, Exporter.stamp())
                    RecordStore.save(context, r, bmp, tg?.name, csv, photoSource)
                }
                // runCatching would swallow a cancellation: a run cancelled at
                // the very end must still not show (RecordStore.save is atomic,
                // so no partial record exists either way).
                (meta.exceptionOrNull() as? CancellationException)?.let { throw it }
                saveMs = (System.nanoTime() - saveStart) / 1_000_000
                saved = meta.getOrNull()
                saveError = meta.exceptionOrNull()?.message
                result = r
                status = "Done in ${r.pipelineMs} ms."
                busy = false
                route = ScanRoute.RESULT
            } catch (e: CancellationException) {
                busy = false
                status = "Cancelled."
                throw e
            } catch (e: Throwable) {
                busy = false
                status = "Failed: ${e::class.java.simpleName}: ${e.message}"
                route = ScanRoute.CORNERS
            }
        }
    }

    /** Cancel a run in flight; no record is left (the pipeline had not saved). */
    fun cancel() {
        job?.cancel()
        job = null
        busy = false
        route = ScanRoute.CORNERS
    }

    /** Export the shown result: the former SheetSection's file name and text, to Downloads/EaCG. */
    fun export() {
        val r = result ?: return
        val bmp = bitmap ?: return
        val tg = resultTarget
        scope.launch {
            val stamp = Exporter.stamp()
            val name = exportName(tg?.name, r.tracer, stamp)
            exported = runCatching {
                "Exported to " + Exporter.exportText(context, name, SheetPipeline.csv(r, photoLabel(bmp), tg?.name, stamp))
            }.getOrElse { "Export failed: ${it.message}" }
        }
    }

    /** The shown result's CSV and file name, exactly as [export] writes them, for the share sheet. */
    fun csvForShare(): Pair<String, String>? {
        val r = result ?: return null
        val bmp = bitmap ?: return null
        val stamp = Exporter.stamp()
        return exportName(resultTarget?.name, r.tracer, stamp) to
            SheetPipeline.csv(r, photoLabel(bmp), resultTarget?.name, stamp)
    }

    /** Rename the record this scan was saved as. */
    fun renameSaved(name: String?) {
        val id = saved?.id ?: return
        scope.launch { saved = runCatching { RecordStore.rename(context, id, name) }.getOrDefault(saved) }
    }

    /** Delete the record this scan was saved as, then back to Home. */
    fun deleteSaved() {
        val id = saved?.id ?: return
        scope.launch { RecordStore.delete(context, id); reset() }
    }

    /** Back to Home, forgetting the photograph. */
    fun reset() {
        job?.cancel()
        bitmap = null; preview = null; corners = null; result = null; truth12 = null; resultTarget = null
        saved = null; exported = null; saveError = null; status = ""; busy = false
        route = ScanRoute.HOME
    }

    companion object {
        const val SHEET_PHOTO_MAX_DIM = 3200

        /** `eacg_sheet_<target|photo>_<tracer>_<stamp>.csv`: frozen, score_device_sheets.py reads it. */
        fun exportName(target: String?, tracer: SheetPipeline.Tracer, stamp: String) =
            "eacg_sheet_${target ?: "photo"}_${tracer.name.lowercase(Locale.US)}_$stamp.csv"
    }
}
