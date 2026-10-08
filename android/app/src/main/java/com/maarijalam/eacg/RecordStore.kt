package com.maarijalam.eacg

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.StringReader
import java.io.StringWriter
import java.util.Locale
import java.util.Properties
import kotlin.math.roundToInt

/**
 * The Records tab's storage: every whole-sheet digitisation, kept so it can
 * be looked at again -- the 12 leads, the page with the digitised trace drawn
 * on it, and the original photograph.
 *
 * One folder per record under the app's PRIVATE files directory
 * (`files/records/<id>/`): an ECG is health data, so it is not put in shared
 * storage, and no permission is needed. Removed with the app; FR10's CSV
 * export to Downloads stays the way to take a record off the phone.
 *
 *     record.properties  what produced it (tracer, corners, calibration, ...)
 *     leads.csv          the 12 leads + rhythm, SheetPipeline.csv's format
 *     traces.txt         per row, the traced page row for each column
 *     photo.jpg          the photograph as digitised (3,200 px decode)
 *     page.jpg           the rectified page at full resolution
 *     thumb.jpg          a 360 px copy of the photograph for the list
 *
 * A record is written to a `.tmp` folder and renamed into place, so a crash
 * mid-save never leaves a half record in the list. Prototype: not encrypted.
 *
 * The file formats are plain JVM code (Properties, text) so they are unit
 * tested without a device (RecordStoreTest); only the image I/O is Android.
 */
object RecordStore {

    const val PHOTO = "photo.jpg"
    const val PAGE = "page.jpg"
    const val THUMB = "thumb.jpg"
    const val LEADS = "leads.csv"
    const val TRACES = "traces.txt"
    const val META = "record.properties"
    private const val THUMB_DIM = 360

    data class Meta(
        val id: String,
        val createdMs: Long,
        val tracer: String,              // "MODEL" or "CLASSICAL"
        val target: String?,             // printed test sheet, if one was chosen
        val calibration: String,
        val pxPerMm: Double,
        val ticksFound: Int,
        val t0Source: String,
        val cornersAuto: Boolean,
        val corners: DoubleArray,
        val pageW: Int, val pageH: Int,
        val photoW: Int, val photoH: Int,
        val medianPcc: Double?,          // only when scored against a test sheet
        val leadsScored: Int,
        val totalMs: Long,
        /** "grid" or "corners"; null on records saved before the grid fit. */
        val geometry: String? = null,
        // Added 2026-10-07 for the redesigned Result and Records screens
        // (docs/ui_plan.md). All null on older records, which still open.
        /** SheetQuality verdict: "ok", "check" or "unreliable". */
        val quality: String? = null,
        /** The warnings, one per line, [encodeWarnings]. */
        val warnings: String? = null,
        val cornersRefined: Boolean? = null,
        val gridRmsMm: Double? = null,
        /** Rules fitted, "VxH". */
        val gridRules: String? = null,
        /** Why the grid fit was refused, when it was. */
        val gridNote: String? = null,
        /** Per-stage times, "stage=ms stage=ms ...", as in the CSV. */
        val timings: String? = null,
        /** StageMemory.format(...), as the CSV's memory_peak_mb. */
        val memory: String? = null,
        val pssEndMb: Double? = null,
        /** Where the photograph came from: "picked" (gallery) or "camera". */
        val photoSource: String? = null,
        /** UI data for the page overlay: time zero and the row centres, page px. */
        val t0X: Double? = null,
        val rowY: String? = null,
        /** Separator ticks kept and replaced, "row k x;row k x;...", page px. */
        val ticks: String? = null,
        val ticksRejected: String? = null,
        /** Each cell's extent, "name rhythm x0 x1 y0 y1;...", page px. */
        val cells: String? = null,
        /** A name the user gave the record (Rename); null shows "PTB-XL n" / "Photo". */
        val name: String? = null
    )

    fun root(context: Context) = File(context.filesDir, "records")

    // ---------------------------------------------------------------- pure

    fun metaToText(m: Meta): String {
        val p = Properties()
        fun put(k: String, v: Any?) { if (v != null) p.setProperty(k, v.toString()) }
        put("id", m.id); put("created_ms", m.createdMs); put("tracer", m.tracer)
        put("target", m.target); put("calibration", m.calibration)
        put("px_per_mm", String.format(Locale.US, "%.6f", m.pxPerMm))
        put("ticks_found", m.ticksFound); put("t0_source", m.t0Source)
        put("corners_auto", m.cornersAuto)
        put("corners", m.corners.joinToString(" ") { String.format(Locale.US, "%.2f", it) })
        put("page_w", m.pageW); put("page_h", m.pageH)
        put("photo_w", m.photoW); put("photo_h", m.photoH)
        put("median_pcc", m.medianPcc?.let { String.format(Locale.US, "%.4f", it) })
        put("leads_scored", m.leadsScored); put("total_ms", m.totalMs)
        put("geometry", m.geometry)
        put("quality", m.quality); put("warnings", m.warnings)
        put("corners_refined", m.cornersRefined)
        put("grid_rms_mm", m.gridRmsMm?.let { String.format(Locale.US, "%.4f", it) })
        put("grid_rules", m.gridRules); put("grid_note", m.gridNote)
        put("timings", m.timings); put("memory", m.memory)
        put("pss_end_mb", m.pssEndMb?.let { String.format(Locale.US, "%.1f", it) })
        put("photo_source", m.photoSource)
        put("t0_x", m.t0X?.let { String.format(Locale.US, "%.2f", it) })
        put("row_y", m.rowY); put("ticks", m.ticks); put("ticks_rejected", m.ticksRejected)
        put("cells", m.cells)
        put("name", m.name)
        return StringWriter().also { p.store(it, "EaCG record") }.toString()
    }

    fun metaFromText(text: String): Meta {
        val p = Properties().apply { load(StringReader(text)) }
        fun s(k: String) = p.getProperty(k) ?: error("record is missing '$k'")
        return Meta(
            id = s("id"), createdMs = s("created_ms").toLong(), tracer = s("tracer"),
            target = p.getProperty("target"), calibration = s("calibration"),
            pxPerMm = s("px_per_mm").toDouble(), ticksFound = s("ticks_found").toInt(),
            t0Source = s("t0_source"), cornersAuto = s("corners_auto").toBoolean(),
            corners = s("corners").split(" ").filter { it.isNotEmpty() }.map { it.toDouble() }.toDoubleArray(),
            pageW = s("page_w").toInt(), pageH = s("page_h").toInt(),
            photoW = s("photo_w").toInt(), photoH = s("photo_h").toInt(),
            medianPcc = p.getProperty("median_pcc")?.toDouble(),
            leadsScored = s("leads_scored").toInt(), totalMs = s("total_ms").toLong(),
            geometry = p.getProperty("geometry"),
            quality = p.getProperty("quality"), warnings = p.getProperty("warnings"),
            cornersRefined = p.getProperty("corners_refined")?.toBoolean(),
            gridRmsMm = p.getProperty("grid_rms_mm")?.toDouble(),
            gridRules = p.getProperty("grid_rules"), gridNote = p.getProperty("grid_note"),
            timings = p.getProperty("timings"), memory = p.getProperty("memory"),
            pssEndMb = p.getProperty("pss_end_mb")?.toDouble(),
            photoSource = p.getProperty("photo_source"),
            t0X = p.getProperty("t0_x")?.toDouble(), rowY = p.getProperty("row_y"),
            ticks = p.getProperty("ticks"), ticksRejected = p.getProperty("ticks_rejected"),
            cells = p.getProperty("cells"),
            name = p.getProperty("name")
        )
    }

    /** A warning as stored: code, severity, leads (comma-joined), message. */
    data class StoredWarning(val code: String, val severity: String, val leads: List<String>,
                             val message: String)

    /** One warning per line, fields tab-separated (Properties escapes both). */
    fun encodeWarnings(ws: List<StoredWarning>): String =
        ws.joinToString("\n") { listOf(it.code, it.severity, it.leads.joinToString(","),
            it.message.replace('\t', ' ').replace('\n', ' ')).joinToString("\t") }

    fun decodeWarnings(text: String?): List<StoredWarning> =
        text.orEmpty().lineSequence().filter { it.isNotBlank() }.map { line ->
            val f = line.split('\t')
            StoredWarning(f[0], f.getOrElse(1) { "warn" },
                f.getOrElse(2) { "" }.split(',').filter { it.isNotEmpty() }, f.getOrElse(3) { "" })
        }.toList()

    /** One line per row: `x0 v v v ...`, NaN written as an empty field. */
    fun tracesToText(rows: List<SheetPipeline.RowTrace>): String = buildString {
        for (r in rows) {
            append(r.x0)
            for (v in r.pos) { append(' '); if (v.isFinite()) append(String.format(Locale.US, "%.2f", v)) else append('-') }
            append('\n')
        }
    }

    fun tracesFromText(text: String): List<SheetPipeline.RowTrace> =
        text.lineSequence().filter { it.isNotBlank() }.map { line ->
            val f = line.trim().split(' ')
            SheetPipeline.RowTrace(f[0].toInt(),
                DoubleArray(f.size - 1) { f[it + 1].let { v -> if (v == "-") Double.NaN else v.toDouble() } })
        }.toList()

    /**
     * SheetPipeline.csv's text back to leads: each 3x4 lead is its 2.5 s
     * window of the 10 s columns (its position in LAYOUT_3X4 gives the
     * window), the rhythm strip all 10 s; `pcc_asis_*` header lines, if any,
     * come back as the lead's score.
     */
    fun leadsFromCsv(text: String): List<SheetPipeline.LeadResult> {
        val pcc = HashMap<String, Double>()
        var header: List<String>? = null
        val cols = ArrayList<ArrayList<Double>>()
        for (line in text.lineSequence()) {
            if (line.isEmpty()) continue
            if (line.startsWith("#")) {
                val kv = line.removePrefix("#").trim()
                if (kv.startsWith("pcc_asis_")) {
                    val i = kv.indexOf(':')
                    kv.substring(i + 1).trim().toDoubleOrNull()?.let { pcc[kv.substring(9, i)] = it }
                }
                continue
            }
            if (header == null) {
                header = line.split(',')
                repeat(header.size) { cols += ArrayList<Double>(5000) }
                continue
            }
            val v = line.split(',')
            for (i in header.indices) cols[i] += v.getOrNull(i)?.takeIf { it.isNotEmpty() }?.toDouble() ?: Double.NaN
        }
        val names = header ?: error("leads.csv has no header")
        val cell = (SheetLayout.CELL_S * SheetLayout.FS).roundToInt()
        val out = ArrayList<SheetPipeline.LeadResult>()
        for (leads in SheetLayout.LAYOUT_3X4) {
            for ((k, name) in leads.withIndex()) {
                val c = names.indexOf(name)
                if (c < 0) continue
                val all = cols[c]
                val a = k * cell
                val mv = DoubleArray(cell) { all.getOrElse(a + it) { Double.NaN } }
                out += SheetPipeline.LeadResult(name, false, k * SheetLayout.CELL_S, mv, pcc[name], null)
            }
        }
        names.indexOf("II_rhythm").takeIf { it >= 0 }?.let { c ->
            out += SheetPipeline.LeadResult("II", true, 0.0, cols[c].toDoubleArray(), pcc["II_rhythm"], null)
        }
        return out
    }

    /** Every complete record under [root], newest first; unreadable ones skipped. */
    fun listIn(root: File): List<Meta> =
        (root.listFiles() ?: emptyArray())
            .filter { it.isDirectory && !it.name.endsWith(".tmp") && File(it, META).isFile }
            .mapNotNull { runCatching { metaFromText(File(it, META).readText()) }.getOrNull() }
            .sortedByDescending { it.createdMs }

    fun sizeBytes(dir: File): Long = dir.walkTopDown().filter { it.isFile }.sumOf { it.length() }

    fun newId(nowMs: Long, stamp: String): String =
        stamp + "_" + (nowMs % 4096).toString(16).padStart(3, '0')

    // ------------------------------------------------------------- android

    suspend fun save(context: Context, r: SheetPipeline.Result, photo: Bitmap,
                     target: String?, csv: String, photoSource: String? = null): Meta = withContext(Dispatchers.IO) {
        val now = System.currentTimeMillis()
        val id = newId(now, Exporter.stamp())
        val scored = r.leads.filter { !it.rhythm }.mapNotNull { it.pcc }.sorted()
        val med = if (scored.isEmpty()) null
            else if (scored.size % 2 == 1) scored[scored.size / 2]
            else 0.5 * (scored[scored.size / 2 - 1] + scored[scored.size / 2])
        fun f(fmt: String, v: Double) = String.format(Locale.US, fmt, v)
        val meta = Meta(id, now, r.tracer.name, target, r.calibration, r.pxPerMm, r.ticksFound,
            r.t0Source, r.cornersAuto, r.corners, r.rectW, r.rectH, photo.width, photo.height,
            med, scored.size, r.pipelineMs, r.geometry,
            quality = r.quality,
            warnings = encodeWarnings(r.warnings.map {
                StoredWarning(it.code, it.severity.name.lowercase(Locale.US), it.leads, it.message) }),
            cornersRefined = r.cornersRefined,
            gridRmsMm = r.gridFit?.rmsMm,
            gridRules = r.gridFit?.let { "${it.nVertical}x${it.nHorizontal}" },
            gridNote = r.gridNote,
            timings = r.timingsMs.entries.joinToString(" ") { "${it.key}=${it.value}" },
            memory = r.memoryBaseline?.let { StageMemory.format(it, r.memoryPeaks) },
            pssEndMb = r.pssEndMb,
            photoSource = photoSource,
            t0X = r.t0X.takeIf { it.isFinite() },
            rowY = r.rowY.takeIf { it.isNotEmpty() }?.joinToString(" "),
            ticks = r.tickPositions.takeIf { it.isNotEmpty() }
                ?.joinToString(";") { (row, k, x) -> "$row $k ${f("%.1f", x)}" },
            ticksRejected = r.ticksRejected.takeIf { it.isNotEmpty() }
                ?.joinToString(";") { (row, k, x) -> "$row $k ${f("%.1f", x)}" },
            cells = r.cells.takeIf { it.isNotEmpty() }
                ?.joinToString(";") { "${it.name} ${it.rhythm} ${it.x0} ${it.x1} ${it.y0} ${it.y1}" })
        val root = root(context).apply { mkdirs() }
        val tmp = File(root, "$id.tmp").apply { deleteRecursively(); mkdirs() }
        try {
            File(tmp, PHOTO).outputStream().use { photo.compress(Bitmap.CompressFormat.JPEG, 92, it) }
            r.pageJpeg?.let { File(tmp, PAGE).writeBytes(it) }
            val s = THUMB_DIM.toDouble() / maxOf(photo.width, photo.height)
            val thumb = Bitmap.createScaledBitmap(photo, (photo.width * s).roundToInt().coerceAtLeast(1),
                (photo.height * s).roundToInt().coerceAtLeast(1), true)
            File(tmp, THUMB).outputStream().use { thumb.compress(Bitmap.CompressFormat.JPEG, 85, it) }
            thumb.recycle()
            File(tmp, LEADS).writeText(csv)
            File(tmp, TRACES).writeText(tracesToText(r.rowTraces))
            File(tmp, META).writeText(metaToText(meta))       // last: marks it complete
            check(tmp.renameTo(File(root, id))) { "could not move the record into place" }
        } catch (e: Throwable) {
            tmp.deleteRecursively()
            throw e
        }
        meta
    }

    /** Longest name kept; longer ones are cut. */
    const val MAX_NAME = 80

    /**
     * Give the record in [dir] a name (blank: back to the default title).
     * Only record.properties changes, rewritten through a temporary file and
     * renamed into place, so a crash cannot leave a record without its
     * metadata. The leads, CSV and images are untouched.
     */
    fun renameIn(dir: File, name: String?): Meta {
        val f = File(dir, META)
        val m = metaFromText(f.readText()).copy(name = name?.trim()?.take(MAX_NAME)?.takeIf { it.isNotEmpty() })
        val tmp = File(dir, "$META.tmp")
        tmp.writeText(metaToText(m))
        if (!tmp.renameTo(f)) {                      // some filesystems will not rename over a file
            f.delete()
            check(tmp.renameTo(f)) { "could not update the record's name" }
        }
        return m
    }

    suspend fun rename(context: Context, id: String, name: String?): Meta =
        withContext(Dispatchers.IO) { renameIn(dir(context, id), name) }

    suspend fun list(context: Context): List<Meta> = withContext(Dispatchers.IO) { listIn(root(context)) }

    fun dir(context: Context, id: String) = File(root(context), id)

    suspend fun leads(context: Context, id: String): List<SheetPipeline.LeadResult> =
        withContext(Dispatchers.IO) { leadsFromCsv(File(dir(context, id), LEADS).readText()) }

    suspend fun traces(context: Context, id: String): List<SheetPipeline.RowTrace> =
        withContext(Dispatchers.IO) {
            File(dir(context, id), TRACES).takeIf { it.isFile }?.let { tracesFromText(it.readText()) } ?: emptyList()
        }

    suspend fun csv(context: Context, id: String): String =
        withContext(Dispatchers.IO) { File(dir(context, id), LEADS).readText() }

    /**
     * Decode a record's image for viewing. RGB_565 halves the memory of a
     * full page (2,900 x 1,950: 11 MB instead of 23) and is ample for looking.
     */
    suspend fun image(context: Context, id: String, name: String, maxDim: Int = 0): Bitmap? =
        withContext(Dispatchers.IO) {
            val f = File(dir(context, id), name)
            if (!f.isFile) return@withContext null
            val o = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.RGB_565 }
            if (maxDim > 0) {
                val b = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                BitmapFactory.decodeFile(f.path, b)
                var sample = 1
                while (maxOf(b.outWidth, b.outHeight) / (sample * 2) >= maxDim) sample *= 2
                o.inSampleSize = sample
            }
            BitmapFactory.decodeFile(f.path, o)
        }

    suspend fun delete(context: Context, id: String): Boolean =
        withContext(Dispatchers.IO) { dir(context, id).deleteRecursively() }

    suspend fun totalBytes(context: Context): Long =
        withContext(Dispatchers.IO) { root(context).takeIf { it.exists() }?.let { sizeBytes(it) } ?: 0L }
}
