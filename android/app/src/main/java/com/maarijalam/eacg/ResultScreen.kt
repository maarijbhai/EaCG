package com.maarijalam.eacg

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import com.maarijalam.eacg.ui.Disclaimer
import com.maarijalam.eacg.ui.EaCard
import com.maarijalam.eacg.ui.EaIcons
import com.maarijalam.eacg.ui.PrimaryButton
import com.maarijalam.eacg.ui.SectionLabel
import com.maarijalam.eacg.ui.TopBar
import com.maarijalam.eacg.ui.drawEcgPaper
import com.maarijalam.eacg.ui.samplePath
import com.maarijalam.eacg.ui.traceStroke
import com.maarijalam.eacg.ui.theme.BadBg
import com.maarijalam.eacg.ui.theme.BadBorder
import com.maarijalam.eacg.ui.theme.BadText
import com.maarijalam.eacg.ui.theme.CardBorder
import com.maarijalam.eacg.ui.theme.DataText
import com.maarijalam.eacg.ui.theme.EcgPaper
import com.maarijalam.eacg.ui.theme.EcgTrace
import com.maarijalam.eacg.ui.theme.Ink
import com.maarijalam.eacg.ui.theme.MissingTick
import com.maarijalam.eacg.ui.theme.Muted
import com.maarijalam.eacg.ui.theme.Primary
import com.maarijalam.eacg.ui.theme.RowDivider
import com.maarijalam.eacg.ui.theme.TruthTrace
import com.maarijalam.eacg.ui.theme.WarnBg
import com.maarijalam.eacg.ui.theme.WarnBorder
import com.maarijalam.eacg.ui.theme.WarnIcon
import com.maarijalam.eacg.ui.theme.WarnIconSmall
import com.maarijalam.eacg.ui.theme.WarnText
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Locale

/**
 * What the Result screen shows, from a live [SheetPipeline.Result] or a saved
 * record (docs/ui_plan.md §3.4) -- so a scan and a reopened record look the
 * same. Display only: every value is copied, nothing recomputed.
 */
class ResultData(
    val title: String,
    val subtitle: String,
    val leads: List<SheetPipeline.LeadResult>?,     // null while a record's leads load
    val quality: String?,
    val warnings: List<RecordStore.StoredWarning>,
    val tracer: String,                              // "MODEL" or "CLASSICAL"
    val cornersAuto: Boolean,
    val cornersRefined: Boolean?,
    val geometry: String?,                           // "grid", "corners", or null (old record)
    val gridRmsMm: Double?,
    val gridRules: String?,
    val gridNote: String?,
    val calibration: String,
    val pxPerMm: Double,
    val ticksFound: Int,
    val t0Source: String,
    val totalMs: Long,
    val timings: String?,
    val memory: String?,
    val pssEndMb: Double?,
    val target: String?,
    /** Ticks kept and replaced, (row, k); null when not recorded (old records). */
    val ticksKept: Set<Pair<Int, Int>>?,
    val ticksReplaced: Set<Pair<Int, Int>>,
    /** Saved record id, if this result is in Records. */
    val recordId: String?,
    /** The classifier's output as stored; null = not run (older record): the tab computes it. */
    val diagnosis: Diagnosis? = null,
) {
    companion object {
        fun parseTicks(s: String?): List<Triple<Int, Int, Double>> =
            s.orEmpty().split(';').filter { it.isNotBlank() }.mapNotNull { e ->
                val f = e.trim().split(' ')
                runCatching { Triple(f[0].toInt(), f[1].toInt(), f[2].toDouble()) }.getOrNull()
            }

        fun fromLive(r: SheetPipeline.Result, meta: RecordStore.Meta?, target: String?): ResultData {
            val now = meta?.createdMs ?: System.currentTimeMillis()
            return ResultData(
                title = meta?.name ?: recordTitle(target), subtitle = "${recordWhen(now)} · 12 leads",
                leads = r.leads, quality = r.quality,
                warnings = r.warnings.map {
                    RecordStore.StoredWarning(it.code, it.severity.name.lowercase(Locale.US), it.leads, it.message)
                },
                tracer = r.tracer.name, cornersAuto = r.cornersAuto, cornersRefined = r.cornersRefined,
                geometry = r.geometry, gridRmsMm = r.gridFit?.rmsMm,
                gridRules = r.gridFit?.let { "${it.nVertical}x${it.nHorizontal}" }, gridNote = r.gridNote,
                calibration = r.calibration, pxPerMm = r.pxPerMm, ticksFound = r.ticksFound,
                t0Source = r.t0Source, totalMs = r.pipelineMs,
                timings = r.timingsMs.entries.joinToString(" ") { "${it.key}=${it.value}" },
                memory = r.memoryBaseline?.let { StageMemory.format(it, r.memoryPeaks) }, pssEndMb = r.pssEndMb,
                target = target,
                ticksKept = r.tickPositions.map { it.first to it.second }.toSet(),
                ticksReplaced = r.ticksRejected.map { it.first to it.second }.toSet(),
                recordId = meta?.id, diagnosis = r.diagnosis)
        }

        fun fromRecord(m: RecordStore.Meta, leads: List<SheetPipeline.LeadResult>?,
                       diagnosis: Diagnosis? = null): ResultData = ResultData(
            title = recordName(m), subtitle = "${recordWhen(m.createdMs)} · 12 leads",
            leads = leads, quality = m.quality, warnings = RecordStore.decodeWarnings(m.warnings),
            tracer = m.tracer, cornersAuto = m.cornersAuto, cornersRefined = m.cornersRefined,
            geometry = m.geometry, gridRmsMm = m.gridRmsMm, gridRules = m.gridRules, gridNote = m.gridNote,
            calibration = m.calibration, pxPerMm = m.pxPerMm, ticksFound = m.ticksFound,
            t0Source = m.t0Source, totalMs = m.totalMs, timings = m.timings, memory = m.memory,
            pssEndMb = m.pssEndMb, target = m.target,
            ticksKept = m.ticks?.let { t -> parseTicks(t).map { it.first to it.second }.toSet() },
            ticksReplaced = parseTicks(m.ticksRejected).map { it.first to it.second }.toSet(),
            recordId = m.id, diagnosis = diagnosis)
    }
}

private enum class ResultTab(val label: String) {
    LEADS("Leads"), DIAGNOSIS("Diagnosis"), OVERLAY("Page overlay"), PHOTO("Photo")
}

/**
 * The Result screen (mock-up artboards 4 and 5), for a scan and for a saved
 * record alike. [onExport] is the existing Exporter call (Downloads/EaCG,
 * frozen file name); [csvForShare] gives the same CSV text and file name for
 * the share sheet; [pageContent] and [photoContent] fill the other two tabs.
 */
@Composable
fun ResultScreen(
    data: ResultData,
    truth12: Map<String, DoubleArray>?,
    onBack: () -> Unit,
    onExport: () -> Unit,
    exportMessage: String?,
    savedNote: String?,
    csvForShare: suspend () -> Pair<String, String>?,
    onDelete: (() -> Unit)?,
    /** Rename the saved record (null: not saved, so nothing to rename). */
    onRename: ((String?) -> Unit)? = null,
    pageContent: @Composable () -> Unit,
    photoContent: @Composable () -> Unit,
    modifier: Modifier = Modifier,
) {
    val proto = LocalAppSettings.current.prototypeMode
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var tab by remember { mutableStateOf(ResultTab.LEADS) }
    var menu by remember { mutableStateOf(false) }
    var confirmDelete by remember { mutableStateOf(false) }
    var shareError by remember { mutableStateOf<String?>(null) }
    var renaming by remember { mutableStateOf(false) }

    if (renaming && onRename != null) {
        var text by remember { mutableStateOf(data.title) }
        AlertDialog(onDismissRequest = { renaming = false },
            title = { Text("Rename record") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedTextField(value = text, onValueChange = { text = it.take(RecordStore.MAX_NAME) },
                        singleLine = true, label = { Text("Name") })
                    Text("Leave empty to go back to the default name. The exported CSV and its file name " +
                        "do not change.", style = MaterialTheme.typography.bodySmall, color = Muted)
                }
            },
            confirmButton = { TextButton(onClick = { renaming = false; onRename(text) }) { Text("Save", color = Primary) } },
            dismissButton = { TextButton(onClick = { renaming = false }) { Text("Cancel") } })
    }

    if (confirmDelete && onDelete != null) {
        AlertDialog(onDismissRequest = { confirmDelete = false },
            title = { Text("Delete this record?") },
            text = { Text("The photograph, the page and the digitised leads are removed from this phone. " +
                "Exported CSV files in Downloads are not affected.") },
            confirmButton = { TextButton(onClick = { confirmDelete = false; onDelete() }) {
                Text("Delete", color = BadText) } },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text("Cancel") } })
    }

    Column(modifier.fillMaxSize()) {
        TopBar(data.title, onBack = onBack, subtitle = data.subtitle) {
            IconButton(onClick = {
                scope.launch {
                    shareError = runCatching {
                        val (name, text) = csvForShare() ?: error("nothing to share")
                        shareCsv(context, name, text); null
                    }.getOrElse { "Share failed: ${it.message}" }
                }
            }, modifier = Modifier.size(48.dp)) {
                Icon(EaIcons.Share, contentDescription = "Share the CSV", tint = com.maarijalam.eacg.ui.theme.Text2)
            }
            Box {
                IconButton(onClick = { menu = true }, modifier = Modifier.size(48.dp)) {
                    Icon(EaIcons.More, contentDescription = "More options", tint = com.maarijalam.eacg.ui.theme.Text2)
                }
                DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                    DropdownMenuItem(text = { Text("Rename") }, enabled = onRename != null,
                        onClick = { menu = false; renaming = true })
                    // The record's photo (as digitised) and straightened page to
                    // Downloads/EaCG, for the desktop replay and for diagnosis.
                    DropdownMenuItem(text = { Text("Save photo and page to Downloads") },
                        enabled = data.recordId != null, onClick = {
                            menu = false
                            val id = data.recordId ?: return@DropdownMenuItem
                            scope.launch {
                                shareError = runCatching {
                                    val dir = RecordStore.dir(context, id)
                                    val out = listOf(RecordStore.PHOTO to "eacg_photo_$id.jpg",
                                        RecordStore.PAGE to "eacg_page_$id.jpg").mapNotNull { (f, n) ->
                                        File(dir, f).takeIf { it.isFile }?.let {
                                            Exporter.exportImage(context, n, withContext(Dispatchers.IO) { it.readBytes() })
                                        }
                                    }
                                    "Saved " + out.joinToString(", ")
                                }.getOrElse { "Save failed: ${it.message}" }
                            }
                        })
                    DropdownMenuItem(text = { Text("Delete record", color = BadText) }, enabled = onDelete != null,
                        onClick = { menu = false; confirmDelete = true })
                }
            }
        }
        UnderlineTabs(ResultTab.entries.map { it.label }, tab.ordinal) { tab = ResultTab.entries[it] }

        when (tab) {
            ResultTab.LEADS -> Column(Modifier.fillMaxWidth().weight(1f).verticalScroll(rememberScrollState())
                .padding(start = 16.dp, end = 16.dp, top = 12.dp, bottom = 16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)) {
                QualityBanner(data) { tab = ResultTab.OVERLAY }
                LeadsCard(data, if (proto) truth12 else null, showScores = proto)
                DetailsList(data)
                if (proto && data.target != null) BenchmarkCard(data)
                if (proto) PrototypeDetails(data)
                PrimaryButton("Export CSV", onExport, Modifier.fillMaxWidth(), icon = EaIcons.Download)
                exportMessage?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = Muted,
                    modifier = Modifier.fillMaxWidth(), textAlign = TextAlign.Center) }
                shareError?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = BadText) }
                Disclaimer(text = listOfNotNull(savedNote, "Research use only; not for diagnosis.").joinToString(" "))
            }
            ResultTab.DIAGNOSIS -> DiagnosisTab(data, proto, Modifier.fillMaxWidth().weight(1f))
            ResultTab.OVERLAY -> Box(Modifier.fillMaxWidth().weight(1f)) { pageContent() }
            ResultTab.PHOTO -> Box(Modifier.fillMaxWidth().weight(1f).padding(16.dp)) { photoContent() }
        }
    }
}

/** Text tabs with a 3 dp maroon underline (no pills; plan §2.3). */
@Composable
fun UnderlineTabs(labels: List<String>, selected: Int, modifier: Modifier = Modifier, onSelect: (Int) -> Unit) {
    Column(modifier.fillMaxWidth().padding(horizontal = 16.dp)) {
        Row(Modifier.fillMaxWidth()) {
            labels.forEachIndexed { i, label ->
                val on = i == selected
                Column(Modifier.weight(1f).height(44.dp).clickable(role = Role.Tab) { onSelect(i) },
                    horizontalAlignment = Alignment.CenterHorizontally) {
                    Box(Modifier.weight(1f), contentAlignment = Alignment.Center) {
                        Text(label, style = MaterialTheme.typography.bodyMedium.copy(
                            fontWeight = if (on) FontWeight.SemiBold else FontWeight.Normal),
                            color = if (on) Primary else Muted)
                    }
                    Box(Modifier.fillMaxWidth().height(3.dp).background(if (on) Primary else Color.Transparent))
                }
            }
        }
        HorizontalDivider(color = CardBorder)
    }
}

/** A banner title for the most specific warning. */
private fun bannerTitle(w: RecordStore.StoredWarning): String = when (w.code) {
    "GRID_REFUSED" -> "Check the page geometry"
    "LEAD_COVERAGE", "LEAD_AMPLITUDE" -> "Check these leads before use"
    else -> "Check timing before use"
}

/** SheetQuality's verdict: nothing for ok, amber for check, red for unreliable (plan §3.4). */
@Composable
private fun QualityBanner(data: ResultData, onCompare: () -> Unit) {
    val q = data.quality ?: return
    if (q == "ok" || data.warnings.isEmpty() && q != "unreliable") return
    val bad = q == "unreliable"
    val bg = if (bad) BadBg else WarnBg
    val border = if (bad) BadBorder else WarnBorder
    val text = if (bad) BadText else WarnText
    val icon = if (bad) BadText else WarnIcon
    val title = if (bad) "Unreliable: a time or geometry reference is wrong"
        else data.warnings.firstOrNull()?.let(::bannerTitle) ?: "Check timing before use"
    val shape = RoundedCornerShape(16.dp)
    Row(Modifier.fillMaxWidth().background(bg, shape).border(1.dp, border, shape)
        .padding(horizontal = 14.dp, vertical = 12.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Icon(EaIcons.Warning, contentDescription = null, tint = icon, modifier = Modifier.size(22.dp).padding(top = 1.dp))
        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(title, style = MaterialTheme.typography.titleSmall, color = text)
            for (w in data.warnings) Text(w.message.ifEmpty { w.code }, style = MaterialTheme.typography.bodyMedium, color = text)
            Text("Compare on the page overlay", style = MaterialTheme.typography.bodyMedium.copy(
                fontWeight = FontWeight.SemiBold), color = icon,
                modifier = Modifier.heightIn(min = 32.dp).clickable(role = Role.Button, onClick = onCompare)
                    .padding(vertical = 6.dp))
        }
    }
}

/**
 * The sheet as printed: three rows of four leads and the rhythm strip, each a
 * pink-grid strip at the paper's own scale (10 s across, 10 mm/mV), lead
 * labels at each cell's start, the separator ticks as short bars -- amber
 * where a tick was not found or was replaced. With [truth12] (prototype mode,
 * benchmarked) the PTB-XL truth is drawn in grey under the digitised trace.
 */
@Composable
private fun LeadsCard(data: ResultData, truth12: Map<String, DoubleArray>?, showScores: Boolean) {
    EaCard(padding = androidx.compose.foundation.layout.PaddingValues(12.dp)) {
        val leads = data.leads
        if (leads == null) {
            Box(Modifier.fillMaxWidth().height(120.dp), contentAlignment = Alignment.Center) {
                CircularProgressIndicator(Modifier.size(24.dp), color = Primary, strokeWidth = 3.dp)
            }
            return@EaCard
        }
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            for (row in 0 until 4) {
                val rowLeads = leads.filter {
                    if (row < 3) !it.rhythm && SheetLayout.LAYOUT_3X4[row].contains(it.name) else it.rhythm
                }.sortedBy { it.tStartS }
                LeadStrip(row, rowLeads, data, truth12, showScores)
            }
            Row(Modifier.fillMaxWidth().padding(top = 2.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                for (t in listOf("0 s", "2.5", "5.0", "7.5", "10 s")) Text(t, style = DataText.copy(
                    fontSize = DataText.fontSize * 0.85f), color = Muted)
            }
        }
    }
}

@Composable
private fun LeadStrip(row: Int, leads: List<SheetPipeline.LeadResult>, data: ResultData,
                      truth12: Map<String, DoubleArray>?, showScores: Boolean) {
    Box(Modifier.fillMaxWidth().height(64.dp).clip(RoundedCornerShape(6.dp))) {
        Canvas(Modifier.fillMaxSize()) {
            val pxMm = size.width / (SheetLayout.RECORDING_S * SheetLayout.MM_PER_S).toFloat()
            drawEcgPaper(minorPx = pxMm)
            val mid = size.height * 0.55f
            val pxMv = (SheetLayout.MM_PER_MV * pxMm).toFloat()
            val secToX = (SheetLayout.MM_PER_S * pxMm).toFloat()
            for (l in leads) {
                val x0 = (l.tStartS * secToX).toFloat()
                truth12?.get(l.name)?.let { tr ->
                    val a = (l.tStartS * SheetLayout.FS).toInt()
                    val seg = tr.copyOfRange(a, minOf(tr.size, a + l.mv.size))
                    if (seg.isNotEmpty()) {
                        val med = SheetLayout.median(seg)
                        drawPath(samplePath(seg.size, { i -> x0 + (i / SheetLayout.FS * secToX).toFloat() },
                            { i -> mid - ((seg[i] - med) * pxMv).toFloat() }, { i -> seg[i].isFinite() }),
                            TruthTrace, style = traceStroke(1.5.dp.toPx()))
                    }
                }
                val v = l.mv
                drawPath(samplePath(v.size, { i -> x0 + (i / SheetLayout.FS * secToX).toFloat() },
                    { i -> mid - (v[i] * pxMv).toFloat() }, { i -> v[i].isFinite() }),
                    EcgTrace, style = traceStroke(1.2.dp.toPx()))
            }
            if (row < 3) for (k in 1..3) {
                val x = (k * SheetLayout.CELL_S * secToX).toFloat()
                val kept = data.ticksKept
                val missing = kept != null && ((row to k) !in kept || (row to k) in data.ticksReplaced)
                drawLine(if (missing) MissingTick else EcgTrace, Offset(x, mid - 7.dp.toPx()),
                    Offset(x, mid + 7.dp.toPx()), 2.dp.toPx())
            }
        }
        val density = LocalDensity.current
        Box(Modifier.fillMaxSize()) {
            for (l in leads) {
                val label = (if (l.rhythm) "II · rhythm" else l.name) +
                    if (showScores && l.pcc != null) String.format(Locale.US, " r=%.2f", l.pcc) else ""
                val frac = (l.tStartS / SheetLayout.RECORDING_S).toFloat()
                androidx.compose.foundation.layout.BoxWithConstraints(Modifier.fillMaxSize()) {
                    Text(label, style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold,
                        fontSize = MaterialTheme.typography.labelSmall.fontSize * 0.92f), color = EcgTrace,
                        modifier = Modifier.padding(start = with(density) { (maxWidth.toPx() * frac).toDp() } + 4.dp, top = 3.dp)
                            .background(EcgPaper.copy(alpha = 0.85f), RoundedCornerShape(3.dp))
                            .padding(horizontal = 3.dp))
                }
            }
        }
    }
}

/** The plain details list (no chips; plan §3.4). */
@Composable
private fun DetailsList(data: ResultData) {
    val rows = mutableListOf<Triple<String, String, Boolean>>()
    rows += Triple("Grid fit", when (data.geometry) {
        "grid" -> "Applied" + (data.gridRmsMm?.let { String.format(Locale.US, " · %.2f mm", it) } ?: "")
        "corners" -> "Corners only" + (if (data.gridNote != null) " (refused)" else "")
        else -> "—"
    }, data.gridNote != null)
    rows += Triple("Lead markers", "${data.ticksFound} of 9 found", data.ticksFound < SheetQuality.MIN_TICKS)
    rows += Triple("Time zero", if (data.t0Source == "pulses") "Calibration pulse" else "Fallback (rhythm strip)",
        data.t0Source != "pulses")
    rows += Triple("Tracer · time on phone", (if (data.tracer == SheetPipeline.Tracer.MODEL.name) "Learned" else "Classical") +
        String.format(Locale.US, " · %.1f s", data.totalMs / 1000.0), false)
    EaCard(padding = androidx.compose.foundation.layout.PaddingValues(horizontal = 14.dp, vertical = 2.dp)) {
        rows.forEachIndexed { i, (label, value, warn) ->
            if (i > 0) HorizontalDivider(color = RowDivider)
            Row(Modifier.fillMaxWidth().heightIn(min = 40.dp), verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(label, Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium, color = Muted)
                if (warn) Icon(EaIcons.Warning, contentDescription = "Warning", tint = WarnIconSmall,
                    modifier = Modifier.size(16.dp))
                Text(value, style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold),
                    color = if (warn) WarnIcon else Ink)
            }
        }
        data.gridNote?.let {
            Text(it, style = MaterialTheme.typography.labelSmall, color = Muted, modifier = Modifier.padding(bottom = 8.dp))
        }
    }
}

/** Prototype mode, benchmarked: the existing headline -- median lead r (as-is) and leads >= 0.9. */
@Composable
private fun BenchmarkCard(data: ResultData) {
    val scored = data.leads.orEmpty().filter { !it.rhythm }.mapNotNull { it.pcc }.sorted()
    if (scored.isEmpty()) return
    val med = if (scored.size % 2 == 1) scored[scored.size / 2] else 0.5 * (scored[scored.size / 2 - 1] + scored[scored.size / 2])
    EaCard(padding = androidx.compose.foundation.layout.PaddingValues(14.dp)) {
        SectionLabel("Benchmark · ${recordTitle(data.target)}")
        Row(Modifier.fillMaxWidth().padding(top = 8.dp), horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            Column(Modifier.weight(1f)) {
                Text(String.format(Locale.US, "%.3f", med), style = DataText.copy(fontSize = DataText.fontSize * 1.6f))
                Text("median lead r (as-is)", style = MaterialTheme.typography.labelSmall, color = Muted)
            }
            Column(Modifier.weight(1f)) {
                Text("${scored.count { it >= 0.9 }} of ${scored.size}", style = DataText.copy(fontSize = DataText.fontSize * 1.6f))
                Text("leads with r ≥ 0.9", style = MaterialTheme.typography.labelSmall, color = Muted)
            }
        }
        Text("Against the PTB-XL signal, no alignment. Per-lead r on each label; truth in grey.",
            style = MaterialTheme.typography.labelSmall, color = Muted, modifier = Modifier.padding(top = 6.dp))
    }
}

/** Prototype mode: everything the former result panel printed, in one expandable card. */
@Composable
private fun PrototypeDetails(data: ResultData) {
    var open by remember { mutableStateOf(false) }
    fun f(fmt: String, v: Double) = String.format(Locale.US, fmt, v)
    EaCard {
        Row(Modifier.fillMaxWidth().heightIn(min = 48.dp).clickable(role = Role.Button) { open = !open }
            .padding(horizontal = 14.dp), verticalAlignment = Alignment.CenterVertically) {
            Text("Prototype details", style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f)
                .semantics { heading() })
            Icon(EaIcons.ChevronRight, contentDescription = if (open) "Collapse" else "Expand", tint = Muted,
                modifier = Modifier.size(20.dp).graphicsLayer(rotationZ = if (open) 90f else 0f))
        }
        if (!open) return@EaCard
        val lines = mutableListOf<Pair<String, String>>()
        lines += "Tracer" to (if (data.tracer == SheetPipeline.Tracer.MODEL.name) "Learned (b3)" else "Classical")
        lines += "Corners" to ((if (data.cornersAuto) "automatic" else "by hand") +
            when (data.cornersRefined) { true -> " · refined"; false -> " · coarse"; null -> "" })
        lines += "Calibration" to "${data.calibration} · ${f("%.3f", data.pxPerMm)} px/mm"
        lines += "Ticks found" to "${data.ticksFound} of 9" +
            (if (data.ticksReplaced.isNotEmpty()) " · ${data.ticksReplaced.size} replaced" else "")
        lines += "Time zero" to (if (data.t0Source == "pulses") "pulses" else data.t0Source)
        lines += "Geometry" to when (data.geometry) {
            "grid" -> "grid fit" + (data.gridRmsMm?.let { " · " + f("%.3f", it) + " mm" } ?: "") +
                (data.gridRules?.let { " · $it rules" } ?: "")
            "corners" -> "corners only" + (data.gridNote?.let { " · $it" } ?: "")
            else -> "—"
        }
        lines += "Pipeline time" to f("%.1f", data.totalMs / 1000.0) + " s"
        memorySummary(data.memory, data.pssEndMb)?.let { lines += "Memory" to it }
        Column(Modifier.padding(start = 14.dp, end = 14.dp, bottom = 12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            for ((k, v) in lines) Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(k, Modifier.weight(0.4f), style = MaterialTheme.typography.labelSmall, color = Muted)
                Text(v, Modifier.weight(0.6f), style = DataText.copy(fontSize = DataText.fontSize * 0.92f))
            }
            data.timings?.let {
                Text("Stage times (ms)", style = MaterialTheme.typography.labelSmall, color = Muted,
                    modifier = Modifier.padding(top = 6.dp))
                Text(it.replace(' ', '\n'), style = DataText.copy(fontSize = DataText.fontSize * 0.92f))
            }
        }
    }
}

/** "native peak N MB (+M over start); process P MB" from a memory_peak_mb string. */
internal fun memorySummary(memory: String?, pss: Double?): String? {
    val parts = memory?.split(' ')?.mapNotNull { e ->
        val kv = e.split('=')
        val nj = kv.getOrNull(1)?.split('/')
        if (kv.size == 2 && nj != null && nj.size == 2) kv[0] to (nj[0].toDoubleOrNull() ?: return@mapNotNull null) else null
    } ?: return pss?.let { String.format(Locale.US, "process %.0f MB", it) }
    val base = parts.firstOrNull { it.first == "baseline" }?.second ?: return null
    val peak = parts.filter { it.first != "baseline" }.maxOfOrNull { it.second } ?: base
    return String.format(Locale.US, "native peak %.0f MB (+%.0f)", peak, peak - base) +
        (pss?.let { String.format(Locale.US, " · process %.0f MB", it) } ?: "")
}

/** Write [text] to the cache and open the share sheet (FileProvider; cache only). */
fun shareCsv(context: Context, name: String, text: String) {
    val dir = File(context.cacheDir, "share").apply { mkdirs() }
    val f = File(dir, name).apply { writeText(text) }
    val uri = FileProvider.getUriForFile(context, "${context.packageName}.files", f)
    val send = Intent(Intent.ACTION_SEND).apply {
        type = "text/csv"
        putExtra(Intent.EXTRA_STREAM, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    context.startActivity(Intent.createChooser(send, "Share CSV").addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
}

/** Decode a JPEG's bytes for viewing (RGB_565: half the memory, ample for looking). */
suspend fun decodeForView(bytes: ByteArray?): Bitmap? = withContext(Dispatchers.Default) {
    bytes ?: return@withContext null
    BitmapFactory.decodeByteArray(bytes, 0, bytes.size,
        BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.RGB_565 })
}

/** An image with pinch-zoom and pan (the Photo tab). */
@Composable
fun ZoomableImage(bmp: Bitmap, modifier: Modifier = Modifier) {
    var scale by remember(bmp) { mutableFloatStateOf(1f) }
    var offset by remember(bmp) { mutableStateOf(Offset.Zero) }
    Box(modifier.fillMaxWidth().aspectRatio(bmp.width.toFloat() / bmp.height).clipToBounds()
        .border(1.dp, CardBorder, RoundedCornerShape(12.dp))
        .pointerInput(bmp) {
            detectTransformGestures { _, pan, zoom, _ ->
                val s = (scale * zoom).coerceIn(1f, 12f)
                val maxX = size.width * (s - 1f) / 2f
                val maxY = size.height * (s - 1f) / 2f
                offset = Offset((offset.x + pan.x).coerceIn(-maxX, maxX), (offset.y + pan.y).coerceIn(-maxY, maxY))
                scale = s
            }
        }) {
        Image(bmp.asImageBitmap(), contentDescription = "The photograph as digitised",
            contentScale = ContentScale.FillBounds,
            modifier = Modifier.fillMaxSize().graphicsLayer(scaleX = scale, scaleY = scale,
                translationX = offset.x, translationY = offset.y))
    }
}

/** A loading placeholder for a tab whose image is still being decoded. */
@Composable
fun LoadingBox(text: String) {
    Column(Modifier.fillMaxWidth().padding(24.dp), horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(8.dp)) {
        CircularProgressIndicator(Modifier.size(24.dp), color = Primary, strokeWidth = 3.dp)
        Text(text, style = MaterialTheme.typography.bodySmall, color = Muted)
    }
}
