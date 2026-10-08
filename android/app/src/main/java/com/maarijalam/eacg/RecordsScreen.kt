package com.maarijalam.eacg

import android.graphics.Bitmap
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
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
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import com.maarijalam.eacg.ui.theme.RowDivider
import com.maarijalam.eacg.ui.theme.Primary
import com.maarijalam.eacg.ui.theme.Muted
import com.maarijalam.eacg.ui.theme.InputOutline
import com.maarijalam.eacg.ui.theme.DataText
import com.maarijalam.eacg.ui.theme.CardSurface
import com.maarijalam.eacg.ui.SectionLabel
import com.maarijalam.eacg.ui.EaIcons
import com.maarijalam.eacg.ui.EaCard
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.Role
import androidx.compose.foundation.layout.heightIn
import androidx.compose.material3.Icon
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.OutlinedTextField

/**
 * The Records tab: every whole-sheet digitisation saved by [RecordStore],
 * newest first. Tap one to see its 12 leads, the page with the digitised
 * trace drawn over it, and the original photograph.
 */
@Composable
fun RecordsScreen(
    modifier: Modifier = Modifier,
    /** A record to open on arrival (Home's Recent list), then [onOpenHandled]. */
    openRequest: RecordStore.Meta? = null,
    onOpenHandled: () -> Unit = {},
) {
    val context = LocalContext.current
    val proto = LocalAppSettings.current.prototypeMode
    var records by remember { mutableStateOf<List<RecordStore.Meta>?>(null) }
    var totalBytes by remember { mutableStateOf(0L) }
    var open by remember { mutableStateOf<RecordStore.Meta?>(null) }
    var reload by remember { mutableStateOf(0) }
    var query by remember { mutableStateOf("") }
    var filter by remember { mutableStateOf(RecordFilter.ALL) }

    LaunchedEffect(openRequest) {
        if (openRequest != null) { open = openRequest; onOpenHandled() }
    }

    LaunchedEffect(reload) {
        records = RecordStore.list(context)
        totalBytes = RecordStore.totalBytes(context)
    }

    val current = open
    if (current != null) {
        BackHandler { open = null }
        RecordResult(current, modifier, onBack = { open = null },
            onDeleted = { open = null; reload++ })
        return
    }

    val all = records
    val filters = RecordFilter.entries.filter { it != RecordFilter.BENCHMARKED || proto }
    if (filter !in filters) filter = RecordFilter.ALL
    val shown = all.orEmpty().filter { filter.accepts(it) && matches(it, query) }

    Column(modifier.fillMaxSize()) {
        Text("Records", style = MaterialTheme.typography.headlineSmall,
            modifier = Modifier.padding(start = 20.dp, end = 20.dp, top = 16.dp, bottom = 8.dp).semantics { heading() })
        OutlinedTextField(value = query, onValueChange = { query = it }, singleLine = true,
            placeholder = { Text("Search by name, target or date", style = MaterialTheme.typography.bodyLarge) },
            leadingIcon = { Icon(EaIcons.Search, contentDescription = null, tint = Muted, modifier = Modifier.size(20.dp)) },
            shape = RoundedCornerShape(26.dp),
            colors = OutlinedTextFieldDefaults.colors(unfocusedBorderColor = InputOutline, focusedBorderColor = Primary,
                unfocusedContainerColor = CardSurface, focusedContainerColor = CardSurface),
            modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp))
        UnderlineTabs(filters.map { f -> f.label + (all?.let { " (${it.count(f::accepts)})" } ?: "") },
            filters.indexOf(filter), Modifier.padding(top = 8.dp)) { filter = filters[it] }

        when {
            all == null -> LoadingBox("Loading records")
            all.isEmpty() -> Text("No records yet. Every sheet you digitise on the Scan tab is kept here.",
                Modifier.padding(20.dp), style = MaterialTheme.typography.bodyMedium, color = Muted)
            else -> LazyColumn(Modifier.fillMaxSize(),
                contentPadding = androidx.compose.foundation.layout.PaddingValues(start = 16.dp, end = 16.dp, top = 12.dp, bottom = 16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)) {
                val groups = shown.groupBy { dayLabel(it.createdMs) }
                if (shown.isEmpty()) item {
                    Text("Nothing matches.", style = MaterialTheme.typography.bodyMedium, color = Muted)
                }
                for ((day, list) in groups) {
                    item(key = "h_$day") { SectionLabel(day, Modifier.padding(top = 4.dp)) }
                    item(key = "g_$day") {
                        EaCard {
                            list.forEachIndexed { i, m ->
                                if (i > 0) HorizontalDivider(color = RowDivider)
                                RecordRow(m, proto) { open = m }
                            }
                        }
                    }
                }
                item(key = "footer") {
                    Text("Stored privately on this phone. Each record keeps the photo, the straightened page and a " +
                        "12-lead CSV. " + String.format(Locale.US, "%d record%s, %.1f MB.", all.size,
                        if (all.size == 1) "" else "s", totalBytes / 1e6),
                        style = MaterialTheme.typography.labelSmall, color = Muted, textAlign = TextAlign.Center,
                        modifier = Modifier.fillMaxWidth().padding(top = 8.dp))
                }
            }
        }
    }
}

/** The Records filters (plan §3.6); Benchmarked only in prototype mode. */
private enum class RecordFilter(val label: String) {
    ALL("All"), CHECK("Check timing"), CLASSICAL("Classical"), BENCHMARKED("Benchmarked");

    fun accepts(m: RecordStore.Meta): Boolean = when (this) {
        ALL -> true
        CHECK -> m.quality == "check" || m.quality == "unreliable"
        CLASSICAL -> m.tracer == SheetPipeline.Tracer.CLASSICAL.name
        BENCHMARKED -> m.target != null
    }
}

/** Search over the record's title, target name and date. */
private fun matches(m: RecordStore.Meta, q: String): Boolean {
    if (q.isBlank()) return true
    val hay = listOfNotNull(recordTitle(m.target), m.target, recordWhen(m.createdMs), dayLabel(m.createdMs),
        SimpleDateFormat("d MMM yyyy", Locale.getDefault()).format(Date(m.createdMs)))
        .joinToString(" ").lowercase(Locale.getDefault())
    return q.trim().lowercase(Locale.getDefault()).split(' ').filter { it.isNotEmpty() }.all { it in hay }
}

/** Name; time · tracer · geometry; status as icon + text; r in prototype mode. */
@Composable
private fun RecordRow(m: RecordStore.Meta, proto: Boolean, onClick: () -> Unit) {
    Row(Modifier.fillMaxWidth().heightIn(min = 64.dp).clickable(role = Role.Button, onClick = onClick)
        .padding(horizontal = 14.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        RecordThumb(m.id)
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text(recordTitle(m.target), style = MaterialTheme.typography.titleSmall)
            val time = SimpleDateFormat("HH:mm", Locale.getDefault()).format(Date(m.createdMs))
            val tracer = if (m.tracer == SheetPipeline.Tracer.MODEL.name) "Learned" else "Classical"
            val geometry = when (m.geometry) { "grid" -> "grid fit"; "corners" -> "corners only"; else -> null }
            Text(listOfNotNull(time, tracer, geometry).joinToString(" · "), style = MaterialTheme.typography.bodySmall,
                color = Muted)
        }
        Column(horizontalAlignment = Alignment.End, verticalArrangement = Arrangement.spacedBy(2.dp)) {
            QualityStatus(m.quality)
            if (proto) m.medianPcc?.let {
                Text(String.format(Locale.US, "r %.3f", it), style = DataText, color = Muted)
            }
        }
    }
}

/**
 * A saved record in the shared Result screen (docs/ui_plan.md §3.4). Export
 * is unchanged: the record's own leads.csv to Downloads/EaCG as
 * `eacg_sheet_<target|photo>_<tracer>_<record id>.csv`.
 */
@Composable
private fun RecordResult(m: RecordStore.Meta, modifier: Modifier, onBack: () -> Unit, onDeleted: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var leads by remember(m.id) { mutableStateOf<List<SheetPipeline.LeadResult>?>(null) }
    var truth by remember(m.id) { mutableStateOf<Map<String, DoubleArray>?>(null) }
    var traces by remember(m.id) { mutableStateOf<List<SheetPipeline.RowTrace>>(emptyList()) }
    var page by remember(m.id) { mutableStateOf<Bitmap?>(null) }
    var photo by remember(m.id) { mutableStateOf<Bitmap?>(null) }
    var message by remember(m.id) { mutableStateOf<String?>(null) }

    LaunchedEffect(m.id) {
        leads = runCatching { RecordStore.leads(context, m.id) }.getOrElse {
            message = "Could not read the leads: ${it.message}"; emptyList()
        }
        traces = runCatching { RecordStore.traces(context, m.id) }.getOrDefault(emptyList())
        truth = m.target?.let { name ->
            runCatching {
                KitTargets.load(context).firstOrNull { it.name == name }?.truth12Asset
                    ?.let { SheetPipeline.loadTruth12(context, it) }
            }.getOrNull()
        }
    }
    val exportName = "eacg_sheet_${m.target ?: "photo"}_${m.tracer.lowercase(Locale.US)}_${m.id}.csv"
    val data = ResultData.fromRecord(m, leads)
    ResultScreen(
        data = data, truth12 = truth, onBack = onBack,
        onExport = {
            scope.launch {
                message = runCatching {
                    "Exported to " + Exporter.exportText(context, exportName, RecordStore.csv(context, m.id))
                }.getOrElse { "Export failed: ${it.message}" }
            }
        },
        exportMessage = message, savedNote = "Saved to Records.",
        csvForShare = { exportName to RecordStore.csv(context, m.id) },
        onDelete = { scope.launch { RecordStore.delete(context, m.id); onDeleted() } },
        pageContent = {
            LaunchedEffect(m.id) { if (page == null) page = RecordStore.image(context, m.id, RecordStore.PAGE) }
            val ticks = ResultData.parseTicks(m.ticks)
            val replaced = ResultData.parseTicks(m.ticksRejected)
            PageTab(page, data, traces, m.t0X ?: Double.NaN,
                m.rowY?.split(' ')?.mapNotNull { it.toIntOrNull() } ?: emptyList(),
                ticks, replaced, parseCells(m.cells), m.pageW, m.pageH, truth)
        },
        photoContent = {
            LaunchedEffect(m.id) { if (photo == null) photo = RecordStore.image(context, m.id, RecordStore.PHOTO) }
            photo?.let { ZoomableImage(it) } ?: LoadingBox("Loading the photograph")
        },
        modifier = modifier)
}

/** A record's cell boxes ("name rhythm x0 x1 y0 y1;..."); empty for older records. */
internal fun parseCells(s: String?): List<SheetPipeline.CellBox> =
    s.orEmpty().split(';').filter { it.isNotBlank() }.mapNotNull { e ->
        val f = e.trim().split(' ')
        runCatching { SheetPipeline.CellBox(f[0], f[1].toBoolean(), f[2].toInt(), f[3].toInt(), f[4].toInt(), f[5].toInt()) }
            .getOrNull()
    }
