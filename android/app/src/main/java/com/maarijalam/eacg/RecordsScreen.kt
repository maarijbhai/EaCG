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
    var records by remember { mutableStateOf<List<RecordStore.Meta>?>(null) }
    var totalBytes by remember { mutableStateOf(0L) }
    var open by remember { mutableStateOf<RecordStore.Meta?>(null) }
    var reload by remember { mutableStateOf(0) }

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

    Column(modifier.padding(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text("Records", style = MaterialTheme.typography.titleLarge)
        val list = records
        when {
            list == null -> CircularProgressIndicator()
            list.isEmpty() -> Text("No records yet. Every sheet you digitise on the Scan tab " +
                "is saved here automatically.", style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            else -> {
                Text("${list.size} record${if (list.size == 1) "" else "s"} · " +
                    String.format(Locale.US, "%.1f MB", totalBytes / 1e6) + " on this phone (private to the app)",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(list, key = { it.id }) { m -> RecordCard(m) { open = m } }
                }
            }
        }
    }
}

private fun when_(ms: Long): String =
    SimpleDateFormat("d MMM yyyy, HH:mm", Locale.getDefault()).format(Date(ms))

private fun tracerLabel(m: RecordStore.Meta) =
    if (m.tracer == SheetPipeline.Tracer.MODEL.name) "Learned tracer" else "Classical tracer"

@Composable
private fun RecordCard(m: RecordStore.Meta, onClick: () -> Unit) {
    val context = LocalContext.current
    var thumb by remember(m.id) { mutableStateOf<Bitmap?>(null) }
    LaunchedEffect(m.id) { thumb = RecordStore.image(context, m.id, RecordStore.THUMB) }
    Card(Modifier.fillMaxWidth().clickable(onClick = onClick),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
        Row(Modifier.padding(10.dp), horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(84.dp).clip(RoundedCornerShape(6.dp))) {
                thumb?.let {
                    Image(it.asImageBitmap(), contentDescription = "Photograph of the sheet",
                        contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())
                }
            }
            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text(when_(m.createdMs), style = MaterialTheme.typography.titleSmall)
                Text("${tracerLabel(m)} · ${m.ticksFound}/9 ticks",
                    style = MaterialTheme.typography.bodySmall)
                val score = m.medianPcc?.let {
                    String.format(Locale.US, " · median r %.2f vs truth", it)
                } ?: ""
                Text((m.target?.let { "Test sheet $it" } ?: "Photograph") + score,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
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
