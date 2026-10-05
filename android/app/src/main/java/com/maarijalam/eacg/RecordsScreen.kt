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
fun RecordsScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    var records by remember { mutableStateOf<List<RecordStore.Meta>?>(null) }
    var totalBytes by remember { mutableStateOf(0L) }
    var open by remember { mutableStateOf<RecordStore.Meta?>(null) }
    var reload by remember { mutableStateOf(0) }

    LaunchedEffect(reload) {
        records = RecordStore.list(context)
        totalBytes = RecordStore.totalBytes(context)
    }

    val current = open
    if (current != null) {
        BackHandler { open = null }
        RecordDetail(current, modifier, onBack = { open = null },
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

private enum class View(val label: String) { LEADS("Leads"), PAGE("Page + trace"), PHOTO("Photo") }

@Composable
private fun RecordDetail(m: RecordStore.Meta, modifier: Modifier, onBack: () -> Unit, onDeleted: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var view by remember { mutableStateOf(View.LEADS) }
    var leads by remember { mutableStateOf<List<SheetPipeline.LeadResult>?>(null) }
    var truth by remember { mutableStateOf<Map<String, DoubleArray>?>(null) }
    var showTruth by remember { mutableStateOf(true) }
    var traces by remember { mutableStateOf<List<SheetPipeline.RowTrace>>(emptyList()) }
    var page by remember { mutableStateOf<Bitmap?>(null) }
    var photo by remember { mutableStateOf<Bitmap?>(null) }
    var showTrace by remember { mutableStateOf(true) }
    var message by remember { mutableStateOf<String?>(null) }
    var confirmDelete by remember { mutableStateOf(false) }

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
    // Full-resolution images are decoded only when their view is open.
    LaunchedEffect(m.id, view) {
        when (view) {
            View.PAGE -> if (page == null) page = RecordStore.image(context, m.id, RecordStore.PAGE)
            View.PHOTO -> if (photo == null) photo = RecordStore.image(context, m.id, RecordStore.PHOTO)
            View.LEADS -> Unit
        }
    }

    if (confirmDelete) {
        AlertDialog(onDismissRequest = { confirmDelete = false },
            title = { Text("Delete this record?") },
            text = { Text("The photograph, the page and the digitised leads are removed from this " +
                "phone. Exported CSV files in Downloads are not affected.") },
            confirmButton = {
                TextButton(onClick = {
                    confirmDelete = false
                    scope.launch { RecordStore.delete(context, m.id); onDeleted() }
                }) { Text("Delete") }
            },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text("Cancel") } })
    }

    Column(modifier.padding(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            TextButton(onClick = onBack) { Text("‹ Records") }
        }
        Text(when_(m.createdMs), style = MaterialTheme.typography.titleMedium)
        Text("${tracerLabel(m)} · corners ${if (m.cornersAuto) "automatic" else "by hand"} · " +
            (m.geometry?.let { if (it == "grid") "grid fit · " else "corners only · " } ?: "") +
            String.format(Locale.US, "%s %.3f px/mm (%d/9 ticks)", m.calibration, m.pxPerMm, m.ticksFound) +
            (m.target?.let { " · test sheet $it" } ?: ""),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant)
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            View.entries.forEach { v ->
                FilterChip(selected = view == v, onClick = { view = v }, label = { Text(v.label) })
            }
        }
        message?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
        Box(Modifier.weight(1f).fillMaxWidth()) {
            when (view) {
                View.LEADS -> Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    val ls = leads
                    if (ls == null) CircularProgressIndicator() else {
                        if (truth != null) {
                            Row(verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                Switch(checked = showTruth, onCheckedChange = { showTruth = it })
                                Text("Show the PTB-XL truth (grey)", style = MaterialTheme.typography.bodySmall)
                            }
                        }
                        for (row in 0 until 4) {
                            val rowLeads = ls.filter {
                                if (row < 3) !it.rhythm && SheetLayout.LAYOUT_3X4[row].contains(it.name) else it.rhythm
                            }
                            Text(rowLeads.joinToString("   ") { l ->
                                l.name + (l.pcc?.let { String.format(Locale.US, "  r=%.2f", it) } ?: "")
                            }, style = MaterialTheme.typography.labelSmall, fontFamily = FontFamily.Monospace)
                            LeadRow(rowLeads, if (showTruth) truth else null)
                        }
                        m.medianPcc?.let {
                            Text(String.format(Locale.US, "Median lead r = %.3f over %d leads (as-is).",
                                it, m.leadsScored), style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
                View.PAGE -> Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Switch(checked = showTrace, onCheckedChange = { showTrace = it })
                        Text("Digitised trace over the page · pinch to zoom",
                            style = MaterialTheme.typography.bodySmall)
                    }
                    val p = page
                    if (p == null) Text("Loading the page…", style = MaterialTheme.typography.bodySmall)
                    else ZoomableImage(p, m.pageW, m.pageH, if (showTrace) traces else emptyList())
                }
                View.PHOTO -> Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text("The photograph as digitised · pinch to zoom",
                        style = MaterialTheme.typography.bodySmall)
                    val p = photo
                    if (p == null) Text("Loading the photograph…", style = MaterialTheme.typography.bodySmall)
                    else ZoomableImage(p, p.width, p.height, emptyList())
                }
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = {
                scope.launch {
                    message = runCatching {
                        val name = "eacg_sheet_${m.target ?: "photo"}_${m.tracer.lowercase(Locale.US)}_${m.id}.csv"
                        "Exported to " + Exporter.exportText(context, name, RecordStore.csv(context, m.id))
                    }.getOrElse { "Export failed: ${it.message}" }
                }
            }, modifier = Modifier.weight(1f)) { Text("Export CSV") }
            OutlinedButton(onClick = { confirmDelete = true }, modifier = Modifier.weight(1f)) { Text("Delete") }
        }
    }
}

/**
 * An image with pinch-zoom and pan, and optionally the traced rows drawn on
 * it in the image's own pixel coordinates ([imgW] x [imgH]).
 */
@Composable
private fun ZoomableImage(bmp: Bitmap, imgW: Int, imgH: Int, traces: List<SheetPipeline.RowTrace>) {
    var scale by remember(bmp) { mutableFloatStateOf(1f) }
    var offset by remember(bmp) { mutableStateOf(Offset.Zero) }
    val traceColour = Color(0xFF1565C0)
    Box(Modifier.fillMaxWidth().aspectRatio(imgW.toFloat() / imgH).clipToBounds()
        .pointerInput(bmp) {
            detectTransformGestures { _, pan, zoom, _ ->
                val s = (scale * zoom).coerceIn(1f, 12f)
                // Keep the image covering the frame: offset bounded by the overhang.
                val maxX = size.width * (s - 1f) / 2f
                val maxY = size.height * (s - 1f) / 2f
                offset = Offset((offset.x + pan.x).coerceIn(-maxX, maxX),
                    (offset.y + pan.y).coerceIn(-maxY, maxY))
                scale = s
            }
        }) {
        Box(Modifier.fillMaxSize().graphicsLayer(scaleX = scale, scaleY = scale,
            translationX = offset.x, translationY = offset.y)) {
            Image(bmp.asImageBitmap(), contentDescription = null, contentScale = ContentScale.FillBounds,
                modifier = Modifier.fillMaxSize())
            if (traces.isNotEmpty()) {
                Canvas(Modifier.fillMaxSize()) {
                    val k = size.width / imgW
                    val stroke = Stroke(width = 1.5.dp.toPx() / scale)
                    for (t in traces) {
                        val path = Path()
                        var pen = false
                        for (i in t.pos.indices) {
                            val y = t.pos[i]
                            if (!y.isFinite()) { pen = false; continue }
                            val px = (t.x0 + i) * k
                            val py = y.toFloat() * k
                            if (pen) path.lineTo(px, py) else path.moveTo(px, py)
                            pen = true
                        }
                        drawPath(path, traceColour, style = stroke)
                    }
                }
            }
        }
    }
}
