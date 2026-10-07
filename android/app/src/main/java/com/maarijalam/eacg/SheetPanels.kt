package com.maarijalam.eacg

import android.graphics.Bitmap
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale

private const val SHEET_PHOTO_MAX_DIM = 3200

private data class SheetState(
    val bitmap: Bitmap? = null,
    val preview: Bitmap? = null,
    val corners: DoubleArray? = null,
    /** True while the corners are exactly what [SheetCorners.find] returned. */
    val cornersAuto: Boolean = false,
    val cornersRefined: Boolean? = null,
    val target: KitTargets.Target? = null,
    val busy: Boolean = false,
    val status: String = "",
    val result: SheetPipeline.Result? = null,
    val truth12: Map<String, DoubleArray>? = null,
    val exported: String? = null,
    /** Regularise the page on its printed grid (SheetPipeline.Geometry.GRID). */
    val gridFit: Boolean = true
)

/**
 * The whole printed sheet, photograph to 12 leads, on the phone.
 *
 * Pick a photograph; the sheet's corners are found automatically
 * ([SheetCorners]) and shown in the [CornerEditor] so they can be checked
 * and, if need be, dragged; then [SheetPipeline] digitises all 12 leads and
 * the rhythm strip with the learned tracer (b3) or the classical one. If the
 * photograph is of one of the printed test sheets, choose it and every lead
 * is scored against its PTB-XL truth.
 */
@Composable
fun SheetSection(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var state by remember { mutableStateOf(SheetState()) }
    var targets by remember { mutableStateOf<List<KitTargets.Target>>(emptyList()) }

    LaunchedEffect(Unit) {
        targets = runCatching { KitTargets.load(context) }.getOrDefault(emptyList())
    }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        scope.launch {
            state = state.copy(busy = true, status = "Loading and finding the sheet",
                result = null, exported = null, corners = null)
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
                state = state.copy(
                    bitmap = bmp, preview = small, busy = false,
                    corners = f?.corners ?: CornerGeometry.initialCorners(bmp.width, bmp.height),
                    cornersAuto = f != null, cornersRefined = f?.refined,
                    status = if (f != null)
                        "Corners found automatically (${if (f.refined) "refined to the grid edge" else "coarse only"}). " +
                            "Check them in the close-ups; drag any that are off."
                    else "Could not find the sheet automatically (${found.exceptionOrNull()?.message}). Drag the four corners."
                )
            }.onFailure { e ->
                state = state.copy(busy = false, status = "Load failed: ${e.message}")
            }
        }
    }

    fun digitise(tracer: SheetPipeline.Tracer) {
        val bmp = state.bitmap ?: return
        val corners = state.corners?.copyOf() ?: return
        scope.launch {
            state = state.copy(busy = true, result = null, exported = null,
                status = "Digitising 12 leads (${tracer.name.lowercase(Locale.US)})...")
            try {
                val truth = state.target?.truth12Asset?.let { SheetPipeline.loadTruth12(context, it) }
                val r = SheetPipeline.run(context, bmp, corners, tracer, truth,
                    cornersAuto = state.cornersAuto, cornersRefined = state.cornersRefined,
                    geometry = if (state.gridFit) SheetPipeline.Geometry.GRID else SheetPipeline.Geometry.CORNERS)
                val total = r.pipelineMs
                state = state.copy(busy = false, result = r, truth12 = truth,
                    status = "Done in ${total} ms. Saving to Records...")
                // Every digitisation is kept (RecordStore); a failed save is
                // reported but never discards the result on screen.
                val saved = runCatching {
                    val csv = SheetPipeline.csv(r, "picked ${bmp.width}x${bmp.height}",
                        state.target?.name, Exporter.stamp())
                    RecordStore.save(context, r, bmp, state.target?.name, csv)
                }
                state = state.copy(status = "Done in ${total} ms. " + saved.fold(
                    { "Saved to Records." }, { "Not saved to Records: ${it.message}" }))
            } catch (c: CancellationException) {
                throw c
            } catch (e: Throwable) {
                state = state.copy(busy = false,
                    status = "Failed: ${e::class.java.simpleName}: ${e.message}")
            }
        }
    }

    Card(modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("Whole sheet: all 12 leads", style = MaterialTheme.typography.titleSmall)
            Text("Pick a photograph of a 3x4 + rhythm printout. The corners are found " +
                "automatically; check them, then digitise.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            OutlinedButton(
                onClick = { picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                enabled = !state.busy, modifier = Modifier.fillMaxWidth()
            ) { Text("Pick photograph") }
            if (targets.isNotEmpty()) {
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    targets.forEach { t ->
                        FilterChip(selected = state.target?.name == t.name,
                            onClick = { state = state.copy(target = if (state.target?.name == t.name) null else t, result = null) },
                            label = { Text(t.label, style = MaterialTheme.typography.labelSmall) })
                    }
                }
            }
            val bmp = state.bitmap
            val preview = state.preview
            val corners = state.corners
            if (bmp != null && preview != null && corners != null) {
                CornerEditor(photo = bmp, preview = preview, corners = corners,
                    onCornersChange = { c ->
                        state = state.copy(corners = c, cornersAuto = false, result = null, exported = null)
                    },
                    enabled = !state.busy, refined = null)
                CornerCheckRow(bmp, corners)
                FilterChip(selected = state.gridFit, enabled = !state.busy,
                    onClick = { state = state.copy(gridFit = !state.gridFit, result = null) },
                    label = { Text("Grid fit: straighten the page on its printed 5 mm grid",
                        style = MaterialTheme.typography.labelSmall) })
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = { digitise(SheetPipeline.Tracer.MODEL) }, enabled = !state.busy,
                        modifier = Modifier.weight(1f)) { Text("Digitise (learned)") }
                    OutlinedButton(onClick = { digitise(SheetPipeline.Tracer.CLASSICAL) }, enabled = !state.busy,
                        modifier = Modifier.weight(1f)) { Text("Classical") }
                }
            }
            if (state.status.isNotEmpty()) {
                Text(state.status, style = MaterialTheme.typography.bodySmall)
            }
            state.result?.let { r ->
                SheetResultPanel(r, state.truth12)
                OutlinedButton(onClick = {
                    scope.launch {
                        val stamp = Exporter.stamp()
                        val name = "eacg_sheet_${state.target?.name ?: "photo"}_${r.tracer.name.lowercase(Locale.US)}_$stamp.csv"
                        state = state.copy(exported = runCatching {
                            Exporter.exportText(context, name,
                                SheetPipeline.csv(r, "picked ${bmp?.width}x${bmp?.height}", state.target?.name, stamp))
                        }.getOrElse { "export failed: ${it.message}" })
                    }
                }, modifier = Modifier.fillMaxWidth()) { Text("Export 12-lead CSV") }
                state.exported?.let {
                    Text(it, style = MaterialTheme.typography.labelSmall, fontFamily = FontFamily.Monospace)
                }
            }
        }
    }
}

@Composable
private fun SheetResultPanel(r: SheetPipeline.Result, truth12: Map<String, DoubleArray>?) {
    fun f(fmt: String, v: Double) = String.format(Locale.US, fmt, v)
    HorizontalDivider()
    Text("${if (r.tracer == SheetPipeline.Tracer.MODEL) "Learned tracer (b3)" else "Classical tracer"}; " +
        "corners ${if (r.cornersAuto) "automatic" else "by hand"}; " +
        "${r.calibration} ${f("%.3f", r.pxPerMm)} px/mm (${r.ticksFound}/9 ticks); t0 from ${r.t0Source}",
        style = MaterialTheme.typography.labelSmall)
    Text(r.gridFit?.let {
        "Geometry: grid fit (${f("%.3f", it.rmsMm)} mm residual, ${it.nVertical}x${it.nHorizontal} rules)"
    } ?: ("Geometry: corners only" + (r.gridNote?.let { " -- $it" } ?: "")),
        style = MaterialTheme.typography.labelSmall)
    Text(r.timingsMs.entries.joinToString("  ") { "${it.key} ${it.value} ms" },
        style = MaterialTheme.typography.labelSmall, fontFamily = FontFamily.Monospace)
    r.memoryBaseline?.let { base ->
        val peak = r.memoryPeaks.values.maxOfOrNull { it.nativeBytes } ?: base.nativeBytes
        Text("Memory: native peak ${f("%.0f", StageMemory.mb(peak))} MB " +
            "(+${f("%.0f", StageMemory.mb(peak - base.nativeBytes))} over start)" +
            (r.pssEndMb?.let { "; process ${f("%.0f", it)} MB" } ?: ""),
            style = MaterialTheme.typography.labelSmall, fontFamily = FontFamily.Monospace)
    }
    // The sheet as printed: three rows of four leads and the rhythm strip.
    for (row in 0 until 4) {
        val leads = r.leads.filter { if (row < 3) !it.rhythm && SheetLayout.LAYOUT_3X4[row].contains(it.name) else it.rhythm }
        val label = leads.joinToString("   ") { l ->
            l.name + (l.pcc?.let { "  r=" + f("%.2f", it) } ?: "")
        }
        Text(label, style = MaterialTheme.typography.labelSmall, fontFamily = FontFamily.Monospace)
        LeadRow(leads, truth12)
    }
    val scored = r.leads.filter { !it.rhythm }.mapNotNull { it.pcc }
    if (scored.isNotEmpty()) {
        val s = scored.sorted()
        val med = if (s.size % 2 == 1) s[s.size / 2] else 0.5 * (s[s.size / 2 - 1] + s[s.size / 2])
        Text("Against the truth: median lead r = ${f("%.3f", med)} over ${s.size} leads " +
            "(as-is, no alignment); ${s.count { it >= 0.9 }} leads >= 0.9.",
            style = MaterialTheme.typography.bodySmall)
    }
}

/** One printed row: each lead's digitised trace (blue) over its truth (grey). */
@Composable
internal fun LeadRow(leads: List<SheetPipeline.LeadResult>, truth12: Map<String, DoubleArray>?) {
    val blue = Color(0xFF2A78D6)
    val grey = Color(0x99555555)
    val grid = Color(0x33E34948)
    Canvas(modifier = Modifier.fillMaxWidth().height(90.dp)) {
        val w = size.width
        val h = size.height
        val secToX = w / SheetLayout.RECORDING_S.toFloat()
        // 0.2 s / 0.5 mV major grid.
        var t = 0f
        while (t <= SheetLayout.RECORDING_S) {
            drawLine(grid, Offset(t * secToX, 0f), Offset(t * secToX, h)); t += 0.2f
        }
        val mvToY = h / 3.0f                      // 3 mV across the row
        val mid = h * 0.55f
        var mv = -1.5f
        while (mv <= 1.5f) { drawLine(grid, Offset(0f, mid - mv * mvToY), Offset(w, mid - mv * mvToY)); mv += 0.5f }
        for (l in leads) {
            val a = (l.tStartS * SheetLayout.FS).toInt()
            truth12?.get(l.name)?.let { tr ->
                val seg = tr.copyOfRange(a, minOf(tr.size, a + l.mv.size))
                val med = SheetLayout.median(seg)
                drawPath(path(seg.map { it - med }.toDoubleArray(), l.tStartS, secToX, mid, mvToY), grey,
                    style = Stroke(width = 2f))
            }
            drawPath(path(l.mv, l.tStartS, secToX, mid, mvToY), blue, style = Stroke(width = 2.5f))
        }
    }
}

private fun path(v: DoubleArray, t0: Double, secToX: Float, mid: Float, mvToY: Float): Path {
    val p = Path()
    var pen = false
    for (i in v.indices) {
        val y = v[i]
        if (!y.isFinite()) { pen = false; continue }
        val x = ((t0 + i / SheetLayout.FS) * secToX).toFloat()
        val yy = mid - y.toFloat() * mvToY
        if (pen) p.lineTo(x, yy) else p.moveTo(x, yy)
        pen = true
    }
    return p
}
