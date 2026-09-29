package com.maarijalam.eacg

import android.graphics.Bitmap
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Everything the photo section knows, replaced wholesale per change. */
private data class PhotoState(
    val bitmap: Bitmap? = null,
    val preview: Bitmap? = null,
    val label: String? = null,
    /** Corner handles in BITMAP pixels, TL TR BR BL, x then y; null until loaded. */
    val corners: DoubleArray? = null,
    val target: KitTargets.Target? = null,
    val busy: Boolean = false,
    val status: String? = null,
    val run: PhotoPipeline.Run? = null,
    val exported: String? = null
)

/** Long side of a picked photograph, px. See the load path. */
private const val PHOTO_MAX_DIM = 3200

private val CORNER_NAMES = listOf("top-left", "top-right", "bottom-right", "bottom-left")

private const val PLACE_HINT = "Drag each blue circle to where the grid lines END at that " +
    "corner -- the outermost tips of the lines, which is often part-way through a square, " +
    "not the last whole square. Use the magnifier to place it exactly."

/**
 * Score a photograph of a printed kit sheet on the handset.
 *
 * The on-device half of the real-photograph experiment: the same photograph
 * that `score_kit_photo.py` scores on the desktop, scored here by the ported
 * pipeline, so the two can be compared number for number and the phone's
 * latency measured on the work it would really do.
 *
 * Four dragged corners stand in for FR1 -- on the corners of the PRINTED
 * IMAGE, where the pink grid meets white paper, placed with a magnifier by
 * [CornerEditor] -- and are then refined to sub-pixel by [PageCorners], so
 * a finger's error does not become a scale error.
 */
@Composable
fun PhotoScoreSection(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var state by remember { mutableStateOf(PhotoState()) }
    var targets by remember { mutableStateOf<List<KitTargets.Target>>(emptyList()) }

    LaunchedEffect(Unit) {
        targets = runCatching { KitTargets.load(context) }.getOrDefault(emptyList())
    }

    val picker = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        scope.launch {
            state = state.copy(busy = true, status = "Loading photograph", run = null,
                exported = null, corners = null)
            runCatching {
                // 3200 px rather than the picker's default 2560: about 10 px/mm on a
                // printed A4 sheet against 8.6, and 31 MB of native heap.
                val bmp = UriSource(context, uri, maxDimension = PHOTO_MAX_DIM).acquire()
                val small = withContext(Dispatchers.Default) {
                    val m = ImageWarp.toMat(bmp)
                    try { ImageWarp.previewBitmap(m, 1080) } finally { m.release() }
                }
                bmp to small
            }.onSuccess { (bmp, small) ->
                state = state.copy(bitmap = bmp, preview = small, busy = false,
                    label = "picked, ${bmp.width} x ${bmp.height}",
                    corners = CornerGeometry.initialCorners(bmp.width, bmp.height),
                    status = PLACE_HINT)
            }.onFailure { e ->
                state = state.copy(busy = false, status = "Load failed: ${e.message}")
            }
        }
    }

    fun score() {
        val bmp = state.bitmap ?: return
        val rough = state.corners?.copyOf() ?: return
        scope.launch {
            state = state.copy(busy = true, status = "Scoring", run = null, exported = null)
            try {
                val truth = state.target?.let { TruthWaveform.load(context, it.truthAsset) }
                val run = PhotoPipeline.run(bmp, rough, state.target, truth)
                val warn = if (run.cornersFromHand)
                    "  Edge refinement fit poorly (%.1f px), so your corners were used as placed: make sure each sits exactly where the grid lines end."
                        .format(java.util.Locale.US, run.edgeRmseMaxPx)
                else ""
                state = state.copy(busy = false, run = run,
                    status = (run.error ?: "Scored in ${run.timings.pipelineMs} ms (pipeline).") + warn)
            } catch (c: CancellationException) {
                throw c
            } catch (e: Throwable) {
                state = state.copy(busy = false,
                    status = "Scoring failed: ${e::class.java.simpleName}: ${e.message}")
            }
        }
    }

    Card(modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("Photograph of a printed sheet", style = MaterialTheme.typography.titleSmall)
            Text(
                "Pick the photograph, choose which printed record it shows, then " +
                    "drag the four corners to where the grid lines end. Place them " +
                    "exactly: if the sheet's edges cannot be refined, your placement is used as it is.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            OutlinedButton(
                onClick = {
                    picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                },
                enabled = !state.busy, modifier = Modifier.fillMaxWidth()
            ) { Text("Pick photograph") }

            if (targets.isNotEmpty()) {
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    targets.forEach { t ->
                        FilterChip(
                            selected = state.target?.name == t.name,
                            onClick = { state = state.copy(target = t, run = null) },
                            label = { Text(t.label, style = MaterialTheme.typography.labelSmall) }
                        )
                    }
                }
            }

            val bmp = state.bitmap
            val preview = state.preview
            val corners = state.corners
            if (bmp != null && preview != null && corners != null) {
                CornerEditor(
                    photo = bmp,
                    preview = preview,
                    corners = corners,
                    onCornersChange = { c ->
                        state = state.copy(corners = c, run = null, exported = null,
                            status = PLACE_HINT)
                    },
                    enabled = !state.busy,
                    refined = state.run?.corners
                )
                Text("Check each corner: the crosshair should sit on the very tips of the grid lines.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                CornerCheckRow(bmp, corners)
                val nearEdge = CornerGeometry.cornersNearFrameEdge(corners, bmp.width, bmp.height)
                if (nearEdge.isNotEmpty()) {
                    Text(
                        "The ${nearEdge.joinToString(" and ") { CORNER_NAMES[it] }} " +
                            "corner is at the edge of the photograph. Refinement needs white " +
                            "paper beyond every edge of the grid: if the sheet runs out of " +
                            "frame, retake the photograph with a border all round.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error
                    )
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(
                        onClick = {
                            state = state.copy(run = null, exported = null, status = PLACE_HINT,
                                corners = CornerGeometry.initialCorners(bmp.width, bmp.height))
                        },
                        enabled = !state.busy,
                        contentPadding = PaddingValues(horizontal = 10.dp)
                    ) { Text("Reset corners") }
                    Button(
                        onClick = { score() },
                        enabled = !state.busy,
                        modifier = Modifier.weight(1f)
                    ) { Text(if (state.busy) "Working ..." else "Score photograph") }
                }
            }

            state.status?.let {
                Text(it, style = MaterialTheme.typography.bodySmall)
            }
            state.run?.let { run ->
                PhotoResults(run, state.target)
                OutlinedButton(
                    onClick = {
                        scope.launch {
                            state = state.copy(exported = runCatching {
                                "Saved to " + Exporter.export(context, run, state.target,
                                    state.label ?: "photo")
                            }.getOrElse { "Export failed: ${it.message}" })
                        }
                    },
                    enabled = !state.busy, modifier = Modifier.fillMaxWidth()
                ) { Text("Export CSV (FR10)") }
                state.exported?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
            }
        }
    }
}

@Composable
private fun PhotoResults(run: PhotoPipeline.Run, target: KitTargets.Target?) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        run.stripPreview?.let {
            Image(it.asImageBitmap(), contentDescription = "located rhythm strip",
                modifier = Modifier.fillMaxWidth().height(70.dp), contentScale = ContentScale.FillBounds)
        }
        Stat("corners used", if (run.cornersFromHand) "as placed" else "refined")
        Stat("refinement would move (max)", "%.1f px".format(run.cornerShiftMaxPx))
        Stat("refinement edge fit (max)", "%.2f px".format(run.edgeRmseMaxPx))
        Stat("rectified", "${run.rectW} x ${run.rectH}")
        run.pxPerMmFromCorners?.let { Stat("px/mm implied by corners", "%.4f".format(it)) }
        run.location?.let { loc ->
            Stat("pulse cut", if (loc.pulseX1 >= 0) "yes" else "no")
            run.levelAngleDeg?.let { a ->
                Stat("strip levelled", if (run.levelApplied) "%+.2f°".format(a) else "no (%+.2f°)".format(a))
            }
            if (target != null && loc.pulseX1 < 0) {
                Text("The printed 1 mV pulse was not found, so it was traced as signal " +
                    "and counted in the duration anchor: expect a lag and a longer anchor.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error)
            }
        }

        for (c in run.calibrations) {
            Text("Calibration: ${c.name}", style = MaterialTheme.typography.labelMedium)
            Stat("px/mm", "%.4f".format(c.pxPerMm) +
                (c.errorVsCornersPct?.let { "  (%+.2f%% vs corners)".format(it) } ?: ""))
            c.alignment?.let { al ->
                if (al.valid) {
                    Stat("error vs truth waveform", "%+.2f%%".format(al.pxPerMmErrorPct))
                    Stat("PCC (aligned)", "%.4f".format(al.pcc))
                } else {
                    Stat("error vs truth waveform", "not measurable")
                    Text("No alignment with the truth over 8.5 s or more " +
                        "(best: %.1f s%s). The calibration is too far off for the ±10%% search to measure."
                            .format(al.nCompared / PhotoPipeline.FS_OUT,
                                if (al.atBound) ", at the search bound" else ""),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            c.error?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
            c.gate?.let { g ->
                Stat("gate", g.verdict.name)
                g.rejections.forEach { Text("reject: $it", style = MaterialTheme.typography.bodySmall) }
            }
            c.fidelity?.let { f ->
                Stat("PCC (as-is)", "%.4f".format(f.pcc))
                Stat("RMSE", "%.4f mV".format(f.rmseMv))
                Stat("gain ratio", "%.4f".format(f.gainRatio))
            }
            if (c.fidelity == null && target == null && c.digitised != null) {
                Text("No record chosen, so not scored.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            c.digitised?.let { WaveformPlot(it.signalMv) }
        }

        Text("Per-stage latency", style = MaterialTheme.typography.labelMedium)
        val t = run.timings
        Stat("corners (FR1 stand-in)", "${t.cornersMs} ms")
        Stat("rectify", "${t.rectifyMs} ms")
        Stat("locate strip (FR4)", "${t.locateMs} ms")
        Stat("strip mask (FR5)", "${t.stripMs} ms")
        Stat("grid (FR3)", "${t.gridMs} ms")
        run.calibrations.forEach { Stat("trace + gate, ${it.name}", "${it.traceMs} ms") }
        Stat("PIPELINE", "${t.pipelineMs} ms")
        Stat("HARNESS (fidelity)", "${t.harnessMs} ms")
    }
}

@Composable
private fun Stat(label: String, value: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically) {
        Text(label, style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(value, style = MaterialTheme.typography.bodyMedium, fontFamily = FontFamily.Monospace)
    }
}
