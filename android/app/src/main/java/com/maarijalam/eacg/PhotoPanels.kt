package com.maarijalam.eacg

import android.graphics.Bitmap
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
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
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
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
    /** Taps in BITMAP pixels, TL TR BR BL order, up to four (x, y) pairs. */
    val taps: List<Pair<Double, Double>> = emptyList(),
    val target: KitTargets.Target? = null,
    val busy: Boolean = false,
    val status: String? = null,
    val run: PhotoPipeline.Run? = null,
    val exported: String? = null
)

/** Long side of a picked photograph, px. See the load path. */
private const val PHOTO_MAX_DIM = 3200

private val CORNER_NAMES = listOf("top-left", "top-right", "bottom-right", "bottom-left")

/**
 * Score a photograph of a printed kit sheet on the handset.
 *
 * The on-device half of the real-photograph experiment: the same photograph
 * that `score_kit_photo.py` scores on the desktop, scored here by the ported
 * pipeline, so the two can be compared number for number and the phone's
 * latency measured on the work it would really do.
 *
 * Four taps stand in for FR1 -- roughly on the corners of the PRINTED IMAGE,
 * where the pink grid meets white paper -- and are then refined to sub-pixel
 * by [PageCorners], so a finger's error does not become a scale error.
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
                exported = null, taps = emptyList())
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
                    status = "Tap the ${CORNER_NAMES[0]} corner of the printed grid")
            }.onFailure { e ->
                state = state.copy(busy = false, status = "Load failed: ${e.message}")
            }
        }
    }

    fun score() {
        val bmp = state.bitmap ?: return
        if (state.taps.size != 4) return
        scope.launch {
            state = state.copy(busy = true, status = "Scoring", run = null, exported = null)
            try {
                val truth = state.target?.let { TruthWaveform.load(context, it.truthAsset) }
                val rough = DoubleArray(8).also { a ->
                    state.taps.forEachIndexed { i, (x, y) -> a[2 * i] = x; a[2 * i + 1] = y }
                }
                val run = PhotoPipeline.run(bmp, rough, state.target, truth)
                state = state.copy(busy = false, run = run,
                    status = run.error ?: "Scored in ${run.timings.pipelineMs} ms (pipeline)")
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
                    "tap the four corners of the printed grid: top-left, top-right, " +
                    "bottom-right, bottom-left. Rough taps are fine; they are refined.",
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

            state.preview?.let { preview ->
                val bmp = state.bitmap!!
                Box(
                    Modifier.fillMaxWidth()
                        .aspectRatio(bmp.width.toFloat() / bmp.height)
                        .pointerInput(bmp, state.taps.size) {
                            detectTapGestures { p ->
                                if (state.taps.size >= 4 || state.busy) return@detectTapGestures
                                // The box has the bitmap's aspect, so the map is linear.
                                val x = p.x / size.width * bmp.width
                                val y = p.y / size.height * bmp.height
                                val taps = state.taps + (x.toDouble() to y.toDouble())
                                state = state.copy(taps = taps, run = null,
                                    status = if (taps.size < 4)
                                        "Tap the ${CORNER_NAMES[taps.size]} corner"
                                    else "Four corners set; score when ready")
                            }
                        }
                ) {
                    Image(preview.asImageBitmap(), contentDescription = "photograph",
                        contentScale = ContentScale.FillBounds, modifier = Modifier.fillMaxSize())
                    Canvas(Modifier.fillMaxSize()) {
                        val pts = state.taps.map { (x, y) ->
                            Offset((x / bmp.width * size.width).toFloat(),
                                (y / bmp.height * size.height).toFloat())
                        }
                        for (i in pts.indices) {
                            drawCircle(Color.Red, 9f, pts[i])
                            if (i > 0) drawLine(Color.Red, pts[i - 1], pts[i], 3f)
                        }
                        if (pts.size == 4) drawLine(Color.Red, pts[3], pts[0], 3f)
                        state.run?.let { r ->
                            val c = r.corners
                            val q = (0 until 4).map {
                                Offset((c[2 * it] / bmp.width * size.width).toFloat(),
                                    (c[2 * it + 1] / bmp.height * size.height).toFloat())
                            }
                            for (i in 0 until 4) drawLine(Color.Green, q[i], q[(i + 1) % 4], 3f)
                        }
                    }
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(
                        onClick = { state = state.copy(taps = emptyList(), run = null,
                            status = "Tap the ${CORNER_NAMES[0]} corner") },
                        enabled = !state.busy && state.taps.isNotEmpty(),
                        contentPadding = PaddingValues(horizontal = 10.dp)
                    ) { Text("Reset corners") }
                    Button(
                        onClick = { score() },
                        enabled = !state.busy && state.taps.size == 4,
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
        Stat("corners moved (max)", "%.1f px".format(run.cornerShiftMaxPx))
        Stat("edge fit residual (max)", "%.2f px".format(run.edgeRmseMaxPx))
        Stat("rectified", "${run.rectW} x ${run.rectH}")
        run.truePxPerMm?.let { Stat("true px/mm (from sheet size)", "%.4f".format(it)) }
        run.location?.let { Stat("pulse cut", if (it.pulseX1 >= 0) "yes" else "no") }

        for (c in run.calibrations) {
            Text("Calibration: ${c.name}", style = MaterialTheme.typography.labelMedium)
            Stat("px/mm", "%.4f".format(c.pxPerMm) +
                (c.errorPct?.let { "  (%+.3f%%)".format(it) } ?: ""))
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
