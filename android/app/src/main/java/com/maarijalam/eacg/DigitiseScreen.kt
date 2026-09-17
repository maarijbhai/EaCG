package com.maarijalam.eacg

import android.graphics.Bitmap
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Everything the screen knows. One object, so adding a field is one edit. */
data class PipelineState(
    val bitmap: Bitmap? = null,
    val status: String = "Ready",
    val busy: Boolean = false,
    val sourceLabel: String? = null,
    val gridPitch: String = "--",
    val latencyMs: String = "--",
    /**
     * Grid estimation time, kept separate from [latencyMs] rather than
     * overwriting it. They measure different stages — decode versus
     * estimate — and the two differ by an order of magnitude, so folding
     * them into one row would quietly relabel whichever number you read.
     */
    val gridMs: String = "--",
    /** Signed error against [ReferenceImage.pxPerMm]; fixtures only. */
    val gridError: String? = null,
    val imageSize: String = "--",
    /**
     * Set only when the loaded image is a bundled fixture with a known true
     * scale. Null for anything the user picked, because for an arbitrary photo
     * there is no right answer to compare against.
     */
    val reference: ReferenceImage? = null,

    // ---- rotation control ------------------------------------------------
    /**
     * Rotation injected by the slider, in degrees, anticlockwise — the same
     * sign convention as `deskew.rotate_image`. This is ground truth: the
     * skew estimator's job is to recover this number back out of the pixels,
     * so it is deliberately a value the user sets rather than one the app
     * discovers.
     */
    val rotationDeg: Float = 0f,
    /**
     * Display-sized copy of [bitmap], made once at load. [bitmap] itself is
     * never drawn: at 3300x2550 it is 33.7 MB as ARGB_8888 and there will be
     * more than one of them on screen shortly.
     */
    val sourcePreview: Bitmap? = null,
    /** Display-sized copy of the rotated image. Null until the slider moves. */
    val rotatedPreview: Bitmap? = null,
    /** True while a warp is in flight, so the slider can say so. */
    val warping: Boolean = false,
    val rotateMs: String = "--"
)

@Composable
fun DigitiseScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var state by remember { mutableStateOf(PipelineState()) }

    /**
     * One load path for every source. The pipeline must not care whether the
     * bitmap came from a fixture or the picker — only the fixture carries a
     * [ReferenceImage] alongside it, and that is used for scoring, not decoding.
     */
    fun load(source: ImageSource, label: String, reference: ReferenceImage?) {
        scope.launch {
            state = state.copy(busy = true, status = "Loading", reference = reference)
            val started = System.nanoTime()
            runCatching {
                val bmp = source.acquire()
                // Estimation is the expensive part and must not run on the
                // main thread; acquire() is already on Dispatchers.IO, this
                // is not, so it gets its own context.
                withContext(Dispatchers.Default) {
                    val grid = GridEstimator.estimate(bmp)
                    // The display copy is made here, on the same background
                    // pass, rather than in the composable. It is a resize of
                    // a possibly 8-megapixel bitmap and has no business on
                    // the main thread.
                    val small = shrinkForDisplay(bmp)
                    Triple(bmp, grid, small)
                }
            }
                .onSuccess { (bmp, grid, preview) ->
                    val ms = (System.nanoTime() - started) / 1_000_000
                    state = state.copy(
                        bitmap = bmp,
                        sourcePreview = preview,
                        rotatedPreview = null,
                        rotationDeg = 0f,
                        rotateMs = "--",
                        busy = false,
                        status = if (grid.valid) "Image loaded" else "Grid estimate failed",
                        sourceLabel = label,
                        imageSize = "${bmp.width} x ${bmp.height}",
                        latencyMs = "$ms",
                        gridPitch = if (grid.valid) "%.3f".format(grid.pxPerMm) else "failed",
                        gridMs = "${grid.elapsedMs}",
                        // Only a fixture has a right answer to be wrong about.
                        gridError = reference?.takeIf { grid.valid }?.let {
                            "%+.2f%%".format((grid.pxPerMm / it.pxPerMm - 1.0) * 100.0)
                        }
                    )
                }
                .onFailure { e ->
                    state = state.copy(
                        busy = false,
                        status = "Failed: ${e.message ?: "unknown error"}"
                    )
                }
        }
    }

    val picker = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        if (uri == null) {
            state = state.copy(status = "No image selected")
            return@rememberLauncherForActivityResult
        }
        load(UriSource(context, uri), "Picked from storage", reference = null)
    }

    /**
     * Applies the slider's angle, off the main thread, without queueing.
     *
     * `collectLatest` is the load-bearing part. A slider drag emits a value
     * every frame, and a Lanczos warp of an 8-megapixel image is far slower
     * than a frame; collecting normally would queue dozens of warps and the
     * image would keep moving for seconds after the finger stopped. Cancelling
     * the previous collection on each new value means at most one warp is in
     * flight and the one that finishes is the one for the angle the user
     * actually chose.
     *
     * Keyed on [PipelineState.bitmap] so the flow restarts when a new image
     * is loaded and dies with the old one.
     */
    LaunchedEffect(state.bitmap) {
        val source = state.bitmap ?: return@LaunchedEffect
        snapshotFlow { state.rotationDeg }.collectLatest { angle ->
            if (angle == 0f) {
                // Not a warp of zero degrees: at 0 the original IS the
                // answer, and warping it anyway would apply an interpolation
                // — a low-pass filter on the grid detail everything
                // downstream measures — for no reason.
                state = state.copy(rotatedPreview = null, rotateMs = "--", warping = false)
                return@collectLatest
            }
            state = state.copy(warping = true)
            val result = runCatching {
                withContext(Dispatchers.Default) {
                    val started = System.nanoTime()
                    val src = ImageWarp.toMat(source)
                    try {
                        val rotated = ImageWarp.rotate(src, angle.toDouble())
                        try {
                            val preview = ImageWarp.previewBitmap(rotated)
                            preview to (System.nanoTime() - started) / 1_000_000
                        } finally {
                            rotated.release()
                        }
                    } finally {
                        src.release()
                    }
                }
            }
            state = result.fold(
                onSuccess = { (preview, ms) ->
                    state.copy(rotatedPreview = preview, rotateMs = "$ms", warping = false)
                },
                onFailure = { e ->
                    state.copy(
                        warping = false,
                        status = "Rotate failed: ${e.message ?: e::class.java.simpleName}"
                    )
                }
            )
        }
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Header()

        OpenCvBanner()

        ImagePanel(
            bitmap = state.rotatedPreview ?: state.sourcePreview,
            busy = state.busy,
            caption = when {
                state.rotatedPreview != null -> "rotated %+.2f deg".format(state.rotationDeg)
                state.sourcePreview != null -> "original"
                else -> null
            }
        )

        RotationControl(
            angle = state.rotationDeg,
            enabled = state.bitmap != null && !state.busy,
            warping = state.warping,
            onChange = { state = state.copy(rotationDeg = it) }
        )

        ReferenceRow(
            enabled = !state.busy,
            onPick = { ref ->
                load(
                    source = AssetSource(context, ref.assetPath),
                    label = "${ref.superclass} @ ${ref.renderDpi} dpi (known)",
                    reference = ref
                )
            }
        )

        Button(
            onClick = {
                picker.launch(
                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                )
            },
            enabled = !state.busy,
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp)
        ) {
            Text("Select ECG image", style = MaterialTheme.typography.titleMedium)
        }

        ResultsPanel(state)
    }
}

/**
 * States the OpenCV load outcome, and states it loudly when it failed.
 *
 * Read once at composition rather than observed: [OpenCvRuntime.initialise]
 * runs in `Application.onCreate`, which has already completed by the time any
 * composable runs, and the verdict never changes afterwards. A `remember`
 * makes that assumption explicit instead of re-reading a value that cannot
 * move.
 *
 * On success this is a single quiet line. The asymmetry is deliberate: a
 * working dependency should not occupy the screen, but a missing native
 * library must not be something the user has to go looking in logcat for.
 */
@Composable
private fun OpenCvBanner() {
    val status = remember { OpenCvRuntime.status }
    when (status) {
        is OpenCvRuntime.Status.Ready -> Text(
            "OpenCV ${status.version} loaded",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        is OpenCvRuntime.Status.Failed -> ErrorCard(
            title = "OpenCV failed to load",
            body = status.reason +
                " Every stage after image loading needs it, so nothing " +
                "downstream of this will work."
        )

        OpenCvRuntime.Status.Pending -> ErrorCard(
            title = "OpenCV was never initialised",
            body = "Application.onCreate did not run OpenCvRuntime.initialise(). " +
                "Check that android:name=\".EaCGApplication\" is still on the " +
                "<application> tag in the manifest."
        )
    }
}

@Composable
private fun ErrorCard(title: String, body: String) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.errorContainer,
            contentColor = MaterialTheme.colorScheme.onErrorContainer
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Text(title, style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold)
            Text(body, style = MaterialTheme.typography.bodySmall,
                fontFamily = FontFamily.Monospace)
        }
    }
}

@Composable
private fun Header() {
    Column {
        Text(
            "ECG Digitiser",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.SemiBold
        )
        Text(
            "On-device paper ECG reconstruction",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

/**
 * Fixture loaders. Small and secondary to the picker on purpose: the picker is
 * for exploring, these are for answering.
 */
@Composable
private fun ReferenceRow(enabled: Boolean, onPick: (ReferenceImage) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(
            "Known-scale fixtures",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            REFERENCE_IMAGES.forEach { ref ->
                OutlinedButton(
                    onClick = { onPick(ref) },
                    enabled = enabled,
                    contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp),
                    modifier = Modifier.weight(1f)
                ) {
                    Text(ref.label, style = MaterialTheme.typography.labelMedium)
                }
            }
        }
    }
}

@Composable
private fun ImagePanel(bitmap: Bitmap?, busy: Boolean, caption: String? = null) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(240.dp)
                .clip(RoundedCornerShape(12.dp))
                .background(MaterialTheme.colorScheme.surfaceVariant),
            contentAlignment = Alignment.Center
        ) {
            when {
                busy -> CircularProgressIndicator()
                bitmap != null -> Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = caption ?: "Selected ECG printout",
                    contentScale = ContentScale.Fit,
                    modifier = Modifier.fillMaxSize().padding(8.dp)
                )
                else -> Text(
                    "No image selected",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
        caption?.let {
            Text(
                it,
                style = MaterialTheme.typography.labelSmall,
                fontFamily = FontFamily.Monospace,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

/**
 * Injects a known rotation.
 *
 * This is not a viewer control and is not throwaway. It is how ground truth
 * gets into the pipeline on the device: the skew estimator has to recover
 * this angle back out of the pixels, and without a known injected value
 * there is nothing on the phone to score it against — a fixture's true grid
 * pitch is known, but its true skew is only zero.
 *
 * Range is plus-or-minus 10 degrees because that is `COARSE_HALF_RANGE_DEG`
 * in the Python: the coarse sweep spans exactly this, so the slider covers
 * the estimator's stated operating range and stops at its edge rather than
 * wandering past it. 200 steps gives 0.1 degree resolution, which is the
 * residual the reconstruction can absorb — finer would be below what the
 * rest of the pipeline can tell apart.
 */
@Composable
private fun RotationControl(
    angle: Float,
    enabled: Boolean,
    warping: Boolean,
    onChange: (Float) -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                "Injected rotation",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                "%+.2f deg%s".format(angle, if (warping) "  (warping)" else ""),
                style = MaterialTheme.typography.labelMedium,
                fontFamily = FontFamily.Monospace
            )
        }
        Slider(
            value = angle,
            onValueChange = onChange,
            valueRange = -ROTATION_RANGE_DEG..ROTATION_RANGE_DEG,
            steps = ROTATION_STEPS,
            enabled = enabled
        )
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text("-10", style = MaterialTheme.typography.labelSmall)
            TextButton(onClick = { onChange(0f) }, enabled = enabled) { Text("reset") }
            Text("+10", style = MaterialTheme.typography.labelSmall)
        }
    }
}

/** Python's `deskew.COARSE_HALF_RANGE_DEG`. */
private const val ROTATION_RANGE_DEG = 10f

/**
 * Slider detents between the endpoints. 199 interior steps over a 20 degree
 * span puts a detent every 0.1 degrees, which is the angular error the
 * waveform can absorb -- measured, 1 degree of residual rotation takes PCC
 * from 0.981 to 0.734.
 */
private const val ROTATION_STEPS = 199

/**
 * Display copy of a just-loaded bitmap, via OpenCV rather than
 * `Bitmap.createScaledBitmap`, so that the reduction is the same INTER_AREA
 * decimation used everywhere else in the pipeline and one code path is
 * responsible for it.
 */
private fun shrinkForDisplay(bmp: Bitmap): Bitmap {
    if (!OpenCvRuntime.isReady) return bmp
    val mat = ImageWarp.toMat(bmp)
    return try {
        ImageWarp.previewBitmap(mat)
    } finally {
        mat.release()
    }
}

@Composable
private fun ResultsPanel(state: PipelineState) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Text(
                "Results",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold
            )
            HorizontalDivider()
            ResultRow("Status", state.status)
            ResultRow("Source", state.sourceLabel ?: "--")
            ResultRow("Image size", state.imageSize)
            ResultRow("Grid pitch", "${state.gridPitch} px/mm")
            // Shown only for fixtures. For a picked photo there is no known
            // answer, and printing a blank row would imply one exists.
            state.reference?.let { ref ->
                ResultRow("True grid pitch", "%.4f px/mm".format(ref.pxPerMm))
                state.gridError?.let { ResultRow("Error vs true", it) }
                ResultRow("True 5 mm box", "%.3f px".format(ref.pxPerMajorBox))
                ResultRow("PTB-XL record", ref.record)
            }
            ResultRow("Load time", "${state.latencyMs} ms")
            ResultRow("Grid estimate", "${state.gridMs} ms")
            ResultRow("Injected rotation", "%+.2f deg".format(state.rotationDeg))
            ResultRow("Rotate (warp)", "${state.rotateMs} ms")
        }
    }
}

@Composable
private fun ResultRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            value,
            style = MaterialTheme.typography.bodyMedium,
            fontFamily = FontFamily.Monospace
        )
    }
}