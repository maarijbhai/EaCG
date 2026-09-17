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
    val reference: ReferenceImage? = null
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
                val grid = withContext(Dispatchers.Default) {
                    GridEstimator.estimate(bmp)
                }
                bmp to grid
            }
                .onSuccess { (bmp, grid) ->
                    val ms = (System.nanoTime() - started) / 1_000_000
                    state = state.copy(
                        bitmap = bmp,
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

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Header()

        OpenCvBanner()

        ImagePanel(bitmap = state.bitmap, busy = state.busy)

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
private fun ImagePanel(bitmap: Bitmap?, busy: Boolean) {
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
                contentDescription = "Selected ECG printout",
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