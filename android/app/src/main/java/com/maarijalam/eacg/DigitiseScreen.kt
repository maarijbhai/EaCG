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
import kotlinx.coroutines.launch

/** Everything the screen knows. One object, so adding a field is one edit. */
data class PipelineState(
    val bitmap: Bitmap? = null,
    val status: String = "Ready",
    val busy: Boolean = false,
    val sourceLabel: String? = null,
    val gridPitch: String = "--",
    val latencyMs: String = "--",
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
            runCatching { source.acquire() }
                .onSuccess { bmp ->
                    val ms = (System.nanoTime() - started) / 1_000_000
                    state = state.copy(
                        bitmap = bmp,
                        busy = false,
                        status = "Image loaded",
                        sourceLabel = label,
                        imageSize = "${bmp.width} x ${bmp.height}",
                        latencyMs = "$ms"
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
                ResultRow("True 5 mm box", "%.3f px".format(ref.pxPerMajorBox))
                ResultRow("PTB-XL record", ref.record)
            }
            ResultRow("Load time", "${state.latencyMs} ms")
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