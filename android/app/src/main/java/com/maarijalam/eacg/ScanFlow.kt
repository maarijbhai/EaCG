package com.maarijalam.eacg

import android.content.ActivityNotFoundException
import android.graphics.Bitmap
import android.net.Uri
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.FileProvider
import java.io.File
import androidx.compose.ui.Modifier
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.ui.unit.dp
import com.maarijalam.eacg.ui.theme.Muted
import com.maarijalam.eacg.ui.theme.Primary

/**
 * The Scan tab: Home → Corners → Processing → Result, driven by one
 * [ScanController] (docs/ui_plan.md §3).
 */
@Composable
fun ScanFlow(
    scan: ScanController,
    onOpenSettings: () -> Unit,
    onSeeAllRecords: () -> Unit,
    onOpenRecord: (RecordStore.Meta) -> Unit,
    modifier: Modifier = Modifier,
) {
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) scan.load(uri, "picked")
    }
    val pick = { picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }

    // In-app camera (plan §3.1): the system camera app writes to a cache file
    // shared through the FileProvider (no camera permission needed); the
    // photo then takes exactly the gallery's path, UriSource at 3,200 px.
    val context = LocalContext.current
    var pendingShot by rememberSaveable { mutableStateOf<String?>(null) }
    val camera = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        val u = pendingShot
        pendingShot = null
        if (u == null) return@rememberLauncherForActivityResult
        if (ok) scan.load(Uri.parse(u), "camera")
        // A cancelled shot leaves no empty gallery entry behind.
        else if (u.startsWith("content://media")) runCatching { context.contentResolver.delete(Uri.parse(u), null, null) }
    }
    val take = {
        // The photo goes to the gallery (Pictures/EaCG, original resolution) so
        // it can be copied off the phone like any other; the cache only below
        // Android 10.
        val name = "eacg_${Exporter.stamp()}.jpg"
        val uri = Exporter.newCameraPhotoUri(context, name) ?: run {
            val dir = File(context.cacheDir, "camera").apply { mkdirs() }
            dir.listFiles()?.forEach { it.delete() }
            FileProvider.getUriForFile(context, "${context.packageName}.files", File(dir, name))
        }
        pendingShot = uri.toString()
        try {
            camera.launch(uri)
        } catch (e: ActivityNotFoundException) {
            pendingShot = null
            if (uri.toString().startsWith("content://media")) runCatching { context.contentResolver.delete(uri, null, null) }
            pick()     // no camera app: fall back to the gallery
        }
    }

    BackHandler(enabled = scan.route != ScanRoute.HOME) {
        when (scan.route) {
            ScanRoute.PROCESSING -> scan.cancel()
            ScanRoute.RESULT, ScanRoute.CORNERS -> scan.reset()
            ScanRoute.HOME -> Unit
        }
    }

    when (scan.route) {
        ScanRoute.HOME -> HomeScreen(
            loading = scan.loading, status = scan.status, savedKey = scan.saved?.id,
            onTakePhoto = take, onChooseFromGallery = pick,
            onOpenSettings = onOpenSettings, onSeeAll = onSeeAllRecords, onOpenRecord = onOpenRecord,
            modifier = modifier)
        ScanRoute.CORNERS -> CornersScreen(scan, modifier)
        ScanRoute.PROCESSING -> ProcessingScreen(scan, modifier)
        ScanRoute.RESULT -> LiveResult(scan, onOpenRecord, modifier)
    }
}

/** The Result screen for the scan just made (saved to Records already, unless saving failed). */
@Composable
private fun LiveResult(scan: ScanController, onOpenRecord: (RecordStore.Meta) -> Unit, modifier: Modifier) {
    val r = scan.result ?: return
    val data = remember(r, scan.saved) { ResultData.fromLive(r, scan.saved, scan.resultTarget?.name) }
    var page by remember(r) { mutableStateOf<Bitmap?>(null) }
    ResultScreen(
        data = data, truth12 = scan.truth12,
        onBack = { scan.reset() },
        onExport = { scan.export() }, exportMessage = scan.exported,
        savedNote = if (scan.saved != null) "Saved to Records." else scan.saveError?.let { "Not saved to Records: $it." },
        csvForShare = { scan.csvForShare() },
        onDelete = if (scan.saved != null) ({ scan.deleteSaved() }) else null,
        onRename = if (scan.saved != null) ({ n: String? -> scan.renameSaved(n) }) else null,
        pageContent = {
            LaunchedEffect(r) { if (page == null) page = decodeForView(r.pageJpeg) }
            PageTab(page, data, r.rowTraces, r.t0X, r.rowY, r.tickPositions, r.ticksRejected, r.cells,
                r.rectW, r.rectH, scan.truth12)
        },
        photoContent = {
            scan.bitmap?.let { ZoomableImage(it) } ?: LoadingBox("No photograph")
        },
        modifier = modifier,
        extraNote = scan.pairNote?.let { note -> {
            Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(note, style = MaterialTheme.typography.bodySmall, color = Muted)
                scan.pairedRecord?.let { m ->
                    TextButton(onClick = { onOpenRecord(m) }) { Text("Open the classical record", color = Primary) }
                }
            }
        } })
}

/** The Page overlay tab's frame: the page, or a loading box until it is decoded. */
@Composable
fun PageTab(page: Bitmap?, data: ResultData, traces: List<SheetPipeline.RowTrace>, t0X: Double, rowY: List<Int>,
            ticks: List<Triple<Int, Int, Double>>, replaced: List<Triple<Int, Int, Double>>,
            cells: List<SheetPipeline.CellBox>, pageW: Int, pageH: Int, truth12: Map<String, DoubleArray>?) {
    if (page == null) { LoadingBox("Loading the page"); return }
    PageOverlay(page, traces, t0X, rowY, ticks, replaced, cells, pageW, pageH, data.leads, data.pxPerMm, truth12)
}
