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
        if (ok && u != null) scan.load(Uri.parse(u), "camera")
    }
    val take = {
        val dir = File(context.cacheDir, "camera").apply { mkdirs() }
        // One shot at a time: the record keeps its own copy of the photo.
        dir.listFiles()?.forEach { it.delete() }
        val f = File(dir, "scan_${Exporter.stamp()}.jpg")
        val uri = FileProvider.getUriForFile(context, "${context.packageName}.files", f)
        pendingShot = uri.toString()
        try {
            camera.launch(uri)
        } catch (e: ActivityNotFoundException) {
            pendingShot = null
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
        ScanRoute.RESULT -> LiveResult(scan, modifier)
    }
}

/** The Result screen for the scan just made (saved to Records already, unless saving failed). */
@Composable
private fun LiveResult(scan: ScanController, modifier: Modifier) {
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
        pageContent = {
            LaunchedEffect(r) { if (page == null) page = decodeForView(r.pageJpeg) }
            PageTab(page, data, r.rowTraces, r.t0X, r.rowY, r.tickPositions, r.ticksRejected, r.cells,
                r.rectW, r.rectH, scan.truth12)
        },
        photoContent = {
            scan.bitmap?.let { ZoomableImage(it) } ?: LoadingBox("No photograph")
        },
        modifier = modifier)
}

/** The Page overlay tab's frame: the page, or a loading box until it is decoded. */
@Composable
fun PageTab(page: Bitmap?, data: ResultData, traces: List<SheetPipeline.RowTrace>, t0X: Double, rowY: List<Int>,
            ticks: List<Triple<Int, Int, Double>>, replaced: List<Triple<Int, Int, Double>>,
            cells: List<SheetPipeline.CellBox>, pageW: Int, pageH: Int, truth12: Map<String, DoubleArray>?) {
    if (page == null) { LoadingBox("Loading the page"); return }
    PageOverlay(page, traces, t0X, rowY, ticks, replaced, cells, pageW, pageH, data.leads, data.pxPerMm, truth12)
}
