package com.maarijalam.eacg

import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.maarijalam.eacg.ui.SecondaryButton
import com.maarijalam.eacg.ui.TopBar

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
            // The in-app camera arrives in UI step 9; until then both open the picker.
            onTakePhoto = pick, onChooseFromGallery = pick,
            onOpenSettings = onOpenSettings, onSeeAll = onSeeAllRecords, onOpenRecord = onOpenRecord,
            modifier = modifier)
        ScanRoute.CORNERS -> CornersScreen(scan, modifier)
        else -> InterimScanPanel(scan, modifier)
    }
}

/**
 * INTERIM (UI steps 4-5): progress and the former result panel, driven by
 * [ScanController], until the Digitising (step 5) and Result (step 6)
 * screens replace it.
 */
@Composable
private fun InterimScanPanel(scan: ScanController, modifier: Modifier) {
    Column(modifier.fillMaxSize()) {
        TopBar(if (scan.busy) "Digitising" else "Result", onBack = { if (scan.busy) scan.cancel() else scan.reset() })
        Column(Modifier.fillMaxWidth().verticalScroll(rememberScrollState()).padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)) {
            if (scan.status.isNotEmpty()) Text(scan.status, style = MaterialTheme.typography.bodySmall)
            scan.saveError?.let { Text("Not saved to Records: $it", style = MaterialTheme.typography.bodySmall) }
            scan.result?.let { r ->
                SheetResultPanel(r, scan.truth12)
                SecondaryButton("Export 12-lead CSV", { scan.export() }, Modifier.fillMaxWidth())
                scan.exported?.let { Text(it, style = MaterialTheme.typography.labelSmall) }
            }
        }
    }
}
