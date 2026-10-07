package com.maarijalam.eacg

import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.maarijalam.eacg.ui.PrimaryButton
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
        else -> InterimScanPanel(scan, modifier)
    }
}

/**
 * INTERIM (UI steps 3-5): the former SheetSection's controls and result,
 * driven by [ScanController], until the Corners (step 4), Digitising (5) and
 * Result (6) screens replace it.
 */
@Composable
private fun InterimScanPanel(scan: ScanController, modifier: Modifier) {
    val proto = LocalAppSettings.current.prototypeMode
    Column(modifier.fillMaxSize()) {
        TopBar("Check corners", onBack = { scan.reset() })
        Column(Modifier.fillMaxWidth().verticalScroll(rememberScrollState()).padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)) {
            val bmp = scan.bitmap
            val preview = scan.preview
            val corners = scan.corners
            if (bmp != null && preview != null && corners != null) {
                CornerEditor(photo = bmp, preview = preview, corners = corners,
                    onCornersChange = { scan.moveCorners(it) }, enabled = !scan.busy, refined = null)
                CornerCheckRow(bmp, corners)
            }
            if (proto) {
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    scan.targets.forEach { t ->
                        FilterChip(selected = scan.target?.name == t.name, enabled = !scan.busy,
                            onClick = { scan.chooseTarget(if (scan.target?.name == t.name) null else t) },
                            label = { Text(t.label, style = MaterialTheme.typography.labelSmall) })
                    }
                }
                FilterChip(selected = scan.gridFit, enabled = !scan.busy, onClick = { scan.gridFit = !scan.gridFit },
                    label = { Text("Grid fit", style = MaterialTheme.typography.labelSmall) })
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                PrimaryButton("Digitise", { scan.tracer = SheetPipeline.Tracer.MODEL; scan.digitise() },
                    Modifier.weight(1f), enabled = !scan.busy)
                if (proto) SecondaryButton("Classical", { scan.tracer = SheetPipeline.Tracer.CLASSICAL; scan.digitise() },
                    Modifier.weight(1f), enabled = !scan.busy)
            }
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
