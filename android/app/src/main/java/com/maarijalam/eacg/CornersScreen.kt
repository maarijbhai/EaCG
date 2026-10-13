package com.maarijalam.eacg

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.maarijalam.eacg.ui.EaIcons
import com.maarijalam.eacg.ui.PrimaryButton
import com.maarijalam.eacg.ui.SecondaryButton
import com.maarijalam.eacg.ui.SectionLabel
import com.maarijalam.eacg.ui.TopBar
import com.maarijalam.eacg.ui.theme.BadText
import com.maarijalam.eacg.ui.theme.ButtonOutline
import com.maarijalam.eacg.ui.theme.CardBorder
import com.maarijalam.eacg.ui.theme.CardSurface
import com.maarijalam.eacg.ui.theme.DarkScreen
import com.maarijalam.eacg.ui.theme.Ink
import com.maarijalam.eacg.ui.theme.Muted
import com.maarijalam.eacg.ui.theme.OnDarkScreen
import com.maarijalam.eacg.ui.theme.Primary
import com.maarijalam.eacg.ui.theme.PrimaryDark
import com.maarijalam.eacg.ui.theme.Text2
import com.maarijalam.eacg.ui.theme.Tint
import com.maarijalam.eacg.ui.theme.WarnBg
import com.maarijalam.eacg.ui.theme.WarnIcon

/**
 * Check corners (docs/ui_plan.md §3.2, mock-up artboard 2): the photograph
 * with the detected quad on a dark ground, then a white sheet with the
 * status, the four close-ups, the research options and the benchmark
 * selector (prototype mode only), and Retake / Digitise.
 */
@Composable
fun CornersScreen(scan: ScanController, modifier: Modifier = Modifier) {
    val proto = LocalAppSettings.current.prototypeMode
    var help by remember { mutableStateOf(false) }
    if (help) HelpDialog { help = false }

    Column(modifier.fillMaxSize().background(DarkScreen)) {
        TopBar("Check corners", onBack = { scan.reset() }, contentColor = OnDarkScreen,
            titleColor = OnDarkScreen) {
            IconButton(onClick = { help = true }, modifier = Modifier.size(48.dp)) {
                Icon(EaIcons.Help, contentDescription = "How to place the corners", tint = OnDarkScreen)
            }
        }
        val bmp = scan.bitmap
        val preview = scan.preview
        val corners = scan.corners
        Box(Modifier.fillMaxWidth().heightIn(max = 340.dp).padding(horizontal = 16.dp, vertical = 8.dp),
            contentAlignment = Alignment.Center) {
            if (bmp != null && preview != null && corners != null) {
                // Fit the photo's aspect inside the box (landscape sheets: full width).
                Box(Modifier.widthIn(max = (320f * bmp.width / bmp.height).dp)) {
                    CornerEditor(photo = bmp, preview = preview, corners = corners,
                        onCornersChange = { scan.moveCorners(it) }, enabled = !scan.busy, refined = null)
                }
            }
        }

        Column(Modifier.weight(1f).fillMaxWidth()
            .background(CardSurface, RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp))) {
            Column(Modifier.weight(1f).verticalScroll(rememberScrollState())
                .padding(start = 16.dp, end = 16.dp, top = 20.dp, bottom = 8.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)) {
                CornerStatus(scan)
                if (bmp != null && corners != null) CornerCloseUps(bmp, corners)
                if (proto) {
                    ResearchOptions(scan)
                    BenchmarkSelector(scan)
                }
                if (scan.status.startsWith("Failed")) {
                    Text(scan.status, style = MaterialTheme.typography.bodySmall, color = BadText)
                }
            }
            Row(Modifier.fillMaxWidth().padding(16.dp), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                SecondaryButton("Retake", { scan.reset() }, Modifier.weight(1f), enabled = !scan.busy)
                PrimaryButton("Digitise", { scan.digitise() }, Modifier.weight(1f),
                    enabled = !scan.busy && corners != null)
            }
        }
    }
}

/** "Corners found automatically" (refined / coarse) or "Could not find the sheet". */
@Composable
private fun CornerStatus(scan: ScanController) {
    val found = scan.cornersFound
    Row(horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.Top) {
        Box(Modifier.size(36.dp).background(if (found) Tint else WarnBg, RoundedCornerShape(18.dp)),
            contentAlignment = Alignment.Center) {
            Icon(if (found) EaIcons.Check else EaIcons.Warning, contentDescription = null,
                tint = if (found) Primary else WarnIcon, modifier = Modifier.size(20.dp))
        }
        Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text(if (found) "Corners found automatically" else "Could not find the sheet — drag the four corners",
                style = MaterialTheme.typography.titleMedium)
            val detail = when {
                !found -> "Place each handle on the outer tip of the grid." +
                    (scan.cornerFailure?.let { " ($it)" } ?: "")
                scan.cornersRefined == true -> "Refined to the grid edge. Drag a handle if it is not on the " +
                    "outer tip of the grid."
                else -> "Coarse only. Drag a handle if it is not on the outer tip of the grid; the grid fit " +
                    "corrects small errors."
            }
            Text(detail, style = MaterialTheme.typography.bodyMedium, color = Muted)
        }
    }
}

/** Tracer (Learned / Classical) and Grid fit, exactly SheetPipeline.Tracer / Geometry. */
@Composable
private fun ResearchOptions(scan: ScanController) {
    val shape = RoundedCornerShape(16.dp)
    Column(Modifier.fillMaxWidth().border(1.dp, CardBorder, shape).padding(horizontal = 14.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)) {
        SectionLabel("Research options")
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Text("Tracer", style = MaterialTheme.typography.bodyLarge, modifier = Modifier.weight(1f))
            Segmented(
                options = listOf("Learned" to SheetPipeline.Tracer.MODEL, "Classical" to SheetPipeline.Tracer.CLASSICAL),
                selected = scan.tracer, enabled = !scan.busy && !scan.bothTracers, onSelect = { scan.tracer = it })
        }
        // Testing sessions: learned then classical on the same photo and
        // corners, both saved and exported (ScanController.digitise).
        Row(Modifier.fillMaxWidth().heightIn(min = 48.dp)
            .clickable(role = Role.Switch, enabled = !scan.busy) { scan.chooseBothTracers(!scan.bothTracers) },
            verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("Run both tracers (testing)", style = MaterialTheme.typography.bodyLarge)
                if (scan.bothTracers) Text("Learned, then classical; both saved and exported",
                    style = MaterialTheme.typography.bodySmall, color = Muted)
            }
            Switch(checked = scan.bothTracers, onCheckedChange = null, enabled = !scan.busy,
                colors = SwitchDefaults.colors(checkedTrackColor = Primary, checkedThumbColor = Color.White))
        }
        Row(Modifier.fillMaxWidth().heightIn(min = 48.dp)
            .clickable(role = Role.Switch, enabled = !scan.busy) { scan.gridFit = !scan.gridFit },
            verticalAlignment = Alignment.CenterVertically) {
            Text("Grid fit", style = MaterialTheme.typography.bodyLarge, modifier = Modifier.weight(1f))
            Switch(checked = scan.gridFit, onCheckedChange = null, enabled = !scan.busy,
                colors = SwitchDefaults.colors(checkedTrackColor = Primary, checkedThumbColor = Color.White))
        }
    }
}

/** The mock-up's two-segment control: tint + maroon for the selected segment. */
@Composable
private fun <T> Segmented(options: List<Pair<String, T>>, selected: T, enabled: Boolean, onSelect: (T) -> Unit) {
    val shape = RoundedCornerShape(20.dp)
    Row(Modifier.border(BorderStroke(1.dp, ButtonOutline), shape).selectableGroup()) {
        options.forEachIndexed { i, (label, value) ->
            val on = value == selected
            val segShape = when (i) {
                0 -> RoundedCornerShape(topStart = 20.dp, bottomStart = 20.dp)
                options.lastIndex -> RoundedCornerShape(topEnd = 20.dp, bottomEnd = 20.dp)
                else -> RoundedCornerShape(0.dp)
            }
            Box(Modifier.height(40.dp).background(if (on) Tint else CardSurface, segShape)
                .selectable(selected = on, enabled = enabled, role = Role.RadioButton) { onSelect(value) }
                .padding(horizontal = 14.dp), contentAlignment = Alignment.Center) {
                Text(label, style = MaterialTheme.typography.bodyMedium.copy(
                    fontWeight = if (on) FontWeight.SemiBold else FontWeight.Normal),
                    color = if (on) PrimaryDark else Text2)
            }
        }
    }
}

/**
 * "Benchmark against": None or one of the printed PTB-XL sheets in
 * assets/kit/targets.txt (KitTargets.load; not a hard-coded list). The
 * choice is remembered; its 12-lead truth is passed to SheetPipeline.run,
 * and its name goes into the record and the CSV file name, as before.
 */
@Composable
private fun BenchmarkSelector(scan: ScanController) {
    var open by remember { mutableStateOf(false) }
    val shape = RoundedCornerShape(16.dp)
    Column(Modifier.fillMaxWidth().border(1.dp, CardBorder, shape).padding(horizontal = 14.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)) {
        SectionLabel("Benchmark against")
        Box {
            Row(Modifier.fillMaxWidth().heightIn(min = 48.dp)
                .border(1.dp, ButtonOutline, RoundedCornerShape(12.dp))
                .clickable(role = Role.DropdownList, enabled = !scan.busy) { open = true }
                .padding(horizontal = 14.dp), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(scan.target?.label ?: "None", style = MaterialTheme.typography.bodyLarge, color = Ink)
                    Text(if (scan.target != null) "Each lead is scored against its PTB-XL signal"
                        else "A photograph of an unknown sheet", style = MaterialTheme.typography.labelSmall,
                        color = Muted)
                }
                Icon(EaIcons.ChevronRight, contentDescription = "Choose", tint = Text2,
                    modifier = Modifier.size(20.dp))
            }
            DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
                DropdownMenuItem(text = { Text("None") }, onClick = { scan.chooseTarget(null); open = false })
                scan.targets.forEach { t ->
                    DropdownMenuItem(text = { Text(t.label) }, onClick = { scan.chooseTarget(t); open = false })
                }
            }
        }
    }
}

@Composable
private fun HelpDialog(onDismiss: () -> Unit) {
    AlertDialog(onDismissRequest = onDismiss,
        confirmButton = { TextButton(onClick = onDismiss) { Text("OK", color = Primary) } },
        title = { Text("Placing the corners") },
        text = {
            Text("Each handle belongs on the very tip of the printed grid -- where the pink grid " +
                "meets the white paper -- not inside the sheet. The close-ups show each corner " +
                "magnified; drag a handle on the photo and a magnifier follows your finger. Small " +
                "errors are corrected by the grid fit.", style = MaterialTheme.typography.bodyMedium)
        })
}
