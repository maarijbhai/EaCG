package com.maarijalam.eacg

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.dp
import com.maarijalam.eacg.ui.Disclaimer
import com.maarijalam.eacg.ui.EaCard
import com.maarijalam.eacg.ui.EaIcons
import com.maarijalam.eacg.ui.SectionLabel
import com.maarijalam.eacg.ui.TopBar
import com.maarijalam.eacg.ui.theme.DataText
import com.maarijalam.eacg.ui.theme.Muted
import com.maarijalam.eacg.ui.theme.Primary
import com.maarijalam.eacg.ui.theme.RowDivider
import com.maarijalam.eacg.ui.theme.Text2

/**
 * Settings (docs/ui_plan.md §5.3): the prototype-mode switch, the way into
 * the Lab (prototype mode only), and About.
 */
@Composable
fun SettingsScreen(onBack: () -> Unit, onOpenLab: () -> Unit, modifier: Modifier = Modifier) {
    val settings = LocalAppSettings.current
    val context = LocalContext.current
    val version = remember {
        runCatching { context.packageManager.getPackageInfo(context.packageName, 0).versionName }
            .getOrNull() ?: "?"
    }
    Column(modifier.fillMaxSize()) {
        TopBar("Settings", onBack = onBack)
        Column(Modifier.fillMaxWidth().verticalScroll(rememberScrollState()).padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(20.dp)) {

            EaCard {
                Row(Modifier.fillMaxWidth()
                    .clickable(role = Role.Switch) { settings.updatePrototypeMode(!settings.prototypeMode) }
                    .padding(horizontal = 14.dp, vertical = 12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                        Text("Prototype mode", style = MaterialTheme.typography.titleSmall)
                        Text("Shows the research controls: tracer and grid-fit choice, the PTB-XL " +
                            "benchmark, timings and memory, and the Lab. Turning it off only hides " +
                            "them; it changes no result.",
                            style = MaterialTheme.typography.bodySmall, color = Muted)
                    }
                    Switch(checked = settings.prototypeMode, onCheckedChange = null,
                        colors = SwitchDefaults.colors(checkedTrackColor = Primary,
                            checkedThumbColor = Color.White))
                }
                if (settings.prototypeMode) {
                    HorizontalDivider(color = RowDivider)
                    Row(Modifier.fillMaxWidth().heightIn(min = 56.dp)
                        .clickable(role = Role.Button, onClick = onOpenLab)
                        .padding(horizontal = 14.dp, vertical = 12.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        Icon(EaIcons.Flask, null, tint = Primary, modifier = Modifier.size(22.dp))
                        Column(Modifier.weight(1f)) {
                            Text("Lab", style = MaterialTheme.typography.titleSmall)
                            Text("Fixtures, deskew check, rhythm-strip path, tracer benchmark",
                                style = MaterialTheme.typography.bodySmall, color = Muted)
                        }
                        Icon(EaIcons.ChevronRight, contentDescription = "Open the Lab", tint = Text2,
                            modifier = Modifier.size(20.dp))
                    }
                }
            }

            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                SectionLabel("About")
                EaCard(padding = PaddingValues(horizontal = 14.dp, vertical = 4.dp)) {
                    AboutRow("App version", version)
                    HorizontalDivider(color = RowDivider)
                    AboutRow("Tracer model", SheetPipeline.MODEL_ASSET.substringAfterLast('/'))
                    HorizontalDivider(color = RowDivider)
                    AboutRow("Fonts", "IBM Plex (SIL OFL 1.1)", mono = false)
                }
            }
            Disclaimer(text = "Not a medical device. Research prototype; not for diagnosis.")
        }
    }
}

@Composable
private fun AboutRow(label: String, value: String, mono: Boolean = true) {
    Row(Modifier.fillMaxWidth().heightIn(min = 40.dp), verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Text(label, Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium, color = Muted)
        Text(value, style = if (mono) DataText else MaterialTheme.typography.bodyMedium)
    }
}

/**
 * The Lab (docs/ui_plan.md §5.4): the original development screen, unchanged
 * in function -- fixtures, the deskew check and its benchmark, the
 * rhythm-strip photo path and the tracer benchmark. They produce
 * results/device/ inputs. The whole-sheet section is not here: it is the
 * Scan flow.
 */
@Composable
fun LabScreen(onBack: () -> Unit, modifier: Modifier = Modifier) {
    Column(modifier.fillMaxSize()) {
        TopBar("Lab", onBack = onBack, subtitle = "Development and benchmark tools")
        DigitiseScreen(modifier = Modifier.weight(1f), includeWholeSheet = false)
    }
}
