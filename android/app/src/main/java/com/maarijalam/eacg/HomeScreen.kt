package com.maarijalam.eacg

import androidx.compose.foundation.Canvas
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import com.maarijalam.eacg.ui.Disclaimer
import com.maarijalam.eacg.ui.EaCard
import com.maarijalam.eacg.ui.EaIcons
import com.maarijalam.eacg.ui.PrimaryButton
import com.maarijalam.eacg.ui.SecondaryButton
import com.maarijalam.eacg.ui.decorativeEcg
import com.maarijalam.eacg.ui.drawEcgPaper
import com.maarijalam.eacg.ui.drawTrace
import com.maarijalam.eacg.ui.theme.CardBorder
import com.maarijalam.eacg.ui.theme.CardSurface
import com.maarijalam.eacg.ui.theme.EcgTrace
import com.maarijalam.eacg.ui.theme.Muted
import com.maarijalam.eacg.ui.theme.Primary
import com.maarijalam.eacg.ui.theme.RowDivider
import com.maarijalam.eacg.ui.theme.Text2
import androidx.compose.ui.graphics.drawscope.translate

/**
 * Home (docs/ui_plan.md §3.1, mock-up artboard 1): what the app does, the two
 * ways in, three capture tips, the newest records, the disclaimer.
 */
@Composable
fun HomeScreen(
    loading: Boolean,
    status: String,
    savedKey: Any?,
    onTakePhoto: () -> Unit,
    onChooseFromGallery: () -> Unit,
    onOpenSettings: () -> Unit,
    onSeeAll: () -> Unit,
    onOpenRecord: (RecordStore.Meta) -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    var recent by remember { mutableStateOf<List<RecordStore.Meta>?>(null) }
    // Reload when a scan was saved (savedKey changes) and on first show.
    LaunchedEffect(savedKey) { recent = runCatching { RecordStore.list(context).take(3) }.getOrDefault(emptyList()) }

    Column(modifier.fillMaxSize()) {
        Row(Modifier.fillMaxWidth().padding(start = 20.dp, end = 8.dp, top = 12.dp, bottom = 8.dp),
            verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text("EaCG", style = MaterialTheme.typography.headlineSmall,
                    modifier = Modifier.semantics { heading() })
                Text("Paper ECG to signal, on this phone", style = MaterialTheme.typography.bodyMedium, color = Muted)
            }
            IconButton(onClick = onOpenSettings, modifier = Modifier.size(48.dp)) {
                Icon(EaIcons.Settings, contentDescription = "Settings", tint = Text2)
            }
        }
        Column(Modifier.fillMaxWidth().verticalScroll(rememberScrollState())
            .padding(start = 16.dp, end = 16.dp, top = 4.dp, bottom = 16.dp),
            verticalArrangement = Arrangement.spacedBy(20.dp)) {

            EaCard(radius = 20.dp, padding = androidx.compose.foundation.layout.PaddingValues(16.dp)) {
                Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    Illustration()
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text("Scan a paper ECG", style = MaterialTheme.typography.titleLarge.copy(
                            fontSize = MaterialTheme.typography.titleLarge.fontSize * 1.1f))
                        Text("Standard 3×4 printout with a lead II rhythm strip. The 12 leads are " +
                            "recovered on the phone; nothing is uploaded.",
                            style = MaterialTheme.typography.bodyLarge, color = Muted)
                    }
                    if (loading) {
                        Row(Modifier.fillMaxWidth().height(52.dp), verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(12.dp, Alignment.CenterHorizontally)) {
                            CircularProgressIndicator(Modifier.size(24.dp), color = Primary, strokeWidth = 3.dp)
                            Text(status.ifEmpty { "Loading" }, style = MaterialTheme.typography.bodyMedium)
                        }
                    } else {
                        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                            PrimaryButton("Take photo", onTakePhoto, Modifier.fillMaxWidth(), icon = EaIcons.Camera)
                            SecondaryButton("Choose from gallery", onChooseFromGallery, Modifier.fillMaxWidth(),
                                icon = EaIcons.Gallery)
                        }
                        if (status.isNotEmpty()) Text(status, style = MaterialTheme.typography.bodySmall, color = Muted)
                    }
                }
            }

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Tip(EaIcons.Page, "Page flat", "No curl or folds", Modifier.weight(1f))
                Tip(EaIcons.Sun, "Even light", "No hand shadow", Modifier.weight(1f))
                Tip(EaIcons.Frame, "Whole sheet", "All 4 edges in view", Modifier.weight(1f))
            }

            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Recent", style = MaterialTheme.typography.titleMedium,
                        modifier = Modifier.weight(1f).semantics { heading() })
                    TextButton(onClick = onSeeAll) {
                        Text("See all", style = MaterialTheme.typography.titleSmall, color = Primary)
                    }
                }
                val list = recent
                EaCard {
                    when {
                        list == null -> Box(Modifier.fillMaxWidth().padding(16.dp)) {
                            CircularProgressIndicator(Modifier.size(20.dp), color = Primary, strokeWidth = 2.dp)
                        }
                        list.isEmpty() -> Text("No records yet. Every sheet you digitise is kept here.",
                            Modifier.padding(14.dp), style = MaterialTheme.typography.bodyMedium, color = Muted)
                        else -> list.forEachIndexed { i, m ->
                            if (i > 0) HorizontalDivider(color = RowDivider)
                            RecentRow(m) { onOpenRecord(m) }
                        }
                    }
                }
            }
            Disclaimer()
        }
    }
}

/** The card's decorative pink-paper ECG (an illustration, not a recording). */
@Composable
private fun Illustration() {
    Canvas(Modifier.fillMaxWidth().height(132.dp).clip(RoundedCornerShape(12.dp))) {
        drawEcgPaper(minorPx = 4.dp.toPx())
        val rowH = size.height / 3
        val w = size.width
        val stroke = 1.4.dp.toPx()
        val amps = floatArrayOf(18f, 12f, 20f)
        for (r in 0 until 3) {
            translate(top = r * rowH) {
                drawTrace(decorativeEcg(w, rowH, 6, amps[r] * density, inverted = r == 1, offset = r * 10f * density),
                    EcgTrace, stroke)
            }
        }
    }
}

@Composable
private fun Tip(icon: ImageVector, title: String, body: String, modifier: Modifier = Modifier) {
    val shape = RoundedCornerShape(14.dp)
    Column(modifier.background(CardSurface, shape).border(1.dp, CardBorder, shape)
        .padding(horizontal = 10.dp, vertical = 12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Icon(icon, contentDescription = null, tint = Primary, modifier = Modifier.size(22.dp))
        Text(title, style = MaterialTheme.typography.labelMedium)
        Text(body, style = MaterialTheme.typography.labelSmall, color = Muted)
    }
}

@Composable
private fun RecentRow(m: RecordStore.Meta, onClick: () -> Unit) {
    Row(Modifier.fillMaxWidth().heightIn(min = 64.dp).clickable(role = Role.Button, onClick = onClick)
        .padding(horizontal = 14.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        RecordThumb(m.id)
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text(recordTitle(m.target), style = MaterialTheme.typography.titleSmall)
            Text("${recordWhen(m.createdMs)} · 12 leads", style = MaterialTheme.typography.bodySmall, color = Muted)
        }
        QualityStatus(m.quality)
    }
}
