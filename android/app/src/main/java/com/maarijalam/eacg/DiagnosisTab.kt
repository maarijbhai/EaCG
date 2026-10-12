package com.maarijalam.eacg

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.maarijalam.eacg.ui.Disclaimer
import com.maarijalam.eacg.ui.EaCard
import com.maarijalam.eacg.ui.EaIcons
import com.maarijalam.eacg.ui.SectionLabel
import com.maarijalam.eacg.ui.theme.BadBg
import com.maarijalam.eacg.ui.theme.BadBorder
import com.maarijalam.eacg.ui.theme.BadText
import com.maarijalam.eacg.ui.theme.DataText
import com.maarijalam.eacg.ui.theme.Ink
import com.maarijalam.eacg.ui.theme.Muted
import com.maarijalam.eacg.ui.theme.Primary
import com.maarijalam.eacg.ui.theme.ProgressTrack
import com.maarijalam.eacg.ui.theme.RowDivider
import com.maarijalam.eacg.ui.theme.Text2
import com.maarijalam.eacg.ui.theme.WarnBg
import com.maarijalam.eacg.ui.theme.WarnBorder
import com.maarijalam.eacg.ui.theme.WarnIcon
import com.maarijalam.eacg.ui.theme.WarnText
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.Locale

/**
 * The Diagnosis tab of the Result screen: what the research classifier
 * ([Classifier]) says about the digitised leads -- one probability bar per
 * PTB-XL superclass, the threshold marked on it, "flagged" at or above it.
 *
 * Shows the output stored with the scan (the pipeline's "classify" stage).
 * A record saved before the classifier existed has none, so it is computed
 * here from the record's leads and said to be so. Never hidden behind the
 * quality verdict, but an unreliable digitisation is stated first: the
 * classifier reads whatever the digitiser produced.
 */
@Composable
fun DiagnosisTab(data: ResultData, proto: Boolean, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    var dx by remember(data.recordId, data.diagnosis) { mutableStateOf(data.diagnosis) }
    var computedNow by remember(data.recordId) { mutableStateOf(false) }
    var error by remember(data.recordId) { mutableStateOf<String?>(null) }
    var meta by remember { mutableStateOf<Classifier.Meta?>(null) }
    var kit by remember { mutableStateOf<Map<Int, Pair<String, String>>>(emptyMap()) }

    LaunchedEffect(data.leads, data.diagnosis) {
        meta = runCatching { withContext(Dispatchers.IO) { Classifier.loadMeta(context) } }.getOrNull()
        kit = runCatching { withContext(Dispatchers.IO) { Classifier.kitLabels(context) } }.getOrDefault(emptyMap())
        val leads = data.leads
        if (dx == null && leads != null && leads.isNotEmpty()) {
            runCatching { withContext(Dispatchers.Default) { Classifier(context).use { it.diagnose(leads) } } }
                .onSuccess { dx = it; computedNow = true }
                .onFailure { error = "The classifier could not run: ${it.message}" }
        }
    }

    Column(modifier.verticalScroll(rememberScrollState())
        .padding(start = 16.dp, end = 16.dp, top = 12.dp, bottom = 16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)) {

        if (data.quality == "unreliable") Notice(bad = true,
            "The digitisation was marked unreliable. These findings are computed from it and inherit its errors.")
        else if (data.quality == "check") Notice(bad = false,
            "The digitisation has warnings (see Leads). Read these findings with them in mind.")

        val d = dx
        when {
            error != null -> Text(error!!, style = MaterialTheme.typography.bodyMedium, color = BadText)
            d == null -> LoadingBox(if (data.leads == null) "Loading the leads…" else "Classifying…")
            else -> {
                Summary(d)
                EaCard(padding = PaddingValues(horizontal = 14.dp, vertical = 6.dp)) {
                    d.classes.forEachIndexed { k, c ->
                        if (k > 0) HorizontalDivider(color = RowDivider)
                        ClassRow(c, d.probs[k], d.thresholds[k], d.flagged(k))
                    }
                }
                if (computedNow) Text("Computed now from the saved leads: this record was made before the " +
                    "classifier was added.", style = MaterialTheme.typography.bodySmall, color = Muted)
            }
        }

        // Prototype mode, a printed test sheet: the record's own PTB-XL label, for checking.
        val id = data.target?.let { Regex("ptbxl(\\d+)_").find(it)?.groupValues?.get(1)?.toIntOrNull() }
        if (proto && id != null) kit[id]?.let { (sup, codes) ->
            EaCard(padding = PaddingValues(14.dp)) {
                SectionLabel("Test sheet label (PTB-XL $id)")
                Text(sup.split(',').joinToString(", ") { "${Diagnosis.NAMES[it] ?: it} ($it)" },
                    style = MaterialTheme.typography.bodyLarge, color = Ink, modifier = Modifier.padding(top = 6.dp))
                Text("Statements: $codes", style = MaterialTheme.typography.bodySmall, color = Muted)
            }
        }

        EaCard(padding = PaddingValues(14.dp)) {
            SectionLabel("About this classifier")
            Text("A 1-D ResNet trained on PTB-XL (folds 1–8) to the five diagnostic superclasses, from only " +
                "what a 3×4 printout shows: each lead's 2.5 s and the 10 s lead II rhythm strip. A class is " +
                "flagged at or above its threshold, chosen on held-out validation records. Several classes can " +
                "be flagged at once.", style = MaterialTheme.typography.bodyMedium, color = Text2,
                modifier = Modifier.padding(top = 6.dp))
            meta?.let { m ->
                Text(String.format(Locale.US, "On PTB-XL test records (true signals): macro AUROC %.3f. " +
                    "It was trained on true signals, not digitised ones.", m.testMacroAuroc),
                    style = MaterialTheme.typography.bodySmall, color = Muted, modifier = Modifier.padding(top = 6.dp))
                if (proto) Text(m.label, style = DataText, color = Muted, modifier = Modifier.padding(top = 4.dp))
            }
        }
        Disclaimer(text = "Research use only. Not a diagnosis and not a medical device; " +
            "a clinician must read the ECG itself.")
    }
}

@Composable
private fun Notice(bad: Boolean, text: String) {
    val shape = RoundedCornerShape(16.dp)
    Row(Modifier.fillMaxWidth().background(if (bad) BadBg else WarnBg, shape)
        .border(1.dp, if (bad) BadBorder else WarnBorder, shape).padding(horizontal = 14.dp, vertical = 12.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Icon(EaIcons.Warning, contentDescription = null, tint = if (bad) BadText else WarnIcon,
            modifier = Modifier.size(22.dp))
        Text(text, style = MaterialTheme.typography.bodyMedium, color = if (bad) BadText else WarnText)
    }
}

/** One line on top: what is flagged, in words. */
@Composable
private fun Summary(d: Diagnosis) {
    val abnormal = d.abnormal
    val (title, sub) = when {
        abnormal.isNotEmpty() -> "Flagged: " + abnormal.joinToString(", ") { Diagnosis.NAMES[it] ?: it } to
            (if ("NORM" in d.flags) "Normal is also above its threshold: the classifier is unsure." else null)
        "NORM" in d.flags -> "No abnormality flagged" to "Only “Normal ECG” is above its threshold."
        else -> "Nothing above threshold" to "No class reached its threshold; read the bars below."
    }
    EaCard(padding = PaddingValues(14.dp)) {
        SectionLabel("Classifier findings")
        Text(title, style = MaterialTheme.typography.titleMedium, color = if (abnormal.isNotEmpty()) Primary else Ink,
            modifier = Modifier.padding(top = 6.dp))
        sub?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = Muted, modifier = Modifier.padding(top = 2.dp)) }
    }
}

/** A class: name, probability, a bar with its threshold marked. */
@Composable
private fun ClassRow(code: String, p: Double, threshold: Double, flagged: Boolean) {
    val pct = String.format(Locale.US, "%.0f%%", 100 * p)
    Column(Modifier.fillMaxWidth().padding(vertical = 10.dp)
        .semantics(mergeDescendants = true) {
            contentDescription = "${Diagnosis.NAMES[code] ?: code}: $pct" + if (flagged) ", flagged" else ""
        }, verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text(Diagnosis.NAMES[code] ?: code, style = MaterialTheme.typography.bodyLarge.copy(
                    fontWeight = if (flagged) FontWeight.SemiBold else FontWeight.Normal), color = Ink)
                Text(code + String.format(Locale.US, " · threshold %.0f%%", 100 * threshold),
                    style = MaterialTheme.typography.bodySmall, color = Muted)
            }
            if (flagged) Text("Flagged", style = MaterialTheme.typography.labelMedium, color = Primary,
                modifier = Modifier.border(1.dp, Primary, RoundedCornerShape(10.dp))
                    .padding(horizontal = 8.dp, vertical = 2.dp))
            Text(pct, style = DataText, color = if (flagged) Primary else Text2,
                modifier = Modifier.padding(start = 10.dp).width(44.dp))
        }
        BoxWithConstraints(Modifier.fillMaxWidth().height(10.dp)) {
            val w = maxWidth
            Box(Modifier.fillMaxWidth().fillMaxHeight().clip(RoundedCornerShape(5.dp)).background(ProgressTrack))
            Box(Modifier.width(w * p.coerceIn(0.0, 1.0).toFloat()).fillMaxHeight().clip(RoundedCornerShape(5.dp))
                .background(if (flagged) Primary else Muted))
            Box(Modifier.offset(x = w * threshold.coerceIn(0.0, 1.0).toFloat() - 1.dp).width(2.dp)
                .height(10.dp).background(Ink))
        }
    }
}
