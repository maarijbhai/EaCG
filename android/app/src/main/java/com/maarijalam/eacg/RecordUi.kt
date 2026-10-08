package com.maarijalam.eacg

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.maarijalam.eacg.ui.EaIcons
import com.maarijalam.eacg.ui.drawEcgPaper
import com.maarijalam.eacg.ui.drawTrace
import com.maarijalam.eacg.ui.samplePath
import com.maarijalam.eacg.ui.theme.BadText
import com.maarijalam.eacg.ui.theme.EcgTrace
import com.maarijalam.eacg.ui.theme.Primary
import com.maarijalam.eacg.ui.theme.WarnIcon
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale

/*
 * How a saved record is named and summarised in lists (Home's Recent,
 * Records). Display only.
 */

private val PTBXL = Regex("""ptbxl(\d+)""")

/** "PTB-XL 40" for a benchmarked record (its target), else "Photo". */
fun recordTitle(target: String?): String =
    target?.let { PTBXL.find(it)?.groupValues?.get(1)?.let { id -> "PTB-XL $id" } ?: it } ?: "Photo"

/** The record's own name if it was renamed, else [recordTitle] of its target. */
fun recordName(m: RecordStore.Meta): String = m.name ?: recordTitle(m.target)

/** "Today, 16:42", "Yesterday, 10:27" or "5 Oct, 10:27" (with the year if not this one). */
fun recordWhen(ms: Long, now: Long = System.currentTimeMillis()): String {
    val time = SimpleDateFormat("HH:mm", Locale.getDefault()).format(Date(ms))
    return "${dayLabel(ms, now)}, $time"
}

/** "Today", "Yesterday", "5 Oct" or "5 Oct 2025": the Records list's day groups. */
fun dayLabel(ms: Long, now: Long = System.currentTimeMillis()): String {
    val c = Calendar.getInstance().apply { timeInMillis = ms }
    val n = Calendar.getInstance().apply { timeInMillis = now }
    val dayDiff = ((startOfDay(n) - startOfDay(c)) / 86_400_000L).toInt()
    return when {
        dayDiff == 0 -> "Today"
        dayDiff == 1 -> "Yesterday"
        c.get(Calendar.YEAR) == n.get(Calendar.YEAR) ->
            SimpleDateFormat("d MMM", Locale.getDefault()).format(Date(ms))
        else -> SimpleDateFormat("d MMM yyyy", Locale.getDefault()).format(Date(ms))
    }
}

private fun startOfDay(c: Calendar): Long = (c.clone() as Calendar).apply {
    set(Calendar.HOUR_OF_DAY, 0); set(Calendar.MINUTE, 0); set(Calendar.SECOND, 0); set(Calendar.MILLISECOND, 0)
}.timeInMillis

/**
 * A quality verdict as an icon plus coloured text (no pill badges; plan §2.3):
 * ✓ Good, ⚠ Check timing, ⚠ Unreliable. Nothing for records saved before
 * the verdict existed.
 */
@Composable
fun QualityStatus(quality: String?, modifier: Modifier = Modifier) {
    val (text, colour, ok) = when (quality) {
        "ok" -> Triple("Good", Primary, true)
        "check" -> Triple("Check timing", WarnIcon, false)
        "unreliable" -> Triple("Unreliable", BadText, false)
        else -> return
    }
    Row(modifier, verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(5.dp)) {
        Icon(if (ok) EaIcons.Check else EaIcons.Warning, contentDescription = null, tint = colour,
            modifier = Modifier.size(16.dp))
        Text(text, style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold), color = colour)
    }
}

/** First 2 s of a record's rhythm strip, cached per record id for the list thumbnails. */
private val thumbCache = object : LinkedHashMap<String, DoubleArray>(32, 0.75f, true) {
    override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, DoubleArray>?) = size > 64
}

/**
 * A record's thumbnail: pink paper with the first 2 s of its digitised lead II
 * rhythm strip (the real signal, read from the record's leads.csv).
 */
@Composable
fun RecordThumb(id: String, width: Dp = 56.dp, height: Dp = 40.dp) {
    val context = LocalContext.current
    var sig by remember(id) { mutableStateOf(synchronized(thumbCache) { thumbCache[id] }) }
    LaunchedEffect(id) {
        if (sig == null) {
            val s = runCatching {
                RecordStore.leads(context, id).firstOrNull { it.rhythm }?.mv
                    ?.copyOfRange(0, (2.0 * SheetLayout.FS).toInt())
            }.getOrNull()
            if (s != null) { synchronized(thumbCache) { thumbCache[id] = s }; sig = s }
        }
    }
    Box(Modifier.size(width, height).clip(RoundedCornerShape(8.dp))) {
        Canvas(Modifier.size(width, height)) {
            drawEcgPaper(minorPx = 4.dp.toPx() / 5f * 1.25f)
            val v = sig ?: return@Canvas
            val fin = v.filter { it.isFinite() }
            if (fin.isEmpty()) return@Canvas
            val lo = fin.min(); val hi = fin.max()
            val span = (hi - lo).coerceAtLeast(0.5)
            val pad = size.height * 0.15f
            val path = samplePath(v.size,
                x = { i -> i * size.width / (v.size - 1) },
                y = { i -> (size.height - pad - ((v[i] - lo) / span * (size.height - 2 * pad))).toFloat() },
                finite = { i -> v[i].isFinite() })
            drawTrace(path, EcgTrace, 1.1.dp.toPx())
        }
    }
}
