package com.maarijalam.eacg

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
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
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.maarijalam.eacg.ui.EaIcons
import com.maarijalam.eacg.ui.TopBar
import com.maarijalam.eacg.ui.drawEcgPaper
import com.maarijalam.eacg.ui.samplePath
import com.maarijalam.eacg.ui.traceStroke
import com.maarijalam.eacg.ui.theme.ButtonOutline
import com.maarijalam.eacg.ui.theme.CardBorder
import com.maarijalam.eacg.ui.theme.CardSurface
import com.maarijalam.eacg.ui.theme.Data
import com.maarijalam.eacg.ui.theme.DataText
import com.maarijalam.eacg.ui.theme.Ink
import com.maarijalam.eacg.ui.theme.Muted
import com.maarijalam.eacg.ui.theme.Primary
import com.maarijalam.eacg.ui.theme.ProgressTrack
import com.maarijalam.eacg.ui.theme.RowDivider
import java.util.Locale

private enum class StageState { DONE, ACTIVE, PENDING, SKIPPED }

/** One line of the stage list: pipeline timing keys it covers, and its copy. */
private class UiStage(val name: String, val keys: List<String>, val detail: String)

/**
 * Digitising (docs/ui_plan.md §3.3, mock-up artboard 3): the page with each
 * row drawn in teal as it is read, the tile count, and the stages with their
 * times as they finish. Every number shown comes from
 * SheetPipeline.Progress; nothing here measures anything.
 */
@Composable
fun ProcessingScreen(scan: ScanController, modifier: Modifier = Modifier) {
    val learned = scan.runTracer == SheetPipeline.Tracer.MODEL
    val stages = listOf(
        UiStage("Straighten page", listOf("corners", "rectify"), "Corners and perspective"),
        UiStage("Grid fit", listOf("grid"), "Every 5 mm line made straight and even"),
        UiStage("Find rows and markers", listOf("locate", "calibrate+segment"), ""),
        UiStage("Read the trace", listOf("trace"), ""),
        UiStage("Cut into 12 leads", listOf("cut", "quality"), "Seconds and millivolts, 500 Hz"),
        UiStage("Classify", listOf("classify"), "Research classifier, five superclasses"),
        UiStage("Save record", emptyList(), "Stored privately on this phone"),
    )
    val ms = scan.stageMs
    fun stateOf(i: Int): StageState {
        val st = stages[i]
        if (st.name == "Grid fit" && !scan.runGridFit) return StageState.SKIPPED
        val done = if (st.keys.isEmpty()) (scan.saveMs ?: -1L) >= 0 else st.keys.all { it in ms }
        if (done) return StageState.DONE
        val firstOpen = stages.indices.firstOrNull { j ->
            val s = stages[j]
            !(s.name == "Grid fit" && !scan.runGridFit) &&
                !(if (s.keys.isEmpty()) (scan.saveMs ?: -1L) >= 0 else s.keys.all { it in ms })
        }
        return if (firstOpen == i) StageState.ACTIVE else StageState.PENDING
    }
    val rowsDone = scan.tracedRows.size

    Column(modifier.fillMaxSize()) {
        TopBar("Digitising", onBack = { scan.cancel() }, backIcon = EaIcons.Close, backLabel = "Cancel")
        Column(Modifier.fillMaxWidth().verticalScroll(rememberScrollState())
            .padding(start = 16.dp, end = 16.dp, top = 8.dp, bottom = 16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)) {

            PagePreview(scan, activeRow = if (stateOf(3) == StageState.ACTIVE) rowsDone else -1)

            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Row(verticalAlignment = Alignment.Bottom) {
                    Text(if (stateOf(3) == StageState.DONE) "Trace read" else "Reading the trace",
                        style = MaterialTheme.typography.titleMedium.copy(fontSize = MaterialTheme.typography.titleMedium.fontSize * 1.1f),
                        modifier = Modifier.weight(1f))
                    Text(if (learned && scan.tilesTotal > 0) "${scan.tilesDone} / ${scan.tilesTotal} tiles"
                        else "$rowsDone / 4 rows", style = DataText, color = Muted)
                }
                val frac = if (learned && scan.tilesTotal > 0) scan.tilesDone.toFloat() / scan.tilesTotal
                    else rowsDone / 4f
                LinearProgressIndicator(progress = { frac },
                    modifier = Modifier.fillMaxWidth().height(8.dp).clip(RoundedCornerShape(4.dp))
                        .semantics { contentDescription = "Trace reading progress" },
                    color = Primary, trackColor = ProgressTrack, strokeCap = StrokeCap.Round,
                    gapSize = 0.dp, drawStopIndicator = {})
            }

            ScanRuns.label(scan.runIndex, scan.runCount, scan.runTracer)?.let {
                Text(it, style = MaterialTheme.typography.titleSmall, color = Primary)
            }
            val shape = RoundedCornerShape(16.dp)
            Column(Modifier.fillMaxWidth().background(CardSurface, shape).border(1.dp, CardBorder, shape)) {
                stages.forEachIndexed { i, st ->
                    if (i > 0) HorizontalDivider(color = RowDivider)
                    val state = stateOf(i)
                    val time: Long? = when {
                        state != StageState.DONE -> null
                        st.keys.isEmpty() -> scan.saveMs
                        else -> st.keys.sumOf { ms[it] ?: 0L }
                    }
                    val detail = when (st.name) {
                        "Find rows and markers" -> listOfNotNull(scan.stageDetail["locate"],
                            scan.stageDetail["calibrate+segment"]).joinToString(" · ")
                            .ifEmpty { "Rows, calibration pulses and lead markers" }
                        "Read the trace" -> (if (learned) "Learned tracer" else "Classical tracer") +
                            if (state == StageState.ACTIVE) ", row ${minOf(rowsDone + 1, 4)} of 4" else ""
                        "Grid fit" -> if (state == StageState.SKIPPED) "Off: corners only" else st.detail
                        "Classify" -> scan.stageDetail["classify"]
                            ?.takeIf { state == StageState.DONE } ?: st.detail
                        else -> st.detail
                    }
                    StageRow(st.name, detail, state, time)
                }
            }
        }
    }
}

@Composable
private fun StageRow(name: String, detail: String, state: StageState, ms: Long?) {
    Row(Modifier.fillMaxWidth().heightIn(min = 56.dp).padding(horizontal = 14.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Box(Modifier.size(28.dp), contentAlignment = Alignment.Center) {
            when (state) {
                StageState.DONE -> Box(Modifier.size(28.dp).background(Primary, RoundedCornerShape(14.dp)),
                    contentAlignment = Alignment.Center) {
                    Icon(EaIcons.Check, contentDescription = "Done", tint = Color.White, modifier = Modifier.size(16.dp))
                }
                StageState.ACTIVE -> CircularProgressIndicator(Modifier.size(24.dp), color = Primary,
                    trackColor = ProgressTrack, strokeWidth = 3.dp)
                StageState.PENDING, StageState.SKIPPED ->
                    Box(Modifier.size(22.dp).border(2.dp, ButtonOutline, RoundedCornerShape(11.dp)))
            }
        }
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(1.dp)) {
            Text(name, style = MaterialTheme.typography.bodyLarge.copy(
                fontWeight = if (state == StageState.ACTIVE) FontWeight.SemiBold else FontWeight.Medium),
                color = if (state == StageState.PENDING || state == StageState.SKIPPED) Muted else Ink)
            if (detail.isNotEmpty()) Text(detail, style = MaterialTheme.typography.labelSmall, color = Muted)
        }
        if (ms != null) Text(if (ms >= 1000) String.format(Locale.US, "%.1f s", ms / 1000.0) else "$ms ms",
            style = DataText, color = Muted)
    }
}

/**
 * The page being read (the pipeline's own preview), each traced row drawn
 * over it in teal, and a light band on the row being read. Before the page
 * exists: plain paper.
 */
@Composable
private fun PagePreview(scan: ScanController, activeRow: Int) {
    val pv = scan.pagePreview
    val (pw, ph) = scan.pageSize
    val shape = RoundedCornerShape(16.dp)
    val aspect = if (pv != null && pw > 0) pw.toFloat() / ph else 297f / 210f
    Box(Modifier.fillMaxWidth().aspectRatio(aspect).clip(shape).border(1.dp, CardBorder, shape)) {
        if (pv != null) {
            Image(pv.asImageBitmap(), contentDescription = "The straightened page",
                contentScale = ContentScale.FillBounds, modifier = Modifier.fillMaxSize())
        } else {
            Canvas(Modifier.fillMaxSize()) { drawEcgPaper(minorPx = 4.dp.toPx()) }
        }
        if (pv != null && pw > 0) {
            Canvas(Modifier.fillMaxSize()) {
                val k = size.width / pw
                for ((_, t) in scan.tracedRows) {
                    val path = samplePath(t.pos.size, x = { i -> (t.x0 + i) * k }, y = { i -> t.pos[i].toFloat() * k },
                        finite = { i -> t.pos[i].isFinite() })
                    drawPath(path, Data, style = traceStroke(2.dp.toPx()))
                }
                // The row being read: a band on its centre as found (SheetPipeline.Progress.Rows).
                val rc = scan.rowCentres
                if (rc != null && activeRow in rc.indices) {
                    val gap = if (rc.size > 1) (rc[1] - rc[0]) * k else size.height / 5
                    val cy = rc[activeRow] * k
                    drawRect(Data.copy(alpha = 0.08f), Offset(0f, cy - gap * 0.45f), Size(size.width, gap * 0.9f))
                }
            }
        }
    }
}
