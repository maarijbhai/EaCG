package com.maarijalam.eacg

import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
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
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.unit.toSize
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.maarijalam.eacg.ui.EaIcons
import com.maarijalam.eacg.ui.dottedStroke
import com.maarijalam.eacg.ui.samplePath
import com.maarijalam.eacg.ui.traceStroke
import com.maarijalam.eacg.ui.theme.CardBorder
import com.maarijalam.eacg.ui.theme.CardSurface
import com.maarijalam.eacg.ui.theme.Data
import com.maarijalam.eacg.ui.theme.EcgTrace
import com.maarijalam.eacg.ui.theme.InputOutline
import com.maarijalam.eacg.ui.theme.MissingTick
import com.maarijalam.eacg.ui.theme.Muted
import com.maarijalam.eacg.ui.theme.PlexMono
import com.maarijalam.eacg.ui.theme.PlexSans
import com.maarijalam.eacg.ui.theme.Primary
import com.maarijalam.eacg.ui.theme.PrimaryDark
import com.maarijalam.eacg.ui.theme.RowDivider
import com.maarijalam.eacg.ui.theme.Text2
import com.maarijalam.eacg.ui.theme.TruthTrace
import java.util.Locale
import kotlin.math.max

/**
 * The page overlay (docs/ui_plan.md §3.5, mock-up artboard 6): the straightened
 * page with the digitised trace drawn thin and dotted in teal over the
 * printed one, the time markers (t0 and the separator ticks; amber where a
 * tick was missing or replaced), pinch / double-tap / button zoom, and a
 * card that steps lead by lead. In prototype mode, "Show truth" adds the
 * PTB-XL signal in grey. Display only: every position drawn is one the
 * pipeline produced (page pixels), nothing is recomputed.
 */
@Composable
fun PageOverlay(
    page: Bitmap, traces: List<SheetPipeline.RowTrace>, t0X: Double, rowY: List<Int>,
    ticks: List<Triple<Int, Int, Double>>, replaced: List<Triple<Int, Int, Double>>,
    cells: List<SheetPipeline.CellBox>, pageW: Int, pageH: Int,
    leads: List<SheetPipeline.LeadResult>?, pxPerMm: Double, truth12: Map<String, DoubleArray>?,
) {
    val proto = LocalAppSettings.current.prototypeMode
    var showTruth by remember { mutableStateOf(false) }
    var zoom by remember(page) { mutableFloatStateOf(1f) }
    var pan by remember(page) { mutableStateOf(Offset.Zero) }
    var viewSize by remember { mutableStateOf(Size.Zero) }
    // Lead order for stepping: the 3x4 block row by row, then the rhythm strip.
    val order = remember(cells) {
        cells.sortedWith(compareBy({ if (it.rhythm) 1 else 0 }, { it.y0 }, { it.x0 }))
    }
    var current by remember(cells) { mutableIntStateOf(-1) }
    val pw = if (pageW > 0) pageW else page.width
    val ph = if (pageH > 0) pageH else page.height
    val image = remember(page) { page.asImageBitmap() }
    val measurer = rememberTextMeasurer()

    fun baseScale() = if (viewSize.width > 0) viewSize.width / pw else 1f
    fun clampPan(p: Offset, z: Float): Offset {
        val k = baseScale() * z
        val minX = minOf(0f, viewSize.width - pw * k)
        val minY = minOf(0f, viewSize.height - ph * k)
        return Offset(p.x.coerceIn(minX, 0f), p.y.coerceIn(minY, maxOf(0f, (viewSize.height - ph * k) / 2)))
    }
    fun zoomAt(newZoom: Float, focus: Offset) {
        val z = newZoom.coerceIn(1f, 12f)
        // Keep the page point under [focus] fixed.
        val k0 = baseScale() * zoom
        val pagePt = (focus - pan) / k0
        val k1 = baseScale() * z
        pan = clampPan(focus - pagePt * k1, z)
        zoom = z
    }
    fun fit() { zoom = 1f; pan = clampPan(Offset(0f, (viewSize.height - ph * baseScale()) / 2), 1f) }
    fun showCell(i: Int) {
        if (order.isEmpty()) return
        val c = order[(i + order.size) % order.size]
        current = (i + order.size) % order.size
        val cw = max(1, c.x1 - c.x0).toFloat()
        val z = (viewSize.width * 0.92f / (cw * baseScale())).coerceIn(1f, 12f)
        val k = baseScale() * z
        val cx = (c.x0 + c.x1) / 2f
        val cy = (c.y0 + c.y1) / 2f
        zoom = z
        pan = clampPan(Offset(viewSize.width / 2 - cx * k, viewSize.height / 2 - cy * k), z)
    }

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(start = 16.dp, end = 16.dp,
        top = 12.dp, bottom = 16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        val shape = RoundedCornerShape(16.dp)
        Box(Modifier.fillMaxWidth().height(430.dp).clip(shape).border(1.dp, CardBorder, shape)
            .background(Color(0xFF2B2B2B))
            .onSizeChanged {
                val first = viewSize == Size.Zero
                viewSize = it.toSize()
                if (first) fit()
            }) {
            Canvas(Modifier.fillMaxSize()
                .semantics { contentDescription = "The straightened page with the digitised trace drawn over it" }
                .pointerInput(page) {
                    detectTransformGestures { centroid, p, z, _ ->
                        zoomAt(zoom * z, centroid)
                        pan = clampPan(pan + p, zoom)
                    }
                }
                .pointerInput(page) {
                    detectTapGestures(onDoubleTap = { at -> if (zoom > 1.5f) fit() else zoomAt(3f, at) })
                }) {
                val k = baseScale() * zoom
                fun sx(x: Double) = (pan.x + x * k).toFloat()
                fun sy(y: Double) = (pan.y + y * k).toFloat()
                drawImage(image, srcOffset = IntOffset.Zero, srcSize = IntSize(page.width, page.height),
                    dstOffset = IntOffset(pan.x.toInt(), pan.y.toInt()),
                    dstSize = IntSize((pw * k).toInt(), (ph * k).toInt()))

                // Rows' vertical extent for the markers: halfway to the neighbours.
                val gap = if (rowY.size > 1) (rowY[1] - rowY[0]).toDouble() else ph / 5.0
                val dashed = PathEffect.dashPathEffect(floatArrayOf(4.dp.toPx(), 3.dp.toPx()))
                if (t0X.isFinite()) {
                    drawLine(Data, Offset(sx(t0X), sy(0.0)), Offset(sx(t0X), sy(ph.toDouble())), 1.5.dp.toPx(),
                        pathEffect = dashed)
                    label(measurer, "t0", Offset(sx(t0X) + 4.dp.toPx(), sy((rowY.firstOrNull() ?: 0) - gap / 2) + 2.dp.toPx()),
                        PrimaryDark, mono = true)
                }
                val kept = ticks.associate { (r, kk, x) -> (r to kk) to x }
                val rej = replaced.map { it.first to it.second }.toSet()
                for (r in 0 until minOf(3, rowY.size)) for (kk in 1..3) {
                    val x = kept[r to kk] ?: (0 until 3).mapNotNull { kept[it to kk] }.takeIf { it.isNotEmpty() }?.average()
                        ?: continue
                    val missing = (r to kk) !in kept || (r to kk) in rej
                    val colour = if (missing) MissingTick else Data
                    val y0 = rowY[r] - gap * 0.45
                    val y1 = rowY[r] + gap * 0.45
                    drawLine(colour, Offset(sx(x), sy(y0)), Offset(sx(x), sy(y1)), 1.5.dp.toPx(), pathEffect = dashed)
                    if (zoom >= 1.6f) label(measurer, "tick ${2.5 * kk} s" + if (missing) " (borrowed)" else "",
                        Offset(sx(x) + 4.dp.toPx(), sy(y1) - 16.dp.toPx()), if (missing) MissingTick else PrimaryDark, mono = true)
                }

                // PTB-XL truth (prototype mode): placed in each lead's cell, its
                // time across the cell's width and 10 mm/mV at the page scale,
                // about the digitised trace's median row. APPROXIMATE: a record
                // keeps the cell extents, not the exact time map.
                if (proto && showTruth && truth12 != null && leads != null && pxPerMm > 0) {
                    for (c in cells) {
                        val lead = leads.firstOrNull { it.name == c.name && it.rhythm == c.rhythm } ?: continue
                        val tr = truth12[c.name] ?: continue
                        val a = (lead.tStartS * SheetLayout.FS).toInt()
                        val seg = tr.copyOfRange(a, minOf(tr.size, a + lead.mv.size))
                        if (seg.isEmpty()) continue
                        val row = traces.getOrNull(cellRow(c, rowY)) ?: continue
                        val ys = (c.x0 until c.x1).mapNotNull { x -> row.pos.getOrNull(x - row.x0)?.takeIf { it.isFinite() } }
                        if (ys.isEmpty()) continue
                        val base = SheetLayout.median(ys.toDoubleArray())
                        val med = SheetLayout.median(seg)
                        val n = seg.size
                        drawPath(samplePath(n, { i -> sx(c.x0 + (c.x1 - c.x0) * i.toDouble() / (n - 1)) },
                            { i -> sy(base - (seg[i] - med) * SheetLayout.MM_PER_MV * pxPerMm) }, { i -> seg[i].isFinite() }),
                            TruthTrace, style = traceStroke(1.5.dp.toPx()))
                    }
                }

                // The digitised trace: thin, dotted, teal (plan §2.3).
                val dots = dottedStroke(1.8.dp.toPx(), density)
                for (t in traces) {
                    drawPath(samplePath(t.pos.size, { i -> sx((t.x0 + i).toDouble()) }, { i -> sy(t.pos[i]) },
                        { i -> t.pos[i].isFinite() }), Data, style = dots)
                }

                // Lead labels at each cell's start.
                for (c in cells) {
                    label(measurer, if (c.rhythm) "II rhythm" else c.name,
                        Offset(sx(c.x0.toDouble()) + 4.dp.toPx(), sy(c.y0.toDouble()) + 4.dp.toPx()), EcgTrace)
                }
            }
            // Zoom in / out / fit.
            Column(Modifier.align(Alignment.BottomEnd).padding(10.dp)
                .background(CardSurface, RoundedCornerShape(14.dp)).border(1.dp, InputOutline, RoundedCornerShape(14.dp))) {
                ZoomButton(EaIcons.Plus, "Zoom in") { zoomAt(zoom * 1.6f, Offset(viewSize.width / 2, viewSize.height / 2)) }
                HorizontalDivider(Modifier.width(44.dp), color = RowDivider)
                ZoomButton(EaIcons.Minus, "Zoom out") { zoomAt(zoom / 1.6f, Offset(viewSize.width / 2, viewSize.height / 2)) }
                HorizontalDivider(Modifier.width(44.dp), color = RowDivider)
                ZoomButton(EaIcons.Fit, "Fit page") { fit(); current = -1 }
            }
        }

        // Legend.
        Row(horizontalArrangement = Arrangement.spacedBy(16.dp), verticalAlignment = Alignment.CenterVertically) {
            LegendItem("Printed trace") { Box(Modifier.width(22.dp).height(1.5.dp).background(EcgTrace)) }
            LegendItem("Digitised") {
                Canvas(Modifier.width(22.dp).height(4.dp)) {
                    drawLine(Data, Offset(0f, size.height / 2), Offset(size.width, size.height / 2), 2.dp.toPx(),
                        cap = androidx.compose.ui.graphics.StrokeCap.Round,
                        pathEffect = PathEffect.dashPathEffect(floatArrayOf(0.1f, 3.2f * density)))
                }
            }
            LegendItem("Marker") { Box(Modifier.width(2.dp).height(14.dp).background(Data)) }
        }

        if (proto && truth12 != null) {
            Row(Modifier.fillMaxWidth().heightIn(min = 48.dp).clickable(role = Role.Switch) { showTruth = !showTruth },
                verticalAlignment = Alignment.CenterVertically) {
                Text("Show the PTB-XL truth (grey, approximate placement)", style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.weight(1f))
                Switch(checked = showTruth, onCheckedChange = null,
                    colors = SwitchDefaults.colors(checkedTrackColor = Primary, checkedThumbColor = Color.White))
            }
        }

        if (order.isNotEmpty()) LeadCard(order, current, onPrev = { showCell(if (current < 0) 0 else current - 1) },
            onNext = { showCell(if (current < 0) 0 else current + 1) })
    }
}

/** Which row a cell belongs to (nearest row centre). */
private fun cellRow(c: SheetPipeline.CellBox, rowY: List<Int>): Int {
    if (rowY.isEmpty()) return 0
    val cy = (c.y0 + c.y1) / 2
    return rowY.indices.minByOrNull { kotlin.math.abs(rowY[it] - cy) } ?: 0
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.label(
    measurer: androidx.compose.ui.text.TextMeasurer, text: String, at: Offset, colour: Color, mono: Boolean = false,
) {
    val style = TextStyle(fontFamily = if (mono) PlexMono else PlexSans, fontSize = if (mono) 11.sp else 13.sp,
        fontWeight = if (mono) FontWeight.Normal else FontWeight.SemiBold, color = colour)
    val layout = measurer.measure(text, style)
    val pad = 3.dp.toPx()
    drawRoundRect(Color.White.copy(alpha = 0.9f), topLeft = at - Offset(pad, pad / 2),
        size = Size(layout.size.width + 2 * pad, layout.size.height + pad),
        cornerRadius = androidx.compose.ui.geometry.CornerRadius(4.dp.toPx()))
    drawText(layout, topLeft = at)
}

@Composable
private fun ZoomButton(icon: androidx.compose.ui.graphics.vector.ImageVector, label: String, onClick: () -> Unit) {
    IconButton(onClick = onClick, modifier = Modifier.size(44.dp)) {
        Icon(icon, contentDescription = label, tint = Text2, modifier = Modifier.size(20.dp))
    }
}

@Composable
private fun LegendItem(text: String, swatch: @Composable () -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        swatch()
        Text(text, style = MaterialTheme.typography.bodySmall, color = Text2)
    }
}

/** "Lead V5 · 7.5 to 10.0 s", "Timed from the 7.5 s tick", with previous / next. */
@Composable
private fun LeadCard(order: List<SheetPipeline.CellBox>, current: Int, onPrev: () -> Unit, onNext: () -> Unit) {
    val shape = RoundedCornerShape(16.dp)
    Row(Modifier.fillMaxWidth().background(CardSurface, shape).border(1.dp, CardBorder, shape)
        .padding(start = 14.dp, end = 8.dp, top = 10.dp, bottom = 10.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            if (current < 0) {
                Text("Step through the leads", style = MaterialTheme.typography.titleSmall)
                Text("Zooms to each lead in turn", style = MaterialTheme.typography.bodySmall, color = Muted)
            } else {
                val c = order[current]
                val col = order.filter { !it.rhythm && kotlin.math.abs(it.y0 - c.y0) < 5 }.sortedBy { it.x0 }.indexOf(c)
                if (c.rhythm) {
                    Text("Lead II rhythm · 0 to 10 s", style = MaterialTheme.typography.titleSmall)
                    Text("Timed from the pulse and the ticks above it", style = MaterialTheme.typography.bodySmall, color = Muted)
                } else {
                    val a = col * SheetLayout.CELL_S
                    Text(String.format(Locale.US, "Lead %s · %.1f to %.1f s", c.name, a, a + SheetLayout.CELL_S),
                        style = MaterialTheme.typography.titleSmall)
                    Text(if (col == 0) "Timed from the calibration pulse"
                        else String.format(Locale.US, "Timed from the %.1f s tick", a),
                        style = MaterialTheme.typography.bodySmall, color = Muted)
                }
            }
        }
        RoundIcon(EaIcons.ChevronLeft, "Previous lead", onPrev)
        RoundIcon(EaIcons.ChevronRight, "Next lead", onNext)
    }
}

@Composable
private fun RoundIcon(icon: androidx.compose.ui.graphics.vector.ImageVector, label: String, onClick: () -> Unit) {
    Box(Modifier.size(44.dp).border(1.dp, InputOutline, RoundedCornerShape(22.dp)).clip(RoundedCornerShape(22.dp))
        .clickable(role = Role.Button, onClick = onClick).semantics { contentDescription = label },
        contentAlignment = Alignment.Center) {
        Icon(icon, contentDescription = null, tint = Text2, modifier = Modifier.size(20.dp))
    }
}
