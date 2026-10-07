package com.maarijalam.eacg.ui

import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import com.maarijalam.eacg.ui.theme.EcgMajor
import com.maarijalam.eacg.ui.theme.EcgMinor
import com.maarijalam.eacg.ui.theme.EcgPaper
import kotlin.math.exp
import kotlin.math.pow

/*
 * The pink ECG paper and the traces drawn on it (docs/ui_plan.md §2.1, §2.3).
 * Display only: nothing here touches a signal's values.
 */

/** Pink paper: 1 mm minor rules every [minorPx], a 5 mm major every fifth. */
fun DrawScope.drawEcgPaper(minorPx: Float, originX: Float = 0f, originY: Float = 0f) {
    drawRect(EcgPaper)
    if (minorPx < 1.5f) return
    var i = 0
    var x = originX
    while (x <= size.width) {
        if (i % 5 != 0) drawLine(EcgMinor, Offset(x, 0f), Offset(x, size.height), 1f)
        x += minorPx; i++
    }
    i = 0
    var y = originY
    while (y <= size.height) {
        if (i % 5 != 0) drawLine(EcgMinor, Offset(0f, y), Offset(size.width, y), 1f)
        y += minorPx; i++
    }
    x = originX
    while (x <= size.width) { drawLine(EcgMajor, Offset(x, 0f), Offset(x, size.height), 1f); x += 5 * minorPx }
    y = originY
    while (y <= size.height) { drawLine(EcgMajor, Offset(0f, y), Offset(size.width, y), 1f); y += 5 * minorPx }
}

/**
 * A polyline through samples, breaking at NaN (a gap stays a gap). [x] and
 * [y] map the sample index and value to canvas coordinates.
 */
fun samplePath(n: Int, x: (Int) -> Float, y: (Int) -> Float, finite: (Int) -> Boolean): Path {
    val p = Path()
    var pen = false
    for (i in 0 until n) {
        if (!finite(i)) { pen = false; continue }
        if (pen) p.lineTo(x(i), y(i)) else p.moveTo(x(i), y(i))
        pen = true
    }
    return p
}

/** Solid trace stroke (the printed trace, or a lead on the paper). */
fun traceStroke(widthPx: Float) = Stroke(width = widthPx, cap = StrokeCap.Round, join = StrokeJoin.Round)

/**
 * The plan's dotted overlay for a DIGITISED trace over the page: ~1.8 px,
 * round caps, dash [0.1, 3.2 dp] -- dots that let the printed ink show
 * through between them.
 */
fun dottedStroke(widthPx: Float, density: Float) = Stroke(
    width = widthPx, cap = StrokeCap.Round, join = StrokeJoin.Round,
    pathEffect = PathEffect.dashPathEffect(floatArrayOf(0.1f, 3.2f * density)))

/**
 * A decorative ECG-like waveform (the mock-up's sum of Gaussians) for
 * illustrations only -- never a recorded signal.
 */
fun decorativeEcg(width: Float, height: Float, beats: Int, amp: Float, inverted: Boolean = false,
                  offset: Float = 0f): Path {
    val p = Path()
    val period = width / beats
    val sgn = if (inverted) -1f else 1f
    var x = 0f
    var first = true
    while (x <= width) {
        val ph = ((x + offset) % period) / period
        fun g(c: Float, s: Float) = exp(-((ph - c) / s).toDouble().pow(2)).toFloat()
        val v = 0.13f * g(0.2f, 0.035f) - 0.12f * g(0.335f, 0.012f) + g(0.36f, 0.012f) -
            0.28f * g(0.385f, 0.012f) + 0.28f * g(0.6f, 0.05f)
        val y = height / 2 - sgn * v * amp
        if (first) p.moveTo(x, y) else p.lineTo(x, y)
        first = false
        x += 0.5f
    }
    return p
}

/** Draw [path] in [color] as a solid trace. */
fun DrawScope.drawTrace(path: Path, color: Color, widthPx: Float) =
    drawPath(path, color, style = traceStroke(widthPx))
