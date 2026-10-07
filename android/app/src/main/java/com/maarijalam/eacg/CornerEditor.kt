package com.maarijalam.eacg

import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.border
import androidx.compose.ui.draw.clip
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.drag
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Four draggable sheet corners over a photograph, with a magnifier.
 *
 * WHY NOT TAPS. The first photograph scored on the A30s (PTB-XL 40,
 * 2026-09-28) was tapped on a 1080 px preview of a 3200 px decode, shown on
 * a 720 px wide screen: one screen pixel is 4-5 photograph pixels, and the
 * taps landed up to 92 px from the sheet corners. [PageCorners] searches
 * +-1.5% of each edge either side of the line it is given -- about 30-42 px
 * on that page -- so the true edge was outside its window along part of the
 * sheet, the edge fits came back at 16 px RMS, and everything downstream
 * inherited the error. Refinement is built for a click within about 10 px.
 *
 * So a corner is dragged, not tapped: it can be corrected rather than
 * redone, and while it moves a magnifier drawn from the FULL-RESOLUTION
 * photograph shows exactly where it sits, above the finger that would
 * otherwise hide it. A drag that does not start on a handle is left alone,
 * so the screen still scrolls.
 *
 * Corners are in bitmap pixels, TL TR BR BL, x then y: 8 values.
 */
@Composable
fun CornerEditor(
    photo: Bitmap,
    preview: Bitmap,
    corners: DoubleArray,
    onCornersChange: (DoubleArray) -> Unit,
    enabled: Boolean,
    refined: DoubleArray?,
    modifier: Modifier = Modifier
) {
    val full: ImageBitmap = remember(photo) { photo.asImageBitmap() }
    val small: ImageBitmap = remember(preview) { preview.asImageBitmap() }
    var active by remember { mutableIntStateOf(-1) }
    val current by rememberUpdatedState(corners)
    val onChange by rememberUpdatedState(onCornersChange)
    val isEnabled by rememberUpdatedState(enabled)
    val density = LocalDensity.current
    val grabPx = with(density) { GRAB_RADIUS.toPx() }

    Box(
        modifier.fillMaxWidth()
            .aspectRatio(photo.width.toFloat() / photo.height)
            .pointerInput(photo) {
                awaitEachGesture {
                    val down = awaitFirstDown(requireUnconsumed = false)
                    if (!isEnabled) return@awaitEachGesture
                    val scale = CornerGeometry.displayScale(
                        size.width.toDouble(), photo.width.toDouble())
                    val idx = CornerGeometry.nearestCorner(
                        current, down.position.x / scale, down.position.y / scale,
                        grabPx / scale)
                    // Not on a handle: leave the gesture to the scrolling screen.
                    if (idx < 0) return@awaitEachGesture
                    down.consume()
                    active = idx
                    try {
                        drag(down.id) { change ->
                            val d = change.position - change.previousPosition
                            val moved = current.copyOf()
                            moved[2 * idx] = (moved[2 * idx] + d.x / scale)
                                .coerceIn(0.0, photo.width - 1.0)
                            moved[2 * idx + 1] = (moved[2 * idx + 1] + d.y / scale)
                                .coerceIn(0.0, photo.height - 1.0)
                            onChange(moved)
                            change.consume()
                        }
                    } finally {
                        active = -1
                    }
                }
            }
    ) {
        Image(small, contentDescription = "photograph",
            contentScale = ContentScale.FillBounds, modifier = Modifier.fillMaxSize())
        Canvas(Modifier.fillMaxSize()) {
            val s = size.width / photo.width
            val pts = (0 until 4).map {
                Offset((corners[2 * it] * s).toFloat(), (corners[2 * it + 1] * s).toFloat())
            }
            val quad = androidx.compose.ui.graphics.Path().apply {
                moveTo(pts[0].x, pts[0].y); for (i in 1 until 4) lineTo(pts[i].x, pts[i].y); close()
            }
            drawPath(quad, QUAD.copy(alpha = 0.10f))
            for (i in 0 until 4) drawLine(QUAD, pts[i], pts[(i + 1) % 4], 2.dp.toPx())
            refined?.let { c ->
                val q = (0 until 4).map {
                    Offset((c[2 * it] * s).toFloat(), (c[2 * it + 1] * s).toFloat())
                }
                for (i in 0 until 4) drawLine(REFINED, q[i], q[(i + 1) % 4], 2.5f)
            }
            // The mock-up's handle: white disc, 3 dp ring (grabbed from GRAB_RADIUS).
            val r = 11.dp.toPx()
            pts.forEachIndexed { i, p ->
                val colour = if (i == active) ACTIVE else HANDLE
                drawCircle(Color.White, r, p)
                drawCircle(colour, r, p, style = Stroke(3.dp.toPx()))
                drawCircle(colour, 2.dp.toPx(), p)
            }
            if (active >= 0) {
                val loupe = LOUPE_SIZE.toPx()
                val margin = 8.dp.toPx()
                val left = pts[active].x < size.width / 2
                val top = pts[active].y < size.height / 2
                // In the quadrant diagonally away from the finger.
                val at = Offset(
                    if (left) size.width - loupe - margin else margin,
                    if (top) size.height - loupe - margin else margin
                )
                drawLoupe(full, corners, active, at, loupe)
            }
        }
    }
}

/** Magnified view of the photograph around corner [idx], drawn at [at]. */
internal fun DrawScope.drawLoupe(
    full: ImageBitmap, corners: DoubleArray, idx: Int, at: Offset, side: Float,
    halfSrc: Int = LOUPE_HALF_SRC_PX
) {
    val cx = corners[2 * idx]
    val cy = corners[2 * idx + 1]
    val win = CornerGeometry.loupeWindow(cx, cy, halfSrc,
        full.width, full.height, side.toDouble())
    drawRect(Color.Black, at, Size(side, side))
    clipRect(at.x, at.y, at.x + side, at.y + side) {
        if (win != null) {
            drawImage(
                full,
                srcOffset = IntOffset(win.srcX, win.srcY),
                srcSize = IntSize(win.srcW, win.srcH),
                dstOffset = IntOffset((at.x + win.dstX).roundToInt(), (at.y + win.dstY).roundToInt()),
                dstSize = IntSize(win.dstW.roundToInt(), win.dstH.roundToInt())
            )
        }
        // The two sheet edges that meet here, in loupe coordinates.
        val k = side / (2.0 * halfSrc)
        fun toLoupe(i: Int) = Offset(
            (at.x + side / 2 + (corners[2 * i] - cx) * k).toFloat(),
            (at.y + side / 2 + (corners[2 * i + 1] - cy) * k).toFloat()
        )
        val c = toLoupe(idx)
        drawLine(HANDLE, c, toLoupe((idx + 1) % 4), 2f)
        drawLine(HANDLE, c, toLoupe((idx + 3) % 4), 2f)
        val arm = side * 0.12f
        drawLine(ACTIVE, c - Offset(arm, 0f), c + Offset(arm, 0f), 2f)
        drawLine(ACTIVE, c - Offset(0f, arm), c + Offset(0f, arm), 2f)
    }
    drawRect(Color.White, at, Size(side, side), style = Stroke(3f))
}

/**
 * All four corners at once, close up, so each can be checked before scoring
 * without dragging it. Run 5's corners looked right on the preview and were
 * 2-9 mm off: at the preview's scale a 5 mm square is ~13 screen pixels.
 * Each tile shows [CHECK_HALF_SRC_PX] photograph pixels either side of its
 * corner, with the two sheet edges that meet there.
 */
@Composable
fun CornerCheckRow(photo: Bitmap, corners: DoubleArray, modifier: Modifier = Modifier) {
    val full: ImageBitmap = remember(photo) { photo.asImageBitmap() }
    androidx.compose.foundation.layout.Row(
        modifier.fillMaxWidth(),
        horizontalArrangement = androidx.compose.foundation.layout.Arrangement.spacedBy(6.dp)
    ) {
        // Top-left, top-right, bottom-left, bottom-right: the sheet's top
        // pair, then its bottom pair.
        for (idx in intArrayOf(0, 1, 3, 2)) {
            Canvas(Modifier.weight(1f).aspectRatio(1f)) {
                drawLoupe(full, corners, idx, Offset.Zero, size.width, CHECK_HALF_SRC_PX)
            }
        }
    }
}

/**
 * The redesign's corner close-ups (docs/ui_plan.md §3.2): the same check as
 * [CornerCheckRow] -- [CHECK_HALF_SRC_PX] photograph pixels either side of
 * each corner, with its two edges and a crosshair -- in four rounded,
 * labelled tiles, Top left ... Bottom right.
 */
@Composable
fun CornerCloseUps(photo: Bitmap, corners: DoubleArray, modifier: Modifier = Modifier) {
    val full: ImageBitmap = remember(photo) { photo.asImageBitmap() }
    val labels = mapOf(0 to "Top left", 1 to "Top right", 3 to "Bottom left", 2 to "Bottom right")
    androidx.compose.foundation.layout.Row(modifier.fillMaxWidth(),
        horizontalArrangement = androidx.compose.foundation.layout.Arrangement.spacedBy(8.dp)) {
        for (idx in intArrayOf(0, 1, 3, 2)) {
            androidx.compose.foundation.layout.Column(Modifier.weight(1f),
                horizontalAlignment = androidx.compose.ui.Alignment.CenterHorizontally,
                verticalArrangement = androidx.compose.foundation.layout.Arrangement.spacedBy(4.dp)) {
                val shape = androidx.compose.foundation.shape.RoundedCornerShape(12.dp)
                Canvas(Modifier.fillMaxWidth().aspectRatio(1f).clip(shape)
                    .border(1.dp, com.maarijalam.eacg.ui.theme.InputOutline, shape)) {
                    drawLoupe(full, corners, idx, Offset.Zero, size.width, CHECK_HALF_SRC_PX)
                }
                androidx.compose.material3.Text(labels.getValue(idx),
                    style = androidx.compose.material3.MaterialTheme.typography.labelSmall,
                    color = com.maarijalam.eacg.ui.theme.Muted)
            }
        }
    }
}

/**
 * The editor's arithmetic, free of Compose so the JVM tests can reach it.
 */
object CornerGeometry {

    /** Display px per bitmap px for a box of [boxWidth] showing [bitmapWidth]. */
    fun displayScale(boxWidth: Double, bitmapWidth: Double): Double = boxWidth / bitmapWidth

    /** Index of the corner within [radius] of (x, y), nearest first; -1 if none. */
    fun nearestCorner(corners: DoubleArray, x: Double, y: Double, radius: Double): Int {
        var best = -1
        var bestD = radius
        for (i in 0 until 4) {
            val d = hypot(corners[2 * i] - x, corners[2 * i + 1] - y)
            if (d <= bestD) { best = i; bestD = d }
        }
        return best
    }

    /** A rectangle inset by [fraction] of each side: where the handles start. */
    fun initialCorners(width: Int, height: Int, fraction: Double = 0.1): DoubleArray {
        val x0 = width * fraction
        val y0 = height * fraction
        val x1 = width * (1 - fraction)
        val y1 = height * (1 - fraction)
        return doubleArrayOf(x0, y0, x1, y0, x1, y1, x0, y1)
    }

    /**
     * Corners closer than [marginFraction] of the long side to the photograph
     * edge, by index. The refinement finds each sheet edge as the step from
     * grid to bare paper, and a corner at the frame edge has no paper beyond
     * it -- the bottom-left of the first A30s photograph, 16 px edge RMS.
     */
    fun cornersNearFrameEdge(
        corners: DoubleArray, width: Int, height: Int, marginFraction: Double = 0.02
    ): List<Int> {
        val m = max(width, height) * marginFraction
        return (0 until 4).filter {
            val x = corners[2 * it]
            val y = corners[2 * it + 1]
            x < m || y < m || x > width - 1 - m || y > height - 1 - m
        }
    }

    class Window(
        val srcX: Int, val srcY: Int, val srcW: Int, val srcH: Int,
        val dstX: Double, val dstY: Double, val dstW: Double, val dstH: Double
    )

    /**
     * The part of the bitmap a loupe of [side] px shows around (cx, cy), with
     * [halfSrc] bitmap px either side, clipped to the bitmap; and where that
     * part lands inside the loupe. Null when nothing of the bitmap is in view.
     */
    fun loupeWindow(
        cx: Double, cy: Double, halfSrc: Int, width: Int, height: Int, side: Double
    ): Window? {
        val left = cx - halfSrc
        val top = cy - halfSrc
        val sl = max(0, kotlin.math.floor(left).toInt())
        val st = max(0, kotlin.math.floor(top).toInt())
        val sr = min(width, kotlin.math.ceil(cx + halfSrc).toInt())
        val sb = min(height, kotlin.math.ceil(cy + halfSrc).toInt())
        if (sr <= sl || sb <= st) return null
        val k = side / (2.0 * halfSrc)
        return Window(sl, st, sr - sl, sb - st,
            (sl - left) * k, (st - top) * k, (sr - sl) * k, (sb - st) * k)
    }
}

/** Finger-sized: a handle is grabbed from this far away, and drawn at about half. */
private val GRAB_RADIUS = 36.dp

/** Loupe side on screen. */
private val LOUPE_SIZE = 132.dp

/**
 * Photograph pixels either side of the corner in the drag loupe: ~9 mm at
 * ~11 px/mm. Was 60 (~5.5 mm), less than the 6-9 mm the bottom corners were
 * off in phone runs 4 and 5 -- a corner that far out never showed the true
 * one in the loupe at all. At 100 a 132 dp loupe (~264 px on the A30s) is
 * 1.3 screen px per photograph px, still about seven times the preview's
 * magnification; the per-corner checks below the photo zoom in further.
 */
private const val LOUPE_HALF_SRC_PX = 100

/** Photograph pixels either side of a corner in its check tile: ~4.5 mm. */
private const val CHECK_HALF_SRC_PX = 50

// The redesign's corner colours (docs/ui_plan.md §2.1): handles and quad in
// cornerOverlay red; the handle being dragged and the loupe crosshair in the
// data teal so they stand out from the quad.
private val HANDLE = com.maarijalam.eacg.ui.theme.CornerOverlay
private val QUAD = com.maarijalam.eacg.ui.theme.CornerLine
private val ACTIVE = com.maarijalam.eacg.ui.theme.Data
private val REFINED = Color(0xFF43A047)
