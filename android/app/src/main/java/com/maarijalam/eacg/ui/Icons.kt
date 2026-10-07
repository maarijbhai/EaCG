package com.maarijalam.eacg.ui

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.vector.addPathNodes
import androidx.compose.ui.graphics.vector.path
import androidx.compose.ui.unit.dp

/**
 * The mock-up's line icons (24-unit viewBox, round caps and joins), from its
 * own SVG path data, so the app matches the approved design without the
 * material-icons dependency. Tinted by `Icon(tint = ...)`: drawn in black.
 */
object EaIcons {

    /** A circle as SVG path data (two arcs). */
    private fun circle(cx: Float, cy: Float, r: Float) =
        "M${cx - r},${cy}a$r,$r 0 1,0 ${2 * r},0a$r,$r 0 1,0 ${-2 * r},0"

    private fun stroke(name: String, width: Float, vararg d: String): ImageVector =
        ImageVector.Builder(name, 24.dp, 24.dp, 24f, 24f).apply {
            for (p in d) addPath(addPathNodes(p), stroke = SolidColor(Color.Black),
                strokeLineWidth = width, strokeLineCap = StrokeCap.Round,
                strokeLineJoin = StrokeJoin.Round)
        }.build()

    private fun fill(name: String, vararg d: String): ImageVector =
        ImageVector.Builder(name, 24.dp, 24.dp, 24f, 24f).apply {
            for (p in d) addPath(addPathNodes(p), fill = SolidColor(Color.Black))
        }.build()

    val Camera = stroke("camera", 2f,
        "M4 8h3l2-3h6l2 3h3a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1z",
        circle(12f, 13.5f, 3.5f))
    val Gallery = stroke("gallery", 2f,
        "M5 4h14a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z",
        circle(9f, 10f, 2f), "M21 17l-5-5-9 8")
    val List = stroke("list", 2f,
        "M8 6h12M8 12h12M8 18h12", circle(4f, 6f, 1f), circle(4f, 12f, 1f), circle(4f, 18f, 1f))
    val Settings = stroke("settings", 1.8f, circle(12f, 12f, 3f),
        "M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z")
    val Back = stroke("back", 2f, "M19 12H5M12 19l-7-7 7-7")
    val Close = stroke("close", 2f, "M18 6L6 18M6 6l12 12")
    val Help = stroke("help", 1.8f, circle(12f, 12f, 9f),
        "M9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6V14", circle(12f, 17.5f, 0.6f))
    val Share = stroke("share", 1.8f, circle(18f, 5f, 2.5f), circle(6f, 12f, 2.5f),
        circle(18f, 19f, 2.5f), "M8.2 10.8l7.6-4.4M8.2 13.2l7.6 4.4")
    val More = fill("more", circle(12f, 5f, 1.7f), circle(12f, 12f, 1.7f), circle(12f, 19f, 1.7f))
    val Check = stroke("check", 2.4f, "M5 12l5 5 9-10")
    val Warning = stroke("warning", 2.2f, "M12 3l10 18H2z", "M12 10v5", circle(12f, 18f, 0.6f))
    val Download = stroke("download", 2f, "M12 3v12M7 10l5 5 5-5M5 21h14")
    val ChevronRight = stroke("chevron", 2f, "M9 6l6 6-6 6")
    // Home's tip tiles.
    val Page = stroke("page", 1.8f, "M5 3h14a1 1 0 0 1 1 1v16a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z",
        "M8 8h8M8 12h8M8 16h5")
    val Sun = stroke("sun", 1.8f, circle(12f, 12f, 4f),
        "M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4")
    val Frame = stroke("frame", 1.8f, "M3 8V4h4M17 4h4v4M21 16v4h-4M7 20H3v-4")
    val Search = stroke("search", 2f, circle(11f, 11f, 7f), "M20 20l-4-4")
    val Plus = stroke("plus", 2f, "M12 5v14M5 12h14")
    val Minus = stroke("minus", 2f, "M5 12h14")
    val Fit = stroke("fit", 2f, "M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5")
    val ChevronLeft = stroke("chevronLeft", 2f, "M15 6l-6 6 6 6")
    val Flask = stroke("flask", 1.8f, "M9 3h6M10 3v6L4.5 18.5A1.7 1.7 0 0 0 6 21h12a1.7 1.7 0 0 0 1.5-2.5L14 9V3",
        "M7 15h10")
}
