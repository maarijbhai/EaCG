package com.maarijalam.eacg

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp

/**
 * The FR5-FR8 half of the screen: pick a strip, digitise it, see the verdict.
 *
 * Split out of [DigitiseScreen] rather than added to it because that file had
 * already reached 874 lines covering acquisition, rotation and deskew, and a
 * screen nobody can find the layout in stops being reviewable.
 */

/**
 * The single-strip fixtures, kept in a row of their own.
 *
 * Deliberately separate from the sheet row, and labelled. The two sets are
 * **not interchangeable**: a sheet is valid input for FR3 and invalid for the
 * tracer, which will splice four leads into one plausible curve and report
 * nothing wrong about it. Putting them in one undifferentiated row would
 * invite exactly the comparison that produces a wrong number.
 */
@Composable
fun StripRow(enabled: Boolean, onPick: (ReferenceImage) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(
            "Single strips — lead II, known waveform",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            STRIP_IMAGES.forEach { ref ->
                OutlinedButton(
                    onClick = { onPick(ref) },
                    enabled = enabled,
                    contentPadding = PaddingValues(horizontal = 8.dp, vertical = 4.dp),
                    modifier = Modifier.weight(1f)
                ) {
                    Text(ref.label, style = MaterialTheme.typography.labelMedium)
                }
            }
        }
    }
}

/** Runs FR3 to FR8 on whatever is loaded. */
@Composable
fun DigitiseControl(enabled: Boolean, busy: Boolean, onRun: () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        OutlinedButton(
            onClick = onRun,
            enabled = enabled && !busy,
            modifier = Modifier.fillMaxWidth().height(48.dp)
        ) {
            Text(
                if (busy) "Digitising ..." else "Digitise (FR5-FR8)",
                style = MaterialTheme.typography.titleSmall
            )
        }
        Text(
            "Traces the strip, calibrates to mV, then gates the result. " +
                "On a 3x4 sheet the gate should REJECT with \"4.0 ink bands " +
                "per column\". Runs on the loaded image, not the rotated " +
                "preview — deskew is a separate stage with its own controls.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

/**
 * The recovered waveform, and everything needed to distrust it.
 *
 * The verdict is shown **above** the plot on purpose. A digitised curve looks
 * convincing whether or not it means anything — the desktop measured a 3x4
 * sheet producing a perfectly smooth trace at PCC 0.273 — so the reader
 * should meet the refusal before the picture.
 */
@Composable
fun DigitisePanel(run: DigitisePipeline.Run?, label: String?) {
    if (run == null) return
    val gate = run.gate

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = when (gate?.verdict) {
                QualityGate.Verdict.REJECT -> MaterialTheme.colorScheme.errorContainer
                QualityGate.Verdict.FLAG -> MaterialTheme.colorScheme.tertiaryContainer
                else -> MaterialTheme.colorScheme.surfaceVariant
            }
        )
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                "Digitisation" + (label?.let { " — $it" } ?: ""),
                style = MaterialTheme.typography.titleSmall
            )

            Text(
                gate?.verdict?.name ?: "NO WAVEFORM",
                style = MaterialTheme.typography.headlineSmall
            )

            gate?.rejections?.forEach {
                Text("reject: $it", style = MaterialTheme.typography.bodySmall)
            }
            gate?.flags?.forEach {
                Text("flag: $it", style = MaterialTheme.typography.bodySmall)
            }
            run.error?.let {
                Text(it, style = MaterialTheme.typography.bodySmall)
            }

            run.digitised?.let { d ->
                WaveformPlot(d.signalMv)
                StatRow("samples", "${d.signalMv.size} @ ${"%.0f".format(d.fs)} Hz")
                StatRow("coverage", "%.3f".format(d.coverage))
                StatRow("baseline", "%.1f px".format(d.baselinePx))
                StatRow("px per mV / per s",
                    "%.2f / %.2f".format(d.pxPerMv, d.pxPerS))
            }
            StatRow("ink bands per column", "%.1f".format(run.tracesPerColumn))

            // Fidelity, and only where a ground truth exists. For anything the
            // user picked there is no right answer to be wrong about, and
            // saying so beats a blank the reader has to interpret.
            if (run.fidelity == null && run.digitised != null) {
                Text(
                    "No committed ground truth for this image, so fidelity is " +
                        "not scored. Expected for the 3x4 sheets and for any " +
                        "image you picked yourself.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            run.fidelity?.let { f ->
                Text("Reconstruction fidelity",
                    style = MaterialTheme.typography.labelMedium)
                StatRow("PCC", "%.4f".format(f.pcc))
                StatRow("RMSE", "%.4f mV".format(f.rmseMv))
                StatRow("gain ratio", "%.4f".format(f.gainRatio))
                StatRow("compared", "${f.nCompared} samples")
                Text(
                    "Read together: PCC is scale- and offset-invariant, so a " +
                        "mis-calibrated trace of the right shape still scores " +
                        "near 1. High RMSE with high PCC is a calibration fault.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            // Checks that could not run. Shown rather than omitted: a gate
            // that silently skips a test reads as one that passed it.
            val unavailable = gate?.unavailable ?: emptyList()
            if (unavailable.isNotEmpty()) {
                Text("Not checked", style = MaterialTheme.typography.labelMedium)
                unavailable.forEach {
                    Text("· $it", style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }

            Text("Per-stage latency", style = MaterialTheme.typography.labelMedium)
            val t = run.timings
            StatRow("to Mat", "${t.toMatMs} ms")
            StatRow("grid (FR3)", "${t.gridMs} ms")
            StatRow("binarise (FR5)", "${t.binariseMs} ms")
            StatRow("mask copy", "${t.maskCopyMs} ms")
            StatRow("trace + calibrate (FR6-7)", "${t.traceMs} ms")
            StatRow("gate (FR8)", "${t.gateMs} ms")
            StatRow("PIPELINE", "${t.pipelineMs} ms")
            StatRow("HARNESS (fidelity)", "${t.harnessMs} ms")
            Text(
                "Only PIPELINE is quotable. Scoring against a committed truth " +
                    "waveform is scaffolding: a deployed capture has no truth " +
                    "and pays none of it.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

/**
 * The recovered waveform as a polyline, autoscaled to its own extent.
 *
 * Autoscaled rather than drawn at a fixed mV per pixel, and that is a
 * limitation worth knowing about while reading it: a trace recovered with the
 * gain wrong by a factor of five looks exactly like a correct one here. The
 * amplitude is in the numbers above, which is where a scale error shows.
 *
 * A NaN run — dropout too wide to bridge — breaks the path rather than being
 * interpolated across, so a hole in the data looks like a hole.
 */
@Composable
fun WaveformPlot(signalMv: DoubleArray, modifier: Modifier = Modifier) {
    val stroke = MaterialTheme.colorScheme.onSurface
    val baselineColour = MaterialTheme.colorScheme.outlineVariant

    Canvas(modifier = modifier.fillMaxWidth().height(140.dp)) {
        if (signalMv.size < 2) return@Canvas
        var lo = Double.POSITIVE_INFINITY
        var hi = Double.NEGATIVE_INFINITY
        for (v in signalMv) {
            if (!v.isFinite()) continue
            if (v < lo) lo = v
            if (v > hi) hi = v
        }
        if (!lo.isFinite() || !hi.isFinite()) return@Canvas
        val span = (hi - lo).takeIf { it > 1e-9 } ?: 1.0

        val pad = 6f
        val h = size.height - 2 * pad
        val w = size.width
        fun yOf(v: Double) = pad + ((hi - v) / span * h).toFloat()

        // Zero, where it falls. Not the middle of the box: a trace sitting
        // entirely above baseline should look like it does.
        if (lo <= 0.0 && hi >= 0.0) {
            val y0 = yOf(0.0)
            drawLine(baselineColour, Offset(0f, y0), Offset(w, y0), strokeWidth = 1f)
        }

        val path = Path()
        var open = false
        for (i in signalMv.indices) {
            val v = signalMv[i]
            if (!v.isFinite()) { open = false; continue }
            val x = i.toFloat() / (signalMv.size - 1) * w
            val y = yOf(v)
            if (open) path.lineTo(x, y) else { path.moveTo(x, y); open = true }
        }
        drawPath(path, color = stroke, style = Stroke(width = 2f))
    }
}

@Composable
private fun StatRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            label,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            value,
            style = MaterialTheme.typography.bodyMedium,
            fontFamily = FontFamily.Monospace
        )
    }
}
