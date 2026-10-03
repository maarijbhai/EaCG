package com.maarijalam.eacg

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.launch
import java.util.Locale

/**
 * The learned-tracer benchmark: one button, then a result block per
 * (variant, threads). Its own state, like the photo section, because it
 * needs no loaded image -- the golden tile is bundled.
 */
@Composable
fun TracerBenchmarkSection() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var running by remember { mutableStateOf(false) }
    var progress by remember { mutableStateOf<String?>(null) }
    var results by remember { mutableStateOf<List<TracerBenchmark.Result>>(emptyList()) }
    var exported by remember { mutableStateOf<String?>(null) }

    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text("Learned tracer (b3) on this phone",
                style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
            Text(
                "Times the model call alone on a bundled 512 x 1024 tile from a real " +
                    "photograph, ${TracerBenchmark.VARIANTS.joinToString("/")} at " +
                    "${TracerBenchmark.THREADS.joinToString("/")} threads: 1 cold + " +
                    "${TracerBenchmark.SHEETS * TracerBenchmark.TILES_PER_SHEET} warm calls each. " +
                    "A full sheet is ${TracerBenchmark.TILES_PER_SHEET} tiles. Keep the screen " +
                    "on and the phone cool; takes a few minutes.",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Button(
                onClick = {
                    running = true; results = emptyList(); exported = null
                    scope.launch {
                        results = TracerBenchmark.run(context) { progress = it }
                        running = false
                        progress = null
                    }
                },
                enabled = !running,
                modifier = Modifier.fillMaxWidth()
            ) { Text(if (running) (progress ?: "Running ...") else "Run tracer benchmark") }

            for (r in results) ResultBlock(r)

            if (results.isNotEmpty()) {
                OutlinedButton(
                    onClick = {
                        scope.launch {
                            val stamp = Exporter.stamp()
                            exported = runCatching {
                                Exporter.exportText(context, "eacg_tracer_bench_$stamp.csv",
                                    TracerBenchmark.csv(results, stamp))
                            }.getOrElse { "export failed: ${it.message}" }
                        }
                    },
                    modifier = Modifier.fillMaxWidth()
                ) { Text("Export benchmark CSV") }
                exported?.let {
                    Text(it, style = MaterialTheme.typography.labelSmall,
                        fontFamily = FontFamily.Monospace)
                }
            }
        }
    }
}

@Composable
private fun ResultBlock(r: TracerBenchmark.Result) {
    HorizontalDivider()
    Text("${r.variant}, ${r.threads} threads",
        style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold)
    if (r.error != null) {
        Text(r.error, color = MaterialTheme.colorScheme.error,
            style = MaterialTheme.typography.labelSmall)
        return
    }
    val s = r.stats
    fun f(fmt: String, v: Any) = String.format(Locale.US, fmt, v)
    Line("model / load", f("%.2f MB", r.modelMb) + " / ${r.loadMs} ms")
    Line("tile cold / mean / p95", "${s.first} / ${f("%.0f", s.mean)} / ${s.p95} ms")
    Line("full sheet (12 tiles)", f("%.0f ms", r.sheetMs))
    Line("native heap load / peak", f("%.1f", r.nativeHeapLoadMb) + " / " +
        f("%.1f MB", r.nativeHeapPeakMb))
    Line("app PSS", f("%.0f MB", r.pssMb))
    Line("vs desktop: max / mean |row|", f("%.3f", r.maxRowDiffPx) + " / " +
        f("%.4f px", r.meanRowDiffPx))
    Line("vs desktop: presence agree", f("%.2f%%", 100 * r.presenceAgree))
}

@Composable
private fun Line(label: String, value: String) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, style = MaterialTheme.typography.bodySmall)
        Text(value, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
    }
}
