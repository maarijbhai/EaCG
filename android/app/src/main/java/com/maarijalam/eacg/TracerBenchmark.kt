package com.maarijalam.eacg

import android.content.Context
import android.os.Debug
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.abs
import kotlin.math.max

/**
 * On-device latency, memory and fidelity of the exported learned tracer.
 *
 * The question this answers before any Kotlin port of the 12-lead path is
 * written: can the A30s run model B fast enough? A full 3x4 + rhythm sheet
 * is four rows, each a 512 x 2,048 canonical frame run as three overlapping
 * 512 x 1,024 tiles (`ml/train_tracer.predict_tiled`), so TILES_PER_SHEET
 * model calls. Each exported variant is timed per tile -- one cold call,
 * then SHEETS x TILES_PER_SHEET warm ones -- at each thread count.
 *
 * Fidelity: the bundled tile (`assets/ml/tile_512x1024.rgb`, a real
 * canonical tile from photograph ptbxl40_20260929_05, made by
 * `ml/make_device_fixture.py`) comes with what each variant outputs for it
 * on the desktop. The phone's output is compared column by column, so a
 * port or runtime difference shows up as a number rather than as a
 * mysteriously different waveform later.
 *
 * What is NOT timed here: canonicalisation, segmentation, tick calibration,
 * decode to mV. Only the model call. CLAUDE.md limitation 9: this is the
 * model's share of PIPELINE, not the whole of it.
 */
object TracerBenchmark {

    const val TILES_PER_SHEET = 12
    const val SHEETS = 3
    val VARIANTS = listOf("int8f", "fp16")
    val THREADS = listOf(2, 4)
    private const val TILE_JSON = "ml/tile_512x1024.json"
    private const val TILE_RGB = "ml/tile_512x1024.rgb"

    data class Result(
        val variant: String,
        val threads: Int,
        val modelMb: Double,
        val loadMs: Long,
        /** Per tile, run order: [0] cold, then warm. */
        val tileMs: List<Long>,
        val nativeHeapLoadMb: Double,
        val nativeHeapPeakMb: Double,
        val pssMb: Double,
        val maxRowDiffPx: Double,
        val meanRowDiffPx: Double,
        val presenceAgree: Double,
        val error: String? = null
    ) {
        val stats: Stats get() = summarise(tileMs)
        /** Model time for a whole sheet from the warm mean. */
        val sheetMs: Double get() = stats.mean * TILES_PER_SHEET
    }

    private fun mapAsset(context: Context, path: String): Pair<MappedByteBuffer, Long> {
        context.assets.openFd(path).use { fd ->
            FileInputStream(fd.fileDescriptor).use { s ->
                val buf = s.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
                return buf to fd.declaredLength
            }
        }
    }

    private fun mb(bytes: Long) = bytes / (1024.0 * 1024.0)

    private fun String.us(vararg args: Any?): String =
        String.format(java.util.Locale.US, this, *args)

    private fun pssMb(): Double {
        val info = Debug.MemoryInfo()
        Debug.getMemoryInfo(info)
        return info.totalPss / 1024.0
    }

    /** Runs every variant at every thread count; [progress] gets a line per step. */
    suspend fun run(context: Context, progress: (String) -> Unit): List<Result> =
        withContext(Dispatchers.Default) {
            val meta = JSONObject(context.assets.open(TILE_JSON).bufferedReader().readText())
            val h = meta.getInt("height")
            val w = meta.getInt("width")
            val rgb = context.assets.open(TILE_RGB).use { it.readBytes() }
            require(rgb.size == h * w * 3) { "tile is ${rgb.size} bytes, expected ${h * w * 3}" }
            // Float input, 0-255: the model's own Rescaling layer normalises.
            val input = ByteBuffer.allocateDirect(h * w * 3 * 4).order(ByteOrder.nativeOrder())
            val fin = input.asFloatBuffer()
            for (b in rgb) fin.put((b.toInt() and 0xFF).toFloat())
            val expected = meta.getJSONObject("expected")

            val results = mutableListOf<Result>()
            for (variant in VARIANTS) {
                val exp = expected.getJSONObject(variant)
                val file = "ml/" + exp.getString("file")
                for (threads in THREADS) {
                    progress("$variant, $threads threads ...")
                    results += runOne(context, file, variant, threads, input, h, w, exp)
                }
            }
            progress("done")
            results
        }

    private fun runOne(
        context: Context, file: String, variant: String, threads: Int,
        input: ByteBuffer, h: Int, w: Int, exp: JSONObject
    ): Result {
        val heap0 = Debug.getNativeHeapAllocatedSize()
        var interp: Interpreter? = null
        try {
            val t0 = System.nanoTime()
            val (model, bytes) = mapAsset(context, file)
            interp = Interpreter(model, Interpreter.Options().setNumThreads(threads))
            val loadMs = (System.nanoTime() - t0) / 1_000_000
            val heapLoad = Debug.getNativeHeapAllocatedSize() - heap0

            val outW = 2 * w
            val row = ByteBuffer.allocateDirect(outW * 4).order(ByteOrder.nativeOrder())
            val pres = ByteBuffer.allocateDirect(outW * 4).order(ByteOrder.nativeOrder())
            val inName = interp.getSignatureInputs(interp.signatureKeys[0]).first()
            val outs = interp.getSignatureOutputs(interp.signatureKeys[0])
            require("row" in outs && "presence" in outs) { "outputs are ${outs.joinToString()}" }

            fun call(): Long {
                input.rewind(); row.rewind(); pres.rewind()
                val s = System.nanoTime()
                interp.runSignature(mapOf(inName to input), mapOf("row" to row, "presence" to pres))
                return (System.nanoTime() - s) / 1_000_000
            }

            val times = mutableListOf(call())          // cold
            // Fidelity from the cold call's output, before anything warms.
            val rowF: FloatBuffer = (row.rewind() as ByteBuffer).asFloatBuffer()
            val presF: FloatBuffer = (pres.rewind() as ByteBuffer).asFloatBuffer()
            val expRow = exp.getJSONArray("row")
            val expPres = exp.getJSONArray("presence")
            var maxD = 0.0; var sumD = 0.0; var nD = 0; var agree = 0
            for (j in 0 until outW) {
                val p = presF.get(j) > 0f
                val pe = expPres.getDouble(j) > 0.0
                if (p == pe) agree++
                if (p && pe) {
                    val d = abs(rowF.get(j) - expRow.getDouble(j))
                    maxD = max(maxD, d); sumD += d; nD++
                }
            }
            var heapPeak = Debug.getNativeHeapAllocatedSize() - heap0
            repeat(SHEETS * TILES_PER_SHEET) {
                times += call()
                heapPeak = max(heapPeak, Debug.getNativeHeapAllocatedSize() - heap0)
            }
            return Result(
                variant, threads, mb(bytes), loadMs, times,
                mb(heapLoad), mb(heapPeak), pssMb(),
                maxD, if (nD > 0) sumD / nD else Double.NaN, agree.toDouble() / outW
            )
        } catch (e: Throwable) {
            return Result(variant, threads, 0.0, 0, listOf(0L), 0.0, 0.0, 0.0,
                Double.NaN, Double.NaN, 0.0, "${e.javaClass.simpleName}: ${e.message}")
        } finally {
            interp?.close()
        }
    }

    /** One CSV per run, for results/device/ via ingest. */
    fun csv(results: List<Result>, stamp: String): String = buildString {
        fun meta(k: String, v: Any?) = append("# $k: ${v ?: ""}\n")
        meta("generated_by", "EaCG TracerBenchmark")
        meta("local_time", stamp)
        meta("device", "${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL} (API ${android.os.Build.VERSION.SDK_INT})")
        meta("tiles_per_sheet", TILES_PER_SHEET)
        meta("warm_calls", SHEETS * TILES_PER_SHEET)
        meta("note", "model call only; canonicalisation/segmentation/decode not timed")
        append("variant,threads,model_mb,load_ms,cold_ms,warm_mean_ms,warm_p95_ms,warm_min_ms," +
            "sheet_ms_est,native_heap_load_mb,native_heap_peak_mb,pss_mb," +
            "max_row_diff_px,mean_row_diff_px,presence_agree,error\n")
        for (r in results) {
            val s = r.stats
            append(listOf(r.variant, r.threads, "%.3f".us(r.modelMb), r.loadMs, s.first,
                "%.1f".us(s.mean), s.p95, s.min, "%.0f".us(r.sheetMs),
                "%.1f".us(r.nativeHeapLoadMb), "%.1f".us(r.nativeHeapPeakMb), "%.1f".us(r.pssMb),
                "%.4f".us(r.maxRowDiffPx), "%.5f".us(r.meanRowDiffPx),
                "%.4f".us(r.presenceAgree), r.error ?: "").joinToString(","))
            append("\n")
        }
    }
}
