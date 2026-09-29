package com.maarijalam.eacg

import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * FR10: get a digitisation off the phone.
 *
 * One CSV per scored photograph, into the public Downloads folder under
 * `EaCG/`, so it can be copied to the desktop with a file manager or USB
 * rather than `adb`. The header lines carry everything needed to interpret
 * or re-score the waveform -- which sheet, which corners, which calibration
 * produced which column, the gate's verdict and the stage timings -- because
 * a waveform without its calibration is a list of numbers.
 *
 * MediaStore on Android 10 and later needs no storage permission for a file
 * the app creates itself. On 8 and 9 (minSdk is 26) it falls back to the
 * app's own external files directory, which needs none either but is only
 * visible over USB; the returned string says which happened.
 */
object Exporter {

    suspend fun export(
        context: Context,
        run: PhotoPipeline.Run,
        target: KitTargets.Target?,
        photoLabel: String
    ): String = withContext(Dispatchers.IO) {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val name = "eacg_${target?.name ?: "photo"}_$stamp.csv"
        val text = render(run, target, photoLabel, stamp)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val values = ContentValues().apply {
                put(MediaStore.Downloads.DISPLAY_NAME, name)
                put(MediaStore.Downloads.MIME_TYPE, "text/csv")
                put(MediaStore.Downloads.RELATIVE_PATH,
                    Environment.DIRECTORY_DOWNLOADS + "/EaCG")
            }
            val uri = context.contentResolver.insert(
                MediaStore.Downloads.EXTERNAL_CONTENT_URI, values
            ) ?: error("MediaStore refused to create $name")
            context.contentResolver.openOutputStream(uri)?.use {
                it.write(text.toByteArray())
            } ?: error("could not open $uri for writing")
            "Downloads/EaCG/$name"
        } else {
            val dir = File(context.getExternalFilesDir(null), "exports").apply { mkdirs() }
            val f = File(dir, name)
            f.writeText(text)
            f.absolutePath
        }
    }

    internal fun render(
        run: PhotoPipeline.Run,
        target: KitTargets.Target?,
        photoLabel: String,
        stamp: String
    ): String = buildString {
        // A missing value is written empty, not as the word "null".
        fun meta(k: String, v: Any?) = append("# $k: ${v ?: ""}\n")
        meta("generated_by", "EaCG PhotoPipeline")
        meta("local_time", stamp)
        meta("device", "${Build.MANUFACTURER} ${Build.MODEL} (API ${Build.VERSION.SDK_INT})")
        meta("photo", photoLabel)
        meta("target", target?.name ?: "none")
        meta("ecg_id", target?.ecgId)
        run.handCorners?.let { h -> meta("corners_hand_px", h.joinToString(" ") { "%.2f".us(it) }) }
        meta("corners_used", if (run.cornersFromHand) "hand (refinement fit too poor)" else "refined")
        meta("corners_used_px", run.corners.joinToString(" ") { "%.2f".us(it) })
        // What the edge refinement found, whether or not it was used.
        meta("refine_shift_max_px", "%.2f".us(run.cornerShiftMaxPx))
        meta("refine_edge_rmse_max_px", "%.3f".us(run.edgeRmseMaxPx))
        meta("rectified_px", "${run.rectW}x${run.rectH}")
        // Rectified width over the sheet's printed width: the scale the
        // corners imply, true only if they are the sheet's corners. The
        // cal.*.truth_* fields are measured against the waveform instead.
        meta("px_per_mm_from_corners", run.pxPerMmFromCorners?.let { "%.5f".us(it) })
        run.location?.let {
            meta("strip_box_px", "${it.x0},${it.y0},${it.x1},${it.y1}")
            meta("strip_t0_px", "%.2f".us(it.t0X))
            meta("pulse_found", it.pulseX1 >= 0)
            meta("level_angle_deg", run.levelAngleDeg?.let { a -> "%.4f".us(a) })
            meta("level_margin", run.levelMargin?.let { a -> "%.3f".us(a) })
            meta("level_applied", run.levelApplied)
            if (target != null && it.pulseX1 < 0) {
                meta("pulse_warning", "every kit sheet prints a 1 mV pulse and none was " +
                    "found: it was traced as signal and its width counted in the duration anchor")
            }
        }
        val t = run.timings
        meta("ms_corners", t.cornersMs)
        meta("ms_rectify", t.rectifyMs)
        meta("ms_locate", t.locateMs)
        meta("ms_strip", t.stripMs)
        meta("ms_grid", t.gridMs)
        meta("ms_pipeline", t.pipelineMs)
        meta("ms_harness", t.harnessMs)
        for (c in run.calibrations) {
            val p = "cal.${c.name}"
            meta("$p.px_per_mm", "%.5f".us(c.pxPerMm))
            meta("$p.error_vs_corners_pct", c.errorVsCornersPct?.let { "%.4f".us(it) })
            c.alignment?.let { al ->
                // Against the truth waveform: the error the harness vouches for.
                // Blank unless the match spans >= 8.5 s and is off the bound.
                meta("$p.truth_align_valid", al.valid)
                meta("$p.truth_align_overlap_s", "%.2f".us(al.nCompared / PhotoPipeline.FS_OUT))
                meta("$p.truth_px_per_mm_error_pct", al.pxPerMmErrorPct?.let { "%.4f".us(it) })
                meta("$p.truth_align_scale", "%.4f".us(al.scale))
                meta("$p.truth_align_lag_samples", al.lagSamples)
                meta("$p.truth_align_at_bound", al.atBound)
                meta("$p.pcc_aligned", if (al.valid) "%.4f".us(al.pcc) else "")
            }
            meta("$p.offset_samples", c.offsetSamples)
            meta("$p.verdict", c.gate?.verdict?.name)
            meta("$p.rejections", c.gate?.rejections?.joinToString(";"))
            meta("$p.flags", c.gate?.flags?.joinToString(";"))
            meta("$p.pcc_asis", c.fidelity?.pcc)
            // The span it was computed over: a 2 s trace can score a fair PCC
            // over its 2 s and mean nothing about the 10 s recording.
            meta("$p.asis_overlap_s", c.fidelity?.let { "%.2f".us(it.nCompared / PhotoPipeline.FS_OUT) })
            meta("$p.rmse_mv", c.fidelity?.rmseMv)
            meta("$p.gain_ratio", c.fidelity?.gainRatio)
            meta("$p.trace_ms", c.traceMs)
            c.error?.let { meta("$p.error", it) }
        }
        // One column per calibration that produced a waveform, at 500 Hz.
        val cols = run.calibrations.filter { it.digitised != null }
        append("t_s")
        for (c in cols) append(",mv_${c.name}")
        append("\n")
        val n = cols.maxOfOrNull { it.digitised!!.signalMv.size } ?: 0
        for (i in 0 until n) {
            append("%.4f".us(i / PhotoPipeline.FS_OUT))
            for (c in cols) {
                val s = c.digitised!!.signalMv
                append(',')
                if (i < s.size && s[i].isFinite()) append("%.5f".us(s[i]))
            }
            append('\n')
        }
    }

    /** Always a decimal point: a German-locale phone would write commas. */
    private fun String.us(vararg args: Any?): String = String.format(Locale.US, this, *args)
}
