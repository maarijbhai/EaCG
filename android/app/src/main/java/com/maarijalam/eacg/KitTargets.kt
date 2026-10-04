package com.maarijalam.eacg

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * The printed kit sheets the app can score a photograph against.
 *
 * Read from `assets/kit/targets.txt`, written by `python/make_kit_assets.py`
 * from `out/kit_targets/`. Each is a PTB-XL fold-10 record rendered by
 * ecg-image-kit and printed at true scale; the app holds its lead II rhythm
 * strip at 500 Hz so the handset can check its own reconstruction, for the
 * reason [TruthWaveform] gives.
 *
 * Flat `key=value` text rather than JSON, as elsewhere in this app: a parser
 * is a porting hazard in exchange for a few bytes.
 */
object KitTargets {

    class Target(
        val name: String,
        val ecgId: Int,
        val stratFold: Int,
        /** Printed size of the gridded image, mm -- exact, from px and dpi. */
        val widthMm: Double,
        val heightMm: Double,
        val fs: Double,
        val n: Int,
        val truthAsset: String,
        /** All 12 leads, 10 s, header line of names; null for older asset sets. */
        val truth12Asset: String? = null
    ) {
        val label: String get() = "PTB-XL $ecgId"
    }

    suspend fun load(context: Context): List<Target> = withContext(Dispatchers.IO) {
        context.assets.open("kit/targets.txt").bufferedReader().useLines { lines ->
            lines.map { it.trim() }
                .filter { it.isNotEmpty() && !it.startsWith("#") }
                .map { line ->
                    val kv = line.split(" ").associate {
                        val i = it.indexOf('=')
                        require(i > 0) { "kit/targets.txt: bad field '$it'" }
                        it.substring(0, i) to it.substring(i + 1)
                    }
                    fun f(k: String) = kv[k] ?: error("kit/targets.txt: no '$k' in '$line'")
                    Target(
                        name = f("name"),
                        ecgId = f("ecg_id").toInt(),
                        stratFold = f("strat_fold").toInt(),
                        widthMm = f("width_mm").toDouble(),
                        heightMm = f("height_mm").toDouble(),
                        fs = f("fs").toDouble(),
                        n = f("n").toInt(),
                        truthAsset = "kit/" + f("truth"),
                        truth12Asset = kv["truth12"]?.let { "kit/$it" }
                    )
                }
                .toList()
        }
    }
}
