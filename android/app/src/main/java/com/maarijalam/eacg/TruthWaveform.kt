package com.maarijalam.eacg

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * The source waveform a bundled strip was rendered from, read off the asset.
 *
 * One float per line in millivolts, written by
 * `python/make_strip_fixtures.py`. Plain text rather than `.npy`: a
 * hand-rolled NumPy header parser on the device would be a porting hazard in
 * exchange for a few kilobytes, and this file is 7.5 KB.
 *
 * WHY THE GROUND TRUTH IS ON THE DEVICE AT ALL
 * So that the handset can score its own reconstruction. A fidelity figure
 * computed on the desktop and quoted for the phone cannot detect a porting
 * bug, and a porting bug is the entire risk the Kotlin side carries — it
 * would show up as a plausible wrong number in the results chapter rather
 * than as a visibly broken app. This is the same move the rotation slider
 * makes for deskew: manufacture a known answer, then ask the device to
 * recover it without sight of it.
 *
 * It is scaffolding, and the timings account for it as such. A deployed
 * capture has no truth to compare against and pays none of this.
 */
object TruthWaveform {

    /**
     * Load and parse. Returns null when [assetPath] is null, so a caller can
     * hand over `reference.truthAsset` without branching.
     *
     * A malformed line is a broken fixture rather than a runtime condition to
     * absorb, so this throws rather than skipping it: silently dropping
     * samples would shorten the reference and shift every comparison after
     * the bad line, which is far harder to notice than a crash on load.
     */
    suspend fun load(context: Context, assetPath: String?): DoubleArray? {
        if (assetPath == null) return null
        return withContext(Dispatchers.IO) {
            context.assets.open(assetPath).bufferedReader().useLines { lines ->
                lines.map { it.trim() }
                    .filter { it.isNotEmpty() }
                    .map { line ->
                        line.toDoubleOrNull()
                            ?: error("$assetPath: cannot parse '$line' as a number")
                    }
                    .toList()
                    .toDoubleArray()
            }
        }
    }
}
