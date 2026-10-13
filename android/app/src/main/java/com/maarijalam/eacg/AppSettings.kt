package com.maarijalam.eacg

import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.runtime.staticCompositionLocalOf

/**
 * App-wide settings, persisted in SharedPreferences.
 *
 * [prototypeMode] (docs/ui_plan.md §4): ON shows the research controls --
 * tracer and grid-fit choice, the PTB-XL benchmark selector and scores,
 * per-stage timings and memory, and the Lab. OFF only HIDES them: nothing
 * it gates changes what the pipeline computes (a scan then runs with the
 * defaults, learned tracer and grid fit). ON by default for the prototype
 * phase.
 */
class AppSettings private constructor(private val context: Context) {

    var prototypeMode by mutableStateOf(prefs().getBoolean(KEY_PROTOTYPE, true))
        private set

    /** The PTB-XL sheet last chosen to benchmark against (its target name), or null. */
    var benchmarkTarget by mutableStateOf(prefs().getString(KEY_TARGET, null))
        private set

    fun updateBenchmarkTarget(name: String?) {
        benchmarkTarget = name
        prefs().edit().putString(KEY_TARGET, name).apply()
    }

    /**
     * Testing sessions (prototype mode only): one Digitise tap runs the learned
     * tracer, then the classical one on the same photo and corners, saving and
     * exporting both. Off by default; remembered like the benchmark target.
     */
    var bothTracers by mutableStateOf(prefs().getBoolean(KEY_BOTH, false))
        private set

    fun updateBothTracers(on: Boolean) {
        bothTracers = on
        prefs().edit().putBoolean(KEY_BOTH, on).apply()
    }

    /**
     * Which way to turn a photograph whose sheet came out sideways
     * (SheetOrientation): clockwise unless the user's corrections said
     * otherwise. Remembered, so a consistent grip is right every time.
     */
    var sidewaysTurnClockwise by mutableStateOf(prefs().getBoolean(KEY_TURN_CW, true))
        private set

    fun updateSidewaysTurnClockwise(cw: Boolean) {
        sidewaysTurnClockwise = cw
        prefs().edit().putBoolean(KEY_TURN_CW, cw).apply()
    }

    fun updatePrototypeMode(on: Boolean) {
        prototypeMode = on
        prefs().edit().putBoolean(KEY_PROTOTYPE, on).apply()
    }

    private fun prefs() = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    companion object {
        private const val PREFS = "eacg_settings"
        private const val KEY_PROTOTYPE = "prototype_mode"
        private const val KEY_TARGET = "benchmark_target"
        private const val KEY_BOTH = "both_tracers"
        private const val KEY_TURN_CW = "sideways_turn_cw"

        @Volatile private var instance: AppSettings? = null

        fun get(context: Context): AppSettings =
            instance ?: synchronized(this) {
                instance ?: AppSettings(context.applicationContext).also { instance = it }
            }
    }
}

/** The settings, provided at the root by MainActivity. */
val LocalAppSettings = staticCompositionLocalOf<AppSettings> {
    error("AppSettings not provided")
}
