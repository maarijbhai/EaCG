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

    fun updatePrototypeMode(on: Boolean) {
        prototypeMode = on
        prefs().edit().putBoolean(KEY_PROTOTYPE, on).apply()
    }

    private fun prefs() = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    companion object {
        private const val PREFS = "eacg_settings"
        private const val KEY_PROTOTYPE = "prototype_mode"

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
