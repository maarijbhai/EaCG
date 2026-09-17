package com.maarijalam.eacg

import android.app.Application

/**
 * Exists for one reason: to load OpenCV before any screen can try to use it.
 *
 * Application.onCreate runs before the first Activity, so by the time
 * [DigitiseScreen] composes, [OpenCvRuntime.status] is settled and the screen
 * can state the outcome instead of discovering it.
 */
class EaCGApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        OpenCvRuntime.initialise()
    }
}
