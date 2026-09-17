package com.maarijalam.eacg

import android.util.Log
import org.opencv.android.OpenCVLoader

/**
 * Loads OpenCV's native library once, at application start, and remembers
 * whether it worked.
 *
 * **Why this is not just a call to `initLocal()` at the first use site.**
 * OpenCV's Java API is a thin JNI shim: every `Mat`, every `Imgproc` call,
 * reaches into `libopencv_java4.so`. If that library never loaded, nothing
 * fails at the point of the mistake -- the failure surfaces later as an
 * `UnsatisfiedLinkError` from inside whichever operation happened to be
 * first, which on this device will be a resize buried in a background
 * coroutine, reported as a coroutine failure with a stack trace pointing at
 * OpenCV internals rather than at the missing library. Loading once, up
 * front, and holding the verdict converts a confusing late crash into a
 * plain statement on screen.
 *
 * The load can fail for exactly one reason that matters here: the APK does
 * not carry a native library for the device's ABI. `app/build.gradle.kts`
 * filters to `arm64-v8a` alone, which is what the Galaxy A30s runs, so on
 * any 64-bit ARM device this succeeds and on a 32-bit-only one the install
 * itself fails earlier. It is still checked rather than assumed, because an
 * ABI filter is exactly the kind of build setting that gets changed for an
 * unrelated reason and silently narrows the set of devices that work.
 *
 * `initLocal()` rather than `initDebug()`: both exist in 4.14.0, but
 * `initLocal` is the entry point for the Maven artifact, which packages the
 * `.so` inside the APK. `initDebug` belongs to the older workflow where the
 * OpenCV Manager app supplied the library separately.
 */
object OpenCvRuntime {

    /** What happened when the library was loaded. */
    sealed interface Status {
        /** Not attempted yet. Seeing this on screen means [initialise] never ran. */
        data object Pending : Status

        data class Ready(val version: String) : Status

        /** Carries the reason, because "OpenCV failed" is not actionable. */
        data class Failed(val reason: String) : Status
    }

    private const val TAG = "OpenCvRuntime"

    @Volatile
    var status: Status = Status.Pending
        private set

    val isReady: Boolean get() = status is Status.Ready

    /**
     * Idempotent. Called from [EaCGApplication.onCreate]; safe to call again.
     *
     * Catches `Throwable` rather than `Exception` on purpose: the failure
     * mode being guarded against is `UnsatisfiedLinkError`, which is an
     * `Error`, not an `Exception`, and would otherwise pass straight through
     * a conventional catch and take the process down.
     */
    @Synchronized
    fun initialise() {
        if (status !is Status.Pending) return
        status = try {
            if (OpenCVLoader.initLocal()) {
                Log.i(TAG, "OpenCV ${OpenCVLoader.OPENCV_VERSION} loaded")
                Status.Ready(OpenCVLoader.OPENCV_VERSION)
            } else {
                // initLocal returns false rather than throwing when the
                // library is absent for this ABI.
                Log.e(TAG, "OpenCVLoader.initLocal() returned false")
                Status.Failed(
                    "initLocal() returned false -- no native library for this " +
                        "device's ABI. The APK is built for arm64-v8a only."
                )
            }
        } catch (t: Throwable) {
            Log.e(TAG, "OpenCV native load threw", t)
            Status.Failed("${t::class.java.simpleName}: ${t.message ?: "no message"}")
        }
    }
}
