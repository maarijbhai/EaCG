package com.maarijalam.eacg

/**
 * Peak memory per pipeline stage (NFR2), for [SheetPipeline.run].
 *
 * A sampler calls [sample] every few tens of milliseconds while the pipeline
 * runs; [close] ends the current stage, the same moment the stage's timing
 * lap is taken, and files the highest reading seen since the previous close
 * under that stage's name. Two quantities, because they live in different
 * places on Android:
 *
 *   native  `Debug.getNativeHeapAllocatedSize()`: OpenCV Mats, the LiteRT
 *           interpreter and its tensor arena, and (API 26+) Bitmap pixels --
 *           nearly all of this pipeline's memory
 *   java    `Runtime.totalMemory() - freeMemory()`: the Kotlin arrays (row
 *           traces, masks, profiles); includes garbage not yet collected,
 *           so it is an upper bound on what is live
 *
 * Both are reported as absolute bytes and against the [baseline] taken
 * before the first stage, so "what this stage added" can be read off.
 *
 * Limits, stated so the numbers are not overread: a sampled peak can miss an
 * allocation shorter than the sampling interval (each [close] also samples,
 * which catches what is still held at a stage's end), and neither figure is
 * the process's total footprint (PSS), which [SheetPipeline] reads once at
 * the end.
 *
 * Pure Kotlin over an injected reader, so the bookkeeping is JVM-tested.
 */
class StageMemory(private val read: () -> Reading) {

    class Reading(val nativeBytes: Long, val javaBytes: Long)

    class Peak(val nativeBytes: Long, val javaBytes: Long)

    val baseline: Reading = read()
    private val peaks = linkedMapOf<String, Peak>()
    private var curNative = baseline.nativeBytes
    private var curJava = baseline.javaBytes

    /** Take one reading into the current stage's running peak. */
    @Synchronized
    fun sample() {
        val r = read()
        if (r.nativeBytes > curNative) curNative = r.nativeBytes
        if (r.javaBytes > curJava) curJava = r.javaBytes
    }

    /**
     * End the current stage as [stage]: one last reading, file the peak,
     * and start the next stage's running peak from the memory held now.
     */
    @Synchronized
    fun close(stage: String) {
        val r = read()
        peaks[stage] = Peak(maxOf(curNative, r.nativeBytes), maxOf(curJava, r.javaBytes))
        curNative = r.nativeBytes
        curJava = r.javaBytes
    }

    /** Stage -> peak, in the order the stages closed. */
    @Synchronized
    fun peaks(): Map<String, Peak> = LinkedHashMap(peaks)

    /** The highest native and Java readings over every closed stage. */
    @Synchronized
    fun overall(): Peak = Peak(
        peaks.values.maxOfOrNull { it.nativeBytes } ?: baseline.nativeBytes,
        peaks.values.maxOfOrNull { it.javaBytes } ?: baseline.javaBytes)

    companion object {
        fun mb(bytes: Long): Double = bytes / (1024.0 * 1024.0)

        /**
         * `stage=native/java` in MB, absolute, after a `baseline=` entry --
         * the format of the CSV's `memory_peak_mb` line.
         */
        fun format(baseline: Reading, peaks: Map<String, Peak>): String {
            fun f(v: Long) = String.format(java.util.Locale.US, "%.1f", mb(v))
            return (listOf("baseline=${f(baseline.nativeBytes)}/${f(baseline.javaBytes)}") +
                peaks.map { (k, p) -> "$k=${f(p.nativeBytes)}/${f(p.javaBytes)}" })
                .joinToString(" ")
        }
    }
}
