package com.maarijalam.eacg

/**
 * Reader for `port_fixtures.txt`, the golden vectors the Python emits.
 *
 * Deliberately parses a flat `key v1 v2 v3` text format with `split`, rather
 * than JSON. A JSON parser would be a test dependency, and `org.json` is
 * stubbed to throw in Android host-side unit tests, so it would have to be
 * a real one. The format is also greppable and diffable: when a value moves,
 * `git diff` shows which one, which matters for a file whose whole purpose
 * is to make a numeric change visible.
 */
class PortFixtures private constructor(private val rows: Map<String, List<String>>) {

    companion object {
        private const val RESOURCE = "/port_fixtures.txt"

        fun load(): PortFixtures {
            val stream = PortFixtures::class.java.getResourceAsStream(RESOURCE)
                ?: error(
                    "$RESOURCE not found on the test classpath. It is generated " +
                        "by `python python/make_port_fixtures.py` and committed " +
                        "under android/app/src/test/resources/."
                )
            val rows = stream.bufferedReader().useLines { lines ->
                lines.map { it.trim() }
                    .filter { it.isNotEmpty() && !it.startsWith("#") }
                    .map { line ->
                        val parts = line.split(" ")
                        parts[0] to parts.drop(1)
                    }
                    .toMap()
            }
            return PortFixtures(rows)
        }
    }

    internal fun row(key: String): List<String> =
        rows[key] ?: error("missing key '$key' in $RESOURCE; regenerate it")

    fun int(key: String, index: Int = 0): Int = row(key)[index].toInt()
    fun long(key: String, index: Int = 0): Long = row(key)[index].toLong()
    fun double(key: String, index: Int = 0): Double = row(key)[index].toDouble()
    fun doubles(key: String): DoubleArray = row(key).map { it.toDouble() }.toDoubleArray()

    /**
     * The shared input, regenerated here rather than transferred.
     *
     * Must match `make_port_fixtures.lcg_plane` exactly. Masking to 31 bits
     * keeps every intermediate inside a signed 32-bit range, so Python's
     * arbitrary-precision integers and this Long arithmetic cannot diverge
     * through an overflow that only one side has.
     */
    internal fun lcgPlane(): SkewEstimator.Plane {
        val rows = int("lcg.rows")
        val cols = int("lcg.cols")
        val mult = long("lcg.mult")
        val add = long("lcg.add")
        val mask = long("lcg.mask")
        var state = long("lcg.seed") and mask
        val data = DoubleArray(rows * cols)
        for (i in data.indices) {
            state = (state * mult + add) and mask
            data[i] = (state % 256L).toDouble()
        }
        return SkewEstimator.Plane(data, rows, cols)
    }

    /**
     * The trace mask, regenerated here rather than transferred.
     *
     * Must match `make_port_fixtures.trace_mask` exactly. Integer arithmetic
     * throughout for the same reason [lcgPlane] uses it: neither side can
     * then diverge through a floating-point detail the other does not share.
     * A single ink band whose row wanders as `8 + (c * c) % band_mod`, with
     * `gap_width` blank columns at every multiple of `gap_mod` — the dropout
     * the gap bridging exists for, and which also puts the first inked column
     * away from zero so the span trim has something to trim. The gaps are two
     * columns wide on purpose: a one-column gap is bridged by every limit
     * down to 1, so the re-blanking half of the function would never run.
     */
    internal fun traceMask(): TraceExtractor.Mask {
        val rows = int("trace.rows")
        val cols = int("trace.cols")
        val gapMod = int("trace.gap_mod")
        val gapWidth = int("trace.gap_width")
        val bandMod = int("trace.band_mod")
        val thickness = int("trace.thickness")
        val data = ByteArray(rows * cols)
        for (c in 0 until cols) {
            if (c % gapMod < gapWidth) continue
            val top = 8 + (c * c) % bandMod
            for (k in 0 until thickness) {
                val r = top + k
                if (r in 0 until rows) data[r * cols + c] = 1
            }
        }
        return TraceExtractor.Mask(data, rows, cols)
    }

    /** Straight bands, one per entry in [at]. A 3x4 sheet's column. */
    internal fun stackedMask(
        rows: Int, cols: Int, at: List<Int>, thickness: Int
    ): TraceExtractor.Mask {
        val data = ByteArray(rows * cols)
        for (top in at) {
            for (k in 0 until thickness) {
                val r = top + k
                if (r in 0 until rows) {
                    for (c in 0 until cols) data[r * cols + c] = 1
                }
            }
        }
        return TraceExtractor.Mask(data, rows, cols)
    }

    fun ints(key: String): List<Int> = row(key).map { it.toInt() }
}
